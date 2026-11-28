/* Inkling 2 Mini sessions on Metal.  Included by ds4.c right before ds4_session_create (the session struct is
 * known by then); every public ds4_session_* function that needs different behaviour for this family starts with a
 * one-line `if (g_ds4_inkling2) return iks_...` hook.
 *
 * A session owns one ik_gpu state.  Invariant after every successful sync/eval: the GPU state has consumed exactly
 * the tokens of s->checkpoint and s->logits holds the logits of the last one (checkpoint_valid).  The ring and conv
 * state cannot be rolled back, so rewind keeps the token prefix and the next sync/eval replays it. */

static int iks_fail(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
    return 1;
}

static int iks_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (e->backend != DS4_BACKEND_METAL) {
        fprintf(stderr, "ds4: inkling2: sessions need --metal (the CPU path is a parity oracle only)\n");
        return 1;
    }
    if (ctx_size < 2) return 1;
    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    uint32_t chunk = ik_default_chunk();
    if (chunk > (uint32_t)ctx_size) chunk = (uint32_t)ctx_size;
    s->prefill_cap = chunk;
    s->ikg = ikg_new(e->ik, (uint32_t)ctx_size, chunk, 1);
    if (!s->ikg) { free(s); return 1; }
    ikg_reset(s->ikg);
    s->logits = xcalloc(IK_VOCAB, sizeof(float));
    s->sample_probs = xcalloc(IK_VOCAB, sizeof(float));
    *out = s;
    return 0;
}

static void iks_free(ds4_session *s) {
    if (!s) return;
    ikg_free(s->ikg);
    token_vec_free(&s->checkpoint);
    token_vec_free(&s->snap_tokens);
    free(s->snap_logits);
    free(s->logits);
    free(s->sample_probs);
    free(s);
}

/* Prefix reuse when the client drops reasoning_content.  The model's own turn is "<|message_model|><|content_thinking|>..." but the
 * official template renders the same turn without the thinking block, so the next prompt diverges from the live tokens one token
 * after the end of the previous prompt.  The GPU ring state cannot be rolled back, so at the end of every prompt sync we keep a copy
 * of the state (ikg_snapshot, tokens and logits).  A later sync that is not an exact continuation of the live state but starts with
 * those tokens restores the copy and only evaluates what follows.  One copy only: a prompt that differs earlier (an edited or
 * dropped older turn) still replays from zero. */
static void iks_take_snapshot(ds4_session *s) {
    s->snap_valid = false;
    if (!s->checkpoint_valid || s->checkpoint.len <= 0 || !s->logits) return;
    if (!ikg_snapshot(s->ikg)) return;
    s->snap_tokens.len = 0;
    for (int i = 0; i < s->checkpoint.len; i++) token_vec_push(&s->snap_tokens, s->checkpoint.v[i]);
    if (!s->snap_logits) s->snap_logits = xmalloc((size_t)IK_VOCAB * sizeof(float));
    memcpy(s->snap_logits, s->logits, (size_t)IK_VOCAB * sizeof(float));
    s->snap_valid = true;
}

/* Live state is an exact prefix of the prompt (the normal continuation). */
static bool iks_live_extends(const ds4_session *s, const ds4_tokens *prompt) {
    if (!s->checkpoint_valid || s->checkpoint.len > prompt->len || s->ikg->pos != (uint32_t)s->checkpoint.len) return false;
    for (int i = 0; i < s->checkpoint.len; i++)
        if (s->checkpoint.v[i] != prompt->v[i]) return false;
    return true;
}

static int iks_restore_prefix(ds4_session *s, const ds4_tokens *prompt) {
    if (!s || !prompt || !s->ikg || !s->snap_valid || s->snap_tokens.len <= 0 || s->snap_tokens.len > prompt->len) return 0;
    if (s->ikg->snap_pos != (uint32_t)s->snap_tokens.len || iks_live_extends(s, prompt)) return 0;
    for (int i = 0; i < s->snap_tokens.len; i++)
        if (s->snap_tokens.v[i] != prompt->v[i]) return 0;
    return s->snap_tokens.len;
}

static void iks_invalidate(ds4_session *s) {
    if (!s) return;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    if (s->ikg) ikg_reset(s->ikg);
}

/* Feed tokens[0..n) through the GPU state in chunks.  want_last: leave the logits of the final token in s->logits.
 * progress_base/total feed the "prefill_chunk" events. */
static int iks_run(ds4_session *s, const int *tok, int n, bool want_last, int progress_base, int progress_total,
                   char *err, size_t errlen) {
    ik_gpu *g = s->ikg;
    for (int c0 = 0; c0 < n;) {
        const int m = n - c0 < (int)g->chunk ? n - c0 : (int)g->chunk;
        const bool last = c0 + m == n;
        const int want0 = m - 1;
        const char *ferr = NULL;
        if (ikg_forward_chunk(g, tok + c0, m, &want0, last && want_last ? 1 : 0, last && want_last ? s->logits : NULL, &ferr)) {
            if (err && errlen) snprintf(err, errlen, "inkling2 forward failed: %s", ferr ? ferr : "unknown error");
            iks_invalidate(s);   /* the state is half written */
            return 1;
        }
        c0 += m;
        ds4_session_report_progress(s, "prefill_chunk", progress_base + c0, progress_total);
        if (s->display_progress) s->display_progress(s->display_progress_ud, "prefill_display", progress_base + c0, progress_total);
        if (c0 < n && s->cancel && s->cancel(s->cancel_ud)) {
            iks_invalidate(s);
            return iks_fail(err, errlen, "cancelled");
        }
    }
    return 0;
}

/* Make the GPU state hold exactly checkpoint[0..len): replay from scratch when it is stale (after a rewind). */
static int iks_catch_up(ds4_session *s, char *err, size_t errlen) {
    ik_gpu *g = s->ikg;
    if (g->pos == (uint32_t)s->checkpoint.len) return 0;
    const uint32_t kept = g->snap_pos;
    ikg_reset(g);
    if (s->checkpoint.len == 0) return 0;
    const int rc = iks_run(s, s->checkpoint.v, s->checkpoint.len, false, 0, s->checkpoint.len, err, errlen);
    /* A completed replay of the same tokens rewrites the global rows with the values they already held, so the kept end state
     * stays valid if the replay covers it.  A cancelled or failed replay leaves it cleared (see ikg_reset). */
    if (rc == 0 && s->snap_valid && kept && kept == (uint32_t)s->snap_tokens.len && kept <= (uint32_t)s->checkpoint.len) {
        bool same = true;
        for (uint32_t i = 0; i < kept && same; i++) same = s->checkpoint.v[i] == s->snap_tokens.v[i];
        if (same) g->snap_pos = kept;
    }
    return rc;
}

static int iks_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0) return iks_fail(err, errlen, "empty prompt");
    if (prompt->len >= s->ctx_size) return iks_fail(err, errlen, "prompt does not fit in the context (--ctx)");
    for (int i = 0; i < prompt->len; i++)
        if (prompt->v[i] < 0 || prompt->v[i] >= IK_VOCAB) return iks_fail(err, errlen, "token id out of range");
    ik_gpu *g = s->ikg;
    int common = 0;
    if (s->checkpoint_valid) {
        const int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
        while (common < n && s->checkpoint.v[common] == prompt->v[common]) common++;
    }
    int start = 0;
    const int restore = iks_restore_prefix(s, prompt);
    if (ds4_inkling2_env_flag("DS4_INKLING2_TRACE_SYNC"))
        fprintf(stderr, "ik-sync: prompt %d, checkpoint %d valid=%d, common %d, pos %u, snap_valid=%d snap_len=%d snap_pos=%u, restore=%d\n", prompt->len, s->checkpoint.len,
                (int)s->checkpoint_valid, common, g->pos, (int)s->snap_valid, s->snap_tokens.len, g->snap_pos, restore);
    if (s->checkpoint_valid && common == s->checkpoint.len && g->pos == (uint32_t)common) {
        if (common == prompt->len) return 0;      /* identical: the logits are current */
        start = common;                           /* the live state is a prefix of the prompt: extend it */
    } else if (restore > 0 && ikg_restore(g)) {   /* diverged after an earlier prompt whose end state we kept: go back to it */
        start = restore;
        s->checkpoint.len = 0;
        for (int i = 0; i < restore; i++) token_vec_push(&s->checkpoint, prompt->v[i]);
        memcpy(s->logits, s->snap_logits, (size_t)IK_VOCAB * sizeof(float));
        s->checkpoint_valid = true;
        if (restore == prompt->len) return 0;     /* the same prompt again: state and logits are the kept ones */
    } else {
        ikg_reset(g);                             /* diverged, shorter or stale: replay everything */
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
    }
    char tmp[200];
    const int rc = iks_run(s, prompt->v + start, prompt->len - start, true, start, prompt->len, err ? err : tmp, err ? errlen : sizeof(tmp));
    if (rc) return rc;
    s->checkpoint.len = 0;
    for (int i = 0; i < prompt->len; i++) token_vec_push(&s->checkpoint, prompt->v[i]);
    s->checkpoint_valid = true;
    iks_take_snapshot(s);
    return 0;
}

static int iks_eval(ds4_session *s, int token, char *err, size_t errlen) {
    if (!s) return iks_fail(err, errlen, "no session");
    if (token < 0 || token >= IK_VOCAB) return iks_fail(err, errlen, "token id out of range");
    if (s->checkpoint.len + 1 > s->ctx_size) return iks_fail(err, errlen, "context is full");
    if (iks_catch_up(s, err, errlen)) return 1;
    if (iks_run(s, &token, 1, true, s->checkpoint.len, s->checkpoint.len + 1, err, errlen)) return 1;
    token_vec_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}

static void iks_rewind(ds4_session *s, int pos) {
    if (!s) return;
    if (pos < 0) pos = 0;
    if (pos >= s->checkpoint.len) return;
    s->checkpoint.len = pos;   /* the GPU state is replayed on the next sync/eval */
    s->checkpoint_valid = false;
}

/* --- logits helpers: this family's vocabulary is IK_VOCAB, not DS4_N_VOCAB --- */

static int iks_argmax_excluding(ds4_session *s, int excluded) {
    if (!s || !s->checkpoint_valid || !s->logits) return -1;
    int best = -1;
    float bv = 0.0f;
    for (int i = 0; i < IK_VOCAB; i++) {
        if (i == excluded) continue;
        if (best < 0 || s->logits[i] > bv) { best = i; bv = s->logits[i]; }
    }
    return best;
}

static int iks_argmax_ignoring_eos(ds4_session *s, ds4_think_mode think_mode) {
    if (!s || !s->checkpoint_valid || !s->logits) return -1;
    int best = -1;
    float bv = 0.0f;
    for (int i = 0; i < IK_VOCAB; i++) {
        if (ds4_token_is_stop_for_think_mode(s->engine, i, think_mode)) continue;
        if (best < 0 || s->logits[i] > bv) { best = i; bv = s->logits[i]; }
    }
    return best;
}

static int iks_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    if (!s || !out || k <= 0 || !s->checkpoint_valid) return 0;
    if (k > IK_VOCAB) k = IK_VOCAB;
    for (int i = 0; i < k; i++) { out[i].id = -1; out[i].logit = DS4_NEG_INF; out[i].logprob = DS4_NEG_INF; }
    float mx = DS4_NEG_INF;
    for (int i = 0; i < IK_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > mx) mx = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(mx)) return 0;
    double sum = 0.0;
    for (int i = 0; i < IK_VOCAB; i++) if (isfinite(s->logits[i])) sum += exp((double)s->logits[i] - (double)mx);
    const double lse = (double)mx + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) out[i].logprob = (float)((double)out[i].logit - lse);
    return k;
}

static int iks_token_logprob(ds4_session *s, int token, ds4_token_score *out) {
    if (!s || !out || token < 0 || token >= IK_VOCAB || !s->checkpoint_valid) return 0;
    float mx = DS4_NEG_INF;
    for (int i = 0; i < IK_VOCAB; i++) if (isfinite(s->logits[i]) && s->logits[i] > mx) mx = s->logits[i];
    if (!isfinite(mx)) return 0;
    double sum = 0.0;
    for (int i = 0; i < IK_VOCAB; i++) if (isfinite(s->logits[i])) sum += exp((double)s->logits[i] - (double)mx);
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = (float)((double)out->logit - ((double)mx + log(sum)));
    return 1;
}

static int iks_copy_logits(ds4_session *s, float *out, int cap) {
    if (!s || !out || cap < IK_VOCAB || !s->checkpoint_valid) return 0;
    memcpy(out, s->logits, (size_t)IK_VOCAB * sizeof(float));
    return IK_VOCAB;
}

/* Batched decode: sessions that sit exactly at their checkpoint decode one token each in a single GPU pass (ikg_forward_multi); the
 * others (rewound, stale, full, bad token) take the single-session path with its error handling.  DS4_INKLING2_NO_BATCH=1 turns the
 * batched pass off (A/B runs, one session after the other as before). */
static bool iks_state_is_live(const ds4_session *s);
static ik_gpu *g_iks_scratch;
static uint32_t g_iks_scratch_cap;

static int iks_run_batch(ds4_session **grp, const int *tok, int ng, char *err, size_t errlen) {
    uint32_t need = 0;
    for (int i = 0; i < ng; i++) if ((uint32_t)grp[i]->ctx_size > need) need = (uint32_t)grp[i]->ctx_size;
    if (!g_iks_scratch || g_iks_scratch_cap < need) {
        ikg_free(g_iks_scratch);
        g_iks_scratch = ikg_new_ex(grp[0]->engine->ik, need, 8, 8, true);
        g_iks_scratch_cap = g_iks_scratch ? need : 0;
        if (!g_iks_scratch) return iks_fail(err, errlen, "cannot allocate the batched decode scratch");
    }
    ik_gpu *gs[8];
    float *lg[8];
    for (int i = 0; i < ng; i++) { gs[i] = grp[i]->ikg; lg[i] = grp[i]->logits; }
    const char *ferr = NULL;
    int bad[8] = {0};
    if (ikg_forward_multi(g_iks_scratch, gs, tok, ng, lg, bad, &ferr)) {
        if (err && errlen) snprintf(err, errlen, "inkling2 batched forward failed: %s", ferr ? ferr : "unknown error");
        for (int i = 0; i < ng; i++) iks_invalidate(grp[i]);   /* every state in the batch is half written */
        return 1;
    }
    /* A session whose logits are not finite fails alone: it is invalidated (position 0, no logits, the next sync replays it) and the
     * caller sees that its position did not advance.  The others are complete and go on. */
    for (int i = 0; i < ng; i++) {
        if (bad[i]) {
            fprintf(stderr, "ds4: inkling2: the logits of one session of a batched step contain NaN or infinity; that session is invalidated, the others go on\n");
            iks_invalidate(grp[i]);
            continue;
        }
        token_vec_push(&grp[i]->checkpoint, tok[i]);
        grp[i]->checkpoint_valid = true;
    }
    return 0;
}

static int iks_eval_batch(ds4_decode_item *items, int count, char *err, size_t errlen) {
    const bool batching = !ds4_inkling2_env_flag("DS4_INKLING2_NO_BATCH");   /* NO_BATCH=0 does not turn batching off */
    ds4_session *grp[8];
    int tok[8], ng = 0, rc = 0;
    for (int i = 0; i <= count && !rc; i++) {
        bool flush = i == count, single = false;
        if (i < count) {
            ds4_session *s = items[i].session;
            const int t = items[i].token;
            const bool ok = batching && iks_state_is_live(s) && t >= 0 && t < IK_VOCAB && s->checkpoint.len + 1 <= s->ctx_size;
            bool seen = false;
            for (int j = 0; j < ng; j++) if (grp[j] == s) seen = true;
            if (ok && !seen) { grp[ng] = s; tok[ng++] = t; flush = ng == 8; }
            else { single = true; flush = true; }
            if (single) {   /* finish the group first, then the odd one out */
                if (ng == 1) rc = iks_eval(grp[0], tok[0], err, errlen);
                else if (ng > 1) rc = iks_run_batch(grp, tok, ng, err, errlen);
                ng = 0;
                if (!rc) rc = iks_eval(s, t, err, errlen);
                continue;
            }
        }
        if (flush && ng > 0) {
            if (ng == 1) rc = iks_eval(grp[0], tok[0], err, errlen);
            else rc = iks_run_batch(grp, tok, ng, err, errlen);
            ng = 0;
        }
    }
    return rc;
}

/* Logit comparison for the self-test.  With DS4_INKLING2_EXACT=1 the kernels are fixed-order and replay must be bit-identical.
 * In the default (fast) mode live decode and replay through the prefill kernels round differently, and over a few hundred router
 * decisions at least one near-tie (8th vs 9th expert) usually flips, which moves some log-probabilities by a nat or more.  So the
 * default-mode check is only a sanity check (state corruption gives distributions that are nearly disjoint): same argmax and
 * total-variation distance between the two softmax distributions at most 0.75; the strict bit-for-bit checks run under EXACT. */
static bool iks_exact_mode(void) { return ds4_inkling2_exact_mode() != 0; }
static bool iks_logits_same(const float *a, const float *b) {
    if (iks_exact_mode()) return memcmp(a, b, IK_VOCAB * sizeof(float)) == 0;
    int ia = 0, ib = 0;
    for (int i = 1; i < IK_VOCAB; i++) { if (a[i] > a[ia]) ia = i; if (b[i] > b[ib]) ib = i; }
    double sa = 0, sb = 0;
    for (int i = 0; i < IK_VOCAB; i++) { sa += exp((double)(a[i] - a[ia])); sb += exp((double)(b[i] - b[ib])); }
    double tv = 0;
    for (int i = 0; i < IK_VOCAB; i++) tv += fabs(exp((double)(a[i] - a[ia])) / sa - exp((double)(b[i] - b[ib])) / sb);
    tv *= 0.5;
    if (getenv("DS4_INKLING2_SESSION_TEST_VERBOSE")) fprintf(stderr, "      logits compare: argmax %d vs %d, total variation %.4f\n", ia, ib, tv);
    return ia == ib && tv <= 0.75;
}

/* Known-answer check: a fixed 14-token prompt ("1, 2, 3, 4, 5," as GPT-4o ids) must give the answer the reference engines give.
 * Reference (llama.cpp, BF16 / NVFP4 / MXFP4 files): top-1 token 220 (a space) with log-probability -0.094 / -0.094 / -0.110, second
 * token 197017 with about -3.7.  The tolerances cover the three quantisations and the fast kernels (measured worst case 0.002 on
 * the CPU-oracle comparison); a build whose kernels produce wrong output (log-prob gaps of tens of nats) cannot pass.  The prompt is
 * longer than 8 tokens, so it runs the batched (prefill) kernels; the last token is also replayed through the one-token (decode)
 * kernels.  Returns the number of failed checks. */
static int iks_known_answer(ds4_engine *e) {
    static const int ka_prompt[14] = {16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11};
    int fails = 0;
    for (int mode = 0; mode < 2; mode++) {   /* 0: all 14 tokens in one batch; 1: 13 in a batch, the last through the decode kernels */
        ds4_session *s = NULL;
        char err[200] = "";
        ds4_tokens t = {0};
        const int n = mode == 0 ? 14 : 13;
        for (int i = 0; i < n; i++) token_vec_push(&t, ka_prompt[i]);
        bool ok = ds4_session_create(&s, e, 2048) == 0 && s && ds4_session_sync(s, &t, err, sizeof(err)) == 0;
        if (ok && mode == 1) ok = ds4_session_eval(s, ka_prompt[13], err, sizeof(err)) == 0;
        ds4_token_score top[2] = {{0}};
        if (ok) ok = ds4_session_top_logprobs(s, top, 2) == 2;
        const bool good = ok && top[0].id == 220 && fabsf(top[0].logprob + 0.094f) < 0.03f &&
                          top[1].id == 197017 && fabsf(top[1].logprob + 3.7f) < 0.3f;
        fprintf(stderr, "%s  known answer (%s): top-1 %d logprob %.4f, top-2 %d logprob %.4f (want 220 / -0.094 +-0.03, 197017 / -3.7 +-0.3)%s%s\n",
                good ? "PASS" : "FAIL", mode ? "decode kernel" : "batched kernels", ok ? top[0].id : -1, ok ? top[0].logprob : 0.0f,
                ok ? top[1].id : -1, ok ? top[1].logprob : 0.0f, ok ? "" : " error: ", ok ? "" : err);
        if (!good) fails++;
        if (s) ds4_session_free(s);
        ds4_tokens_free(&t);
    }
    return fails;
}

/* Batched decode checks: N sessions decoded in one pass equal the same sessions decoded one at a time.  EXACT: bit for bit; default
 * mode: same argmax every step and the worst top-5 log-probability difference is printed (limit 0.4 as for the 9k parity; outliers above 0.02 are listed). */
static int iks_batch_selftest(ds4_engine *e, const ds4_tokens *prompt, int *checks_out) {
    int fails = 0, checks = 0;
#define IKB_CHECK(cond, what) do { checks++; if (cond) fprintf(stderr, "PASS  %s\n", what); else { fails++; fprintf(stderr, "FAIL  %s\n", what); } } while (0)
    enum { NS = 5, STEPS = 12 };
    const int lens[NS] = { 30, 200, 1300, 77, 450 };
    ds4_session *bat[NS], *ref[NS];
    char err[200] = "";
    bool made = true;
    for (int i = 0; i < NS; i++) {
        bat[i] = ref[i] = NULL;
        made = made && ds4_session_create(&bat[i], e, 2048) == 0 && bat[i] && ds4_session_create(&ref[i], e, 2048) == 0 && ref[i];
    }
    IKB_CHECK(made, "batch test: 5 + 5 sessions (ctx 2048)");
    if (!made) { *checks_out += checks; return fails + 1; }
    float *la = xmalloc(IK_VOCAB * sizeof(float)), *lb = xmalloc(IK_VOCAB * sizeof(float));
    bool synced = true;
    for (int i = 0; i < NS; i++) {
        ds4_tokens P = {0};
        for (int k = 0; k < lens[i]; k++) token_vec_push(&P, prompt->v[(k * (3 + 2 * i) + i * 5) % prompt->len]);
        synced = synced && ds4_session_sync(bat[i], &P, err, sizeof(err)) == 0 && ds4_session_sync(ref[i], &P, err, sizeof(err)) == 0;
        ds4_tokens_free(&P);
    }
    IKB_CHECK(synced, "batch test: sessions of 30, 200, 1300 (ring wrapped), 77 and 450 tokens synced");
    bool all_same = true, argmax_same = true, batched_ok = true;
    double worst = 0.0;
    for (int step = 0; step < STEPS && batched_ok; step++) {
        ds4_decode_item items[NS];
        int n = 0, idx[NS];
        for (int i = 0; i < NS; i++) {
            if (step % 4 == 3 && i == 1) continue;   /* a session sitting out a step: the others still batch */
            items[n].session = bat[i];
            items[n].token = ds4_session_argmax(ref[i]);
            idx[n++] = i;
        }
        batched_ok = ds4_sessions_eval_batch(items, n, err, sizeof(err)) == 0;
        for (int j = 0; j < n && batched_ok; j++) {
            const int i = idx[j];
            if (ds4_session_eval(ref[i], items[j].token, err, sizeof(err))) { batched_ok = false; break; }
            ds4_session_copy_logits(bat[i], la, IK_VOCAB);
            ds4_session_copy_logits(ref[i], lb, IK_VOCAB);
            if (memcmp(la, lb, IK_VOCAB * sizeof(float)) != 0) all_same = false;
            ds4_token_score ta[5], tb[5];
            ds4_session_top_logprobs(bat[i], ta, 5);
            ds4_session_top_logprobs(ref[i], tb, 5);
            if (ta[0].id != tb[0].id) argmax_same = false;
            double sw = 0.0;
            for (int k = 0; k < 5; k++) for (int m = 0; m < 5; m++) if (ta[k].id == tb[m].id && fabs((double)ta[k].logprob - tb[m].logprob) > sw) sw = fabs((double)ta[k].logprob - tb[m].logprob);
            if (sw > worst) worst = sw;
            if (sw > 0.02 && !iks_exact_mode()) fprintf(stderr, "      step %d session %d (%d tokens): top-5 log-probability difference %.4f\n", step, i, ds4_session_pos(ref[i]), sw);
        }
    }
    IKB_CHECK(batched_ok, "batched decode of 5 sessions ran 12 steps without error");
    fprintf(stderr, "      worst top-5 log-probability difference batch vs one at a time: %.5f\n", worst);
    IKB_CHECK(argmax_same, "every step: same top-1 token as one at a time");
    if (iks_exact_mode()) IKB_CHECK(all_same, "EXACT: batched logits equal one-at-a-time logits bit for bit"); else IKB_CHECK(worst <= 0.4, "default: worst top-5 log-probability difference within the documented decode tolerance (0.4)");
    /* the batch left every session in a state that continues correctly one at a time */
    bool cont = true;
    for (int i = 0; i < NS; i++) {
        const int t = ds4_session_argmax(ref[i]);
        cont = cont && ds4_session_eval(bat[i], t, err, sizeof(err)) == 0 && ds4_session_eval(ref[i], t, err, sizeof(err)) == 0;
        ds4_token_score ta[2], tb[2];
        ds4_session_top_logprobs(bat[i], ta, 2);
        ds4_session_top_logprobs(ref[i], tb, 2);
        cont = cont && ta[0].id == tb[0].id && fabs((double)ta[0].logprob - tb[0].logprob) <= 0.05;
    }
    IKB_CHECK(cont, "after a batch each session continues one at a time with the same answer");
    /* a rewound session in the batch falls back to the single path and the others still batch */
    ds4_session_rewind(bat[0], ds4_session_pos(bat[0]) - 3);
    ds4_session_rewind(ref[0], ds4_session_pos(ref[0]) - 3);
    {
        ds4_decode_item items[3] = { { bat[0], 220 }, { bat[1], 220 }, { bat[2], 220 } };
        const int rc = ds4_sessions_eval_batch(items, 3, err, sizeof(err));
        const int rr = ds4_session_eval(ref[0], 220, err, sizeof(err)) | ds4_session_eval(ref[1], 220, err, sizeof(err)) | ds4_session_eval(ref[2], 220, err, sizeof(err));
        ds4_token_score ta, tb;
        ds4_session_top_logprobs(bat[0], &ta, 1);
        ds4_session_top_logprobs(ref[0], &tb, 1);
        IKB_CHECK(rc == 0 && rr == 0 && ta.id == tb.id, "a rewound session in a batch is replayed, the rest batch");
    }
    /* NaN logits in one session of a batched step: only that session fails (invalidated, no logits, the next sync replays it); the
     * others advance and equal a one-at-a-time step.  The test hook poisons row 1 of the batch. */
    {
        ds4_decode_item items[3] = { { bat[0], 220 }, { bat[1], 220 }, { bat[2], 220 } };
        const int p0 = ds4_session_pos(bat[0]), p1 = ds4_session_pos(bat[1]), p2 = ds4_session_pos(bat[2]);
        setenv("DS4_INKLING2_TEST_POISON_ROW", "1", 1);
        const int rc = ds4_sessions_eval_batch(items, 3, err, sizeof(err));
        unsetenv("DS4_INKLING2_TEST_POISON_ROW");
        IKB_CHECK(rc == 0, "NaN in one session of a batched step: the call itself succeeds");
        IKB_CHECK(ds4_session_pos(bat[0]) == p0 + 1 && ds4_session_pos(bat[2]) == p2 + 1, "the other sessions of the step advanced");
        IKB_CHECK(ds4_session_pos(bat[1]) != p1 + 1 && ds4_session_argmax(bat[1]) == -1, "the poisoned session did not advance and has no logits");
        const int r0 = ds4_session_eval(ref[0], 220, err, sizeof(err)), r2 = ds4_session_eval(ref[2], 220, err, sizeof(err));
        ds4_session_copy_logits(bat[0], la, IK_VOCAB);
        ds4_session_copy_logits(ref[0], lb, IK_VOCAB);
        bool same = r0 == 0 && iks_logits_same(la, lb);
        ds4_session_copy_logits(bat[2], la, IK_VOCAB);
        ds4_session_copy_logits(ref[2], lb, IK_VOCAB);
        same = same && r2 == 0 && iks_logits_same(la, lb);
        IKB_CHECK(same, "the other sessions' logits equal a one-at-a-time step");
        const int r1 = ds4_session_eval(ref[1], 220, err, sizeof(err));
        IKB_CHECK(r1 == 0 && ds4_session_sync(bat[1], ds4_session_tokens(ref[1]), err, sizeof(err)) == 0 && ds4_session_pos(bat[1]) == ds4_session_pos(ref[1]),
                  "the poisoned session recovers with a sync");
        ds4_session_copy_logits(bat[1], la, IK_VOCAB);
        ds4_session_copy_logits(ref[1], lb, IK_VOCAB);
        IKB_CHECK(iks_logits_same(la, lb), "and then equals the one-at-a-time session");
    }
    /* a bad token is refused and the others are untouched */
    {
        ds4_decode_item items[2] = { { bat[3], 220 }, { bat[4], 999999999 } };
        IKB_CHECK(ds4_sessions_eval_batch(items, 2, err, sizeof(err)) != 0 && err[0], "a bad token in a batch is refused with a message");
    }
    for (int i = 0; i < NS; i++) { ds4_session_free(bat[i]); ds4_session_free(ref[i]); }
    free(la); free(lb);
#undef IKB_CHECK
    *checks_out += checks;
    return fails;
}

/* Context boundary checks: prompts of ctx and ctx+1 tokens are refused up front (cheap, no prefill); a prompt of ctx-1 tokens is accepted, one
 * more token fits, the next one is refused as "context is full"; the session stays usable.  DS4_INKLING2_BOUNDARY_CTX=N sets the context
 * (default 1024); for a big N only the two refusals and a short prompt run (a prompt of N-1 tokens would be a full-context prefill). */
static int iks_boundary_selftest(ds4_engine *e, const ds4_tokens *prompt, int *checks_out) {
    int fails = 0, checks = 0;
#define IKC_CHECK(cond, what) do { checks++; if (cond) fprintf(stderr, "PASS  %s\n", what); else { fails++; fprintf(stderr, "FAIL  %s\n", what); } } while (0)
    const char *env = getenv("DS4_INKLING2_BOUNDARY_CTX");
    const int ctx = env && atoi(env) > 8 ? atoi(env) : 1024;
    char err[200] = "";
    ds4_session *s = NULL;
    IKC_CHECK(ds4_session_create(&s, e, ctx) == 0 && s, "boundary test: session at the chosen context");
    if (!s) { *checks_out += checks; return fails + 1; }
    ds4_tokens P = {0};
    for (int k = 0; k < ctx + 1; k++) token_vec_push(&P, prompt->v[(k * 7 + k / 5) % prompt->len]);
    ds4_tokens Q = {0};
    Q.v = P.v; Q.len = ctx; Q.cap = P.cap;
    IKC_CHECK(ds4_session_sync(s, &Q, err, sizeof(err)) != 0 && strstr(err, "does not fit"), "a prompt of exactly ctx tokens is refused up front");
    IKC_CHECK(ds4_session_sync(s, &P, err, sizeof(err)) != 0 && strstr(err, "does not fit"), "a prompt of ctx+1 tokens is refused up front");
    if (ctx <= 4096) {
        Q.len = ctx - 1;
        IKC_CHECK(ds4_session_sync(s, &Q, err, sizeof(err)) == 0 && ds4_session_pos(s) == ctx - 1, "a prompt of ctx-1 tokens is accepted");
        IKC_CHECK(ds4_session_eval(s, P.v[0], err, sizeof(err)) == 0 && ds4_session_pos(s) == ctx, "one more token fits (position reaches ctx)");
        IKC_CHECK(ds4_session_eval(s, P.v[1], err, sizeof(err)) != 0 && strstr(err, "full"), "the next token is refused: context is full");
        ds4_tokens B = {0};
        token_vec_push(&B, ds4_session_tokens(s)->v[0]);
        token_vec_push(&B, 220);
        IKC_CHECK(ds4_session_sync(s, &B, err, sizeof(err)) == 0 && ds4_session_argmax(s) >= 0, "afterwards the session syncs a new short prompt");
        ds4_tokens_free(&B);
    } else {
        ds4_tokens B = {0};
        for (int k = 0; k < 40; k++) token_vec_push(&B, P.v[k]);
        IKC_CHECK(ds4_session_sync(s, &B, err, sizeof(err)) == 0 && ds4_session_argmax(s) >= 0, "a short prompt works at this context");
        ds4_tokens_free(&B);
    }
    ds4_session_free(s);
    Q.v = NULL;
    ds4_tokens_free(&P);
#undef IKC_CHECK
    *checks_out += checks;
    return fails;
}

/* Disk KV payload checks: restore then continue equals never having saved; damaged or foreign files are refused and leave the session usable. */
static int iks_payload_selftest(ds4_engine *e, const ds4_tokens *prompt, int *checks_out) {
    int fails = 0, checks = 0;
#define IKP_CHECK(cond, what) do { checks++; if (cond) fprintf(stderr, "PASS  %s\n", what); else { fails++; fprintf(stderr, "FAIL  %s\n", what); } } while (0)
    char err[200] = "";
    const int plen = 1300, nmore = 6;   /* beyond the 1024-row window: the ring has wrapped */
    ds4_tokens P = {0};
    for (int i = 0; i < plen; i++) token_vec_push(&P, prompt->v[(i * 7 + i / 3) % prompt->len]);
    ds4_session *a = NULL, *b = NULL;
    IKP_CHECK(ds4_session_create(&a, e, 2048) == 0 && a && ds4_session_create(&b, e, 2048) == 0 && b, "payload test: two sessions (ctx 2048)");
    if (!a || !b) { *checks_out += checks; return fails + 1; }
    IKP_CHECK(ds4_session_payload_bytes(a) == 0, "payload size of a session without state is 0");
    IKP_CHECK(ds4_session_sync(a, &P, err, sizeof(err)) == 0, "payload test: sync 1300 tokens");
    for (int i = 0; i < 3; i++) ds4_session_eval(a, P.v[i], err, sizeof(err));
    const uint64_t want = ds4_session_payload_bytes(a);
    FILE *f = tmpfile();
    IKP_CHECK(f && ds4_session_save_payload(a, f, err, sizeof(err)) == 0 && (uint64_t)ftello(f) == want && want > 0, "save writes exactly payload_bytes");
    fprintf(stderr, "      payload %llu bytes (%.1f MiB) for %d tokens\n", (unsigned long long)want, (double)want / 1048576.0, ds4_session_pos(a));
    uint8_t *img = xmalloc((size_t)want);
    rewind(f);
    const bool rd = fread(img, 1, (size_t)want, f) == (size_t)want;
    fclose(f);
    IKP_CHECK(rd, "payload read back");
    float *la = xmalloc(IK_VOCAB * sizeof(float)), *lb = xmalloc(IK_VOCAB * sizeof(float));
#define IKP_LOAD(sess, buf, len, claim) ({ FILE *ff = tmpfile(); fwrite((buf), 1, (len), ff); rewind(ff); int r_ = ds4_session_load_payload((sess), ff, (claim), err, sizeof(err)); fclose(ff); r_; })
    IKP_CHECK(IKP_LOAD(b, img, (size_t)want, want) == 0 && ds4_session_pos(b) == plen + 3, "restore into a second session");
    ds4_session_copy_logits(a, la, IK_VOCAB);
    ds4_session_copy_logits(b, lb, IK_VOCAB);
    IKP_CHECK(memcmp(la, lb, IK_VOCAB * sizeof(float)) == 0, "restored logits are the saved ones, bit for bit");
    bool same = true;
    for (int i = 0; i < nmore && same; i++) {
        const int t = P.v[(100 + i * 13) % plen];
        ds4_session_eval(a, t, err, sizeof(err));
        ds4_session_eval(b, t, err, sizeof(err));
        ds4_session_copy_logits(a, la, IK_VOCAB);
        ds4_session_copy_logits(b, lb, IK_VOCAB);
        same = memcmp(la, lb, IK_VOCAB * sizeof(float)) == 0;
    }
    IKP_CHECK(same, "continuing after restore gives the same logits as never having saved (6 tokens, bit for bit)");
    /* an incremental follow-up: restore, then sync a longer prompt: only the new tokens are evaluated */
    ds4_tokens P2 = {0};
    for (int i = 0; i < P.len; i++) token_vec_push(&P2, P.v[i]);
    for (int i = 0; i < 3; i++) token_vec_push(&P2, P.v[i]);
    for (int i = 0; i < 20; i++) token_vec_push(&P2, (P.v[i] + 5) % IK_VOCAB);
    IKP_CHECK(IKP_LOAD(b, img, (size_t)want, want) == 0 && ds4_session_sync(b, &P2, err, sizeof(err)) == 0 && ds4_session_pos(b) == P2.len,
              "after a restore, a longer prompt only evaluates the new tokens");
    /* refusals */
    uint8_t *bad = xmalloc((size_t)want);
    memcpy(bad, img, (size_t)want);
    IKP_CHECK(IKP_LOAD(b, bad, (size_t)want - 5000, want - 5000) != 0 && err[0], "a truncated payload is refused (size does not match the header)");
    fprintf(stderr, "      -> %s\n", err);
    IKP_CHECK(ds4_session_pos(b) == P2.len, "a refusal at the header leaves the session state untouched");
    IKP_CHECK(IKP_LOAD(b, bad, (size_t)want - 5000, want) != 0 && err[0], "a file shorter than its recorded size is refused");
    fprintf(stderr, "      -> %s\n", err);
    bad[want / 2] ^= 0x10;
    IKP_CHECK(IKP_LOAD(b, bad, (size_t)want, want) != 0 && strstr(err, "checksum"), "one flipped bit in the K/V data is refused");
    fprintf(stderr, "      -> %s\n", err);
    IKP_CHECK(ds4_session_argmax(b) == -1, "a refused payload leaves no live state");
    bad[want / 2] ^= 0x10;
    bad[12 * 4] ^= 0x01;
    IKP_CHECK(IKP_LOAD(b, bad, (size_t)want, want) != 0 && strstr(err, "another model family"), "a payload of another family (tag) is refused");
    fprintf(stderr, "      -> %s\n", err);
    bad[12 * 4] ^= 0x01;
    bad[4 * 4] ^= 0x01;
    IKP_CHECK(IKP_LOAD(b, bad, (size_t)want, want) != 0 && strstr(err, "another model file"), "a payload of another model file is refused");
    fprintf(stderr, "      -> %s\n", err);
    bad[4 * 4] ^= 0x01;
    bad[0] ^= 0x01;
    IKP_CHECK(IKP_LOAD(b, bad, (size_t)want, want) != 0, "a payload with a wrong magic is refused");
    bad[0] ^= 0x01;
    ds4_session *c = NULL;
    IKP_CHECK(ds4_session_create(&c, e, 1024) == 0 && c && IKP_LOAD(c, img, (size_t)want, want) != 0 && strstr(err, "different --ctx"), "a payload written with another --ctx is refused");
    fprintf(stderr, "      -> %s\n", err);
    if (c) ds4_session_free(c);
    /* after refusals the session is fully usable and equals a cold run */
    IKP_CHECK(ds4_session_sync(b, &P, err, sizeof(err)) == 0 && ds4_session_pos(b) == plen, "after refusals the session syncs from scratch");
    ds4_session_free(a);
    ds4_session_free(b);
    free(img); free(bad); free(la); free(lb);
    ds4_tokens_free(&P); ds4_tokens_free(&P2);
#undef IKP_LOAD
#undef IKP_CHECK
    *checks_out += checks;
    return fails;
}

static int iks_test_cancel_calls;
static bool iks_test_cancel_after_first(void *ud) { (void)ud; return ++iks_test_cancel_calls >= 1; }

/* DS4_INKLING2_SESSION_TEST=1 with --metal --first-token-test: exercises the public session API the way the CLI and
 * server use it (sync, eval, rewind, replay, error paths) and prints one PASS/FAIL line per check. */
static int iks_selftest(ds4_engine *e, const ds4_tokens *prompt_in) {
    int fails = 0, checks = 0;
    ds4_tokens prompt_env = {0};
    const ds4_tokens *prompt = prompt_in;
    const char *tok_env = getenv("DS4_INKLING2_FT_TOKENS");   /* same convention as the oracle driver */
    if (tok_env && tok_env[0]) {
        int *v = NULL;
        const int nv = ik_parse_ints(tok_env, &v);
        for (int i = 0; i < nv; i++) token_vec_push(&prompt_env, v[i]);
        free(v);
        prompt = &prompt_env;
    }
    if (prompt->len < 2) { fprintf(stderr, "session test needs a prompt of at least two tokens\n"); return 1; }
    { const int kf = iks_known_answer(e); checks += 2; fails += kf; }
#define IKS_CHECK(cond, what) do { checks++; if (cond) fprintf(stderr, "PASS  %s\n", what); else { fails++; fprintf(stderr, "FAIL  %s\n", what); } } while (0)
    char err[200] = "";
    ds4_session *s = NULL;
    IKS_CHECK(ds4_session_create(&s, e, 2048) == 0 && s, "session create (ctx 2048)");
    if (!s) return 1;
    const int plen = prompt->len;
    float *L[16];
    for (int i = 0; i < 16; i++) L[i] = xmalloc(IK_VOCAB * sizeof(float));
    ds4_tokens empty = {0};
    IKS_CHECK(ds4_session_sync(s, &empty, err, sizeof(err)) != 0 && err[0], "sync of an empty prompt is refused with a message");
    IKS_CHECK(ds4_session_eval(s, 999999999, err, sizeof(err)) != 0, "eval of an out-of-range token is refused");
    IKS_CHECK(ds4_session_argmax(s) == -1, "argmax of a session without logits returns -1");
    IKS_CHECK(ds4_session_sync(s, prompt, err, sizeof(err)) == 0 && ds4_session_pos(s) == plen, "sync of the prompt");
    IKS_CHECK(ds4_session_copy_logits(s, L[0], IK_VOCAB) == IK_VOCAB, "copy logits");
    int toks[8];
    for (int i = 0; i < 6; i++) {
        toks[i] = ds4_session_argmax(s);
        if (ds4_session_eval(s, toks[i], err, sizeof(err)) != 0) { IKS_CHECK(false, "eval during the greedy run"); break; }
        ds4_session_copy_logits(s, L[i + 1], IK_VOCAB);
    }
    IKS_CHECK(ds4_session_pos(s) == plen + 6, "six greedy evals advance the position");
    /* non-finite logits are an error returned to the caller (not token 0), and the session recovers after a sync */
    {
        setenv("DS4_INKLING2_TEST_POISON_LOGITS", "1", 1);
        char perr[200] = "";
        const int prc = ds4_session_eval(s, toks[0], perr, sizeof(perr));
        unsetenv("DS4_INKLING2_TEST_POISON_LOGITS");
        IKS_CHECK(prc != 0 && strstr(perr, "NaN"), "NaN logits make eval fail with a message");
        IKS_CHECK(ds4_session_argmax(s) == -1, "after a NaN failure the session has no logits (argmax -1, not token 0)");
        ds4_tokens again = {0};
        for (int i = 0; i < plen; i++) token_vec_push(&again, prompt->v[i]);
        for (int i = 0; i < 6; i++) token_vec_push(&again, toks[i]);
        IKS_CHECK(ds4_session_sync(s, &again, err, sizeof(err)) == 0 && ds4_session_pos(s) == plen + 6, "the session recovers with a sync");
        ds4_tokens_free(&again);
    }
    ds4_token_score top[20], one;
    IKS_CHECK(ds4_session_top_logprobs(s, top, 20) == 20 && top[0].id == ds4_session_argmax(s) && top[0].logprob <= 0.0f, "top_logprobs agrees with argmax");
    double psum = 0;
    for (int i = 0; i < 20; i++) psum += exp((double)top[i].logprob);
    IKS_CHECK(psum > 0.0 && psum <= 1.0 + 1e-4, "top-20 probabilities sum to at most 1");
    IKS_CHECK(ds4_session_token_logprob(s, top[3].id, &one) == 1 && fabsf(one.logprob - top[3].logprob) < 1e-5f, "token_logprob matches top_logprobs");
    /* rewind to after toks[2], then continue without a sync: the GPU state is replayed on the next eval */
    ds4_session_rewind(s, plen + 3);
    IKS_CHECK(ds4_session_pos(s) == plen + 3 && ds4_session_argmax(s) == -1, "rewind keeps the prefix and drops the logits");
    IKS_CHECK(ds4_session_eval(s, toks[3], err, sizeof(err)) == 0 && ds4_session_pos(s) == plen + 4, "eval after rewind replays the prefix");
    float *cur = xmalloc(IK_VOCAB * sizeof(float));
    ds4_session_copy_logits(s, cur, IK_VOCAB);
    IKS_CHECK(iks_logits_same(cur, L[4]), "logits after rewind+replay+eval match the first run (EXACT: bit-identical)");
    ds4_session_eval(s, toks[4], err, sizeof(err)); ds4_session_eval(s, toks[5], err, sizeof(err));
    ds4_session_copy_logits(s, cur, IK_VOCAB);
    IKS_CHECK(iks_logits_same(cur, L[6]), "two more evals reproduce the first run (EXACT: bit for bit)");
    /* sync after a rewind: same tokens again */
    ds4_tokens full = {0};
    for (int i = 0; i < plen; i++) token_vec_push(&full, prompt->v[i]);
    for (int i = 0; i < 6; i++) token_vec_push(&full, toks[i]);
    ds4_session_rewind(s, plen);
    IKS_CHECK(ds4_session_sync(s, &full, err, sizeof(err)) == 0 && ds4_session_pos(s) == plen + 6, "sync of prompt + six tokens after a rewind");
    ds4_session_copy_logits(s, cur, IK_VOCAB);
    IKS_CHECK(iks_logits_same(cur, L[6]), "logits after the replaying sync match (EXACT: bit-identical)");
    IKS_CHECK(ds4_session_sync(s, &full, err, sizeof(err)) == 0 && ds4_session_pos(s) == plen + 6, "sync of an identical prompt is a no-op");
    /* extension: sync a longer prompt on top of the live state */
    token_vec_push(&full, toks[0]);
    IKS_CHECK(ds4_session_sync(s, &full, err, sizeof(err)) == 0 && ds4_session_pos(s) == plen + 7, "sync extends the live state");
    /* divergence: change the last token */
    full.v[full.len - 1] = (toks[0] + 1) % IK_VOCAB;
    IKS_CHECK(ds4_session_sync(s, &full, err, sizeof(err)) == 0 && ds4_session_pos(s) == plen + 7, "sync of a diverging prompt rebuilds");
    ds4_session_copy_logits(s, cur, IK_VOCAB);
    /* a fresh session on the same tokens must give the same logits */
    ds4_session *s2 = NULL;
    IKS_CHECK(ds4_session_create(&s2, e, 2048) == 0 && s2, "second session");
    if (s2) {
        IKS_CHECK(ds4_session_sync(s2, &full, err, sizeof(err)) == 0, "sync in the second session");
        float *c2 = xmalloc(IK_VOCAB * sizeof(float));
        ds4_session_copy_logits(s2, c2, IK_VOCAB);
        IKS_CHECK(iks_logits_same(cur, c2), "two sessions agree (live decode vs fresh prefill; EXACT: bit for bit)");
        free(c2);
        ds4_session_free(s2);
    }
    ds4_tokens shortp = {0};
    for (int i = 0; i < plen - 1; i++) token_vec_push(&shortp, prompt->v[i]);
    const int src = ds4_session_sync(s, &shortp, err, sizeof(err));
    if (src || ds4_session_pos(s) != plen - 1) fprintf(stderr, "      sync rc=%d err='%s' pos=%d want %d\n", src, err, ds4_session_pos(s), plen - 1);
    IKS_CHECK(src == 0 && ds4_session_pos(s) == plen - 1, "sync of a strict prefix replays");
    ds4_tokens big = {0};
    for (int i = 0; i < 2048; i++) token_vec_push(&big, prompt->v[i % plen]);
    IKS_CHECK(ds4_session_sync(s, &big, err, sizeof(err)) != 0 && err[0], "a prompt that fills the context is refused");
    { const int pf = iks_payload_selftest(e, prompt, &checks); fails += pf; }
    { const int bf = iks_batch_selftest(e, prompt, &checks); fails += bf; }
    { const int cf = iks_boundary_selftest(e, prompt, &checks); fails += cf; }
    int acc[4];
    IKS_CHECK(ds4_session_eval_speculative_argmax(s, 1, 4, -1, acc, 4, err, sizeof(err)) < 0 && strstr(err, "not supported"), "speculative decoding is refused");
    /* prefix reuse after the live state diverged (a client that drops reasoning_content): restore the kept end state of the earlier prompt */
    {
        ds4_tokens pa = {0}, pb = {0}, pc = {0};
        for (int i = 0; i < plen; i++) token_vec_push(&pa, prompt->v[i]);
        for (int i = 0; i < plen; i++) token_vec_push(&pb, prompt->v[i]);
        for (int i = 0; i < 3; i++) token_vec_push(&pb, (toks[0] + 7 + i) % IK_VOCAB);     /* not what the model generated */
        for (int i = 0; i < plen; i++) token_vec_push(&pc, prompt->v[i]);
        pc.v[plen - 1] = (pc.v[plen - 1] + 1) % IK_VOCAB;                                    /* differs inside the kept prompt */
        float *ref = xmalloc(IK_VOCAB * sizeof(float));
        IKS_CHECK(ds4_session_sync(s, &pa, err, sizeof(err)) == 0, "sync of the prompt (keeps its end state)");
        ds4_session_copy_logits(s, ref, IK_VOCAB);
        for (int i = 0; i < 6; i++) ds4_session_eval(s, toks[i], err, sizeof(err));
        IKS_CHECK(ds4_session_restore_prefix(s, &pb) == plen, "after decoding, a prompt that diverges right after the old prompt can restore it");
        IKS_CHECK(ds4_session_restore_prefix(s, &pc) == 0, "a prompt that differs inside the old prompt cannot");
        IKS_CHECK(ds4_session_sync(s, &pb, err, sizeof(err)) == 0 && ds4_session_pos(s) == plen + 3, "sync restores the old prompt's state and evaluates the new tokens");
        ds4_session *s3 = NULL;
        float *fresh = xmalloc(IK_VOCAB * sizeof(float)), *cur2 = xmalloc(IK_VOCAB * sizeof(float)), *kept = xmalloc(IK_VOCAB * sizeof(float));
        ds4_session_copy_logits(s, kept, IK_VOCAB);
        if (ds4_session_create(&s3, e, 2048) == 0 && s3 && ds4_session_sync(s3, &pb, err, sizeof(err)) == 0) {
            ds4_session_copy_logits(s3, fresh, IK_VOCAB);
            IKS_CHECK(iks_logits_same(kept, fresh), "restored state gives the logits of a cold prefill (EXACT: bit for bit)");
            ds4_session_eval(s3, toks[0], err, sizeof(err));
            ds4_session_copy_logits(s3, fresh, IK_VOCAB);
        } else IKS_CHECK(false, "second session for the cold comparison");
        ds4_session_eval(s, toks[0], err, sizeof(err));
        ds4_session_copy_logits(s, cur2, IK_VOCAB);
        IKS_CHECK(iks_logits_same(cur2, fresh), "decoding after a restore matches the cold session");
        if (s3) ds4_session_free(s3);
        for (int i = 1; i < 6; i++) ds4_session_eval(s, toks[i], err, sizeof(err));
        IKS_CHECK(ds4_session_sync(s, &pb, err, sizeof(err)) == 0 && ds4_session_pos(s) == plen + 3, "the same prompt again restores its end state");
        ds4_session_copy_logits(s, cur2, IK_VOCAB);
        IKS_CHECK(memcmp(cur2, kept, IK_VOCAB * sizeof(float)) == 0, "and its logits are the kept ones, bit for bit");
        IKS_CHECK(ds4_session_sync(s, &pc, err, sizeof(err)) == 0 && ds4_session_pos(s) == plen, "a prompt that differs inside the old prompt replays from zero");
        free(ref); free(fresh); free(cur2); free(kept);
        ds4_tokens_free(&pa); ds4_tokens_free(&pb); ds4_tokens_free(&pc);
    }
    /* A cancelled or failed prefill of another conversation must not leave the kept end state of the earlier prompt restorable:
     * its global K/V rows were overwritten by the replay.  Two ways to break the replay: cancel after the first chunk, and a
     * NaN failure in the last one.  The follow-up of the first conversation must equal a cold run. */
    for (int mode = 0; mode < 2; mode++) {
        ds4_tokens pa = {0}, pa2 = {0}, pc = {0};
        for (int i = 0; i < 600; i++) token_vec_push(&pa, prompt->v[(i * 3) % plen]);
        for (int i = 0; i < 1100; i++) token_vec_push(&pc, (prompt->v[(i * 5 + 1) % plen] + 1 + i / plen) % IK_VOCAB);
        for (int i = 0; i < 600; i++) token_vec_push(&pa2, pa.v[i]);
        for (int i = 0; i < 8; i++) token_vec_push(&pa2, (toks[i % 6] + i) % IK_VOCAB);
        float *cold = xmalloc(IK_VOCAB * sizeof(float)), *got = xmalloc(IK_VOCAB * sizeof(float));
        ds4_session *sc = NULL;
        IKS_CHECK(ds4_session_create(&sc, e, 2048) == 0 && sc && ds4_session_sync(sc, &pa2, err, sizeof(err)) == 0, "stale-state test: cold run of the follow-up");
        if (sc) { ds4_session_copy_logits(sc, cold, IK_VOCAB); ds4_session_free(sc); }
        IKS_CHECK(ds4_session_sync(s, &pa, err, sizeof(err)) == 0, mode ? "stale-state test (failed prefill): sync of conversation A" : "stale-state test (cancel): sync of conversation A");
        for (int i = 0; i < 5; i++) ds4_session_eval(s, toks[i], err, sizeof(err));
        int rc;
        if (mode == 0) {
            iks_test_cancel_calls = 0;
            ds4_session_set_cancel(s, iks_test_cancel_after_first, NULL);
            rc = ds4_session_sync(s, &pc, err, sizeof(err));
            ds4_session_set_cancel(s, NULL, NULL);
            IKS_CHECK(rc != 0 && strstr(err, "cancel"), "a cancelled prefill of conversation C reports the cancel");
        } else {
            setenv("DS4_INKLING2_TEST_POISON_LOGITS", "1", 1);
            rc = ds4_session_sync(s, &pc, err, sizeof(err));
            unsetenv("DS4_INKLING2_TEST_POISON_LOGITS");
            IKS_CHECK(rc != 0 && strstr(err, "NaN"), "a failed prefill (NaN) of conversation C reports the failure");
        }
        IKS_CHECK(ds4_session_restore_prefix(s, &pa2) == 0, "after the broken prefill the old prompt is no longer restorable");
        IKS_CHECK(ds4_session_sync(s, &pa2, err, sizeof(err)) == 0 && ds4_session_pos(s) == pa2.len, "follow-up of conversation A syncs");
        ds4_session_copy_logits(s, got, IK_VOCAB);
        IKS_CHECK(iks_logits_same(got, cold), "follow-up of A equals a cold run, not a mix with C (EXACT: bit for bit)");
        free(cold); free(got);
        ds4_tokens_free(&pa); ds4_tokens_free(&pa2); ds4_tokens_free(&pc);
    }
    ds4_session_free(s);
    ds4_tokens_free(&full); ds4_tokens_free(&shortp); ds4_tokens_free(&big);
    for (int i = 0; i < 16; i++) free(L[i]);
    free(cur);
    ds4_tokens_free(&prompt_env);
    fprintf(stderr, "inkling2 session test: %d checks, %d failed\n", checks, fails);
#undef IKS_CHECK
    return fails ? 1 : 0;
}

/* ds4_engine_generate_argmax (CLI greedy path) for this family: a session loop, no sampling. */
static int iks_generate_argmax(ds4_engine *e, const ds4_tokens *prompt, int n_predict, int ctx_size,
                               ds4_token_emit_fn emit, ds4_generation_done_fn done, void *emit_ud,
                               ds4_session_progress_fn progress, void *progress_ud) {
    ds4_session *s = NULL;
    char err[256] = "";
    if (ds4_session_create(&s, e, ctx_size) != 0) return 1;
    const double t0 = ik_now();
    ds4_session_set_progress(s, progress, progress_ud);
    if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(s, NULL, NULL);
        fprintf(stderr, "ds4: prompt processing failed: %s\n", err);
        ds4_session_free(s);
        return 1;
    }
    ds4_session_set_progress(s, NULL, NULL);
    const double t1 = ik_now();
    int rc = 0, n_generated = 0;
    int token = ds4_session_argmax(s);
    for (int i = 0; i < n_predict && ds4_session_pos(s) < ctx_size; i++) {
        if (token < 0) { fprintf(stderr, "ds4: argmax failed\n"); rc = 1; break; }
        if (ds4_token_is_stop(e, token)) break;
        if (emit) emit(emit_ud, token);
        n_generated++;
        if (i == n_predict - 1 || ds4_session_pos(s) + 1 >= ctx_size) break;
        if (ds4_session_eval(s, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: decode failed: %s\n", err);
            rc = 1;
            break;
        }
        token = ds4_session_argmax(s);
    }
    const double t2 = ik_now();
    if (done) done(emit_ud);
    ds4_log(stderr, DS4_LOG_TIMING, "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            t1 > t0 ? (double)prompt->len / (t1 - t0) : 0.0, t2 > t1 ? (double)n_generated / (t2 - t1) : 0.0);
    ds4_session_free(s);
    return rc;
}

/* --- disk KV payload -------------------------------------------------------------------------------------------------------
 * What a session saves so that restore-then-continue equals never having saved:
 *   13 u32 header (same field count as the other families; h[12] is the family tag IKS_PAYLOAD_TAG, so a DeepSeek/Qwen/GLM payload is never
 *   read here and the reverse), the token ids, the f32 logits of the last token, then per layer-visit slot (48): the conv tails (3 rows of
 *   6656 f32), K rows and V rows as raw half.  Global slots keep rows 0..ntok-1; window slots keep only the last min(ntok, 1024) positions in
 *   position order (the ring is rebuilt at (p % rows), so a payload does not depend on the chunk size or on the ring length).  A 64-bit
 *   checksum over everything after the header closes the payload; a truncated, bit-flipped or foreign file never becomes a live state.
 *   Header: h[2] = --ctx the file was written with (must match: the global slots are sized by it), h[4]/h[5] = fingerprint of the model file
 *   (size + sampled bytes), h[6] slots, h[7] tokens, h[8] layers, h[9] K/V row width, h[10] conv channels, h[11] vocabulary. */
#define IKS_PAYLOAD_TAG 0x494B3202u
#define IKS_IO_CHUNK (4u << 20)

static uint64_t iks_hash_block(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, b + i, 8);
        h = (h ^ w) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 31;
    }
    for (; i < n; i++) h = (h ^ b[i]) * 0x100000001B3ull;
    return (h ^ (h >> 29)) + n;
}

static uint64_t iks_model_fingerprint(const ds4_engine *e) {
    const ds4_model *m = e->ik->m;
    uint64_t h = 0xcbf29ce484222325ull ^ (uint64_t)m->size;
    for (int i = 0; i < 32; i++) {
        const uint64_t off = (uint64_t)((__uint128_t)m->size * (uint64_t)i / 32u) & ~4095ull;
        const uint64_t n = off + 4096 <= m->size ? 4096 : (m->size > off ? m->size - off : 0);
        h = iks_hash_block(h, (const uint8_t *)m->map + off, (size_t)n);
    }
    return h;
}

static uint64_t iks_live_rows(int slot, uint32_t ntok) {
    return ik_layer_is_global(ik_slot_layer(slot)) ? ntok : (ntok < IK_EXTENT ? ntok : IK_EXTENT);
}

static uint64_t iks_payload_size(uint32_t ntok) {
    uint64_t b = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * 4 + (uint64_t)ntok * 4 + (uint64_t)IK_VOCAB * 4 + 8;
    for (int s = 0; s < IK_NSLOT; s++) b += (uint64_t)IK_CONVC * 3 * 4 + 2 * iks_live_rows(s, ntok) * IK_HK * IK_D * 2;
    return b;
}

static bool iks_state_is_live(const ds4_session *s) {
    return s && s->ikg && s->checkpoint_valid && s->checkpoint.len > 0 && s->ikg->pos == (uint32_t)s->checkpoint.len;
}

static uint64_t iks_payload_bytes(ds4_session *s) {
    return iks_state_is_live(s) ? iks_payload_size((uint32_t)s->checkpoint.len) : 0;
}

/* `count` rows starting at position p0, in position order, as at most two GPU spans per wrap of the ring. */
static int iks_rows_io(FILE *fp, ds4_gpu_tensor *t, uint32_t rows, uint64_t p0, uint64_t count, bool save, uint8_t *buf,
                       uint64_t *hash, uint64_t *remaining, char *err, size_t errlen) {
    const uint64_t rb = (uint64_t)IK_HK * IK_D * 2;
    while (count > 0) {
        const uint64_t row = p0 % rows, run = count < rows - row ? count : rows - row;
        for (uint64_t done = 0, total = run * rb; done < total;) {
            const size_t n = total - done > IKS_IO_CHUNK ? IKS_IO_CHUNK : (size_t)(total - done);
            if (save) {
                if (!ds4_gpu_tensor_read(t, row * rb + done, buf, n)) return iks_fail(err, errlen, "failed to read accelerator session tensor");
                if (fwrite(buf, 1, n, fp) != n) return iks_fail(err, errlen, "failed to write session payload");
                *hash = iks_hash_block(*hash, buf, n);
            } else {
                if (*remaining < n) return iks_fail(err, errlen, "truncated session payload");
                if (fread(buf, 1, n, fp) != n) return iks_fail(err, errlen, "failed to read session payload");
                *remaining -= n;
                *hash = iks_hash_block(*hash, buf, n);
                if (!ds4_gpu_tensor_write(t, row * rb + done, buf, n)) return iks_fail(err, errlen, "failed to restore accelerator session tensor");
            }
            done += n;
        }
        p0 += run;
        count -= run;
    }
    return 0;
}

static int iks_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    if (!iks_state_is_live(s)) return iks_fail(err, errlen, "session has no live state to save (rewound or stale)");
    ik_gpu *g = s->ikg;
    const uint32_t ntok = (uint32_t)s->checkpoint.len;
    const uint64_t fpr = iks_model_fingerprint(s->engine);
    const uint32_t hd[DS4_SESSION_PAYLOAD_U32_FIELDS] = { DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, (uint32_t)s->ctx_size, g->chunk,
        (uint32_t)fpr, (uint32_t)(fpr >> 32), IK_NSLOT, ntok, IK_NLAYER, IK_HK * IK_D, IK_CONVC, IK_VOCAB, IKS_PAYLOAD_TAG };
    uint8_t *buf = xmalloc(IKS_IO_CHUNK);
    uint64_t hash = 0x1234567887654321ull;
    int rc = 0;
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS && rc == 0; i++) rc = payload_write_u32(fp, hd[i], err, errlen);
    for (int i = 0; i < s->checkpoint.len && rc == 0; i++) {
        uint8_t b[4];
        payload_put_u32(b, (uint32_t)s->checkpoint.v[i]);
        hash = iks_hash_block(hash, b, 4);
        rc = payload_write_bytes(fp, b, 4, err, errlen);
    }
    if (rc == 0) {
        hash = iks_hash_block(hash, s->logits, (size_t)IK_VOCAB * 4);
        rc = payload_write_bytes(fp, s->logits, (uint64_t)IK_VOCAB * 4, err, errlen);
    }
    for (int sl = 0; sl < IK_NSLOT && rc == 0; sl++) {
        const uint64_t cb = (uint64_t)IK_CONVC * 3 * 4, live = iks_live_rows(sl, ntok);
        if (!ds4_gpu_tensor_read(g->conv[sl], 0, buf, cb)) { rc = iks_fail(err, errlen, "failed to read accelerator session tensor"); break; }
        hash = iks_hash_block(hash, buf, (size_t)cb);
        rc = payload_write_bytes(fp, buf, cb, err, errlen);
        if (rc == 0) rc = iks_rows_io(fp, g->k[sl], g->rows[sl], ntok - live, live, true, buf, &hash, NULL, err, errlen);
        if (rc == 0) rc = iks_rows_io(fp, g->v[sl], g->rows[sl], ntok - live, live, true, buf, &hash, NULL, err, errlen);
    }
    if (rc == 0) {
        uint8_t hb[8];
        for (int i = 0; i < 8; i++) hb[i] = (uint8_t)(hash >> (8 * i));
        rc = payload_write_bytes(fp, hb, 8, err, errlen);
    }
    free(buf);
    return rc;
}

static int iks_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s || !fp || !s->ikg) return iks_fail(err, errlen, "invalid session payload load");
    ik_gpu *g = s->ikg;
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++)
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) return iks_fail(err, errlen, "unsupported session payload version");
    if (h[12] != IKS_PAYLOAD_TAG) return iks_fail(err, errlen, "KV checkpoint was written by another model family");
    const uint64_t fpr = iks_model_fingerprint(s->engine);
    if (h[4] != (uint32_t)fpr || h[5] != (uint32_t)(fpr >> 32)) return iks_fail(err, errlen, "KV checkpoint was written for another model file");
    if (h[6] != IK_NSLOT || h[8] != IK_NLAYER || h[9] != IK_HK * IK_D || h[10] != IK_CONVC || h[11] != IK_VOCAB)
        return iks_fail(err, errlen, "KV checkpoint was written for a different Inkling layout");
    if (h[2] != (uint32_t)s->ctx_size) return iks_fail(err, errlen, "KV checkpoint was written with a different --ctx");
    const uint32_t ntok = h[7];
    if (ntok == 0 || ntok >= (uint32_t)s->ctx_size) return iks_fail(err, errlen, "KV checkpoint token count does not fit the context");
    if (payload_bytes != iks_payload_size(ntok)) return iks_fail(err, errlen, "KV checkpoint size does not match its header (truncated or corrupt)");
    /* from here the GPU state is overwritten: any failure leaves the session invalid */
    uint8_t *buf = xmalloc(IKS_IO_CHUNK);
    uint64_t hash = 0x1234567887654321ull;
    int rc = 0;
    token_vec toks = {0};
    for (uint32_t i = 0; i < ntok && rc == 0; i++) {
        uint8_t b[4];
        rc = payload_read_bytes(fp, b, 4, &remaining, err, errlen);
        if (rc) break;
        hash = iks_hash_block(hash, b, 4);
        const uint32_t t = payload_get_u32(b);
        if (t >= IK_VOCAB) { rc = iks_fail(err, errlen, "KV checkpoint holds a token id outside the vocabulary"); break; }
        token_vec_push(&toks, (int)t);
    }
    float *logits = xmalloc((size_t)IK_VOCAB * 4);
    if (rc == 0) rc = payload_read_bytes(fp, logits, (uint64_t)IK_VOCAB * 4, &remaining, err, errlen);
    if (rc == 0) hash = iks_hash_block(hash, logits, (size_t)IK_VOCAB * 4);
    if (rc == 0) {
        s->checkpoint_valid = false;
        s->checkpoint.len = 0;
        s->snap_valid = false;
        g->snap_pos = 0;
    }
    for (int sl = 0; sl < IK_NSLOT && rc == 0; sl++) {
        const uint64_t cb = (uint64_t)IK_CONVC * 3 * 4, live = iks_live_rows(sl, ntok);
        rc = payload_read_bytes(fp, buf, cb, &remaining, err, errlen);
        if (rc == 0) {
            hash = iks_hash_block(hash, buf, (size_t)cb);
            if (!ds4_gpu_tensor_write(g->conv[sl], 0, buf, cb)) rc = iks_fail(err, errlen, "failed to restore accelerator session tensor");
        }
        if (rc == 0) rc = iks_rows_io(fp, g->k[sl], g->rows[sl], ntok - live, live, false, buf, &hash, &remaining, err, errlen);
        if (rc == 0) rc = iks_rows_io(fp, g->v[sl], g->rows[sl], ntok - live, live, false, buf, &hash, &remaining, err, errlen);
    }
    if (rc == 0) {
        uint8_t hb[8];
        rc = payload_read_bytes(fp, hb, 8, &remaining, err, errlen);
        uint64_t want = 0;
        for (int i = 0; i < 8; i++) want |= (uint64_t)hb[i] << (8 * i);
        if (rc == 0 && (want != hash || remaining != 0)) rc = iks_fail(err, errlen, "KV checkpoint checksum mismatch (corrupt file)");
    }
    if (rc == 0) {
        for (int i = 0; i < toks.len; i++) token_vec_push(&s->checkpoint, toks.v[i]);
        memcpy(s->logits, logits, (size_t)IK_VOCAB * 4);
        g->pos = ntok;
        s->checkpoint_valid = true;
        iks_take_snapshot(s);   /* a follow-up that drops reasoning_content restores the restored prompt's end state */
    } else {
        iks_invalidate(s);
    }
    token_vec_free(&toks);
    free(logits);
    free(buf);
    return rc;
}
