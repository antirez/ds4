/* Recursive Flint sessions (fork-owned; docs/RECURSIVE_FLINT.md step 2).
 * Textually included into ds4.c right after ds4_flint.inc.  The session
 * driver behind the ds4_session_* hooks: prefill in chunks of 2048 tokens
 * through ds4_flint_gpu_prefill, decode through the megakernel, payloads
 * (disk checkpoints and snapshots), rewind, plus the two CLI check modes.
 *
 * State per session beyond ds4_session's token checkpoint: the GPU KV
 * blocks, `ring_initial` (the prefill length the 35 sliding rings are laid
 * out for; rec's `initial`), and the logits of the step before the last
 * eval so a one-token rewind restores them exactly.  Longer rewinds give
 * up the state and the next sync rebuilds it (the rings only hold the
 * last 513 positions, so nothing older can be recovered). */

typedef struct flint_session {
    ds4_flint_gpu_session *gpu;
    uint32_t ctx_cap;
    uint32_t ring_initial;
    float *prev_logits;
    bool prev_logits_valid;
    uint16_t *bf16;              /* 150144 bf16 logits scratch */
} flint_session;

#define DS4_FLINT_PAYLOAD_TAG UINT32_C(0x544e4c46)   /* "FLNT" */

static int ds4_session_tp_register(ds4_session *s);
static bool ds4_session_cancelled(ds4_session *s);

static flint_session *flint_of(ds4_session *s) { return (flint_session *)s->flint; }

static void flint_widen_logits(const uint16_t *bf16, float *out) {
    for (uint32_t i = 0; i < DS4_FLINT_GPU_N_VOCAB; i++) out[i] = flint_bf16_to_f32(bf16[i]);
}

static uint32_t flint_prefill_chunk(void) {
    const char *env = getenv("DS4_FLINT_PREFILL_CHUNK");
    uint32_t chunk = env && env[0] ? (uint32_t)atoi(env) : DS4_FLINT_GPU_PREFILL_CAP;
    if (chunk == 0 || chunk > DS4_FLINT_GPU_PREFILL_CAP) chunk = DS4_FLINT_GPU_PREFILL_CAP;
    return chunk;
}

static int flint_session_create(ds4_session *s, ds4_engine *e, int ctx_size, ds4_session **out) {
    if (!g_flint_gpu_open) {
        fprintf(stderr, "ds4: the Flint GPU engine is not open\n");
        free(s);
        return 1;
    }
    if ((uint32_t)ctx_size > DS4_FLINT_GPU_MAX_CTX) {
        fprintf(stderr, "ds4: Flint contexts above %u tokens are not supported\n", DS4_FLINT_GPU_MAX_CTX);
        free(s);
        return 1;
    }
    flint_session *f = xcalloc(1, sizeof(*f));
    f->gpu = ds4_flint_gpu_session_create((uint32_t)ctx_size);
    if (!f->gpu) {
        free(f);
        free(s);
        return 1;
    }
    f->ctx_cap = (uint32_t)ctx_size;
    f->prev_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    f->bf16 = xmalloc((size_t)DS4_FLINT_GPU_N_VOCAB * sizeof(uint16_t));
    s->flint = f;
    s->prefill_cap = (uint32_t)ctx_size;
    s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    s->sample_probs = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->sample_probs[0]));
    (void)e;
    if (!ds4_session_tp_register(s)) {
        ds4_session_free(s);
        return 1;
    }
    *out = s;
    return 0;
}

static void flint_session_free(ds4_session *s) {
    flint_session *f = flint_of(s);
    if (!f) return;
    ds4_flint_gpu_session_free(f->gpu);
    free(f->prev_logits);
    free(f->bf16);
    free(f);
    s->flint = NULL;
}

/* Prefill prompt[start..len) in chunks; the checkpoint grows per chunk so an
 * interruption keeps what was done. */
static int flint_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    flint_session *f = flint_of(s);
    int start = 0;
    if (s->checkpoint_valid && prompt->len >= s->checkpoint.len && ds4_tokens_starts_with(prompt, &s->checkpoint)) {
        start = s->checkpoint.len;
    } else {
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
        f->ring_initial = 0;
        f->prev_logits_valid = false;
    }
    for (int i = start; i < prompt->len; i++) {
        if (prompt->v[i] < 0 || prompt->v[i] >= (int)DS4_N_VOCAB) {
            snprintf(err, errlen, "token id %d at position %d is outside the vocabulary", prompt->v[i], i);
            return 1;
        }
    }
    if (start == prompt->len) return 0;   /* the logits after the last token are already current */
    if ((uint32_t)prompt->len > f->ctx_cap) {
        snprintf(err, errlen, "prompt length %d exceeds the session capacity %u", prompt->len, f->ctx_cap);
        return 1;
    }
    const uint32_t chunk = flint_prefill_chunk();
    for (int i = start; i < prompt->len; i += (int)chunk) {
        if (ds4_session_cancelled(s)) {
            snprintf(err, errlen, "interrupted");
            return DS4_SESSION_SYNC_INTERRUPTED;
        }
        const uint32_t n = (uint32_t)(prompt->len - i) < chunk ? (uint32_t)(prompt->len - i) : chunk;
        if (!ds4_flint_gpu_prefill(f->gpu, prompt->v + i, n, (uint32_t)i, f->ring_initial, f->bf16, NULL, NULL, err, errlen)) {
            s->checkpoint_valid = false;
            return 1;
        }
        f->ring_initial = (uint32_t)i + n;
        for (uint32_t j = 0; j < n; j++) token_vec_push(&s->checkpoint, prompt->v[i + (int)j]);
        s->checkpoint_valid = true;
        if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i + (int)n, prompt->len);
    }
    flint_widen_logits(f->bf16, s->logits);
    f->prev_logits_valid = false;
    s->mtp_draft_valid = false;
    return 0;
}

static int flint_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    flint_session *f = flint_of(s);
    const uint32_t pos = (uint32_t)s->checkpoint.len;
    if (pos >= f->ctx_cap) {
        if (errlen) snprintf(err, errlen, "context is full");
        return 1;
    }
    if (token < 0 || token >= (int)DS4_N_VOCAB) {
        if (errlen) snprintf(err, errlen, "token id %d is outside the vocabulary", token);
        return 1;
    }
    memcpy(f->prev_logits, s->logits, (size_t)DS4_N_VOCAB * sizeof(float));
    f->prev_logits_valid = s->checkpoint_valid;
    if (!ds4_flint_gpu_decode(f->gpu, (uint32_t)token, pos, f->ring_initial, f->bf16, NULL, err, errlen)) {
        s->checkpoint_valid = false;
        f->prev_logits_valid = false;
        return 1;
    }
    flint_widen_logits(f->bf16, s->logits);
    token_vec_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    return 0;
}

/* Returns whether the backend state is consistent with checkpoint length pos. */
static bool flint_session_rewind(ds4_session *s, int pos) {
    flint_session *f = flint_of(s);
    if (pos == s->checkpoint.len - 1 && f->prev_logits_valid) {
        memcpy(s->logits, f->prev_logits, (size_t)DS4_N_VOCAB * sizeof(float));
        f->prev_logits_valid = false;
        return true;
    }
    f->prev_logits_valid = false;
    return false;
}

/* Payload: ds4's 13-word header (tag FLNT), tokens, f32 logits, ring_initial,
 * then per layer K and V: global layers rows [0, len), sliding layers all
 * 513 ring slots; each as [4 heads][rows][128] bf16. */
static uint64_t flint_payload_kv_bytes(uint32_t rows) {
    return (uint64_t)DS4_FLINT_N_GLOBAL_LAYER * 2u * 4u * rows * 128u * 2u +
           (uint64_t)(DS4_N_LAYER - DS4_FLINT_N_GLOBAL_LAYER) * 2u * 4u * DS4_FLINT_GPU_SWA * 128u * 2u;
}

static uint64_t flint_session_payload_bytes(ds4_session *s) {
    if (!s->checkpoint_valid || !flint_of(s)) return 0;
    return (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * 4 + (uint64_t)s->checkpoint.len * 4 +
           (uint64_t)DS4_N_VOCAB * 4 + 4 + flint_payload_kv_bytes((uint32_t)s->checkpoint.len);
}

static int flint_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    flint_session *f = flint_of(s);
    const uint32_t rows = (uint32_t)s->checkpoint.len;
    const uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, (uint32_t)s->ctx_size, s->prefill_cap,
        f->ctx_cap, f->ctx_cap, 0u, rows, DS4_N_LAYER, DS4_N_HEAD_DIM, 0u, DS4_N_VOCAB, DS4_FLINT_PAYLOAD_TAG,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * 4, err, errlen) != 0) return 1;
    if (payload_write_u32(fp, f->ring_initial, err, errlen) != 0) return 1;
    const uint32_t step = 4096;
    uint16_t *buf = xmalloc((size_t)step * 128 * 2);
    int rc = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER && rc == 0; il++) {
        const uint32_t n = ds4_flint_layer_is_global(il) ? rows : DS4_FLINT_GPU_SWA;
        for (int key = 1; key >= 0 && rc == 0; key--) {
            for (uint32_t head = 0; head < 4 && rc == 0; head++) {
                for (uint32_t r0 = 0; r0 < n && rc == 0; r0 += step) {
                    const uint32_t cnt = n - r0 < step ? n - r0 : step;
                    if (!ds4_flint_gpu_kv_read(f->gpu, il, key, head, r0, cnt, buf)) {
                        payload_set_err(err, errlen, "Flint KV read failed");
                        rc = 1;
                        break;
                    }
                    rc = payload_write_bytes(fp, buf, (uint64_t)cnt * 128 * 2, err, errlen);
                }
            }
        }
    }
    free(buf);
    return rc;
}

static int flint_session_load_payload(ds4_session *s, FILE *fp, const uint32_t *h, uint64_t *remaining,
                                      char *err, size_t errlen) {
    flint_session *f = flint_of(s);
    const uint32_t rows = h[7];
    if (h[12] != DS4_FLINT_PAYLOAD_TAG || h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM || h[11] != DS4_N_VOCAB) {
        payload_set_err(err, errlen, "KV checkpoint was written by a different model family or shape");
        return 1;
    }
    if (rows > f->ctx_cap || rows > (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint is longer than this session's context");
        return 1;
    }
    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < rows; i++) {
        uint32_t tok;
        if (payload_read_u32(fp, &tok, remaining, err, errlen) != 0) { token_vec_free(&new_checkpoint); return 1; }
        if (tok >= DS4_N_VOCAB) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint token id is outside the vocabulary");
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    s->checkpoint_valid = false;
    f->prev_logits_valid = false;
    int rc = payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * 4, remaining, err, errlen);
    uint32_t ring_initial = 0;
    if (rc == 0) rc = payload_read_u32(fp, &ring_initial, remaining, err, errlen);
    if (rc == 0 && *remaining != flint_payload_kv_bytes(rows)) {
        payload_set_err(err, errlen, "Flint checkpoint tensor lengths do not match its layout");
        rc = 1;
    }
    const uint32_t step = 4096;
    uint16_t *buf = xmalloc((size_t)step * 128 * 2);
    for (uint32_t il = 0; il < DS4_N_LAYER && rc == 0; il++) {
        const uint32_t n = ds4_flint_layer_is_global(il) ? rows : DS4_FLINT_GPU_SWA;
        for (int key = 1; key >= 0 && rc == 0; key--) {
            for (uint32_t head = 0; head < 4 && rc == 0; head++) {
                for (uint32_t r0 = 0; r0 < n && rc == 0; r0 += step) {
                    const uint32_t cnt = n - r0 < step ? n - r0 : step;
                    rc = payload_read_bytes(fp, buf, (uint64_t)cnt * 128 * 2, remaining, err, errlen);
                    if (rc == 0 && !ds4_flint_gpu_kv_write(f->gpu, il, key, head, r0, cnt, buf)) {
                        payload_set_err(err, errlen, "Flint KV write failed");
                        rc = 1;
                    }
                }
            }
        }
    }
    free(buf);
    if (rc != 0) { token_vec_free(&new_checkpoint); return 1; }
    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    f->ring_initial = ring_initial;
    return 0;
}

/* ---- CLI check modes (tests/flint/prefill_compare.py, tests/flint/check_golden.py) ---- */

/* IN: "FLPT", u32 version 1, u32 n, i32 tokens[n].  OUT: "FLPO", u32 n, u32 50,
 * u32 2048, u32 150144, u16 hidden[50][n][2048] (embedding output, then each
 * layer's output), u16 logits[n][150144] (batched head over every row), u16
 * last[150144] (the vector head of the last row, what sampling uses). */
int ds4_flint_prefill_trace(ds4_engine *e, const char *in_path, const char *out_path) {
    if (!e || !g_flint_gpu_open) {
        fprintf(stderr, "ds4: the Flint prefill trace needs an open Flint engine\n");
        return 1;
    }
    FILE *in = fopen(in_path, "rb");
    if (!in) { perror(in_path); return 1; }
    char magic[4];
    uint32_t version = 0, n = 0;
    if (fread(magic, 1, 4, in) != 4 || memcmp(magic, "FLPT", 4) != 0 || !flint_read_u32(in, &version) || version != 1 ||
        !flint_read_u32(in, &n) || n == 0 || n > 65536) {
        fprintf(stderr, "ds4: %s is not a Flint prefill trace input\n", in_path);
        fclose(in);
        return 1;
    }
    int32_t *tokens = xcalloc(n, sizeof(int32_t));
    if (fread(tokens, 4, n, in) != n) { fprintf(stderr, "ds4: short token read\n"); free(tokens); fclose(in); return 1; }
    fclose(in);
    ds4_flint_gpu_session *gpu = ds4_flint_gpu_session_create(n + 1);
    if (!gpu) { free(tokens); return 1; }
    const uint32_t chunk = flint_prefill_chunk();
    uint16_t *hidden = xmalloc((size_t)50 * n * 2048 * 2);
    uint16_t *logits = xmalloc((size_t)n * DS4_FLINT_GPU_N_VOCAB * 2);
    uint16_t *last = xmalloc((size_t)DS4_FLINT_GPU_N_VOCAB * 2);
    uint16_t *hidden_chunk = xmalloc((size_t)50 * chunk * 2048 * 2);
    uint32_t ring_initial = 0;
    int rc = 0;
    const double t0 = flint_now_ms();
    for (uint32_t i = 0; i < n && rc == 0; i += chunk) {
        const uint32_t nc = n - i < chunk ? n - i : chunk;
        char err[256] = {0};
        if (!ds4_flint_gpu_prefill(gpu, tokens + i, nc, i, ring_initial, last, hidden_chunk,
                                   logits + (size_t)i * DS4_FLINT_GPU_N_VOCAB, err, sizeof(err))) {
            fprintf(stderr, "ds4: Flint prefill failed: %s\n", err);
            rc = 1;
            break;
        }
        for (uint32_t layer = 0; layer < 50; layer++) {
            memcpy(hidden + ((size_t)layer * n + i) * 2048, hidden_chunk + (size_t)layer * nc * 2048, (size_t)nc * 2048 * 2);
        }
        ring_initial = i + nc;
    }
    fprintf(stderr, "ds4: Flint prefill trace: %u tokens in %.1f ms (per-layer sync included)\n", n, flint_now_ms() - t0);
    if (rc == 0) {
        FILE *outf = fopen(out_path, "wb");
        if (!outf) { perror(out_path); rc = 1; }
        else {
            const uint32_t header[4] = { n, 50, 2048, DS4_FLINT_GPU_N_VOCAB };
            fwrite("FLPO", 1, 4, outf);
            fwrite(header, 4, 4, outf);
            fwrite(hidden, 2, (size_t)50 * n * 2048, outf);
            fwrite(logits, 2, (size_t)n * DS4_FLINT_GPU_N_VOCAB, outf);
            fwrite(last, 2, DS4_FLINT_GPU_N_VOCAB, outf);
            fclose(outf);
        }
    }
    free(hidden); free(logits); free(last); free(hidden_chunk); free(tokens);
    ds4_flint_gpu_session_free(gpu);
    return rc;
}

static float flint_logsumexp(const float *x, uint32_t n) {
    float m = x[0];
    for (uint32_t i = 1; i < n; i++) if (x[i] > m) m = x[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += exp((double)(x[i] - m));
    return m + (float)log(sum);
}

/* Generation through the session API, as the server would drive it.
 * IN: "FLGN", u32 version 2, u32 n_prompt, u32 n_expected, u32 max_new,
 * u32 ctx (0: prompt + max_new + 1), u32 n_follow, i32 prompt[],
 * i32 expected[], i32 follow[].  With expected tokens the run is
 * teacher-forced (each expected token is fed whatever ds4's argmax was);
 * without, greedy until a stop token.  With follow tokens (teacher-forced
 * runs only) the session then syncs prompt + expected + follow, reusing its
 * decode KV as the prefix (gate 2 of the plan), and a fresh session prefills
 * the same tokens; both logits vectors are written.
 * OUT: "FLGO", u32 n_steps, per step {i32 fed, i32 argmax, f32 logprob(fed),
 * f32 logprob(argmax)}, f64 prefill_ms, f64 decode_ms, u32 has_follow,
 * [f32 reused[V], f32 fresh[V], f64 reuse_prefill_ms, f64 fresh_prefill_ms,
 *  u32 n2, per step {f32 kl(fresh||reused), f32 dlp(fed), u32 top1_same}]:
 * the fresh session's greedy continuation of n2 = 64 tokens after the
 * follow-up, fed to both sessions, so the comparison runs along a reply. */
int ds4_flint_generate_check(ds4_engine *e, const char *in_path, const char *out_path) {
    if (!e || !g_flint_gpu_open) {
        fprintf(stderr, "ds4: the Flint generate check needs an open Flint engine\n");
        return 1;
    }
    FILE *in = fopen(in_path, "rb");
    if (!in) { perror(in_path); return 1; }
    char magic[4];
    uint32_t version = 0, n_prompt = 0, n_expected = 0, max_new = 0, ctx = 0, n_follow = 0;
    if (fread(magic, 1, 4, in) != 4 || memcmp(magic, "FLGN", 4) != 0 || !flint_read_u32(in, &version) ||
        (version != 1 && version != 2) ||
        !flint_read_u32(in, &n_prompt) || !flint_read_u32(in, &n_expected) || !flint_read_u32(in, &max_new) ||
        !flint_read_u32(in, &ctx) || (version == 2 && !flint_read_u32(in, &n_follow)) ||
        n_prompt == 0 || n_prompt > 262144 || n_expected > 65536 || n_follow > 65536) {
        fprintf(stderr, "ds4: %s is not a Flint generate input\n", in_path);
        fclose(in);
        return 1;
    }
    ds4_tokens prompt = {0};
    for (uint32_t i = 0; i < n_prompt; i++) {
        int32_t t;
        if (fread(&t, 4, 1, in) != 1) { fclose(in); ds4_tokens_free(&prompt); return 1; }
        token_vec_push(&prompt, t);
    }
    int32_t *expected = n_expected ? xcalloc(n_expected, 4) : NULL;
    if (n_expected && fread(expected, 4, n_expected, in) != n_expected) { fclose(in); free(expected); ds4_tokens_free(&prompt); return 1; }
    int32_t *follow = n_follow ? xcalloc(n_follow, 4) : NULL;
    if (n_follow && fread(follow, 4, n_follow, in) != n_follow) { fclose(in); free(expected); free(follow); ds4_tokens_free(&prompt); return 1; }
    fclose(in);
    if (ctx == 0) ctx = n_prompt + max_new + n_follow + 65;   /* room for the 64-step reuse continuation */
    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, (int)ctx) != 0) { free(expected); ds4_tokens_free(&prompt); return 1; }
    char err[256] = {0};
    const double t0 = flint_now_ms();
    if (ds4_session_sync(s, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: Flint prefill failed: %s\n", err);
        ds4_session_free(s); free(expected); ds4_tokens_free(&prompt);
        return 1;
    }
    const double t1 = flint_now_ms();
    FILE *outf = fopen(out_path, "wb");
    if (!outf) { perror(out_path); ds4_session_free(s); free(expected); ds4_tokens_free(&prompt); return 1; }
    fwrite("FLGO", 1, 4, outf);
    uint32_t n_steps = 0;
    fwrite(&n_steps, 4, 1, outf);   /* patched at the end */
    const uint32_t limit = n_expected ? (n_expected < max_new ? n_expected : max_new) : max_new;
    int rc = 0;
    for (uint32_t step = 0; step < limit; step++) {
        const int argmax = ds4_session_argmax(s);
        const int fed = n_expected ? expected[step] : argmax;
        const float lse = flint_logsumexp(s->logits, DS4_N_VOCAB);
        const float lp_fed = s->logits[fed] - lse, lp_arg = s->logits[argmax] - lse;
        const int32_t ints[2] = { fed, argmax };
        const float floats[2] = { lp_fed, lp_arg };
        fwrite(ints, 4, 2, outf);
        fwrite(floats, 4, 2, outf);
        n_steps++;
        if (!n_expected && ds4_token_is_stop(e, fed)) break;
        if (step + 1 == limit) break;
        if (ds4_session_eval(s, fed, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: Flint decode failed at step %u: %s\n", step, err);
            rc = 1;
            break;
        }
    }
    const double t2 = flint_now_ms();
    const double times[2] = { t1 - t0, t2 - t1 };
    fwrite(times, 8, 2, outf);
    const uint32_t has_follow = (rc == 0 && n_follow && n_expected) ? 1u : 0u;
    fwrite(&has_follow, 4, 1, outf);
    if (has_follow) {
        /* the session holds prompt + the fed tokens (decode KV); sync the full
         * sequence on it and on a fresh session */
        ds4_tokens full = {0};
        for (int i = 0; i < prompt.len; i++) token_vec_push(&full, prompt.v[i]);
        for (uint32_t i = 0; i + 1 < n_steps; i++) token_vec_push(&full, expected[i]);
        token_vec_push(&full, expected[n_steps - 1]);
        for (uint32_t i = 0; i < n_follow; i++) token_vec_push(&full, follow[i]);
        const double r0 = flint_now_ms();
        if (ds4_session_sync(s, &full, err, sizeof(err)) != 0) { fprintf(stderr, "ds4: reuse sync failed: %s\n", err); rc = 1; }
        const double r1 = flint_now_ms();
        fwrite(s->logits, 4, DS4_N_VOCAB, outf);
        ds4_session *fresh = NULL;
        if (rc == 0 && ds4_session_create(&fresh, e, (int)ctx) != 0) rc = 1;
        if (rc == 0 && ds4_session_sync(fresh, &full, err, sizeof(err)) != 0) { fprintf(stderr, "ds4: fresh sync failed: %s\n", err); rc = 1; }
        const double r2 = flint_now_ms();
        if (fresh) fwrite(fresh->logits, 4, DS4_N_VOCAB, outf);
        const double rtimes[2] = { r1 - r0, r2 - r1 };
        fwrite(rtimes, 8, 2, outf);
        fprintf(stderr, "ds4: Flint reuse check: %d tokens; reused-KV sync %.0f ms, fresh prefill %.0f ms\n", full.len, r1 - r0, r2 - r1);
        uint32_t n2 = rc == 0 ? 64u : 0u;
        if (n2 && (uint32_t)full.len + n2 + 1 > ctx) n2 = 0;
        fwrite(&n2, 4, 1, outf);
        float *lpa = xmalloc((size_t)DS4_N_VOCAB * 4), *lpb = xmalloc((size_t)DS4_N_VOCAB * 4);
        for (uint32_t i = 0; i < n2; i++) {
            const float la = flint_logsumexp(s->logits, DS4_N_VOCAB), lb = flint_logsumexp(fresh->logits, DS4_N_VOCAB);
            double kl = 0.0;
            for (uint32_t v = 0; v < DS4_N_VOCAB; v++) {
                lpa[v] = s->logits[v] - la; lpb[v] = fresh->logits[v] - lb;
                kl += exp((double)lpb[v]) * ((double)lpb[v] - (double)lpa[v]);
            }
            const int fed = ds4_session_argmax(fresh);
            const float rec[2] = { (float)kl, fabsf(lpa[fed] - lpb[fed]) };
            const uint32_t same = ds4_session_argmax(s) == fed;
            fwrite(rec, 4, 2, outf);
            fwrite(&same, 4, 1, outf);
            if (i + 1 == n2) break;
            if (ds4_session_eval(s, fed, err, sizeof(err)) != 0 || ds4_session_eval(fresh, fed, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4: reuse continuation failed: %s\n", err);
                rc = 1;
                break;
            }
        }
        free(lpa); free(lpb);
        ds4_session_free(fresh);
        ds4_tokens_free(&full);
    }
    fseek(outf, 4, SEEK_SET);
    fwrite(&n_steps, 4, 1, outf);
    fclose(outf);
    fprintf(stderr, "ds4: Flint generate: prefill %u tokens %.0f ms, %u steps %.0f ms (%.1f ms/token)\n",
            n_prompt, t1 - t0, n_steps, t2 - t1, n_steps > 1 ? (t2 - t1) / (n_steps - 1) : 0.0);
    ds4_session_free(s);
    free(expected);
    free(follow);
    ds4_tokens_free(&prompt);
    return rc;
}
