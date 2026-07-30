#include "pulsar_engine_internal.h"



void pulsar_linux_graph_backend_set_oom_score(pulsar_backend backend) {
#if defined(__linux__)
    static bool attempted = false;
    if (attempted) return;
    attempted = true;

    const int score = 1000;
    FILE *fp = fopen("/proc/self/oom_score_adj", "w");
    if (!fp) {
        fprintf(stderr,
                "pulsar: failed to set Linux %s backend oom_score_adj=%d: %s\n",
                pulsar_backend_name(backend),
                score,
                strerror(errno));
        return;
    }
    if (fprintf(fp, "%d\n", score) < 0) {
        const int err = errno;
        fclose(fp);
        fprintf(stderr,
                "pulsar: failed to write Linux %s backend oom_score_adj=%d: %s\n",
                pulsar_backend_name(backend),
                score,
                strerror(err));
        return;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr,
                "pulsar: failed to close Linux %s backend oom_score_adj=%d: %s\n",
                pulsar_backend_name(backend),
                score,
                strerror(errno));
        return;
    }
    fprintf(stderr,
            "pulsar: Linux %s backend set oom_score_adj=%d\n",
            pulsar_backend_name(backend),
            score);
#else
    (void)backend;
#endif
}



bool pulsar_think_mode_enabled(pulsar_think_mode mode) {
    return mode == PULSAR_THINK_HIGH || mode == PULSAR_THINK_MAX;
}



const char *pulsar_think_mode_name(pulsar_think_mode mode) {
    switch (mode) {
    case PULSAR_THINK_NONE: return "none";
    case PULSAR_THINK_HIGH: return "high";
    case PULSAR_THINK_MAX:  return "max";
    }
    return "unknown";
}



const char *pulsar_think_max_prefix(void) {
    return PULSAR_REASONING_EFFORT_MAX_PREFIX;
}



uint32_t pulsar_think_max_min_context(void) {
    return PULSAR_THINK_MAX_MIN_CONTEXT;
}



pulsar_think_mode pulsar_think_mode_for_context(pulsar_think_mode mode, int ctx_size) {
    if (mode == PULSAR_THINK_MAX && (uint32_t)(ctx_size > 0 ? ctx_size : 0) < PULSAR_THINK_MAX_MIN_CONTEXT) {
        return PULSAR_THINK_HIGH;
    }
    return mode;
}



void pulsar_release_instance_lock(void) {
    if (g_pulsar_lock_fd >= 0) {
        close(g_pulsar_lock_fd);
        g_pulsar_lock_fd = -1;
    }
}



/* Refuse to start a second pulsar/ds4 process.  The model can map tens of GiB,
 * so a stale accidental second run is more dangerous than a normal CLI error. */
void pulsar_acquire_instance_lock(void) {
    const char *path = getenv("PULSAR_LOCK_FILE");
    /* The default lock path stays "/tmp/ds4.lock" DELIBERATELY after the
     * Pulsar rebrand: an old ds4-server and a new pulsar binary must contend
     * on the SAME lock during the transition — two live instances would OOM
     * the GB10. Do not rename this path (or the PULSAR_* env names). */
    if (!path || !path[0]) path = "/tmp/ds4.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "pulsar: failed to open lock file %s: %s\n", path, strerror(errno));
        exit(2);
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                char *end = NULL;
                owner = strtol(buf, &end, 10);
            }
            if (owner > 0) {
                fprintf(stderr, "pulsar: another pulsar/ds4 process is already running (pid %ld); refusing to start\n", owner);
            } else {
                fprintf(stderr, "pulsar: another pulsar/ds4 process is already running; refusing to start\n");
            }
            close(fd);
            exit(2);
        }
        fprintf(stderr, "pulsar: failed to lock %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }

    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "pulsar: failed to truncate lock file %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }
    dprintf(fd, "%ld\n", (long)getpid());
    g_pulsar_lock_fd = fd;
    atexit(pulsar_release_instance_lock);
}



/* =========================================================================
 * Public free-function facades (C++ port).
 * =========================================================================
 *
 * The whole pulsar_session_* / pulsar_engine_* verb family now lives on the
 * classes in pulsar_engine_internal.h; the public API declared in pulsar.h
 * keeps its exact free-function signatures as one-line forwarders defined
 * here.  Server/CLI/agent/lib and the tests/ gate harnesses keep calling
 * these; engine internals call the members directly.
 *
 * Null-object tolerance: most of the original bodies accepted a NULL object
 * (returning a constant).  Calling a member through a NULL pointer is UB, so
 * each facade keeps that guard HERE, returning the same constant the old
 * body's null path returned.  (The handful of err-writing entry points do
 * not duplicate the err-message formatting for the never-exercised s==NULL
 * case — every live call site holds a valid session; all other argument
 * validation still runs unchanged inside the member body.) */

int pulsar_engine_open(pulsar_engine **out, const pulsar_engine_options *opt) { return pulsar_engine::open(out, opt); }
void pulsar_engine_close(pulsar_engine *e) { if (e) e->destroy(); }
void pulsar_engine_summary(pulsar_engine *e) { e->summary(); }
int pulsar_engine_vocab_size(pulsar_engine *e) { return e ? e->vocab_size() : 0; }
int pulsar_engine_logits_width(const pulsar_engine *e) { return e ? e->logits_width() : 0; }
const char *pulsar_engine_model_name(pulsar_engine *e) { return e->model_name(); }
int pulsar_engine_layer_count(pulsar_engine *e) { return e ? e->layer_count() : (int)PULSAR_N_LAYER; }
void pulsar_engine_spec_metrics(pulsar_engine *e, pulsar_spec_metrics *out) { if (e) { e->spec_metrics(out); } else if (out) { memset(out, 0, sizeof(*out)); } }
uint32_t pulsar_engine_layer_compress_ratio(pulsar_engine *e, uint32_t layer) { return e ? e->layer_compress_ratio(layer) : (layer >= PULSAR_N_LAYER ? 0 : pulsar_layer_compress_ratio(layer)); }
uint64_t pulsar_engine_hidden_f32_values(pulsar_engine *e) { return e ? e->hidden_f32_values() : (uint64_t)PULSAR_N_HC * PULSAR_N_EMBD; }
int pulsar_engine_model_id(pulsar_engine *e) { return e ? e->model_id() : (int)PULSAR_MODEL_VARIANT; }
bool pulsar_engine_is_pruned(pulsar_engine *e) { return e ? e->is_pruned() : false; }
uint64_t pulsar_engine_session_cost_bytes(pulsar_engine *e, int ctx_size) { return e ? e->session_cost_bytes(ctx_size) : 0; }
uint64_t pulsar_engine_session_cost_bytes_banked(pulsar_engine *e, int ctx_size, int n_banks) { return e ? e->session_cost_bytes_banked(ctx_size, n_banks) : 0; }
uint64_t pulsar_engine_demand_paged_bytes_per_bank(pulsar_engine *e, int ctx_size) { return e ? e->demand_paged_bytes_per_bank(ctx_size) : 0; }
uint64_t pulsar_engine_weights_resident_bytes(pulsar_engine *e) { return e ? e->weights_resident_bytes() : 0; }
int pulsar_engine_generate_argmax(pulsar_engine *e, const pulsar_tokens *prompt,
                               int n_predict, int ctx_size,
                               pulsar_token_emit_fn emit,
                               pulsar_generation_done_fn done,
                               void *emit_ud,
                               pulsar_session_progress_fn progress,
                               void *progress_ud) { return e->generate_argmax(prompt, n_predict, ctx_size, emit, done, emit_ud, progress, progress_ud); }
int pulsar_engine_collect_imatrix(pulsar_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) { return e ? e->collect_imatrix(dataset_path, output_path, ctx_size, max_prompts, max_tokens) : 1; }
void pulsar_engine_dump_tokens(pulsar_engine *e, const pulsar_tokens *tokens) { e->dump_tokens(tokens); }
int pulsar_engine_head_test(pulsar_engine *e, const pulsar_tokens *prompt) { return e->head_test(prompt); }
int pulsar_engine_gpu_graph_test(pulsar_engine *e, const pulsar_tokens *prompt) { return e->gpu_graph_test(prompt); }
int pulsar_engine_routed_quant_bits(pulsar_engine *e) { return e ? e->routed_quant_bits() : 0; }
bool pulsar_engine_has_output_head(pulsar_engine *e) { return e && e->has_output_head(); }
bool pulsar_engine_has_dspark(pulsar_engine *e) { return e && e->has_dspark(); }

int pulsar_session_create(pulsar_session **out, pulsar_engine *e, int ctx_size) { return pulsar_session::create(out, e, ctx_size); }
void pulsar_session_free(pulsar_session *s) { if (s) s->destroy(); }
void pulsar_session_set_progress(pulsar_session *s, pulsar_session_progress_fn fn, void *ud) { if (s) s->set_progress(fn, ud); }
void pulsar_session_set_display_progress(pulsar_session *s, pulsar_session_progress_fn fn, void *ud) { if (s) s->set_display_progress(fn, ud); }
void pulsar_session_set_cancel(pulsar_session *s, pulsar_session_cancel_fn fn, void *ud) { if (s) s->set_cancel(fn, ud); }
void pulsar_session_report_progress(pulsar_session *s, const char *event, int current, int total) { if (s) s->report_progress(event, current, total); }
void pulsar_session_spec_metrics(const pulsar_session *s, pulsar_spec_metrics *out) { if (s) { s->spec_metrics(out); } else if (out) { memset(out, 0, sizeof(*out)); } }
uint64_t pulsar_session_touched_kv_bytes(const pulsar_session *s) { return s ? s->touched_kv_bytes() : 0; }
bool pulsar_session_bank_free_physical(pulsar_session *s, uint32_t bank) { return s ? s->bank_free_physical(bank) : false; }
bool pulsar_session_bank_alloc_physical(pulsar_session *s, uint32_t bank) { return s ? s->bank_alloc_physical(bank) : false; }
bool pulsar_session_bank_is_evicted(const pulsar_session *s, uint32_t bank) { return s ? s->bank_is_evicted(bank) : false; }
uint64_t pulsar_session_bank_touched_kv_bytes(pulsar_session *s, uint32_t bank) { return s ? s->bank_touched_kv_bytes(bank) : 0; }
int pulsar_session_bank_kv_save(pulsar_session *s, uint32_t bank, FILE *fp, char *err, size_t errlen) { return s ? s->bank_kv_save(bank, fp, err, errlen) : 1; }
int pulsar_session_bank_kv_load(pulsar_session *s, uint32_t bank, FILE *fp, char *err, size_t errlen) { return s ? s->bank_kv_load(bank, fp, err, errlen) : 1; }
uint64_t pulsar_session_quantum_growth_bytes_per_bank(pulsar_session *s, uint32_t q) { return s->quantum_growth_bytes_per_bank(q); }
int pulsar_session_bank_fork(pulsar_session *s, uint32_t src, uint32_t dst, const int *tokens, int n_cached) { return s ? s->bank_fork(src, dst, tokens, n_cached) : 1; }
bool pulsar_session_bank_fork_pinned(const pulsar_session *s, uint32_t bank) { return s ? s->bank_fork_pinned(bank) : false; }
int pulsar_session_bank_fork_partial(pulsar_session *s, uint32_t src, uint32_t dst, const int *tokens, int n_cached) { return s ? s->bank_fork_partial(src, dst, tokens, n_cached) : 1; }
int pulsar_session_sync(pulsar_session *s, const pulsar_tokens *prompt, char *err, size_t errlen) { return s ? s->sync(prompt, err, errlen) : 1; }
pulsar_session_rewrite_result pulsar_session_rewrite_from_common(pulsar_session *s, const pulsar_tokens *prompt, int common, char *err, size_t errlen) { return s ? s->rewrite_from_common(prompt, common, err, errlen) : PULSAR_SESSION_REWRITE_ERROR; }
int pulsar_session_common_prefix(pulsar_session *s, const pulsar_tokens *prompt) { return s->common_prefix(prompt); }
int pulsar_session_argmax(pulsar_session *s) { return s->argmax(); }
int pulsar_session_argmax_excluding(pulsar_session *s, int excluded_id) { return s ? s->argmax_excluding(excluded_id) : -1; }
int pulsar_session_sample(pulsar_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) { return s->sample(temperature, top_k, top_p, min_p, rng); }
int pulsar_session_top_logprobs(pulsar_session *s, pulsar_token_score *out, int k) { return s ? s->top_logprobs(out, k) : 0; }
int pulsar_session_token_logprob(pulsar_session *s, int token, pulsar_token_score *out) { return s ? s->token_logprob(token, out) : 0; }
int pulsar_session_copy_logits(pulsar_session *s, float *out, int cap) { return s ? s->copy_logits(out, cap) : 0; }
int pulsar_session_set_logits(pulsar_session *s, const float *logits, int n) { return s ? s->set_logits(logits, n) : 1; }
int pulsar_session_eval(pulsar_session *s, int token, char *err, size_t errlen) { return s ? s->eval(token, err, errlen) : 1; }
int pulsar_session_decode_multiseq(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n, float *logits, int logits_cap, char *err, size_t errlen) { return s ? s->decode_multiseq(reqs, n, logits, logits_cap, err, errlen) : 1; }
int pulsar_session_decode_mixed(pulsar_session *s, const pulsar_multiseq_req *reqs, uint32_t n_rows, float *logits, int logits_cap, uint32_t *out_n_rows, uint32_t max_head_runs, char *err, size_t errlen) { return s ? s->decode_mixed(reqs, n_rows, logits, logits_cap, out_n_rows, max_head_runs, err, errlen) : 1; }
int pulsar_session_bank_count(pulsar_session *s) { return s ? s->bank_count() : 0; }
int pulsar_session_bank_repoint(pulsar_session *s, uint32_t bank) { return s ? s->bank_repoint(bank) : 1; }
void pulsar_session_bank_state_save(pulsar_session *s, uint32_t bank) { if (s) s->bank_state_save(bank); }
bool pulsar_session_bank_state_restore(pulsar_session *s, uint32_t bank) { return s ? s->bank_state_restore(bank) : false; }
int pulsar_session_bank_pos(pulsar_session *s, uint32_t bank) { return s->bank_pos(bank); }
const pulsar_tokens *pulsar_session_bank_tokens(pulsar_session *s, uint32_t bank) { return s->bank_tokens(bank); }
int pulsar_session_bank_common_prefix(pulsar_session *s, uint32_t bank, const pulsar_tokens *prompt) { return s->bank_common_prefix(bank, prompt); }
void pulsar_session_note_committed_tokens(pulsar_session *s, const int *toks, int n) { if (s) s->note_committed_tokens(toks, n); }
int pulsar_session_generate_speculative(pulsar_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng, int max_tokens, int eos_token, int *accepted, int accepted_cap, char *err, size_t errlen) { return s ? s->generate_speculative(temperature, top_k, top_p, min_p, rng, max_tokens, eos_token, accepted, accepted_cap, err, errlen) : 0; }
int pulsar_session_eval_speculative_block(pulsar_session *s, int first_token, int max_tokens, int eos_token, int *accepted, int accepted_cap, char *err, size_t errlen) { return s ? s->eval_speculative_block(first_token, max_tokens, eos_token, accepted, accepted_cap, err, errlen) : 0; }
void pulsar_session_invalidate(pulsar_session *s) { s->invalidate(); }
void pulsar_session_rewind(pulsar_session *s, int pos) { s->rewind(pos); }
int pulsar_session_pos(pulsar_session *s) { return s->pos(); }
int pulsar_session_ctx(pulsar_session *s) { return s->ctx(); }
uint32_t pulsar_session_prefill_quantum_min_suffix(const pulsar_session *s) { return s ? s->prefill_quantum_min_suffix() : 0; }
const pulsar_tokens *pulsar_session_tokens(pulsar_session *s) { return s ? s->tokens() : NULL; }
uint64_t pulsar_session_payload_bytes(pulsar_session *s) { return s ? s->payload_bytes() : 0; }
int pulsar_session_stage_payload(pulsar_session *s, pulsar_session_payload_file *out, char *err, size_t errlen) { return s ? s->stage_payload(out, err, errlen) : 1; }
int pulsar_session_save_payload(pulsar_session *s, FILE *fp, char *err, size_t errlen) { return s ? s->save_payload(fp, err, errlen) : 1; }
int pulsar_session_load_payload(pulsar_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) { return s ? s->load_payload(fp, payload_bytes, err, errlen) : 1; }
int pulsar_session_save_snapshot(pulsar_session *s, pulsar_session_snapshot *snap, char *err, size_t errlen) { return s ? s->save_snapshot(snap, err, errlen) : 1; }
int pulsar_session_load_snapshot(pulsar_session *s, const pulsar_session_snapshot *snap, char *err, size_t errlen) { return s ? s->load_snapshot(snap, err, errlen) : 1; }

