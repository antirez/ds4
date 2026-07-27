#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_CUDA,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*ds4_session_cancel_fn)(void *ud);

#define DS4_SESSION_SYNC_INTERRUPTED 2

typedef struct {
    const char *model_path;
    const char *dspark_path;
    /* Drafter is auto-enabled when the main GGUF contains dspark.* tensors;
     * this opts out (memory saving for sampled-only workloads). */
    bool dspark_disable;
    /* "FILE:PREFIX" — swap routed-expert tensors whose name starts with
     * PREFIX for the same-named tensors in FILE (a donor GGUF). Measurement
     * aid for per-layer quant-format KL probes; see gguf-tools/prisma. */
    const char *expert_overlay;
    ds4_backend backend;
    int n_threads;
    uint32_t prefill_chunk;
    int dspark_draft_tokens;
    float dspark_confidence;
    const char *directional_steering_file;
    float directional_steering_attn;
    float directional_steering_ffn;
    bool warm_weights;
    bool quality;
    bool inspect_only;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
/* Tokenizer table length. NOT the logits row width — see
 * ds4_engine_logits_width, and never size a logits buffer from this. */
int ds4_engine_vocab_size(ds4_engine *e);
/* Row width (in floats) of every logits buffer the engine writes: the shape
 * profile's n_vocab. Size logits buffers with THIS — it is the stride
 * ds4_session_decode_multiseq writes its rows at, and the loader does not
 * check it against ds4_engine_vocab_size. */
int ds4_engine_logits_width(const ds4_engine *e);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);

/* DSpark speculative-decode counters for the server /metrics endpoint. All
 * cumulative/monotonic since engine open. accepted_per_pos[i] counts how often
 * draft position i was accepted; rate[i] = accepted_per_pos[i]/num_drafts. */
typedef struct {
    uint64_t accepted_tokens;       /* accepted draft tokens */
    uint64_t draft_tokens;          /* proposed/verified draft tokens */
    uint64_t num_drafts;            /* draft rounds (verify steps with drafts) */
    uint64_t gen_tokens;            /* tokens emitted by the spec loop */
    uint64_t accepted_per_pos[16];  /* accepted count per draft position */
    int      max_draft;             /* configured draft depth (dspark_draft_tokens) */
    bool     has_dspark;            /* spec decode active */
} ds4_spec_metrics;
void ds4_engine_spec_metrics(ds4_engine *e, ds4_spec_metrics *out);
/* Per-session cumulative counters (accepted/draft/num_drafts/gen_tokens only;
 * accepted_per_pos left zero). Snapshot + diff across one request for a
 * per-response accept-rate the global engine counters cannot attribute under
 * concurrent decode. */
void ds4_session_spec_metrics(const ds4_session *s, ds4_spec_metrics *out);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
/* Like ds4_context_memory_estimate_with_prefill, but the persistent KV caches
 * are sized with their real packed element/row widths (f16 raw + DS4_ATTN_PACK
 * attn comp + MXFP4 indexer) instead of the sizeof(float) upper bound.
 * WARNING: this covers ONLY the persistent KV rows plus a small scratch term —
 * NOT the full per-session graph (prefill batch buffers, drafter state, …).
 * For admission accounting use ds4_engine_session_cost_bytes, which prices the
 * whole session; pricing sessions with this estimate under-admitted by ~10x
 * and hard-locked the GB10 (2026-07-13). */
ds4_context_memory ds4_context_memory_estimate_packed(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
/* TRUE total per-session GPU byte cost of ds4_session_create(e, ctx_size):
 * persistent KV caches + the full prefill working set (scaled by the engine's
 * prefill chunk) + speculative/DSpark drafter state when the engine has a
 * drafter loaded.  Shares sizing code with the graph allocator; this is the
 * number admission control must use.  Returns 0 if no session could be
 * created (no graph backend / weights not loaded). */
uint64_t ds4_engine_session_cost_bytes(ds4_engine *e, int ctx_size);
/* Same, priced for an EXPLICIT bank-pool size (Tier-2 auto-sizing): the server
 * evaluates the (banks, ctx) fit table before committing DS4_MSEQ_BANKS. n_banks
 * >= 1; 1 is the classic single-session cost. */
uint64_t ds4_engine_session_cost_bytes_banked(ds4_engine *e, int ctx_size,
                                              int n_banks);
/* Tier-2 overcommit (task #55): demand-paged comp+index bytes for ONE bank at a
 * context — the physical-on-touch VA the overcommit auto-size reserves but does
 * NOT charge at admission (only the eager floor is charged). 0 if no session
 * could be priced. */
uint64_t ds4_engine_demand_paged_bytes_per_bank(ds4_engine *e, int ctx_size);
/* Tier-2 overcommit (task #55): EXACT touched (physically resident) demand-paged
 * KV bytes across the pool, summed from the per-bank compressor frontier
 * (deterministic; no MemAvailable). The eviction guard triggers on this; the
 * accounting-exactness gate proves it matches the physical cudaMemGetInfo delta.
 * For the current bank the live frontier is used; idle banks use their captured
 * frontier — capture the current bank first if you need it fully current. */
uint64_t ds4_session_touched_kv_bytes(const ds4_session *s);
/* Tier-2 task #55 increment 2b — per-bank physical evict/restore for the proactive
 * eviction guard. free_physical: DIRECT cudaFree of one idle bank's split comp/index
 * (reclaims physical on GB10); caller must have snapshotted the bank's KV to DISK
 * first (host RAM reclaims nothing on unified memory) and repointed away from it.
 * alloc_physical: reallocate that bank's comp/index (VA; physical on touch) + rebuild
 * the base-pointer table; caller then reloads KV H2D from the disk snapshot.
 * is_evicted: whether a bank's physical is currently freed. bank_touched_kv_bytes:
 * one bank's exact resident comp/index KV from its frontier (guard Δ + victim pick). */
bool ds4_session_bank_free_physical(ds4_session *s, uint32_t bank);
bool ds4_session_bank_alloc_physical(ds4_session *s, uint32_t bank);
bool ds4_session_bank_is_evicted(const ds4_session *s, uint32_t bank);
uint64_t ds4_session_bank_touched_kv_bytes(ds4_session *s, uint32_t bank);
/* Raw per-bank comp/index KV disk snapshot (the eviction guard's bit-identical
 * mechanism; the D2H staging is transient — freed before free_physical). save:
 * precondition bank is installed (cur). load: reallocs physical + rebuilds the
 * base table + reinstalls counters, leaving bank installed. Return 0 on success. */
int ds4_session_bank_kv_save(ds4_session *s, uint32_t bank, FILE *fp, char *err, size_t errlen);
int ds4_session_bank_kv_load(ds4_session *s, uint32_t bank, FILE *fp, char *err, size_t errlen);
/* Conservative per-bank comp/index growth over one q-token decode quantum (the
 * guard's Delta term; over-charges the index side so the guard fires early). */
uint64_t ds4_session_quantum_growth_bytes_per_bank(ds4_session *s, uint32_t q);
/* Tier-2 PATH-A partial-prefix KV-reuse (plan-33 increment A) — FULL-PREFIX fork.
 * Clone bank `src`'s committed KV + host carry into bank `dst` (dst continues src's
 * conversation; the caller re-prefills only the divergent suffix). VALIDATES the
 * request tokens[0..n_cached) against src's committed history BEFORE any device
 * write (mismatch/short history -> refuse, non-zero return -> caller cold-prefills);
 * n_cached must equal src's committed length (full-prefix). Pins src against the
 * eviction guard for the clone. Returns 0 on success. fork_pinned: whether a bank
 * is currently mid-fork (the guard's victim picker must skip it). */
int  ds4_session_bank_fork(ds4_session *s, uint32_t src, uint32_t dst,
                           const int *tokens, int n_cached);
bool ds4_session_bank_fork_pinned(const ds4_session *s, uint32_t bank);
/* plan-33 increment C — PARTIAL-prefix fork: the request shares only
 * tokens[0..n_cached) with src's history. Cuts at R = ((n_cached-4)/128)*128,
 * validates tokens[0..R+4) vs src BEFORE any device write, clones [0,R) (+ the
 * byte-stashed ratio-4 boundary row, emit-restored during the replay), and makes
 * dst's committed history tokens[0..R) — the caller then re-prefills [R, ...).
 * src==dst = in-place truncate-reuse. 0 on success; non-zero -> cold prefill. */
int  ds4_session_bank_fork_partial(ds4_session *s, uint32_t src, uint32_t dst,
                                   const int *tokens, int n_cached);
/* GPU bytes the session's create actually allocated (allocator delta measured
 * across ds4_session_create).  Reconcile against
 * ds4_engine_session_cost_bytes after each create; commit this actual to any
 * memory ledger. */
uint64_t ds4_session_resident_bytes(const ds4_session *s);
/* Resident (mmap'd, read-only, shared) weight footprint in bytes: the main
 * GGUF plus an external drafter and expert overlay when mapped separately.
 * This competes with per-session KV for the unified-memory budget. */
uint64_t ds4_engine_weights_resident_bytes(ds4_engine *e);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_gpu_graph_test(ds4_engine *e, const ds4_tokens *prompt);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* Optional cooperative cancellation.  ds4_session_sync() checks it only at
 * safe boundaries where the live checkpoint is either unchanged or represents a
 * valid token prefix, and returns DS4_SESSION_SYNC_INTERRUPTED when it stops. */
void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
/* Tier-2 batched multi-session decode (bank pool, DS4_MSEQ_BANKS >= 2): one
 * decode request per co-scheduled session, all advanced one token by ONE
 * weight sweep.  reqs[k] = {TRUE bank id, absolute position of `token`,
 * current input token}.
 *
 * logits (out): row k is bank[k]'s next-token distribution, rows strided by
 * the engine's logits row width (the same width ds4_session_copy_logits
 * returns); logits_cap is the TOTAL float capacity of the buffer and is
 * rejected when it cannot hold n rows.  Sampling stays per-session on the
 * host with each session's own sampler/rng state.
 *
 * DETERMINISM: a session's row is reproducible at a FIXED batch composition,
 * and co-scheduling additional sessions does not perturb the existing rows
 * (per-row math is batch-width invariant for n >= 2).  The batched sweep is
 * NOT bit-identical to classic single-token decode, however — it is a
 * different kernel path — so a session's greedy token stream can diverge
 * from the same prompt decoded alone through ds4_session_eval once a
 * near-tie tips.  That is the two-mode contract (plan §2.4), not a bug.
 *
 * Requirements: each bank prefilled to exactly `pos` through the classic
 * single-bank path with its per-bank counters captured, one request per
 * bank, no position-0 rows, and NO speculation co-scheduled (n >= 2 is
 * plain decode by contract).
 *
 * `logits` receives n rows of ds4_engine_logits_width(engine) floats; row k
 * corresponds to reqs[k] (i.e. bank reqs[k].bank). `logits_cap` is the
 * buffer's capacity IN FLOATS and must be at least n * that width — size it
 * with ds4_engine_logits_width, never ds4_engine_vocab_size.
 *
 * On success the session's classic single-bank bookkeeping is INVALIDATED
 * (the scalar frontier counters now hold a cross-bank superset and the
 * checkpoint no longer describes any one bank): the caller owns per-bank
 * histories, and classic per-bank work must re-establish state explicitly.
 * Until it does, the classic single-session entries that would decode
 * against those superset counters FAIL LOUD rather than corrupt KV:
 * ds4_session_eval, ds4_session_generate_speculative and
 * ds4_session_eval_speculative_block all return an error. A ds4_session_sync
 * (which rebuilds from zero) re-establishes state and clears the condition.
 *
 * The session's own s->logits is NOT updated by a multiseq step (the step has
 * no single "the session's" row — every row belongs to a bank). Accordingly
 * ds4_session_argmax(s) / ds4_session_token_logprob(s, ...) keep returning
 * the pre-step classic distribution and must not be used to interpret a
 * multiseq result: sample from the `logits` rows returned here instead.
 *
 * Returns 0 on success; 1 on a recoverable rejection (bad args/contract; no
 * state mutated); -1 on a fatal mid-sweep failure (tear the session down). */
typedef struct {
    uint32_t bank;   /* TRUE bank id in the session's pool */
    int32_t  pos;    /* absolute position of `token` (bank's committed length) */
    int      token;  /* input token id decoded at `pos` */
} ds4_multiseq_req;
int ds4_session_decode_multiseq(ds4_session *s, const ds4_multiseq_req *reqs,
                                uint32_t n, float *logits, int logits_cap,
                                char *err, size_t errlen);
/* plan-34 phase-2: mixed prefill+decode batched step. SAME contract, kernel path,
 * and post-step invalidation as ds4_session_decode_multiseq, but the per-row
 * descriptor scratch is HEAP-allocated (up to the session's prefill_cap rows)
 * rather than the fixed DS4_MSEQ_MAX stack — the row representation the fused
 * mixed-batch step grows into. INCREMENT 1 is a byte-identical refactor: for a
 * decode-only batch (1 row per bank, n_rows == bank count) it produces the SAME
 * per-row logits as ds4_session_decode_multiseq. n_rows must be <= prefill_cap;
 * the underlying kernel still bounds the live row count to DS4_MSEQ_MAX this
 * increment (later increments lift it). Returns 0 / 1 / -1 as decode_multiseq. */
/* On success `logits` receives ONE row per per-bank RUN (the LAST row of each
 * run — plan-34 inc 3), in run order (ascending first-appearance of the bank);
 * *out_n_rows (may be NULL) is set to that count (== the number of distinct
 * banks). For a decode-only batch (1 row per bank) that is n_rows rows in bank
 * order — unchanged. For a K-row prefill run it is a single row (the K-th token's
 * logits). Size `logits` for the run count, not n_rows. */
/* max_head_runs (plan-34 inc 5, LEVER 1): emit head logits for only the FIRST
 * max_head_runs runs (0 = ALL runs = inc-3/4 behavior). A fused mixed step whose
 * trailing prefill run is on an INTERMEDIATE chunk passes n_dec (the decode banks
 * only): the prefill run's intermediate logits are never consumed, so skipping them
 * is unobservable AND lets the head take the single-block identity path (no
 * two-block gather/resync, no wasted K-row prefill head GEMM). *out_n_rows is then
 * the emitted count (== min(n_runs, max_head_runs), with 0 meaning n_runs). */
int ds4_session_decode_mixed(ds4_session *s, const ds4_multiseq_req *reqs,
                             uint32_t n_rows, float *logits, int logits_cap,
                             uint32_t *out_n_rows, uint32_t max_head_runs,
                             char *err, size_t errlen);
/* Tier-2 unified bank model (server-facing).  A bank-pooled session's graph
 * hosts up to N co-scheduled conversations as banks; each server slot maps to a
 * bank id.  The pool size is chosen at session create (DS4_MSEQ_BANKS today);
 * ds4_session_bank_count reports it (1 when not pooled).
 *
 * A conversation lives in ONE bank for its lifetime.  To do classic/spec work
 * on bank b the caller repoints the device views AND restores b's host carry;
 * ds4_session_bank_restore does both (+ clears the multiseq poison cheaply, no
 * re-prefill).  ds4_session_bank_save snapshots the live bank before switching
 * away.  Typical server flow:
 *   prefill:  ds4_session_bank_repoint(s,b); ds4_session_invalidate(s);
 *             ds4_session_sync(s, prompt, ...); ds4_session_bank_save(s,b);
 *   decode 1: ds4_session_bank_state_restore(s,b);
 *             ds4_session_generate_speculative(...);
 *             ds4_session_bank_state_save(s,b);
 *   decode N: ds4_session_decode_multiseq(s, reqs, N, ...) over the live banks
 *             (each captured via bank_state_save first); sample per row on host.
 * repoint/save/restore must run only between fully synchronized forwards. */
int  ds4_session_bank_count(ds4_session *s);
int  ds4_session_bank_repoint(ds4_session *s, uint32_t bank);
void ds4_session_bank_state_save(ds4_session *s, uint32_t bank);
bool ds4_session_bank_state_restore(ds4_session *s, uint32_t bank);
/* Per-bank frontier readers for a bank-pooled session: the committed length,
 * token history, and common-prefix-with-prompt of ONE bank, correct even when
 * that bank is not the currently-installed one (the live bank reads the live
 * checkpoint; others read their saved host carry).  ds4_session_pos/_tokens/
 * _common_prefix describe only the live bank, so server routing and /metrics
 * must use these for non-active banks.  Pure host reads (no CUDA); the caller
 * must keep idle banks' carries current (bank_state_save at job end). Return
 * 0 / NULL for an out-of-range or never-populated bank. */
int  ds4_session_bank_pos(ds4_session *s, uint32_t bank);
const ds4_tokens *ds4_session_bank_tokens(ds4_session *s, uint32_t bank);
int  ds4_session_bank_common_prefix(ds4_session *s, uint32_t bank,
                                    const ds4_tokens *prompt);
/* Tier-2: reconcile the host checkpoint after a run of ds4_session_decode_multiseq
 * steps advanced the LIVE (just bank_state_restore'd) bank's device KV frontier
 * without touching the host token history. Append the tokens multiseq committed,
 * in order, so ds4_session_pos/_tokens/_common_prefix and any subsequent classic
 * eval/sync/store see the true frontier. Pure host bookkeeping — no eval, no
 * CUDA. The device per-bank counters are already correct (the multiseq driver
 * maintained them and bank_state_restore installed them); this only catches the
 * host checkpoint up to them. */
void ds4_session_note_committed_tokens(ds4_session *s, const int *toks, int n);
int ds4_session_generate_speculative(ds4_session *s, float temperature, int top_k,
                                     float top_p, float min_p, uint64_t *rng,
                                     int max_tokens, int eos_token,
                                     int *accepted, int accepted_cap,
                                     char *err, size_t errlen);
int ds4_session_eval_speculative_block(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
/* Multi-session serving: prefill quantum policy. A server that time-slices
 * prefill interrupts ds4_session_sync() at chunk boundaries (via the cancel
 * callback) and re-issues the sync to resume. Returns the minimum remaining
 * suffix (target minus checkpoint, in tokens) that must still be pending for
 * an interruption to reproduce the uninterrupted chunk boundaries — hence
 * bit-identical KV state — or 0 when interrupting this session is never
 * exact and sync must run to completion. */
uint32_t ds4_session_prefill_quantum_min_suffix(const ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_output_head(ds4_engine *e);
bool ds4_engine_has_dspark(ds4_engine *e);
int ds4_engine_dspark_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

#ifdef __cplusplus
}
#endif

#endif
