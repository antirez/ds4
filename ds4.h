#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CPU,
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

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);

typedef struct {
    const char *model_path;
    const char *mtp_path;
    ds4_backend backend;
    int n_threads;
    int mtp_draft_tokens;
    float mtp_margin;
    bool warm_weights;
    bool quality;
    /* Pipeline-parallel layer range.  Defaults (0, 0) select the full model
     * [0, DS4_N_LAYER): n_layer_end <= 0 normalizes to DS4_N_LAYER.  Partition
     * the model across a head process (e.g. [0, 22)) and a tail RPC worker
     * (e.g. [22, 43)) by setting these.  Single-host operation uses the
     * default and remains bit-identical to pre-RPC behavior. */
    int n_layer_start;
    int n_layer_end;

    /* Optional RPC peer.  When rpc_peer_host is set, the engine acts as the
     * head: after this engine's local slice runs (layers [n_layer_start,
     * n_layer_end)), the residual is shipped to the configured peer over
     * TCP, and the peer's logits reply replaces local sampling input.  The
     * peer must run ds4-rpc-worker with --layer-start equal to n_layer_end
     * and --layer-end DS4_N_LAYER.  Unused on the worker side.  When unset
     * the engine is fully self-sufficient. */
    const char *rpc_peer_host;
    int         rpc_peer_port;
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

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

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

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live graph cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the graph is
 * refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Disk KV cache payload helpers.  The server owns the outer file header and
 * policy; the engine owns the DS4-specific serialized graph state. */
uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);

/* Pipeline-parallel residual stream transfer (the cur_hc tensor: hyper-
 * connection slots times embedding width, in floats).  The returned size is
 * model-fixed.  Export reads the residual that comes out of this engine's
 * last owned layer; import installs an incoming residual ahead of the next
 * forward pass.  Both are no-ops for engines that own the full layer range,
 * but the API works in all cases so single-host and pipeline code paths can
 * share call sites.  Metal backend only; CPU returns an error string. */
uint64_t ds4_residual_hc_floats(void);
int ds4_session_export_residual_hc(ds4_session *s, float *out, uint64_t n_floats,
                                   char *err, size_t errlen);
int ds4_session_import_residual_hc(ds4_session *s, const float *in, uint64_t n_floats,
                                   char *err, size_t errlen);

/* Shape accessors so the RPC transport and worker don't have to duplicate
 * DS4_N_* compile-time constants in their own code.  All four values are
 * model-fixed today; if a future DS4 variant changes them they will become
 * engine-derived and these accessors will keep callers stable. */
uint32_t ds4_model_n_layer(void);
uint32_t ds4_model_n_embd(void);
uint32_t ds4_model_n_hc(void);
uint32_t ds4_model_n_vocab(void);

/* Resolve the GGUF model path for a CLI/server launch.  Priority:
 *   1. If explicit_path is non-NULL/non-empty, return it unchanged.
 *   2. Else if quant ("q2" or "q4") is non-NULL, return the canonical
 *      path for that quant.  If the file is missing, fail.
 *   3. Else probe ./gguf/ for the canonical Q2 and Q4 files.  If both
 *      are present, prefer Q2 (the smaller / safer-for-128GB default).
 *      If only one is present, return it.  If neither, return
 *      "ds4flash.gguf" so the historical symlink mechanism still works.
 *
 * Returned pointer is either the caller's explicit_path or static
 * storage; do not free.  Returns NULL on error with a human-readable
 * reason in err. */
const char *ds4_resolve_model_path(const char *explicit_path,
                                   const char *quant,
                                   char *err, size_t errlen);

/* Read-only view of the last logits produced by ds4_session_eval or the
 * tail of ds4_session_sync.  Length is ds4_model_n_vocab() floats.  Returns
 * NULL if the session has not yet produced logits.  The RPC worker uses
 * this to ship a full logit vector back to the head without copying through
 * top_logprobs. */
const float *ds4_session_logits(const ds4_session *s);

#endif
