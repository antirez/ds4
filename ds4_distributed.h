#ifndef DS4_DISTRIBUTED_H
#define DS4_DISTRIBUTED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4.h"

/* Distributed inference is an engine backend, not a separate frontend API.
 * Programs parse distributed options here, then keep using the normal
 * ds4_session_* calls. Only worker/coordinator serving modes call
 * ds4_dist_run() directly.
 */
typedef ds4_distributed_options ds4_dist_options;
typedef struct ds4_dist_session ds4_dist_session;

/* Options used by standalone `./ds4 --role coordinator -p ...` generation.
 * Interactive tools and the server go through the normal ds4_session API.
 */
typedef struct {
    const char *prompt;
    const char *system;
    const char *dump_logits_path;
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    ds4_think_mode think_mode;
} ds4_dist_generation_options;

typedef enum {
    DS4_DIST_CLI_ERROR = -1,
    DS4_DIST_CLI_NOT_MATCHED = 0,
    DS4_DIST_CLI_MATCHED = 1,
} ds4_dist_cli_parse_result;

/* Shared option parsing. */
bool ds4_dist_enabled(const ds4_dist_options *opt);
ds4_dist_options *ds4_dist_options_create(void);
void ds4_dist_options_free(ds4_dist_options *opt);
void ds4_dist_usage(FILE *fp);
ds4_dist_cli_parse_result ds4_dist_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_dist_options *opt,
        char *err,
        size_t errlen);

/* Applies distributed layer-loading choices to the engine options before the
 * model is loaded.
 */
int ds4_dist_prepare_engine_options(
        const ds4_dist_options *opt,
        ds4_engine_options *engine,
        char *err,
        size_t errlen);

/* Coordinator session backend used by ds4.c. These mirror the normal session
 * operations; callers outside the engine should not need to call them directly.
 */
int ds4_dist_session_create(
        ds4_dist_session **out,
        ds4_engine *engine,
        const ds4_dist_options *opt,
        ds4_session *owner,
        int ctx_size,
        char *err,
        size_t errlen,
        ds4_dist_session *parent);
void ds4_dist_session_free(ds4_dist_session *d);

/* Returns 1 when the coordinator has full layer coverage, 0 when workers are
 * still missing, and -1 for configuration or internal errors.
 */
int ds4_dist_session_route_ready(ds4_dist_session *d, char *err, size_t errlen);

/* Synchronize the distributed KV state to the requested prompt timeline. */
int ds4_dist_session_sync(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        const ds4_tokens *prompt,
        float *logits,
        char *err,
        size_t errlen);

/* Evaluate one additional token across the current distributed route. */
int ds4_dist_session_eval(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int token,
        float *logits,
        char *err,
        size_t errlen);

/* Maximum rows in one multi-session decode span. */
#define DS4_DIST_MAX_MULTI_ROWS 16u

/* True when the route's worker negotiated row-batched decode spans. */
bool ds4_dist_session_batch_cap(ds4_dist_session *d);

/* Coordinator local layer range for this dist session. */
int ds4_dist_session_local_layers(const ds4_dist_session *d,
                                  uint32_t *start,
                                  uint32_t *end);

/* The session's wire id (per-row ownership in batched spans). */
uint64_t ds4_dist_session_id(const ds4_dist_session *d);

/* One decode span carrying rows owned by different coordinator sessions:
 * tokens[i] is evaluated for the session plane row_session_ids[i] with the
 * precomputed leader-side hidden row hidden_rows + i*hidden_f32_values;
 * logits_rows receives count × vocab f32 in row order. The owner session
 * drives request-id sequencing and the span's timeline prefix hash. */
int ds4_dist_eval_batch_span(
        ds4_dist_session *owner,
        ds4_session *owner_session,
        const int *tokens,
        const uint64_t *row_session_ids,
        const float *hidden_rows,
        uint32_t count,
        float *logits_rows,
        char *err,
        size_t errlen);

/* Legacy-MTP speculative decode cycle over the pipeline split.  Decodes
 * first_token, then drafts, verifies, and commits as many follow-up tokens
 * as the worker's MTP head and the target model agree on.  Requires the
 * --mtp support model and --mtp-draft >= 2 on both machines; otherwise
 * callers should use the ordinary per-token path. */
int ds4_dist_session_mtp_spec_cycle(
        ds4_dist_session *d,
        ds4_session *owner,
        int first_token,
        int max_tokens,
        int eos_token,
        int *accepted,
        int accepted_cap,
        char *err,
        size_t errlen);

/* Print the dist speculative-decode telemetry counters accumulated by
 * ds4_dist_session_mtp_spec_cycle.  Gated by DS4_DSPARK_STATS like the
 * single-node DSpark stats line, so an external validator can enable both
 * with the same env knob.  Emits nothing when no speculative cycle ran. */
void ds4_dist_session_print_spec_stats(const ds4_dist_session *d);

/* Save/load use the normal DSV4 payload format. The coordinator gathers or
 * pushes remote layer shards internally so saved files are topology-neutral.
 */
int ds4_dist_session_save_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        char *err,
        size_t errlen);
int ds4_dist_session_load_payload(
        ds4_dist_session *d,
        ds4_session *owner,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen);

/* Standalone distributed mode. Workers stay in this loop; coordinator one-shot
 * mode uses it for `./ds4 --role coordinator -p ...`.
 */
int ds4_dist_run(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen);

#endif
