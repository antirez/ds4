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
        size_t errlen);
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
