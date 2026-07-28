#ifndef DS4_MODEL_PROVIDER_BUILTIN_H
#define DS4_MODEL_PROVIDER_BUILTIN_H

#include "ds4_model_provider.h"

const ds4_model_provider_v1 *ds4_deepseek_model_provider(void);
const ds4_model_provider_v1 *ds4_glm_model_provider(void);
bool ds4_model_provider_valid(const ds4_model_provider_v1 *provider);

int ds4_deepseek_session_create(ds4_session **out,
                                ds4_engine *engine,
                                int context_size);
int ds4_glm_session_create(ds4_session **out,
                           ds4_engine *engine,
                           int context_size);
void ds4_deepseek_session_destroy(ds4_session *session);
void ds4_glm_session_destroy(ds4_session *session);
int ds4_deepseek_session_sync(ds4_session *session,
                              const ds4_tokens *prompt,
                              char *err,
                              size_t errlen);
int ds4_glm_session_sync(ds4_session *session,
                         const ds4_tokens *prompt,
                         char *err,
                         size_t errlen);
int ds4_deepseek_session_eval(ds4_session *session,
                              int token,
                              bool probe_support_model,
                              char *err,
                              size_t errlen);
int ds4_glm_session_eval(ds4_session *session,
                         int token,
                         bool probe_support_model,
                         char *err,
                         size_t errlen);
int ds4_builtin_sessions_eval_batch(ds4_decode_item *items,
                                    int count,
                                    char *err,
                                    size_t errlen);
int ds4_builtin_sessions_eval_batch_with_prefill(
        ds4_decode_item *items,
        int count,
        ds4_session *prefill_session,
        const ds4_tokens *prefill_prompt,
        char *err,
        size_t errlen);
int ds4_deepseek_session_eval_speculative(
        ds4_session *session,
        int first_token,
        int max_tokens,
        int eos_token,
        int *accepted,
        int accepted_cap,
        char *err,
        size_t errlen);
int ds4_glm_session_eval_speculative(
        ds4_session *session,
        int first_token,
        int max_tokens,
        int eos_token,
        int *accepted,
        int accepted_cap,
        char *err,
        size_t errlen);
void ds4_deepseek_session_invalidate(ds4_session *session);
void ds4_glm_session_invalidate(ds4_session *session);
void ds4_deepseek_session_rewind(ds4_session *session, int position);
void ds4_glm_session_rewind(ds4_session *session, int position);
int ds4_deepseek_session_layer_slice_reset(ds4_session *session,
                                           char *err,
                                           size_t errlen);
int ds4_glm_session_layer_slice_reset(ds4_session *session,
                                      char *err,
                                      size_t errlen);
int ds4_deepseek_session_eval_output_head(
        ds4_session *session,
        const float *hidden_state,
        uint32_t token_count,
        float *logits,
        char *err,
        size_t errlen);
int ds4_glm_session_eval_output_head(
        ds4_session *session,
        const float *hidden_state,
        uint32_t token_count,
        float *logits,
        char *err,
        size_t errlen);
int ds4_deepseek_session_eval_layer_slice(
        ds4_session *session,
        const int *tokens,
        uint32_t token_count,
        uint32_t position,
        uint32_t layer_start,
        uint32_t layer_end,
        const float *input_hidden_state,
        float *output_hidden_state,
        bool output_logits,
        float *logits,
        char *err,
        size_t errlen);
int ds4_glm_session_eval_layer_slice(
        ds4_session *session,
        const int *tokens,
        uint32_t token_count,
        uint32_t position,
        uint32_t layer_start,
        uint32_t layer_end,
        const float *input_hidden_state,
        float *output_hidden_state,
        bool output_logits,
        float *logits,
        char *err,
        size_t errlen);
uint64_t ds4_deepseek_session_payload_bytes(ds4_session *session);
uint64_t ds4_glm_session_payload_bytes(ds4_session *session);
int ds4_deepseek_session_save_payload(ds4_session *session,
                                      FILE *file,
                                      char *err,
                                      size_t errlen);
int ds4_glm_session_save_payload(ds4_session *session,
                                 FILE *file,
                                 char *err,
                                 size_t errlen);
int ds4_deepseek_session_load_payload(ds4_session *session,
                                      FILE *file,
                                      uint64_t payload_bytes,
                                      char *err,
                                      size_t errlen);
int ds4_glm_session_load_payload(ds4_session *session,
                                 FILE *file,
                                 uint64_t payload_bytes,
                                 char *err,
                                 size_t errlen);
uint64_t ds4_deepseek_session_layer_payload_bytes(
        ds4_session *session,
        uint32_t layer_start,
        uint32_t layer_end);
uint64_t ds4_glm_session_layer_payload_bytes(
        ds4_session *session,
        uint32_t layer_start,
        uint32_t layer_end);
int ds4_deepseek_session_save_layer_payload(
        ds4_session *session,
        FILE *file,
        uint32_t layer_start,
        uint32_t layer_end,
        char *err,
        size_t errlen);
int ds4_glm_session_save_layer_payload(
        ds4_session *session,
        FILE *file,
        uint32_t layer_start,
        uint32_t layer_end,
        char *err,
        size_t errlen);
int ds4_deepseek_session_load_layer_payload(
        ds4_session *session,
        FILE *file,
        uint64_t payload_bytes,
        const int *tokens,
        uint32_t token_count,
        uint32_t layer_start,
        uint32_t layer_end,
        char *err,
        size_t errlen);
int ds4_glm_session_load_layer_payload(
        ds4_session *session,
        FILE *file,
        uint64_t payload_bytes,
        const int *tokens,
        uint32_t token_count,
        uint32_t layer_start,
        uint32_t layer_end,
        char *err,
        size_t errlen);

#endif
