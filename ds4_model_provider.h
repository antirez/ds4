#ifndef DS4_MODEL_PROVIDER_H
#define DS4_MODEL_PROVIDER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4.h"

/*
 * Source-level boundary between the engine core and a model integration.
 *
 * A provider owns whole-model orchestration. It may call its custom kernels
 * directly and keep model-specific graph state private; the core only enters
 * through these lifecycle operations. This is intentionally not an operator
 * or individual-kernel interface.
 */
#define DS4_MODEL_PROVIDER_ABI_VERSION 1u

typedef struct ds4_model_provider_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *id;

    int (*session_create)(ds4_session **out,
                          ds4_engine *engine,
                          int context_size);

    /*
     * Release only provider-owned state embedded in the session. The core
     * releases transport, checkpoint, sampling, and session storage.
     */
    void (*session_destroy)(ds4_session *session);

    int (*session_sync)(ds4_session *session,
                        const ds4_tokens *prompt,
                        char *err,
                        size_t errlen);

    int (*session_eval)(ds4_session *session,
                        int token,
                        bool probe_support_model,
                        char *err,
                        size_t errlen);

    int (*sessions_eval_batch)(ds4_decode_item *items,
                               int count,
                               char *err,
                               size_t errlen);

    int (*sessions_eval_batch_with_prefill)(
            ds4_decode_item *items,
            int count,
            ds4_session *prefill_session,
            const ds4_tokens *prefill_prompt,
            char *err,
            size_t errlen);

    int (*session_eval_speculative)(ds4_session *session,
                                    int first_token,
                                    int max_tokens,
                                    int eos_token,
                                    int *accepted,
                                    int accepted_cap,
                                    char *err,
                                    size_t errlen);

    /* Update provider-private cache frontiers after core checkpoint changes. */
    void (*session_invalidate)(ds4_session *session);
    void (*session_rewind)(ds4_session *session, int position);

    /* Pipeline-parallel execution of a contiguous transformer slice. */
    int (*session_layer_slice_reset)(ds4_session *session,
                                     char *err,
                                     size_t errlen);

    int (*session_eval_output_head)(ds4_session *session,
                                    const float *hidden_state,
                                    uint32_t token_count,
                                    float *logits,
                                    char *err,
                                    size_t errlen);

    int (*session_eval_layer_slice)(ds4_session *session,
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

    uint64_t (*session_payload_bytes)(ds4_session *session);

    int (*session_save_payload)(ds4_session *session,
                                FILE *file,
                                char *err,
                                size_t errlen);

    int (*session_load_payload)(ds4_session *session,
                                FILE *file,
                                uint64_t payload_bytes,
                                char *err,
                                size_t errlen);

    uint64_t (*session_layer_payload_bytes)(ds4_session *session,
                                            uint32_t layer_start,
                                            uint32_t layer_end);

    int (*session_save_layer_payload)(ds4_session *session,
                                      FILE *file,
                                      uint32_t layer_start,
                                      uint32_t layer_end,
                                      char *err,
                                      size_t errlen);

    int (*session_load_layer_payload)(ds4_session *session,
                                      FILE *file,
                                      uint64_t payload_bytes,
                                      const int *tokens,
                                      uint32_t token_count,
                                      uint32_t layer_start,
                                      uint32_t layer_end,
                                      char *err,
                                      size_t errlen);
} ds4_model_provider_v1;

#endif
