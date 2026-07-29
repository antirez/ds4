#ifndef DS4_MODEL_PROVIDER_BUILTIN_H
#define DS4_MODEL_PROVIDER_BUILTIN_H

#include "ds4_model_provider.h"

/*
 * Shared entry points used by the built-in providers. Model-specific
 * lifecycle declarations live beside each provider under models/<model>/.
 */
bool ds4_model_provider_valid(const ds4_model_provider_v1 *provider);

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

#endif
