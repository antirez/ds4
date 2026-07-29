#include "provider.h"

#include "../../ds4_model_provider_builtin.h"

static const ds4_model_provider_v1 DS4_DEEPSEEK_PROVIDER = {
    .abi_version = DS4_MODEL_PROVIDER_ABI_VERSION,
    .struct_size = sizeof(ds4_model_provider_v1),
    .id = "deepseek-v4",
    .session_create = ds4_deepseek_session_create,
    .session_destroy = ds4_deepseek_session_destroy,
    .session_sync = ds4_deepseek_session_sync,
    .session_eval = ds4_deepseek_session_eval,
    .sessions_eval_batch = ds4_builtin_sessions_eval_batch,
    .sessions_eval_batch_with_prefill =
        ds4_builtin_sessions_eval_batch_with_prefill,
    .session_eval_speculative = ds4_deepseek_session_eval_speculative,
    .session_invalidate = ds4_deepseek_session_invalidate,
    .session_rewind = ds4_deepseek_session_rewind,
    .session_layer_slice_reset = ds4_deepseek_session_layer_slice_reset,
    .session_eval_output_head = ds4_deepseek_session_eval_output_head,
    .session_eval_layer_slice = ds4_deepseek_session_eval_layer_slice,
    .session_payload_bytes = ds4_deepseek_session_payload_bytes,
    .session_save_payload = ds4_deepseek_session_save_payload,
    .session_load_payload = ds4_deepseek_session_load_payload,
    .session_layer_payload_bytes =
        ds4_deepseek_session_layer_payload_bytes,
    .session_save_layer_payload =
        ds4_deepseek_session_save_layer_payload,
    .session_load_layer_payload =
        ds4_deepseek_session_load_layer_payload,
};

const ds4_model_provider_v1 *ds4_deepseek_model_provider(void) {
    return &DS4_DEEPSEEK_PROVIDER;
}
