#include "ds4_model_provider_builtin.h"

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

static const ds4_model_provider_v1 DS4_GLM_PROVIDER = {
    .abi_version = DS4_MODEL_PROVIDER_ABI_VERSION,
    .struct_size = sizeof(ds4_model_provider_v1),
    .id = "glm-dsa",
    .session_create = ds4_glm_session_create,
    .session_destroy = ds4_glm_session_destroy,
    .session_sync = ds4_glm_session_sync,
    .session_eval = ds4_glm_session_eval,
    .sessions_eval_batch = ds4_builtin_sessions_eval_batch,
    .sessions_eval_batch_with_prefill =
        ds4_builtin_sessions_eval_batch_with_prefill,
    .session_eval_speculative = ds4_glm_session_eval_speculative,
    .session_invalidate = ds4_glm_session_invalidate,
    .session_rewind = ds4_glm_session_rewind,
    .session_layer_slice_reset = ds4_glm_session_layer_slice_reset,
    .session_eval_output_head = ds4_glm_session_eval_output_head,
    .session_eval_layer_slice = ds4_glm_session_eval_layer_slice,
    .session_payload_bytes = ds4_glm_session_payload_bytes,
    .session_save_payload = ds4_glm_session_save_payload,
    .session_load_payload = ds4_glm_session_load_payload,
    .session_layer_payload_bytes = ds4_glm_session_layer_payload_bytes,
    .session_save_layer_payload = ds4_glm_session_save_layer_payload,
    .session_load_layer_payload = ds4_glm_session_load_layer_payload,
};

const ds4_model_provider_v1 *ds4_deepseek_model_provider(void) {
    return &DS4_DEEPSEEK_PROVIDER;
}

const ds4_model_provider_v1 *ds4_glm_model_provider(void) {
    return &DS4_GLM_PROVIDER;
}

bool ds4_model_provider_valid(const ds4_model_provider_v1 *provider) {
    return provider &&
           provider->abi_version == DS4_MODEL_PROVIDER_ABI_VERSION &&
           provider->struct_size >= sizeof(ds4_model_provider_v1) &&
           provider->id &&
           provider->session_create &&
           provider->session_destroy &&
           provider->session_sync &&
           provider->session_eval &&
           provider->sessions_eval_batch &&
           provider->sessions_eval_batch_with_prefill &&
           provider->session_eval_speculative &&
           provider->session_invalidate &&
           provider->session_rewind &&
           provider->session_layer_slice_reset &&
           provider->session_eval_output_head &&
           provider->session_eval_layer_slice &&
           provider->session_payload_bytes &&
           provider->session_save_payload &&
           provider->session_load_payload &&
           provider->session_layer_payload_bytes &&
           provider->session_save_layer_payload &&
           provider->session_load_layer_payload;
}
