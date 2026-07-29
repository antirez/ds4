#include "ds4_model_provider_builtin.h"

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
