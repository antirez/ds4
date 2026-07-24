#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu_mgpu.h"
#include "ds4_gpu.h"
#include "ds4_gpu_args.h"

/* g_gpu / g_n_gpus / g_gpu_peer_ok storage, ds4_gpu_init_multi,
 * ds4_gpu_set_current_device[_fenced], the tensor_alloc_on family,
 * tensor_free_in_place, tensor_device, the tensor_copy_xdev/wait_xdev
 * families, and ds4_gpu_tier_free_vram now live in rocm/ds4_rocm_mgpu.cuh
 * (included from ds4_rocm.cu). That header needs direct access to
 * runtime internals (the model-map cache, per-device selective weight
 * cache, hipBLAS/hipBLASLt state) that only exist in ds4_rocm.cu's
 * translation unit -- this file (ds4_rocm_compat.cu) is a separate TU with
 * no visibility into those statics, so it can only host functions that
 * are self-contained HIP calls or thin wrappers around the public
 * ds4_gpu.h API. */

extern "C" int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *dst,
                                           const ds4_gpu_tensor *src,
                                           uint64_t bytes) {
    if (!dst || !src || bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0) return 1;
    return hipMemcpyAsync(dst->ptr, src->ptr, (size_t)bytes,
                          hipMemcpyDeviceToDevice, 0) == hipSuccess;
}

extern "C" int ds4_gpu_args_probe_auto_cuda(
        const int *device_filter, int filter_len, ds4_gpu_config *out,
        size_t safety_margin_bytes, char *errbuf, size_t errbuflen) {
    if (!out) {
        if (errbuf && errbuflen) snprintf(errbuf, errbuflen, "internal: NULL out");
        return 1;
    }
    int visible = 0;
    hipError_t rc = hipGetDeviceCount(&visible);
    if (rc != hipSuccess || visible <= 0) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen, "hipGetDeviceCount failed: %s",
                     rc == hipSuccess ? "no devices" : hipGetErrorString(rc));
        }
        return 1;
    }
    /* Build the device list: either the explicit filter or 0..visible-1. */
    int devs[DS4_MAX_GPUS];
    int n_dev = 0;
    if (device_filter && filter_len > 0) {
        if (filter_len > DS4_MAX_GPUS) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen,
                         "--gpu-devices filter has %d entries (max %d)",
                         filter_len, DS4_MAX_GPUS);
            }
            return 1;
        }
        for (int i = 0; i < filter_len; i++) {
            int d = device_filter[i];
            if (d < 0 || d >= visible) {
                if (errbuf && errbuflen) {
                    snprintf(errbuf, errbuflen,
                             "--gpu-devices: device %d not in 0..%d",
                             d, visible - 1);
                }
                return 1;
            }
            devs[n_dev++] = d;
        }
    } else {
        int cap = visible < DS4_MAX_GPUS ? visible : DS4_MAX_GPUS;
        for (int i = 0; i < cap; i++) devs[n_dev++] = i;
    }
    memset(out, 0, sizeof(*out));
    out->n_gpus = n_dev;
    out->safety_margin_bytes = safety_margin_bytes;
    for (int i = 0; i < n_dev; i++) {
        int d = devs[i];
        if (hipSetDevice(d) != hipSuccess) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen, "hipSetDevice(%d) failed", d);
            }
            return 1;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        rc = hipMemGetInfo(&free_bytes, &total_bytes);
        if (rc != hipSuccess) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen, "hipMemGetInfo on device %d failed: %s",
                         d, hipGetErrorString(rc));
            }
            return 1;
        }
        /* Reserve = max(2 GiB, 5% of free); mirrors CUDA's auto-probe
         * reserve. Explicit --gpu-vram budgets bypass this probe. */
        const size_t reserve_floor = (size_t)2ull * 1024ull * 1024ull * 1024ull;
        const size_t reserve_pct = free_bytes / 20u;
        const size_t reserve = reserve_floor > reserve_pct ? reserve_floor : reserve_pct;
        out->device_indices[i] = d;
        out->vram_bytes[i] = free_bytes > reserve ? free_bytes - reserve : 0;
    }
    return 0;
}

extern "C" void ds4_gpu_enable_q8_dequant_gemm(void) {
}

static int g_rocm_q8_cache_suppressed = 0;

extern "C" int ds4_gpu_q8_cache_suppressed(void) {
    return g_rocm_q8_cache_suppressed;
}

extern "C" void ds4_gpu_set_q8_cache_suppressed(int suppressed) {
    g_rocm_q8_cache_suppressed = suppressed != 0;
}

extern "C" int ds4_gpu_set_decode_fast_attention(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_set_decode_score_vec4(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                      weight_offset, in_dim, out_dim, x,
                                      n_rows);
}

extern "C" int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1, const void *model_map,
        uint64_t model_size, uint64_t weight0_offset,
        uint64_t weight1_offset, uint64_t in_dim, uint64_t out0_dim,
        uint64_t out1_dim, const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_q8_0_pair_tensor(
            out0, out1, model_map, model_size, weight0_offset, weight1_offset,
            in_dim, out0_dim, out1_dim, x, n_rows);
}

extern "C" int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_f16_tensor(out, model_map, model_size, weight_offset,
                                     4096u, 256u, x, n_rows);
}

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size,
        uint64_t q_weight_offset, uint32_t q_n,
        ds4_gpu_tensor *kv_out, const ds4_gpu_tensor *kv,
        uint64_t kv_weight_offset, uint32_t kv_n, uint32_t rows,
        uint32_t kv_n_head, uint32_t kv_head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float eps) {
    return ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                   q_out, q, model_map, model_size, q_weight_offset, q_n,
                   kv_out, kv, kv_weight_offset, kv_n, rows, eps) != 0 &&
           ds4_gpu_rope_tail_tensor(
                   kv_out, rows, kv_n_head, kv_head_dim, n_rot, pos0,
                   n_ctx_orig, inverse, freq_base, freq_scale, ext_factor,
                   attn_factor, beta_fast, beta_slow) != 0;
}

extern "C" int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint32_t n_vocab,
        uint32_t token, uint32_t n_embd) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_embed_token_q8_0_tensor(out, model_map, model_size,
                                           weight_offset, n_vocab, token,
                                           n_embd);
}

extern "C" int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_vocab, uint32_t n_tokens,
        uint32_t n_embd) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_embed_tokens_q8_0_tensor(out, tokens, model_map, model_size,
                                            weight_offset, n_vocab, n_tokens,
                                            n_embd);
}

extern "C" int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t in_dim,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (weight_type == 8u) {
        return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                          weight_offset, in_dim, out_dim, x,
                                          n_tok);
    }
    if (weight_type == 1u) {
        return ds4_gpu_matmul_f16_tensor(out, model_map, model_size,
                                         weight_offset, in_dim, out_dim, x,
                                         n_tok);
    }
    return 0;
}

extern "C" int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t in_dim,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (weight_type == 8u) {
        return ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
                out, model_map, model_size, weight_offset, in_dim, out_dim,
                x, n_tok);
    }
    return ds4_gpu_matmul_quant_tensor(out, model_map, model_size,
                                       weight_offset, weight_type, in_dim,
                                       out_dim, x, n_tok);
}

extern "C" int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *kv_norm,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t n_head) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_k_b_project_tensor(out, kv_norm, model_map, model_size,
                                           weight_offset, n_tokens,
                                           kv_lora_dim, qk_nope, n_head);
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t qk_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_qk_lowrank_q8_0_tensor(
            qk_low, q, model_map, model_size, weight_offset, n_head,
            kv_lora_dim, qk_nope, qk_dim);
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_head,
        uint32_t kv_lora_dim, uint32_t qk_nope, uint32_t qk_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
            qk_low, q, model_map, model_size, weight_offset, n_tokens, n_head,
            kv_lora_dim, qk_nope, qk_dim);
}

extern "C" int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *lora,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_head,
        uint32_t kv_lora_dim, uint32_t value_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
            heads, lora, model_map, model_size, weight_offset, n_tokens,
            n_head, kv_lora_dim, value_dim);
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        uint32_t value_weight_type, const ds4_gpu_tensor *selected,
        uint32_t n_selected, uint32_t cache_cap, bool cache_f16,
        uint32_t n_head, uint32_t kv_lora_dim, uint32_t qk_nope,
        uint32_t qk_rope, uint32_t value_dim, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {
    if (value_weight_type != 8u) return 0;
    return ds4_gpu_glm_attention_indexed_decode_tensor(
            heads, q, qk_low, kv_lora_cache, k_rope_cache, model_map,
            model_size, value_weight_offset, selected, n_selected, cache_cap,
            cache_f16, n_head, kv_lora_dim, qk_nope, qk_rope, value_dim,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow);
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        uint32_t value_weight_type, const ds4_gpu_tensor *selected,
        uint32_t n_tokens, uint32_t n_selected, uint32_t cache_cap,
        bool cache_f16, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t qk_rope, uint32_t value_dim,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow) {
    if (value_weight_type != 8u) return 0;
    return ds4_gpu_glm_attention_indexed_batch_tensor(
            heads, q, qk_low, kv_lora_cache, k_rope_cache, model_map,
            model_size, value_weight_offset, selected, n_tokens, n_selected,
            cache_cap, cache_f16, n_head, kv_lora_dim, qk_nope, qk_rope,
            value_dim, n_ctx_orig, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow);
}
