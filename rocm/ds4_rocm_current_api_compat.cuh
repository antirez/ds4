// Folded RMS preparation and every prepared-RHS consumer currently use the
// default stream. Decline capture before scratch allocation: the common arena
// can grow/free its previous buffer and is not graph-owned.
static bool rocm_hc_prefill_stream_ready(void) {
#ifdef __HIP_PLATFORM_AMD__
    hipStream_t stream = nullptr;
    if (hipblasGetStream(g_cublas, &stream) != HIPBLAS_STATUS_SUCCESS || stream != nullptr) return false;
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    return hipStreamIsCapturing(stream, &capture) == hipSuccess &&
           capture == hipStreamCaptureStatusNone;
#else
    return false;
#endif
}

extern "C" int ds4_gpu_matmul_f16_rms_fold_tensor(
        ds4_gpu_tensor *out, const void *model_map,
        uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok, float norm_eps) {
    using namespace ds4_hc_norm_mix;
    ds4_rocm_hc_prefill::sizes bytes;
    if (!out || !x ||
        !ds4_rocm_hc_prefill::eligible(in_dim, out_dim, n_tok,
            ds4_gpu_get_execution_phase(), g_quality_mode, g_glm_model, g_cublas_ready,
            (uintptr_t)out->ptr, out->bytes, (uintptr_t)x->ptr, x->bytes,
            (uintptr_t)model_map, model_size, weight_offset, norm_eps, &bytes) ||
        !rocm_hc_prefill_stream_ready()) return 0;

    const char *wptr = cuda_model_range_ptr(
            model_map, weight_offset, bytes.weights, "HC prefill RMS-fold F16");
    if (!range_valid((uintptr_t)wptr, bytes.weights) ||
        ((uintptr_t)wptr & 1u) ||
        overlaps((uintptr_t)out->ptr, bytes.output, (uintptr_t)wptr, bytes.weights)) return 0;
    // The existing allocation may alias a caller's view. Check before growth
    // could free it, and check the returned region again before submitting.
    if (g_cuda_tmp &&
        (overlaps((uintptr_t)g_cuda_tmp, g_cuda_tmp_bytes, (uintptr_t)out->ptr, bytes.output) ||
         overlaps((uintptr_t)g_cuda_tmp, g_cuda_tmp_bytes, (uintptr_t)x->ptr, bytes.input) ||
         overlaps((uintptr_t)g_cuda_tmp, g_cuda_tmp_bytes, (uintptr_t)wptr, bytes.weights))) return 0;
    __half *xh = (__half *)cuda_tmp_alloc(bytes.half, "HC prefill RMS-fold activations");
    if (!range_valid((uintptr_t)xh, bytes.half) || ((uintptr_t)xh & 1u) ||
        overlaps((uintptr_t)xh, bytes.half, (uintptr_t)out->ptr, bytes.output) ||
        overlaps((uintptr_t)xh, bytes.half, (uintptr_t)x->ptr, bytes.input) ||
        overlaps((uintptr_t)xh, bytes.half, (uintptr_t)wptr, bytes.weights)) return 0;

    rocm_hc_rms_norm_f16_kernel<<<(uint32_t)n_tok, 256u>>>(
            xh, (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)n_tok, norm_eps);
    if (!cuda_ok(cudaGetLastError(), "HC prefill RMS-fold activation launch")) return -1;
    return rocm_matmul_f16_prepared_tensor(out, (const __half *)wptr, xh,
            in_dim, out_dim, n_tok, true) ? 1 : -1;
}

static int rocm_hc_norm_mix_device_supported(void) {
    if (!g_cublas_ready || g_quality_mode) return 0;
#ifdef __HIP_PLATFORM_AMD__
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    static thread_local int cached_device = -1;
    static thread_local int supported = 0;
    if (cached_device != device) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return 0;
        supported = prop.warpSize == 32 && prop.maxThreadsPerBlock >= 256 &&
                    strncmp(prop.gcnArchName, "gfx1151", 7) == 0 &&
                    (prop.gcnArchName[7] == '\0' || prop.gcnArchName[7] == ':');
        cached_device = device;
    }
    return supported;
#else
    return 0;
#endif
}

extern "C" int ds4_gpu_hc_rms_norm_mix_f16_available(void) {
    // The scalar-scratch candidate avoids 64 KiB of normalized output, but
    // applies the scale in each of 24 projection rows (376832 extra FP32
    // multiplies). Keep graph dispatch on the existing path until native
    // gfx1151 parity and timing justify this compute/memory tradeoff. The
    // explicit tensor API below remains callable by the native A/B fixture.
    return 0;
}

extern "C" int ds4_gpu_hc_rms_norm_mix_f16_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n, uint32_t out_dim, float eps) {
    using namespace ds4_hc_norm_mix;
    if (!out || !x ||
        !eligible(n, out_dim, ds4_gpu_get_execution_phase(), g_quality_mode,
                  (uintptr_t)out->ptr, out->bytes, (uintptr_t)x->ptr, x->bytes,
                  (uintptr_t)model_map, model_size, weight_offset) ||
        !(eps > 0.0f && eps <= 3.402823466e38F) ||
        !rocm_hc_norm_mix_device_supported()) return 0;
    const char *wptr = cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "HC norm/mix f16");
    if (!range_valid((uintptr_t)wptr, weight_bytes) ||
        ((uintptr_t)wptr & 1u) != 0u ||
        overlaps((uintptr_t)out->ptr, out_bytes, (uintptr_t)wptr, weight_bytes)) return 0;
    if (g_cuda_tmp &&
        (overlaps((uintptr_t)g_cuda_tmp, g_cuda_tmp_bytes, (uintptr_t)out->ptr, out_bytes) ||
         overlaps((uintptr_t)g_cuda_tmp, g_cuda_tmp_bytes, (uintptr_t)x->ptr, x_bytes) ||
         overlaps((uintptr_t)g_cuda_tmp, g_cuda_tmp_bytes, (uintptr_t)wptr, weight_bytes))) return 0;
    float *scale = (float *)cuda_tmp_alloc(sizeof(float), "HC norm/mix scale");
    if (!range_valid((uintptr_t)scale, sizeof(float)) ||
        overlaps((uintptr_t)scale, sizeof(float), (uintptr_t)out->ptr, out_bytes) ||
        overlaps((uintptr_t)scale, sizeof(float), (uintptr_t)x->ptr, x_bytes) ||
        overlaps((uintptr_t)scale, sizeof(float), (uintptr_t)wptr, weight_bytes)) return 0;

    // Retain two launches and the original ordered F32 dot. A single fused
    // launch would duplicate the 16384-element RMS per row or constrain the
    // 24 projection rows to too few workgroups. Neither is enabled unmeasured.
    // The common scratch is consumed on the default stream before reuse.
    ds4_hc_rms_scale_kernel<<<1u, 256u>>>(scale, (const float *)x->ptr, n, eps);
    if (!cuda_ok(cudaGetLastError(), "HC norm/mix scale launch")) return -1;
    ds4_hc_f16_project_scaled_ordered_kernel<<<out_dim, 32u>>>(
            (float *)out->ptr, (const __half *)wptr, (const float *)x->ptr,
            scale, n, out_dim);
    return cuda_ok(cudaGetLastError(), "HC norm/mix ordered launch") ? 1 : -1;
}

extern "C" int ds4_gpu_tensor_read_after_selected_event(
        const ds4_gpu_tensor *tensor,
        uint64_t offset,
        void *data,
        uint64_t bytes,
        uint64_t event_value,
        const char *label) {
    if (!tensor || !data || offset > tensor->bytes ||
        bytes > tensor->bytes - offset ||
        event_value == 0 ||
        !g_selected_readback_event) {
        return 0;
    }
    if (!g_selected_readback_stream) {
        cudaError_t err =
            cudaStreamCreateWithFlags(&g_selected_readback_stream,
                                      cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "selected readback stream creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
#ifdef __HIP_PLATFORM_AMD__
    cudaError_t err = hipStreamWaitEvent(g_selected_readback_stream,
                                         g_selected_readback_event,
                                         0);
#else
    cudaError_t err = cudaStreamWaitEvent(g_selected_readback_stream,
                                          g_selected_readback_event,
                                          0);
#endif
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback stream wait failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemcpyAsync(data,
                          (const char *)tensor->ptr + offset,
                          (size_t)bytes,
                          cudaMemcpyDeviceToHost,
                          g_selected_readback_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback copy failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaStreamSynchronize(g_selected_readback_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback sync failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    int ok = ds4_gpu_set_model_fd(fd);
    g_model_fd_host_base = model_map ? model_map : g_model_host_base;
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_f32_to_f16(
        ds4_gpu_tensor *dst,
        uint64_t dst_offset,
        const ds4_gpu_tensor *src,
        uint64_t src_offset,
        uint64_t count) {
    if (!dst || !src || !dst->ptr || !src->ptr) return 0;
    if ((dst_offset % sizeof(__half)) != 0 || (src_offset % sizeof(float)) != 0) return 0;
    if (dst_offset > dst->bytes || src_offset > src->bytes) return 0;
    if (count > (UINT64_MAX / sizeof(__half)) || count > (UINT64_MAX / sizeof(float))) return 0;
    uint64_t dst_bytes = count * sizeof(__half);
    uint64_t src_bytes = count * sizeof(float);
    if (dst_bytes > dst->bytes - dst_offset || src_bytes > src->bytes - src_offset) return 0;
    if (count == 0) return 1;
    f32_to_f16_kernel<<<(count + 255u) / 256u, 256>>>(
            (__half *)((char *)dst->ptr + dst_offset),
            (const float *)((const char *)src->ptr + src_offset),
            count);
    return cuda_ok(cudaGetLastError(), "tensor copy f32 to f16 launch");
}

extern "C" int ds4_gpu_pro_q4_expert_table_auto_available(void) {
    return 0;
}

extern "C" int ds4_gpu_preload_q4_expert_tables(
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t n_total_expert) {
    (void)model_map;
    (void)model_size;
    (void)gate_offset;
    (void)up_offset;
    (void)down_offset;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    (void)n_total_expert;
    return 0;
}

extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    if (enabled && !g_ssd_streaming_mode) {
        (void)ds4_gpu_release_q4_attn_q_b_f16_sidecars();
    }
    g_ssd_streaming_mode = enabled ? 1 : 0;
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_routed_moe_selected_override_n = 0;
    g_stream_selected_cache.loaded = 0;
    g_stream_batch_selected_cache.loaded = 0;
}

extern "C" void ds4_gpu_set_glm_model(bool enabled) {
    g_glm_model = enabled ? 1 : 0;
}

extern "C" void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    g_stream_expert_cache_budget = experts;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    (void)bytes;
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    size_t free_b = 0;
    size_t total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    (void)free_b;
    return (uint64_t)total_b;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    return g_ssd_streaming_mode ? g_stream_expert_cache_budget : 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return (uint32_t)g_stream_resident_experts.size();
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
}

extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    cuda_stream_resident_cache_release();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return ds4_gpu_stream_expert_cache_configured_count();
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!table) return 0;
    if (!cuda_stream_selected_load(table->model_map,
                                   table->model_size,
                                   table->layer,
                                   selected_ids,
                                   table->n_total_expert,
                                   n_selected,
                                   table->gate_offset,
                                   table->up_offset,
                                   table->down_offset,
                                   table->gate_expert_bytes,
                                   table->down_expert_bytes)) {
        return 0;
    }
    return cuda_stream_selected_finish_pending_missing(0);
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!table) return 0;
    return cuda_stream_selected_load(table->model_map,
                                     table->model_size,
                                     table->layer,
                                     selected_ids,
                                     table->n_total_expert,
                                     n_selected,
                                     table->gate_offset,
                                     table->up_offset,
                                     table->down_offset,
                                     table->gate_expert_bytes,
                                     table->down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected) {
    if (!table) return 0;
    const ds4_gpu_tensor *selected_exec = NULL;
    const char **gate_ptrs = NULL;
    const char **up_ptrs = NULL;
    const char **down_ptrs = NULL;
    uint32_t unique = 0;
    return cuda_stream_batch_selected_prepare_from_host(table->model_map,
                                                        table->model_size,
                                                        table->layer,
                                                        selected_ids,
                                                        n_tokens,
                                                        table->n_total_expert,
                                                        n_selected,
                                                        table->gate_offset,
                                                        table->up_offset,
                                                        table->down_offset,
                                                        table->gate_expert_bytes,
                                                        table->down_expert_bytes,
                                                        &selected_exec,
                                                        &gate_ptrs,
                                                        &up_ptrs,
                                                        &down_ptrs,
                                                        &unique,
                                                        1);
}

extern "C" int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table) {
    if (!table) return 0;
    return cuda_stream_layer_expert_cache_load(table->model_map,
                                               table->model_size,
                                               table->layer,
                                               table->n_total_expert,
                                               table->gate_offset,
                                               table->up_offset,
                                               table->down_offset,
                                               table->gate_expert_bytes,
                                               table->down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor             *selected,
        uint32_t                          n_tokens,
        uint32_t                          n_seed_tokens,
        uint32_t                          n_selected) {
    if (!table) return 0;
    return cuda_stream_layer_expert_cache_seed_selected(table->model_map,
                                                        table->model_size,
                                                        table->layer,
                                                        selected,
                                                        n_tokens,
                                                        n_seed_tokens,
                                                        table->n_total_expert,
                                                        n_selected,
                                                        table->gate_offset,
                                                        table->up_offset,
                                                        table->down_offset,
                                                        table->gate_expert_bytes,
                                                        table->down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_finish_pending_batch(void) {
    return cuda_stream_batch_selected_finish_pending_missing();
}

extern "C" int ds4_gpu_stream_expert_cache_release_layer_cache(void) {
    cuda_stream_layer_expert_cache_release();
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    if (!table) return 0;
    return cuda_stream_resident_seed_experts(table->model_map,
                                             table->model_size,
                                             table->layer,
                                             expert_ids,
                                             expert_priorities,
                                             n_experts,
                                             table->n_total_expert,
                                             table->gate_offset,
                                             table->up_offset,
                                             table->down_offset,
                                             table->gate_expert_bytes,
                                             table->down_expert_bytes);
}

extern "C" int ds4_gpu_routed_moe_set_selected_override(
        const int32_t *selected,
        uint32_t n_selected) {
    if (n_selected > DS4_ROCM_N_EXPERT_USED || (!selected && n_selected != 0)) return 0;
    for (uint32_t i = 0; i < n_selected; i++) {
        g_routed_moe_selected_override[i] = selected[i];
    }
    g_routed_moe_selected_override_n = n_selected;
    return 1;
}
