extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (!event_value) return 0;
    *event_value = 0;
    if (!g_selected_readback_event) {
        cudaError_t err =
            cudaEventCreateWithFlags(&g_selected_readback_event,
                                     cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "selected readback event creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    cudaError_t err = cudaEventRecord(g_selected_readback_event, 0);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback event record failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *event_value = ++g_selected_readback_event_value;
    return 1;
}

extern "C" int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label) {
    if (event_value == 0 || !g_selected_readback_event) return 0;
    cudaError_t err = cudaEventSynchronize(g_selected_readback_event);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "selected readback wait failed for %s: %s\n",
                label ? label : "selected-id readback",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label) {
    return ds4_gpu_commit_and_wait_selected_readback(event_value, label);
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
    g_ssd_streaming_mode = enabled ? 1 : 0;
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_routed_moe_selected_override_n = 0;
    g_stream_selected_cache.loaded = 0;
    g_stream_batch_selected_cache.loaded = 0;
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

extern "C" int ds4_gpu_memory_info(uint64_t *free_bytes, uint64_t *total_bytes) {
    size_t free_b = 0;
    size_t total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    if (free_bytes) *free_bytes = (uint64_t)free_b;
    if (total_bytes) *total_bytes = (uint64_t)total_b;
    return 1;
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

extern "C" int ds4_gpu_stream_expert_cache_release_layer_cache(void) {
    cuda_stream_layer_expert_cache_release();
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    (void)table;
    (void)expert_ids;
    (void)expert_priorities;
    (void)n_experts;
    return 1;
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
