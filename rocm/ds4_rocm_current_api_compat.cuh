extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (event_value) *event_value = 0;
    return 1;
}

extern "C" int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label) {
    (void)event_value;
    (void)label;
    return ds4_gpu_end_commands();
}

extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label) {
    (void)event_value;
    (void)label;
    return ds4_gpu_synchronize();
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
    (void)enabled;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    (void)experts;
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    return 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    return 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return 0;
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return 0;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    (void)model_map;
    (void)model_size;
    (void)layer;
    (void)selected_ids;
    (void)n_total_expert;
    (void)n_selected;
    (void)gate_offset;
    (void)up_offset;
    (void)down_offset;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return 0;
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    return ds4_gpu_stream_expert_cache_seed_selected(
            model_map, model_size, layer, selected_ids, n_total_expert,
            n_selected, gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes);
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const int32_t *expert_ids,
        const uint32_t *expert_priorities,
        uint32_t n_experts,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    (void)model_map;
    (void)model_size;
    (void)layer;
    (void)expert_ids;
    (void)expert_priorities;
    (void)n_experts;
    (void)n_total_expert;
    (void)gate_offset;
    (void)up_offset;
    (void)down_offset;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return 0;
}

extern "C" int ds4_gpu_routed_moe_set_selected_override(
        const int32_t *selected,
        uint32_t n_selected) {
    (void)selected;
    (void)n_selected;
    return 0;
}
