/* --gpu-vram auto probe. Defined here (in the .cu unit) so the
 * C-side parser (ds4_gpu_args.c) does not need to include
 * <cuda_runtime.h>. Returns 0 on success, nonzero on error
 * (errbuf populated). See ds4_gpu_args.h.
 *
 * Side-effect-light: changes cudaSetDevice during probing; callers
 * that care about the active device should reset it themselves
 * before continuing. (The mgpu init path resets it anyway.) */
extern "C" int ds4_gpu_args_probe_auto_cuda(const int      *device_filter,
                                              int             filter_len,
                                              ds4_gpu_config *out,
                                              size_t          safety_margin_bytes,
                                              char           *errbuf,
                                              size_t          errbuflen) {
    if (!out) {
        if (errbuf && errbuflen) snprintf(errbuf, errbuflen, "internal: NULL out");
        return 1;
    }
    int visible = 0;
    cudaError_t rc = cudaGetDeviceCount(&visible);
    if (rc != cudaSuccess || visible <= 0) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen,
                     "cudaGetDeviceCount failed: %s",
                     rc == cudaSuccess ? "no devices" : cudaGetErrorString(rc));
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
    out->n_gpus = n_dev;
    out->safety_margin_bytes = safety_margin_bytes;
    for (int i = 0; i < n_dev; i++) {
        int d = devs[i];
        rc = cudaSetDevice(d);
        if (rc != cudaSuccess) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen,
                         "cudaSetDevice(%d) failed: %s",
                         d, cudaGetErrorString(rc));
            }
            return 1;
        }
        size_t free_b = 0, total_b = 0;
        rc = cudaMemGetInfo(&free_b, &total_b);
        if (rc != cudaSuccess) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen,
                         "cudaMemGetInfo on device %d failed: %s",
                         d, cudaGetErrorString(rc));
            }
            return 1;
        }
        /* Auto-mode reserve. Auto-probe is the only place we override
         * the user's stated budget, so this is where the conservative-
         * on-the-user's-behalf reserve belongs. The
         * engine path (engine_classify_multi_tier) still subtracts the
         * user-supplied safety_margin_bytes + the cuBLAS workspace from
         * whatever budget we hand back; that math is unchanged and
         * applies on top of the reserve we trim here.
         *
         * Reserve = max(2 GiB, 5 % of free). Why these numbers:
         *   - 2 GiB floor covers runtime scratch / Q8 dequant caches /
         *     MTP optional state on small GPUs (8-12 GB cards) where
         *     5 % is < 1 GiB and not enough headroom.
         *   - 5 % of free scales the reserve up on larger cards where
         *     workspace + KV growth needs proportionally more room.
         * Explicit --gpu-vram 47,37 budgets do not go through this
         * probe and are unaffected. */
        const size_t reserve_floor = (size_t)2ull * 1024ull * 1024ull * 1024ull;
        const size_t reserve_pct = free_b / 20u;
        const size_t reserve = reserve_floor > reserve_pct ? reserve_floor : reserve_pct;
        const size_t budget = free_b > reserve ? (free_b - reserve) : 0;
        (void)safety_margin_bytes;
        out->device_indices[i] = d;
        out->vram_bytes[i] = budget;
    }
    return 0;
}

typedef struct ds4_gpu_stream_expert_table {
    const void *model_map;
    uint64_t    model_size;
    uint32_t    layer;
    uint32_t    n_total_expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
} ds4_gpu_stream_expert_table;

static int cuda_stream_selected_ensure_bytes(
        char **ptr, uint64_t *capacity, uint64_t bytes, const char *label) {
    if (*ptr && *capacity >= bytes) return 1;
    if (*ptr) {
        (void)cudaFree(*ptr);
        *ptr = NULL;
        *capacity = 0;
    }
    if (bytes == 0 || bytes > (uint64_t)SIZE_MAX) return 0;
    cudaError_t err = cudaMalloc((void **)ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA streaming %s allocation failed for %.2f MiB: %s\n",
                label, (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *capacity = bytes;
    return 1;
}

static int cuda_stream_selected_ensure_i32(uint64_t count) {
    if (count == 0 || count > UINT64_MAX / sizeof(int32_t)) return 0;
    const uint64_t bytes = count * sizeof(int32_t);
    return cuda_stream_selected_ensure_bytes(
            (char **)&g_stream_selected_cache.slot_selected_ptr,
            &g_stream_selected_cache.slot_selected_capacity,
            bytes,
            "selected-id remap");
}

static int cuda_stream_selected_ranges_valid(
        const ds4_gpu_stream_expert_table *table) {
    if (!table || !table->model_map || table->model_size == 0 ||
        table->n_total_expert == 0 || table->gate_expert_bytes == 0 ||
        table->down_expert_bytes == 0) {
        return 0;
    }
    if ((uint64_t)table->n_total_expert >
            UINT64_MAX / table->gate_expert_bytes ||
        (uint64_t)table->n_total_expert >
            UINT64_MAX / table->down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_bytes =
        (uint64_t)table->n_total_expert * table->gate_expert_bytes;
    const uint64_t down_bytes =
        (uint64_t)table->n_total_expert * table->down_expert_bytes;
    return table->gate_offset <= table->model_size &&
           gate_bytes <= table->model_size - table->gate_offset &&
           table->up_offset <= table->model_size &&
           gate_bytes <= table->model_size - table->up_offset &&
           table->down_offset <= table->model_size &&
           down_bytes <= table->model_size - table->down_offset;
}

static int cuda_stream_selected_cache_begin_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *selected_ids,
        uint32_t slot_count) {
    cuda_stream_selected_cache_invalidate();
    if (!g_ssd_streaming_mode) return 1;
    if (!cuda_stream_selected_ranges_valid(table) || !selected_ids ||
        slot_count == 0) {
        return 0;
    }
    if (g_n_gpus != 1) {
        fprintf(stderr,
                "ds4: CUDA SSD streaming requires single-GPU placement\n");
        return 0;
    }

    std::vector<int32_t> expert_to_slot;
    std::vector<int32_t> compact_ids;
    std::vector<int32_t> slot_ids;
    try {
        expert_to_slot.assign(table->n_total_expert, -1);
        compact_ids.reserve(slot_count < table->n_total_expert ?
                            slot_count : table->n_total_expert);
        slot_ids.resize(slot_count);
    } catch (...) {
        return 0;
    }
    for (uint32_t i = 0; i < slot_count; i++) {
        const int32_t expert = selected_ids[i];
        if (expert < 0 || (uint32_t)expert >= table->n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming expert id %d is outside 0..%u at layer %u\n",
                    expert, table->n_total_expert, table->layer);
            return 0;
        }
        int32_t compact = expert_to_slot[(uint32_t)expert];
        if (compact < 0) {
            compact = (int32_t)compact_ids.size();
            expert_to_slot[(uint32_t)expert] = compact;
            compact_ids.push_back(expert);
        }
        slot_ids[i] = compact;
    }
    if (compact_ids.empty() || compact_ids.size() > UINT32_MAX) return 0;
    const uint64_t compact_count = compact_ids.size();
    if (compact_count > UINT64_MAX / table->gate_expert_bytes ||
        compact_count > UINT64_MAX / table->down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_bytes = compact_count * table->gate_expert_bytes;
    const uint64_t down_bytes = compact_count * table->down_expert_bytes;
    const int logical_tier = 0;
    if (g_stream_selected_cache.logical_tier != logical_tier &&
        (g_stream_selected_cache.gate_ptr ||
         g_stream_selected_cache.up_ptr ||
         g_stream_selected_cache.down_ptr ||
         g_stream_selected_cache.slot_selected_ptr)) {
        cuda_stream_selected_cache_release();
    }
    if (ds4_gpu_set_current_device(logical_tier) != 0 ||
        !cuda_stream_selected_ensure_bytes(
                &g_stream_selected_cache.gate_ptr,
                &g_stream_selected_cache.gate_capacity,
                gate_bytes, "gate experts") ||
        !cuda_stream_selected_ensure_bytes(
                &g_stream_selected_cache.up_ptr,
                &g_stream_selected_cache.up_capacity,
                gate_bytes, "up experts") ||
        !cuda_stream_selected_ensure_bytes(
                &g_stream_selected_cache.down_ptr,
                &g_stream_selected_cache.down_capacity,
                down_bytes, "down experts") ||
        !cuda_stream_selected_ensure_i32(slot_count)) {
        cuda_stream_selected_cache_invalidate();
        return 0;
    }

    for (uint32_t i = 0; i < compact_ids.size(); i++) {
        const uint64_t expert = (uint32_t)compact_ids[i];
        const uint64_t gate_src =
            table->gate_offset + expert * table->gate_expert_bytes;
        const uint64_t up_src =
            table->up_offset + expert * table->gate_expert_bytes;
        const uint64_t down_src =
            table->down_offset + expert * table->down_expert_bytes;
        const uint64_t gate_dst = (uint64_t)i * table->gate_expert_bytes;
        const uint64_t down_dst = (uint64_t)i * table->down_expert_bytes;
        if (!cuda_model_copy_to_device_streamed(
                    g_stream_selected_cache.gate_ptr + gate_dst,
                    table->model_map, table->model_size,
                    gate_src, table->gate_expert_bytes,
                    "stream gate expert copy") ||
            !cuda_model_copy_to_device_streamed(
                    g_stream_selected_cache.up_ptr + gate_dst,
                    table->model_map, table->model_size,
                    up_src, table->gate_expert_bytes,
                    "stream up expert copy") ||
            !cuda_model_copy_to_device_streamed(
                    g_stream_selected_cache.down_ptr + down_dst,
                    table->model_map, table->model_size,
                    down_src, table->down_expert_bytes,
                    "stream down expert copy")) {
            cuda_stream_selected_cache_invalidate();
            return 0;
        }
    }
    if (!cuda_ok(cudaMemcpy(g_stream_selected_cache.slot_selected_ptr,
                            slot_ids.data(),
                            (size_t)slot_count * sizeof(int32_t),
                            cudaMemcpyHostToDevice),
                 "stream selected-id remap copy")) {
        cuda_stream_selected_cache_invalidate();
        return 0;
    }

    g_stream_selected_cache.logical_tier = logical_tier;
    g_stream_selected_cache.model_map = table->model_map;
    g_stream_selected_cache.layer = table->layer;
    g_stream_selected_cache.n_total_expert = table->n_total_expert;
    g_stream_selected_cache.slot_count = slot_count;
    g_stream_selected_cache.compact_count = (uint32_t)compact_count;
    g_stream_selected_cache.gate_offset = table->gate_offset;
    g_stream_selected_cache.up_offset = table->up_offset;
    g_stream_selected_cache.down_offset = table->down_offset;
    g_stream_selected_cache.gate_expert_bytes = table->gate_expert_bytes;
    g_stream_selected_cache.down_expert_bytes = table->down_expert_bytes;
    g_stream_selected_cache.slot_selected_tensor.ptr =
        g_stream_selected_cache.slot_selected_ptr;
    g_stream_selected_cache.slot_selected_tensor.bytes =
        (uint64_t)slot_count * sizeof(int32_t);
    g_stream_selected_cache.slot_selected_tensor.owner = 0;
    g_stream_selected_cache.slot_selected_tensor.device_id = logical_tier;
    g_stream_selected_cache.valid = 1;
    return 1;
}
