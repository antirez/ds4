int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!model_map || model_size == 0) return 0;
    if (map_offset > model_size || map_size == 0 || map_size > model_size - map_offset) return 0;
    max_tensor_bytes = ds4_gpu_effective_model_max_tensor_bytes(map_size, max_tensor_bytes);

    @autoreleasepool {
        if (g_model_map_ptr == model_map &&
            g_model_map_size == model_size &&
            g_model_mapped_offset == map_offset &&
            g_model_mapped_size == map_size &&
            g_model_mapped_max_tensor_bytes == max_tensor_bytes) {
            return 1;
        }

        for (uint32_t i = 0; i < g_model_view_count; i++) {
            if (g_model_views[i].model_map == model_map &&
                g_model_views[i].model_size == model_size &&
                map_offset >= g_model_views[i].model_offset &&
                map_offset + map_size <= g_model_views[i].model_offset + g_model_views[i].bytes) {
                return 1;
            }
        }

        ds4_gpu_model_residency_clear();
        if (!ds4_gpu_map_model_views(model_map, model_size, map_offset, map_size, max_tensor_bytes)) {
            ds4_gpu_model_residency_clear();
            return 0;
        }
        g_model_map_ptr = model_map;
        g_model_map_size = model_size;
        g_model_mapped_offset = map_offset;
        g_model_mapped_size = map_size;
        g_model_mapped_max_tensor_bytes = max_tensor_bytes;
        if (ds4_gpu_model_map_log_enabled()) {
            fprintf(stderr,
                    "ds4: Metal mapped mmaped model as %u overlapping shared buffers\n",
                    g_model_view_count);
        }
        return 1;
    }
}

static int ds4_gpu_model_views_cover_spans(
        const void     *model_map,
        uint64_t        model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t        count);

int ds4_gpu_set_model_map_spans(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    if (count == 1) {
        return ds4_gpu_set_model_map_range(model_map,
                                           model_size,
                                           offsets[0],
                                           sizes[0],
                                           max_tensor_bytes);
    }
    if (ds4_gpu_model_views_cover_spans(model_map, model_size, offsets, sizes, count)) {
        return 1;
    }

    @autoreleasepool {
        const double t0 = ds4_gpu_now_ms();
        max_tensor_bytes = ds4_gpu_effective_model_max_tensor_bytes(model_size, max_tensor_bytes);

        ds4_gpu_model_residency_clear();
        ds4_gpu_model_views_clear();

        uint64_t mapped_total = 0;
        uint64_t first_offset = UINT64_MAX;
        for (uint32_t i = 0; i < count; i++) {
            if (offsets[i] > model_size || sizes[i] == 0 || sizes[i] > model_size - offsets[i]) {
                fprintf(stderr, "ds4: Metal model span %u is outside the GGUF mapping\n", i);
                ds4_gpu_model_residency_clear();
                ds4_gpu_model_views_clear();
                return 0;
            }
            if (offsets[i] < first_offset) first_offset = offsets[i];
            uint64_t effective_max = max_tensor_bytes;
            if (effective_max > sizes[i]) effective_max = sizes[i];
            if (!ds4_gpu_add_model_view_range(model_map,
                                              model_size,
                                              offsets[i],
                                              sizes[i],
                                              effective_max,
                                              true,
                                              &mapped_total)) {
                ds4_gpu_model_residency_clear();
                ds4_gpu_model_views_clear();
                return 0;
            }
        }
        if (!ds4_gpu_finish_model_views(t0, mapped_total, first_offset)) {
            ds4_gpu_model_residency_clear();
            ds4_gpu_model_views_clear();
            return 0;
        }
        g_model_map_ptr = model_map;
        g_model_map_size = model_size;
        g_model_mapped_offset = first_offset == UINT64_MAX ? 0 : first_offset;
        g_model_mapped_size = mapped_total;
        g_model_mapped_max_tensor_bytes = max_tensor_bytes;
        if (ds4_gpu_model_map_log_enabled()) {
            fprintf(stderr,
                    "ds4: Metal mapped mmaped model as %u disjoint shared buffers across %u tensor spans\n",
                    g_model_view_count,
                    count);
        }
        return 1;
    }
}

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    return ds4_gpu_set_model_map_range(model_map, model_size, 0, model_size, 0);
}

int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    return 1;
}

static int ds4_gpu_model_views_cover_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    size) {
    if (!model_map || model_size == 0 || size == 0 ||
        offset > model_size || size > model_size - offset) {
        return 0;
    }
    const uint64_t end = offset + size;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (g_model_views[i].model_map != model_map ||
            g_model_views[i].model_size != model_size) {
            continue;
        }
        const uint64_t view_start = g_model_views[i].model_offset;
        const uint64_t view_end = view_start + g_model_views[i].bytes;
        if (offset >= view_start && end <= view_end) return 1;
    }
    return 0;
}

static int ds4_gpu_model_views_cover_spans(
        const void     *model_map,
        uint64_t        model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t        count) {
    if (!offsets || !sizes || count == 0) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!ds4_gpu_model_views_cover_range(model_map,
                                             model_size,
                                             offsets[i],
                                             sizes[i])) {
            return 0;
        }
    }
    return 1;
}

int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    (void)fd;
    (void)model_map;
    return 1;
}

static id<MTLBuffer> ds4_gpu_wrap_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    (void)model_map;
    if (model_size == 0 || offset > model_size || len > model_size - offset) {
        fprintf(stderr, "ds4: Metal model range is outside the mapped model\n");
        return nil;
    }

    const uint64_t end = offset + len;
    for (uint32_t i = 0; i < g_model_view_count; i++) {
        if (g_model_views[i].model_map != model_map ||
            g_model_views[i].model_size != model_size) {
            continue;
        }
        const uint64_t view_start = g_model_views[i].model_offset;
        const uint64_t view_end = view_start + g_model_views[i].bytes;
        if (offset >= view_start && end <= view_end) {
            *inner_offset = offset - view_start;
            return g_model_views[i].buffer;
        }
    }

    fprintf(stderr,
            "ds4: Metal model range %.2f..%.2f GiB is not covered by mapped model views\n",
            ds4_gpu_gib(offset),
            ds4_gpu_gib(end));
    return nil;
}

typedef enum {
    DS4_GPU_EXACT_VIEW_CACHED,
    DS4_GPU_EXACT_VIEW_TRANSIENT,
    DS4_GPU_EXACT_VIEW_OWNED,
} ds4_gpu_exact_view_lifetime;

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range_impl(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset,
        ds4_gpu_exact_view_lifetime lifetime) {
    const bool cache_view = lifetime == DS4_GPU_EXACT_VIEW_CACHED;
    const bool transient_view = lifetime == DS4_GPU_EXACT_VIEW_TRANSIENT;
    if (!model_map || !g_device ||
        (cache_view && !g_model_buffer_cache) ||
        (transient_view && !g_transient_buffers) ||
        model_size == 0 || offset > model_size || len > model_size - offset) {
        fprintf(stderr, "ds4: Metal exact model range is outside the mapped model\n");
        return nil;
    }

    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t page_offset = offset & ~(page - 1);
    const uint64_t leading = offset - page_offset;
    if (len > UINT64_MAX - leading ||
        leading + len > UINT64_MAX - (page - 1)) {
        fprintf(stderr, "ds4: Metal exact model range overflows page alignment\n");
        return nil;
    }
    uint64_t view_bytes = round_up_u64(leading + len, page);
    if (view_bytes > model_size - page_offset) view_bytes = model_size - page_offset;
    if (leading + len > view_bytes) {
        fprintf(stderr, "ds4: Metal exact model range alignment exceeds mapped model\n");
        return nil;
    }
    if (view_bytes > (uint64_t)[g_device maxBufferLength]) {
        fprintf(stderr,
                "ds4: Metal exact model range %.2f GiB exceeds maxBufferLength %.2f GiB\n",
                ds4_gpu_gib(view_bytes),
                ds4_gpu_gib((uint64_t)[g_device maxBufferLength]));
        return nil;
    }

    NSString *key = nil;
    id<MTLBuffer> buffer = nil;
    if (cache_view) {
        key = [NSString stringWithFormat:@"%p:%llu:%llu:%llu",
               model_map,
               (unsigned long long)model_size,
               (unsigned long long)page_offset,
               (unsigned long long)view_bytes];
        buffer = [g_model_buffer_cache objectForKey:key];
    }
    if (!buffer) {
        const uintptr_t base = (uintptr_t)model_map;
        buffer = [g_device newBufferWithBytesNoCopy:(void *)(base + page_offset)
                                             length:(NSUInteger)view_bytes
                                            options:ds4_gpu_model_resource_options()
                                        deallocator:nil];
        if (!buffer) {
            fprintf(stderr,
                    "ds4: Metal could not wrap exact mmaped model range at %.2f GiB, size %.2f MiB\n",
                    ds4_gpu_gib(page_offset),
                    ds4_gpu_mib(view_bytes));
            return nil;
        }
        if (cache_view) {
            buffer.label = @"ds4_model_exact_view";
        } else if (transient_view) {
            buffer.label = @"ds4_model_exact_transient_view";
        } else {
            buffer.label = @"ds4_model_exact_owned_view";
        }
        if (cache_view) {
            [g_model_buffer_cache setObject:buffer forKey:key];
            ds4_gpu_model_buffer_cache_note_insert(view_bytes);
        } else if (transient_view) {
            [g_transient_buffers addObject:buffer];
        }
    }

    if (inner_offset) *inner_offset = leading;
    return buffer;
}

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    return ds4_gpu_wrap_model_exact_range_impl(model_map,
                                               model_size,
                                               offset,
                                               len,
                                               inner_offset,
                                               DS4_GPU_EXACT_VIEW_CACHED);
}

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range_transient(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    return ds4_gpu_wrap_model_exact_range_impl(model_map,
                                               model_size,
                                               offset,
                                               len,
                                               inner_offset,
                                               DS4_GPU_EXACT_VIEW_TRANSIENT);
}

static id<MTLBuffer> ds4_gpu_wrap_model_exact_range_owned(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t   *inner_offset) {
    return ds4_gpu_wrap_model_exact_range_impl(model_map,
                                               model_size,
                                               offset,
                                               len,
                                               inner_offset,
                                               DS4_GPU_EXACT_VIEW_OWNED);
}

static id<MTLBuffer> ds4_gpu_wrap_q8_decode_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t    n_tokens,
        uint64_t   *inner_offset) {
    const uint64_t exact_decode_max_mib =
        ds4_gpu_env_u64("DS4_METAL_Q8_DECODE_EXACT_VIEW_MAX_MIB",
                        1024u,
                        1u,
                        4096u);
    const uint64_t exact_decode_max_bytes =
        exact_decode_max_mib * 1024ull * 1024ull;
    const bool exact_decode_weight_view =
        n_tokens == 1u &&
        len <= exact_decode_max_bytes &&
        getenv("DS4_METAL_ENABLE_Q8_DECODE_EXACT_VIEWS") != NULL &&
        getenv("DS4_METAL_DISABLE_Q8_DECODE_EXACT_VIEWS") == NULL;
    return exact_decode_weight_view ?
        ds4_gpu_wrap_model_exact_range(model_map,
                                       model_size,
                                       offset,
                                       len,
                                       inner_offset) :
        ds4_gpu_wrap_model_range(model_map,
                                 model_size,
                                 offset,
                                 len,
                                 inner_offset);
}

static id<MTLBuffer> ds4_gpu_wrap_f32_decode_model_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len,
        uint64_t    n_tokens,
        uint64_t   *inner_offset) {
    const uint64_t exact_decode_max_mib =
        ds4_gpu_env_u64("DS4_METAL_F32_DECODE_EXACT_VIEW_MAX_MIB",
                        64u,
                        1u,
                        4096u);
    const uint64_t exact_decode_max_bytes =
        exact_decode_max_mib * 1024ull * 1024ull;
    const bool exact_decode_weight_view =
        n_tokens == 1u &&
        len <= exact_decode_max_bytes &&
        getenv("DS4_METAL_ENABLE_F32_DECODE_EXACT_VIEWS") != NULL &&
        getenv("DS4_METAL_DISABLE_F32_DECODE_EXACT_VIEWS") == NULL;
    return exact_decode_weight_view ?
        ds4_gpu_wrap_model_exact_range(model_map,
                                       model_size,
                                       offset,
                                       len,
                                       inner_offset) :
        ds4_gpu_wrap_model_range(model_map,
                                 model_size,
                                 offset,
                                 len,
                                 inner_offset);
}
