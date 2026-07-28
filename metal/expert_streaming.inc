uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    if (budget > DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES) {
        budget = DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES;
    }
    return budget;
}

uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return g_stream_expert_cache_entry_count;
}

uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                      down_expert_bytes)) {
        return 0;
    }
    return ds4_gpu_stream_expert_cache_configured_budget();
}

static int ds4_gpu_stream_expert_cache_note_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    if (gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        fprintf(stderr, "ds4: Metal streaming expert cache byte size overflow\n");
        return 0;
    }
    /*
     * The cache is a single-size-class slab allocator: the expert byte size is
     * frozen on first sight (or pre-seeded at startup from the model's slab
     * class) and off-size layers are REJECTED rather than adopted. A rejected
     * layer (mixed-precision boost: Q4_K experts among IQ2 layers) falls back
     * to the mapped-model per-expert path; last-writer-wins here would instead
     * poison the slab size class and deadlock slab reuse.
     */
    const uint64_t bytes = gate_expert_bytes * 2ull + down_expert_bytes;
    if (g_stream_expert_cache_expert_bytes == 0) {
        g_stream_expert_cache_expert_bytes = bytes;
        return 1;
    }
    return bytes == g_stream_expert_cache_expert_bytes;
}

static uint32_t ds4_gpu_stream_expert_cache_requested_budget(void) {
    if (!g_ssd_streaming_mode) return 0;
    if (g_stream_expert_cache_budget_override != 0) {
        return g_stream_expert_cache_budget_override;
    }
    return 0;
}

static uint32_t ds4_gpu_stream_expert_cache_configured_budget(void) {
    return ds4_gpu_stream_expert_cache_requested_budget();
}

static uint32_t ds4_gpu_stream_expert_cache_effective_cap(
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected) {
    (void)layer;
    if (ds4_gpu_stream_expert_cache_configured_budget() == 0) return 0;

    /*
     * The residency policy is global: every layer can use any expert slot it
     * routes to, and global pruning decides which existing entry is least
     * valuable.  A per-layer cap made cache size depend on model depth rather
     * than the actual byte budget.
     */
    uint32_t cap = n_total_expert;
    if (cap < n_selected) cap = n_selected;
    if (cap > DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) {
        cap = DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
    }
    return cap;
}

static int ds4_gpu_stream_expert_timing_summary_enabled(void) {
    static int checked = 0;
    static int enabled = 0;
    if (!checked) {
        enabled =
            (getenv("DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY") != NULL ||
             getenv("DS4_METAL_STREAMING_EXPERT_PROFILE_SUMMARY") != NULL) &&
            getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_TIMING_SUMMARY") == NULL;
        checked = 1;
    }
    return enabled;
}

static uint32_t ds4_gpu_stream_expert_popcount(uint32_t mask) {
    return (uint32_t)__builtin_popcount(mask);
}

static int ds4_gpu_stream_expert_split_worthwhile(
        uint32_t resident_mask,
        uint32_t missing_mask) {
    if (resident_mask == 0 || missing_mask == 0) return 0;
    /*
     * The split path pays an extra command stage and a second routed-expert
     * bind.  It is worthwhile when several experts are missing and their SSD
     * reads can be hidden by resident expert work.  With one or two misses,
     * especially in large caches, a single unsplit routed pass is faster.
     */
    return ds4_gpu_stream_expert_popcount(missing_mask) >= 3u;
}

static void ds4_gpu_stream_expert_timing_note_selected(
        double sync_ms,
        double copy_ms,
        double bind_ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_selected_calls++;
    g_stream_expert_timing_selected_sync_ms += sync_ms;
    g_stream_expert_timing_selected_copy_ms += copy_ms;
    g_stream_expert_timing_selected_read_ms += sync_ms + copy_ms;
    g_stream_expert_timing_selected_bind_ms += bind_ms;
}

static void ds4_gpu_stream_expert_timing_note_split(
        uint32_t resident_mask,
        uint32_t missing_mask,
        double   resident_ms,
        double   missing_ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_split_layers++;
    g_stream_expert_timing_split_resident_experts +=
        ds4_gpu_stream_expert_popcount(resident_mask);
    g_stream_expert_timing_split_missing_experts +=
        ds4_gpu_stream_expert_popcount(missing_mask);
    g_stream_expert_timing_split_resident_ms += resident_ms;
    g_stream_expert_timing_split_missing_ms += missing_ms;
}

static void ds4_gpu_stream_expert_timing_note_split_missing_detail(
        double load_ms,
        double slot_ms,
        double prune_ms,
        double addr_ms,
        double wait_ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_split_missing_load_ms += load_ms;
    g_stream_expert_timing_split_missing_slot_ms += slot_ms;
    g_stream_expert_timing_split_missing_prune_ms += prune_ms;
    g_stream_expert_timing_split_missing_addr_ms += addr_ms;
    g_stream_expert_timing_split_missing_wait_ms += wait_ms;
}

static void ds4_gpu_stream_expert_timing_note_load_detail(
        double prepare_ms,
        double pread_ms,
        double modify_ms,
        double install_ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_load_calls++;
    g_stream_expert_timing_load_prepare_ms += prepare_ms;
    g_stream_expert_timing_load_pread_ms += pread_ms;
    g_stream_expert_timing_load_modify_ms += modify_ms;
    g_stream_expert_timing_load_install_ms += install_ms;
}

static void ds4_gpu_stream_expert_timing_note_prepare_batch_reuse(double ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_prepare_batch_reuse_calls++;
    g_stream_expert_timing_prepare_batch_reuse_ms += ms;
}

static void ds4_gpu_stream_expert_timing_note_prepare_buffer(double ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_prepare_buffer_calls++;
    g_stream_expert_timing_prepare_buffer_ms += ms;
}

static void ds4_gpu_stream_expert_timing_note_prepare_task(
        uint32_t experts,
        double   ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_prepare_task_experts += experts;
    g_stream_expert_timing_prepare_task_ms += ms;
}

static void ds4_gpu_stream_expert_timing_note_reuse_scan(
        uint64_t entries,
        double   ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_reuse_scan_calls++;
    g_stream_expert_timing_reuse_scan_entries += entries;
    g_stream_expert_timing_reuse_scan_ms += ms;
}

static void ds4_gpu_stream_expert_timing_note_reuse_clear(double ms) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    g_stream_expert_timing_reuse_clear_ms += ms;
}

static void ds4_gpu_stream_expert_timing_note_cache_class(
        uint32_t resident_mask,
        uint32_t missing_mask) {
    if (!ds4_gpu_stream_expert_timing_summary_enabled()) return;
    const uint32_t resident = ds4_gpu_stream_expert_popcount(resident_mask);
    const uint32_t missing = ds4_gpu_stream_expert_popcount(missing_mask);
    if (missing == 0) {
        g_stream_expert_timing_cache_all_resident_layers++;
    } else if (resident == 0) {
        g_stream_expert_timing_cache_all_missing_layers++;
    } else {
        g_stream_expert_timing_cache_mixed_layers++;
    }
    g_stream_expert_timing_cache_resident_experts += resident;
    g_stream_expert_timing_cache_missing_experts += missing;
}

static int ds4_gpu_stream_expert_readahead_enabled(void) {
    return g_ssd_streaming_mode &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD") == NULL;
}

static void ds4_gpu_stream_expert_readahead_range(uint64_t offset, uint64_t len) {
    if (!ds4_gpu_stream_expert_readahead_enabled() || g_model_fd < 0 || len == 0) {
        return;
    }

#if defined(F_RDADVISE)
    const int timing = ds4_gpu_stream_expert_timing_summary_enabled();
    const double t0 = timing ? ds4_gpu_now_ms() : 0.0;
    uint64_t pos = offset;
    uint64_t rem = len;
    while (rem > 0) {
        const uint64_t chunk64 =
            rem > (uint64_t)INT_MAX ? (uint64_t)INT_MAX : rem;
        if (pos > (uint64_t)LLONG_MAX) break;

        struct radvisory ra;
        ra.ra_offset = (off_t)pos;
        ra.ra_count = (int)chunk64;
        (void)fcntl(g_model_fd, F_RDADVISE, &ra);

        pos += chunk64;
        rem -= chunk64;
    }
    if (timing) {
        g_stream_expert_timing_readahead_calls++;
        g_stream_expert_timing_readahead_bytes += len;
        g_stream_expert_timing_readahead_ms += ds4_gpu_now_ms() - t0;
    }
#else
    (void)offset;
    (void)len;
#endif
}

typedef struct {
    uint64_t offset;
    uint64_t len;
    uint8_t *dst;
    uint64_t read_bytes;
    double ms;
    int ok;
} ds4_gpu_stream_expert_pread_task;

typedef struct {
    int active;
    const void *model_map;
    uint64_t model_size;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint32_t missing_mask;
    uint32_t load_slots[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint32_t source_slots[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint32_t n_loads;
    uint32_t n_tasks;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    int32_t selected_ids[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint64_t gate_abs_offsets[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint64_t up_abs_offsets[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint64_t down_abs_offsets[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    __strong id<MTLBuffer> gate_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    __strong id<MTLBuffer> up_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    __strong id<MTLBuffer> down_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    NSUInteger gate_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    NSUInteger up_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    NSUInteger down_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    ds4_gpu_stream_expert_pread_task tasks[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED * 3u];
    double start_ms;
    double prepare_ms;
} ds4_gpu_stream_expert_pending_load;

static ds4_gpu_stream_expert_pending_load g_stream_expert_pending_load;

typedef struct {
    int active;
    const void *model_map;
    uint64_t model_size;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    int32_t selected_ids[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
} ds4_gpu_glm_stream_selected_prefetch;

static ds4_gpu_glm_stream_selected_prefetch g_glm_stream_selected_prefetch;

typedef struct {
    ds4_gpu_stream_expert_pread_task *tasks;
    uint32_t n_tasks;
    uint32_t worker_index;
    uint32_t n_workers;
} ds4_gpu_stream_expert_pread_worker_args;

static void ds4_gpu_stream_expert_cache_note_pread(
        uint32_t layer,
        uint64_t bytes,
        double   ms) {
    if (g_stream_expert_cache_pread_bytes > UINT64_MAX - bytes) {
        g_stream_expert_cache_pread_bytes = UINT64_MAX;
    } else {
        g_stream_expert_cache_pread_bytes += bytes;
    }
    g_stream_expert_cache_pread_ms += ms;
    if (layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER) {
        if (g_stream_expert_cache_layer_pread_bytes[layer] > UINT64_MAX - bytes) {
            g_stream_expert_cache_layer_pread_bytes[layer] = UINT64_MAX;
        } else {
            g_stream_expert_cache_layer_pread_bytes[layer] += bytes;
        }
        g_stream_expert_cache_layer_pread_ms[layer] += ms;
    }
}

static uint32_t ds4_gpu_stream_expert_pread_thread_limit(void) {
    uint32_t threads = 9;
    const char *env = getenv("DS4_METAL_STREAMING_EXPERT_PREAD_THREADS");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0') {
            threads = v > UINT32_MAX ? UINT32_MAX : (uint32_t)v;
        }
    }
    if (threads == 0) threads = 1;
    if (threads > 18) threads = 18;
    return threads;
}

static uint32_t ds4_gpu_stream_expert_pread_thread_count(uint32_t n_tasks) {
    if (n_tasks <= 1) return n_tasks;
    uint32_t threads = ds4_gpu_stream_expert_pread_thread_limit();
    if (threads > n_tasks) threads = n_tasks;
    return threads;
}

static int ds4_gpu_stream_expert_pread_into(
        uint64_t  offset,
        uint64_t  len,
        uint8_t  *dst,
        uint64_t *read_bytes,
        double   *ms_out) {
    if (read_bytes) *read_bytes = 0;
    if (ms_out) *ms_out = 0.0;
    if (g_model_fd < 0 ||
        !dst ||
        len == 0 ||
        offset > (uint64_t)LLONG_MAX ||
        len > (uint64_t)LLONG_MAX - offset) {
        return 0;
    }

    const double t0 = ds4_gpu_now_ms();
    uint64_t pos = 0;
    int ok = 1;
    while (pos < len) {
        const uint64_t rem = len - pos;
        const size_t want = rem > (uint64_t)SSIZE_MAX ? (size_t)SSIZE_MAX : (size_t)rem;
        ssize_t nread;
        do {
            nread = pread(g_model_fd, dst + pos, want, (off_t)(offset + pos));
        } while (nread < 0 && errno == EINTR);
        if (nread <= 0) {
            ok = 0;
            break;
        }
        pos += (uint64_t)nread;
    }
    const double dt = ds4_gpu_now_ms() - t0;
    if (read_bytes) *read_bytes = pos;
    if (ms_out) *ms_out = dt;
    if (!ok || pos != len) {
        fprintf(stderr,
                "ds4: Metal streaming expert explicit pread failed offset=%.2f GiB len=%.2f MiB read=%.2f MiB\n",
                ds4_gpu_gib(offset),
                ds4_gpu_mib(len),
                ds4_gpu_mib(pos));
        return 0;
    }
    return 1;
}

static void *ds4_gpu_stream_expert_pread_worker(void *arg) {
    ds4_gpu_stream_expert_pread_worker_args *wa =
        (ds4_gpu_stream_expert_pread_worker_args *)arg;
    for (uint32_t i = wa->worker_index; i < wa->n_tasks; i += wa->n_workers) {
        ds4_gpu_stream_expert_pread_task *task = &wa->tasks[i];
        task->ok = ds4_gpu_stream_expert_pread_into(task->offset,
                                                    task->len,
                                                    task->dst,
                                                    &task->read_bytes,
                                                    &task->ms);
    }
    return NULL;
}

static pthread_mutex_t g_stream_expert_pread_pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_stream_expert_pread_pool_start_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_stream_expert_pread_pool_done_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_stream_expert_pread_pool_threads[18];
static uint32_t g_stream_expert_pread_pool_thread_count;
static uint32_t g_stream_expert_pread_pool_active_workers;
static uint32_t g_stream_expert_pread_pool_remaining_workers;
static uint32_t g_stream_expert_pread_pool_n_tasks;
static uint32_t g_stream_expert_pread_pool_next_task;
static uint64_t g_stream_expert_pread_pool_generation;
static ds4_gpu_stream_expert_pread_task *g_stream_expert_pread_pool_tasks;
static int g_stream_expert_pread_pool_initialized;
static int g_stream_expert_pread_pool_stopping;

static int ds4_gpu_stream_expert_pread_pool_enabled(void) {
    const char *env = getenv("DS4_METAL_STREAMING_EXPERT_PREAD_POOL");
    return !(env && strcmp(env, "0") == 0);
}

static void *ds4_gpu_stream_expert_pread_pool_worker(void *arg) {
    const uint32_t worker_index = (uint32_t)(uintptr_t)arg;
    uint64_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&g_stream_expert_pread_pool_mutex);
        while (!g_stream_expert_pread_pool_stopping &&
               g_stream_expert_pread_pool_generation == seen_generation) {
            pthread_cond_wait(&g_stream_expert_pread_pool_start_cond,
                              &g_stream_expert_pread_pool_mutex);
        }
        if (g_stream_expert_pread_pool_stopping) {
            pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
            break;
        }

        seen_generation = g_stream_expert_pread_pool_generation;
        if (worker_index >= g_stream_expert_pread_pool_active_workers) {
            pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
            continue;
        }

        for (;;) {
            const uint32_t task_index =
                g_stream_expert_pread_pool_next_task++;
            if (task_index >= g_stream_expert_pread_pool_n_tasks) break;

            ds4_gpu_stream_expert_pread_task *task =
                &g_stream_expert_pread_pool_tasks[task_index];
            pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);

            task->ok = ds4_gpu_stream_expert_pread_into(task->offset,
                                                        task->len,
                                                        task->dst,
                                                        &task->read_bytes,
                                                        &task->ms);

            pthread_mutex_lock(&g_stream_expert_pread_pool_mutex);
        }

        if (g_stream_expert_pread_pool_remaining_workers > 0 &&
            --g_stream_expert_pread_pool_remaining_workers == 0) {
            g_stream_expert_pread_pool_tasks = NULL;
            g_stream_expert_pread_pool_n_tasks = 0;
            g_stream_expert_pread_pool_active_workers = 0;
            pthread_cond_signal(&g_stream_expert_pread_pool_done_cond);
        }
        pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
    }

    return NULL;
}

static int ds4_gpu_stream_expert_pread_pool_init(uint32_t n_threads) {
    if (g_stream_expert_pread_pool_initialized) return 1;
    if (!ds4_gpu_stream_expert_pread_pool_enabled() || n_threads <= 1) return 0;
    if (n_threads > 18) n_threads = 18;

    pthread_mutex_lock(&g_stream_expert_pread_pool_mutex);
    g_stream_expert_pread_pool_thread_count = n_threads;
    g_stream_expert_pread_pool_stopping = 0;
    g_stream_expert_pread_pool_generation = 0;
    g_stream_expert_pread_pool_tasks = NULL;
    g_stream_expert_pread_pool_n_tasks = 0;
    g_stream_expert_pread_pool_next_task = 0;
    g_stream_expert_pread_pool_active_workers = 0;
    g_stream_expert_pread_pool_remaining_workers = 0;
    pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);

    uint32_t started = 0;
    for (uint32_t i = 0; i < n_threads; i++) {
        const int rc = pthread_create(&g_stream_expert_pread_pool_threads[i],
                                      NULL,
                                      ds4_gpu_stream_expert_pread_pool_worker,
                                      (void *)(uintptr_t)i);
        if (rc != 0) {
            fprintf(stderr,
                    "ds4: Metal streaming expert pread pool thread creation failed: %s\n",
                    strerror(rc));
            pthread_mutex_lock(&g_stream_expert_pread_pool_mutex);
            g_stream_expert_pread_pool_stopping = 1;
            g_stream_expert_pread_pool_generation++;
            pthread_cond_broadcast(&g_stream_expert_pread_pool_start_cond);
            pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
            for (uint32_t j = 0; j < started; j++) {
                (void)pthread_join(g_stream_expert_pread_pool_threads[j], NULL);
            }
            pthread_mutex_lock(&g_stream_expert_pread_pool_mutex);
            g_stream_expert_pread_pool_thread_count = 0;
            g_stream_expert_pread_pool_stopping = 0;
            pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
            return 0;
        }
        started++;
    }

    g_stream_expert_pread_pool_initialized = 1;
    return 1;
}

static int ds4_gpu_stream_expert_pread_pool_begin(
        ds4_gpu_stream_expert_pread_task *tasks,
        uint32_t n_tasks,
        uint32_t n_workers) {
    if (n_workers <= 1) return 0;
    const uint32_t limit = ds4_gpu_stream_expert_pread_thread_limit();
    if (!ds4_gpu_stream_expert_pread_pool_init(limit)) return 0;

    pthread_mutex_lock(&g_stream_expert_pread_pool_mutex);
    if (!g_stream_expert_pread_pool_initialized ||
        g_stream_expert_pread_pool_stopping ||
        g_stream_expert_pread_pool_thread_count == 0) {
        pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
        return 0;
    }
    if (n_workers > g_stream_expert_pread_pool_thread_count) {
        n_workers = g_stream_expert_pread_pool_thread_count;
    }
    if (n_workers == 0) {
        pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
        return 0;
    }

    if (g_stream_expert_pread_pool_remaining_workers != 0 ||
        g_stream_expert_pread_pool_tasks != NULL) {
        pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
        return 0;
    }

    g_stream_expert_pread_pool_tasks = tasks;
    g_stream_expert_pread_pool_n_tasks = n_tasks;
    g_stream_expert_pread_pool_next_task = 0;
    g_stream_expert_pread_pool_active_workers = n_workers;
    g_stream_expert_pread_pool_remaining_workers = n_workers;
    g_stream_expert_pread_pool_generation++;
    pthread_cond_broadcast(&g_stream_expert_pread_pool_start_cond);
    pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
    return 1;
}

static int ds4_gpu_stream_expert_pread_pool_wait(void) {
    if (!g_stream_expert_pread_pool_initialized) return 0;

    pthread_mutex_lock(&g_stream_expert_pread_pool_mutex);
    while (g_stream_expert_pread_pool_remaining_workers != 0) {
        pthread_cond_wait(&g_stream_expert_pread_pool_done_cond,
                          &g_stream_expert_pread_pool_mutex);
    }
    pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
    return 1;
}

static int ds4_gpu_stream_expert_pread_pool_dispatch(
        ds4_gpu_stream_expert_pread_task *tasks,
        uint32_t n_tasks,
        uint32_t n_workers) {
    if (!ds4_gpu_stream_expert_pread_pool_begin(tasks, n_tasks, n_workers)) {
        return 0;
    }
    return ds4_gpu_stream_expert_pread_pool_wait();
}

static void ds4_gpu_stream_expert_pread_pool_shutdown(void) {
    if (!g_stream_expert_pread_pool_initialized) return;

    pthread_mutex_lock(&g_stream_expert_pread_pool_mutex);
    g_stream_expert_pread_pool_stopping = 1;
    g_stream_expert_pread_pool_generation++;
    pthread_cond_broadcast(&g_stream_expert_pread_pool_start_cond);
    pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);

    const uint32_t n_threads = g_stream_expert_pread_pool_thread_count;
    for (uint32_t i = 0; i < n_threads; i++) {
        (void)pthread_join(g_stream_expert_pread_pool_threads[i], NULL);
    }

    pthread_mutex_lock(&g_stream_expert_pread_pool_mutex);
    g_stream_expert_pread_pool_thread_count = 0;
    g_stream_expert_pread_pool_active_workers = 0;
    g_stream_expert_pread_pool_remaining_workers = 0;
    g_stream_expert_pread_pool_n_tasks = 0;
    g_stream_expert_pread_pool_next_task = 0;
    g_stream_expert_pread_pool_tasks = NULL;
    g_stream_expert_pread_pool_initialized = 0;
    g_stream_expert_pread_pool_stopping = 0;
    pthread_mutex_unlock(&g_stream_expert_pread_pool_mutex);
}

static int ds4_gpu_stream_expert_pread_tasks(
        ds4_gpu_stream_expert_pread_task *tasks,
        uint32_t n_tasks,
        uint64_t *total_bytes,
        double *wall_ms) {
    if (total_bytes) *total_bytes = 0;
    if (wall_ms) *wall_ms = 0.0;
    if (!tasks || n_tasks == 0) return 1;

    const uint32_t n_workers =
        ds4_gpu_stream_expert_pread_thread_count(n_tasks);
    const double t0 = ds4_gpu_now_ms();
    int ok = 1;
    if (n_workers <= 1) {
        ds4_gpu_stream_expert_pread_worker_args wa = {
            .tasks = tasks,
            .n_tasks = n_tasks,
            .worker_index = 0,
            .n_workers = 1,
        };
        (void)ds4_gpu_stream_expert_pread_worker(&wa);
    } else if (!ds4_gpu_stream_expert_pread_pool_dispatch(tasks,
                                                          n_tasks,
                                                          n_workers)) {
        pthread_t threads[18];
        ds4_gpu_stream_expert_pread_worker_args args[18];
        uint32_t started = 0;
        for (uint32_t i = 0; i < n_workers; i++) {
            args[i].tasks = tasks;
            args[i].n_tasks = n_tasks;
            args[i].worker_index = i;
            args[i].n_workers = n_workers;
            const int rc = pthread_create(&threads[i],
                                          NULL,
                                          ds4_gpu_stream_expert_pread_worker,
                                          &args[i]);
            if (rc != 0) {
                fprintf(stderr,
                        "ds4: Metal streaming expert pread thread creation failed: %s\n",
                        strerror(rc));
                ok = 0;
                break;
            }
            started++;
        }
        for (uint32_t i = 0; i < started; i++) {
            if (pthread_join(threads[i], NULL) != 0) ok = 0;
        }
    }
    const double dt = ds4_gpu_now_ms() - t0;

    uint64_t bytes = 0;
    for (uint32_t i = 0; i < n_tasks; i++) {
        if (!tasks[i].ok) ok = 0;
        if (bytes > UINT64_MAX - tasks[i].read_bytes) {
            bytes = UINT64_MAX;
        } else {
            bytes += tasks[i].read_bytes;
        }
    }
    if (total_bytes) *total_bytes = bytes;
    if (wall_ms) *wall_ms = dt;
    return ok;
}

static id<MTLBuffer> ds4_gpu_stream_expert_alloc_buffer(
        uint64_t  len,
        NSString *label) {
    if (!g_device ||
        len == 0 ||
        len > (uint64_t)NSUIntegerMax) {
        return nil;
    }

    id<MTLBuffer> buffer = [g_device newBufferWithLength:(NSUInteger)len
                                                 options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr,
                "ds4: Metal streaming expert explicit buffer allocation failed (%.2f MiB)\n",
                ds4_gpu_mib(len));
        return nil;
    }
    buffer.label = label;
    g_stream_expert_cache_buffer_allocs++;
    return buffer;
}

static int ds4_gpu_stream_expert_combined_buffer_enabled(void) {
    return g_ssd_streaming_mode &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_COMBINED_BUFFER") == NULL;
}

static int ds4_gpu_stream_expert_slab_enabled(void) {
    return ds4_gpu_stream_expert_combined_buffer_enabled() &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS") == NULL;
}

/*
 * Large PRO caches otherwise create thousands of small shared Metal buffers.
 * Slabs keep the buffer object set small while locking pages only for slots
 * that actually hold a streamed expert.
 */
static uint64_t ds4_gpu_stream_expert_slab_target_bytes(void) {
    const uint64_t mib = 1024ull * 1024ull;
    uint64_t target = 4096ull * mib;
    const char *env = getenv("DS4_METAL_STREAMING_EXPERT_SLAB_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && *end == '\0' && v != 0) {
            target = v > UINT64_MAX / mib ? UINT64_MAX : (uint64_t)v * mib;
        }
    }
    return target;
}

static id<MTLBuffer> ds4_gpu_stream_expert_alloc_slab_buffer(
        uint64_t  len,
        NSString *label) {
    if (!g_device ||
        len == 0 ||
        len > (uint64_t)NSUIntegerMax) {
        return nil;
    }

    id<MTLBuffer> buffer = [g_device newBufferWithLength:(NSUInteger)len
                                                 options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr,
                "ds4: Metal streaming expert slab allocation failed (%.2f MiB)\n",
                ds4_gpu_mib(len));
        return nil;
    }
    buffer.label = label;
    g_stream_expert_cache_buffer_allocs++;
    return buffer;
}

static int ds4_gpu_stream_expert_slab_slot_range(
        uint32_t slot,
        uint32_t *slab_index,
        uint64_t *slot_base) {
    for (uint32_t i = 0; i < g_stream_expert_cache_slab_count; i++) {
        const uint32_t start = g_stream_expert_cache_slab_start_slot[i];
        const uint32_t count = g_stream_expert_cache_slab_slot_count[i];
        if (slot < start || slot >= start + count) continue;
        if (slab_index) *slab_index = i;
        if (slot_base) {
            *slot_base =
                (uint64_t)(slot - start) * g_stream_expert_cache_slab_slot_bytes;
        }
        return 1;
    }
    return 0;
}

static int ds4_gpu_stream_expert_slab_slot_for_buffer(
        id<MTLBuffer> buffer,
        NSUInteger    gate_inner,
        uint32_t     *slot_out) {
    if (!buffer || g_stream_expert_cache_slab_slot_bytes == 0 || !slot_out) {
        return 0;
    }
    for (uint32_t i = 0; i < g_stream_expert_cache_slab_count; i++) {
        if (g_stream_expert_cache_slabs[i] != buffer) continue;
        const uint64_t inner = (uint64_t)gate_inner;
        const uint64_t slot_bytes = g_stream_expert_cache_slab_slot_bytes;
        if (slot_bytes == 0 || inner % slot_bytes != 0) return 0;
        const uint64_t local_slot = inner / slot_bytes;
        if (local_slot >= g_stream_expert_cache_slab_slot_count[i]) return 0;
        const uint64_t slot =
            (uint64_t)g_stream_expert_cache_slab_start_slot[i] + local_slot;
        if (slot > UINT32_MAX) return 0;
        *slot_out = (uint32_t)slot;
        return 1;
    }
    return 0;
}

static void ds4_gpu_stream_expert_slab_push_free_slot(uint32_t slot) {
    if (slot >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES ||
        g_stream_expert_cache_free_slot_count >=
            DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES) {
        return;
    }
    g_stream_expert_cache_free_slots[g_stream_expert_cache_free_slot_count++] =
        slot;
}

static int ds4_gpu_stream_expert_slab_slot_buffers(
        uint32_t slot,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        __strong id<MTLBuffer> *gate_buf,
        __strong id<MTLBuffer> *up_buf,
        __strong id<MTLBuffer> *down_buf,
        NSUInteger *gate_inner,
        NSUInteger *up_inner,
        NSUInteger *down_inner) {
    uint32_t slab = UINT32_MAX;
    uint64_t base = 0;
    if (!ds4_gpu_stream_expert_slab_slot_range(slot, &slab, &base) ||
        slab >= g_stream_expert_cache_slab_count ||
        !g_stream_expert_cache_slabs[slab]) {
        return 0;
    }
    if (gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull ||
        base > UINT64_MAX - (gate_expert_bytes * 2ull + down_expert_bytes) ||
        base + gate_expert_bytes * 2ull + down_expert_bytes >
            (uint64_t)NSUIntegerMax) {
        return 0;
    }
    id<MTLBuffer> b = g_stream_expert_cache_slabs[slab];
    *gate_buf = b;
    *up_buf = b;
    *down_buf = b;
    *gate_inner = (NSUInteger)base;
    *up_inner = (NSUInteger)(base + gate_expert_bytes);
    *down_inner = (NSUInteger)(base + gate_expert_bytes * 2ull);
    return 1;
}

static int ds4_gpu_stream_expert_alloc_slab_slot(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        __strong id<MTLBuffer> *gate_buf,
        __strong id<MTLBuffer> *up_buf,
        __strong id<MTLBuffer> *down_buf,
        NSUInteger *gate_inner,
        NSUInteger *up_inner,
        NSUInteger *down_inner) {
    if (!ds4_gpu_stream_expert_slab_enabled() ||
        !gate_buf || !up_buf || !down_buf ||
        !gate_inner || !up_inner || !down_inner ||
        gate_expert_bytes == 0 || down_expert_bytes == 0 ||
        gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        return 0;
    }

    uint64_t slot_bytes = gate_expert_bytes * 2ull + down_expert_bytes;
    if (slot_bytes == 0 || slot_bytes > (uint64_t)NSUIntegerMax) return 0;
    const uint64_t page = (uint64_t)getpagesize();
    if (page != 0) {
        slot_bytes = round_up_u64(slot_bytes, page);
        if (slot_bytes == 0 || slot_bytes > (uint64_t)NSUIntegerMax) return 0;
    }
    if (g_stream_expert_cache_slab_slot_bytes != 0 &&
        g_stream_expert_cache_slab_slot_bytes != slot_bytes) {
        return 0;
    }
    g_stream_expert_cache_slab_slot_bytes = slot_bytes;

    if (g_stream_expert_cache_free_slot_count != 0) {
        const uint32_t slot =
            g_stream_expert_cache_free_slots[--g_stream_expert_cache_free_slot_count];
        return ds4_gpu_stream_expert_slab_slot_buffers(slot,
                                                       gate_expert_bytes,
                                                       down_expert_bytes,
                                                       gate_buf,
                                                       up_buf,
                                                       down_buf,
                                                       gate_inner,
                                                       up_inner,
                                                       down_inner);
    }

    uint32_t slab = g_stream_expert_cache_slab_count;
    if (slab != 0 &&
        g_stream_expert_cache_slab_slots_used[slab - 1] <
            g_stream_expert_cache_slab_slot_count[slab - 1]) {
        slab--;
    } else {
        if (g_stream_expert_cache_slab_count >=
            DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS) {
            return 0;
        }
        const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
        if (budget != 0 && g_stream_expert_cache_slab_total_slots >= budget) {
            return 0;
        }
        uint64_t target = ds4_gpu_stream_expert_slab_target_bytes();
        uint64_t slots64 = target / slot_bytes;
        if (slots64 == 0) slots64 = 1;
        if (slots64 > UINT32_MAX) slots64 = UINT32_MAX;
        uint32_t slots = (uint32_t)slots64;
        if (budget != 0) {
            const uint32_t remaining =
                budget - g_stream_expert_cache_slab_total_slots;
            if (slots > remaining) slots = remaining;
        }
        if (slots == 0) return 0;
        id<MTLBuffer> slab_buffer = nil;
        while (slots != 0) {
            if ((uint64_t)slots <= UINT64_MAX / slot_bytes &&
                (uint64_t)slots * slot_bytes <= (uint64_t)NSUIntegerMax) {
                slab_buffer =
                    ds4_gpu_stream_expert_alloc_slab_buffer(
                            (uint64_t)slots * slot_bytes,
                            @"ds4_stream_expert_slab");
                if (slab_buffer) break;
            }
            slots /= 2u;
        }
        if (!slab_buffer || slots == 0) return 0;

        slab = g_stream_expert_cache_slab_count++;
        g_stream_expert_cache_slabs[slab] = slab_buffer;
        g_stream_expert_cache_slab_start_slot[slab] =
            g_stream_expert_cache_slab_total_slots;
        g_stream_expert_cache_slab_slot_count[slab] = slots;
        g_stream_expert_cache_slab_slots_used[slab] = 0;
        g_stream_expert_cache_slab_total_slots += slots;
    }

    const uint32_t local_slot = g_stream_expert_cache_slab_slots_used[slab]++;
    const uint32_t slot =
        g_stream_expert_cache_slab_start_slot[slab] + local_slot;
    return ds4_gpu_stream_expert_slab_slot_buffers(slot,
                                                   gate_expert_bytes,
                                                   down_expert_bytes,
                                                   gate_buf,
                                                   up_buf,
                                                   down_buf,
                                                   gate_inner,
                                                   up_inner,
                                                   down_inner);
}

static uint64_t ds4_gpu_stream_expert_buffer_object_count(
        id<MTLBuffer> gate,
        id<MTLBuffer> up,
        id<MTLBuffer> down) {
    if (!gate || !up || !down) return 0;
    if (gate == up && gate == down) return 1;
    if (gate == up || gate == down || up == down) return 2;
    return 3;
}

static int ds4_gpu_stream_expert_evict_dontneed_enabled(void) {
    return g_ssd_streaming_mode &&
           getenv("DS4_METAL_ENABLE_STREAMING_EXPERT_EVICT_DONTNEED") != NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_EVICT_DONTNEED") == NULL;
}

static void ds4_gpu_stream_expert_evict_dontneed_range(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    len) {
    if (!ds4_gpu_stream_expert_evict_dontneed_enabled() ||
        !model_map ||
        model_size == 0 ||
        offset > model_size ||
        len == 0 ||
        len > model_size - offset) {
        return;
    }

#if defined(POSIX_MADV_DONTNEED)
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t page_offset = offset & ~(page - 1);
    const uint64_t leading = offset - page_offset;
    if (len > UINT64_MAX - leading ||
        leading + len > UINT64_MAX - (page - 1)) {
        return;
    }
    uint64_t advise_bytes = round_up_u64(leading + len, page);
    if (advise_bytes > model_size - page_offset) {
        advise_bytes = model_size - page_offset;
    }
    if (advise_bytes == 0 || advise_bytes > (uint64_t)SIZE_MAX) return;

    const uintptr_t base = (uintptr_t)model_map;
    if (page_offset > (uint64_t)(UINTPTR_MAX - base)) return;
    void *addr = (void *)(base + (uintptr_t)page_offset);
    const int rc = posix_madvise(addr, (size_t)advise_bytes, POSIX_MADV_DONTNEED);
    if (rc == 0) {
        if (g_stream_expert_cache_evict_advise_bytes > UINT64_MAX - advise_bytes) {
            g_stream_expert_cache_evict_advise_bytes = UINT64_MAX;
        } else {
            g_stream_expert_cache_evict_advise_bytes += advise_bytes;
        }
    } else if (getenv("DS4_METAL_STREAMING_EXPERT_EVICT_DONTNEED_PROFILE") != NULL) {
        fprintf(stderr,
                "ds4: Metal streaming expert evict DONTNEED failed offset=%.2f GiB len=%.2f MiB: %s\n",
                ds4_gpu_gib(offset),
                ds4_gpu_mib(len),
                strerror(rc));
    }
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)len;
#endif
}

static int ds4_gpu_stream_expert_split_requested(void) {
    return g_ssd_streaming_mode;
}

static uint32_t ds4_gpu_stream_expert_split_min_decode_tokens(void) {
    return 4;
}

static uint32_t ds4_gpu_stream_expert_split_min_cached(void) {
    uint32_t min_cached = 1024;
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    if (budget != 0 && budget < min_cached * 2u) {
        min_cached = budget / 2u;
    }
    return min_cached;
}

static int ds4_gpu_stream_expert_split_ready(void) {
    if (!ds4_gpu_stream_expert_split_requested()) return 0;
    if (g_stream_expert_cache_decode_tokens <
        ds4_gpu_stream_expert_split_min_decode_tokens()) {
        return 0;
    }
    return g_stream_expert_cache_entry_count >=
           ds4_gpu_stream_expert_split_min_cached();
}

static void ds4_gpu_stream_expert_cache_decay_route_hotness(void) {
    for (uint32_t layer = 0;
         layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
         layer++) {
        for (uint32_t expert = 0;
             expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
             expert++) {
            g_stream_expert_cache_route_hotness[layer][expert] >>= 1;
        }
    }
}

void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
    memset(g_stream_expert_cache_route_hotness,
           0,
           sizeof(g_stream_expert_cache_route_hotness));
    g_stream_expert_cache_hotness_decay_token =
        g_stream_expert_cache_decode_tokens;
}

static void ds4_gpu_stream_expert_cache_maybe_decay_route_hotness(void) {
    if (g_stream_expert_cache_decode_tokens == 0) return;
    if (g_stream_expert_cache_hotness_decay_token == 0) {
        g_stream_expert_cache_hotness_decay_token =
            g_stream_expert_cache_decode_tokens;
        return;
    }
    while (g_stream_expert_cache_decode_tokens -
           g_stream_expert_cache_hotness_decay_token >=
           DS4_METAL_STREAM_EXPERT_HOTNESS_DECAY_TOKENS) {
        ds4_gpu_stream_expert_cache_decay_route_hotness();
        g_stream_expert_cache_hotness_decay_token +=
            DS4_METAL_STREAM_EXPERT_HOTNESS_DECAY_TOKENS;
    }
}

static void ds4_gpu_stream_expert_cache_note_route_hotness(
        uint32_t layer,
        uint32_t expert,
        uint32_t amount) {
    if (layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        expert >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        amount == 0) {
        return;
    }
    uint32_t *hotness = &g_stream_expert_cache_route_hotness[layer][expert];
    if (*hotness > UINT32_MAX - amount) {
        *hotness = UINT32_MAX;
    } else {
        *hotness += amount;
    }
}

static void ds4_gpu_stream_expert_cache_note_selected_hotness(
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_selected) {
    if (!selected_ids ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_selected == 0) {
        return;
    }
    ds4_gpu_stream_expert_cache_maybe_decay_route_hotness();
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 ||
            selected_ids[i] >=
                (int32_t)DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) {
            continue;
        }
        ds4_gpu_stream_expert_cache_note_route_hotness(
                layer,
                (uint32_t)selected_ids[i],
                1);
    }
}

static void ds4_gpu_stream_expert_cache_note_frequency_hotness(
        uint32_t        layer,
        const uint32_t *frequency,
        uint32_t        n_total_expert) {
    if (!frequency || layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER) {
        return;
    }
    if (n_total_expert > DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) {
        n_total_expert = DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
    }
    ds4_gpu_stream_expert_cache_maybe_decay_route_hotness();
    for (uint32_t expert = 0; expert < n_total_expert; expert++) {
        ds4_gpu_stream_expert_cache_note_route_hotness(layer,
                                                       expert,
                                                       frequency[expert]);
    }
}

static void ds4_gpu_stream_expert_cache_note_token(uint32_t layer_index) {
    if (!g_ssd_streaming_mode || layer_index != 0 ||
        g_stream_expert_cache_decode_tokens == UINT64_MAX) {
        return;
    }
    g_stream_expert_cache_decode_tokens++;
    ds4_gpu_stream_expert_cache_maybe_decay_route_hotness();
}

static void ds4_gpu_stream_expert_cache_note_decode_token(void) {
    if (!g_ssd_streaming_mode ||
        g_stream_expert_cache_decode_tokens == UINT64_MAX) {
        return;
    }
    g_stream_expert_cache_decode_tokens++;
    ds4_gpu_stream_expert_cache_maybe_decay_route_hotness();
}

static int ds4_gpu_stream_compact_addr_requested(void) {
    return g_ssd_streaming_mode &&
           getenv("DS4_METAL_ENABLE_STREAMING_COMPACT_ADDR") != NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_COMPACT_ADDR") == NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") == NULL;
}

static int ds4_gpu_stream_expert_addr_table_requested(void) {
    return g_ssd_streaming_mode &&
           (getenv("DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE") != NULL ||
            getenv("DS4_METAL_ENABLE_STREAMING_EXPERT_HIT_VALIDATOR") != NULL ||
            getenv("DS4_METAL_ENABLE_STREAMING_EXPERT_MASKED_ADDR") != NULL ||
            g_stream_prefill_batch_selected_addr_building ||
            g_glm_stream_expert_addr_table_building ||
            (getenv("DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") != NULL &&
             getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") == NULL) ||
            ds4_gpu_stream_expert_split_requested()) &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") == NULL;
}

static int ds4_gpu_stream_expert_addr_table_kernel_requested(void) {
    return g_ssd_streaming_mode &&
           (getenv("DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE") != NULL ||
            getenv("DS4_METAL_ENABLE_STREAMING_EXPERT_HIT_VALIDATOR") != NULL ||
            getenv("DS4_METAL_ENABLE_STREAMING_EXPERT_MASKED_ADDR") != NULL ||
            ds4_gpu_stream_expert_split_ready()) &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") == NULL;
}

static int ds4_gpu_stream_expert_masked_addr_requested(void) {
    return g_ssd_streaming_mode &&
           (getenv("DS4_METAL_ENABLE_STREAMING_EXPERT_MASKED_ADDR") != NULL ||
            ds4_gpu_stream_expert_split_ready()) &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_MASKED_ADDR") == NULL;
}

static int ds4_gpu_stream_expert_hit_validator_requested(void) {
    return g_ssd_streaming_mode &&
           getenv("DS4_METAL_ENABLE_STREAMING_EXPERT_HIT_VALIDATOR") != NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_HIT_VALIDATOR") == NULL;
}

static uint32_t ds4_gpu_stream_prefill_batch_selected_addr_auto_max(
        uint32_t n_total_expert) {
    const char *env = getenv("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX");
    if (env && env[0]) {
        char *end = NULL;
        const long v = strtol(env, &end, 10);
        if (end != env) {
            if (v <= 0) return 0;
            if ((unsigned long)v > (unsigned long)UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
    if (n_total_expert == 384) return 800u;
    if (n_total_expert == 256) return 760u;
    return 0;
}

static uint32_t ds4_gpu_stream_prefill_batch_selected_addr_auto_min(
        uint32_t n_total_expert) {
    const char *env = getenv("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MIN");
    if (env && env[0]) {
        char *end = NULL;
        const long v = strtol(env, &end, 10);
        if (end != env) {
            if (v <= 0) return 0;
            if ((unsigned long)v > (unsigned long)UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
    if (n_total_expert == 384 || n_total_expert == 256) return 2u;
    return 0;
}

static int ds4_gpu_stream_prefill_batch_selected_addr_enabled(
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t gate_type,
        uint32_t down_type) {
    if (!g_ssd_streaming_mode ||
        n_tokens <= 1 ||
        n_total_expert == 0 ||
        n_expert != 6 ||
        gate_type != DS4_METAL_TENSOR_IQ2_XXS ||
        down_type != DS4_METAL_TENSOR_Q2_K ||
        g_quality_mode ||
        getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") != NULL ||
        getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") != NULL ||
        getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL ||
        getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") != NULL) {
        return 0;
    }
    /* All unique experts for one layer must fit simultaneously because the
     * address-table kernels consume them in one dispatch. Once the global
     * cache fills, preparation reuses entries owned by other layers. */
    if (ds4_gpu_stream_expert_cache_configured_count() < n_total_expert) {
        return 0;
    }
    if (getenv("DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") != NULL) {
        return 1;
    }
    const uint32_t max_tokens =
        ds4_gpu_stream_prefill_batch_selected_addr_auto_max(n_total_expert);
    const uint32_t min_tokens =
        ds4_gpu_stream_prefill_batch_selected_addr_auto_min(n_total_expert);
    return max_tokens != 0 && n_tokens >= min_tokens && n_tokens <= max_tokens;
}

static int ds4_gpu_glm_streaming_prefill_full_layer_active(void) {
    return g_glm_streaming_prefill_full_layer_runtime ||
           getenv("DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER") != NULL;
}

static int ds4_gpu_stream_full_expert_addr_table_requested(void) {
    return g_ssd_streaming_mode &&
           getenv("DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE") != NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_FULL_EXPERT_ADDR_TABLE") == NULL;
}

static uint64_t ds4_gpu_buffer_address(id<MTLBuffer> buffer, NSUInteger inner) {
    if (!buffer) return 0;
#if TARGET_OS_OSX
    if (@available(macOS 13.0, *)) {
        return (uint64_t)[buffer gpuAddress] + (uint64_t)inner;
    }
#endif
    return 0;
}

static int ds4_gpu_stream_compact_addr_ensure_buffers(uint32_t layer) {
    if (!g_device || layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER) return 0;

    const NSUInteger addr_bytes = 6u * sizeof(uint64_t);
    const NSUInteger ids_bytes = 6u * sizeof(int32_t);
    for (uint32_t i = 0; i < 4; i++) {
        id<MTLBuffer> current = nil;
        NSUInteger bytes = addr_bytes;
        NSString *label = @"ds4_stream_compact_gate_addresses";
        switch (i) {
        case 0:
            current = g_stream_compact_gate_addr_buffers[layer];
            label = @"ds4_stream_compact_gate_addresses";
            break;
        case 1:
            current = g_stream_compact_up_addr_buffers[layer];
            label = @"ds4_stream_compact_up_addresses";
            break;
        case 2:
            current = g_stream_compact_down_addr_buffers[layer];
            label = @"ds4_stream_compact_down_addresses";
            break;
        default:
            current = g_stream_compact_selected_buffers[layer];
            bytes = ids_bytes;
            label = @"ds4_stream_compact_selected_ids";
            break;
        }
        if (current) continue;
        id<MTLBuffer> b = [g_device newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared];
        if (!b) {
            fprintf(stderr, "ds4: Metal streaming compact address buffer allocation failed\n");
            return 0;
        }
        b.label = label;
        memset([b contents], 0, bytes);
        [b didModifyRange:NSMakeRange(0, bytes)];
        switch (i) {
        case 0:
            g_stream_compact_gate_addr_buffers[layer] = b;
            break;
        case 1:
            g_stream_compact_up_addr_buffers[layer] = b;
            break;
        case 2:
            g_stream_compact_down_addr_buffers[layer] = b;
            break;
        default:
            g_stream_compact_selected_buffers[layer] = b;
            break;
        }
    }
    return 1;
}

static int ds4_gpu_stream_compact_addr_prepare(
        uint32_t layer,
        ds4_gpu_stream_expert_cache_entry * const entries[6],
        uint32_t n_entries,
        id<MTLBuffer> *gate_addrs,
        id<MTLBuffer> *up_addrs,
        id<MTLBuffer> *down_addrs,
        id<MTLBuffer> *selected_ids) {
    if (layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        !entries || !gate_addrs || !up_addrs || !down_addrs || !selected_ids ||
        n_entries == 0 || n_entries > 6) {
        return 0;
    }
    if (!ds4_gpu_stream_compact_addr_ensure_buffers(layer)) return 0;

    uint64_t gate_values[6] = {0, 0, 0, 0, 0, 0};
    uint64_t up_values[6] = {0, 0, 0, 0, 0, 0};
    uint64_t down_values[6] = {0, 0, 0, 0, 0, 0};
    int32_t slot_ids[6] = {0, 1, 2, 3, 4, 5};

    for (uint32_t i = 0; i < n_entries; i++) {
        ds4_gpu_stream_expert_cache_entry *e = entries[i];
        if (!e || !e->gate_buffer || !e->up_buffer || !e->down_buffer) {
            return 0;
        }
        gate_values[i] = ds4_gpu_buffer_address(e->gate_buffer, e->gate_inner);
        up_values[i] = ds4_gpu_buffer_address(e->up_buffer, e->up_inner);
        down_values[i] = ds4_gpu_buffer_address(e->down_buffer, e->down_inner);
        if (gate_values[i] == 0 || up_values[i] == 0 || down_values[i] == 0) {
            fprintf(stderr, "ds4: Metal streaming compact address path requires GPU addresses\n");
            return 0;
        }
    }

    const NSUInteger addr_bytes = 6u * sizeof(uint64_t);
    const NSUInteger ids_bytes = 6u * sizeof(int32_t);
    id<MTLBuffer> gb = g_stream_compact_gate_addr_buffers[layer];
    id<MTLBuffer> ub = g_stream_compact_up_addr_buffers[layer];
    id<MTLBuffer> db = g_stream_compact_down_addr_buffers[layer];
    id<MTLBuffer> ib = g_stream_compact_selected_buffers[layer];
    memcpy([gb contents], gate_values, addr_bytes);
    memcpy([ub contents], up_values, addr_bytes);
    memcpy([db contents], down_values, addr_bytes);
    memcpy([ib contents], slot_ids, ids_bytes);
    [gb didModifyRange:NSMakeRange(0, addr_bytes)];
    [ub didModifyRange:NSMakeRange(0, addr_bytes)];
    [db didModifyRange:NSMakeRange(0, addr_bytes)];
    [ib didModifyRange:NSMakeRange(0, ids_bytes)];

    *gate_addrs = gb;
    *up_addrs = ub;
    *down_addrs = db;
    *selected_ids = ib;
    return 1;
}

static int ds4_gpu_stream_selected_ids_prepare(
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_selected,
        id<MTLBuffer> *selected_buf,
        NSUInteger    *selected_off) {
    if (!g_device ||
        !selected_ids ||
        !selected_buf ||
        !selected_off ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_selected == 0 ||
        n_selected > DS4_METAL_MAX_ROUTED_EXPERT_USED) {
        return 0;
    }

    const NSUInteger bytes =
        (NSUInteger)DS4_METAL_MAX_ROUTED_EXPERT_USED * sizeof(int32_t);
    id<MTLBuffer> b = g_stream_selected_id_buffers[layer];
    if (!b) {
        b = [g_device newBufferWithLength:bytes
                                  options:MTLResourceStorageModeShared];
        if (!b) {
            fprintf(stderr, "ds4: Metal streaming selected-id buffer allocation failed\n");
            return 0;
        }
        b.label = @"ds4_stream_selected_ids";
        g_stream_selected_id_buffers[layer] = b;
    }

    int32_t ids[DS4_METAL_MAX_ROUTED_EXPERT_USED] = {0};
    memcpy(ids, selected_ids, (size_t)n_selected * sizeof(ids[0]));
    memcpy([b contents], ids, bytes);
    [b didModifyRange:NSMakeRange(0, bytes)];

    *selected_buf = b;
    *selected_off = 0;
    return 1;
}

static int ds4_gpu_stream_expert_cache_ensure_addr_buffers(uint32_t layer) {
    if (!g_device ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER) {
        return 0;
    }

    const NSUInteger bytes =
        (NSUInteger)DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT * sizeof(uint64_t);
    id<MTLBuffer> buffers[3] = {
        g_stream_expert_cache_gate_addr_buffers[layer],
        g_stream_expert_cache_up_addr_buffers[layer],
        g_stream_expert_cache_down_addr_buffers[layer],
    };
    for (uint32_t i = 0; i < 3; i++) {
        if (buffers[i]) continue;
        id<MTLBuffer> b = [g_device newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared];
        if (!b) {
            fprintf(stderr, "ds4: Metal streaming expert address table allocation failed\n");
            return 0;
        }
        b.label =
            i == 0 ? @"ds4_stream_expert_gate_addresses" :
            (i == 1 ? @"ds4_stream_expert_up_addresses" :
                      @"ds4_stream_expert_down_addresses");
        memset([b contents], 0, bytes);
        [b didModifyRange:NSMakeRange(0, bytes)];
        if (i == 0) {
            g_stream_expert_cache_gate_addr_buffers[layer] = b;
        } else if (i == 1) {
            g_stream_expert_cache_up_addr_buffers[layer] = b;
        } else {
            g_stream_expert_cache_down_addr_buffers[layer] = b;
        }
    }
    return 1;
}

static void ds4_gpu_stream_expert_cache_zero_addr_slot(uint32_t layer, uint32_t expert) {
    if (layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        expert >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) {
        return;
    }

    id<MTLBuffer> buffers[3] = {
        g_stream_expert_cache_gate_addr_buffers[layer],
        g_stream_expert_cache_up_addr_buffers[layer],
        g_stream_expert_cache_down_addr_buffers[layer],
    };
    const NSUInteger off = (NSUInteger)expert * sizeof(uint64_t);
    for (uint32_t i = 0; i < 3; i++) {
        if (!buffers[i]) continue;
        uint64_t *addr = (uint64_t *)((uint8_t *)[buffers[i] contents] + off);
        *addr = 0;
        [buffers[i] didModifyRange:NSMakeRange(off, sizeof(uint64_t))];
    }
}

static int ds4_gpu_stream_expert_cache_set_addr_slot_raw(
        uint32_t      layer,
        uint32_t      expert,
        id<MTLBuffer> gate_buffer,
        NSUInteger    gate_inner,
        id<MTLBuffer> up_buffer,
        NSUInteger    up_inner,
        id<MTLBuffer> down_buffer,
        NSUInteger    down_inner) {
    if (!ds4_gpu_stream_expert_cache_ensure_addr_buffers(layer)) return 0;

    const uint64_t values[3] = {
        ds4_gpu_buffer_address(gate_buffer, gate_inner),
        ds4_gpu_buffer_address(up_buffer, up_inner),
        ds4_gpu_buffer_address(down_buffer, down_inner),
    };
    if (values[0] == 0 || values[1] == 0 || values[2] == 0) {
        fprintf(stderr, "ds4: Metal streaming expert address table requires GPU addresses\n");
        return 0;
    }

    id<MTLBuffer> buffers[3] = {
        g_stream_expert_cache_gate_addr_buffers[layer],
        g_stream_expert_cache_up_addr_buffers[layer],
        g_stream_expert_cache_down_addr_buffers[layer],
    };
    const NSUInteger off = (NSUInteger)expert * sizeof(uint64_t);
    for (uint32_t i = 0; i < 3; i++) {
        uint64_t *addr = (uint64_t *)((uint8_t *)[buffers[i] contents] + off);
        *addr = values[i];
        [buffers[i] didModifyRange:NSMakeRange(off, sizeof(uint64_t))];
    }
    return 1;
}

static int ds4_gpu_stream_expert_cache_set_addr_slot(
        uint32_t      layer,
        uint32_t      expert,
        id<MTLBuffer> gate_buffer,
        NSUInteger    gate_inner,
        id<MTLBuffer> up_buffer,
        NSUInteger    up_inner,
        id<MTLBuffer> down_buffer,
        NSUInteger    down_inner) {
    if (!ds4_gpu_stream_expert_addr_table_requested()) return 1;
    return ds4_gpu_stream_expert_cache_set_addr_slot_raw(layer,
                                                         expert,
                                                         gate_buffer,
                                                         gate_inner,
                                                         up_buffer,
                                                         up_inner,
                                                         down_buffer,
                                                         down_inner);
}

static int ds4_gpu_stream_expert_cache_addr_buffers(
        uint32_t       layer,
        id<MTLBuffer> *gate,
        id<MTLBuffer> *up,
        id<MTLBuffer> *down) {
    if (!ds4_gpu_stream_expert_cache_ensure_addr_buffers(layer)) return 0;
    if (gate) *gate = g_stream_expert_cache_gate_addr_buffers[layer];
    if (up) *up = g_stream_expert_cache_up_addr_buffers[layer];
    if (down) *down = g_stream_expert_cache_down_addr_buffers[layer];
    return g_stream_expert_cache_gate_addr_buffers[layer] &&
           g_stream_expert_cache_up_addr_buffers[layer] &&
           g_stream_expert_cache_down_addr_buffers[layer];
}

static void ds4_gpu_stream_full_expert_addr_clear_layer(uint32_t layer) {
    if (layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER) return;
    ds4_gpu_stream_expert_cache_entry *e = &g_stream_full_expert_addr_entry[layer];
    e->gate_buffer = nil;
    e->up_buffer = nil;
    e->down_buffer = nil;
    e->model_map = NULL;
    e->model_size = 0;
    e->gate_abs_offset = 0;
    e->up_abs_offset = 0;
    e->down_abs_offset = 0;
    e->gate_expert_bytes = 0;
    e->down_expert_bytes = 0;
    e->logical_bytes = 0;
    e->last_used = 0;
    e->use_count = 0;
    e->gate_inner = 0;
    e->up_inner = 0;
    e->down_inner = 0;
    e->slab_slot = 0;
    e->valid = 0;
    e->slab_backed = 0;
}

static int ds4_gpu_stream_full_expert_addr_table_prepare(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        uint32_t       n_total_expert,
        uint64_t       gate_abs_offset,
        uint64_t       up_abs_offset,
        uint64_t       down_abs_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        id<MTLBuffer> *gate_addrs,
        id<MTLBuffer> *up_addrs,
        id<MTLBuffer> *down_addrs,
        ds4_gpu_stream_expert_cache_entry **entry_out) {
    if (!ds4_gpu_stream_full_expert_addr_table_requested()) return 0;
    if (!model_map || model_size == 0 ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_total_expert == 0 ||
        n_total_expert > DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        n_total_expert > UINT64_MAX / gate_expert_bytes ||
        n_total_expert > UINT64_MAX / down_expert_bytes) {
        return 0;
    }

    const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_abs_offset > model_size ||
        up_abs_offset > model_size ||
        down_abs_offset > model_size ||
        gate_tensor_bytes > model_size - gate_abs_offset ||
        gate_tensor_bytes > model_size - up_abs_offset ||
        down_tensor_bytes > model_size - down_abs_offset) {
        return 0;
    }

    ds4_gpu_stream_expert_cache_entry *entry =
        &g_stream_full_expert_addr_entry[layer];
    if (!entry->valid ||
        entry->model_map != model_map ||
        entry->model_size != model_size ||
        entry->gate_abs_offset != gate_abs_offset ||
        entry->up_abs_offset != up_abs_offset ||
        entry->down_abs_offset != down_abs_offset ||
        entry->gate_expert_bytes != gate_expert_bytes ||
        entry->down_expert_bytes != down_expert_bytes ||
        !entry->gate_buffer ||
        !entry->up_buffer ||
        !entry->down_buffer) {
        ds4_gpu_stream_full_expert_addr_clear_layer(layer);

        uint64_t gate_inner = 0;
        uint64_t up_inner = 0;
        uint64_t down_inner = 0;
        id<MTLBuffer> gate_buf =
            ds4_gpu_wrap_model_exact_range_owned(model_map,
                                                 model_size,
                                                 gate_abs_offset,
                                                 gate_tensor_bytes,
                                                 &gate_inner);
        id<MTLBuffer> up_buf =
            ds4_gpu_wrap_model_exact_range_owned(model_map,
                                                 model_size,
                                                 up_abs_offset,
                                                 gate_tensor_bytes,
                                                 &up_inner);
        id<MTLBuffer> down_buf =
            ds4_gpu_wrap_model_exact_range_owned(model_map,
                                                 model_size,
                                                 down_abs_offset,
                                                 down_tensor_bytes,
                                                 &down_inner);
        if (!gate_buf || !up_buf || !down_buf) return 0;

        entry->gate_buffer = gate_buf;
        entry->up_buffer = up_buf;
        entry->down_buffer = down_buf;
        entry->model_map = model_map;
        entry->model_size = model_size;
        entry->gate_abs_offset = gate_abs_offset;
        entry->up_abs_offset = up_abs_offset;
        entry->down_abs_offset = down_abs_offset;
        entry->gate_expert_bytes = gate_expert_bytes;
        entry->down_expert_bytes = down_expert_bytes;
        entry->logical_bytes = gate_tensor_bytes * 2ull + down_tensor_bytes;
        entry->gate_inner = (NSUInteger)gate_inner;
        entry->up_inner = (NSUInteger)up_inner;
        entry->down_inner = (NSUInteger)down_inner;
        entry->valid = 1;

        if (!ds4_gpu_stream_expert_cache_ensure_addr_buffers(layer)) {
            ds4_gpu_stream_full_expert_addr_clear_layer(layer);
            return 0;
        }

        const uint64_t gate_base = ds4_gpu_buffer_address(entry->gate_buffer,
                                                          entry->gate_inner);
        const uint64_t up_base = ds4_gpu_buffer_address(entry->up_buffer,
                                                        entry->up_inner);
        const uint64_t down_base = ds4_gpu_buffer_address(entry->down_buffer,
                                                          entry->down_inner);
        if (gate_base == 0 || up_base == 0 || down_base == 0) {
            fprintf(stderr, "ds4: Metal full streaming expert address table requires GPU addresses\n");
            ds4_gpu_stream_full_expert_addr_clear_layer(layer);
            return 0;
        }

        id<MTLBuffer> buffers[3] = {
            g_stream_expert_cache_gate_addr_buffers[layer],
            g_stream_expert_cache_up_addr_buffers[layer],
            g_stream_expert_cache_down_addr_buffers[layer],
        };
        uint64_t *gate_addr = (uint64_t *)[buffers[0] contents];
        uint64_t *up_addr = (uint64_t *)[buffers[1] contents];
        uint64_t *down_addr = (uint64_t *)[buffers[2] contents];
        for (uint32_t expert = 0; expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; expert++) {
            if (expert < n_total_expert) {
                gate_addr[expert] = gate_base + (uint64_t)expert * gate_expert_bytes;
                up_addr[expert] = up_base + (uint64_t)expert * gate_expert_bytes;
                down_addr[expert] = down_base + (uint64_t)expert * down_expert_bytes;
            } else {
                gate_addr[expert] = 0;
                up_addr[expert] = 0;
                down_addr[expert] = 0;
            }
        }
        const NSUInteger bytes =
            (NSUInteger)DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT * sizeof(uint64_t);
        for (uint32_t i = 0; i < 3; i++) {
            [buffers[i] didModifyRange:NSMakeRange(0, bytes)];
        }
    }

    if (gate_addrs) *gate_addrs = g_stream_expert_cache_gate_addr_buffers[layer];
    if (up_addrs) *up_addrs = g_stream_expert_cache_up_addr_buffers[layer];
    if (down_addrs) *down_addrs = g_stream_expert_cache_down_addr_buffers[layer];
    if (entry_out) *entry_out = entry;
    return entry->valid &&
           g_stream_expert_cache_gate_addr_buffers[layer] &&
           g_stream_expert_cache_up_addr_buffers[layer] &&
           g_stream_expert_cache_down_addr_buffers[layer];
}

static id<MTLBuffer> ds4_gpu_stream_expert_validate_status_buffer(void) {
    if (!g_device) return nil;
    if (g_stream_expert_validate_status_buffer) {
        return g_stream_expert_validate_status_buffer;
    }

    const NSUInteger bytes =
        (NSUInteger)DS4_METAL_STREAM_EXPERT_VALIDATE_WORDS * sizeof(uint32_t);
    id<MTLBuffer> b = [g_device newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
    if (!b) {
        fprintf(stderr, "ds4: Metal streaming expert validator allocation failed\n");
        return nil;
    }
    b.label = @"ds4_stream_expert_validate_status";
    memset([b contents], 0, bytes);
    [b didModifyRange:NSMakeRange(0, bytes)];
    g_stream_expert_validate_status_buffer = b;
    return b;
}

static int ds4_gpu_encode_stream_expert_cache_validate(
        id<MTLCommandBuffer> cb,
        const ds4_gpu_stream_expert_validate_args *args,
        id<MTLBuffer> selected,
        NSUInteger selected_off,
        id<MTLBuffer> gate_addrs,
        id<MTLBuffer> up_addrs,
        id<MTLBuffer> down_addrs,
        id<MTLBuffer> status) {
    if (!cb || !args || !selected || !gate_addrs || !up_addrs || !down_addrs ||
        !status || !g_moe_stream_expert_cache_validate_pipeline ||
        args->n_total_expert == 0 || args->n_total_expert > 384 ||
        args->n_expert == 0 || args->n_expert > 6) {
        return 0;
    }

    id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
    [enc setComputePipelineState:g_moe_stream_expert_cache_validate_pipeline];
    [enc setBytes:args length:sizeof(*args) atIndex:0];
    [enc setBuffer:selected offset:selected_off atIndex:1];
    [enc setBuffer:gate_addrs offset:0 atIndex:2];
    [enc setBuffer:up_addrs offset:0 atIndex:3];
    [enc setBuffer:down_addrs offset:0 atIndex:4];
    [enc setBuffer:status offset:0 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    ds4_gpu_end_compute_encoder(cb, enc);
    return 1;
}

static int ds4_gpu_stream_expert_cache_validate_selected(
        const ds4_gpu_tensor *selected,
        id<MTLBuffer> gate_addrs,
        id<MTLBuffer> up_addrs,
        id<MTLBuffer> down_addrs,
        uint32_t n_total_expert,
        uint32_t n_expert,
        int32_t selected_ids[6],
        uint32_t *all_cached,
        uint32_t *miss_mask,
        uint32_t *invalid_mask) {
    if (!selected || !selected_ids || !all_cached || !miss_mask || !invalid_mask) {
        return 0;
    }
    id<MTLBuffer> selectedbuf = ds4_gpu_tensor_buffer(selected);
    id<MTLBuffer> status = ds4_gpu_stream_expert_validate_status_buffer();
    if (!selectedbuf || !status) return 0;

    const NSUInteger status_bytes =
        (NSUInteger)DS4_METAL_STREAM_EXPERT_VALIDATE_WORDS * sizeof(uint32_t);
    memset([status contents], 0, status_bytes);
    [status didModifyRange:NSMakeRange(0, status_bytes)];

    ds4_gpu_stream_expert_validate_args args = {
        .n_total_expert = n_total_expert,
        .n_expert = n_expert,
    };

    const int had_batch = g_batch_cb != nil;
    int owned = 0;
    id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
    if (!cb) return 0;
    if (!ds4_gpu_encode_stream_expert_cache_validate(cb,
                                                     &args,
                                                     selectedbuf,
                                                     ds4_gpu_tensor_offset(selected),
                                                     gate_addrs,
                                                     up_addrs,
                                                     down_addrs,
                                                     status)) {
        return 0;
    }

    if (had_batch) {
        if (ds4_gpu_end_commands() == 0) return 0;
    } else if (!ds4_gpu_finish_command_buffer(cb, owned,
                                              "streaming expert cache validator")) {
        return 0;
    }

    const uint32_t *words = (const uint32_t *)[status contents];
    *all_cached = words[0];
    *miss_mask = words[1];
    *invalid_mask = words[2];
    for (uint32_t i = 0; i < 6; i++) {
        selected_ids[i] = (int32_t)words[4 + i];
    }

    if (had_batch && ds4_gpu_begin_commands() == 0) return 0;
    return 1;
}

static void ds4_gpu_stream_expert_cache_clear_entry_internal(
        uint32_t                                  layer,
        uint32_t                                  expert,
        int                                       count_eviction,
        int                                       recycle_slab_slot,
        ds4_gpu_stream_expert_reusable_buffers   *reuse) {
    if (layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        expert >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) {
        return;
    }
    if (reuse) {
        reuse->gate_buffer = nil;
        reuse->up_buffer = nil;
        reuse->down_buffer = nil;
        reuse->gate_inner = 0;
        reuse->up_inner = 0;
        reuse->down_inner = 0;
    }
    ds4_gpu_stream_expert_cache_entry *e =
        &g_stream_expert_cache[layer][expert];
    if (!e->valid) return;
    if (ds4_gpu_stream_expert_cache_entry_inflight(e)) {
        return;
    }

    const uint64_t bytes = e->logical_bytes;
    ds4_gpu_stream_expert_evict_dontneed_range(e->model_map,
                                               e->model_size,
                                               e->gate_abs_offset,
                                               e->gate_expert_bytes);
    ds4_gpu_stream_expert_evict_dontneed_range(e->model_map,
                                               e->model_size,
                                               e->up_abs_offset,
                                               e->gate_expert_bytes);
    ds4_gpu_stream_expert_evict_dontneed_range(e->model_map,
                                               e->model_size,
                                               e->down_abs_offset,
                                               e->down_expert_bytes);
    ds4_gpu_stream_expert_cache_zero_addr_slot(layer, expert);
    if (reuse) {
        reuse->gate_buffer = e->gate_buffer;
        reuse->up_buffer = e->up_buffer;
        reuse->down_buffer = e->down_buffer;
        reuse->gate_inner = e->gate_inner;
        reuse->up_inner = e->up_inner;
        reuse->down_inner = e->down_inner;
    } else if (e->slab_backed && recycle_slab_slot) {
        ds4_gpu_stream_expert_slab_push_free_slot(e->slab_slot);
    }
    e->gate_buffer = nil;
    e->up_buffer = nil;
    e->down_buffer = nil;
    e->model_map = NULL;
    e->model_size = 0;
    e->gate_abs_offset = 0;
    e->up_abs_offset = 0;
    e->down_abs_offset = 0;
    e->gate_expert_bytes = 0;
    e->down_expert_bytes = 0;
    e->logical_bytes = 0;
    e->last_used = 0;
    e->use_count = 0;
    e->gate_inner = 0;
    e->up_inner = 0;
    e->down_inner = 0;
    e->inflight_seq = 0;
    e->slab_slot = 0;
    e->valid = 0;
    e->slab_backed = 0;

    if (g_stream_expert_cache_layer_count[layer] > 0) {
        g_stream_expert_cache_layer_count[layer]--;
    }
    if (g_stream_expert_cache_entry_count > 0) {
        g_stream_expert_cache_entry_count--;
    }
    if (g_stream_expert_cache_bytes >= bytes) {
        g_stream_expert_cache_bytes -= bytes;
    } else {
        g_stream_expert_cache_bytes = 0;
    }
    if (count_eviction) {
        g_stream_expert_cache_evictions++;
        g_stream_expert_cache_layer_evictions[layer]++;
    }
}

static void ds4_gpu_stream_expert_cache_clear_entry(
        uint32_t layer,
        uint32_t expert,
        int      count_eviction) {
    ds4_gpu_stream_expert_cache_clear_entry_internal(layer,
                                                     expert,
                                                     count_eviction,
                                                     1,
                                                     NULL);
}

static void ds4_gpu_stream_expert_cache_clear_all(int reset_stats) {
    ds4_gpu_stream_expert_pending_load_clear();
    g_stream_expert_cache_done_seq = g_stream_expert_cache_cb_seq;
    g_stream_expert_cache_batch_seq = 0;
    g_stream_expert_cache_owned_seq = 0;
    g_stream_expert_cache_pending_max_seq = 0;
    for (uint32_t layer = 0; layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; layer++) {
        for (uint32_t expert = 0; expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; expert++) {
            g_stream_expert_cache[layer][expert].inflight_seq = 0;
            ds4_gpu_stream_expert_cache_clear_entry(layer, expert, 0);
        }
        ds4_gpu_stream_full_expert_addr_clear_layer(layer);
        g_stream_expert_cache_layer_count[layer] = 0;
        id<MTLBuffer> buffers[3] = {
            g_stream_expert_cache_gate_addr_buffers[layer],
            g_stream_expert_cache_up_addr_buffers[layer],
            g_stream_expert_cache_down_addr_buffers[layer],
        };
        for (uint32_t i = 0; i < 3; i++) {
            if (!buffers[i]) continue;
            const NSUInteger bytes =
                (NSUInteger)DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT * sizeof(uint64_t);
            memset([buffers[i] contents], 0, bytes);
            [buffers[i] didModifyRange:NSMakeRange(0, bytes)];
        }
    }
    g_stream_expert_cache_bytes = 0;
    g_stream_expert_cache_entry_count = 0;
    for (uint32_t i = 0; i < g_stream_expert_cache_slab_count; i++) {
        g_stream_expert_cache_slabs[i] = nil;
        g_stream_expert_cache_slab_start_slot[i] = 0;
        g_stream_expert_cache_slab_slot_count[i] = 0;
        g_stream_expert_cache_slab_slots_used[i] = 0;
    }
    g_stream_expert_cache_slab_count = 0;
    g_stream_expert_cache_slab_total_slots = 0;
    g_stream_expert_cache_free_slot_count = 0;
    g_stream_expert_cache_slab_slot_bytes = 0;
    if (reset_stats) {
        g_stream_expert_cache_hits = 0;
        g_stream_expert_cache_misses = 0;
        g_stream_expert_cache_evictions = 0;
        g_stream_expert_cache_wraps = 0;
        g_stream_expert_cache_clock = 0;
        g_stream_expert_cache_evict_advise_bytes = 0;
        g_stream_expert_cache_willneed_advise_bytes = 0;
        g_stream_expert_cache_pread_bytes = 0;
        g_stream_expert_cache_pread_ms = 0.0;
        g_stream_expert_cache_buffer_allocs = 0;
        g_stream_expert_cache_buffer_reuses = 0;
        g_stream_expert_cache_decode_tokens = 0;
        g_stream_expert_cache_hotness_decay_token = 0;
        memset(g_stream_expert_cache_route_hotness,
               0,
               sizeof(g_stream_expert_cache_route_hotness));
        g_stream_expert_timing_selected_calls = 0;
        g_stream_expert_timing_selected_read_ms = 0.0;
        g_stream_expert_timing_selected_sync_ms = 0.0;
        g_stream_expert_timing_selected_copy_ms = 0.0;
        g_stream_expert_timing_selected_bind_ms = 0.0;
        g_stream_expert_timing_split_layers = 0;
        g_stream_expert_timing_split_resident_experts = 0;
        g_stream_expert_timing_split_missing_experts = 0;
        g_stream_expert_timing_split_resident_ms = 0.0;
        g_stream_expert_timing_split_missing_ms = 0.0;
        g_stream_expert_timing_split_missing_load_ms = 0.0;
        g_stream_expert_timing_split_missing_slot_ms = 0.0;
        g_stream_expert_timing_split_missing_prune_ms = 0.0;
        g_stream_expert_timing_split_missing_addr_ms = 0.0;
        g_stream_expert_timing_split_missing_wait_ms = 0.0;
        g_stream_expert_timing_load_calls = 0;
        g_stream_expert_timing_load_prepare_ms = 0.0;
        g_stream_expert_timing_load_pread_ms = 0.0;
        g_stream_expert_timing_load_modify_ms = 0.0;
        g_stream_expert_timing_load_install_ms = 0.0;
        g_stream_expert_timing_prepare_batch_reuse_calls = 0;
        g_stream_expert_timing_prepare_batch_reuse_ms = 0.0;
        g_stream_expert_timing_prepare_buffer_calls = 0;
        g_stream_expert_timing_prepare_buffer_ms = 0.0;
        g_stream_expert_timing_prepare_task_experts = 0;
        g_stream_expert_timing_prepare_task_ms = 0.0;
        g_stream_expert_timing_reuse_scan_calls = 0;
        g_stream_expert_timing_reuse_scan_entries = 0;
        g_stream_expert_timing_reuse_scan_ms = 0.0;
        g_stream_expert_timing_reuse_clear_ms = 0.0;
        g_stream_expert_timing_readahead_calls = 0;
        g_stream_expert_timing_readahead_bytes = 0;
        g_stream_expert_timing_readahead_ms = 0.0;
        g_stream_expert_timing_cache_all_resident_layers = 0;
        g_stream_expert_timing_cache_all_missing_layers = 0;
        g_stream_expert_timing_cache_mixed_layers = 0;
        g_stream_expert_timing_cache_resident_experts = 0;
        g_stream_expert_timing_cache_missing_experts = 0;
        g_stream_expert_timing_last_report =
            (ds4_gpu_stream_expert_timing_snapshot){0};
        memset(g_stream_expert_cache_layer_hits,
               0,
               sizeof(g_stream_expert_cache_layer_hits));
        memset(g_stream_expert_cache_layer_misses,
               0,
               sizeof(g_stream_expert_cache_layer_misses));
        memset(g_stream_expert_cache_layer_evictions,
               0,
               sizeof(g_stream_expert_cache_layer_evictions));
        memset(g_stream_expert_cache_layer_pread_bytes,
               0,
               sizeof(g_stream_expert_cache_layer_pread_bytes));
        memset(g_stream_expert_cache_layer_pread_ms,
               0,
               sizeof(g_stream_expert_cache_layer_pread_ms));
    }
}

static int ds4_gpu_stream_expert_cache_is_protected(
        uint32_t expert,
        const int32_t *protect_ids,
        uint32_t n_protect) {
    if (!protect_ids) return 0;
    for (uint32_t i = 0; i < n_protect; i++) {
        if (protect_ids[i] >= 0 && (uint32_t)protect_ids[i] == expert) {
            return 1;
        }
    }
    return 0;
}

static void ds4_gpu_stream_expert_cache_prune_layer(
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        const int32_t *protect_ids,
        uint32_t n_protect) {
    if (layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER) return;
    uint32_t cap = ds4_gpu_stream_expert_cache_effective_cap(layer,
                                                             n_total_expert,
                                                             n_selected);
    if (cap == 0) return;

    /*
     * Route hotness counts selected experts even when they miss.  Hit-count
     * LFU penalizes experts that are repeatedly selected but evicted before a
     * second hit, which keeps too many decode layers in the mixed-cache path.
     */
    while (g_stream_expert_cache_layer_count[layer] > cap) {
        uint32_t victim = UINT32_MAX;
        uint32_t lowest_hotness = UINT32_MAX;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t expert = 0; expert < n_total_expert; expert++) {
            ds4_gpu_stream_expert_cache_entry *e =
                &g_stream_expert_cache[layer][expert];
            if (!e->valid ||
                ds4_gpu_stream_expert_cache_entry_inflight(e) ||
                ds4_gpu_stream_expert_cache_is_protected(expert, protect_ids, n_protect)) {
                continue;
            }
            const uint32_t hotness =
                g_stream_expert_cache_route_hotness[layer][expert];
            if (hotness < lowest_hotness ||
                (hotness == lowest_hotness && e->last_used < oldest)) {
                lowest_hotness = hotness;
                oldest = e->last_used;
                victim = expert;
            }
        }
        if (victim == UINT32_MAX) break;
        ds4_gpu_stream_expert_cache_clear_entry(layer, victim, 1);
    }
}

static int ds4_gpu_stream_expert_cache_entry_protected(
        uint32_t layer,
        uint32_t expert,
        uint32_t protect_layer,
        const int32_t *protect_ids,
        uint32_t n_protect) {
    if (layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER &&
        expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT &&
        ds4_gpu_stream_expert_cache_entry_inflight(
                &g_stream_expert_cache[layer][expert])) {
        return 1;
    }
    return layer == protect_layer &&
           ds4_gpu_stream_expert_cache_is_protected(expert,
                                                    protect_ids,
                                                    n_protect);
}

static int ds4_gpu_stream_expert_cache_entry_reusable(
        const ds4_gpu_stream_expert_cache_entry *e,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    return e &&
           e->valid &&
           e->gate_buffer &&
           e->up_buffer &&
           e->down_buffer &&
           e->gate_expert_bytes == gate_expert_bytes &&
           e->down_expert_bytes == down_expert_bytes;
}

static int ds4_gpu_stream_expert_cache_take_reusable(
        int                                     force_reuse,
        uint32_t                                protect_layer,
        const int32_t                          *protect_ids,
        uint32_t                                n_protect,
        uint64_t                                gate_expert_bytes,
        uint64_t                                down_expert_bytes,
        ds4_gpu_stream_expert_reusable_buffers *reuse) {
    if (!reuse) return 0;
    reuse->gate_buffer = nil;
    reuse->up_buffer = nil;
    reuse->down_buffer = nil;
    reuse->gate_inner = 0;
    reuse->up_inner = 0;
    reuse->down_inner = 0;

    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    if (budget == 0 ||
        (!force_reuse && g_stream_expert_cache_entry_count < budget)) {
        return 0;
    }

    int waited_inflight = 0;
retry:
    ;
    uint32_t victim_layer = UINT32_MAX;
    uint32_t victim_expert = UINT32_MAX;
    uint32_t lowest_hotness = UINT32_MAX;
    uint64_t oldest = UINT64_MAX;
    int skipped_inflight = 0;
    const int timing = ds4_gpu_stream_expert_timing_summary_enabled();
    const double scan_t0 = timing ? ds4_gpu_now_ms() : 0.0;
    uint64_t scan_entries = 0;
    for (uint32_t layer = 0;
         layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
         layer++) {
        for (uint32_t expert = 0;
             expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
             expert++) {
            scan_entries++;
            ds4_gpu_stream_expert_cache_entry *e =
                &g_stream_expert_cache[layer][expert];
            if (!ds4_gpu_stream_expert_cache_entry_reusable(e,
                                                            gate_expert_bytes,
                                                            down_expert_bytes)) {
                continue;
            }
            if (ds4_gpu_stream_expert_cache_entry_inflight(e)) {
                skipped_inflight = 1;
                continue;
            }
            if (ds4_gpu_stream_expert_cache_entry_protected(layer,
                                                            expert,
                                                            protect_layer,
                                                            protect_ids,
                                                            n_protect)) {
                continue;
            }
            const uint32_t hotness =
                g_stream_expert_cache_route_hotness[layer][expert];
            if (hotness < lowest_hotness ||
                (hotness == lowest_hotness && e->last_used < oldest)) {
                lowest_hotness = hotness;
                oldest = e->last_used;
                victim_layer = layer;
                victim_expert = expert;
            }
        }
    }
    if (timing) {
        ds4_gpu_stream_expert_timing_note_reuse_scan(scan_entries,
                                                     ds4_gpu_now_ms() - scan_t0);
    }

    if (victim_layer == UINT32_MAX || victim_expert == UINT32_MAX) {
        if (skipped_inflight && !waited_inflight &&
            !ds4_gpu_stream_expert_cache_on_service_thread()) {
            waited_inflight = 1;
            if (!ds4_gpu_stream_expert_cache_wait_inflight(
                    "streaming expert cache reuse")) {
                return 0;
            }
            goto retry;
        }
        return 0;
    }
    const double clear_t0 = timing ? ds4_gpu_now_ms() : 0.0;
    ds4_gpu_stream_expert_cache_clear_entry_internal(victim_layer,
                                                     victim_expert,
                                                     1,
                                                     1,
                                                     reuse);
    if (timing) {
        ds4_gpu_stream_expert_timing_note_reuse_clear(ds4_gpu_now_ms() -
                                                      clear_t0);
    }
    if (!reuse->gate_buffer || !reuse->up_buffer || !reuse->down_buffer) {
        reuse->gate_buffer = nil;
        reuse->up_buffer = nil;
        reuse->down_buffer = nil;
        reuse->gate_inner = 0;
        reuse->up_inner = 0;
        reuse->down_inner = 0;
        return 0;
    }
    g_stream_expert_cache_buffer_reuses +=
        ds4_gpu_stream_expert_buffer_object_count(reuse->gate_buffer,
                                                  reuse->up_buffer,
                                                  reuse->down_buffer);
    return 1;
}

static uint32_t ds4_gpu_stream_expert_cache_take_reusable_batch(
        uint32_t                                n_needed,
        uint32_t                                protect_layer,
        const int32_t                          *protect_ids,
        uint32_t                                n_protect,
        uint64_t                                gate_expert_bytes,
        uint64_t                                down_expert_bytes,
        ds4_gpu_stream_expert_reusable_buffers *reuses) {
    if (!reuses || n_needed == 0) return 0;
    if (n_needed > DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED) {
        n_needed = DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED;
    }
    for (uint32_t i = 0; i < n_needed; i++) {
        reuses[i].gate_buffer = nil;
        reuses[i].up_buffer = nil;
        reuses[i].down_buffer = nil;
        reuses[i].gate_inner = 0;
        reuses[i].up_inner = 0;
        reuses[i].down_inner = 0;
    }

    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    if (budget == 0 || g_stream_expert_cache_entry_count < budget) {
        return 0;
    }

    int waited_inflight = 0;
retry:
    ;
    uint32_t victim_layers[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint32_t victim_experts[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint32_t victim_hotness[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint64_t victim_last_used[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint32_t victim_count = 0;
    int skipped_inflight = 0;
    const int timing = ds4_gpu_stream_expert_timing_summary_enabled();
    const double scan_t0 = timing ? ds4_gpu_now_ms() : 0.0;
    uint64_t scan_entries = 0;
    for (uint32_t i = 0; i < n_needed; i++) {
        victim_layers[i] = UINT32_MAX;
        victim_experts[i] = UINT32_MAX;
        victim_hotness[i] = UINT32_MAX;
        victim_last_used[i] = UINT64_MAX;
    }

    for (uint32_t layer = 0;
         layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
         layer++) {
        for (uint32_t expert = 0;
             expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
             expert++) {
            scan_entries++;
            ds4_gpu_stream_expert_cache_entry *e =
                &g_stream_expert_cache[layer][expert];
            if (!ds4_gpu_stream_expert_cache_entry_reusable(e,
                                                            gate_expert_bytes,
                                                            down_expert_bytes)) {
                continue;
            }
            if (ds4_gpu_stream_expert_cache_entry_inflight(e)) {
                skipped_inflight = 1;
                continue;
            }
            if (ds4_gpu_stream_expert_cache_entry_protected(layer,
                                                            expert,
                                                            protect_layer,
                                                            protect_ids,
                                                            n_protect)) {
                continue;
            }

            const uint32_t hotness =
                g_stream_expert_cache_route_hotness[layer][expert];
            const uint64_t last_used = e->last_used;
            if (victim_count < n_needed) {
                victim_layers[victim_count] = layer;
                victim_experts[victim_count] = expert;
                victim_hotness[victim_count] = hotness;
                victim_last_used[victim_count] = last_used;
                victim_count++;
                continue;
            }

            uint32_t worst = 0;
            for (uint32_t i = 1; i < victim_count; i++) {
                if (victim_hotness[i] > victim_hotness[worst] ||
                    (victim_hotness[i] == victim_hotness[worst] &&
                     victim_last_used[i] > victim_last_used[worst])) {
                    worst = i;
                }
            }
            if (hotness < victim_hotness[worst] ||
                (hotness == victim_hotness[worst] &&
                 last_used < victim_last_used[worst])) {
                victim_layers[worst] = layer;
                victim_experts[worst] = expert;
                victim_hotness[worst] = hotness;
                victim_last_used[worst] = last_used;
            }
        }
    }
    if (timing) {
        ds4_gpu_stream_expert_timing_note_reuse_scan(scan_entries,
                                                     ds4_gpu_now_ms() - scan_t0);
    }

    if (victim_count == 0) {
        if (skipped_inflight && !waited_inflight) {
            waited_inflight = 1;
            if (!ds4_gpu_stream_expert_cache_wait_inflight(
                    "streaming expert cache batch reuse")) {
                return 0;
            }
            goto retry;
        }
        return 0;
    }

    uint32_t reuse_count = 0;
    const double clear_t0 = timing ? ds4_gpu_now_ms() : 0.0;
    for (uint32_t i = 0; i < victim_count; i++) {
        if (victim_layers[i] == UINT32_MAX ||
            victim_experts[i] == UINT32_MAX) {
            continue;
        }
        ds4_gpu_stream_expert_cache_clear_entry_internal(victim_layers[i],
                                                         victim_experts[i],
                                                         1,
                                                         1,
                                                         &reuses[reuse_count]);
        if (!reuses[reuse_count].gate_buffer ||
            !reuses[reuse_count].up_buffer ||
            !reuses[reuse_count].down_buffer) {
            reuses[reuse_count].gate_buffer = nil;
            reuses[reuse_count].up_buffer = nil;
            reuses[reuse_count].down_buffer = nil;
            reuses[reuse_count].gate_inner = 0;
            reuses[reuse_count].up_inner = 0;
            reuses[reuse_count].down_inner = 0;
            continue;
        }
        g_stream_expert_cache_buffer_reuses +=
            ds4_gpu_stream_expert_buffer_object_count(
                    reuses[reuse_count].gate_buffer,
                    reuses[reuse_count].up_buffer,
                    reuses[reuse_count].down_buffer);
        reuse_count++;
    }
    if (timing) {
        ds4_gpu_stream_expert_timing_note_reuse_clear(ds4_gpu_now_ms() -
                                                      clear_t0);
    }
    return reuse_count;
}

static int ds4_gpu_stream_expert_batch_reuse_enabled(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        return 0;
    }
    const uint64_t slot_bytes = gate_expert_bytes * 2ull + down_expert_bytes;
    /*
     * One global victim scan per selected miss is measurable for GLM Q2-size
     * slots, but batching larger Q4-size slots regressed short decode on M5.
     * Keep larger slots on the older single-victim path until profiling says
     * otherwise.
     */
    return slot_bytes <= 16ull * 1024ull * 1024ull;
}

static int ds4_gpu_stream_expert_cache_prepare_load_buffers(
        uint32_t       layer,
        uint32_t       expert,
        uint32_t       protect_layer,
        const int32_t *protect_ids,
        uint32_t       n_protect,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        int            force_reuse,
        __strong id<MTLBuffer> *gate_buf,
        __strong id<MTLBuffer> *up_buf,
        __strong id<MTLBuffer> *down_buf,
        NSUInteger    *gate_inner,
        NSUInteger    *up_inner,
        NSUInteger    *down_inner) {
    if (!gate_buf || !up_buf || !down_buf ||
        !gate_inner || !up_inner || !down_inner ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        expert >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) {
        return 0;
    }

    *gate_buf = nil;
    *up_buf = nil;
    *down_buf = nil;
    *gate_inner = 0;
    *up_inner = 0;
    *down_inner = 0;

    ds4_gpu_stream_expert_reusable_buffers reuse = { nil, nil, nil, 0, 0, 0 };
    ds4_gpu_stream_expert_cache_entry *e =
        &g_stream_expert_cache[layer][expert];
    if (e->valid && ds4_gpu_stream_expert_cache_entry_inflight(e)) {
        if (ds4_gpu_stream_expert_cache_on_service_thread()) return 0;
        if (!ds4_gpu_stream_expert_cache_wait_inflight(
                "streaming expert cache replacement")) {
            return 0;
        }
        e = &g_stream_expert_cache[layer][expert];
    }
    if (ds4_gpu_stream_expert_cache_entry_reusable(e,
                                                   gate_expert_bytes,
                                                   down_expert_bytes)) {
        ds4_gpu_stream_expert_cache_clear_entry_internal(layer,
                                                         expert,
                                                         0,
                                                         1,
                                                         &reuse);
        if (reuse.gate_buffer && reuse.up_buffer && reuse.down_buffer) {
            g_stream_expert_cache_buffer_reuses +=
                ds4_gpu_stream_expert_buffer_object_count(reuse.gate_buffer,
                                                          reuse.up_buffer,
                                                          reuse.down_buffer);
        }
    } else if (e->valid) {
        ds4_gpu_stream_expert_cache_clear_entry(layer, expert, 0);
    }

    if (!reuse.gate_buffer || !reuse.up_buffer || !reuse.down_buffer) {
        if (!ds4_gpu_stream_expert_cache_take_reusable(force_reuse,
                                                       protect_layer,
                                                       protect_ids,
                                                       n_protect,
                                                       gate_expert_bytes,
                                                       down_expert_bytes,
                                                       &reuse)) {
            reuse.gate_buffer = nil;
            reuse.up_buffer = nil;
            reuse.down_buffer = nil;
        }
    }

    if (reuse.gate_buffer && reuse.up_buffer && reuse.down_buffer) {
        *gate_buf = reuse.gate_buffer;
        *up_buf = reuse.up_buffer;
        *down_buf = reuse.down_buffer;
        *gate_inner = reuse.gate_inner;
        *up_inner = reuse.up_inner;
        *down_inner = reuse.down_inner;
        return 1;
    }

    if (ds4_gpu_stream_expert_combined_buffer_enabled()) {
        if (gate_expert_bytes > UINT64_MAX - gate_expert_bytes ||
            gate_expert_bytes * 2ull > UINT64_MAX - down_expert_bytes ||
            gate_expert_bytes > (uint64_t)NSUIntegerMax ||
            gate_expert_bytes * 2ull > (uint64_t)NSUIntegerMax ||
            gate_expert_bytes * 2ull + down_expert_bytes >
                (uint64_t)NSUIntegerMax) {
            return 0;
        }
        if (ds4_gpu_stream_expert_alloc_slab_slot(gate_expert_bytes,
                                                  down_expert_bytes,
                                                  gate_buf,
                                                  up_buf,
                                                  down_buf,
                                                  gate_inner,
                                                  up_inner,
                                                  down_inner)) {
            return 1;
        }
        const uint64_t up_off = gate_expert_bytes;
        const uint64_t down_off = gate_expert_bytes * 2ull;
        const uint64_t combined_bytes = down_off + down_expert_bytes;
        id<MTLBuffer> combined =
            ds4_gpu_stream_expert_alloc_buffer(combined_bytes,
                                               @"ds4_stream_expert_combined");
        if (!combined) return 0;
        *gate_buf = combined;
        *up_buf = combined;
        *down_buf = combined;
        *gate_inner = 0;
        *up_inner = (NSUInteger)up_off;
        *down_inner = (NSUInteger)down_off;
        return 1;
    }

    *gate_buf = ds4_gpu_stream_expert_alloc_buffer(gate_expert_bytes,
                                                   @"ds4_stream_expert_gate");
    *up_buf = ds4_gpu_stream_expert_alloc_buffer(gate_expert_bytes,
                                                 @"ds4_stream_expert_up");
    *down_buf = ds4_gpu_stream_expert_alloc_buffer(down_expert_bytes,
                                                   @"ds4_stream_expert_down");
    return *gate_buf && *up_buf && *down_buf;
}

static void ds4_gpu_stream_expert_cache_prune_global(
        uint32_t protect_layer,
        const int32_t *protect_ids,
        uint32_t n_protect) {
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    if (budget == 0 || g_stream_expert_cache_entry_count <= budget) return;

    while (g_stream_expert_cache_entry_count > budget) {
        uint32_t victim_layer = UINT32_MAX;
        uint32_t victim_expert = UINT32_MAX;
        uint32_t lowest_hotness = UINT32_MAX;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t layer = 0;
             layer < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
             layer++) {
            for (uint32_t expert = 0;
                 expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
                 expert++) {
                ds4_gpu_stream_expert_cache_entry *e =
                    &g_stream_expert_cache[layer][expert];
                if (!e->valid ||
                    ds4_gpu_stream_expert_cache_entry_protected(layer,
                                                                expert,
                                                                protect_layer,
                                                                protect_ids,
                                                                n_protect)) {
                    continue;
                }
                const uint32_t hotness =
                    g_stream_expert_cache_route_hotness[layer][expert];
                if (hotness < lowest_hotness ||
                    (hotness == lowest_hotness && e->last_used < oldest)) {
                    lowest_hotness = hotness;
                    oldest = e->last_used;
                    victim_layer = layer;
                    victim_expert = expert;
                }
            }
        }
        if (victim_layer == UINT32_MAX || victim_expert == UINT32_MAX) break;
        ds4_gpu_stream_expert_cache_clear_entry(victim_layer, victim_expert, 1);
    }
}

static int ds4_gpu_stream_expert_cache_entry_matches(
        const ds4_gpu_stream_expert_cache_entry *e,
        const void *model_map,
        uint64_t    model_size,
        uint64_t    gate_abs_offset,
        uint64_t    up_abs_offset,
        uint64_t    down_abs_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes) {
    return e &&
           e->valid &&
           e->model_map == model_map &&
           e->model_size == model_size &&
           e->gate_abs_offset == gate_abs_offset &&
           e->up_abs_offset == up_abs_offset &&
           e->down_abs_offset == down_abs_offset &&
           e->gate_expert_bytes == gate_expert_bytes &&
           e->down_expert_bytes == down_expert_bytes &&
           e->gate_buffer && e->up_buffer && e->down_buffer;
}

static ds4_gpu_stream_expert_cache_entry *ds4_gpu_stream_expert_cache_peek(
        const void *model_map,
        uint64_t    model_size,
        uint32_t    layer,
        uint32_t    expert,
        uint32_t    n_total_expert,
        uint32_t    n_selected,
        uint64_t    gate_abs_offset,
        uint64_t    up_abs_offset,
        uint64_t    down_abs_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes) {
    if (!g_ssd_streaming_mode ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        expert >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        expert >= n_total_expert ||
        !ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                      down_expert_bytes) ||
        ds4_gpu_stream_expert_cache_effective_cap(layer,
                                                  n_total_expert,
                                                  n_selected) == 0) {
        return NULL;
    }

    ds4_gpu_stream_expert_cache_entry *e =
        &g_stream_expert_cache[layer][expert];
    if (!ds4_gpu_stream_expert_cache_entry_matches(e,
                                                   model_map,
                                                   model_size,
                                                   gate_abs_offset,
                                                   up_abs_offset,
                                                   down_abs_offset,
                                                   gate_expert_bytes,
                                                   down_expert_bytes)) {
        return NULL;
    }

    e->last_used = ++g_stream_expert_cache_clock;
    e->use_count++;
    g_stream_expert_cache_hits++;
    g_stream_expert_cache_layer_hits[layer]++;
    return e;
}

static ds4_gpu_stream_expert_cache_entry *
ds4_gpu_stream_expert_cache_install_loaded(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        uint32_t       expert,
        uint64_t       gate_abs_offset,
        uint64_t       up_abs_offset,
        uint64_t       down_abs_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        id<MTLBuffer>  gate_buf,
        id<MTLBuffer>  up_buf,
        id<MTLBuffer>  down_buf,
        NSUInteger     gate_inner,
        NSUInteger     up_inner,
        NSUInteger     down_inner) {
    if (!gate_buf || !up_buf || !down_buf ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        expert >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) {
        return NULL;
    }
    if (gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        fprintf(stderr, "ds4: Metal streaming expert cache byte size overflow\n");
        return NULL;
    }
    const uint64_t logical_bytes = gate_expert_bytes * 2ull + down_expert_bytes;

    ds4_gpu_stream_expert_cache_entry *e =
        &g_stream_expert_cache[layer][expert];
    if (e->valid) {
        if (ds4_gpu_stream_expert_cache_entry_inflight(e) &&
            ds4_gpu_stream_expert_cache_on_service_thread()) {
            return NULL;
        }
        if (ds4_gpu_stream_expert_cache_entry_inflight(e) &&
            !ds4_gpu_stream_expert_cache_wait_inflight(
                    "streaming expert cache install")) {
            return NULL;
        }
        if (ds4_gpu_stream_expert_cache_entry_inflight(e)) return NULL;
        ds4_gpu_stream_expert_cache_clear_entry(layer, expert, 0);
        if (e->valid) return NULL;
    }

    if (!ds4_gpu_stream_expert_cache_set_addr_slot(layer,
                                                   expert,
                                                   gate_buf,
                                                   gate_inner,
                                                   up_buf,
                                                   up_inner,
                                                   down_buf,
                                                   down_inner)) {
        return NULL;
    }

    e->gate_buffer = gate_buf;
    e->up_buffer = up_buf;
    e->down_buffer = down_buf;
    e->model_map = model_map;
    e->model_size = model_size;
    e->gate_abs_offset = gate_abs_offset;
    e->up_abs_offset = up_abs_offset;
    e->down_abs_offset = down_abs_offset;
    e->gate_expert_bytes = gate_expert_bytes;
    e->down_expert_bytes = down_expert_bytes;
    e->logical_bytes = logical_bytes;
    e->last_used = ++g_stream_expert_cache_clock;
    e->use_count = 1;
    e->gate_inner = gate_inner;
    e->up_inner = up_inner;
    e->down_inner = down_inner;
    e->inflight_seq = 0;
    uint32_t slab_slot = 0;
    if (gate_buf == up_buf &&
        gate_buf == down_buf &&
        ds4_gpu_stream_expert_slab_slot_for_buffer(gate_buf,
                                                   gate_inner,
                                                   &slab_slot)) {
        e->slab_backed = 1;
        e->slab_slot = slab_slot;
    } else {
        e->slab_backed = 0;
        e->slab_slot = 0;
    }
    e->valid = 1;
    g_stream_expert_cache_layer_count[layer]++;
    if (g_stream_expert_cache_entry_count < UINT32_MAX) {
        g_stream_expert_cache_entry_count++;
    }
    if (g_stream_expert_cache_bytes > UINT64_MAX - logical_bytes) {
        g_stream_expert_cache_bytes = UINT64_MAX;
    } else {
        g_stream_expert_cache_bytes += logical_bytes;
    }
    g_stream_expert_cache_misses++;
    g_stream_expert_cache_layer_misses[layer]++;
    g_stream_expert_cache_wraps += 3;
    return e;
}

static ds4_gpu_stream_expert_cache_entry *ds4_gpu_stream_expert_cache_get_protected(
        const void *model_map,
        uint64_t    model_size,
        uint32_t    layer,
        uint32_t    expert,
        uint32_t    n_total_expert,
        uint32_t    n_selected,
        uint64_t    gate_abs_offset,
        uint64_t    up_abs_offset,
        uint64_t    down_abs_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        const int32_t *protect_ids,
        uint32_t    n_protect) {
    if (!g_ssd_streaming_mode ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        expert >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        expert >= n_total_expert ||
        !ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                      down_expert_bytes) ||
        ds4_gpu_stream_expert_cache_effective_cap(layer,
                                                  n_total_expert,
                                                  n_selected) == 0) {
        return NULL;
    }

    ds4_gpu_stream_expert_cache_entry *e =
        &g_stream_expert_cache[layer][expert];
    if (ds4_gpu_stream_expert_cache_entry_matches(e,
                                                  model_map,
                                                  model_size,
                                                  gate_abs_offset,
                                                  up_abs_offset,
                                                  down_abs_offset,
                                                  gate_expert_bytes,
                                                  down_expert_bytes)) {
        e->last_used = ++g_stream_expert_cache_clock;
        e->use_count++;
        g_stream_expert_cache_hits++;
        g_stream_expert_cache_layer_hits[layer]++;
        return e;
    }

    ds4_gpu_stream_expert_readahead_range(gate_abs_offset, gate_expert_bytes);
    ds4_gpu_stream_expert_readahead_range(up_abs_offset, gate_expert_bytes);
    ds4_gpu_stream_expert_readahead_range(down_abs_offset, down_expert_bytes);

    id<MTLBuffer> gate_buf = nil;
    id<MTLBuffer> up_buf = nil;
    id<MTLBuffer> down_buf = nil;
    NSUInteger gate_inner = 0;
    NSUInteger up_inner = 0;
    NSUInteger down_inner = 0;
    const int32_t protect_one = (int32_t)expert;
    if (!protect_ids || n_protect == 0) {
        protect_ids = &protect_one;
        n_protect = 1;
    }
    const uint32_t cache_budget =
        ds4_gpu_stream_expert_cache_configured_budget();
    const int force_reuse =
        cache_budget != 0 && g_stream_expert_cache_entry_count >= cache_budget;
    if (!ds4_gpu_stream_expert_cache_prepare_load_buffers(layer,
                                                          expert,
                                                          layer,
                                                          protect_ids,
                                                          n_protect,
                                                          gate_expert_bytes,
                                                          down_expert_bytes,
                                                          force_reuse,
                                                          &gate_buf,
                                                          &up_buf,
                                                          &down_buf,
                                                          &gate_inner,
                                                          &up_inner,
                                                          &down_inner)) {
        return NULL;
    }
    if (!gate_buf || !up_buf || !down_buf) return NULL;

    uint8_t *gate_dst = (uint8_t *)[gate_buf contents] + gate_inner;
    uint8_t *up_dst = (uint8_t *)[up_buf contents] + up_inner;
    uint8_t *down_dst = (uint8_t *)[down_buf contents] + down_inner;
    if (!gate_dst || !up_dst || !down_dst) return NULL;

    ds4_gpu_stream_expert_pread_task tasks[3] = {
        {
            .offset = gate_abs_offset,
            .len = gate_expert_bytes,
            .dst = gate_dst,
        },
        {
            .offset = up_abs_offset,
            .len = gate_expert_bytes,
            .dst = up_dst,
        },
        {
            .offset = down_abs_offset,
            .len = down_expert_bytes,
            .dst = down_dst,
        },
    };
    uint64_t read_bytes = 0;
    double read_ms = 0.0;
    if (!ds4_gpu_stream_expert_pread_tasks(tasks, 3, &read_bytes, &read_ms)) {
        return NULL;
    }
    ds4_gpu_stream_expert_cache_note_pread(layer, read_bytes, read_ms);

    [gate_buf didModifyRange:NSMakeRange(gate_inner, (NSUInteger)gate_expert_bytes)];
    [up_buf didModifyRange:NSMakeRange(up_inner, (NSUInteger)gate_expert_bytes)];
    [down_buf didModifyRange:NSMakeRange(down_inner, (NSUInteger)down_expert_bytes)];
    if (getenv("DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE") != NULL) {
        fprintf(stderr,
                "ds4: Metal streaming expert parallel pread layer=%u experts=1 tensors=3 "
                "threads=%u bytes=%.2f GiB wall=%.3f ms\n",
                layer,
                ds4_gpu_stream_expert_pread_thread_count(3),
                ds4_gpu_gib(read_bytes),
                read_ms);
    }
    return ds4_gpu_stream_expert_cache_install_loaded(model_map,
                                                      model_size,
                                                      layer,
                                                      expert,
                                                      gate_abs_offset,
                                                      up_abs_offset,
                                                      down_abs_offset,
                                                      gate_expert_bytes,
                                                      down_expert_bytes,
                                                      gate_buf,
                                                      up_buf,
                                                      down_buf,
                                                      gate_inner,
                                                      up_inner,
                                                      down_inner);
}

static ds4_gpu_stream_expert_cache_entry *ds4_gpu_stream_expert_cache_get(
        const void *model_map,
        uint64_t    model_size,
        uint32_t    layer,
        uint32_t    expert,
        uint32_t    n_total_expert,
        uint32_t    n_selected,
        uint64_t    gate_abs_offset,
        uint64_t    up_abs_offset,
        uint64_t    down_abs_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes) {
    return ds4_gpu_stream_expert_cache_get_protected(model_map,
                                                     model_size,
                                                     layer,
                                                     expert,
                                                     n_total_expert,
                                                     n_selected,
                                                     gate_abs_offset,
                                                     up_abs_offset,
                                                     down_abs_offset,
                                                     gate_expert_bytes,
                                                     down_expert_bytes,
                                                     NULL,
                                                     0);
}

static int ds4_gpu_stream_expert_pending_load_profile_enabled(void) {
    return getenv("DS4_METAL_STREAMING_EXPERT_EARLY_LOAD_PROFILE") != NULL;
}

static void ds4_gpu_stream_expert_pending_load_release_buffers(
        ds4_gpu_stream_expert_pending_load *p) {
    if (!p) return;
    for (uint32_t i = 0; i < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED; i++) {
        p->gate_bufs[i] = nil;
        p->up_bufs[i] = nil;
        p->down_bufs[i] = nil;
        p->gate_inners[i] = 0;
        p->up_inners[i] = 0;
        p->down_inners[i] = 0;
    }
}

static int ds4_gpu_stream_expert_pending_load_install(
        ds4_gpu_stream_expert_pending_load *p,
        ds4_gpu_stream_expert_cache_entry *entries[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED],
        double elapsed_ms) {
    if (!p || p->n_loads == 0) return 1;

    uint64_t read_bytes = 0;
    int ok = 1;
    for (uint32_t i = 0; i < p->n_tasks; i++) {
        if (!p->tasks[i].ok) ok = 0;
        if (read_bytes > UINT64_MAX - p->tasks[i].read_bytes) {
            read_bytes = UINT64_MAX;
        } else {
            read_bytes += p->tasks[i].read_bytes;
        }
    }
    if (!ok) return 0;

    ds4_gpu_stream_expert_cache_note_pread(p->layer, read_bytes, elapsed_ms);
    const int load_timing = ds4_gpu_stream_expert_timing_summary_enabled();
    double load_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
    double load_modify_ms = 0.0;
    double load_install_ms = 0.0;
    for (uint32_t load_i = 0; load_i < p->n_loads; load_i++) {
        [p->gate_bufs[load_i] didModifyRange:NSMakeRange(p->gate_inners[load_i], (NSUInteger)p->gate_expert_bytes)];
        [p->up_bufs[load_i] didModifyRange:NSMakeRange(p->up_inners[load_i], (NSUInteger)p->gate_expert_bytes)];
        [p->down_bufs[load_i] didModifyRange:NSMakeRange(p->down_inners[load_i], (NSUInteger)p->down_expert_bytes)];
    }
    if (load_timing) {
        const double now_ms = ds4_gpu_now_ms();
        load_modify_ms = now_ms - load_t0;
        load_t0 = now_ms;
    }

    ds4_gpu_stream_expert_cache_entry
        *loaded_entries[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    for (uint32_t i = 0; i < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED; i++) {
        loaded_entries[i] = NULL;
    }
    for (uint32_t load_i = 0; load_i < p->n_loads; load_i++) {
        const uint32_t slot = p->load_slots[load_i];
        const uint32_t expert = (uint32_t)p->selected_ids[slot];
        ds4_gpu_stream_expert_cache_entry *entry =
            ds4_gpu_stream_expert_cache_install_loaded(p->model_map,
                                                       p->model_size,
                                                       p->layer,
                                                       expert,
                                                       p->gate_abs_offsets[slot],
                                                       p->up_abs_offsets[slot],
                                                       p->down_abs_offsets[slot],
                                                       p->gate_expert_bytes,
                                                       p->down_expert_bytes,
                                                       p->gate_bufs[load_i],
                                                       p->up_bufs[load_i],
                                                       p->down_bufs[load_i],
                                                       p->gate_inners[load_i],
                                                       p->up_inners[load_i],
                                                       p->down_inners[load_i]);
        if (!entry) return 0;
        loaded_entries[slot] = entry;
        if (entries) entries[slot] = entry;
    }
    for (uint32_t i = 0; i < p->n_selected; i++) {
        if ((p->missing_mask & (1u << i)) == 0) continue;
        if (entries && entries[i]) continue;
        const uint32_t source = p->source_slots[i];
        if (source >= p->n_selected) return 0;
        ds4_gpu_stream_expert_cache_entry *entry = entries && entries[source] ?
            entries[source] : loaded_entries[source];
        if (!entry) return 0;
        entry->use_count++;
        if (entries) entries[i] = entry;
    }
    if (load_timing) {
        load_install_ms = ds4_gpu_now_ms() - load_t0;
        ds4_gpu_stream_expert_timing_note_load_detail(p->prepare_ms,
                                                      elapsed_ms,
                                                      load_modify_ms,
                                                      load_install_ms);
    }
    if (ds4_gpu_stream_expert_pending_load_profile_enabled()) {
        fprintf(stderr,
                "ds4: Metal streaming expert early-load finish layer=%u experts=%u tensors=%u bytes=%.2f GiB wall=%.3f ms\n",
                p->layer,
                p->n_loads,
                p->n_tasks,
                ds4_gpu_gib(read_bytes),
                elapsed_ms);
    }
    return 1;
}

static int ds4_gpu_stream_expert_pending_load_finish(
        ds4_gpu_stream_expert_cache_entry *entries[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED]) {
    ds4_gpu_stream_expert_pending_load *p = &g_stream_expert_pending_load;
    if (!p->active) return 1;

    const double start_ms = p->start_ms;
    if (!ds4_gpu_stream_expert_pread_pool_wait()) {
        ds4_gpu_stream_expert_pending_load_release_buffers(p);
        p->active = 0;
        return 0;
    }
    const double elapsed_ms = ds4_gpu_now_ms() - start_ms;
    p->active = 0;
    const int ok = ds4_gpu_stream_expert_pending_load_install(p,
                                                              entries,
                                                              elapsed_ms);
    ds4_gpu_stream_expert_pending_load_release_buffers(p);
    p->n_tasks = 0;
    p->n_loads = 0;
    p->prepare_ms = 0.0;
    return ok;
}

static void ds4_gpu_stream_expert_pending_load_clear(void) {
    if (!g_stream_expert_pending_load.active) {
        ds4_gpu_stream_expert_pending_load_release_buffers(
                &g_stream_expert_pending_load);
        return;
    }
    (void)ds4_gpu_stream_expert_pending_load_finish(NULL);
}

static int ds4_gpu_stream_expert_pending_load_matches(
        const void   *model_map,
        uint64_t      model_size,
        uint32_t      layer,
        const int32_t *selected_ids,
        uint32_t      n_total_expert,
        uint32_t      n_selected,
        uint64_t      gate_expert_bytes,
        uint64_t      down_expert_bytes) {
    ds4_gpu_stream_expert_pending_load *p = &g_stream_expert_pending_load;
    if (!p->active ||
        p->model_map != model_map ||
        p->model_size != model_size ||
        p->layer != layer ||
        p->n_total_expert != n_total_expert ||
        p->n_selected != n_selected ||
        p->gate_expert_bytes != gate_expert_bytes ||
        p->down_expert_bytes != down_expert_bytes) {
        return 0;
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        if (p->selected_ids[i] != selected_ids[i]) return 0;
    }
    return 1;
}

int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode ||
        getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD") != NULL) {
        return 1;
    }
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !selected_ids ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_selected == 0 ||
        n_selected > DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED ||
        n_total_expert == 0 ||
        n_total_expert > DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        !ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                      down_expert_bytes) ||
        ds4_gpu_stream_expert_cache_effective_cap(layer,
                                                  n_total_expert,
                                                  n_selected) == 0) {
        return 1;
    }
    if (!g_initialized && !ds4_gpu_init()) return 0;

    if (ds4_gpu_stream_expert_pending_load_matches(model_map,
                                                   model_size,
                                                   layer,
                                                   selected_ids,
                                                   n_total_expert,
                                                   n_selected,
                                                   gate_expert_bytes,
                                                   down_expert_bytes)) {
        return 1;
    }

    ds4_gpu_stream_expert_pending_load_clear();
    ds4_gpu_stream_expert_pending_load *p = &g_stream_expert_pending_load;
    p->active = 0;
    p->model_map = model_map;
    p->model_size = model_size;
    p->layer = layer;
    p->n_total_expert = n_total_expert;
    p->n_selected = n_selected;
    p->missing_mask = 0;
    p->n_loads = 0;
    p->n_tasks = 0;
    p->gate_expert_bytes = gate_expert_bytes;
    p->down_expert_bytes = down_expert_bytes;
    p->prepare_ms = 0.0;
    const int load_timing = ds4_gpu_stream_expert_timing_summary_enabled();
    const double load_prepare_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
    for (uint32_t i = 0; i < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED; i++) {
        p->selected_ids[i] = -1;
        p->load_slots[i] = 0;
        p->source_slots[i] = UINT32_MAX;
        p->gate_abs_offsets[i] = 0;
        p->up_abs_offsets[i] = 0;
        p->down_abs_offsets[i] = 0;
        p->gate_bufs[i] = nil;
        p->up_bufs[i] = nil;
        p->down_bufs[i] = nil;
        p->gate_inners[i] = 0;
        p->up_inners[i] = 0;
        p->down_inners[i] = 0;
    }
    memset(p->tasks, 0, sizeof(p->tasks));

    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: Metal streaming early-load expert id %d is outside 0..%u\n",
                    selected_ids[i],
                    n_total_expert);
            return 0;
        }
        p->selected_ids[i] = selected_ids[i];
        const uint64_t expert_id = (uint64_t)(uint32_t)selected_ids[i];
        if (expert_id > UINT64_MAX / gate_expert_bytes ||
            expert_id > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: Metal streaming early-load offset overflow\n");
            return 0;
        }
        const uint64_t gate_rel = expert_id * gate_expert_bytes;
        const uint64_t down_rel = expert_id * down_expert_bytes;
        if (gate_rel > UINT64_MAX - gate_offset ||
            gate_rel > UINT64_MAX - up_offset ||
            down_rel > UINT64_MAX - down_offset) {
            fprintf(stderr, "ds4: Metal streaming early-load offset overflow\n");
            return 0;
        }
        p->gate_abs_offsets[i] = gate_offset + gate_rel;
        p->up_abs_offsets[i] = up_offset + gate_rel;
        p->down_abs_offsets[i] = down_offset + down_rel;

        ds4_gpu_stream_expert_cache_entry *e =
            &g_stream_expert_cache[layer][(uint32_t)selected_ids[i]];
        if (ds4_gpu_stream_expert_cache_entry_matches(e,
                                                      model_map,
                                                      model_size,
                                                      p->gate_abs_offsets[i],
                                                      p->up_abs_offsets[i],
                                                      p->down_abs_offsets[i],
                                                      gate_expert_bytes,
                                                      down_expert_bytes)) {
            continue;
        }

        uint32_t source = UINT32_MAX;
        for (uint32_t prev = 0; prev < i; prev++) {
            if (selected_ids[prev] == selected_ids[i] &&
                p->gate_abs_offsets[prev] == p->gate_abs_offsets[i] &&
                p->up_abs_offsets[prev] == p->up_abs_offsets[i] &&
                p->down_abs_offsets[prev] == p->down_abs_offsets[i] &&
                (p->missing_mask & (1u << prev)) != 0) {
                source = p->source_slots[prev] != UINT32_MAX ?
                         p->source_slots[prev] : prev;
                break;
            }
        }
        if (source != UINT32_MAX) {
            p->source_slots[i] = source;
            p->missing_mask |= 1u << i;
            continue;
        }
        p->source_slots[i] = i;
        p->missing_mask |= 1u << i;
        p->load_slots[p->n_loads++] = i;
    }
    if (p->n_loads == 0) return 1;

    const uint32_t cache_budget =
        ds4_gpu_stream_expert_cache_configured_budget();
    uint32_t reserved_entries = g_stream_expert_cache_entry_count;
    ds4_gpu_stream_expert_reusable_buffers
        batch_reuse[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    for (uint32_t i = 0; i < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED; i++) {
        batch_reuse[i] =
            (ds4_gpu_stream_expert_reusable_buffers){ nil, nil, nil, 0, 0, 0 };
    }
    uint32_t batch_reuse_count = 0;
    if (cache_budget != 0 &&
        reserved_entries >= cache_budget &&
        p->n_loads > 1 &&
        ds4_gpu_stream_expert_batch_reuse_enabled(gate_expert_bytes,
                                                  down_expert_bytes)) {
        const double reuse_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
        batch_reuse_count =
            ds4_gpu_stream_expert_cache_take_reusable_batch(
                    p->n_loads,
                    layer,
                    selected_ids,
                    n_selected,
                    gate_expert_bytes,
                    down_expert_bytes,
                    batch_reuse);
        if (load_timing) {
            ds4_gpu_stream_expert_timing_note_prepare_batch_reuse(
                    ds4_gpu_now_ms() - reuse_t0);
        }
    }
    for (uint32_t load_i = 0; load_i < p->n_loads; load_i++) {
        const uint32_t slot = p->load_slots[load_i];
        const uint32_t expert = (uint32_t)p->selected_ids[slot];
        const int force_reuse =
            cache_budget != 0 && reserved_entries >= cache_budget;

        ds4_gpu_stream_expert_readahead_range(p->gate_abs_offsets[slot],
                                              gate_expert_bytes);
        ds4_gpu_stream_expert_readahead_range(p->up_abs_offsets[slot],
                                              gate_expert_bytes);
        ds4_gpu_stream_expert_readahead_range(p->down_abs_offsets[slot],
                                              down_expert_bytes);

        if (load_i < batch_reuse_count &&
            batch_reuse[load_i].gate_buffer &&
            batch_reuse[load_i].up_buffer &&
            batch_reuse[load_i].down_buffer) {
            p->gate_bufs[load_i] = batch_reuse[load_i].gate_buffer;
            p->up_bufs[load_i] = batch_reuse[load_i].up_buffer;
            p->down_bufs[load_i] = batch_reuse[load_i].down_buffer;
            p->gate_inners[load_i] = batch_reuse[load_i].gate_inner;
            p->up_inners[load_i] = batch_reuse[load_i].up_inner;
            p->down_inners[load_i] = batch_reuse[load_i].down_inner;
        } else {
            const double buffer_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
            const int prepared =
                ds4_gpu_stream_expert_cache_prepare_load_buffers(layer,
                                                                 expert,
                                                                 layer,
                                                                 selected_ids,
                                                                 n_selected,
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes,
                                                                 force_reuse,
                                                                 &p->gate_bufs[load_i],
                                                                 &p->up_bufs[load_i],
                                                                 &p->down_bufs[load_i],
                                                                 &p->gate_inners[load_i],
                                                                 &p->up_inners[load_i],
                                                                 &p->down_inners[load_i]);
            if (load_timing) {
                ds4_gpu_stream_expert_timing_note_prepare_buffer(
                        ds4_gpu_now_ms() - buffer_t0);
            }
            if (!prepared) {
                ds4_gpu_stream_expert_pending_load_release_buffers(p);
                return 0;
            }
        }
        if (!force_reuse && reserved_entries < UINT32_MAX) {
            reserved_entries++;
        }
        uint8_t *gate_dst = (uint8_t *)[p->gate_bufs[load_i] contents] +
                             p->gate_inners[load_i];
        uint8_t *up_dst = (uint8_t *)[p->up_bufs[load_i] contents] +
                           p->up_inners[load_i];
        uint8_t *down_dst = (uint8_t *)[p->down_bufs[load_i] contents] +
                             p->down_inners[load_i];
        if (!gate_dst || !up_dst || !down_dst) {
            ds4_gpu_stream_expert_pending_load_release_buffers(p);
            return 0;
        }
        const double task_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
        p->tasks[p->n_tasks++] = (ds4_gpu_stream_expert_pread_task) {
            .offset = p->gate_abs_offsets[slot],
            .len = gate_expert_bytes,
            .dst = gate_dst,
        };
        p->tasks[p->n_tasks++] = (ds4_gpu_stream_expert_pread_task) {
            .offset = p->up_abs_offsets[slot],
            .len = gate_expert_bytes,
            .dst = up_dst,
        };
        p->tasks[p->n_tasks++] = (ds4_gpu_stream_expert_pread_task) {
            .offset = p->down_abs_offsets[slot],
            .len = down_expert_bytes,
            .dst = down_dst,
        };
        if (load_timing) {
            ds4_gpu_stream_expert_timing_note_prepare_task(
                    1,
                    ds4_gpu_now_ms() - task_t0);
        }
    }

    const uint32_t n_workers =
        ds4_gpu_stream_expert_pread_thread_count(p->n_tasks);
    p->start_ms = ds4_gpu_now_ms();
    if (load_timing) {
        p->prepare_ms = p->start_ms - load_prepare_t0;
    }
    if (ds4_gpu_stream_expert_pread_pool_begin(p->tasks,
                                               p->n_tasks,
                                               n_workers)) {
        p->active = 1;
        if (ds4_gpu_stream_expert_pending_load_profile_enabled()) {
            fprintf(stderr,
                    "ds4: Metal streaming expert early-load begin layer=%u experts=%u tensors=%u threads=%u\n",
                    layer,
                    p->n_loads,
                    p->n_tasks,
                    n_workers);
        }
        return 1;
    }

    uint64_t read_bytes = 0;
    double read_ms = 0.0;
    if (!ds4_gpu_stream_expert_pread_tasks(p->tasks,
                                           p->n_tasks,
                                           &read_bytes,
                                           &read_ms)) {
        ds4_gpu_stream_expert_pending_load_release_buffers(p);
        return 0;
    }
    (void)read_bytes;
    if (!ds4_gpu_stream_expert_pending_load_install(p, NULL, read_ms)) {
        ds4_gpu_stream_expert_pending_load_release_buffers(p);
        return 0;
    }
    ds4_gpu_stream_expert_pending_load_release_buffers(p);
    p->n_tasks = 0;
    p->n_loads = 0;
    p->prepare_ms = 0.0;
    return 1;
}

static int ds4_gpu_stream_expert_cache_load_selected_missing(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *selected_ids,
        uint32_t       n_total_expert,
        uint32_t       n_selected,
        const uint64_t *gate_abs_offsets,
        const uint64_t *up_abs_offsets,
        const uint64_t *down_abs_offsets,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        uint32_t       missing_mask,
        ds4_gpu_stream_expert_cache_entry **entries) {
    if (!g_ssd_streaming_mode ||
        !model_map ||
        !selected_ids ||
        !gate_abs_offsets ||
        !up_abs_offsets ||
        !down_abs_offsets ||
        !entries ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_selected == 0 ||
        n_selected > DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED ||
        n_total_expert == 0 ||
        n_total_expert > DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        !ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                      down_expert_bytes) ||
        ds4_gpu_stream_expert_cache_effective_cap(layer,
                                                  n_total_expert,
                                                  n_selected) == 0) {
        return 0;
    }
    missing_mask &= (1u << n_selected) - 1u;
    if (missing_mask == 0) return 1;
    if (g_stream_expert_pending_load.active &&
        n_selected <= DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED) {
        if (ds4_gpu_stream_expert_pending_load_matches(model_map,
                                                       model_size,
                                                       layer,
                                                       selected_ids,
                                                       n_total_expert,
                                                       n_selected,
                                                       gate_expert_bytes,
                                                       down_expert_bytes)) {
            if (!ds4_gpu_stream_expert_pending_load_finish(entries)) return 0;
            for (uint32_t i = 0; i < n_selected; i++) {
                if ((missing_mask & (1u << i)) != 0 && entries[i]) {
                    missing_mask &= ~(1u << i);
                }
            }
            if (missing_mask == 0) return 1;
        } else {
            ds4_gpu_stream_expert_pending_load_clear();
        }
    } else if (g_stream_expert_pending_load.active) {
        ds4_gpu_stream_expert_pending_load_clear();
    }

    const int load_timing = ds4_gpu_stream_expert_timing_summary_enabled();
    double load_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
    double load_prepare_ms = 0.0;
    double load_modify_ms = 0.0;
    double load_install_ms = 0.0;

    uint32_t load_slots[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint32_t source_slots[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    for (uint32_t i = 0; i < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED; i++) {
        load_slots[i] = 0;
        source_slots[i] = UINT32_MAX;
    }
    uint32_t n_loads = 0;
    for (uint32_t i = 0; i < n_selected; i++) {
        if ((missing_mask & (1u << i)) == 0) continue;
        if (entries[i]) {
            source_slots[i] = i;
            continue;
        }
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: Metal streaming selected missing expert id %d is outside 0..%u\n",
                    selected_ids[i],
                    n_total_expert);
            return 0;
        }
        uint32_t source = UINT32_MAX;
        for (uint32_t prev = 0; prev < i; prev++) {
            if (selected_ids[prev] == selected_ids[i] &&
                gate_abs_offsets[prev] == gate_abs_offsets[i] &&
                up_abs_offsets[prev] == up_abs_offsets[i] &&
                down_abs_offsets[prev] == down_abs_offsets[i] &&
                (entries[prev] || (missing_mask & (1u << prev)) != 0)) {
                source = source_slots[prev] != UINT32_MAX ?
                         source_slots[prev] : prev;
                break;
            }
        }
        if (source != UINT32_MAX) {
            source_slots[i] = source;
            continue;
        }
        source_slots[i] = i;
        load_slots[n_loads++] = i;
    }
    if (n_loads == 0) {
        for (uint32_t i = 0; i < n_selected; i++) {
            if ((missing_mask & (1u << i)) == 0 || entries[i]) continue;
            const uint32_t source = source_slots[i];
            if (source >= n_selected || !entries[source]) return 0;
            entries[i] = entries[source];
            entries[i]->use_count++;
        }
        return 1;
    }

    __strong id<MTLBuffer> gate_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    __strong id<MTLBuffer> up_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    __strong id<MTLBuffer> down_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    NSUInteger gate_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    NSUInteger up_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    NSUInteger down_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    for (uint32_t i = 0; i < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED; i++) {
        gate_bufs[i] = nil;
        up_bufs[i] = nil;
        down_bufs[i] = nil;
        gate_inners[i] = 0;
        up_inners[i] = 0;
        down_inners[i] = 0;
    }
    ds4_gpu_stream_expert_pread_task tasks[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED * 3u];
    memset(tasks, 0, sizeof(tasks));
    uint32_t n_tasks = 0;
    const uint32_t cache_budget =
        ds4_gpu_stream_expert_cache_configured_budget();
    uint32_t reserved_entries = g_stream_expert_cache_entry_count;
    ds4_gpu_stream_expert_reusable_buffers
        batch_reuse[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    for (uint32_t i = 0; i < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED; i++) {
        batch_reuse[i] =
            (ds4_gpu_stream_expert_reusable_buffers){ nil, nil, nil, 0, 0, 0 };
    }
    uint32_t batch_reuse_count = 0;
    if (cache_budget != 0 &&
        reserved_entries >= cache_budget &&
        n_loads > 1 &&
        ds4_gpu_stream_expert_batch_reuse_enabled(gate_expert_bytes,
                                                  down_expert_bytes)) {
        const double reuse_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
        batch_reuse_count =
            ds4_gpu_stream_expert_cache_take_reusable_batch(
                    n_loads,
                    layer,
                    selected_ids,
                    n_selected,
                    gate_expert_bytes,
                    down_expert_bytes,
                    batch_reuse);
        if (load_timing) {
            ds4_gpu_stream_expert_timing_note_prepare_batch_reuse(
                    ds4_gpu_now_ms() - reuse_t0);
        }
    }

    for (uint32_t load_i = 0; load_i < n_loads; load_i++) {
        const uint32_t slot = load_slots[load_i];
        const uint32_t expert = (uint32_t)selected_ids[slot];
        const int force_reuse =
            cache_budget != 0 && reserved_entries >= cache_budget;

        ds4_gpu_stream_expert_readahead_range(gate_abs_offsets[slot],
                                              gate_expert_bytes);
        ds4_gpu_stream_expert_readahead_range(up_abs_offsets[slot],
                                              gate_expert_bytes);
        ds4_gpu_stream_expert_readahead_range(down_abs_offsets[slot],
                                              down_expert_bytes);

        if (load_i < batch_reuse_count &&
            batch_reuse[load_i].gate_buffer &&
            batch_reuse[load_i].up_buffer &&
            batch_reuse[load_i].down_buffer) {
            gate_bufs[load_i] = batch_reuse[load_i].gate_buffer;
            up_bufs[load_i] = batch_reuse[load_i].up_buffer;
            down_bufs[load_i] = batch_reuse[load_i].down_buffer;
            gate_inners[load_i] = batch_reuse[load_i].gate_inner;
            up_inners[load_i] = batch_reuse[load_i].up_inner;
            down_inners[load_i] = batch_reuse[load_i].down_inner;
        } else {
            const double buffer_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
            const int prepared =
                ds4_gpu_stream_expert_cache_prepare_load_buffers(layer,
                                                                 expert,
                                                                 layer,
                                                                 selected_ids,
                                                                 n_selected,
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes,
                                                                 force_reuse,
                                                                 &gate_bufs[load_i],
                                                                 &up_bufs[load_i],
                                                                 &down_bufs[load_i],
                                                                 &gate_inners[load_i],
                                                                 &up_inners[load_i],
                                                                 &down_inners[load_i]);
            if (load_timing) {
                ds4_gpu_stream_expert_timing_note_prepare_buffer(
                        ds4_gpu_now_ms() - buffer_t0);
            }
            if (!prepared) {
                return 0;
            }
        }
        if (!force_reuse && reserved_entries < UINT32_MAX) {
            reserved_entries++;
        }
        if (!gate_bufs[load_i] || !up_bufs[load_i] || !down_bufs[load_i]) {
            return 0;
        }

        uint8_t *gate_dst = (uint8_t *)[gate_bufs[load_i] contents] +
                             gate_inners[load_i];
        uint8_t *up_dst = (uint8_t *)[up_bufs[load_i] contents] +
                           up_inners[load_i];
        uint8_t *down_dst = (uint8_t *)[down_bufs[load_i] contents] +
                             down_inners[load_i];
        if (!gate_dst || !up_dst || !down_dst) return 0;

        const double task_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
        tasks[n_tasks++] = (ds4_gpu_stream_expert_pread_task) {
            .offset = gate_abs_offsets[slot],
            .len = gate_expert_bytes,
            .dst = gate_dst,
        };
        tasks[n_tasks++] = (ds4_gpu_stream_expert_pread_task) {
            .offset = up_abs_offsets[slot],
            .len = gate_expert_bytes,
            .dst = up_dst,
        };
        tasks[n_tasks++] = (ds4_gpu_stream_expert_pread_task) {
            .offset = down_abs_offsets[slot],
            .len = down_expert_bytes,
            .dst = down_dst,
        };
        if (load_timing) {
            ds4_gpu_stream_expert_timing_note_prepare_task(
                    1,
                    ds4_gpu_now_ms() - task_t0);
        }
    }

    if (load_timing) {
        const double now_ms = ds4_gpu_now_ms();
        load_prepare_ms = now_ms - load_t0;
        load_t0 = now_ms;
    }

    uint64_t read_bytes = 0;
    double read_ms = 0.0;
    const int ok = ds4_gpu_stream_expert_pread_tasks(tasks,
                                                     n_tasks,
                                                     &read_bytes,
                                                     &read_ms);
    if (!ok) return 0;
    if (load_timing) {
        load_t0 = ds4_gpu_now_ms();
    }
    ds4_gpu_stream_expert_cache_note_pread(layer, read_bytes, read_ms);

    for (uint32_t load_i = 0; load_i < n_loads; load_i++) {
        [gate_bufs[load_i] didModifyRange:NSMakeRange(gate_inners[load_i], (NSUInteger)gate_expert_bytes)];
        [up_bufs[load_i] didModifyRange:NSMakeRange(up_inners[load_i], (NSUInteger)gate_expert_bytes)];
        [down_bufs[load_i] didModifyRange:NSMakeRange(down_inners[load_i], (NSUInteger)down_expert_bytes)];
    }
    if (load_timing) {
        const double now_ms = ds4_gpu_now_ms();
        load_modify_ms = now_ms - load_t0;
        load_t0 = now_ms;
    }
    if (getenv("DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE") != NULL) {
        fprintf(stderr,
                "ds4: Metal streaming expert parallel pread layer=%u experts=%u tensors=%u "
                "threads=%u bytes=%.2f GiB wall=%.3f ms\n",
                layer,
                n_loads,
                n_tasks,
                ds4_gpu_stream_expert_pread_thread_count(n_tasks),
                ds4_gpu_gib(read_bytes),
                read_ms);
    }

    for (uint32_t load_i = 0; load_i < n_loads; load_i++) {
        const uint32_t slot = load_slots[load_i];
        const uint32_t expert = (uint32_t)selected_ids[slot];
        ds4_gpu_stream_expert_cache_entry *entry =
            ds4_gpu_stream_expert_cache_install_loaded(model_map,
                                                       model_size,
                                                       layer,
                                                       expert,
                                                       gate_abs_offsets[slot],
                                                       up_abs_offsets[slot],
                                                       down_abs_offsets[slot],
                                                       gate_expert_bytes,
                                                       down_expert_bytes,
                                                       gate_bufs[load_i],
                                                       up_bufs[load_i],
                                                       down_bufs[load_i],
                                                       gate_inners[load_i],
                                                       up_inners[load_i],
                                                       down_inners[load_i]);
        if (!entry) return 0;
        entries[slot] = entry;
    }

    for (uint32_t i = 0; i < n_selected; i++) {
        if ((missing_mask & (1u << i)) == 0 || entries[i]) continue;
        const uint32_t source = source_slots[i];
        if (source >= n_selected || !entries[source]) return 0;
        entries[i] = entries[source];
        entries[i]->use_count++;
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        if ((missing_mask & (1u << i)) != 0 && !entries[i]) return 0;
    }
    if (load_timing) {
        load_install_ms = ds4_gpu_now_ms() - load_t0;
        ds4_gpu_stream_expert_timing_note_load_detail(load_prepare_ms,
                                                      read_ms,
                                                      load_modify_ms,
                                                      load_install_ms);
    }
    return 1;
}

static void ds4_gpu_glm_stream_selected_prefetch_set(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    g_glm_stream_selected_prefetch.active = 0;
    if (!table || !selected_ids ||
        n_selected == 0 ||
        n_selected > DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED) {
        return;
    }
    g_glm_stream_selected_prefetch.model_map = table->model_map;
    g_glm_stream_selected_prefetch.model_size = table->model_size;
    g_glm_stream_selected_prefetch.layer = table->layer;
    g_glm_stream_selected_prefetch.n_total_expert = table->n_total_expert;
    g_glm_stream_selected_prefetch.n_selected = n_selected;
    g_glm_stream_selected_prefetch.gate_offset = table->gate_offset;
    g_glm_stream_selected_prefetch.up_offset = table->up_offset;
    g_glm_stream_selected_prefetch.down_offset = table->down_offset;
    g_glm_stream_selected_prefetch.gate_expert_bytes = table->gate_expert_bytes;
    g_glm_stream_selected_prefetch.down_expert_bytes = table->down_expert_bytes;
    for (uint32_t i = 0; i < n_selected; i++) {
        g_glm_stream_selected_prefetch.selected_ids[i] = selected_ids[i];
    }
    g_glm_stream_selected_prefetch.active = 1;
}

static int ds4_gpu_glm_stream_selected_prefetch_take(
        const void *model_map,
        uint64_t    model_size,
        uint32_t    layer,
        uint32_t    n_total_expert,
        uint32_t    n_selected,
        uint64_t    gate_offset,
        uint64_t    up_offset,
        uint64_t    down_offset,
        uint64_t    gate_expert_bytes,
        uint64_t    down_expert_bytes,
        int32_t    *selected_ids_out) {
    if (!selected_ids_out || !g_glm_stream_selected_prefetch.active) return 0;
    ds4_gpu_glm_stream_selected_prefetch *p = &g_glm_stream_selected_prefetch;
    if (p->model_map != model_map ||
        p->model_size != model_size ||
        p->layer != layer ||
        p->n_total_expert != n_total_expert ||
        p->n_selected != n_selected ||
        p->gate_offset != gate_offset ||
        p->up_offset != up_offset ||
        p->down_offset != down_offset ||
        p->gate_expert_bytes != gate_expert_bytes ||
        p->down_expert_bytes != down_expert_bytes) {
        return 0;
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        selected_ids_out[i] = p->selected_ids[i];
    }
    p->active = 0;
    return 1;
}

int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_selected) {
    g_glm_stream_selected_prefetch.active = 0;
    if (!g_ssd_streaming_mode ||
        getenv("DS4_METAL_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD") != NULL) {
        return 1;
    }
    if (!table || !selected ||
        n_selected == 0 ||
        n_selected > DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED) {
        return 1;
    }

    const int timing = ds4_gpu_stream_expert_timing_summary_enabled();
    double t0 = timing ? ds4_gpu_now_ms() : 0.0;
    double selected_sync_ms = 0.0;
    double selected_copy_ms = 0.0;
    double selected_bind_ms = 0.0;
    const int had_batch = g_batch_cb != nil;
    if (had_batch) {
        if (ds4_gpu_end_commands() == 0) return 0;
        if (timing) {
            selected_sync_ms = ds4_gpu_now_ms() - t0;
            t0 = ds4_gpu_now_ms();
        }
    }

    int32_t selected_ids[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    for (uint32_t i = 0; i < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED; i++) {
        selected_ids[i] = -1;
    }
    int ok = ds4_gpu_tensor_read(selected,
                                 0,
                                 selected_ids,
                                 (uint64_t)n_selected * sizeof(selected_ids[0])) != 0;
    if (timing) {
        selected_copy_ms = ds4_gpu_now_ms() - t0;
        t0 = ds4_gpu_now_ms();
    }
    if (ok) {
        ok = ds4_gpu_stream_expert_cache_begin_selected_load(table,
                                                             selected_ids,
                                                             n_selected) != 0;
    }
    if (timing) {
        selected_bind_ms = ds4_gpu_now_ms() - t0;
        ds4_gpu_stream_expert_timing_note_selected(selected_sync_ms,
                                                   selected_copy_ms,
                                                   selected_bind_ms);
    }
    if (ok) {
        ds4_gpu_glm_stream_selected_prefetch_set(table,
                                                 selected_ids,
                                                 n_selected);
    }
    if (had_batch && ds4_gpu_begin_commands() == 0) ok = 0;
    return ok;
}

static void ds4_gpu_stream_expert_cache_clear_layer(uint32_t layer) {
    if (layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER) return;
    for (uint32_t expert = 0;
         expert < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
         expert++) {
        ds4_gpu_stream_expert_cache_clear_entry(layer, expert, 0);
    }
    g_stream_expert_cache_layer_count[layer] = 0;
}

static int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const ds4_gpu_tensor *selected,
        uint32_t       n_tokens,
        uint32_t       n_total_expert,
        uint32_t       n_selected,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        id<MTLBuffer> *gate_addrs,
        id<MTLBuffer> *up_addrs,
        id<MTLBuffer> *down_addrs,
        ds4_gpu_stream_expert_cache_entry **resources,
        uint32_t      *n_resources,
        uint32_t      *unique_out,
        id<MTLBuffer> *overflow_gate,
        id<MTLBuffer> *overflow_up,
        id<MTLBuffer> *overflow_down) {
    if (overflow_gate) *overflow_gate = nil;
    if (overflow_up) *overflow_up = nil;
    if (overflow_down) *overflow_down = nil;
    if (!g_ssd_streaming_mode ||
        !model_map ||
        !selected ||
        !gate_addrs ||
        !up_addrs ||
        !down_addrs ||
        !resources ||
        !n_resources ||
        !unique_out ||
        !overflow_gate ||
        !overflow_up ||
        !overflow_down ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_tokens == 0 ||
        n_selected == 0 ||
        n_selected > DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED ||
        n_total_expert == 0 ||
        n_total_expert > DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }
    if (n_tokens > UINT32_MAX / n_selected) return 0;

    const uint64_t n_ids = (uint64_t)n_tokens * n_selected;
    if (n_ids > SIZE_MAX / sizeof(int32_t)) return 0;
    int32_t *ids = malloc((size_t)n_ids * sizeof(ids[0]));
    if (!ids) return 0;

    const bool profile =
        getenv("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_PROFILE") != NULL;
    const double t0 = profile ? ds4_gpu_now_ms() : 0.0;
    int ok = ds4_gpu_tensor_read(selected,
                                 0,
                                 ids,
                                 n_ids * sizeof(ids[0]));
    const double t_read = profile ? ds4_gpu_now_ms() : 0.0;

    bool seen[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT] = { false };
    uint32_t frequency[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT] = { 0 };
    int32_t unique_ids[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    uint32_t unique_count = 0;
    *n_resources = 0;
    *unique_out = 0;
    if (ok) {
        if (!ds4_gpu_stream_expert_cache_ensure_addr_buffers(layer)) {
            ok = 0;
        }
    }
    if (ok) {
        for (uint64_t i = 0; i < n_ids; i++) {
            const int32_t selected_id = ids[i];
            if (selected_id < 0 || (uint32_t)selected_id >= n_total_expert) {
                fprintf(stderr,
                        "ds4: Metal streaming batch selected expert id %d is outside 0..%u at layer %u\n",
                        selected_id,
                        n_total_expert,
                        layer);
                ok = 0;
                break;
            }
            frequency[(uint32_t)selected_id]++;
            if (!seen[(uint32_t)selected_id]) {
                seen[(uint32_t)selected_id] = true;
                unique_ids[unique_count++] = selected_id;
            }
        }
    }
    if (ok) {
        ds4_gpu_stream_expert_cache_note_frequency_hotness(layer,
                                                           frequency,
                                                           n_total_expert);
    }
    /*
     * When the layer's unique selected set does not fit the cache budget, the
     * extra experts are addressed straight into whole-tensor mapped model views
     * instead of falling back to a different MoE kernel path. The address-table
     * kernels read identical expert bytes either way, so the cache/view split
     * does not change the computed logits.
     */
    uint32_t view_served = 0;
    uint64_t overflow_gate_inner = 0;
    uint64_t overflow_up_inner = 0;
    uint64_t overflow_down_inner = 0;

    uint64_t unique_gate_offsets[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    uint64_t unique_up_offsets[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    uint64_t unique_down_offsets[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    ds4_gpu_stream_expert_cache_entry
        *unique_entries[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    uint32_t load_unique[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    __strong id<MTLBuffer>
        gate_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    __strong id<MTLBuffer>
        up_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    __strong id<MTLBuffer>
        down_bufs[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    NSUInteger gate_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    NSUInteger up_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    NSUInteger down_inners[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    for (uint32_t i = 0; i < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; i++) {
        unique_gate_offsets[i] = 0;
        unique_up_offsets[i] = 0;
        unique_down_offsets[i] = 0;
        unique_entries[i] = NULL;
        load_unique[i] = 0;
        gate_bufs[i] = nil;
        up_bufs[i] = nil;
        down_bufs[i] = nil;
        gate_inners[i] = 0;
        up_inners[i] = 0;
        down_inners[i] = 0;
    }

    ds4_gpu_stream_expert_pread_task *tasks = NULL;
    uint32_t n_loads = 0;
    uint32_t n_tasks = 0;
    double load_prepare_ms = 0.0;
    double load_modify_ms = 0.0;
    double load_install_ms = 0.0;
    double load_timing_t0 = ds4_gpu_stream_expert_timing_summary_enabled() ?
                            ds4_gpu_now_ms() : 0.0;
    const int load_timing = load_timing_t0 != 0.0;
    if (ok && unique_count != 0) {
        tasks = calloc((size_t)unique_count * 3u, sizeof(tasks[0]));
        if (!tasks) ok = 0;
    }
    if (ok) {
        const uint32_t cache_budget =
            ds4_gpu_stream_expert_cache_configured_budget();
        uint32_t reserved_entries = g_stream_expert_cache_entry_count;

        for (uint32_t u = 0; u < unique_count; u++) {
            const uint32_t expert = (uint32_t)unique_ids[u];

            if ((uint64_t)expert > UINT64_MAX / gate_expert_bytes ||
                (uint64_t)expert > UINT64_MAX / down_expert_bytes) {
                fprintf(stderr, "ds4: Metal streaming batch selected expert offset overflow\n");
                ok = 0;
                break;
            }
            const uint64_t gate_rel = (uint64_t)expert * gate_expert_bytes;
            const uint64_t down_rel = (uint64_t)expert * down_expert_bytes;
            if (gate_rel > UINT64_MAX - gate_offset ||
                gate_rel > UINT64_MAX - up_offset ||
                down_rel > UINT64_MAX - down_offset) {
                fprintf(stderr, "ds4: Metal streaming batch selected expert offset overflow\n");
                ok = 0;
                break;
            }
            unique_gate_offsets[u] = gate_offset + gate_rel;
            unique_up_offsets[u] = up_offset + gate_rel;
            unique_down_offsets[u] = down_offset + down_rel;

            ds4_gpu_stream_expert_cache_entry *entry =
                ds4_gpu_stream_expert_cache_peek(model_map,
                                                 model_size,
                                                 layer,
                                                 expert,
                                                 n_total_expert,
                                                 n_selected,
                                                 unique_gate_offsets[u],
                                                 unique_up_offsets[u],
                                                 unique_down_offsets[u],
                                                 gate_expert_bytes,
                                                 down_expert_bytes);
            if (entry) {
                unique_entries[u] = entry;
                continue;
            }

            const int force_reuse =
                cache_budget != 0 && reserved_entries >= cache_budget;
            ds4_gpu_stream_expert_readahead_range(unique_gate_offsets[u],
                                                  gate_expert_bytes);
            ds4_gpu_stream_expert_readahead_range(unique_up_offsets[u],
                                                  gate_expert_bytes);
            ds4_gpu_stream_expert_readahead_range(unique_down_offsets[u],
                                                  down_expert_bytes);
            const double buffer_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
            const int prepared =
                ds4_gpu_stream_expert_cache_prepare_load_buffers(layer,
                                                                 expert,
                                                                 layer,
                                                                 unique_ids,
                                                                 unique_count,
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes,
                                                                 force_reuse,
                                                                 &gate_bufs[n_loads],
                                                                 &up_bufs[n_loads],
                                                                 &down_bufs[n_loads],
                                                                 &gate_inners[n_loads],
                                                                 &up_inners[n_loads],
                                                                 &down_inners[n_loads]);
            if (load_timing) {
                ds4_gpu_stream_expert_timing_note_prepare_buffer(
                        ds4_gpu_now_ms() - buffer_t0);
            }
            if (!prepared) {
                ok = 0;
                break;
            }
            if (!force_reuse && reserved_entries < UINT32_MAX) {
                reserved_entries++;
            }
            if (!gate_bufs[n_loads] ||
                !up_bufs[n_loads] ||
                !down_bufs[n_loads]) {
                ok = 0;
                break;
            }

            uint8_t *gate_dst = (uint8_t *)[gate_bufs[n_loads] contents] +
                                 gate_inners[n_loads];
            uint8_t *up_dst = (uint8_t *)[up_bufs[n_loads] contents] +
                               up_inners[n_loads];
            uint8_t *down_dst = (uint8_t *)[down_bufs[n_loads] contents] +
                                 down_inners[n_loads];
            if (!gate_dst || !up_dst || !down_dst) {
                ok = 0;
                break;
            }

            load_unique[n_loads] = u;
            const double task_t0 = load_timing ? ds4_gpu_now_ms() : 0.0;
            tasks[n_tasks++] = (ds4_gpu_stream_expert_pread_task) {
                .offset = unique_gate_offsets[u],
                .len = gate_expert_bytes,
                .dst = gate_dst,
            };
            tasks[n_tasks++] = (ds4_gpu_stream_expert_pread_task) {
                .offset = unique_up_offsets[u],
                .len = gate_expert_bytes,
                .dst = up_dst,
            };
            tasks[n_tasks++] = (ds4_gpu_stream_expert_pread_task) {
                .offset = unique_down_offsets[u],
                .len = down_expert_bytes,
                .dst = down_dst,
            };
            if (load_timing) {
                ds4_gpu_stream_expert_timing_note_prepare_task(
                        1,
                        ds4_gpu_now_ms() - task_t0);
            }
            n_loads++;
        }
    }
    if (ok && n_loads != 0) {
        if (load_timing_t0 != 0.0) {
            const double now_ms = ds4_gpu_now_ms();
            load_prepare_ms = now_ms - load_timing_t0;
            load_timing_t0 = now_ms;
        }
        uint64_t read_bytes = 0;
        double read_ms = 0.0;
        ok = ds4_gpu_stream_expert_pread_tasks(tasks,
                                               n_tasks,
                                               &read_bytes,
                                               &read_ms);
        if (ok) {
            ds4_gpu_stream_expert_cache_note_pread(layer, read_bytes, read_ms);
        }
        if (load_timing_t0 != 0.0) {
            load_timing_t0 = ds4_gpu_now_ms();
        }
        if (ok) {
            for (uint32_t load_i = 0; load_i < n_loads; load_i++) {
                [gate_bufs[load_i] didModifyRange:NSMakeRange(gate_inners[load_i], (NSUInteger)gate_expert_bytes)];
                [up_bufs[load_i] didModifyRange:NSMakeRange(up_inners[load_i], (NSUInteger)gate_expert_bytes)];
                [down_bufs[load_i] didModifyRange:NSMakeRange(down_inners[load_i], (NSUInteger)down_expert_bytes)];
            }
        }
        if (load_timing_t0 != 0.0) {
            const double now_ms = ds4_gpu_now_ms();
            load_modify_ms = now_ms - load_timing_t0;
            load_timing_t0 = now_ms;
        }
        if (ok && getenv("DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE") != NULL) {
            fprintf(stderr,
                    "ds4: Metal streaming batch expert parallel pread layer=%u experts=%u tensors=%u "
                    "threads=%u bytes=%.2f GiB wall=%.3f ms\n",
                    layer,
                    n_loads,
                    n_tasks,
                    ds4_gpu_stream_expert_pread_thread_count(n_tasks),
                    ds4_gpu_gib(read_bytes),
                    read_ms);
        }
        if (ok) {
            for (uint32_t load_i = 0; load_i < n_loads; load_i++) {
                const uint32_t u = load_unique[load_i];
                const uint32_t expert = (uint32_t)unique_ids[u];
                ds4_gpu_stream_expert_cache_entry *entry =
                    ds4_gpu_stream_expert_cache_install_loaded(model_map,
                                                               model_size,
                                                               layer,
                                                               expert,
                                                               unique_gate_offsets[u],
                                                               unique_up_offsets[u],
                                                               unique_down_offsets[u],
                                                               gate_expert_bytes,
                                                               down_expert_bytes,
                                                               gate_bufs[load_i],
                                                               up_bufs[load_i],
                                                               down_bufs[load_i],
                                                               gate_inners[load_i],
                                                               up_inners[load_i],
                                                               down_inners[load_i]);
                if (!entry) {
                    ok = 0;
                    break;
                }
                unique_entries[u] = entry;
            }
        }
        if (load_timing_t0 != 0.0) {
            load_install_ms = ds4_gpu_now_ms() - load_timing_t0;
            ds4_gpu_stream_expert_timing_note_load_detail(load_prepare_ms,
                                                          read_ms,
                                                          load_modify_ms,
                                                          load_install_ms);
        }
    }
    if (tasks) free(tasks);
    if (ok) {
        for (uint32_t u = 0; u < unique_count; u++) {
            ds4_gpu_stream_expert_cache_entry *entry = unique_entries[u];
            const uint32_t expert = (uint32_t)unique_ids[u];
            if (!entry) {
                const uint64_t gate_rel = unique_gate_offsets[u] - gate_offset;
                const uint64_t down_rel = unique_down_offsets[u] - down_offset;
                if (!*overflow_gate) {
                    const uint64_t gate_tensor_bytes =
                        (uint64_t)n_total_expert * gate_expert_bytes;
                    const uint64_t down_tensor_bytes =
                        (uint64_t)n_total_expert * down_expert_bytes;
                    uint64_t gate_view_inner = 0;
                    uint64_t up_view_inner = 0;
                    uint64_t down_view_inner = 0;
                    id<MTLBuffer> gv = ds4_gpu_wrap_model_range(model_map,
                                                                model_size,
                                                                gate_offset,
                                                                gate_tensor_bytes,
                                                                &gate_view_inner);
                    id<MTLBuffer> uv = ds4_gpu_wrap_model_range(model_map,
                                                                model_size,
                                                                up_offset,
                                                                gate_tensor_bytes,
                                                                &up_view_inner);
                    id<MTLBuffer> dv = ds4_gpu_wrap_model_range(model_map,
                                                                model_size,
                                                                down_offset,
                                                                down_tensor_bytes,
                                                                &down_view_inner);
                    if (!gv || !uv || !dv) {
                        fprintf(stderr,
                                "ds4: Metal streaming prefill batch selected addr "
                                "failed to map overflow expert views at layer %u\n",
                                layer);
                        ok = 0;
                        break;
                    }
                    *overflow_gate = gv;
                    *overflow_up = uv;
                    *overflow_down = dv;
                    overflow_gate_inner = gate_view_inner;
                    overflow_up_inner = up_view_inner;
                    overflow_down_inner = down_view_inner;
                }
                if (!ds4_gpu_stream_expert_cache_set_addr_slot(
                            layer,
                            expert,
                            *overflow_gate,
                            (NSUInteger)(overflow_gate_inner + gate_rel),
                            *overflow_up,
                            (NSUInteger)(overflow_up_inner + gate_rel),
                            *overflow_down,
                            (NSUInteger)(overflow_down_inner + down_rel))) {
                    ok = 0;
                    break;
                }
                view_served++;
                continue;
            }
            const uint32_t extra_uses =
                frequency[expert] > 0 ? frequency[expert] - 1u : 0;
            if (extra_uses != 0) {
                if (entry->use_count > UINT64_MAX - extra_uses) {
                    entry->use_count = UINT64_MAX;
                } else {
                    entry->use_count += extra_uses;
                }
                if (g_stream_expert_cache_hits > UINT64_MAX - extra_uses) {
                    g_stream_expert_cache_hits = UINT64_MAX;
                } else {
                    g_stream_expert_cache_hits += extra_uses;
                }
                if (g_stream_expert_cache_layer_hits[layer] >
                    UINT64_MAX - extra_uses) {
                    g_stream_expert_cache_layer_hits[layer] = UINT64_MAX;
                } else {
                    g_stream_expert_cache_layer_hits[layer] += extra_uses;
                }
            }
            resources[*n_resources] = entry;
            (*n_resources)++;
        }
    }
    free(ids);

    if (!ok || (*n_resources == 0 && view_served == 0)) {
        ds4_gpu_stream_expert_cache_clear_layer(layer);
        return 0;
    }
    if (!ds4_gpu_stream_expert_cache_addr_buffers(layer,
                                                  gate_addrs,
                                                  up_addrs,
                                                  down_addrs)) {
        ds4_gpu_stream_expert_cache_clear_layer(layer);
        return 0;
    }
    *unique_out = *n_resources + view_served;
    if (view_served != 0 &&
        ds4_gpu_stream_expert_timing_summary_enabled()) {
        fprintf(stderr,
                "ds4: Metal streaming prefill batch selected addr layer=%u "
                "served %u/%u unique experts via mapped views (cache budget %u)\n",
                layer,
                view_served,
                unique_count,
                ds4_gpu_stream_expert_cache_configured_budget());
    }
    if (profile) {
        const double t_done = ds4_gpu_now_ms();
        uint64_t per_expert_bytes = UINT64_MAX;
        uint64_t logical_bytes = UINT64_MAX;
        if (gate_expert_bytes <= (UINT64_MAX - down_expert_bytes) / 2ull) {
            per_expert_bytes = gate_expert_bytes * 2ull + down_expert_bytes;
            if (per_expert_bytes == 0 ||
                *n_resources <= UINT64_MAX / per_expert_bytes) {
                logical_bytes = (uint64_t)(*n_resources) * per_expert_bytes;
            }
        }
        fprintf(stderr,
                "ds4: Metal streaming prefill batch selected addr layer=%u "
                "tokens=%u unique=%u read=%.3f ms wrap=%.3f ms bytes=%.2f GiB\n",
                layer,
                n_tokens,
                *n_resources,
                t_read - t0,
                t_done - t_read,
                ds4_gpu_gib(logical_bytes));
    }
    return 1;
}

int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !selected_ids ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_selected == 0 ||
        n_selected > 6 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        !ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                      down_expert_bytes) ||
        ds4_gpu_stream_expert_cache_effective_cap(layer,
                                                  n_total_expert,
                                                  n_selected) == 0) {
        return 1;
    }
    if (!g_initialized && !ds4_gpu_init()) return 0;

    ds4_gpu_stream_expert_cache_note_selected_hotness(layer,
                                                      selected_ids,
                                                      n_selected);
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: Metal prefill expert-cache seed selected expert id %d is outside 0..%u\n",
                    selected_ids[i],
                    n_total_expert);
            return 0;
        }
        const uint64_t expert_id = (uint64_t)(uint32_t)selected_ids[i];
        if (expert_id > UINT64_MAX / gate_expert_bytes ||
            expert_id > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: Metal prefill expert-cache seed offset overflow\n");
            return 0;
        }
        const uint64_t gate_rel = expert_id * gate_expert_bytes;
        const uint64_t down_rel = expert_id * down_expert_bytes;
        if (gate_rel > UINT64_MAX - gate_offset ||
            gate_rel > UINT64_MAX - up_offset ||
            down_rel > UINT64_MAX - down_offset) {
            fprintf(stderr, "ds4: Metal prefill expert-cache seed offset overflow\n");
            return 0;
        }

        if (!ds4_gpu_stream_expert_cache_get(model_map,
                                             model_size,
                                             layer,
                                             (uint32_t)selected_ids[i],
                                             n_total_expert,
                                             n_selected,
                                             gate_offset + gate_rel,
                                             up_offset + gate_rel,
                                             down_offset + down_rel,
                                             gate_expert_bytes,
                                             down_expert_bytes)) {
            return 0;
        }
    }
    ds4_gpu_stream_expert_cache_prune_layer(layer,
                                            n_total_expert,
                                            n_selected,
                                            selected_ids,
                                            n_selected);
    ds4_gpu_stream_expert_cache_prune_global(layer,
                                             selected_ids,
                                             n_selected);
    return 1;
}

int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !expert_ids ||
        layer >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER ||
        n_experts == 0 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT ||
        !ds4_gpu_stream_expert_cache_note_expert_size(gate_expert_bytes,
                                                      down_expert_bytes) ||
        ds4_gpu_stream_expert_cache_effective_cap(layer,
                                                  n_total_expert,
                                                  1) == 0) {
        return 1;
    }
    if (!g_initialized && !ds4_gpu_init()) return 0;

    ds4_gpu_stream_expert_cache_maybe_decay_route_hotness();
    uint32_t remaining = n_experts;
    while (remaining != 0) {
        const uint32_t batch =
            remaining > 6u ? 6u : remaining;
        remaining -= batch;

        int32_t selected_ids[6] = { -1, -1, -1, -1, -1, -1 };
        uint64_t gate_abs_offsets[6] = { 0, 0, 0, 0, 0, 0 };
        uint64_t up_abs_offsets[6] = { 0, 0, 0, 0, 0, 0 };
        uint64_t down_abs_offsets[6] = { 0, 0, 0, 0, 0, 0 };
        ds4_gpu_stream_expert_cache_entry *entries[6] = {
            NULL, NULL, NULL, NULL, NULL, NULL
        };
        uint32_t missing_mask = 0;

        for (uint32_t i = 0; i < batch; i++) {
            const int32_t expert = expert_ids[remaining + i];
            const uint32_t priority =
                expert_priorities ? expert_priorities[remaining + i] : 0;
            if (expert < 0 || (uint32_t)expert >= n_total_expert) {
                fprintf(stderr,
                        "ds4: Metal streaming hotlist seed expert id %d is outside 0..%u\n",
                        expert,
                        n_total_expert);
                return 0;
            }
            const uint64_t expert_id = (uint64_t)(uint32_t)expert;
            if (expert_id > UINT64_MAX / gate_expert_bytes ||
                expert_id > UINT64_MAX / down_expert_bytes) {
                fprintf(stderr, "ds4: Metal streaming hotlist seed offset overflow\n");
                return 0;
            }
            const uint64_t gate_rel = expert_id * gate_expert_bytes;
            const uint64_t down_rel = expert_id * down_expert_bytes;
            if (gate_rel > UINT64_MAX - gate_offset ||
                gate_rel > UINT64_MAX - up_offset ||
                down_rel > UINT64_MAX - down_offset) {
                fprintf(stderr, "ds4: Metal streaming hotlist seed offset overflow\n");
                return 0;
            }

            ds4_gpu_stream_expert_cache_note_route_hotness(
                    layer,
                    (uint32_t)expert,
                    priority != 0 ? priority : 1u);
            selected_ids[i] = expert;
            gate_abs_offsets[i] = gate_offset + gate_rel;
            up_abs_offsets[i] = up_offset + gate_rel;
            down_abs_offsets[i] = down_offset + down_rel;
            entries[i] = ds4_gpu_stream_expert_cache_peek(model_map,
                                                          model_size,
                                                          layer,
                                                          (uint32_t)expert,
                                                          n_total_expert,
                                                          batch,
                                                          gate_abs_offsets[i],
                                                          up_abs_offsets[i],
                                                          down_abs_offsets[i],
                                                          gate_expert_bytes,
                                                          down_expert_bytes);
            if (!entries[i]) missing_mask |= 1u << i;
            if (entries[i] && priority != 0 &&
                entries[i]->use_count < (uint64_t)priority) {
                entries[i]->use_count = (uint64_t)priority;
            }
        }

        if (missing_mask != 0 &&
            !ds4_gpu_stream_expert_cache_load_selected_missing(model_map,
                                                               model_size,
                                                               layer,
                                                               selected_ids,
                                                               n_total_expert,
                                                               batch,
                                                               gate_abs_offsets,
                                                               up_abs_offsets,
                                                               down_abs_offsets,
                                                               gate_expert_bytes,
                                                               down_expert_bytes,
                                                               missing_mask,
                                                               entries)) {
            return 0;
        }
        if (expert_priorities) {
            for (uint32_t i = 0; i < batch; i++) {
                const uint32_t priority = expert_priorities[remaining + i];
                if (entries[i] && priority != 0 &&
                    entries[i]->use_count < (uint64_t)priority) {
                    entries[i]->use_count = (uint64_t)priority;
                }
            }
        }
    }

    const uint32_t protect_n = n_experts < 6u ? n_experts : 6u;
    ds4_gpu_stream_expert_cache_prune_layer(layer,
                                            n_total_expert,
                                            1,
                                            expert_ids,
                                            protect_n);
    ds4_gpu_stream_expert_cache_prune_global(layer,
                                             expert_ids,
                                             protect_n);
    return 1;
}

static uint32_t ds4_gpu_q4_expert_table_group_size(uint32_t n_total_expert) {
    const char *env = getenv("DS4_METAL_Q4_EXPERT_TABLE_GROUP_SIZE");
    if (!env || !env[0]) return 1;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || v < 2 || v > n_total_expert) {
        return 1;
    }
    return (uint32_t)v;
}

static bool ds4_gpu_q4_table_queue_residency_requested(void) {
    return getenv("DS4_METAL_Q4_TABLE_QUEUE_RESIDENCY_SET") != NULL;
}

static bool ds4_gpu_q4_table_queue_residency_available(void) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        return g_queue && [g_queue respondsToSelector:@selector(addResidencySet:)];
    }
#endif
    return false;
}

static bool ds4_gpu_q4_non_streaming_opt_in_enabled(void) {
    return getenv("DS4_METAL_ENABLE_Q4_SELECTED_EXPERT_VIEWS") != NULL ||
           getenv("DS4_METAL_ENABLE_PRO_Q4_SELECTED_EXPERT_VIEWS") != NULL ||
           getenv("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") != NULL ||
           getenv("DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE") != NULL ||
           getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") != NULL ||
           getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO") != NULL;
}

static bool ds4_gpu_q4_selected_paths_allowed(void) {
    if (g_ssd_streaming_mode) return true;
    if (g_glm_model_mode) return false;
    return ds4_gpu_q4_non_streaming_opt_in_enabled();
}

int ds4_gpu_pro_q4_expert_table_auto_available(void) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    return (g_ssd_streaming_mode ||
            getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") != NULL) &&
           getenv("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO") == NULL &&
           getenv("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") == NULL &&
           ds4_gpu_q4_table_queue_residency_available();
}

static bool ds4_gpu_q4_table_queue_residency_enabled(bool auto_queue_residency) {
    return (auto_queue_residency || ds4_gpu_q4_table_queue_residency_requested()) &&
           ds4_gpu_q4_table_queue_residency_available();
}

static bool ds4_gpu_q4_table_model_residency_enabled(void) {
    return getenv("DS4_METAL_Q4_TABLE_MODEL_RESIDENCY_SET") != NULL;
}

static bool ds4_gpu_pro_q4_expert_indirect_shape_supported(
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint64_t gate_tensor_bytes,
        uint64_t down_tensor_bytes) {
    const uint64_t q4_selected_min_tensor_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    return n_total_expert == 384 &&
           n_expert == 6 &&
           gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
           down_tensor_bytes >= q4_selected_min_tensor_bytes;
}

static bool ds4_gpu_pro_q4_expert_table_auto_enabled(
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint64_t gate_tensor_bytes,
        uint64_t down_tensor_bytes) {
    /*
     * This path lets the shader choose among exact per-expert resources.
     * It is only automatic when a Metal queue residency set can make every
     * indirect expert resource visible up front; otherwise the selected
     * exact-slice path remains the fallback.
     */
    return ds4_gpu_pro_q4_expert_indirect_shape_supported(n_total_expert,
                                                          n_expert,
                                                          gate_tensor_bytes,
                                                          down_tensor_bytes) &&
           ds4_gpu_pro_q4_expert_table_auto_available();
}

static bool ds4_gpu_pro_q4_expert_address_auto_enabled(
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint64_t gate_tensor_bytes,
        uint64_t down_tensor_bytes) {
    /*
     * GPU-address expert tables are useful for experiments, but they are not
     * safe as an automatic path unless every indirect expert resource is made
     * visible to Metal. Keep this behind an explicit opt-in while the selected
     * active-slice path remains the correctness/performance baseline.
     */
    return getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO") != NULL &&
           ds4_gpu_q4_selected_paths_allowed() &&
           ds4_gpu_pro_q4_expert_indirect_shape_supported(n_total_expert,
                                                          n_expert,
                                                          gate_tensor_bytes,
                                                          down_tensor_bytes) &&
           g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline != nil &&
           g_moe_mul_mv_addr_q4_k_sum6_pipeline != nil &&
           getenv("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO") == NULL &&
           getenv("DS4_METAL_DISABLE_PRO_Q4_EXPERT_ADDRESS_AUTO") == NULL &&
           getenv("DS4_METAL_DISABLE_Q4_EXPERT_ADDRESS_TABLE") == NULL &&
           ds4_gpu_q4_table_queue_residency_available();
}

static bool ds4_gpu_q4_table_bind_anchors_enabled(void) {
    return getenv("DS4_METAL_Q4_TABLE_BIND_ANCHORS") != NULL;
}

static int ds4_gpu_use_model_residency_set(id<MTLCommandBuffer> cb) {
    if (!ds4_gpu_q4_table_model_residency_enabled()) return 1;
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (cb && g_model_residency_set && [cb respondsToSelector:@selector(useResidencySet:)]) {
            [cb useResidencySet:g_model_residency_set];
            return 1;
        }
    }
#endif
    fprintf(stderr, "ds4: Metal Q4 table model residency set is not available\n");
    return 0;
}

static int ds4_gpu_bind_q4_expert_table_anchors(
        id<MTLComputeCommandEncoder> enc,
        DS4MetalQ4ExpertTable       *table,
        NSUInteger                   first_index,
        NSUInteger                   max_count) {
    if (!ds4_gpu_q4_table_bind_anchors_enabled()) return 1;
    if (!enc || !table || !table.expertBuffers) return 0;

    const NSUInteger count = [table.expertBuffers count];
    if (count > max_count) {
        fprintf(stderr,
                "ds4: Metal Q4 table anchor count %lu exceeds available slots %lu\n",
                (unsigned long)count,
                (unsigned long)max_count);
        return 0;
    }
    for (NSUInteger i = 0; i < count; i++) {
        [enc setBuffer:[table.expertBuffers objectAtIndex:i]
                offset:0
               atIndex:first_index + i];
    }
    return 1;
}

static id ds4_gpu_q4_expert_table_residency_set(NSMutableArray<id<MTLBuffer>> *buffers) {
    if (!buffers || [buffers count] == 0 ||
        getenv("DS4_METAL_Q4_TABLE_RESIDENCY_SET") == NULL ||
        getenv("DS4_METAL_Q4_TABLE_PER_TENSOR_RESIDENCY_SET") == NULL) {
        return nil;
    }
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ds4_q4_expert_table";
        desc.initialCapacity = [buffers count];
        NSError *error = nil;
        id residency_set = [g_device newResidencySetWithDescriptor:desc error:&error];
        if (!residency_set) {
            fprintf(stderr, "ds4: Metal Q4 expert table residency set creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return nil;
        }
        for (id<MTLBuffer> buffer in buffers) {
            [residency_set addAllocation:buffer];
        }
        [residency_set commit];
        [residency_set requestResidency];
        return residency_set;
    }
#endif
    return nil;
}

static void ds4_gpu_q4_residency_add_table(id residency_set,
                                           DS4MetalQ4ExpertTable *table) {
#if TARGET_OS_OSX
    if (!residency_set || !table) return;
    if (@available(macOS 15.0, *)) {
        if (table.argumentBuffer) {
            [residency_set addAllocation:table.argumentBuffer];
        }
        if (table.addressBuffer) {
            [residency_set addAllocation:table.addressBuffer];
        }
        for (id<MTLBuffer> buffer in table.expertBuffers) {
            [residency_set addAllocation:buffer];
        }
    }
#else
    (void)residency_set;
    (void)table;
#endif
}

static id ds4_gpu_q4_expert_layer_residency_set(DS4MetalQ4ExpertTable *gate_table,
                                                DS4MetalQ4ExpertTable *up_table,
                                                DS4MetalQ4ExpertTable *down_table,
                                                bool auto_queue_residency) {
    const bool queue_residency =
        ds4_gpu_q4_table_queue_residency_enabled(auto_queue_residency);
    if (!gate_table || !up_table || !down_table ||
        !g_device || !g_q4_expert_layer_residency_cache ||
        (getenv("DS4_METAL_Q4_TABLE_RESIDENCY_SET") == NULL &&
         !queue_residency)) {
        return nil;
    }
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        NSString *key = [NSString stringWithFormat:@"%p:%p:%p",
                         gate_table, up_table, down_table];
        DS4MetalQ4LayerResidency *cached =
            [g_q4_expert_layer_residency_cache objectForKey:key];
        if (cached) return cached.residencySet;

        const NSUInteger capacity =
            [gate_table.expertBuffers count] +
            [up_table.expertBuffers count] +
            [down_table.expertBuffers count] + 3u;
        MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ds4_q4_expert_layer";
        desc.initialCapacity = capacity;
        NSError *error = nil;
        id residency_set = [g_device newResidencySetWithDescriptor:desc error:&error];
        if (!residency_set) {
            fprintf(stderr,
                    "ds4: Metal Q4 expert layer residency set creation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return nil;
        }

        ds4_gpu_q4_residency_add_table(residency_set, gate_table);
        ds4_gpu_q4_residency_add_table(residency_set, up_table);
        ds4_gpu_q4_residency_add_table(residency_set, down_table);
        [residency_set commit];
        [residency_set requestResidency];

        DS4MetalQ4LayerResidency *entry = [DS4MetalQ4LayerResidency new];
        entry.residencySet = residency_set;
        if (queue_residency) {
            [g_queue addResidencySet:residency_set];
            entry.addedToQueue = YES;
        }
        [g_q4_expert_layer_residency_cache setObject:entry forKey:key];
        return residency_set;
    }
#endif
    return nil;
}

static DS4MetalQ4ExpertTable *ds4_gpu_q4_expert_table(
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                tensor_offset,
        uint64_t                expert_bytes,
        uint32_t                n_total_expert,
        id<MTLArgumentEncoder>  encoder);

static DS4MetalQ4ExpertTable *ds4_gpu_q4_expert_address_table(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    tensor_offset,
        uint64_t    expert_bytes,
        uint32_t    n_total_expert);

int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                     uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                     uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                     uint32_t n_total_expert) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (!model_map || model_size == 0 || gate_expert_bytes == 0 ||
        down_expert_bytes == 0 || n_total_expert == 0) {
        return 0;
    }
    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: Metal Q4 expert table preload byte size overflow\n");
        return 0;
    }

    const uint64_t gate_tensor_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_tensor_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    const bool address_auto =
        ds4_gpu_pro_q4_expert_address_auto_enabled(n_total_expert,
                                                   6,
                                                   gate_tensor_bytes,
                                                   down_tensor_bytes);
    const bool table_auto =
        ds4_gpu_pro_q4_expert_table_auto_enabled(n_total_expert,
                                                 6,
                                                 gate_tensor_bytes,
                                                 down_tensor_bytes);
    if (!address_auto && !table_auto) {
        return 1;
    }

    if (address_auto) {
        @autoreleasepool {
            DS4MetalQ4ExpertTable *gate_table =
                ds4_gpu_q4_expert_address_table(model_map,
                                                model_size,
                                                gate_offset,
                                                gate_expert_bytes,
                                                n_total_expert);
            DS4MetalQ4ExpertTable *up_table =
                ds4_gpu_q4_expert_address_table(model_map,
                                                model_size,
                                                up_offset,
                                                gate_expert_bytes,
                                                n_total_expert);
            DS4MetalQ4ExpertTable *down_table =
                ds4_gpu_q4_expert_address_table(model_map,
                                                model_size,
                                                down_offset,
                                                down_expert_bytes,
                                                n_total_expert);
            if (!gate_table || !up_table || !down_table) {
                return 0;
            }
            id residency = ds4_gpu_q4_expert_layer_residency_set(gate_table,
                                                                 up_table,
                                                                 down_table,
                                                                 true);
            if (!residency) {
                fprintf(stderr, "ds4: Metal Q4 expert address table preload failed to create queue residency set\n");
                return 0;
            }
        }
        return 1;
    }

    if (!g_moe_table_q4_pair_gate_encoder || !g_moe_table_q4_pair_up_encoder ||
        !g_moe_table_q4_sum_down_encoder) {
        fprintf(stderr, "ds4: Metal Q4 expert table preload missing argument encoders\n");
        return 0;
    }

    @autoreleasepool {
        DS4MetalQ4ExpertTable *gate_table =
            ds4_gpu_q4_expert_table(model_map,
                                    model_size,
                                    gate_offset,
                                    gate_expert_bytes,
                                    n_total_expert,
                                    g_moe_table_q4_pair_gate_encoder);
        DS4MetalQ4ExpertTable *up_table =
            ds4_gpu_q4_expert_table(model_map,
                                    model_size,
                                    up_offset,
                                    gate_expert_bytes,
                                    n_total_expert,
                                    g_moe_table_q4_pair_up_encoder);
        DS4MetalQ4ExpertTable *down_table =
            ds4_gpu_q4_expert_table(model_map,
                                    model_size,
                                    down_offset,
                                    down_expert_bytes,
                                    n_total_expert,
                                    g_moe_table_q4_sum_down_encoder);
        if (!gate_table || !up_table || !down_table) {
            return 0;
        }
        id residency = ds4_gpu_q4_expert_layer_residency_set(gate_table,
                                                             up_table,
                                                             down_table,
                                                             true);
        if (!residency) {
            fprintf(stderr, "ds4: Metal Q4 expert table preload failed to create queue residency set\n");
            return 0;
        }
    }
    return 1;
}

static DS4MetalQ4ExpertTable *ds4_gpu_q4_expert_table(
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                tensor_offset,
        uint64_t                expert_bytes,
        uint32_t                n_total_expert,
        id<MTLArgumentEncoder>  encoder) {
    if (!model_map || !g_device || !g_q4_expert_table_cache || !encoder ||
        model_size == 0 || expert_bytes == 0 ||
        n_total_expert == 0 || n_total_expert > 384) {
        return nil;
    }
    if ((uint64_t)n_total_expert > UINT64_MAX / expert_bytes) {
        fprintf(stderr, "ds4: Metal Q4 expert table byte size overflow\n");
        return nil;
    }
    const uint64_t tensor_bytes = (uint64_t)n_total_expert * expert_bytes;
    if (tensor_offset > model_size || tensor_bytes > model_size - tensor_offset) {
        fprintf(stderr, "ds4: Metal Q4 expert table is outside the mapped model\n");
        return nil;
    }

    const uint32_t table_group_size =
        ds4_gpu_q4_expert_table_group_size(n_total_expert);
    NSString *key = [NSString stringWithFormat:@"%p:%llu:%llu:%llu:%u:%llu:%u",
                     model_map,
                     (unsigned long long)model_size,
                     (unsigned long long)tensor_offset,
                     (unsigned long long)expert_bytes,
                     n_total_expert,
                     (unsigned long long)[encoder encodedLength],
                     table_group_size];
    DS4MetalQ4ExpertTable *cached = [g_q4_expert_table_cache objectForKey:key];
    if (cached) return cached;

    id<MTLBuffer> arg_buffer =
        [g_device newBufferWithLength:[encoder encodedLength]
                               options:MTLResourceStorageModeShared];
    if (!arg_buffer) {
        fprintf(stderr, "ds4: Metal Q4 expert table argument buffer allocation failed\n");
        return nil;
    }
    arg_buffer.label = @"ds4_q4_expert_table";
    [encoder setArgumentBuffer:arg_buffer offset:0];

    NSMutableArray<id<MTLBuffer>> *expert_buffers =
        [NSMutableArray arrayWithCapacity:table_group_size > 1 ?
            (n_total_expert + table_group_size - 1u) / table_group_size :
            n_total_expert];
    if (!expert_buffers) return nil;

    if (table_group_size > 1) {
        for (uint32_t first = 0; first < n_total_expert; first += table_group_size) {
            const uint32_t remaining = n_total_expert - first;
            const uint32_t group_n =
                remaining < table_group_size ? remaining : table_group_size;
            if ((uint64_t)first > UINT64_MAX / expert_bytes ||
                (uint64_t)group_n > UINT64_MAX / expert_bytes) {
                fprintf(stderr, "ds4: Metal Q4 expert table group byte overflow\n");
                return nil;
            }
            const uint64_t rel = (uint64_t)first * expert_bytes;
            const uint64_t group_bytes = (uint64_t)group_n * expert_bytes;
            if (rel > UINT64_MAX - tensor_offset) {
                fprintf(stderr, "ds4: Metal Q4 expert table group offset overflow\n");
                return nil;
            }
            uint64_t inner = 0;
            id<MTLBuffer> group_buf =
                ds4_gpu_wrap_model_range(model_map,
                                         model_size,
                                         tensor_offset + rel,
                                         group_bytes,
                                         &inner);
            if (!group_buf) return nil;
            for (uint32_t j = 0; j < group_n; j++) {
                const uint64_t expert_inner = inner + (uint64_t)j * expert_bytes;
                [encoder setBuffer:group_buf offset:(NSUInteger)expert_inner atIndex:first + j];
            }
            [expert_buffers addObject:group_buf];
        }
    } else {
        for (uint32_t i = 0; i < n_total_expert; i++) {
            const uint64_t rel = (uint64_t)i * expert_bytes;
            if (rel > UINT64_MAX - tensor_offset) {
                fprintf(stderr, "ds4: Metal Q4 expert table offset overflow\n");
                return nil;
            }
            uint64_t inner = 0;
            id<MTLBuffer> expert_buf =
                ds4_gpu_wrap_model_exact_range_owned(model_map,
                                                     model_size,
                                                     tensor_offset + rel,
                                                     expert_bytes,
                                                     &inner);
            if (!expert_buf) return nil;
            [encoder setBuffer:expert_buf offset:(NSUInteger)inner atIndex:i];
            [expert_buffers addObject:expert_buf];
        }
    }
    [arg_buffer didModifyRange:NSMakeRange(0, [encoder encodedLength])];

    if (getenv("DS4_METAL_Q4_EXPERT_TABLE_PROFILE") != NULL) {
        fprintf(stderr,
                "ds4: Metal Q4 expert table: experts=%u group=%u buffers=%lu expert_bytes=%.2f MiB\n",
                n_total_expert,
                table_group_size,
                (unsigned long)[expert_buffers count],
                ds4_gpu_mib(expert_bytes));
    }

    DS4MetalQ4ExpertTable *table = [DS4MetalQ4ExpertTable new];
    table.argumentBuffer = arg_buffer;
    table.expertBuffers = expert_buffers;
    table.residencySet = ds4_gpu_q4_expert_table_residency_set(expert_buffers);
    table.nExpert = n_total_expert;
    table.expertBytes = expert_bytes;
    [g_q4_expert_table_cache setObject:table forKey:key];
    return table;
}

static DS4MetalQ4ExpertTable *ds4_gpu_q4_expert_address_table(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    tensor_offset,
        uint64_t    expert_bytes,
        uint32_t    n_total_expert) {
    if (!model_map || !g_device || !g_q4_expert_table_cache ||
        model_size == 0 || expert_bytes == 0 ||
        n_total_expert == 0 || n_total_expert > 384) {
        return nil;
    }
    if ((uint64_t)n_total_expert > UINT64_MAX / expert_bytes) {
        fprintf(stderr, "ds4: Metal Q4 expert address table byte size overflow\n");
        return nil;
    }
    const uint64_t tensor_bytes = (uint64_t)n_total_expert * expert_bytes;
    if (tensor_offset > model_size || tensor_bytes > model_size - tensor_offset) {
        fprintf(stderr, "ds4: Metal Q4 expert address table is outside the mapped model\n");
        return nil;
    }

    const uint32_t table_group_size =
        ds4_gpu_q4_expert_table_group_size(n_total_expert);
    NSString *key = [NSString stringWithFormat:@"addr:%p:%llu:%llu:%llu:%u:%u",
                     model_map,
                     (unsigned long long)model_size,
                     (unsigned long long)tensor_offset,
                     (unsigned long long)expert_bytes,
                     n_total_expert,
                     table_group_size];
    DS4MetalQ4ExpertTable *cached = [g_q4_expert_table_cache objectForKey:key];
    if (cached) return cached;

    id<MTLBuffer> address_buffer =
        [g_device newBufferWithLength:(NSUInteger)n_total_expert * sizeof(uint64_t)
                               options:MTLResourceStorageModeShared];
    if (!address_buffer) {
        fprintf(stderr, "ds4: Metal Q4 expert address table allocation failed\n");
        return nil;
    }
    address_buffer.label = @"ds4_q4_expert_address_table";
    uint64_t *addresses = (uint64_t *)[address_buffer contents];
    if (!addresses) return nil;

    NSMutableArray<id<MTLBuffer>> *expert_buffers =
        [NSMutableArray arrayWithCapacity:table_group_size > 1 ?
            (n_total_expert + table_group_size - 1u) / table_group_size :
            n_total_expert];
    if (!expert_buffers) return nil;

#if TARGET_OS_OSX
    if (@available(macOS 13.0, *)) {
        for (uint32_t first = 0; first < n_total_expert; first += table_group_size) {
            const uint32_t remaining = n_total_expert - first;
            const uint32_t group_n =
                remaining < table_group_size ? remaining : table_group_size;
            if ((uint64_t)first > UINT64_MAX / expert_bytes ||
                (uint64_t)group_n > UINT64_MAX / expert_bytes) {
                fprintf(stderr, "ds4: Metal Q4 expert address table group byte overflow\n");
                return nil;
            }
            const uint64_t rel = (uint64_t)first * expert_bytes;
            const uint64_t group_bytes = (uint64_t)group_n * expert_bytes;
            if (rel > UINT64_MAX - tensor_offset) {
                fprintf(stderr, "ds4: Metal Q4 expert address table group offset overflow\n");
                return nil;
            }
            uint64_t inner = 0;
            id<MTLBuffer> group_buf = nil;
            if (table_group_size > 1) {
                group_buf = ds4_gpu_wrap_model_range(model_map,
                                                     model_size,
                                                     tensor_offset + rel,
                                                     group_bytes,
                                                     &inner);
            } else {
                group_buf = ds4_gpu_wrap_model_exact_range_owned(model_map,
                                                                 model_size,
                                                                 tensor_offset + rel,
                                                                 expert_bytes,
                                                                 &inner);
            }
            if (!group_buf) return nil;
            const uint64_t base_address = (uint64_t)[group_buf gpuAddress] + inner;
            for (uint32_t j = 0; j < group_n; j++) {
                addresses[first + j] = base_address + (uint64_t)j * expert_bytes;
            }
            [expert_buffers addObject:group_buf];
        }
    } else
#endif
    {
        fprintf(stderr, "ds4: Metal GPU addresses require macOS 13 or newer\n");
        return nil;
    }

    [address_buffer didModifyRange:NSMakeRange(0,
        (NSUInteger)n_total_expert * sizeof(uint64_t))];

    if (getenv("DS4_METAL_Q4_EXPERT_TABLE_PROFILE") != NULL) {
        fprintf(stderr,
                "ds4: Metal Q4 expert address table: experts=%u group=%u buffers=%lu expert_bytes=%.2f MiB\n",
                n_total_expert,
                table_group_size,
                (unsigned long)[expert_buffers count],
                ds4_gpu_mib(expert_bytes));
    }

    DS4MetalQ4ExpertTable *table = [DS4MetalQ4ExpertTable new];
    table.addressBuffer = address_buffer;
    table.expertBuffers = expert_buffers;
    table.nExpert = n_total_expert;
    table.expertBytes = expert_bytes;
    [g_q4_expert_table_cache setObject:table forKey:key];
    return table;
}

static void ds4_gpu_use_q4_expert_table_resources(
        id<MTLCommandBuffer>        cb,
        id<MTLComputeCommandEncoder> enc,
        DS4MetalQ4ExpertTable       *table,
        bool                         queue_residency) {
    if (!enc || !table) return;
    if (table.argumentBuffer) {
        [enc useResource:table.argumentBuffer usage:MTLResourceUsageRead];
    }
    if (table.addressBuffer) {
        [enc useResource:table.addressBuffer usage:MTLResourceUsageRead];
    }
    if (!queue_residency &&
        getenv("DS4_METAL_Q4_TABLE_RESIDENCY_SET") != NULL &&
        table.residencySet &&
        cb &&
        [cb respondsToSelector:@selector(useResidencySet:)]) {
        [cb useResidencySet:table.residencySet];
    }
    if (getenv("DS4_METAL_Q4_TABLE_USE_RESOURCES") == NULL &&
        getenv("DS4_METAL_Q4_ADDR_USE_RESOURCES") == NULL) {
        return;
    }
    const NSUInteger count = [table.expertBuffers count];
    for (NSUInteger base = 0; base < count; base += 64u) {
        const NSUInteger n = count - base < 64u ? count - base : 64u;
        __unsafe_unretained id<MTLResource> resources[64];
        for (NSUInteger i = 0; i < n; i++) {
            resources[i] = [table.expertBuffers objectAtIndex:base + i];
        }
        [enc useResources:resources count:n usage:MTLResourceUsageRead];
    }
}
