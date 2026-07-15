static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static int g_ssd_streaming_mode;
static cudaStream_t g_model_upload_stream;
static cudaStream_t g_selected_readback_stream;
static cudaEvent_t g_selected_readback_event;
static uint64_t g_selected_readback_event_value;
static cublasHandle_t g_cublas;
static int g_cublas_ready;
#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm_hipblaslt.cuh"
#endif
static int g_quality_mode;

enum {
    DS4_ROCM_N_EXPERT = 256u,
    DS4_ROCM_MAX_N_EXPERT = 384u,
    DS4_ROCM_N_EXPERT_USED = 6u,
    DS4_ROCM_STREAM_READ_WORKERS = DS4_ROCM_N_EXPERT_USED * 3u,
    DS4_ROCM_STREAM_READ_MAX_JOBS = DS4_ROCM_MAX_N_EXPERT * 3u,
    DS4_ROCM_COMPRESSOR_MAX_RATIO = 128u
};
#define DS4_ROCM_EXPERT_WEIGHT_SCALE 1.5f
#define DS4_ROCM_EXPERT_WEIGHT_SCALE_TOL 1.0e-6f

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct cuda_model_image {
    const void *host_base;
    uint64_t size;
    char *device_ptr;
};

struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_q8_f16_transpose_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_stream_selected_cache {
    int loaded;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    int32_t selected_ids[DS4_ROCM_N_EXPERT_USED];
    char *gate;
    char *up;
    char *down;
    uint64_t gate_capacity;
    uint64_t down_capacity;
    int32_t *slot_ids;
    const char **gate_ptrs;
    const char **up_ptrs;
    const char **down_ptrs;
    ds4_gpu_tensor slot_tensor;
};

struct cuda_stream_resident_expert {
    const void *model_map;
    uint32_t layer;
    int32_t expert;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    char *base;
    char *gate;
    char *up;
    char *down;
    uint64_t bytes;
    uint64_t last_used;
};

struct cuda_stream_resident_key {
    const void *model_map;
    uint32_t layer;
    int32_t expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;

    bool operator==(const cuda_stream_resident_key &o) const {
        return model_map == o.model_map &&
               layer == o.layer &&
               expert == o.expert &&
               gate_offset == o.gate_offset &&
               up_offset == o.up_offset &&
               down_offset == o.down_offset &&
               gate_expert_bytes == o.gate_expert_bytes &&
               down_expert_bytes == o.down_expert_bytes;
    }
};

struct cuda_stream_resident_key_hash {
    size_t operator()(const cuda_stream_resident_key &k) const {
        uint64_t h = (uint64_t)(uintptr_t)k.model_map;
        h ^= (uint64_t)k.layer + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= (uint64_t)(uint32_t)k.expert + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= k.gate_offset + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= k.up_offset + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= k.down_offset + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= k.gate_expert_bytes + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= k.down_expert_bytes + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

struct cuda_stream_batch_selected_cache {
    int loaded;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint32_t n_tokens;
    uint32_t n_unique;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    int32_t *selected_ids;
    uint64_t selected_capacity;
    uint8_t *pair_missing;
    uint64_t pair_missing_capacity;
    const char **gate_ptrs;
    const char **up_ptrs;
    const char **down_ptrs;
    const char **resident_gate_ptrs;
    const char **resident_up_ptrs;
    const char **missing_gate_ptrs;
    const char **missing_up_ptrs;
    uint32_t ptr_capacity;
    ds4_gpu_tensor selected_tensor;
};

struct cuda_stream_layer_expert_cache {
    int active;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t bytes;
    uint64_t capacity;
    char *base;
    char *gate;
    char *up;
    char *down;
};

struct cuda_stream_cache_stats {
    uint64_t selected_calls;
    uint64_t selected_slots;
    uint64_t selected_hits;
    uint64_t selected_misses;
    uint64_t batch_calls;
    uint64_t batch_unique;
    uint64_t batch_hits;
    uint64_t batch_misses;
    uint64_t seed_calls;
    uint64_t seed_unique;
    uint64_t layer_loads;
    uint64_t layer_load_bytes;
    uint64_t layer_resident_flushes;
    uint64_t allocs;
    uint64_t alloc_bytes;
    uint64_t evictions;
    uint64_t evict_bytes;
    uint64_t max_resident_count;
    uint64_t max_resident_bytes;
};

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::vector<cuda_model_image> g_model_images;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f16_transpose_range> g_q8_f16_transpose_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_transpose_by_offset;
static uint64_t g_model_range_bytes;
static uint64_t g_q8_f16_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_disabled_for_multi_model;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static void *g_cuda_tmp;
static uint64_t g_cuda_tmp_bytes;
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;
static uint32_t g_stream_expert_cache_budget;
static cuda_stream_selected_cache g_stream_selected_cache;
static cuda_stream_batch_selected_cache g_stream_batch_selected_cache;
static cuda_stream_layer_expert_cache g_stream_layer_expert_cache[2];
static std::vector<cuda_stream_resident_expert> g_stream_resident_experts;
static std::unordered_map<cuda_stream_resident_key,
                          size_t,
                          cuda_stream_resident_key_hash> g_stream_resident_index;
static uint64_t g_stream_resident_bytes;
static uint64_t g_stream_resident_clock;
static cuda_stream_cache_stats g_stream_cache_stats;
static int g_stream_cache_stats_enabled = -1;
static int32_t g_routed_moe_selected_override[DS4_ROCM_N_EXPERT_USED];
static uint32_t g_routed_moe_selected_override_n;
static uint64_t g_stream_selected_stage_counter;
static cudaEvent_t g_stream_selected_reuse_event;
static int g_stream_selected_reuse_event_pending;
static void *g_stream_read_stage_raw[DS4_ROCM_STREAM_READ_WORKERS];
static uint64_t g_stream_read_stage_bytes[DS4_ROCM_STREAM_READ_WORKERS];
static cudaStream_t g_stream_read_upload_streams[DS4_ROCM_STREAM_READ_WORKERS];
static pthread_t g_stream_read_threads[DS4_ROCM_STREAM_READ_WORKERS];
static uint32_t g_stream_read_thread_ids[DS4_ROCM_STREAM_READ_WORKERS];
static pthread_mutex_t g_stream_read_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_stream_read_work_cond = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_stream_read_done_cond = PTHREAD_COND_INITIALIZER;
static int g_stream_read_pool_started;
static int g_stream_read_pool_stop;
static struct cuda_stream_read_job *g_stream_read_active_jobs;
static uint32_t g_stream_read_active_count;
static uint32_t g_stream_read_active_next;
static uint32_t g_stream_read_active_done;
static int g_stream_read_active_ok;

static int cuda_ok(cudaError_t err, const char *what);
static uint64_t cuda_model_copy_chunk_bytes(void);
static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes);
static int cuda_model_stage_pool_alloc(uint64_t bytes);
static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset);
static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload);
static int cuda_stream_selected_reuse_wait(const char *what);
static void cuda_stream_read_pool_shutdown(void);

static int cuda_u64_mul_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return 0;
    if (a != 0u && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int cuda_u64_mul3_checked(uint64_t a, uint64_t b, uint64_t c, uint64_t *out) {
    uint64_t tmp = 0;
    return cuda_u64_mul_checked(a, b, &tmp) && cuda_u64_mul_checked(tmp, c, out);
}

static int cuda_stream_cache_stats_on(void) {
    if (g_stream_cache_stats_enabled < 0) {
        g_stream_cache_stats_enabled =
            getenv("DS4_ROCM_STREAM_CACHE_STATS") != NULL ? 1 : 0;
    }
    return g_stream_cache_stats_enabled;
}

static void cuda_stream_cache_stats_note_resident(void) {
    if (!cuda_stream_cache_stats_on()) return;
    const uint64_t count = (uint64_t)g_stream_resident_experts.size();
    if (count > g_stream_cache_stats.max_resident_count) {
        g_stream_cache_stats.max_resident_count = count;
    }
    if (g_stream_resident_bytes > g_stream_cache_stats.max_resident_bytes) {
        g_stream_cache_stats.max_resident_bytes = g_stream_resident_bytes;
    }
}

static void cuda_stream_cache_stats_print(const char *label) {
    if (!cuda_stream_cache_stats_on()) return;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "stream cache stats %s: "
            "selected calls=%llu slots=%llu hits=%llu misses=%llu; "
            "batch calls=%llu unique=%llu hits=%llu misses=%llu; "
            "seed calls=%llu unique=%llu; "
            "full-layer loads=%llu bytes=%.2f GiB resident-flushes=%llu; "
            "allocs=%llu alloc=%.2f GiB evictions=%llu evicted=%.2f GiB; "
            "resident current=%zu/%.2f GiB max=%llu/%.2f GiB budget=%u\n",
            label ? label : "",
            (unsigned long long)g_stream_cache_stats.selected_calls,
            (unsigned long long)g_stream_cache_stats.selected_slots,
            (unsigned long long)g_stream_cache_stats.selected_hits,
            (unsigned long long)g_stream_cache_stats.selected_misses,
            (unsigned long long)g_stream_cache_stats.batch_calls,
            (unsigned long long)g_stream_cache_stats.batch_unique,
            (unsigned long long)g_stream_cache_stats.batch_hits,
            (unsigned long long)g_stream_cache_stats.batch_misses,
            (unsigned long long)g_stream_cache_stats.seed_calls,
            (unsigned long long)g_stream_cache_stats.seed_unique,
            (unsigned long long)g_stream_cache_stats.layer_loads,
            (double)g_stream_cache_stats.layer_load_bytes / 1073741824.0,
            (unsigned long long)g_stream_cache_stats.layer_resident_flushes,
            (unsigned long long)g_stream_cache_stats.allocs,
            (double)g_stream_cache_stats.alloc_bytes / 1073741824.0,
            (unsigned long long)g_stream_cache_stats.evictions,
            (double)g_stream_cache_stats.evict_bytes / 1073741824.0,
            g_stream_resident_experts.size(),
            (double)g_stream_resident_bytes / 1073741824.0,
            (unsigned long long)g_stream_cache_stats.max_resident_count,
            (double)g_stream_cache_stats.max_resident_bytes / 1073741824.0,
            g_stream_expert_cache_budget);
}

static int cuda_model_range_fits(uint64_t model_size, uint64_t offset, uint64_t bytes) {
    return offset <= model_size && bytes <= model_size - offset;
}

static int cuda_tensor_has_bytes(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && t->bytes >= bytes;
}

static int cuda_tensor_has_elems(const ds4_gpu_tensor *t, uint64_t elems, uint64_t elem_size) {
    uint64_t bytes = 0;
    return cuda_u64_mul_checked(elems, elem_size, &bytes) && cuda_tensor_has_bytes(t, bytes);
}

static int cuda_tensor_has_elems2(const ds4_gpu_tensor *t, uint64_t a, uint64_t b, uint64_t elem_size) {
    uint64_t bytes = 0;
    return cuda_u64_mul3_checked(a, b, elem_size, &bytes) && cuda_tensor_has_bytes(t, bytes);
}

static int cuda_tensor_has_elems3(const ds4_gpu_tensor *t, uint64_t a, uint64_t b, uint64_t c, uint64_t elem_size) {
    uint64_t ab = 0, elems = 0, bytes = 0;
    return cuda_u64_mul_checked(a, b, &ab) &&
           cuda_u64_mul_checked(ab, c, &elems) &&
           cuda_u64_mul_checked(elems, elem_size, &bytes) &&
           cuda_tensor_has_bytes(t, bytes);
}

static int cuda_tensor_has_f32(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(float));
}

static int cuda_tensor_has_i32(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(int32_t));
}

static int cuda_tensor_has_f16(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(__half));
}

static int cuda_tensor_has_u16(const ds4_gpu_tensor *t, uint64_t elems) {
    return cuda_tensor_has_elems(t, elems, sizeof(uint16_t));
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f16_transpose_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

static void cuda_shared_gate_up_async_cleanup(void);

static void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_cuda_tmp_bytes >= bytes) return g_cuda_tmp;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_cuda_tmp = ptr;
    g_cuda_tmp_bytes = bytes;
    return g_cuda_tmp;
}

static int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_ROCM_ATTENTION_SCORE_CAP - DS4_ROCM_ATTENTION_RAW_SCORE_CAP;
}

static int cuda_model_image_find(const void *model_map) {
    if (!model_map) return -1;
    for (size_t i = 0; i < g_model_images.size(); i++) {
        if (g_model_images[i].host_base == model_map) return (int)i;
    }
    return -1;
}

static const char *cuda_model_image_ptr(const void *model_map, uint64_t offset) {
    const int idx = cuda_model_image_find(model_map);
    if (idx < 0) return NULL;
    const cuda_model_image &img = g_model_images[(size_t)idx];
    if (offset > img.size) return NULL;
    return img.device_ptr + offset;
}

static int cuda_model_image_owned(const void *model_map) {
    return cuda_model_image_find(model_map) >= 0;
}

static uint64_t cuda_model_image_bytes(void) {
    uint64_t bytes = 0;
    for (const cuda_model_image &img : g_model_images) bytes += img.size;
    return bytes;
}

static void cuda_model_image_release_all(void) {
    for (const cuda_model_image &img : g_model_images) {
        if (img.device_ptr) (void)cudaFree(img.device_ptr);
    }
    g_model_images.clear();
}

static void cuda_stream_resident_cache_release(void) {
    for (cuda_stream_resident_expert &e : g_stream_resident_experts) {
        if (e.base) (void)cudaFree(e.base);
    }
    g_stream_resident_experts.clear();
    g_stream_resident_index.clear();
    g_stream_resident_bytes = 0;
    g_stream_resident_clock = 0;
}

static void cuda_stream_layer_expert_cache_release(void) {
    for (uint32_t i = 0; i < 2u; i++) {
        cuda_stream_layer_expert_cache &c = g_stream_layer_expert_cache[i];
        if (c.base) (void)cudaFree(c.base);
        memset(&c, 0, sizeof(c));
    }
}

static void cuda_stream_read_stage_release(void) {
    cuda_stream_read_pool_shutdown();
    for (uint32_t i = 0; i < DS4_ROCM_STREAM_READ_WORKERS; i++) {
        if (g_stream_read_stage_raw[i]) {
            (void)cudaFreeHost(g_stream_read_stage_raw[i]);
            g_stream_read_stage_raw[i] = NULL;
            g_stream_read_stage_bytes[i] = 0;
        }
        if (g_stream_read_upload_streams[i]) {
            (void)cudaStreamDestroy(g_stream_read_upload_streams[i]);
            g_stream_read_upload_streams[i] = NULL;
        }
    }
}

static void cuda_stream_batch_selected_cache_release(void) {
    if (g_stream_batch_selected_cache.selected_ids) {
        (void)cudaFree(g_stream_batch_selected_cache.selected_ids);
    }
    if (g_stream_batch_selected_cache.pair_missing) {
        (void)cudaFree(g_stream_batch_selected_cache.pair_missing);
    }
    if (g_stream_batch_selected_cache.gate_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.gate_ptrs);
    }
    if (g_stream_batch_selected_cache.up_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.up_ptrs);
    }
    if (g_stream_batch_selected_cache.down_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.down_ptrs);
    }
    if (g_stream_batch_selected_cache.resident_gate_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.resident_gate_ptrs);
    }
    if (g_stream_batch_selected_cache.resident_up_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.resident_up_ptrs);
    }
    if (g_stream_batch_selected_cache.missing_gate_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.missing_gate_ptrs);
    }
    if (g_stream_batch_selected_cache.missing_up_ptrs) {
        (void)cudaFree(g_stream_batch_selected_cache.missing_up_ptrs);
    }
    memset(&g_stream_batch_selected_cache, 0, sizeof(g_stream_batch_selected_cache));
}

static void cuda_stream_selected_cache_release(void) {
    (void)cuda_stream_selected_reuse_wait("streaming selected cache release");
    if (g_stream_selected_cache.gate) (void)cudaFree(g_stream_selected_cache.gate);
    if (g_stream_selected_cache.up) (void)cudaFree(g_stream_selected_cache.up);
    if (g_stream_selected_cache.down) (void)cudaFree(g_stream_selected_cache.down);
    if (g_stream_selected_cache.slot_ids) (void)cudaFree(g_stream_selected_cache.slot_ids);
    if (g_stream_selected_cache.gate_ptrs) (void)cudaFree(g_stream_selected_cache.gate_ptrs);
    if (g_stream_selected_cache.up_ptrs) (void)cudaFree(g_stream_selected_cache.up_ptrs);
    if (g_stream_selected_cache.down_ptrs) (void)cudaFree(g_stream_selected_cache.down_ptrs);
    if (g_stream_selected_reuse_event) {
        (void)cudaEventDestroy(g_stream_selected_reuse_event);
        g_stream_selected_reuse_event = NULL;
    }
    g_stream_selected_reuse_event_pending = 0;
    memset(&g_stream_selected_cache, 0, sizeof(g_stream_selected_cache));
    cuda_stream_batch_selected_cache_release();
    cuda_stream_resident_cache_release();
    cuda_stream_layer_expert_cache_release();
    cuda_stream_read_stage_release();
    g_routed_moe_selected_override_n = 0;
}

static int cuda_stream_selected_ensure_stream(void) {
    if (g_model_upload_stream) return 1;
    cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected upload stream creation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_selected_reuse_ensure_event(void) {
    if (g_stream_selected_reuse_event) return 1;
    cudaError_t err =
        cudaEventCreateWithFlags(&g_stream_selected_reuse_event,
                                 cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected reuse event creation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_selected_reuse_wait(const char *what) {
    if (!g_stream_selected_reuse_event_pending) return 1;
    cudaError_t err = cudaEventSynchronize(g_stream_selected_reuse_event);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "%s wait failed: %s\n",
                what ? what : "streaming selected cache reuse",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_selected_reuse_event_pending = 0;
    return 1;
}

static int cuda_stream_selected_mark_inflight(void) {
    if (!g_ssd_streaming_mode) return 1;
    if (!cuda_stream_selected_reuse_ensure_event()) return 0;
    cudaError_t err = cudaEventRecord(g_stream_selected_reuse_event, 0);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected reuse event record failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_stream_selected_reuse_event_pending = 1;
    return 1;
}

static int cuda_stream_selected_ensure_buffers(uint64_t gate_bytes, uint64_t down_bytes) {
    if (gate_bytes == 0 || down_bytes == 0) return 0;
    cudaError_t err = cudaSuccess;
    if (g_stream_selected_cache.gate_capacity < gate_bytes) {
        if (g_stream_selected_cache.gate) (void)cudaFree(g_stream_selected_cache.gate);
        if (g_stream_selected_cache.up) (void)cudaFree(g_stream_selected_cache.up);
        g_stream_selected_cache.gate = NULL;
        g_stream_selected_cache.up = NULL;
        g_stream_selected_cache.gate_capacity = 0;
        err = cudaMalloc((void **)&g_stream_selected_cache.gate, (size_t)gate_bytes);
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_selected_cache.up, (size_t)gate_bytes);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected gate/up alloc failed (%.2f MiB): %s\n",
                    (double)gate_bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_selected_cache.gate_capacity = gate_bytes;
    }
    if (g_stream_selected_cache.down_capacity < down_bytes) {
        if (g_stream_selected_cache.down) (void)cudaFree(g_stream_selected_cache.down);
        g_stream_selected_cache.down = NULL;
        g_stream_selected_cache.down_capacity = 0;
        err = cudaMalloc((void **)&g_stream_selected_cache.down, (size_t)down_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected down alloc failed (%.2f MiB): %s\n",
                    (double)down_bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_selected_cache.down_capacity = down_bytes;
    }
    if (!g_stream_selected_cache.slot_ids) {
        err = cudaMalloc((void **)&g_stream_selected_cache.slot_ids,
                         DS4_ROCM_N_EXPERT_USED * sizeof(int32_t));
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected slot-id alloc failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        int32_t slots[DS4_ROCM_N_EXPERT_USED];
        for (uint32_t i = 0; i < DS4_ROCM_N_EXPERT_USED; i++) slots[i] = (int32_t)i;
        err = cudaMemcpy(g_stream_selected_cache.slot_ids,
                         slots,
                         sizeof(slots),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected slot-id upload failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_selected_cache.slot_tensor.ptr = g_stream_selected_cache.slot_ids;
        g_stream_selected_cache.slot_tensor.bytes =
            DS4_ROCM_N_EXPERT_USED * sizeof(int32_t);
        g_stream_selected_cache.slot_tensor.owner = 0;
    }
    if (!g_stream_selected_cache.gate_ptrs) {
        err = cudaMalloc((void **)&g_stream_selected_cache.gate_ptrs,
                         DS4_ROCM_N_EXPERT_USED * sizeof(char *));
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_selected_cache.up_ptrs,
                             DS4_ROCM_N_EXPERT_USED * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_selected_cache.down_ptrs,
                             DS4_ROCM_N_EXPERT_USED * sizeof(char *));
        }
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected pointer table alloc failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    return 1;
}

static int cuda_stream_batch_selected_ensure_buffers(
        uint64_t n_ids,
        uint32_t n_unique) {
    if (n_ids == 0 || n_unique == 0) return 0;
    cudaError_t err = cudaSuccess;
    const uint64_t selected_bytes = n_ids * sizeof(int32_t);
    if (g_stream_batch_selected_cache.selected_capacity < selected_bytes) {
        if (g_stream_batch_selected_cache.selected_ids) {
            (void)cudaFree(g_stream_batch_selected_cache.selected_ids);
            g_stream_batch_selected_cache.selected_ids = NULL;
            g_stream_batch_selected_cache.selected_capacity = 0;
        }
        err = cudaMalloc((void **)&g_stream_batch_selected_cache.selected_ids,
                         (size_t)selected_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch selected-id alloc failed "
                    "(%.2f MiB): %s\n",
                    (double)selected_bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_batch_selected_cache.selected_capacity = selected_bytes;
    }
    if (g_stream_batch_selected_cache.pair_missing_capacity < n_ids) {
        if (g_stream_batch_selected_cache.pair_missing) {
            (void)cudaFree(g_stream_batch_selected_cache.pair_missing);
            g_stream_batch_selected_cache.pair_missing = NULL;
            g_stream_batch_selected_cache.pair_missing_capacity = 0;
        }
        err = cudaMalloc((void **)&g_stream_batch_selected_cache.pair_missing,
                         (size_t)n_ids);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch selected split-flag alloc failed "
                    "(%.2f MiB): %s\n",
                    (double)n_ids / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_batch_selected_cache.pair_missing_capacity = n_ids;
    }
    if (g_stream_batch_selected_cache.ptr_capacity < n_unique) {
        if (g_stream_batch_selected_cache.gate_ptrs) {
            (void)cudaFree(g_stream_batch_selected_cache.gate_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.up_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.down_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.resident_gate_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.resident_up_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.missing_gate_ptrs);
            (void)cudaFree(g_stream_batch_selected_cache.missing_up_ptrs);
            g_stream_batch_selected_cache.gate_ptrs = NULL;
            g_stream_batch_selected_cache.up_ptrs = NULL;
            g_stream_batch_selected_cache.down_ptrs = NULL;
            g_stream_batch_selected_cache.resident_gate_ptrs = NULL;
            g_stream_batch_selected_cache.resident_up_ptrs = NULL;
            g_stream_batch_selected_cache.missing_gate_ptrs = NULL;
            g_stream_batch_selected_cache.missing_up_ptrs = NULL;
            g_stream_batch_selected_cache.ptr_capacity = 0;
        }
        err = cudaMalloc((void **)&g_stream_batch_selected_cache.gate_ptrs,
                         (size_t)n_unique * sizeof(char *));
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.up_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.down_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.resident_gate_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.resident_up_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.missing_gate_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err == cudaSuccess) {
            err = cudaMalloc((void **)&g_stream_batch_selected_cache.missing_up_ptrs,
                             (size_t)n_unique * sizeof(char *));
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch pointer-table alloc failed "
                    "(unique=%u): %s\n",
                    n_unique,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_batch_selected_cache.ptr_capacity = n_unique;
    }
    g_stream_batch_selected_cache.selected_tensor.ptr =
        g_stream_batch_selected_cache.selected_ids;
    g_stream_batch_selected_cache.selected_tensor.bytes = selected_bytes;
    g_stream_batch_selected_cache.selected_tensor.owner = 0;
    return 1;
}

static int cuda_stream_batch_selected_cache_apply(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint32_t n_tokens,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char ***gate_ptrs,
        const char ***up_ptrs,
        const char ***down_ptrs,
        uint32_t *unique_out) {
    if (!selected_exec || !gate_ptrs || !up_ptrs || !down_ptrs || !unique_out) {
        return 0;
    }
    if (!g_stream_batch_selected_cache.loaded ||
        g_stream_batch_selected_cache.model_map != model_map ||
        g_stream_batch_selected_cache.layer != layer ||
        g_stream_batch_selected_cache.n_total_expert != n_total_expert ||
        g_stream_batch_selected_cache.n_selected != n_selected ||
        g_stream_batch_selected_cache.n_tokens != n_tokens ||
        g_stream_batch_selected_cache.gate_offset != gate_offset ||
        g_stream_batch_selected_cache.up_offset != up_offset ||
        g_stream_batch_selected_cache.down_offset != down_offset ||
        g_stream_batch_selected_cache.gate_expert_bytes != gate_expert_bytes ||
        g_stream_batch_selected_cache.down_expert_bytes != down_expert_bytes ||
        !g_stream_batch_selected_cache.selected_ids ||
        !g_stream_batch_selected_cache.gate_ptrs ||
        !g_stream_batch_selected_cache.up_ptrs ||
        !g_stream_batch_selected_cache.down_ptrs ||
        g_stream_batch_selected_cache.n_unique == 0) {
        return 0;
    }
    *selected_exec = &g_stream_batch_selected_cache.selected_tensor;
    *gate_ptrs = g_stream_batch_selected_cache.gate_ptrs;
    *up_ptrs = g_stream_batch_selected_cache.up_ptrs;
    *down_ptrs = g_stream_batch_selected_cache.down_ptrs;
    *unique_out = g_stream_batch_selected_cache.n_unique;
    return 1;
}

static int cuda_stream_selected_is_current(
        const cuda_stream_resident_expert &e,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_selected) {
    if (!selected_ids || e.layer != layer) return 0;
    for (uint32_t i = 0; i < n_selected; i++) {
        if (e.expert == selected_ids[i]) return 1;
    }
    return 0;
}

static cuda_stream_resident_key cuda_stream_resident_make_key(
        const void *model_map,
        uint32_t layer,
        int32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    cuda_stream_resident_key k;
    k.model_map = model_map;
    k.layer = layer;
    k.expert = expert;
    k.gate_offset = gate_offset;
    k.up_offset = up_offset;
    k.down_offset = down_offset;
    k.gate_expert_bytes = gate_expert_bytes;
    k.down_expert_bytes = down_expert_bytes;
    return k;
}

static cuda_stream_resident_key cuda_stream_resident_entry_key(
        const cuda_stream_resident_expert &e) {
    return cuda_stream_resident_make_key(e.model_map,
                                         e.layer,
                                         e.expert,
                                         e.gate_offset,
                                         e.up_offset,
                                         e.down_offset,
                                         e.gate_expert_bytes,
                                         e.down_expert_bytes);
}

static int cuda_stream_resident_find(
        const void *model_map,
        uint32_t layer,
        int32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    const cuda_stream_resident_key key =
        cuda_stream_resident_make_key(model_map,
                                      layer,
                                      expert,
                                      gate_offset,
                                      up_offset,
                                      down_offset,
                                      gate_expert_bytes,
                                      down_expert_bytes);
    const auto it = g_stream_resident_index.find(key);
    if (it != g_stream_resident_index.end() &&
        it->second < g_stream_resident_experts.size()) {
        return (int)it->second;
    }
    return -1;
}

static void cuda_stream_resident_evict_at(size_t idx) {
    if (idx >= g_stream_resident_experts.size()) return;
    cuda_stream_resident_expert &e = g_stream_resident_experts[idx];
    const cuda_stream_resident_key evicted_key =
        cuda_stream_resident_entry_key(e);
    if (cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.evictions++;
        g_stream_cache_stats.evict_bytes += e.bytes;
    }
    if (e.base) (void)cudaFree(e.base);
    if (g_stream_resident_bytes >= e.bytes) {
        g_stream_resident_bytes -= e.bytes;
    } else {
        g_stream_resident_bytes = 0;
    }
    g_stream_resident_index.erase(evicted_key);
    const size_t last = g_stream_resident_experts.size() - 1u;
    if (idx != last) {
        g_stream_resident_experts[idx] = g_stream_resident_experts[last];
        g_stream_resident_index[cuda_stream_resident_entry_key(
                g_stream_resident_experts[idx])] = idx;
    }
    g_stream_resident_experts.pop_back();
}

static int cuda_stream_resident_evict_one(
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_selected) {
    size_t victim = (size_t)-1;
    uint64_t oldest = UINT64_MAX;
    for (size_t i = 0; i < g_stream_resident_experts.size(); i++) {
        const cuda_stream_resident_expert &e = g_stream_resident_experts[i];
        if (cuda_stream_selected_is_current(e, layer, selected_ids, n_selected)) {
            continue;
        }
        if (e.last_used < oldest) {
            oldest = e.last_used;
            victim = i;
        }
    }
    if (victim == (size_t)-1) return 0;
    cuda_stream_resident_evict_at(victim);
    return 1;
}

static uint64_t cuda_stream_resident_free_reserve_bytes(void) {
    return 16ull * 1024ull * 1024ull * 1024ull;
}

static int cuda_stream_resident_make_room(
        uint64_t bytes,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_selected) {
    while (g_stream_resident_experts.size() >= g_stream_expert_cache_budget) {
        if (!cuda_stream_resident_evict_one(layer, selected_ids, n_selected)) {
            break;
        }
    }

    size_t free_b = 0;
    size_t total_b = 0;
    const uint64_t reserve = cuda_stream_resident_free_reserve_bytes();
    while (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        (void)total_b;
        if ((uint64_t)free_b >= reserve &&
            bytes <= (uint64_t)free_b - reserve) {
            return 1;
        }
        if (!cuda_stream_resident_evict_one(layer, selected_ids, n_selected)) {
            return 0;
        }
    }
    (void)cudaGetLastError();
    return 1;
}

static int cuda_stream_resident_alloc(
        const void *model_map,
        uint32_t layer,
        int32_t expert,
        const int32_t *selected_ids,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (g_stream_expert_cache_budget == 0) return -1;
    uint64_t bytes = 0;
    uint64_t gate_pair = 0;
    if (!cuda_u64_mul_checked(2u, gate_expert_bytes, &gate_pair) ||
        gate_pair > UINT64_MAX - down_expert_bytes) {
        return -1;
    }
    bytes = gate_pair + down_expert_bytes;

    if (!cuda_stream_resident_make_room(bytes, layer, selected_ids, n_selected)) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming expert cache cannot keep %.2f MiB "
                "for layer=%u expert=%d while preserving %.2f GiB free\n",
                (double)bytes / 1048576.0,
                layer,
                expert,
                (double)cuda_stream_resident_free_reserve_bytes() / 1073741824.0);
        return -1;
    }

    void *base = NULL;
    cudaError_t err = cudaMalloc(&base, (size_t)bytes);
    while (err != cudaSuccess && cuda_stream_resident_evict_one(layer, selected_ids, n_selected)) {
        (void)cudaGetLastError();
        err = cudaMalloc(&base, (size_t)bytes);
    }
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming expert cache allocation failed "
                "for layer=%u expert=%d (%.2f MiB): %s\n",
                layer,
                expert,
                (double)bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return -1;
    }

    cuda_stream_resident_expert e;
    memset(&e, 0, sizeof(e));
    e.model_map = model_map;
    e.layer = layer;
    e.expert = expert;
    e.gate_expert_bytes = gate_expert_bytes;
    e.down_expert_bytes = down_expert_bytes;
    e.gate_offset = gate_offset;
    e.up_offset = up_offset;
    e.down_offset = down_offset;
    e.base = (char *)base;
    e.gate = e.base;
    e.up = e.base + gate_expert_bytes;
    e.down = e.base + 2u * gate_expert_bytes;
    e.bytes = bytes;
    e.last_used = ++g_stream_resident_clock;
    g_stream_resident_experts.push_back(e);
    g_stream_resident_index[cuda_stream_resident_entry_key(e)] =
        g_stream_resident_experts.size() - 1u;
    g_stream_resident_bytes += bytes;
    if (cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.allocs++;
        g_stream_cache_stats.alloc_bytes += bytes;
    }
    cuda_stream_cache_stats_note_resident();
    return (int)g_stream_resident_experts.size() - 1;
}

typedef struct cuda_stream_read_job {
    char *dst;
    uint64_t offset;
    uint64_t bytes;
    void *host_raw;
    void *host_buf;
    int ok;
    int uploaded;
    int errnum;
} cuda_stream_read_job;

struct cuda_stream_batch_selected_pending {
    int active;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint32_t n_tokens;
    uint32_t n_unique;
    uint32_t resident_count;
    uint32_t missing_count;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    cuda_stream_read_job read_jobs[DS4_ROCM_STREAM_READ_MAX_JOBS];
    uint32_t read_job_count;
};

static cuda_stream_batch_selected_pending g_stream_batch_selected_pending;

struct cuda_stream_selected_pending {
    int active;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t resident_mask;
    uint32_t missing_mask;
    int32_t selected_ids[DS4_ROCM_N_EXPERT_USED];
    cuda_stream_read_job read_jobs[DS4_ROCM_N_EXPERT_USED * 3u];
    uint32_t read_job_count;
};

static cuda_stream_selected_pending g_stream_selected_pending;

static void cuda_stream_read_job_run(cuda_stream_read_job *job) {
    job->ok = 0;
    job->uploaded = 0;
    job->errnum = 0;
    if (!job || !job->host_buf || job->bytes == 0 || g_model_fd < 0) {
        if (job) job->errnum = EINVAL;
        return;
    }
    if (cuda_pread_full(g_model_fd, job->host_buf, job->bytes, job->offset)) {
        job->ok = 1;
    } else {
        job->errnum = errno ? errno : EIO;
    }
}

static int cuda_stream_read_job_upload(
        cuda_stream_read_job *job,
        cudaStream_t stream) {
    if (!job || !job->ok || !job->dst || !job->host_buf || !stream) {
        if (job) job->errnum = EINVAL;
        return 0;
    }
    cudaError_t err = cudaMemcpyAsync(job->dst,
                                      job->host_buf,
                                      (size_t)job->bytes,
                                      cudaMemcpyHostToDevice,
                                      stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming read-worker upload failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        job->ok = 0;
        job->errnum = EIO;
        return 0;
    }
    job->uploaded = 1;
    cuda_model_drop_file_pages(job->offset, job->bytes);
    return 1;
}

static void *cuda_stream_read_worker(void *arg) {
    const uint32_t worker_id = arg ? *(const uint32_t *)arg : 0u;
    (void)cudaSetDevice(0);
    for (;;) {
        pthread_mutex_lock(&g_stream_read_mutex);
        while (!g_stream_read_pool_stop &&
               (!g_stream_read_active_jobs ||
                g_stream_read_active_next >= g_stream_read_active_count)) {
            pthread_cond_wait(&g_stream_read_work_cond, &g_stream_read_mutex);
        }
        if (g_stream_read_pool_stop) {
            pthread_mutex_unlock(&g_stream_read_mutex);
            break;
        }
        const uint32_t idx = g_stream_read_active_next++;
        cuda_stream_read_job *job = &g_stream_read_active_jobs[idx];
        if (worker_id < DS4_ROCM_STREAM_READ_WORKERS) {
            job->host_raw = g_stream_read_stage_raw[worker_id];
            job->host_buf = g_stream_read_stage_raw[worker_id];
        }
        pthread_mutex_unlock(&g_stream_read_mutex);

        cuda_stream_read_job_run(job);
        if (job->ok) {
            (void)cuda_stream_read_job_upload(
                    job,
                    worker_id < DS4_ROCM_STREAM_READ_WORKERS ?
                        g_stream_read_upload_streams[worker_id] : NULL);
        }

        pthread_mutex_lock(&g_stream_read_mutex);
        if (!job->ok) g_stream_read_active_ok = 0;
        g_stream_read_active_done++;
        if (g_stream_read_active_done >= g_stream_read_active_count) {
            pthread_cond_signal(&g_stream_read_done_cond);
        }
        pthread_mutex_unlock(&g_stream_read_mutex);
    }
    return NULL;
}

static void cuda_stream_read_upload_streams_destroy(void) {
    for (uint32_t i = 0; i < DS4_ROCM_STREAM_READ_WORKERS; i++) {
        if (g_stream_read_upload_streams[i]) {
            (void)cudaStreamDestroy(g_stream_read_upload_streams[i]);
            g_stream_read_upload_streams[i] = NULL;
        }
    }
}

static int cuda_stream_read_upload_streams_ensure(void) {
    for (uint32_t i = 0; i < DS4_ROCM_STREAM_READ_WORKERS; i++) {
        if (g_stream_read_upload_streams[i]) continue;
        cudaError_t err = cudaStreamCreateWithFlags(
                &g_stream_read_upload_streams[i],
                cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming read upload stream creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_stream_read_upload_streams_destroy();
            return 0;
        }
    }
    return 1;
}

static int cuda_stream_read_pool_ensure(void) {
    if (g_stream_read_pool_started) return 1;
    pthread_mutex_lock(&g_stream_read_mutex);
    if (g_stream_read_pool_started) {
        pthread_mutex_unlock(&g_stream_read_mutex);
        return 1;
    }
    g_stream_read_pool_stop = 0;
    g_stream_read_active_jobs = NULL;
    g_stream_read_active_count = 0;
    g_stream_read_active_next = 0;
    g_stream_read_active_done = 0;
    g_stream_read_active_ok = 1;
    if (!cuda_stream_read_upload_streams_ensure()) {
        pthread_mutex_unlock(&g_stream_read_mutex);
        return 0;
    }
    for (uint32_t i = 0; i < DS4_ROCM_STREAM_READ_WORKERS; i++) {
        g_stream_read_thread_ids[i] = i;
        const int rc = pthread_create(&g_stream_read_threads[i],
                                      NULL,
                                      cuda_stream_read_worker,
                                      &g_stream_read_thread_ids[i]);
        if (rc != 0) {
            g_stream_read_pool_stop = 1;
            pthread_cond_broadcast(&g_stream_read_work_cond);
            pthread_mutex_unlock(&g_stream_read_mutex);
            for (uint32_t j = 0; j < i; j++) {
                (void)pthread_join(g_stream_read_threads[j], NULL);
            }
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming read worker creation failed: %s\n",
                    strerror(rc));
            cuda_stream_read_upload_streams_destroy();
            return 0;
        }
    }
    g_stream_read_pool_started = 1;
    pthread_mutex_unlock(&g_stream_read_mutex);
    return 1;
}

static void cuda_stream_read_pool_shutdown(void) {
    if (!g_stream_read_pool_started) return;
    pthread_mutex_lock(&g_stream_read_mutex);
    g_stream_read_pool_stop = 1;
    pthread_cond_broadcast(&g_stream_read_work_cond);
    pthread_mutex_unlock(&g_stream_read_mutex);
    for (uint32_t i = 0; i < DS4_ROCM_STREAM_READ_WORKERS; i++) {
        (void)pthread_join(g_stream_read_threads[i], NULL);
    }
    pthread_mutex_lock(&g_stream_read_mutex);
    g_stream_read_pool_started = 0;
    g_stream_read_pool_stop = 0;
    g_stream_read_active_jobs = NULL;
    g_stream_read_active_count = 0;
    g_stream_read_active_next = 0;
    g_stream_read_active_done = 0;
    g_stream_read_active_ok = 1;
    pthread_mutex_unlock(&g_stream_read_mutex);
}

static int cuda_stream_read_jobs_prepare(cuda_stream_read_job *jobs, uint32_t count) {
    if (!jobs || count == 0) return 1;
    if (count > DS4_ROCM_STREAM_READ_MAX_JOBS) return 0;

    uint64_t max_bytes = 0;
    for (uint32_t i = 0; i < count; i++) {
        jobs[i].ok = 0;
        jobs[i].uploaded = 0;
        jobs[i].errnum = 0;
        jobs[i].host_raw = NULL;
        jobs[i].host_buf = NULL;
        if (jobs[i].bytes > max_bytes) max_bytes = jobs[i].bytes;
    }

    for (uint32_t i = 0; i < DS4_ROCM_STREAM_READ_WORKERS; i++) {
        if (g_stream_read_stage_bytes[i] < max_bytes) {
            if (g_stream_read_stage_raw[i]) {
                (void)cudaFreeHost(g_stream_read_stage_raw[i]);
                g_stream_read_stage_raw[i] = NULL;
                g_stream_read_stage_bytes[i] = 0;
            }
            cudaError_t err = cudaMallocHost(&g_stream_read_stage_raw[i],
                                             (size_t)max_bytes);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "streaming read pinned allocation failed "
                        "(%.2f MiB): %s\n",
                        (double)max_bytes / 1048576.0,
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
                return 0;
            }
            g_stream_read_stage_bytes[i] = max_bytes;
        }
    }
    return 1;
}

static int cuda_stream_read_jobs_start(cuda_stream_read_job *jobs, uint32_t count) {
    if (!jobs || count == 0) return 1;
    if (!cuda_stream_read_jobs_prepare(jobs, count)) return 0;
    if (!cuda_stream_read_pool_ensure()) return 0;
    pthread_mutex_lock(&g_stream_read_mutex);
    if (g_stream_read_active_jobs != NULL) {
        pthread_mutex_unlock(&g_stream_read_mutex);
        fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming read pool already has active work\n");
        return 0;
    }
    g_stream_read_active_jobs = jobs;
    g_stream_read_active_count = count;
    g_stream_read_active_next = 0;
    g_stream_read_active_done = 0;
    g_stream_read_active_ok = 1;
    pthread_cond_broadcast(&g_stream_read_work_cond);
    pthread_mutex_unlock(&g_stream_read_mutex);
    return 1;
}

static int cuda_stream_read_jobs_wait(cuda_stream_read_job *jobs, uint32_t count) {
    if (!jobs || count == 0) return 1;
    pthread_mutex_lock(&g_stream_read_mutex);
    if (g_stream_read_active_jobs != jobs) {
        pthread_mutex_unlock(&g_stream_read_mutex);
        fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming read wait received inactive job set\n");
        return 0;
    }
    while (g_stream_read_active_done < g_stream_read_active_count) {
        pthread_cond_wait(&g_stream_read_done_cond, &g_stream_read_mutex);
    }
    const int pool_ok = g_stream_read_active_ok;
    g_stream_read_active_jobs = NULL;
    g_stream_read_active_count = 0;
    g_stream_read_active_next = 0;
    g_stream_read_active_done = 0;
    g_stream_read_active_ok = 1;
    pthread_mutex_unlock(&g_stream_read_mutex);

    int ok = 1;
    if (!pool_ok) ok = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!jobs[i].ok) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming read failed at offset %.2f GiB "
                    "size %.2f MiB: %s\n",
                    (double)jobs[i].offset / 1073741824.0,
                    (double)jobs[i].bytes / 1048576.0,
                    strerror(jobs[i].errnum ? jobs[i].errnum : EIO));
            ok = 0;
        }
    }
    return ok;
}

static int cuda_stream_read_jobs_parallel(cuda_stream_read_job *jobs, uint32_t count) {
    if (!jobs || count == 0) return 1;
    return cuda_stream_read_jobs_start(jobs, count) &&
           cuda_stream_read_jobs_wait(jobs, count);
}

static void cuda_stream_read_jobs_free(cuda_stream_read_job *jobs, uint32_t count) {
    if (!jobs) return;
    for (uint32_t i = 0; i < count; i++) {
        jobs[i].host_raw = NULL;
        jobs[i].host_buf = NULL;
        jobs[i].uploaded = 0;
    }
}

static int cuda_stream_selected_upload_read_jobs(
        cuda_stream_read_job *jobs,
        uint32_t count);

static int cuda_stream_batch_selected_pending_matches(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint32_t n_tokens,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    return g_stream_batch_selected_pending.active &&
           g_stream_batch_selected_pending.model_map == model_map &&
           g_stream_batch_selected_pending.layer == layer &&
           g_stream_batch_selected_pending.n_total_expert == n_total_expert &&
           g_stream_batch_selected_pending.n_selected == n_selected &&
           g_stream_batch_selected_pending.n_tokens == n_tokens &&
           g_stream_batch_selected_pending.gate_offset == gate_offset &&
           g_stream_batch_selected_pending.up_offset == up_offset &&
           g_stream_batch_selected_pending.down_offset == down_offset &&
           g_stream_batch_selected_pending.gate_expert_bytes == gate_expert_bytes &&
           g_stream_batch_selected_pending.down_expert_bytes == down_expert_bytes;
}

static int cuda_stream_batch_selected_apply_split(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint32_t n_tokens,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char ***resident_gate_ptrs,
        const char ***resident_up_ptrs,
        const char ***missing_gate_ptrs,
        const char ***missing_up_ptrs,
        const char ***down_ptrs,
        const uint8_t **pair_missing,
        uint32_t *resident_count,
        uint32_t *missing_count,
        uint32_t *unique_out) {
    if (!selected_exec || !resident_gate_ptrs || !resident_up_ptrs ||
        !missing_gate_ptrs || !missing_up_ptrs || !down_ptrs ||
        !pair_missing ||
        !resident_count || !missing_count || !unique_out ||
        !cuda_stream_batch_selected_pending_matches(model_map,
                                                    layer,
                                                    n_total_expert,
                                                    n_selected,
                                                    n_tokens,
                                                    gate_offset,
                                                    up_offset,
                                                    down_offset,
                                                    gate_expert_bytes,
                                                    down_expert_bytes) ||
        !g_stream_batch_selected_cache.selected_ids ||
        !g_stream_batch_selected_cache.resident_gate_ptrs ||
        !g_stream_batch_selected_cache.resident_up_ptrs ||
        !g_stream_batch_selected_cache.missing_gate_ptrs ||
        !g_stream_batch_selected_cache.missing_up_ptrs ||
        !g_stream_batch_selected_cache.down_ptrs ||
        !g_stream_batch_selected_cache.pair_missing ||
        g_stream_batch_selected_pending.missing_count == 0) {
        return 0;
    }
    *selected_exec = &g_stream_batch_selected_cache.selected_tensor;
    *resident_gate_ptrs = g_stream_batch_selected_cache.resident_gate_ptrs;
    *resident_up_ptrs = g_stream_batch_selected_cache.resident_up_ptrs;
    *missing_gate_ptrs = g_stream_batch_selected_cache.missing_gate_ptrs;
    *missing_up_ptrs = g_stream_batch_selected_cache.missing_up_ptrs;
    *down_ptrs = g_stream_batch_selected_cache.down_ptrs;
    *pair_missing = g_stream_batch_selected_cache.pair_missing;
    *resident_count = g_stream_batch_selected_pending.resident_count;
    *missing_count = g_stream_batch_selected_pending.missing_count;
    *unique_out = g_stream_batch_selected_pending.n_unique;
    return 1;
}

static int cuda_stream_batch_selected_finish_pending_missing(void) {
    if (!g_stream_batch_selected_pending.active) return 1;
    const uint32_t read_job_count =
        g_stream_batch_selected_pending.read_job_count;
    if (!cuda_stream_read_jobs_wait(g_stream_batch_selected_pending.read_jobs,
                                    read_job_count) ||
        !cuda_stream_selected_upload_read_jobs(
                g_stream_batch_selected_pending.read_jobs,
                read_job_count)) {
        cuda_stream_read_jobs_free(g_stream_batch_selected_pending.read_jobs,
                                   read_job_count);
        memset(&g_stream_batch_selected_pending, 0,
               sizeof(g_stream_batch_selected_pending));
        cuda_stream_resident_cache_release();
        return 0;
    }
    cuda_stream_read_jobs_free(g_stream_batch_selected_pending.read_jobs,
                               read_job_count);
    g_stream_batch_selected_cache.loaded = 1;
    memset(&g_stream_batch_selected_pending, 0,
           sizeof(g_stream_batch_selected_pending));
    return 1;
}

static void cuda_stream_batch_selected_abort_pending(void) {
    if (!g_stream_batch_selected_pending.active) return;
    const uint32_t read_job_count =
        g_stream_batch_selected_pending.read_job_count;
    (void)cuda_stream_read_jobs_wait(g_stream_batch_selected_pending.read_jobs,
                                     read_job_count);
    cuda_stream_read_jobs_free(g_stream_batch_selected_pending.read_jobs,
                               read_job_count);
    memset(&g_stream_batch_selected_pending, 0,
           sizeof(g_stream_batch_selected_pending));
}

static int cuda_stream_selected_upload_read_jobs(
        cuda_stream_read_job *jobs,
        uint32_t count) {
    if (!jobs || count == 0) return 1;
    int need_upload = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!jobs[i].uploaded) {
            need_upload = 1;
            break;
        }
    }
    if (!need_upload) return 1;
    if (!cuda_stream_selected_ensure_stream()) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (jobs[i].uploaded) continue;
        cudaError_t err = cudaMemcpyAsync(jobs[i].dst,
                                          jobs[i].host_buf,
                                          (size_t)jobs[i].bytes,
                                          cudaMemcpyHostToDevice,
                                          g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected cached upload failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_drop_file_pages(jobs[i].offset, jobs[i].bytes);
    }
    cudaError_t err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected upload sync failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_flush_read_jobs(
        cuda_stream_read_job *jobs,
        uint32_t *count) {
    if (!count || *count == 0) return 1;
    if (!cuda_stream_read_jobs_parallel(jobs, *count) ||
        !cuda_stream_selected_upload_read_jobs(jobs, *count)) {
        cuda_stream_read_jobs_free(jobs, *count);
        *count = 0;
        return 0;
    }
    cuda_stream_read_jobs_free(jobs, *count);
    *count = 0;
    return 1;
}

static int cuda_stream_layer_expert_cache_apply(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const char **gate_w,
        const char **up_w,
        const char **down_w) {
    if (!g_ssd_streaming_mode || !gate_w || !up_w || !down_w) return 0;
    for (uint32_t i = 0; i < 2u; i++) {
        const cuda_stream_layer_expert_cache &c = g_stream_layer_expert_cache[i];
        if (c.active &&
            c.model_map == model_map &&
            c.layer == layer &&
            c.n_total_expert == n_total_expert &&
            c.gate_offset == gate_offset &&
            c.up_offset == up_offset &&
            c.down_offset == down_offset &&
            c.gate_expert_bytes == gate_expert_bytes &&
            c.down_expert_bytes == down_expert_bytes &&
            c.gate && c.up && c.down) {
            *gate_w = c.gate;
            *up_w = c.up;
            *down_w = c.down;
            return 1;
        }
    }
    return 0;
}

static int cuda_stream_layer_expert_cache_load(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_ssd_streaming_mode ||
        !model_map ||
        model_size == 0 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_STREAM_READ_MAX_JOBS / 3u ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        g_model_fd < 0 ||
        (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base)) {
        return 0;
    }

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    uint64_t gate_pair_bytes = 0;
    uint64_t total_bytes = 0;
    if (!cuda_u64_mul_checked(n_total_expert, gate_expert_bytes, &gate_bytes) ||
        !cuda_u64_mul_checked(n_total_expert, down_expert_bytes, &down_bytes) ||
        !cuda_u64_mul_checked(2u, gate_bytes, &gate_pair_bytes) ||
        gate_pair_bytes > UINT64_MAX - down_bytes) {
        return 0;
    }
    total_bytes = gate_pair_bytes + down_bytes;
    if (cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.layer_loads++;
        g_stream_cache_stats.layer_load_bytes += total_bytes;
    }

    if (gate_offset > model_size ||
        up_offset > model_size ||
        down_offset > model_size ||
        gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }

    cuda_stream_layer_expert_cache &slot =
        g_stream_layer_expert_cache[layer & 1u];
    slot.active = 0;
    if (slot.capacity < total_bytes) {
        if (slot.base) {
            (void)cudaFree(slot.base);
            memset(&slot, 0, sizeof(slot));
        }
        if (cuda_stream_cache_stats_on() &&
            !g_stream_resident_experts.empty()) {
            g_stream_cache_stats.layer_resident_flushes++;
        }
        cuda_stream_resident_cache_release();
        void *base = NULL;
        cudaError_t err = cudaMalloc(&base, (size_t)total_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer expert cache allocation "
                    "failed for layer=%u (%.2f GiB): %s\n",
                    layer,
                    (double)total_bytes / 1073741824.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        slot.base = (char *)base;
        slot.capacity = total_bytes;
    }

    slot.bytes = total_bytes;
    slot.gate = slot.base;
    slot.up = slot.base + gate_bytes;
    slot.down = slot.base + gate_pair_bytes;

    const uint64_t read_chunk = 32ull * 1048576ull;
    const uint64_t gate_chunks =
        (gate_bytes + read_chunk - 1u) / read_chunk;
    const uint64_t down_chunks =
        (down_bytes + read_chunk - 1u) / read_chunk;
    const uint64_t read_job_count64 = gate_chunks * 2u + down_chunks;
    if (read_job_count64 == 0 ||
        read_job_count64 > DS4_ROCM_STREAM_READ_MAX_JOBS ||
        read_job_count64 > UINT32_MAX) {
        return 0;
    }
    const uint32_t read_job_count = (uint32_t)read_job_count64;
    cuda_stream_read_job *jobs =
        (cuda_stream_read_job *)calloc((size_t)read_job_count, sizeof(jobs[0]));
    if (!jobs) return 0;

    int ok = 1;
    uint32_t j = 0;
    for (uint64_t off = 0; off < gate_bytes; off += read_chunk) {
        const uint64_t n = gate_bytes - off < read_chunk ? gate_bytes - off : read_chunk;
        jobs[j++] = {slot.gate + off, gate_offset + off, n, NULL, NULL, 0, 0};
    }
    for (uint64_t off = 0; off < gate_bytes; off += read_chunk) {
        const uint64_t n = gate_bytes - off < read_chunk ? gate_bytes - off : read_chunk;
        jobs[j++] = {slot.up + off, up_offset + off, n, NULL, NULL, 0, 0};
    }
    for (uint64_t off = 0; off < down_bytes; off += read_chunk) {
        const uint64_t n = down_bytes - off < read_chunk ? down_bytes - off : read_chunk;
        jobs[j++] = {slot.down + off, down_offset + off, n, NULL, NULL, 0, 0};
    }
    if (j != read_job_count ||
        !cuda_stream_read_jobs_parallel(jobs, read_job_count)) {
        ok = 0;
    }
    cuda_stream_read_jobs_free(jobs, read_job_count);
    free(jobs);
    if (!ok) return 0;

    slot.active = 1;
    slot.model_map = model_map;
    slot.layer = layer;
    slot.n_total_expert = n_total_expert;
    slot.gate_offset = gate_offset;
    slot.up_offset = up_offset;
    slot.down_offset = down_offset;
    slot.gate_expert_bytes = gate_expert_bytes;
    slot.down_expert_bytes = down_expert_bytes;
    return 1;
}

static void cuda_stream_selected_cache_header(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        const int32_t *selected_ids,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    g_stream_selected_cache.model_map = model_map;
    g_stream_selected_cache.layer = layer;
    g_stream_selected_cache.n_total_expert = n_total_expert;
    g_stream_selected_cache.n_selected = n_selected;
    g_stream_selected_cache.gate_expert_bytes = gate_expert_bytes;
    g_stream_selected_cache.down_expert_bytes = down_expert_bytes;
    for (uint32_t i = 0; i < n_selected; i++) {
        g_stream_selected_cache.selected_ids[i] = selected_ids[i];
    }
}

static int cuda_stream_selected_compact_mask(
        const void *model_map,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t mask) {
    (void)n_total_expert;
    if (mask == 0) return 1;
    if (!selected_ids || !cuda_stream_selected_ensure_stream()) return 0;
    cudaError_t err = cudaSuccess;
    for (uint32_t i = 0; i < n_selected; i++) {
        if ((mask & (1u << i)) == 0) continue;
        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            selected_ids[i],
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        if (idx < 0) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected resident expert missing during compact\n");
            return 0;
        }
        cuda_stream_resident_expert &entry =
            g_stream_resident_experts[(size_t)idx];
        entry.last_used = ++g_stream_resident_clock;
        err = cudaMemcpyAsync(g_stream_selected_cache.gate +
                                  (uint64_t)i * gate_expert_bytes,
                              entry.gate,
                              (size_t)gate_expert_bytes,
                              cudaMemcpyDeviceToDevice,
                              g_model_upload_stream);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_selected_cache.up +
                                      (uint64_t)i * gate_expert_bytes,
                                  entry.up,
                                  (size_t)gate_expert_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_selected_cache.down +
                                      (uint64_t)i * down_expert_bytes,
                                  entry.down,
                                  (size_t)down_expert_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  g_model_upload_stream);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected compact copy failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected compact sync failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_selected_prepare_ptrs(
        const void *model_map,
        uint32_t layer,
        const int32_t *selected_ids,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!selected_ids ||
        n_selected == 0 ||
        n_selected > DS4_ROCM_N_EXPERT_USED ||
        !g_stream_selected_cache.gate_ptrs ||
        !g_stream_selected_cache.up_ptrs ||
        !g_stream_selected_cache.down_ptrs ||
        !cuda_stream_selected_ensure_stream()) {
        return 0;
    }
    const char *gate_ptrs[DS4_ROCM_N_EXPERT_USED] = {0};
    const char *up_ptrs[DS4_ROCM_N_EXPERT_USED] = {0};
    const char *down_ptrs[DS4_ROCM_N_EXPERT_USED] = {0};
    for (uint32_t i = 0; i < n_selected; i++) {
        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            selected_ids[i],
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        if (idx < 0) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected pointer expert missing\n");
            return 0;
        }
        cuda_stream_resident_expert &entry =
            g_stream_resident_experts[(size_t)idx];
        entry.last_used = ++g_stream_resident_clock;
        gate_ptrs[i] = entry.gate;
        up_ptrs[i] = entry.up;
        down_ptrs[i] = entry.down;
    }
    cudaError_t err = cudaMemcpyAsync(g_stream_selected_cache.gate_ptrs,
                                      gate_ptrs,
                                      n_selected * sizeof(gate_ptrs[0]),
                                      cudaMemcpyHostToDevice,
                                      g_model_upload_stream);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(g_stream_selected_cache.up_ptrs,
                              up_ptrs,
                              n_selected * sizeof(up_ptrs[0]),
                              cudaMemcpyHostToDevice,
                              g_model_upload_stream);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(g_stream_selected_cache.down_ptrs,
                              down_ptrs,
                              n_selected * sizeof(down_ptrs[0]),
                              cudaMemcpyHostToDevice,
                              g_model_upload_stream);
    }
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected pointer upload failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "streaming selected pointer upload sync failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cuda_stream_batch_selected_prepare_from_host(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const int32_t *ids,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char ***gate_ptrs,
        const char ***up_ptrs,
        const char ***down_ptrs,
        uint32_t *unique_out,
        int begin_pending) {
    if (!g_ssd_streaming_mode ||
        !model_map ||
        !ids ||
        !selected_exec ||
        !gate_ptrs ||
        !up_ptrs ||
        !down_ptrs ||
        !unique_out ||
        n_tokens <= 1 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_selected == 0 ||
        n_selected > DS4_ROCM_N_EXPERT_USED ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }
    if (cuda_stream_batch_selected_cache_apply(model_map,
                                               layer,
                                               n_total_expert,
                                               n_selected,
                                               n_tokens,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               selected_exec,
                                               gate_ptrs,
                                               up_ptrs,
                                               down_ptrs,
                                               unique_out)) {
        return 1;
    }
    g_stream_batch_selected_cache.loaded = 0;

    uint64_t n_ids64 = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_selected, &n_ids64) ||
        n_ids64 > SIZE_MAX / sizeof(int32_t)) {
        return 0;
    }
    int32_t *compact_ids = (int32_t *)malloc((size_t)n_ids64 * sizeof(compact_ids[0]));
    if (!compact_ids) {
        free(compact_ids);
        return 0;
    }
    uint8_t *pair_missing = (uint8_t *)malloc((size_t)n_ids64);
    if (!pair_missing) {
        free(compact_ids);
        return 0;
    }

    int ok = 1;
    int32_t map[DS4_ROCM_MAX_N_EXPERT];
    int32_t unique_ids[DS4_ROCM_MAX_N_EXPERT];
    uint8_t unique_missing[DS4_ROCM_MAX_N_EXPERT] = {0};
    for (uint32_t i = 0; i < DS4_ROCM_MAX_N_EXPERT; i++) map[i] = -1;
    uint32_t unique_count = 0;
    if (ok) {
        for (uint64_t i = 0; i < n_ids64; i++) {
            const int32_t expert = ids[i];
            if (expert < 0 || (uint32_t)expert >= n_total_expert) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "streaming batch selected expert id %d outside 0..%u "
                        "(layer=%u)\n",
                        expert,
                        n_total_expert,
                        layer);
                ok = 0;
                break;
            }
            int32_t slot = map[(uint32_t)expert];
            if (slot < 0) {
                if (unique_count >= DS4_ROCM_MAX_N_EXPERT) {
                    ok = 0;
                    break;
                }
                slot = (int32_t)unique_count;
                map[(uint32_t)expert] = slot;
                unique_ids[unique_count++] = expert;
            }
            compact_ids[i] = slot;
        }
    }
    if (ok && unique_count == 0) ok = 0;
    if (ok && !cuda_stream_batch_selected_ensure_buffers(n_ids64, unique_count)) {
        ok = 0;
    }
    if (ok && !cuda_stream_selected_ensure_stream()) ok = 0;
    if (ok && cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.batch_calls++;
        g_stream_cache_stats.batch_unique += unique_count;
    }

    cuda_stream_read_job read_jobs[DS4_ROCM_STREAM_READ_MAX_JOBS];
    memset(read_jobs, 0, sizeof(read_jobs));
    uint32_t read_job_count = 0;
    const int use_fd =
        g_model_fd >= 0 &&
        (g_model_fd_host_base == NULL || model_map == g_model_fd_host_base);

    const char *gate_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *up_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *down_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *resident_gate_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *resident_up_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *missing_gate_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *missing_up_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    uint32_t resident_count = 0;
    uint32_t missing_count = 0;

    for (uint32_t u = 0; ok && u < unique_count; u++) {
        const int32_t expert_i = unique_ids[u];
        const uint64_t expert = (uint64_t)(uint32_t)expert_i;
        uint64_t gate_rel = 0;
        uint64_t down_rel = 0;
        if (!cuda_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !cuda_u64_mul_checked(expert, down_expert_bytes, &down_rel) ||
            gate_rel > model_size ||
            down_rel > model_size ||
            gate_offset > model_size ||
            up_offset > model_size ||
            down_offset > model_size ||
            gate_rel > model_size - gate_offset ||
            gate_rel > model_size - up_offset ||
            down_rel > model_size - down_offset ||
            gate_expert_bytes > model_size - gate_offset - gate_rel ||
            gate_expert_bytes > model_size - up_offset - gate_rel ||
            down_expert_bytes > model_size - down_offset - down_rel) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming batch selected expert offset overflow\n");
            ok = 0;
            break;
        }

        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            expert_i,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        const int was_resident = idx >= 0;
        if (cuda_stream_cache_stats_on()) {
            if (was_resident) {
                g_stream_cache_stats.batch_hits++;
            } else {
                g_stream_cache_stats.batch_misses++;
            }
        }
        if (idx < 0) {
            idx = cuda_stream_resident_alloc(model_map,
                                             layer,
                                             expert_i,
                                             unique_ids,
                                             unique_count,
                                             gate_offset,
                                             up_offset,
                                             down_offset,
                                             gate_expert_bytes,
                                             down_expert_bytes);
            if (idx < 0) {
                ok = 0;
                break;
            }
            cuda_stream_resident_expert &entry =
                g_stream_resident_experts[(size_t)idx];
            if (use_fd) {
                if (read_job_count + 3u > DS4_ROCM_STREAM_READ_MAX_JOBS) {
                    if (!cuda_stream_flush_read_jobs(read_jobs, &read_job_count)) {
                        ok = 0;
                        break;
                    }
                }
                read_jobs[read_job_count++] =
                    {entry.gate, gate_offset + gate_rel, gate_expert_bytes,
                     NULL, NULL, 0, 0};
                read_jobs[read_job_count++] =
                    {entry.up, up_offset + gate_rel, gate_expert_bytes,
                     NULL, NULL, 0, 0};
                read_jobs[read_job_count++] =
                    {entry.down, down_offset + down_rel, down_expert_bytes,
                     NULL, NULL, 0, 0};
            } else {
                cudaError_t err = cudaMemcpyAsync(entry.gate,
                                                  (const char *)model_map + gate_offset + gate_rel,
                                                  (size_t)gate_expert_bytes,
                                                  cudaMemcpyHostToDevice,
                                                  g_model_upload_stream);
                if (err == cudaSuccess) {
                    err = cudaMemcpyAsync(entry.up,
                                          (const char *)model_map + up_offset + gate_rel,
                                          (size_t)gate_expert_bytes,
                                          cudaMemcpyHostToDevice,
                                          g_model_upload_stream);
                }
                if (err == cudaSuccess) {
                    err = cudaMemcpyAsync(entry.down,
                                          (const char *)model_map + down_offset + down_rel,
                                          (size_t)down_expert_bytes,
                                          cudaMemcpyHostToDevice,
                                          g_model_upload_stream);
                }
                if (err != cudaSuccess) {
                    fprintf(stderr,
                            DS4_GPU_LOG_PREFIX "streaming batch selected cached copy failed: %s\n",
                            cudaGetErrorString(err));
                    (void)cudaGetLastError();
                    ok = 0;
                    break;
                }
            }
        }
        if (idx >= 0) {
            cuda_stream_resident_expert &entry =
                g_stream_resident_experts[(size_t)idx];
            entry.last_used = ++g_stream_resident_clock;
            gate_host[u] = entry.gate;
            up_host[u] = entry.up;
            down_host[u] = entry.down;
            if (was_resident) {
                resident_gate_host[u] = entry.gate;
                resident_up_host[u] = entry.up;
                resident_count++;
            } else {
                unique_missing[u] = 1;
                missing_gate_host[u] = entry.gate;
                missing_up_host[u] = entry.up;
                missing_count++;
            }
        }
    }

    if (ok) {
        for (uint64_t i = 0; i < n_ids64; i++) {
            const int32_t slot = compact_ids[i];
            if (slot < 0 || (uint32_t)slot >= unique_count) {
                ok = 0;
                break;
            }
            pair_missing[i] = unique_missing[(uint32_t)slot];
        }
    }

    if (ok && begin_pending && use_fd && read_job_count != 0) {
        memset(&g_stream_batch_selected_pending, 0,
               sizeof(g_stream_batch_selected_pending));
        g_stream_batch_selected_pending.active = 1;
        g_stream_batch_selected_pending.model_map = model_map;
        g_stream_batch_selected_pending.layer = layer;
        g_stream_batch_selected_pending.n_total_expert = n_total_expert;
        g_stream_batch_selected_pending.n_selected = n_selected;
        g_stream_batch_selected_pending.n_tokens = n_tokens;
        g_stream_batch_selected_pending.n_unique = unique_count;
        g_stream_batch_selected_pending.resident_count = resident_count;
        g_stream_batch_selected_pending.missing_count = missing_count;
        g_stream_batch_selected_pending.gate_offset = gate_offset;
        g_stream_batch_selected_pending.up_offset = up_offset;
        g_stream_batch_selected_pending.down_offset = down_offset;
        g_stream_batch_selected_pending.gate_expert_bytes = gate_expert_bytes;
        g_stream_batch_selected_pending.down_expert_bytes = down_expert_bytes;
        g_stream_batch_selected_pending.read_job_count = read_job_count;
        memcpy(g_stream_batch_selected_pending.read_jobs,
               read_jobs,
               (size_t)read_job_count * sizeof(read_jobs[0]));
        if (!cuda_stream_read_jobs_start(g_stream_batch_selected_pending.read_jobs,
                                         read_job_count)) {
            memset(&g_stream_batch_selected_pending, 0,
                   sizeof(g_stream_batch_selected_pending));
            ok = 0;
        } else {
            read_job_count = 0;
        }
    }
    if (ok && !cuda_stream_flush_read_jobs(read_jobs, &read_job_count)) {
        ok = 0;
    }
    if (ok && !use_fd) {
        cudaError_t err = cudaStreamSynchronize(g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch selected upload sync failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            ok = 0;
        }
    }
    if (ok) {
        cudaError_t err = cudaMemcpyAsync(g_stream_batch_selected_cache.selected_ids,
                                          compact_ids,
                                          (size_t)n_ids64 * sizeof(compact_ids[0]),
                                          cudaMemcpyHostToDevice,
                                          g_model_upload_stream);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.pair_missing,
                                  pair_missing,
                                  (size_t)n_ids64,
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.gate_ptrs,
                                  gate_host,
                                  unique_count * sizeof(gate_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.up_ptrs,
                                  up_host,
                                  unique_count * sizeof(up_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.down_ptrs,
                                  down_host,
                                  unique_count * sizeof(down_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.resident_gate_ptrs,
                                  resident_gate_host,
                                  unique_count * sizeof(resident_gate_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.resident_up_ptrs,
                                  resident_up_host,
                                  unique_count * sizeof(resident_up_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.missing_gate_ptrs,
                                  missing_gate_host,
                                  unique_count * sizeof(missing_gate_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.missing_up_ptrs,
                                  missing_up_host,
                                  unique_count * sizeof(missing_up_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming batch selected table upload failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            if (g_stream_batch_selected_pending.active) {
                cuda_stream_batch_selected_abort_pending();
            }
            ok = 0;
        }
    }

    if (ok) {
        g_stream_batch_selected_cache.loaded =
            g_stream_batch_selected_pending.active ? 0 : 1;
        g_stream_batch_selected_cache.model_map = model_map;
        g_stream_batch_selected_cache.layer = layer;
        g_stream_batch_selected_cache.n_total_expert = n_total_expert;
        g_stream_batch_selected_cache.n_selected = n_selected;
        g_stream_batch_selected_cache.n_tokens = n_tokens;
        g_stream_batch_selected_cache.n_unique = unique_count;
        g_stream_batch_selected_cache.gate_offset = gate_offset;
        g_stream_batch_selected_cache.up_offset = up_offset;
        g_stream_batch_selected_cache.down_offset = down_offset;
        g_stream_batch_selected_cache.gate_expert_bytes = gate_expert_bytes;
        g_stream_batch_selected_cache.down_expert_bytes = down_expert_bytes;
        *selected_exec = &g_stream_batch_selected_cache.selected_tensor;
        *gate_ptrs = g_stream_batch_selected_cache.gate_ptrs;
        *up_ptrs = g_stream_batch_selected_cache.up_ptrs;
        *down_ptrs = g_stream_batch_selected_cache.down_ptrs;
        *unique_out = unique_count;
    } else {
        g_stream_batch_selected_cache.loaded = 0;
        if (g_stream_batch_selected_pending.active) {
            cuda_stream_batch_selected_abort_pending();
        }
        if (read_job_count != 0) cuda_stream_read_jobs_free(read_jobs, read_job_count);
    }

    free(compact_ids);
    free(pair_missing);
    return ok;
}

static int cuda_stream_batch_selected_prepare(
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char ***gate_ptrs,
        const char ***up_ptrs,
        const char ***down_ptrs,
        uint32_t *unique_out) {
    if (!selected ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_selected, sizeof(int32_t))) {
        return 0;
    }
    if (cuda_stream_batch_selected_cache_apply(model_map,
                                               layer,
                                               n_total_expert,
                                               n_selected,
                                               n_tokens,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               selected_exec,
                                               gate_ptrs,
                                               up_ptrs,
                                               down_ptrs,
                                               unique_out)) {
        return 1;
    }

    uint64_t n_ids64 = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_selected, &n_ids64) ||
        n_ids64 > SIZE_MAX / sizeof(int32_t)) {
        return 0;
    }
    int32_t *ids = (int32_t *)malloc((size_t)n_ids64 * sizeof(ids[0]));
    if (!ids) return 0;

    const int copy_ok = cuda_ok(cudaMemcpy(ids,
                                           selected->ptr,
                                           (size_t)n_ids64 * sizeof(ids[0]),
                                           cudaMemcpyDeviceToHost),
                                "streaming batch selected ids copy");
    const int ok = copy_ok &&
        cuda_stream_batch_selected_prepare_from_host(model_map,
                                                     model_size,
                                                     layer,
                                                     ids,
                                                     n_tokens,
                                                     n_total_expert,
                                                     n_selected,
                                                     gate_offset,
                                                     up_offset,
                                                     down_offset,
                                                     gate_expert_bytes,
                                                     down_expert_bytes,
                                                     selected_exec,
                                                     gate_ptrs,
                                                     up_ptrs,
                                                     down_ptrs,
                                                     unique_out,
                                                     0);
    free(ids);
    return ok;
}

static int cuda_stream_layer_expert_cache_prepare_batch(
        const void *model_map,
        uint32_t layer,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char ***gate_ptrs,
        const char ***up_ptrs,
        const char ***down_ptrs,
        uint32_t *unique_out) {
    if (!selected ||
        !selected_exec ||
        !gate_ptrs ||
        !up_ptrs ||
        !down_ptrs ||
        !unique_out ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_selected, sizeof(int32_t)) ||
        n_tokens <= 1 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_selected == 0 ||
        n_selected > DS4_ROCM_N_EXPERT_USED ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }
    if (cuda_stream_batch_selected_cache_apply(model_map,
                                               layer,
                                               n_total_expert,
                                               n_selected,
                                               n_tokens,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               selected_exec,
                                               gate_ptrs,
                                               up_ptrs,
                                               down_ptrs,
                                               unique_out)) {
        return 1;
    }

    const char *layer_gate = NULL;
    const char *layer_up = NULL;
    const char *layer_down = NULL;
    if (!cuda_stream_layer_expert_cache_apply(model_map,
                                              layer,
                                              n_total_expert,
                                              gate_offset,
                                              up_offset,
                                              down_offset,
                                              gate_expert_bytes,
                                              down_expert_bytes,
                                              &layer_gate,
                                              &layer_up,
                                              &layer_down)) {
        return 0;
    }

    uint64_t n_ids64 = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_selected, &n_ids64) ||
        n_ids64 > SIZE_MAX / sizeof(int32_t)) {
        return 0;
    }
    int32_t *ids = (int32_t *)malloc((size_t)n_ids64 * sizeof(ids[0]));
    int32_t *compact_ids =
        (int32_t *)malloc((size_t)n_ids64 * sizeof(compact_ids[0]));
    if (!ids || !compact_ids) {
        free(ids);
        free(compact_ids);
        return 0;
    }

    int ok = cuda_ok(cudaMemcpy(ids,
                                selected->ptr,
                                (size_t)n_ids64 * sizeof(ids[0]),
                                cudaMemcpyDeviceToHost),
                     "streaming full-layer selected ids copy");

    int32_t map[DS4_ROCM_MAX_N_EXPERT];
    int32_t unique_ids[DS4_ROCM_MAX_N_EXPERT];
    for (uint32_t i = 0; i < DS4_ROCM_MAX_N_EXPERT; i++) map[i] = -1;
    uint32_t unique_count = 0;
    for (uint64_t i = 0; ok && i < n_ids64; i++) {
        const int32_t expert = ids[i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer selected expert id %d "
                    "outside 0..%u (layer=%u)\n",
                    expert,
                    n_total_expert,
                    layer);
            ok = 0;
            break;
        }
        int32_t slot = map[(uint32_t)expert];
        if (slot < 0) {
            if (unique_count >= DS4_ROCM_MAX_N_EXPERT) {
                ok = 0;
                break;
            }
            slot = (int32_t)unique_count;
            map[(uint32_t)expert] = slot;
            unique_ids[unique_count++] = expert;
        }
        compact_ids[i] = slot;
    }
    if (ok && unique_count == 0) ok = 0;
    if (ok && !cuda_stream_batch_selected_ensure_buffers(n_ids64, unique_count)) {
        ok = 0;
    }

    const char *gate_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *up_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    const char *down_host[DS4_ROCM_MAX_N_EXPERT] = {0};
    for (uint32_t u = 0; ok && u < unique_count; u++) {
        const uint64_t expert = (uint64_t)(uint32_t)unique_ids[u];
        uint64_t gate_rel = 0;
        uint64_t down_rel = 0;
        if (!cuda_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !cuda_u64_mul_checked(expert, down_expert_bytes, &down_rel)) {
            ok = 0;
            break;
        }
        gate_host[u] = layer_gate + gate_rel;
        up_host[u] = layer_up + gate_rel;
        down_host[u] = layer_down + down_rel;
    }

    if (ok) {
        cudaError_t err = cudaMemcpyAsync(g_stream_batch_selected_cache.selected_ids,
                                          compact_ids,
                                          (size_t)n_ids64 * sizeof(compact_ids[0]),
                                          cudaMemcpyHostToDevice,
                                          g_model_upload_stream);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.gate_ptrs,
                                  gate_host,
                                  unique_count * sizeof(gate_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.up_ptrs,
                                  up_host,
                                  unique_count * sizeof(up_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(g_stream_batch_selected_cache.down_ptrs,
                                  down_host,
                                  unique_count * sizeof(down_host[0]),
                                  cudaMemcpyHostToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) err = cudaStreamSynchronize(g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer selected table upload failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            ok = 0;
        }
    }

    if (ok) {
        g_stream_batch_selected_cache.loaded = 1;
        g_stream_batch_selected_cache.model_map = model_map;
        g_stream_batch_selected_cache.layer = layer;
        g_stream_batch_selected_cache.n_total_expert = n_total_expert;
        g_stream_batch_selected_cache.n_selected = n_selected;
        g_stream_batch_selected_cache.n_tokens = n_tokens;
        g_stream_batch_selected_cache.n_unique = unique_count;
        g_stream_batch_selected_cache.gate_offset = gate_offset;
        g_stream_batch_selected_cache.up_offset = up_offset;
        g_stream_batch_selected_cache.down_offset = down_offset;
        g_stream_batch_selected_cache.gate_expert_bytes = gate_expert_bytes;
        g_stream_batch_selected_cache.down_expert_bytes = down_expert_bytes;
        *selected_exec = &g_stream_batch_selected_cache.selected_tensor;
        *gate_ptrs = g_stream_batch_selected_cache.gate_ptrs;
        *up_ptrs = g_stream_batch_selected_cache.up_ptrs;
        *down_ptrs = g_stream_batch_selected_cache.down_ptrs;
        *unique_out = unique_count;
    } else {
        g_stream_batch_selected_cache.loaded = 0;
    }

    free(ids);
    free(compact_ids);
    return ok;
}

static int cuda_stream_layer_expert_cache_seed_selected(
        const void *model_map,
        uint32_t layer,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens,
        uint32_t n_seed_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_ssd_streaming_mode ||
        !model_map ||
        !selected ||
        n_tokens == 0 ||
        n_seed_tokens == 0 ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_selected == 0 ||
        n_selected > DS4_ROCM_N_EXPERT_USED ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_selected, sizeof(int32_t))) {
        return 0;
    }

    const char *layer_gate = NULL;
    const char *layer_up = NULL;
    const char *layer_down = NULL;
    if (!cuda_stream_layer_expert_cache_apply(model_map,
                                              layer,
                                              n_total_expert,
                                              gate_offset,
                                              up_offset,
                                              down_offset,
                                              gate_expert_bytes,
                                              down_expert_bytes,
                                              &layer_gate,
                                              &layer_up,
                                              &layer_down)) {
        return 0;
    }

    if (n_seed_tokens > n_tokens) n_seed_tokens = n_tokens;
    const uint64_t n_ids64 = (uint64_t)n_seed_tokens * n_selected;
    if (n_ids64 == 0 || n_ids64 > SIZE_MAX / sizeof(int32_t)) return 0;
    int32_t ids_stack[DS4_ROCM_N_EXPERT_USED * 16u];
    int32_t *ids_heap = NULL;
    int32_t *ids = ids_stack;
    if (n_ids64 > sizeof(ids_stack) / sizeof(ids_stack[0])) {
        ids_heap = (int32_t *)malloc((size_t)n_ids64 * sizeof(ids_heap[0]));
        if (!ids_heap) return 0;
        ids = ids_heap;
    }

    const uint64_t src_off =
        (uint64_t)(n_tokens - n_seed_tokens) * n_selected * sizeof(int32_t);
    int ok = cuda_ok(cudaMemcpy(ids,
                                (const char *)selected->ptr + src_off,
                                (size_t)n_ids64 * sizeof(ids[0]),
                                cudaMemcpyDeviceToHost),
                     "streaming full-layer seed selected ids copy");

    int32_t unique_stack[DS4_ROCM_N_EXPERT_USED * 16u];
    int32_t *unique_heap = NULL;
    int32_t *unique = unique_stack;
    if (n_ids64 > sizeof(unique_stack) / sizeof(unique_stack[0])) {
        unique_heap = (int32_t *)malloc((size_t)n_ids64 * sizeof(unique_heap[0]));
        if (!unique_heap) ok = 0;
        unique = unique_heap;
    }
    uint32_t unique_count = 0;
    bool seen[DS4_ROCM_MAX_N_EXPERT] = {0};
    for (uint64_t i = 0; ok && i < n_ids64; i++) {
        const int32_t expert = ids[i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer seed expert id %d "
                    "outside 0..%u (layer=%u)\n",
                    expert,
                    n_total_expert,
                    layer);
            ok = 0;
            break;
        }
        if (seen[(uint32_t)expert]) continue;
        seen[(uint32_t)expert] = true;
        unique[unique_count++] = expert;
    }
    if (ok && cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.seed_calls++;
        g_stream_cache_stats.seed_unique += unique_count;
    }

    if (ok && unique_count != 0 && !cuda_stream_selected_ensure_stream()) {
        ok = 0;
    }

    for (uint32_t u = 0; ok && u < unique_count; u++) {
        const int32_t expert_i32 = unique[u];
        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            expert_i32,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        if (idx >= 0) {
            g_stream_resident_experts[(size_t)idx].last_used =
                ++g_stream_resident_clock;
            continue;
        }

        idx = cuda_stream_resident_alloc(model_map,
                                         layer,
                                         expert_i32,
                                         unique,
                                         unique_count,
                                         gate_offset,
                                         up_offset,
                                         down_offset,
                                         gate_expert_bytes,
                                         down_expert_bytes);
        if (idx < 0) {
            ok = 0;
            break;
        }

        const uint64_t expert = (uint64_t)(uint32_t)expert_i32;
        uint64_t gate_rel = 0;
        uint64_t down_rel = 0;
        if (!cuda_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !cuda_u64_mul_checked(expert, down_expert_bytes, &down_rel)) {
            ok = 0;
            break;
        }

        cuda_stream_resident_expert &entry =
            g_stream_resident_experts[(size_t)idx];
        cudaError_t err = cudaMemcpyAsync(entry.gate,
                                          layer_gate + gate_rel,
                                          (size_t)gate_expert_bytes,
                                          cudaMemcpyDeviceToDevice,
                                          g_model_upload_stream);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(entry.up,
                                  layer_up + gate_rel,
                                  (size_t)gate_expert_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  g_model_upload_stream);
        }
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(entry.down,
                                  layer_down + down_rel,
                                  (size_t)down_expert_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  g_model_upload_stream);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer seed D2D copy failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            ok = 0;
            break;
        }
    }

    if (ok && unique_count != 0) {
        cudaError_t err = cudaStreamSynchronize(g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming full-layer seed sync failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            ok = 0;
        }
    }

    if (!ok) cuda_stream_resident_cache_release();
    free(ids_heap);
    free(unique_heap);
    return ok;
}

static int cuda_stream_selected_copy_range(
        const void *model_map,
        char *dst,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (!model_map || !dst || bytes == 0) return 0;
    if (!cuda_stream_selected_ensure_stream()) return 0;

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    const int use_fd =
        g_model_fd >= 0 &&
        (g_model_fd_host_base == NULL || model_map == g_model_fd_host_base);
    if (use_fd && !cuda_model_stage_pool_alloc(stage_bytes)) return 0;

    uint64_t copied = 0;
    while (copied < bytes) {
        const uint64_t n = bytes - copied < chunk ? bytes - copied : chunk;
        const uint64_t chunk_idx = use_fd ? g_stream_selected_stage_counter++ : 0u;
        const uint64_t bi = use_fd ? chunk_idx % 4u : 0u;
        const char *payload = NULL;
        if (use_fd) {
            if (chunk_idx >= 4u) {
                cudaError_t err = cudaEventSynchronize(g_model_stage_event[bi]);
                if (err != cudaSuccess) {
                    fprintf(stderr,
                            DS4_GPU_LOG_PREFIX "streaming selected staging wait failed for %s: %s\n",
                            what ? what : "expert",
                            cudaGetErrorString(err));
                    (void)cudaGetLastError();
                    return 0;
                }
            }
            if (!cuda_model_stage_read(g_model_stage[bi],
                                       g_model_stage_bytes,
                                       offset + copied,
                                       n,
                                       &payload)) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "streaming selected read failed for %s at %.2f MiB: %s\n",
                        what ? what : "expert",
                        (double)copied / 1048576.0,
                        strerror(errno));
                return 0;
            }
        } else {
            payload = (const char *)model_map + offset + copied;
        }
        cudaError_t err = cudaMemcpyAsync(dst + copied,
                                          payload,
                                          (size_t)n,
                                          cudaMemcpyHostToDevice,
                                          g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        if (use_fd) {
            err = cudaEventRecord(g_model_stage_event[bi],
                                  g_model_upload_stream);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "streaming selected staging record failed for %s at %.2f MiB: %s\n",
                        what ? what : "expert",
                        (double)copied / 1048576.0,
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
                return 0;
            }
        }
        copied += n;
    }
    return 1;
}

static int cuda_stream_selected_load(
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
    g_stream_selected_cache.loaded = 0;
    memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || !selected_ids ||
        n_total_expert == 0 ||
        n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_selected == 0 ||
        n_selected > DS4_ROCM_N_EXPERT_USED ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0) {
        return 0;
    }
    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    if (!cuda_u64_mul_checked(n_selected, gate_expert_bytes, &gate_bytes) ||
        !cuda_u64_mul_checked(n_selected, down_expert_bytes, &down_bytes)) {
        return 0;
    }
    if (!cuda_stream_selected_reuse_wait("streaming selected cache reuse")) {
        return 0;
    }
    if (!cuda_stream_selected_ensure_buffers(gate_bytes, down_bytes)) return 0;
    if (!cuda_stream_selected_ensure_stream()) return 0;
    cuda_stream_selected_cache_header(model_map,
                                      layer,
                                      n_total_expert,
                                      n_selected,
                                      selected_ids,
                                      gate_expert_bytes,
                                      down_expert_bytes);
    if (cuda_stream_cache_stats_on()) {
        g_stream_cache_stats.selected_calls++;
        g_stream_cache_stats.selected_slots += n_selected;
    }

    cuda_stream_read_job read_jobs[DS4_ROCM_N_EXPERT_USED * 3u];
    memset(read_jobs, 0, sizeof(read_jobs));
    uint32_t read_job_count = 0;
    uint32_t resident_mask = 0;
    uint32_t missing_mask = 0;
    const int use_fd =
        g_model_fd >= 0 &&
        (g_model_fd_host_base == NULL || model_map == g_model_fd_host_base);

    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "streaming selected expert id %d outside 0..%u "
                    "(layer=%u slot=%u selected=[%d,%d,%d,%d,%d,%d])\n",
                    selected_ids[i],
                    n_total_expert,
                    layer,
                    i,
                    n_selected > 0 ? selected_ids[0] : -1,
                    n_selected > 1 ? selected_ids[1] : -1,
                    n_selected > 2 ? selected_ids[2] : -1,
                    n_selected > 3 ? selected_ids[3] : -1,
                    n_selected > 4 ? selected_ids[4] : -1,
                    n_selected > 5 ? selected_ids[5] : -1);
            return 0;
        }
        const uint64_t expert = (uint64_t)(uint32_t)selected_ids[i];
        uint64_t gate_rel = 0;
        uint64_t down_rel = 0;
        if (!cuda_u64_mul_checked(expert, gate_expert_bytes, &gate_rel) ||
            !cuda_u64_mul_checked(expert, down_expert_bytes, &down_rel) ||
            gate_rel > model_size ||
            down_rel > model_size ||
            gate_offset > model_size ||
            up_offset > model_size ||
            down_offset > model_size ||
            gate_rel > model_size - gate_offset ||
            gate_rel > model_size - up_offset ||
            down_rel > model_size - down_offset ||
            gate_expert_bytes > model_size - gate_offset - gate_rel ||
            gate_expert_bytes > model_size - up_offset - gate_rel ||
            down_expert_bytes > model_size - down_offset - down_rel) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected expert offset overflow\n");
            return 0;
        }

        int idx = cuda_stream_resident_find(model_map,
                                            layer,
                                            selected_ids[i],
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes);
        if (cuda_stream_cache_stats_on()) {
            if (idx >= 0) {
                g_stream_cache_stats.selected_hits++;
            } else {
                g_stream_cache_stats.selected_misses++;
            }
        }
        if (idx >= 0) {
            g_stream_resident_experts[(size_t)idx].last_used =
                ++g_stream_resident_clock;
            resident_mask |= 1u << i;
            continue;
        }

        idx = cuda_stream_resident_alloc(model_map,
                                         layer,
                                         selected_ids[i],
                                         selected_ids,
                                         n_selected,
                                         gate_offset,
                                         up_offset,
                                         down_offset,
                                         gate_expert_bytes,
                                         down_expert_bytes);
        if (idx < 0) return 0;
        missing_mask |= 1u << i;
        cuda_stream_resident_expert &entry =
            g_stream_resident_experts[(size_t)idx];

        if (use_fd) {
            if (read_job_count + 3u > DS4_ROCM_N_EXPERT_USED * 3u) return 0;
            read_jobs[read_job_count++] =
                {entry.gate, gate_offset + gate_rel, gate_expert_bytes,
                 NULL, NULL, 0, 0};
            read_jobs[read_job_count++] =
                {entry.up, up_offset + gate_rel, gate_expert_bytes,
                 NULL, NULL, 0, 0};
            read_jobs[read_job_count++] =
                {entry.down, down_offset + down_rel, down_expert_bytes,
                 NULL, NULL, 0, 0};
        } else {
            cudaError_t err = cudaMemcpyAsync(entry.gate,
                                              (const char *)model_map + gate_offset + gate_rel,
                                              (size_t)gate_expert_bytes,
                                              cudaMemcpyHostToDevice,
                                              g_model_upload_stream);
            if (err == cudaSuccess) {
                err = cudaMemcpyAsync(entry.up,
                                      (const char *)model_map + up_offset + gate_rel,
                                      (size_t)gate_expert_bytes,
                                      cudaMemcpyHostToDevice,
                                      g_model_upload_stream);
            }
            if (err == cudaSuccess) {
                err = cudaMemcpyAsync(entry.down,
                                      (const char *)model_map + down_offset + down_rel,
                                      (size_t)down_expert_bytes,
                                      cudaMemcpyHostToDevice,
                                      g_model_upload_stream);
            }
            if (err != cudaSuccess) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "streaming selected cached copy failed: %s\n",
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
                cuda_stream_resident_cache_release();
                return 0;
            }
        }
    }

    if (use_fd && read_job_count != 0 && resident_mask != 0) {
        g_stream_selected_pending.active = 1;
        g_stream_selected_pending.model_map = model_map;
        g_stream_selected_pending.layer = layer;
        g_stream_selected_pending.n_total_expert = n_total_expert;
        g_stream_selected_pending.n_selected = n_selected;
        g_stream_selected_pending.gate_offset = gate_offset;
        g_stream_selected_pending.up_offset = up_offset;
        g_stream_selected_pending.down_offset = down_offset;
        g_stream_selected_pending.gate_expert_bytes = gate_expert_bytes;
        g_stream_selected_pending.down_expert_bytes = down_expert_bytes;
        g_stream_selected_pending.resident_mask = resident_mask;
        g_stream_selected_pending.missing_mask = missing_mask;
        g_stream_selected_pending.read_job_count = read_job_count;
        for (uint32_t i = 0; i < n_selected; i++) {
            g_stream_selected_pending.selected_ids[i] = selected_ids[i];
        }
        memcpy(g_stream_selected_pending.read_jobs,
               read_jobs,
               (size_t)read_job_count * sizeof(read_jobs[0]));
        if (!cuda_stream_read_jobs_start(g_stream_selected_pending.read_jobs,
                                         read_job_count)) {
            memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
            cuda_stream_read_jobs_free(read_jobs, read_job_count);
            cuda_stream_resident_cache_release();
            return 0;
        }
        if (!cuda_stream_selected_prepare_ptrs(model_map,
                                               layer,
                                               selected_ids,
                                               n_selected,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes)) {
            (void)cuda_stream_read_jobs_wait(g_stream_selected_pending.read_jobs,
                                             read_job_count);
            cuda_stream_read_jobs_free(g_stream_selected_pending.read_jobs,
                                       read_job_count);
            memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
            cuda_stream_read_jobs_free(read_jobs, read_job_count);
            cuda_stream_resident_cache_release();
            return 0;
        }
        return 1;
    }

    if (resident_mask != 0 &&
        !cuda_stream_selected_compact_mask(model_map,
                                           layer,
                                           selected_ids,
                                           n_total_expert,
                                           n_selected,
                                           gate_offset,
                                           up_offset,
                                           down_offset,
                                           gate_expert_bytes,
                                           down_expert_bytes,
                                           resident_mask)) {
        cuda_stream_read_jobs_free(read_jobs, read_job_count);
        cuda_stream_resident_cache_release();
        return 0;
    }

    if (read_job_count != 0) {
        if (!cuda_stream_read_jobs_parallel(read_jobs, read_job_count) ||
            !cuda_stream_selected_upload_read_jobs(read_jobs, read_job_count)) {
            cuda_stream_read_jobs_free(read_jobs, read_job_count);
            cuda_stream_resident_cache_release();
            return 0;
        }
    } else {
        cudaError_t err = cudaStreamSynchronize(g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "streaming selected upload sync failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_stream_read_jobs_free(read_jobs, read_job_count);
            cuda_stream_resident_cache_release();
            return 0;
        }
    }
    cuda_stream_read_jobs_free(read_jobs, read_job_count);

    {
        const uint32_t all_mask =
            n_selected >= 32u ? 0xffffffffu : ((1u << n_selected) - 1u);
        const uint32_t compact_mask = resident_mask != 0 ? missing_mask : all_mask;
        if (!cuda_stream_selected_compact_mask(model_map,
                                               layer,
                                               selected_ids,
                                               n_total_expert,
                                               n_selected,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               compact_mask)) {
            cuda_stream_resident_cache_release();
            return 0;
        }
    }

    g_stream_selected_cache.loaded = 1;
    return 1;
}

static int cuda_stream_selected_pending_matches(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!g_stream_selected_pending.active ||
        g_routed_moe_selected_override_n != n_selected ||
        g_stream_selected_pending.model_map != model_map ||
        g_stream_selected_pending.layer != layer ||
        g_stream_selected_pending.n_total_expert != n_total_expert ||
        g_stream_selected_pending.n_selected != n_selected ||
        g_stream_selected_pending.gate_expert_bytes != gate_expert_bytes ||
        g_stream_selected_pending.down_expert_bytes != down_expert_bytes) {
        return 0;
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        if (g_stream_selected_pending.selected_ids[i] !=
            g_routed_moe_selected_override[i]) {
            return 0;
        }
    }
    return 1;
}

static int cuda_stream_selected_finish_pending_missing(uint32_t compact_mask);

static int cuda_stream_selected_apply_split(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char **gate_w,
        const char **up_w,
        const char **down_w,
        const char ***gate_ptrs,
        const char ***up_ptrs,
        const char ***down_ptrs,
        uint32_t *resident_mask,
        uint32_t *missing_mask) {
    if (!g_ssd_streaming_mode ||
        !selected_exec ||
        !gate_w ||
        !up_w ||
        !down_w ||
        !gate_ptrs ||
        !up_ptrs ||
        !down_ptrs ||
        !resident_mask ||
        !missing_mask ||
        !cuda_stream_selected_pending_matches(model_map,
                                              layer,
                                              n_total_expert,
                                              n_selected,
                                              gate_expert_bytes,
                                              down_expert_bytes)) {
        return 0;
    }
    if (g_stream_selected_pending.resident_mask == 0 ||
        g_stream_selected_pending.missing_mask == 0 ||
        !g_stream_selected_cache.gate_ptrs ||
        !g_stream_selected_cache.up_ptrs ||
        !g_stream_selected_cache.down_ptrs) {
        return 0;
    }
    *selected_exec = &g_stream_selected_cache.slot_tensor;
    *gate_w = g_stream_selected_cache.gate;
    *up_w = g_stream_selected_cache.up;
    *down_w = g_stream_selected_cache.down;
    *gate_ptrs = g_stream_selected_cache.gate_ptrs;
    *up_ptrs = g_stream_selected_cache.up_ptrs;
    *down_ptrs = g_stream_selected_cache.down_ptrs;
    *resident_mask = g_stream_selected_pending.resident_mask;
    *missing_mask = g_stream_selected_pending.missing_mask;
    g_routed_moe_selected_override_n = 0;
    return 1;
}

static int cuda_stream_selected_finish_pending_missing(uint32_t compact_mask) {
    if (!g_stream_selected_pending.active) return 1;
    const uint32_t read_job_count = g_stream_selected_pending.read_job_count;
    if (!cuda_stream_read_jobs_wait(g_stream_selected_pending.read_jobs,
                                    read_job_count) ||
        !cuda_stream_selected_upload_read_jobs(g_stream_selected_pending.read_jobs,
                                              read_job_count)) {
        cuda_stream_read_jobs_free(g_stream_selected_pending.read_jobs,
                                   read_job_count);
        memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
        cuda_stream_resident_cache_release();
        return 0;
    }
    cuda_stream_read_jobs_free(g_stream_selected_pending.read_jobs,
                               read_job_count);
    if (compact_mask != 0 &&
        !cuda_stream_selected_compact_mask(
                g_stream_selected_pending.model_map,
                g_stream_selected_pending.layer,
                g_stream_selected_pending.selected_ids,
                g_stream_selected_pending.n_total_expert,
                g_stream_selected_pending.n_selected,
                g_stream_selected_pending.gate_offset,
                g_stream_selected_pending.up_offset,
                g_stream_selected_pending.down_offset,
                g_stream_selected_pending.gate_expert_bytes,
                g_stream_selected_pending.down_expert_bytes,
                compact_mask)) {
        memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
        cuda_stream_resident_cache_release();
        return 0;
    }
    g_stream_selected_cache.loaded = compact_mask != 0 ? 1 : 0;
    memset(&g_stream_selected_pending, 0, sizeof(g_stream_selected_pending));
    return 1;
}

static int cuda_stream_selected_apply(
        const void *model_map,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const ds4_gpu_tensor **selected_exec,
        const char **gate_w,
        const char **up_w,
        const char **down_w) {
    if (g_ssd_streaming_mode &&
        !g_stream_selected_cache.loaded &&
        getenv("DS4_ROCM_DISABLE_STREAMING_SPLIT_SELECTED") != NULL &&
        cuda_stream_selected_pending_matches(model_map,
                                             layer,
                                             n_total_expert,
                                             n_selected,
                                             gate_expert_bytes,
                                             down_expert_bytes)) {
        const uint32_t compact_mask =
            g_stream_selected_pending.resident_mask |
            g_stream_selected_pending.missing_mask;
        if (!cuda_stream_selected_finish_pending_missing(compact_mask)) {
            return 0;
        }
    }
    if (!g_ssd_streaming_mode ||
        !g_stream_selected_cache.loaded ||
        !selected_exec ||
        !gate_w ||
        !up_w ||
        !down_w ||
        g_routed_moe_selected_override_n != n_selected ||
        g_stream_selected_cache.model_map != model_map ||
        g_stream_selected_cache.layer != layer ||
        g_stream_selected_cache.n_total_expert != n_total_expert ||
        g_stream_selected_cache.n_selected != n_selected ||
        g_stream_selected_cache.gate_expert_bytes != gate_expert_bytes ||
        g_stream_selected_cache.down_expert_bytes != down_expert_bytes) {
        return 0;
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        if (g_stream_selected_cache.selected_ids[i] !=
            g_routed_moe_selected_override[i]) {
            return 0;
        }
    }
    *selected_exec = &g_stream_selected_cache.slot_tensor;
    *gate_w = g_stream_selected_cache.gate;
    *up_w = g_stream_selected_cache.up;
    *down_w = g_stream_selected_cache.down;
    g_routed_moe_selected_override_n = 0;
    return 1;
}

static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    const char *owned = cuda_model_image_ptr(model_map, offset);
    if (owned) return owned;
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

static const char *cuda_model_range_copy_uncached(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    const char *src = (const char *)model_map + offset;
    err = cudaMemcpy(dev, src, (size_t)bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range copy failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return NULL;
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_bytes += bytes;
    return (const char *)dev;
}

static const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    if (bytes == 0) return cuda_model_ptr(model_map, offset);
    if (cuda_model_image_owned(model_map)) return cuda_model_ptr(model_map, offset);

    const uint64_t end = offset + bytes;
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && end >= offset && bytes <= r.bytes) return r.device_ptr;
    }
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
        if (r.host_base == model_map && r.host_registered && r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return r.registered_device_base + (h0 - r0);
        }
    }

    const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
    if (fd_ptr) return fd_ptr;

    if (model_map != g_model_host_base) {
        return cuda_model_range_copy_uncached(model_map, offset, bytes, what);
    }

    cudaError_t err = cudaSuccess;
    if (g_model_range_mapping_supported && model_map == g_model_host_base) {
        const long page_sz_l = sysconf(_SC_PAGESIZE);
        const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
        const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
        const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
        const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
        const uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
        void *reg_dev = NULL;
        err = cudaHostRegister((void *)reg_addr,
                               (size_t)reg_bytes,
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        if (err == cudaSuccess) {
            err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
            if (err == cudaSuccess && reg_dev) {
                char *dev_ptr = (char *)reg_dev + reg_delta;
                g_model_ranges.push_back({model_map, offset, bytes, dev_ptr, (void *)reg_addr, (char *)reg_dev, reg_bytes, 1, 0});
                g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
                return dev_ptr;
            }
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range map pointer failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaHostUnregister((void *)reg_addr);
            (void)cudaGetLastError();
        } else {
            if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) g_model_range_mapping_supported = 0;
            (void)cudaGetLastError();
        }
    }

    void *dev = NULL;
    err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    return (const char *)dev;
}

static int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;
    if (cuda_model_image_owned(model_map)) return 1;

    const uint64_t end = offset + bytes;
    if (end < offset) return 0;
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            offset >= r.offset &&
            end <= r.offset + r.bytes) {
            return 1;
        }
        if (r.host_base == model_map &&
            r.host_registered &&
            r.registered_base &&
            r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return 1;
        }
    }
    return 0;
}

static void cuda_q8_f16_cache_release_all(void) {
    for (const cuda_q8_f16_transpose_range &r : g_q8_f16_transpose_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_transpose_ranges.clear();
    g_q8_f16_transpose_by_offset.clear();
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

static int cuda_env_present(const char *env) {
    if (env != NULL) return env[0] != '\0' && strcmp(env, "0") != 0;
    return 0;
}

static uint32_t cuda_rows_per_block_or_default(uint32_t v, uint32_t def) {
    return (v == 1u || v == 2u || v == 4u || v == 8u || v == 16u || v == 32u) ? v : def;
}

struct ds4_rocm_runtime_config {
    int initialized;
    int q8_prequant_decode;
    int disable_splitk_attn_out_low;
    int disable_shared_gate_up_fused_w32;
    int attention_output_cublas_all;
    int shared_down_cublas;
    int graph_dump;
    uint32_t q8_decode_rpb;
    uint32_t q8_hc_decode_rpb;
    uint32_t attn_out_low_decode_rpb;
    uint32_t moe_decode_rpb;
    int oldhip_attention_decode;
};

static ds4_rocm_runtime_config g_rocm_cfg;

static const ds4_rocm_runtime_config *cuda_runtime_config(void) {
    if (!g_rocm_cfg.initialized) {
        g_rocm_cfg.q8_prequant_decode = !g_quality_mode;
        g_rocm_cfg.disable_splitk_attn_out_low = !g_quality_mode;
        g_rocm_cfg.disable_shared_gate_up_fused_w32 = !g_quality_mode;
        g_rocm_cfg.attention_output_cublas_all = !g_quality_mode;
        g_rocm_cfg.shared_down_cublas = !g_quality_mode;
        g_rocm_cfg.graph_dump = cuda_env_present(getenv("DS4_METAL_GRAPH_DUMP_PREFIX"));
        g_rocm_cfg.q8_decode_rpb = g_quality_mode ? 8u : 1u;
        g_rocm_cfg.q8_hc_decode_rpb = g_quality_mode ? 8u : 16u;
        g_rocm_cfg.attn_out_low_decode_rpb = g_quality_mode ? 8u : 32u;
        g_rocm_cfg.moe_decode_rpb = g_quality_mode ? 8u : 1u;
        g_rocm_cfg.oldhip_attention_decode = !g_quality_mode;
        g_rocm_cfg.initialized = 1;
    }
    return &g_rocm_cfg;
}

static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    return UINT64_MAX;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    if (g_ssd_streaming_mode) {
        return cuda_stream_resident_free_reserve_bytes();
    }
    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* The expanded Q8->F16 cache is only an acceleration path.  Keep enough
     * device memory free for cuBLAS workspaces, transient graph buffers, and
     * driver bookkeeping instead of letting optional cached weights consume the
     * last few GiB on 96 GiB cards. */
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u; /* 5% */
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void cuda_q8_f16_cache_budget_notice(
        const char *reason,
        uint64_t request_bytes,
        uint64_t free_bytes,
        uint64_t total_bytes,
        uint64_t reserve_bytes,
        uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
}

static int cuda_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    const uint64_t limit = cuda_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes ||
        free_bytes - request_bytes < reserve_bytes) {
        cuda_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

static void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX "q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty()) {
        (void)cudaDeviceSynchronize();
        cuda_q8_f16_cache_release_all();
    }
    (void)cudaGetLastError();
}

static int cuda_q8_f16_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_quality_mode) return 0;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (g_q8_f16_disabled_for_multi_model) return 0;
    if (getenv("DS4_CUDA_NO_Q8_F16_CACHE") != NULL) return 0;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return 1;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return 1;
    }
    if (strstr(label, "ffn_gate_shexp") != NULL ||
        strstr(label, "ffn_up_shexp") != NULL ||
        strstr(label, "ffn_down_shexp") != NULL) {
        return 1;
    }
    return (in_dim == 4096u && out_dim == 2048u) ||
           (in_dim == 2048u && out_dim == 4096u) ||
           (in_dim == 4096u && out_dim == 1024u) ||
           (in_dim == 4096u && out_dim == 512u) ||
           (in_dim == 1024u && out_dim == 32768u);
}

static int cuda_q8_label_is_attention_output(const char *label) {
    return label &&
           (strstr(label, "attn_output_a") != NULL ||
            strstr(label, "attn_output_b") != NULL ||
            strstr(label, "attention_output_a") != NULL ||
            strstr(label, "attention_output_b") != NULL);
}

static int cuda_q8_f16_preload_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (cuda_q8_label_is_attention_output(label) &&
        !cuda_runtime_config()->attention_output_cublas_all) {
        return 0;
    }
    return cuda_q8_f16_cache_allowed(label, in_dim, out_dim);
}

static const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_by_offset.find(offset);
    if (exact != g_q8_f16_by_offset.end()) {
        const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    uint64_t out_bytes = 0;
    if (in_dim == 0u || out_dim == 0u ||
        !cuda_u64_mul3_checked(in_dim, out_dim, sizeof(__half), &out_bytes)) return NULL;
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;

    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    return dev;
}

static const __half *cuda_q8_f16_transpose_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_transpose_by_offset.find(offset);
    if (exact != g_q8_f16_transpose_by_offset.end()) {
        const cuda_q8_f16_transpose_range &r = g_q8_f16_transpose_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    for (const cuda_q8_f16_transpose_range &r : g_q8_f16_transpose_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;
    uint64_t out_bytes = 0;
    if (in_dim == 0u || out_dim == 0u ||
        !cuda_u64_mul3_checked(in_dim, out_dim, sizeof(__half), &out_bytes)) return NULL;
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;
    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;
    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "q8 fp16 transpose cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("transpose allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_transpose_kernel<<<(n + 255u) / 256u, 256>>>(dev,
                                                                     (const unsigned char *)q8,
                                                                     in_dim,
                                                                     out_dim,
                                                                     blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 transpose dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("transpose launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_transpose_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_transpose_by_offset[offset] = g_q8_f16_transpose_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    return dev;
}

static uint32_t cuda_prefill_warmup_tokens(void) {
    uint32_t n_tok = 2048u;
    const char *chunk_env = getenv("DS4_METAL_PREFILL_CHUNK");
    if (chunk_env && chunk_env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(chunk_env, &end, 10);
        if (end != chunk_env && *end == '\0' && v > 0 && v <= 4096u) n_tok = (uint32_t)v;
    }
    return n_tok;
}

static void cuda_q8_f16_warmup_attention_output_a_gemm(const __half *out_a_f16,
                                                       uint64_t group_dim,
                                                       uint64_t rank,
                                                       uint32_t n_groups) {
    static int warmed = 0;
    if (warmed || !g_cublas_ready || !out_a_f16 || group_dim == 0 || rank == 0 || n_groups == 0) return;
    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    if (!cfg->attention_output_cublas_all) return;
    warmed = 1;
    const uint32_t n_tok = cuda_prefill_warmup_tokens();
    const uint64_t heads_h_count = (uint64_t)n_groups * n_tok * group_dim;
    const uint64_t low_h_count = (uint64_t)n_groups * n_tok * rank;
    const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
    const uint64_t low_h_off = (heads_h_bytes + 255ull) & ~255ull;
    if (low_h_count > (UINT64_MAX - low_h_off) / sizeof(__half)) return;
    void *tmp = cuda_tmp_alloc(low_h_off + low_h_count * sizeof(__half), "attention output a warmup");
    if (!tmp) return;
    __half *heads_h = (__half *)tmp;
    __half *low_h = (__half *)((char *)tmp + low_h_off);
    if (cudaMemset(heads_h, 0, (size_t)heads_h_bytes) != cudaSuccess) return;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas,
                                                   CUBLAS_OP_T,
                                                   CUBLAS_OP_N,
                                                   (int)rank,
                                                   (int)n_tok,
                                                   (int)group_dim,
                                                   &alpha,
                                                   out_a_f16,
                                                   CUDA_R_16F,
                                                   (int)group_dim,
                                                   (long long)rank * (long long)group_dim,
                                                   heads_h,
                                                   CUDA_R_16F,
                                                   (int)group_dim,
                                                   (long long)n_tok * (long long)group_dim,
                                                   &beta,
                                                   low_h,
                                                   CUDA_R_16F,
                                                   (int)(n_groups * rank),
                                                   (long long)rank,
                                                   (int)n_groups,
                                                   CUBLAS_COMPUTE_32F,
                                                   CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) (void)cudaDeviceSynchronize();
}

static void cuda_q8_f16_warmup_attention_output_b_gemm(const __half *out_b_f16_t,
                                                       uint64_t low_dim,
                                                       uint64_t out_dim) {
    static int warmed = 0;
    if (warmed || !g_cublas_ready || !out_b_f16_t || low_dim == 0 || out_dim == 0) return;
    if (!cuda_runtime_config()->attention_output_cublas_all) return;
    warmed = 1;
    const uint32_t n_tok = cuda_prefill_warmup_tokens();
    const uint64_t low_h_count = (uint64_t)n_tok * low_dim;
    const uint64_t out_count = (uint64_t)n_tok * out_dim;
    const uint64_t low_h_bytes = low_h_count * sizeof(__half);
    const uint64_t out_off = (low_h_bytes + 255ull) & ~255ull;
    if (out_count > (UINT64_MAX - out_off) / sizeof(float)) return;
    void *tmp = cuda_tmp_alloc(out_off + out_count * sizeof(float), "attention output b warmup");
    if (!tmp) return;
    __half *low_h = (__half *)tmp;
    float *out = (float *)((char *)tmp + out_off);
    if (cudaMemset(low_h, 0, (size_t)low_h_bytes) != cudaSuccess) return;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)low_dim,
                                     &alpha,
                                     out_b_f16_t,
                                     CUDA_R_16F,
                                     (int)out_dim,
                                     low_h,
                                     CUDA_R_16F,
                                     (int)low_dim,
                                     &beta,
                                     out,
                                     CUDA_R_32F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) (void)cudaDeviceSynchronize();
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "%s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cuda_model_load_progress_enabled(void) {
    return 1;
}

static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    if (!cuda_model_load_progress_enabled()) return;

    const double now = cuda_wall_sec();
    if (!g_model_load_progress_started) {
        g_model_load_progress_started = 1;
        g_model_load_progress_tty = isatty(STDERR_FILENO) != 0;
        g_model_load_progress_next = (g_model_load_progress_tty ? 2ull : 16ull) *
                                     1024ull * 1024ull * 1024ull;
        g_model_load_progress_last = now;
        if (g_model_load_progress_tty) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors into device cache: 0.00 GiB");
        } else {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors into device cache\n");
        }
    }

    if (cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (g_model_load_progress_tty ? 2.0 : 10.0)) {
        return;
    }

    if (g_model_load_progress_tty) {
        fprintf(stderr, "\r" DS4_GPU_LOG_PREFIX "loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "loading model tensors %.2f GiB cached\n",
                (double)cached_bytes / 1073741824.0);
    }
    fflush(stderr);
    g_model_load_progress_last = now;
    const uint64_t step = (g_model_load_progress_tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}

static uint64_t cuda_model_copy_chunk_bytes(void) {
    return 64ull * 1048576ull;
}

static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (!model_map || bytes == 0 || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 || bytes == 0) return;
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)offset;
    (void)bytes;
#endif
}

static uint64_t cuda_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t cuda_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

static void *cuda_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static int cuda_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    uint64_t alloc_bytes = bytes;
    if (g_model_direct_align > 1u) {
        const uint64_t pad = g_model_direct_align - 1u;
        if (alloc_bytes > UINT64_MAX - pad) return 0;
        alloc_bytes += pad;
    }
    if (alloc_bytes > (uint64_t)SIZE_MAX) return 0;
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)alloc_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging event creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX) ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (g_model_direct_fd >= 0 && g_model_direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = cuda_round_down(offset, g_model_direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = cuda_round_up(delta + bytes, g_model_direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (cuda_pread_full(g_model_direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
            const int direct_errno = errno;
            if (direct_errno == EINVAL || direct_errno == EFAULT || direct_errno == ENOTSUP || direct_errno == EOPNOTSUPP) {
                (void)close(g_model_direct_fd);
                g_model_direct_fd = -1;
                g_model_direct_align = 1;
            }
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return cuda_pread_full(g_model_fd, stage, bytes, offset);
}

static uint64_t cuda_model_cache_limit_bytes(void) {
    return UINT64_MAX;
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t bytes = 1792ull * 1048576ull;
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

static char *cuda_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) return NULL;
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (cuda_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = cuda_model_arena_chunk_bytes(aligned);
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    return (char *)dev;
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        return cuda_model_ptr(model_map, offset);
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        return cuda_model_ptr(model_map, offset);
    }
    cudaError_t err = cudaSuccess;

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging record failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, g_model_registered_size, offset + copied, n);
        copied += n;
        cuda_model_load_progress_note(g_model_range_bytes + copied);
        chunk_idx++;
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    return (const char *)dev;
}

static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (cuda_model_image_owned(model_map)) {
        g_model_host_base = model_map;
        g_model_device_base = cuda_model_image_ptr(model_map, 0);
        g_model_registered_size = model_size;
        g_model_device_owned = 1;
        return 1;
    }

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, DS4_GPU_LOG_PREFIX "chunk-copying %.2f GiB model image\n",
            (double)model_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) {
        (void)cudaFree(dev);
        return 0;
    }

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < model_size) {
        const uint64_t n = (model_size - copied < chunk) ? (model_size - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging wait failed: %s\n", cudaGetErrorString(err));
                (void)cudaFree(dev);
                (void)cudaGetLastError();
                return 0;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   copied, n, &payload)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staged read failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, strerror(errno));
            (void)cudaFree(dev);
            return 0;
        }
        err = cudaMemcpyAsync((char *)dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "model staging record failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_drop_file_pages(copied, n);
        cuda_model_discard_source_pages(model_map, model_size, copied, n);
        copied += n;
        chunk_idx++;
        cuda_model_load_progress_note(copied > map_offset ? copied - map_offset : 0);
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "model upload sync failed: %s\n", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }
    g_model_images.push_back({model_map, model_size, (char *)dev});
    g_model_host_base = model_map;
    g_model_device_base = (const char *)dev;
    g_model_registered_size = model_size;
    g_model_device_owned = 1;
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
            t1 - t0,
            (double)map_size / 1073741824.0);
    return 1;
}

static void cuda_model_range_release_all(void) {
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_registered && r.registered_base) {
            (void)cudaHostUnregister(r.registered_base);
        } else if (r.device_ptr && !r.arena_allocated) {
            (void)cudaFree(r.device_ptr);
        }
    }
    for (const cuda_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)cudaFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
    g_stream_selected_cache.loaded = 0;
    cuda_stream_resident_cache_release();
    cuda_model_load_progress_reset();
}

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: " DS4_GPU_BLAS_NAME " %s failed: status %d\n", what, (int)st);
    return 0;
}


extern "C" int ds4_gpu_init(void) {
    int dev = 0;
    if (!cuda_ok(cudaSetDevice(dev), "set device")) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    }
    if (!g_cublas_ready) {
        if (!cublas_ok(cublasCreate(&g_cublas), "create handle")) return 0;
        const cublasMath_t math_mode = g_quality_mode ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
        g_cublas_ready = 1;
    }
#ifdef __HIP_PLATFORM_AMD__
    if (!g_hipblaslt_ready) {
        if (hipblaslt_ok(hipblasLtCreate(&g_hipblaslt), "create handle")) {
            g_hipblaslt_ready = 1;
        }
    }
#endif
    return 1;
}

extern "C" void ds4_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    cuda_stream_cache_stats_print("cleanup");
    cuda_shared_gate_up_async_cleanup();
#ifdef __HIP_PLATFORM_AMD__
    hipblaslt_gemm_plan_clear();
#endif
    if (g_cublas_ready) {
        (void)cublasDestroy(g_cublas);
        g_cublas_ready = 0;
        g_cublas = NULL;
    }
#ifdef __HIP_PLATFORM_AMD__
    if (g_hipblaslt_ready) {
        (void)hipblasLtDestroy(g_hipblaslt);
        g_hipblaslt_ready = 0;
        g_hipblaslt = NULL;
    }
#endif
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    cuda_stream_selected_cache_release();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_disabled_for_multi_model = 0;
    g_q8_f16_budget_notice_printed = 0;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
    if (g_selected_readback_stream) {
        (void)cudaStreamDestroy(g_selected_readback_stream);
        g_selected_readback_stream = NULL;
    }
    if (g_selected_readback_event) {
        (void)cudaEventDestroy(g_selected_readback_event);
        g_selected_readback_event = NULL;
    }
    g_selected_readback_event_value = 0;
    cuda_model_image_release_all();
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_fd = -1;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

static uint64_t cuda_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    /* Very large KV caches are where device-only cudaMalloc() can make a
     * unified-memory machine unresponsive.  Managed memory restores the old
     * demand-paged behavior for this one long-lived allocation class only. */
    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    return t;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) (void)cudaFree(tensor->ptr);
    free(tensor);
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    (void)cudaDeviceSynchronize();
    return tensor->ptr;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
    return cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice), "tensor write");
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost), "tensor read");
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    return cuda_ok(cudaMemcpy((char *)dst->ptr + dst_offset,
                              (const char *)src->ptr + src_offset,
                              (size_t)bytes,
                              cudaMemcpyDeviceToDevice),
                   "tensor copy");
}

extern "C" int ds4_gpu_begin_commands(void) { return 1; }
extern "C" int ds4_gpu_flush_commands(void) { return cuda_ok(cudaDeviceSynchronize(), "flush"); }
extern "C" int ds4_gpu_end_commands(void) {
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}
extern "C" int ds4_gpu_synchronize(void) { return cuda_ok(cudaDeviceSynchronize(), "synchronize"); }

extern "C" int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;
    const int multi_model =
        g_model_host_base != NULL &&
        (g_model_host_base != model_map || g_model_registered_size != model_size);
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    if (multi_model) {
        /*
         * MTP loads a second GGUF mapping.  Its weights are small, but on UMA
         * ROCm systems the optional expanded Q8->F16 cache can consume the
         * memory margin needed for session/context tensors once both model
         * mappings are resident.  The cache is only a speed path; the normal
         * Q8 kernels remain available and keep MTP startup reliable.
         */
        g_q8_f16_disabled_for_multi_model = 1;
    }
    g_model_host_base = model_map;
    g_model_device_base = cuda_model_image_owned(model_map) ?
                          cuda_model_image_ptr(model_map, 0) :
                          (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_device_owned = cuda_model_image_owned(model_map);
    g_model_range_mapping_supported = 1;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }

    /* Strix Halo uses the staged full-copy path in ds4_gpu_set_model_map_range().
     * Avoid host-registering the mmap here: that would make the staged copier
     * believe the model is already device-resident. */
    return 1;
}

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 ||
        map_offset > model_size ||
        map_size > model_size - map_offset) {
        return 0;
    }
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (g_ssd_streaming_mode) {
        if (!cuda_model_range_ptr(model_map, map_offset, map_size, "stream_range")) return 0;
        return cuda_model_range_is_cached(model_map, map_offset, map_size);
    }
    /*
     * Do not eagerly copy a contiguous model image here.  On Strix Halo the
     * caller immediately follows with accelerator_cache_model_tensors(), which
     * prepares the exact tensor spans selected by --layers.  Copying here would
     * either allocate the whole GGUF image or, for sparse span sets, an oversized
     * envelope before the precise tensor-span cache gets a chance to run.
     */
    return 1;
}

extern "C" int ds4_gpu_set_model_map_spans(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size ||
            sizes[i] == 0 ||
            sizes[i] > model_size - offsets[i]) {
            return 0;
        }
    }
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (g_ssd_streaming_mode) {
        for (uint32_t i = 0; i < count; i++) {
            if (!cuda_model_range_ptr(model_map, offsets[i], sizes[i], "stream_span")) return 0;
            if (!cuda_model_range_is_cached(model_map, offsets[i], sizes[i])) return 0;
        }
        return 1;
    }
    /*
     * The spans can be sparse distributed layer slices.  Materializing their
     * min..max envelope can be much larger than the actual selected tensors.
     * Leave the precise per-tensor preparation to accelerator_cache_model_tensors().
     */
    return 1;
}

extern "C" int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    g_model_fd_host_base = g_model_host_base;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
            }
        }
#endif
    }
    return 1;
}

extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    if (!cuda_model_range_ptr(model_map, offset, bytes, label ? label : "model_tensor")) return 0;
    return cuda_model_range_is_cached(model_map, offset, bytes);
}

extern "C" int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    static int optional_q8_preload_disabled = 0;
    if (optional_q8_preload_disabled) return 1;
    const char *cache_label = label ? label : "q8_0";
    if (!cuda_q8_f16_preload_allowed(cache_label, in_dim, out_dim)) return 1;
    const int preload_transposed_b = !g_quality_mode &&
                                     strstr(cache_label, "attn_output_b") != NULL;
    if (preload_transposed_b) {
        const __half *f16_t = cuda_q8_f16_transpose_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label);
        if (f16_t) {
            if (strstr(cache_label, "attn_output_b") != NULL && in_dim == 8192u && out_dim == 4096u) {
                cuda_q8_f16_warmup_attention_output_b_gemm(f16_t, in_dim, out_dim);
            }
            return 1;
        }
    } else {
        const __half *f16 = cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label);
        if (f16) {
            if (strstr(cache_label, "attn_output_a") != NULL && in_dim == 4096u && out_dim == 8192u) {
                cuda_q8_f16_warmup_attention_output_a_gemm(f16, in_dim, 1024u, 8u);
            }
            return 1;
        }
    }
    optional_q8_preload_disabled = 1;
    return 1;
}

extern "C" void ds4_gpu_release_q8_f16_cache(void) {
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
}

extern "C" void ds4_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "memory %s: query failed: %s\n",
                label ? label : "", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return;
    }
    const uint64_t used_b = (uint64_t)total_b - (uint64_t)free_b;
    const char *placement = cuda_model_image_bytes() ? "device_copy" : "mapped/range_cache";
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "memory %s: used=%.2f GiB free=%.2f GiB total=%.2f GiB "
            "placement=%s model_image=%.2f GiB range_cache=%.2f GiB "
            "q8_f16_cache=%.2f GiB scratch=%.2f GiB",
            label ? label : "",
            (double)used_b / 1073741824.0,
            (double)free_b / 1073741824.0,
            (double)total_b / 1073741824.0,
            placement,
            (double)cuda_model_image_bytes() / 1073741824.0,
            (double)g_model_range_bytes / 1073741824.0,
            (double)g_q8_f16_bytes / 1073741824.0,
            (double)g_cuda_tmp_bytes / 1073741824.0);
    fprintf(stderr, "\n");
}

extern "C" void ds4_gpu_set_quality(bool quality) {
    const int new_quality_mode = quality ? 1 : 0;
    if (g_quality_mode != new_quality_mode) {
        g_rocm_cfg.initialized = 0;
    }
    g_quality_mode = new_quality_mode;
    if (g_cublas_ready) {
        const cublasMath_t math_mode = g_quality_mode ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
    }
}
