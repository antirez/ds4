#include "pulsar_cuda_internal.h"


static const void *g_model_host_base;


static const char *g_model_device_base;


static uint64_t g_model_registered_size;


static int g_model_registered;


static int g_model_device_owned;


static int g_model_range_mapping_supported = 1;


static int g_model_hmm_direct;


static int g_model_fd = -1;


static const void *g_model_fd_host_base;


static int g_model_direct_fd = -1;


static uint64_t g_model_direct_align = 1;


static uint64_t g_model_file_size;


static int g_model_cache_full;


static int g_model_mapping_failure_notice_printed;


static cudaStream_t g_model_prefetch_stream;


static cudaStream_t g_model_upload_stream;


cublasHandle_t g_cublas;


int g_cublas_ready;


int g_quality_mode;


static cudaGraphExec_t g_decode_graph_exec;
static int g_decode_graph_capturing;
static int g_decode_graphs_off = -1;
static uint64_t g_decode_graph_updates, g_decode_graph_instantiates;





static std::vector<cuda_model_range> g_model_ranges;


static std::vector<cuda_model_arena> g_model_arenas;


static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;




static uint64_t g_model_range_bytes;


static uint64_t g_model_load_progress_next;


static uint64_t g_model_load_progress_last_bytes = UINT64_MAX;


static uint64_t g_model_load_progress_last_cgib = UINT64_MAX;


static double g_model_load_progress_last;


static int g_model_load_progress_started;


static int g_model_load_progress_tty;


static void *g_cuda_tmp;


static uint64_t g_cuda_tmp_bytes;


static void *g_model_stage_raw[4];


static void *g_model_stage[4];


static cudaEvent_t g_model_stage_event[4];


static uint64_t g_model_stage_bytes;



int cuda_ok(cudaError_t err, const char *what);


static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);


static const char *cuda_model_direct_fallback_ptr(const void *model_map, uint64_t offset);




static uint64_t cuda_model_cache_limit_bytes(void);


static uint64_t cuda_model_local_model_limit_bytes(void);


static int cuda_model_cache_limit_explicit(void);


void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
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
        fprintf(stderr, "pulsar: CUDA temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_cuda_tmp = ptr;
    g_cuda_tmp_bytes = bytes;
    return g_cuda_tmp;
}



int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= PULSAR_CUDA_ATTENTION_SCORE_CAP - PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP;
}



static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}



static const char *cuda_model_range_register_mapped(const void *model_map,
                                                    uint64_t offset,
                                                    uint64_t bytes,
                                                    const char *what) {
    if (!g_model_range_mapping_supported || bytes == 0) return NULL;

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
    uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
    if (model_map == g_model_host_base &&
        g_model_registered_size >= 88ull * 1073741824ull &&
        g_model_registered_size <= 96ull * 1073741824ull &&
        g_model_range_bytes >= 80ull * 1073741824ull) {
        const uintptr_t model_base = (uintptr_t)model_map;
        const uintptr_t model_end = model_base + (uintptr_t)g_model_registered_size;
        if (model_end > model_base && model_end > reg_addr) {
            const uint64_t tail_bytes = (uint64_t)(model_end - reg_addr);
            reg_bytes = (tail_bytes + page_sz - 1u) & ~(page_sz - 1u);
        }
    }
    void *reg_dev = NULL;

    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    if (getenv("PULSAR_CUDA_HOST_REGISTER_PLAIN") != NULL) {
        flags = cudaHostRegisterMapped;
    }

    cudaError_t err = cudaHostRegister((void *)reg_addr,
                                       (size_t)reg_bytes,
                                       flags);
    if (err != cudaSuccess &&
        (flags & cudaHostRegisterReadOnly) != 0 &&
        (err == cudaErrorNotSupported || err == cudaErrorInvalidValue)) {
        (void)cudaGetLastError();
        err = cudaHostRegister((void *)reg_addr,
                               (size_t)reg_bytes,
                               cudaHostRegisterMapped);
    }
    if (err == cudaSuccess) {
        err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
        if (err == cudaSuccess && reg_dev) {
            char *dev_ptr = (char *)reg_dev + reg_delta;
            g_model_ranges.push_back({model_map, offset, bytes, dev_ptr, (void *)reg_addr, (char *)reg_dev, reg_bytes, 1, 0});
            g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
            if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "pulsar: CUDA mapped %s %.2f MiB\n",
                        what ? what : "weights",
                        (double)bytes / 1048576.0);
            }
            return dev_ptr;
        }
        fprintf(stderr, "pulsar: CUDA model range map pointer failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaHostUnregister((void *)reg_addr);
        (void)cudaGetLastError();
        return NULL;
    }

    if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) {
        g_model_range_mapping_supported = 0;
    }
    if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "pulsar: CUDA model range map skipped for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
    }
    (void)cudaGetLastError();
    return NULL;
}



/* Allocate a device-resident copy of [offset, offset+bytes) from model_map and
 * push it into g_model_ranges so future cuda_model_range_ptr lookups hit it.
 * Returns the device pointer on success, NULL on cudaMalloc/cudaMemcpy failure.
 * Caller is responsible for any policy gating (budget cap, env opt-out, etc.) */
static const char *cuda_model_range_populate_device_copy(const void *model_map,
                                                          uint64_t offset,
                                                          uint64_t bytes,
                                                          const char *what) {
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "pulsar: CUDA skipped device copy for %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return NULL;
    }

    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "pulsar: CUDA model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "pulsar: CUDA model range copy failed for %s at %.2f/%.2f MiB: %s\n",
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
    if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "pulsar: CUDA cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}



const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    if (bytes == 0) return cuda_model_ptr(model_map, offset);

    /* Device-resident HBM cache hits win over UVA-mapped registered pointers:
     * direct HBM reads are ~10% faster than mapped reads through host page
     * tables (measured on plain decode at GB10).  Cache lookup runs first; the
     * registered-mapped shortcut below is the cold fallback when an allocation
     * hasn't been pre-populated. */
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

    if (g_model_device_owned || g_model_registered) return cuda_model_ptr(model_map, offset);
    if (g_model_hmm_direct &&
        getenv("PULSAR_CUDA_WEIGHT_CACHE") == NULL &&
        getenv("PULSAR_CUDA_WEIGHT_PRELOAD") == NULL) {
        return cuda_model_ptr(model_map, offset);
    }
    const char *direct_env = getenv("PULSAR_CUDA_DIRECT_MODEL");
    if (direct_env && direct_env[0]) return cuda_model_ptr(model_map, offset);

    if (getenv("PULSAR_CUDA_NO_FD_CACHE") == NULL) {
        const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
        if (fd_ptr) return fd_ptr;
    }

    const char *mapped = cuda_model_range_register_mapped(model_map, offset, bytes, what);
    if (mapped) return mapped;

    return cuda_model_range_populate_device_copy(model_map, offset, bytes, what);
}



static int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;
    if (g_model_device_owned || g_model_registered || g_model_hmm_direct) return 1;

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






int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "pulsar: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}



static double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}



static int cuda_model_load_progress_enabled(void) {
    if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE") != NULL) return 0;
    return 1;
}



static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last_bytes = UINT64_MAX;
    g_model_load_progress_last_cgib = UINT64_MAX;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}



static void cuda_model_load_progress_finish(void) {
    if (!g_model_load_progress_started) return;
    if (g_model_load_progress_tty) {
        fputc('\n', stderr);
        fflush(stderr);
    }
    g_model_load_progress_started = 0;
}



static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    if (!cuda_model_load_progress_enabled()) return;

    const double now = cuda_wall_sec();
    const int tty = isatty(STDERR_FILENO) != 0;
    const uint64_t step = (tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    const uint64_t display_cgib =
        cached_bytes > (UINT64_MAX - gib / 2ull) / 100ull ?
        UINT64_MAX : (cached_bytes * 100ull + gib / 2ull) / gib;
    if (g_model_load_progress_next == 0) {
        g_model_load_progress_next = step;
    }
    if (g_model_load_progress_last != 0.0 &&
        (cached_bytes == g_model_load_progress_last_bytes ||
         display_cgib == g_model_load_progress_last_cgib)) {
        return;
    }
    if (g_model_load_progress_last != 0.0 &&
        cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (tty ? 2.0 : 10.0)) {
        return;
    }

    g_model_load_progress_started = 1;
    g_model_load_progress_tty = tty;
    if (g_model_load_progress_tty) {
        fprintf(stderr, "\r\033[Kpulsar: CUDA loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        if (g_model_load_progress_last == 0.0) {
            fprintf(stderr, "pulsar: CUDA loading model tensors into device cache\n");
        } else {
            fprintf(stderr, "pulsar: CUDA loading model tensors %.2f GiB cached\n",
                    (double)cached_bytes / 1073741824.0);
        }
    }
    fflush(stderr);
    g_model_load_progress_last_bytes = cached_bytes;
    g_model_load_progress_last_cgib = display_cgib;
    g_model_load_progress_last = now;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}



static int cuda_model_prefetch_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("PULSAR_CUDA_NO_MODEL_PREFETCH") != NULL ||
        getenv("PULSAR_CUDA_COPY_MODEL") != NULL ||
        getenv("PULSAR_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("PULSAR_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    int pageable = 0;
    cudaError_t err = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device);
    if (err != cudaSuccess || !pageable) {
        (void)cudaGetLastError();
        return 0;
    }
    cudaMemLocation loc;
    memset(&loc, 0, sizeof(loc));
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + map_offset);
    const uintptr_t pre_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t pre_delta = (uint64_t)(host_addr - pre_addr);
    const uint64_t pre_bytes = (pre_delta + map_size + page_sz - 1u) & ~(page_sz - 1u);
    void *pre_ptr = (void *)pre_addr;

    const double t0 = cuda_wall_sec();
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetReadMostly, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "pulsar: CUDA model read-mostly advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetPreferredLocation, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "pulsar: CUDA model preferred-location advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    if (!g_model_prefetch_stream) {
        err = cudaStreamCreateWithFlags(&g_model_prefetch_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "pulsar: CUDA model prefetch stream creation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }

    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, 0, g_model_prefetch_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "pulsar: CUDA model prefetch skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    if (getenv("PULSAR_CUDA_MODEL_PREFETCH_SYNC") != NULL) {
        err = cudaStreamSynchronize(g_model_prefetch_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "pulsar: CUDA model prefetch sync failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "pulsar: CUDA ATS/HMM prefetch queued %.2f GiB of model tensors in %.3fs\n",
            (double)map_size / 1073741824.0,
            t1 - t0);
    g_model_hmm_direct = 1;
    return 1;
}



static uint64_t cuda_model_copy_chunk_bytes(void) {
    uint64_t mb = 64;
    const char *env = getenv("PULSAR_CUDA_MODEL_COPY_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 16) mb = 16;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}



static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (getenv("PULSAR_CUDA_KEEP_MODEL_PAGES") != NULL || !model_map || bytes == 0 || offset > model_size) return;
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
    if (g_model_fd < 0 || getenv("PULSAR_CUDA_KEEP_MODEL_PAGES") != NULL || bytes == 0) return;
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
            fprintf(stderr, "pulsar: CUDA model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "pulsar: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "pulsar: CUDA model staging event creation failed: %s\n", cudaGetErrorString(err));
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
                if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "pulsar: CUDA direct model read disabled: %s\n", strerror(direct_errno));
                }
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
    uint64_t gb = 0;
    const char *env = getenv("PULSAR_CUDA_WEIGHT_CACHE_LIMIT_GB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) gb = (uint64_t)v;
        return gb * 1073741824ull;
    }
    /* One Spark can run the IQ2 model (~81 GiB) and larger mixed models
     * (~91 GiB) via the old startup tensor cache.  Keep enough headroom for
     * scratch and KV; anything bigger is rejected at load so the model stays
     * fully resident, unless the operator opts into a larger cache budget
     * explicitly. */
    return 96ull * 1073741824ull;
}



static uint64_t cuda_model_local_model_limit_bytes(void) {
    const uint64_t default_limit = 96ull * 1073741824ull;
    if (!cuda_model_cache_limit_explicit()) return default_limit;
    const uint64_t explicit_limit = cuda_model_cache_limit_bytes();
    return explicit_limit > default_limit ? explicit_limit : default_limit;
}



static int cuda_model_cache_limit_explicit(void) {
    const char *env = getenv("PULSAR_CUDA_WEIGHT_CACHE_LIMIT_GB");
    return env && env[0];
}



static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t mb = 1792;
    const char *env = getenv("PULSAR_CUDA_WEIGHT_ARENA_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 256) mb = 256;
    if (mb > 8192) mb = 8192;
    uint64_t bytes = mb * 1048576ull;
    if (need > bytes / 2u) {
        const uint64_t align = 64ull * 1048576ull;
        return (need + align - 1u) & ~(align - 1u);
    }
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
        fprintf(stderr, "pulsar: CUDA model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE")) {
        uint64_t arena_bytes = 0;
        for (const cuda_model_arena &a : g_model_arenas) arena_bytes += a.bytes;
        fprintf(stderr, "pulsar: CUDA model arena allocated %.2f MiB (arenas %.2f GiB)\n",
                (double)chunk / 1048576.0,
                (double)arena_bytes / 1073741824.0);
    }
    return (char *)dev;
}



/* A raw host pointer is safe for kernels only after CUDA owns, registered, or
 * HMM-prefetched the mapping.  Otherwise let the caller try per-range mapping
 * or a device copy instead of surfacing an async illegal access later. */
static const char *cuda_model_direct_fallback_ptr(const void *model_map, uint64_t offset) {
    if (g_model_device_owned || g_model_registered || g_model_hmm_direct ||
        getenv("PULSAR_CUDA_DIRECT_MODEL") != NULL) {
        return cuda_model_ptr(model_map, offset);
    }
    return NULL;
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
        if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "pulsar: CUDA direct %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        if (getenv("PULSAR_CUDA_STRICT_WEIGHT_CACHE") != NULL) return NULL;
        return cuda_model_direct_fallback_ptr(model_map, offset);
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
                fprintf(stderr, "pulsar: CUDA model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, "pulsar: CUDA model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "pulsar: CUDA model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "pulsar: CUDA model staging record failed for %s: %s\n",
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
        fprintf(stderr, "pulsar: CUDA model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "pulsar: CUDA fd-cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}



static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("PULSAR_CUDA_NO_MODEL_COPY") != NULL ||
        getenv("PULSAR_CUDA_DIRECT_MODEL") != NULL ||
        getenv("PULSAR_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("PULSAR_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }
    if (g_model_device_owned || g_model_registered) return 1;

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "pulsar: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, "pulsar: CUDA chunk-copying %.2f GiB model image\n",
            (double)model_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    void *stage = NULL;
    err = cudaMallocHost(&stage, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "pulsar: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }

    if (map_offset > 0) {
        uint64_t copied_header = 0;
        while (copied_header < map_offset) {
            const uint64_t n = (map_offset - copied_header < chunk) ? (map_offset - copied_header) : chunk;
            memcpy(stage, (const char *)model_map + copied_header, (size_t)n);
            err = cudaMemcpy((char *)dev + copied_header, stage, (size_t)n, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "pulsar: CUDA model header copy failed: %s\n", cudaGetErrorString(err));
                (void)cudaFreeHost(stage);
                (void)cudaFree(dev);
                (void)cudaGetLastError();
                return 0;
            }
            copied_header += n;
        }
    }

    uint64_t copied = 0;
    double last_report = t0;
    while (copied < map_size) {
        const uint64_t n = (map_size - copied < chunk) ? (map_size - copied) : chunk;
        const uint64_t off = map_offset + copied;
        memcpy(stage, (const char *)model_map + off, (size_t)n);
        err = cudaMemcpy((char *)dev + off, stage, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "pulsar: CUDA model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFreeHost(stage);
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_discard_source_pages(model_map, model_size, off, n);
        copied += n;
        const double now = cuda_wall_sec();
        if (getenv("PULSAR_CUDA_MODEL_COPY_VERBOSE") != NULL && now - last_report >= 2.0) {
            fprintf(stderr, "pulsar: CUDA model chunk copy %.2f/%.2f GiB\n",
                    (double)copied / 1073741824.0,
                    (double)map_size / 1073741824.0);
            last_report = now;
        }
    }

    (void)cudaFreeHost(stage);
    g_model_device_base = (const char *)dev;
    g_model_device_owned = 1;
    g_model_hmm_direct = 0;
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "pulsar: CUDA model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
            t1 - t0,
            (double)map_size / 1073741824.0);
    return 1;
}



static void cuda_model_range_release_all(void) {
    cuda_model_load_progress_finish();
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
}






int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "pulsar: cuBLAS %s failed: status %d\n", what, (int)st);
    return 0;
}



/* Wrong-arch build trap (registration block at the bottom of pulsar_gpu.h):
 * every nvcc-compiled TU must carry device code for this GPU's SM family
 * (same SM major — sm_120f serves every sm_12x device, so the family match
 * cannot false-positive on the correct GB10 build).  Walk the registered TUs
 * BEFORE any kernel can launch and fail startup with the rebuild command
 * instead of dying mid-run in an opaque device assert. */
static int cuda_tu_archs_ok(const cudaDeviceProp *prop) {
#ifdef __CUDA_ARCH_LIST__
    for (const pulsar_tu_archs *r = pulsar_tu_archs_head; r; r = r->next) {
        int ok = 0;
        for (int i = 0; i < r->n_archs; i++)
            if (r->archs[i] / 100 == prop->major) ok = 1;
        if (!ok) {
            fprintf(stderr,
                    "pulsar: wrong-arch build: %s was compiled for sm_%d but this GPU "
                    "(%s) is sm_%d%d — rebuild with `make cuda-spark` (a plain "
                    "`make` drops CUDA_ARCH=sm_120f)\n",
                    r->file, r->n_archs > 0 ? r->archs[0] / 10 : 0,
                    prop->name, prop->major, prop->minor);
            return 0;
        }
    }
#else
    (void)prop;
#endif
    return 1;
}



int pulsar_gpu_init(void) {
    int dev = 0;
    if (!cuda_ok(cudaSetDevice(dev), "set device")) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        fprintf(stderr, "pulsar: CUDA backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
        if (!cuda_tu_archs_ok(&prop)) return 0;
    }
    if (!g_cublas_ready) {
        if (!cublas_ok(cublasCreate(&g_cublas), "create handle")) return 0;
        /* The whole build uses the per-thread default stream (Makefile
         * --default-stream per-thread) so the decode tape is capturable into
         * a CUDA graph; cuBLAS must launch onto the same stream. */
        (void)cublasSetStream(g_cublas, cudaStreamPerThread);
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("PULSAR_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
        g_cublas_ready = 1;
    }
    return 1;
}



void pulsar_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    if (g_decode_graph_exec) {
        (void)cudaGraphExecDestroy(g_decode_graph_exec);
        g_decode_graph_exec = NULL;
    }
    if (g_cublas_ready) {
        (void)cublasDestroy(g_cublas);
        g_cublas_ready = 0;
        g_cublas = NULL;
    }
    /* Invalidate the fp8 weight-pointer cache BEFORE freeing the model arenas
     * it points into -- a later engine open in this process would otherwise be
     * served dangling MXFP8_LT pointers (the second engine's mmap typically
     * reuses the same base address, so the cache guard cannot catch it). */
    cuda_fp8_weight_cache_clear();
    cuda_model_range_release_all();
    cuda_model_load_progress_reset();
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
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
    }
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_model_registered = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_fd = -1;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
    g_model_mapping_failure_notice_printed = 0;
    if (g_model_prefetch_stream) {
        (void)cudaStreamDestroy(g_model_prefetch_stream);
        g_model_prefetch_stream = NULL;
    }
}



__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);



/* Running total of live pulsar_gpu_tensor_alloc/_managed bytes (views excluded:
 * they don't own memory).  This is the ground truth the server's per-session
 * memory ledger reconciles against (pulsar_session_create snapshots it around the
 * graph allocation), catching drift between the sizing estimate and what the
 * allocator really did.  Atomic because drafter conf/token staging tensors are
 * allocated per spec step; one relaxed add is noise next to the cudaMalloc it
 * accompanies. */
static uint64_t g_tensor_alloc_bytes;

uint64_t pulsar_gpu_tensor_alloc_bytes_current(void) {
    return __atomic_load_n(&g_tensor_alloc_bytes, __ATOMIC_RELAXED);
}

pulsar_gpu_tensor *pulsar_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    pulsar_gpu_tensor *t = (pulsar_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    __atomic_add_fetch(&g_tensor_alloc_bytes, bytes, __ATOMIC_RELAXED);
    return t;
}



pulsar_gpu_tensor *pulsar_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    pulsar_gpu_tensor *t = (pulsar_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    __atomic_add_fetch(&g_tensor_alloc_bytes, bytes, __ATOMIC_RELAXED);
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



int pulsar_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
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



pulsar_gpu_tensor *pulsar_gpu_tensor_view(const pulsar_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    pulsar_gpu_tensor *t = (pulsar_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    return t;
}



void pulsar_gpu_tensor_free(pulsar_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) {
        (void)cudaFree(tensor->ptr);
        __atomic_sub_fetch(&g_tensor_alloc_bytes, tensor->bytes, __ATOMIC_RELAXED);
    }
    free(tensor);
}



uint64_t pulsar_gpu_tensor_bytes(const pulsar_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}



void *pulsar_gpu_tensor_contents(pulsar_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    (void)cudaDeviceSynchronize();
    return tensor->ptr;
}



/* Raw device pointer WITHOUT a synchronize — for building device pointer tables
 * (Tier-2 per-bank comp/index base tables) at allocation time, where the caller
 * controls ordering. Do NOT use to read tensor contents on the host. */
void *pulsar_gpu_tensor_device_ptr(const pulsar_gpu_tensor *tensor) {
    return tensor ? tensor->ptr : NULL;
}



int pulsar_gpu_tensor_fill_f32(pulsar_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
    return cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
}



int pulsar_gpu_tensor_write(pulsar_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice), "tensor write");
}



int pulsar_gpu_tensor_read(const pulsar_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost), "tensor read");
}




int pulsar_gpu_tensor_copy(pulsar_gpu_tensor *dst, uint64_t dst_offset,
                                     const pulsar_gpu_tensor *src, uint64_t src_offset,
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


/* ---- Batched tensor copy: one kernel over a device-side descriptor table. ----
 * Built for the spec-frontier snapshot/restore paths, which copy ~126 small
 * per-layer state tensors per fused step; as individual cudaMemcpy calls those
 * were ~190 launches/step of pure launch overhead. Descriptors are prepared
 * once (the tensors are fixed allocations) and replayed with a single launch.
 * Copies are 16-byte vectorized; ds4 tensor allocations are 256B-aligned and
 * prepare rejects any byte count that is not a multiple of 16. */
typedef struct {
    void *dst;
    const void *src;
    uint64_t bytes;
} pulsar_copy_desc;

__global__ static void batched_copy_kernel(const pulsar_copy_desc *descs, uint32_t n_descs) {
    const uint32_t d = blockIdx.y;
    if (d >= n_descs) return;
    const pulsar_copy_desc dc = descs[d];
    const uint64_t n16 = dc.bytes >> 4;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n16;
         i += (uint64_t)gridDim.x * blockDim.x) {
        ((float4 *)dc.dst)[i] = ((const float4 *)dc.src)[i];
    }
}

void *pulsar_gpu_batched_copy_prepare(
        pulsar_gpu_tensor **dst,
        pulsar_gpu_tensor **src,
        const uint64_t *bytes,
        uint32_t n) {
    if (!dst || !src || !bytes || n == 0) return NULL;
    pulsar_copy_desc *h = (pulsar_copy_desc *)malloc(n * sizeof(pulsar_copy_desc));
    if (!h) return NULL;
    for (uint32_t i = 0; i < n; i++) {
        if (!dst[i] || !src[i] || !dst[i]->ptr || !src[i]->ptr ||
            bytes[i] == 0 || (bytes[i] & 15u) ||
            bytes[i] > dst[i]->bytes || bytes[i] > src[i]->bytes) {
            free(h);
            return NULL;
        }
        h[i].dst = dst[i]->ptr;
        h[i].src = src[i]->ptr;
        h[i].bytes = bytes[i];
    }
    pulsar_copy_desc *d = NULL;
    if (cudaMalloc(&d, n * sizeof(pulsar_copy_desc)) != cudaSuccess) { free(h); return NULL; }
    if (!cuda_ok(cudaMemcpy(d, h, n * sizeof(pulsar_copy_desc), cudaMemcpyHostToDevice),
                 "batched copy descs")) {
        cudaFree(d);
        free(h);
        return NULL;
    }
    free(h);
    return d;
}

void pulsar_gpu_batched_copy_free(void *handle) {
    if (handle) cudaFree(handle);
}

int pulsar_gpu_batched_copy_run(void *handle, uint32_t n_descs, uint64_t max_bytes) {
    if (!handle || n_descs == 0) return 0;
    uint32_t chunks = (uint32_t)(((max_bytes >> 4) + 255) / 256);
    if (chunks < 1u) chunks = 1u;
    if (chunks > 64u) chunks = 64u;
    dim3 grid(chunks, n_descs);
    batched_copy_kernel<<<grid, 256>>>((const pulsar_copy_desc *)handle, n_descs);
    return cuda_ok(cudaGetLastError(), "batched copy launch");
}



int pulsar_gpu_begin_commands(void) { return 1; }


int pulsar_gpu_flush_commands(void) { return cuda_ok(cudaDeviceSynchronize(), "flush"); }


/* =========================================================================
 * Decode CUDA graph capture.
 *
 * The per-token decode tape re-encodes the same kernel sequence every step;
 * submitting it as ~500 individual launches pays a per-launch gap that adds
 * up to the measured ~3 ms/step floor. Capture the tape into a graph each
 * token (relaxed mode: the rare cuda_tmp_alloc growth and lazy weight-cache
 * mallocs execute immediately and are not part of the graph), then reuse one
 * instantiated exec via cudaGraphExecUpdate — instantiation only happens
 * again when the topology changes (context-tier boundaries). Opt-in via
 * PULSAR_CUDA_GRAPHS=1; see the note in pulsar_gpu_decode_graph_begin.
 * ========================================================================= */
int pulsar_gpu_decode_graph_begin(void) {
    /* Opt-in (PULSAR_CUDA_GRAPHS=1): capture-per-token measured ~2% SLOWER than
     * direct submission (ExecUpdate on a ~500-node graph outweighs the
     * launch-gap savings when the tape is re-encoded every token anyway).
     * The capture path is kept as working infrastructure for the follow-up
     * that reuses a stable tape with parameter-only updates. */
    if (g_decode_graphs_off < 0)
        g_decode_graphs_off = getenv("PULSAR_CUDA_GRAPHS") == NULL;
    if (g_decode_graphs_off) return 0;
    if (cudaStreamBeginCapture(cudaStreamPerThread,
                               cudaStreamCaptureModeRelaxed) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr,
                "pulsar: CUDA decode graph capture unavailable; using direct submission\n");
        g_decode_graphs_off = 1;
        return 0;
    }
    g_decode_graph_capturing = 1;
    return 1;
}

int pulsar_gpu_decode_graph_end(void) {
    if (!g_decode_graph_capturing) return 0;
    g_decode_graph_capturing = 0;
    cudaGraph_t graph = NULL;
    if (!cuda_ok(cudaStreamEndCapture(cudaStreamPerThread, &graph),
                 "decode graph capture")) {
        return 0;
    }
    int ok = 1;
    if (g_decode_graph_exec) {
        cudaGraphExecUpdateResultInfo info;
        if (cudaGraphExecUpdate(g_decode_graph_exec, graph, &info) == cudaSuccess) {
            g_decode_graph_updates++;
        } else {
            (void)cudaGetLastError();
            (void)cudaGraphExecDestroy(g_decode_graph_exec);
            g_decode_graph_exec = NULL;
        }
    }
    if (!g_decode_graph_exec) {
        ok = cuda_ok(cudaGraphInstantiate(&g_decode_graph_exec, graph, 0),
                     "decode graph instantiate");
        if (ok) g_decode_graph_instantiates++;
    }
    (void)cudaGraphDestroy(graph);
    if (ok) ok = cuda_ok(cudaGraphLaunch(g_decode_graph_exec, cudaStreamPerThread),
                         "decode graph launch");
    if (ok) ok = cuda_ok(cudaStreamSynchronize(cudaStreamPerThread),
                         "decode graph sync");
    if (ok && getenv("PULSAR_CUDA_GRAPH_STATS") != NULL &&
        (g_decode_graph_updates + g_decode_graph_instantiates) % 256 == 0) {
        fprintf(stderr, "pulsar: decode graphs: %llu updates, %llu instantiates\n",
                (unsigned long long)g_decode_graph_updates,
                (unsigned long long)g_decode_graph_instantiates);
    }
    return ok;
}

int pulsar_gpu_end_commands(void) {
    cuda_model_load_progress_finish();
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}


int pulsar_gpu_synchronize(void) {
    cuda_model_load_progress_finish();
    return cuda_ok(cudaDeviceSynchronize(), "synchronize");
}



static int cuda_model_set_host_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    const int same_backing_model =
        g_model_host_base == model_map &&
        g_model_registered_size == model_size;
    cuda_model_range_release_all();
    if (!same_backing_model) {
        cuda_model_load_progress_reset();
    }
    if (!same_backing_model) {
        if (g_model_device_owned && g_model_device_base) {
            (void)cudaFree((void *)g_model_device_base);
            g_model_device_owned = 0;
        }
        if (g_model_registered && g_model_host_base) {
            (void)cudaHostUnregister((void *)g_model_host_base);
            g_model_registered = 0;
        }
        g_model_host_base = model_map;
        g_model_device_base = (const char *)model_map;
        g_model_registered_size = model_size;
    } else if (!g_model_device_owned && !g_model_registered) {
        g_model_device_base = (const char *)model_map;
    }
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_cache_full = 0;
    g_model_mapping_failure_notice_printed = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }
    return 1;
}



int pulsar_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!cuda_model_set_host_map(model_map, model_size)) return 0;

    const char *copy_env = getenv("PULSAR_CUDA_COPY_MODEL");
    if (copy_env && copy_env[0]) {
        void *dev = NULL;
        const double t0 = clock() / (double)CLOCKS_PER_SEC;
        cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
        if (err == cudaSuccess) {
            fprintf(stderr, "pulsar: CUDA copying %.2f GiB model to device memory\n",
                    (double)model_size / 1073741824.0);
            err = cudaMemcpy(dev, model_map, (size_t)model_size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                g_model_device_base = (const char *)dev;
                g_model_device_owned = 1;
                const double t1 = clock() / (double)CLOCKS_PER_SEC;
                fprintf(stderr, "pulsar: CUDA model copy complete in %.3fs\n", t1 - t0);
                return 1;
            }
            fprintf(stderr, "pulsar: CUDA model copy failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
        } else {
            fprintf(stderr, "pulsar: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    }

    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    if (getenv("PULSAR_CUDA_HOST_REGISTER_PLAIN") != NULL) {
        flags = cudaHostRegisterMapped;
    }
    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size,
                                       flags);
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_device_base = (const char *)dev;
            g_model_registered = 1;
            fprintf(stderr, "pulsar: CUDA registered %.2f GiB model mapping for device access\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr, "pulsar: CUDA host registration pointer lookup failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr, "pulsar: CUDA host registration skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        const uint64_t limit = cuda_model_local_model_limit_bytes();
        if (!cuda_model_cache_limit_explicit() && model_size > limit) {
            fprintf(stderr,
                    "pulsar: CUDA model %.2f GiB exceeds the default single-GPU "
                    "startup cache budget %.2f GiB; set "
                    "PULSAR_CUDA_WEIGHT_CACHE_LIMIT_GB explicitly if the model "
                    "plus scratch and KV still fit in memory\n",
                    (double)model_size / 1073741824.0,
                    (double)limit / 1073741824.0);
            return 0;
        }
    }
    return 1;
}



int pulsar_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!pulsar_gpu_set_model_map(model_map, model_size)) return 0;
    if (getenv("PULSAR_CUDA_COPY_MODEL_CHUNKED") != NULL &&
        !cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
        (void)cuda_model_prefetch_range(model_map, model_size, map_offset, map_size);
    }
    return 1;
}




int pulsar_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    g_model_fd = fd;
    g_model_fd_host_base = model_map;
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
        if (getenv("PULSAR_CUDA_NO_DIRECT_IO") == NULL) {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
                if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "pulsar: CUDA model direct I/O enabled (align=%llu)\n",
                            (unsigned long long)g_model_direct_align);
                }
            } else if (getenv("PULSAR_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "pulsar: CUDA model direct I/O unavailable: %s\n", strerror(errno));
            }
        }
#endif
    }
    return 1;
}



int pulsar_gpu_set_model_fd(int fd) {
    return pulsar_gpu_set_model_fd_for_map(fd, g_model_host_base);
}



int pulsar_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    if (cuda_model_range_is_cached(model_map, offset, bytes)) return 1;

    const char *ptr = cuda_model_range_ptr(model_map, offset, bytes, label ? label : "model_tensor");
    if (!ptr || !cuda_model_range_is_cached(model_map, offset, bytes)) {
        if (!g_model_mapping_failure_notice_printed) {
            fprintf(stderr,
                    "pulsar: CUDA failed to prepare model tensor spans for device access\n");
            g_model_mapping_failure_notice_printed = 1;
        }
        return 0;
    }
    return 1;
}



/* Device-cache a tensor span from an ARBITRARY GGUF file (--expert-overlay
 * donor). The main weight-cache paths key on the base model's fd/map, so the
 * overlay's spans are uploaded here explicitly: pread from the overlay's own
 * fd through a pinned staging buffer, then registered in g_model_ranges under
 * the overlay's host map so cuda_model_range_ptr resolves them like any other
 * cached range. Runs at startup only (synchronous). */
int pulsar_gpu_cache_external_range(const void *host_base_key, int fd,
                                            uint64_t offset, uint64_t bytes,
                                            const char *label) {
    if (!host_base_key || fd < 0 || bytes == 0) return 0;

    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "pulsar: CUDA overlay range alloc failed for %s (%.2f MiB): %s\n",
                label ? label : "overlay", (double)bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    void *stage = NULL;
    err = cudaMallocHost(&stage, (size_t)(bytes < chunk ? bytes : chunk));
    if (err != cudaSuccess) {
        fprintf(stderr, "pulsar: CUDA overlay staging alloc failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }

    for (uint64_t done = 0; done < bytes;) {
        const uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        uint64_t got = 0;
        while (got < n) {
            const ssize_t r = pread(fd, (char *)stage + got, (size_t)(n - got),
                                    (off_t)(offset + done + got));
            if (r <= 0) {
                fprintf(stderr, "pulsar: CUDA overlay read failed for %s at %.2f MiB: %s\n",
                        label ? label : "overlay", (double)(done + got) / 1048576.0,
                        r == 0 ? "unexpected EOF" : strerror(errno));
                (void)cudaFreeHost(stage);
                (void)cudaFree(dev);
                return 0;
            }
            got += (uint64_t)r;
        }
        err = cudaMemcpy((char *)dev + done, stage, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "pulsar: CUDA overlay upload failed for %s at %.2f MiB: %s\n",
                    label ? label : "overlay", (double)done / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFreeHost(stage);
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        done += n;
    }
    (void)cudaFreeHost(stage);

    g_model_ranges.push_back({host_base_key, offset, bytes, (char *)dev,
                              NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    return 1;
}



void pulsar_gpu_mem_info(uint64_t *free_out, uint64_t *total_out) {
    size_t free_b = 0, total_b = 0;
    (void)cudaMemGetInfo(&free_b, &total_b);
    if (free_out) *free_out = (uint64_t)free_b;
    if (total_out) *total_out = (uint64_t)total_b;
}



void pulsar_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    (void)cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr, "pulsar: CUDA memory report %s: free %.2f MiB total %.2f MiB\n",
            label ? label : "", (double)free_b / 1048576.0, (double)total_b / 1048576.0);
}



void pulsar_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
    if (g_cublas_ready) {
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("PULSAR_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
    }
}






__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

