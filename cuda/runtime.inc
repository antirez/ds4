/* =========================================================================
 * Multi-GPU plumbing (device-aware CUDA).
 * ========================================================================= */

static_assert(DS4_MAX_GPUS == 16, "DS4_MAX_GPUS stack tables sized for 16");

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];
int         g_n_gpus = 0;
int         g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS];

/* Per-pair pinned-host bounce buffers, indexed [src][dst]. Lazily grown
 * to the largest copy seen for that pair. Each pair is its own allocation
 * so concurrent fan-out copies from a single source GPU to multiple
 * destinations cannot race for staging memory. */
static void   *g_xdev_bounce[DS4_MAX_GPUS][DS4_MAX_GPUS];
static size_t  g_xdev_bounce_bytes[DS4_MAX_GPUS][DS4_MAX_GPUS];

/* Internal helper: resolve a tensor's device index. -1 (untagged) is
 * treated as device 0 for legacy callers. */
static inline int ds4_tensor_device_idx(const ds4_gpu_tensor *t) {
    if (!t) return 0;
    int d = t->device_id;
    if (d < 0) return 0;
    return d;
}

/* Debug/override flags are read once per CUDA init. The hot decode path calls
 * the xdev helpers many times per token, so they must not re-enter getenv(). */
static void cuda_xdev_env_refresh(void) {
    g_xdev_sync_debug = getenv("DS4_CUDA_SYNC_XDEV") != NULL;
    g_xdev_force_cuda_peer = getenv("DS4_FORCE_CUDA_PEER") != NULL;
    g_xdev_force_host_bounce = getenv("DS4_FORCE_HOST_BOUNCE") != NULL;
}

static void cuda_decode_dispatch_env_refresh(void) {
    g_cuda_disable_qkv_rms_fused = getenv("DS4_CUDA_DISABLE_QKV_RMS_FUSED") != NULL;
    g_cuda_no_window_attention = getenv("DS4_CUDA_NO_WINDOW_ATTENTION") != NULL;
    g_cuda_decode_heads8_online = getenv("DS4_CUDA_DECODE_HEADS8_ONLINE") != NULL;
    g_cuda_decode_score4 = getenv("DS4_CUDA_DECODE_SCORE4") != NULL;
    g_cuda_decode_score8 = getenv("DS4_CUDA_DECODE_SCORE8") != NULL;
    g_cuda_no_decode_value512 = getenv("DS4_CUDA_NO_DECODE_VALUE512") != NULL;
    g_cuda_no_top1 = getenv("DS4_CUDA_NO_TOP1") != NULL;
    g_cuda_end_stream_sync = getenv("DS4_CUDA_END_STREAM_SYNC") != NULL;
    g_cuda_no_setdevice_cache = getenv("DS4_CUDA_NO_SETDEVICE_CACHE") != NULL;
    g_cuda_exact_score_split_graph =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_GRAPH") != NULL;
    g_cuda_exact_score_split_ldg =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_LDG") != NULL;
    g_cuda_exact_score_split_vec4 =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_VEC4") != NULL;
    g_cuda_exact_score_split_vec4_plain =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_VEC4_PLAIN") != NULL;
    g_cuda_exact_score_split_dim2 =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_DIM2") != NULL &&
        getenv("DS4_CUDA_NO_EXACT_SCORE_SPLIT_DIM2") == NULL;
    g_cuda_exact_score_split_fuse_inv_rope =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_FUSE_INV_ROPE") != NULL;
    g_cuda_moe_decode_graph = getenv("DS4_CUDA_MOE_DECODE_GRAPH") != NULL;
}

/* WITH_DEVICE(d) { ... } scope macro.
 *
 * Save the calling thread's current CUDA device, switch to device `d`,
 * run the body exactly once, then restore the previous device. If the
 * required CUDA calls fail, the body still runs (we don't have a clean
 * way to early-exit a containing function from a macro), but the next
 * CUDA call inside the body will surface the error naturally.
 *
 * Implementation: a for-loop with two synthetic variables. Iter 0 runs
 * the body; on iter 1, the iteration step restores the previous device
 * via cudaSetDevice and sets _wd_first = 0 so the loop exits. The
 * single-statement-body restriction of for-loops is removed by the
 * required `{ ... }` block in the call site.
 */
#define WITH_DEVICE(d)                                                      \
    for (int _wd_prev = -1, _wd_first = 1;                                  \
         _wd_first;                                                          \
         _wd_first = 0,                                                      \
         (_wd_prev >= 0 ? (void)cudaSetDevice(_wd_prev) : (void)0))         \
        if (cudaGetDevice(&_wd_prev) != cudaSuccess) { /* leave */ } else   \
        if (cudaSetDevice(d)           != cudaSuccess) { /* leave */ } else

/* =========================================================================
 * Per-device selective model cache (selective model cache).
 *
 * The public API in ds4_gpu.h declares ds4_tensor_range and the
 * device_cache_tensors / lookup_cache entry points. ds4_cuda.cu does NOT
 * include ds4_gpu.h historically (a pre-existing project convention), so
 * we redeclare the struct here with the same layout the header uses.
 * The implementation links by C linkage; struct compatibility is by
 * field layout. */
typedef struct {
    uint64_t source_offset;
    uint64_t bytes;
    int      target_device;
} ds4_tensor_range;

struct cuda_device_cache {
    void   *base;     /* device-side slab base */
    size_t  bytes;
    int     present;
};
static cuda_device_cache g_dev_cache[DS4_MAX_GPUS];

struct cache_range_entry {
    uint64_t source_offset;
    uint64_t bytes;
    int      device_id;
    void    *device_ptr;
};
static std::vector<cache_range_entry> g_cache_ranges;

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

struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
    int device_id;          /* physical CUDA device id; 0 in single-tier */
};

struct cuda_q8_f32_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    float *device_ptr;
    int device_id;          /* physical CUDA device id; 0 in single-tier */
};

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f32_by_offset;
static uint64_t g_model_range_bytes;
static uint64_t g_q8_f16_bytes;
static uint64_t g_q8_f32_bytes;
static int g_q8_cache_suppressed;
static int g_q8_f16_disabled_after_oom;
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
static void *g_stream_selected_stage_raw[4];
static void *g_stream_selected_stage[4];
static cudaEvent_t g_stream_selected_stage_event[4];
static uint64_t g_stream_selected_stage_bytes;
static cudaStream_t g_stream_selected_upload_stream;

static int cuda_ok(cudaError_t err, const char *what);
static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);

/* Forward declaration: defined later in this file. The resolver wrapper
 * below uses it for multi-tier dispatch. */
extern "C" int ds4_gpu_lookup_cache_strict(uint64_t source_offset,
                                            uint64_t bytes,
                                            int      expected_device,
                                            void   **out_device_ptr);
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
        fprintf(stderr, "ds4: CUDA temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_cuda_tmp = ptr;
    g_cuda_tmp_bytes = bytes;
    return g_cuda_tmp;
}

/* Per-tier scratch accessor.
 *
 * Behavior:
 *   - At g_n_gpus <= 1 (single-tier), delegates to cuda_tmp_alloc which
 *     manages the legacy g_cuda_tmp slab. This guarantees byte-identical
 *     behavior to pre-task code for the gpu_cfg == NULL case.
 *   - For multi-tier, grows the per-device g_gpu[logical_tier].scratch
 *     slab on the corresponding physical device. Cleanup is already
 *     handled by ds4_gpu_cleanup (which walks g_gpu[i].scratch).
 *
 * The legacy g_cuda_tmp slab is untouched: init-time / preload callers
 * still use cuda_tmp_alloc directly. No aliasing between g_cuda_tmp
 * and g_gpu[0].scratch — they are independently owned and freed.
 *
 * Added for multi-GPU execution (multi-GPU execution), step A3 of the
 * spec (sub-area 2). */
static void *cuda_tmp_alloc_on(int logical_tier, uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_n_gpus <= 1) {
        return cuda_tmp_alloc(bytes, what);
    }
    if (logical_tier < 0 || logical_tier >= g_n_gpus) {
        fprintf(stderr, "ds4: cuda_tmp_alloc_on: bad tier %d (n_gpus=%d, what=%s)\n",
                logical_tier, g_n_gpus, what ? what : "?");
        return NULL;
    }
    ds4_gpu_ctx *ctx = &g_gpu[logical_tier];
    if (ctx->scratch_bytes >= bytes) return ctx->scratch;
    int prev = -1;
    cudaError_t derr = cudaGetDevice(&prev);
    if (derr != cudaSuccess) {
        fprintf(stderr,
                "ds4: cudaGetDevice failed before scratch alloc on tier %d (dev=%d, what=%s): %s\n",
                logical_tier, ctx->device_id, what ? what : "scratch",
                cudaGetErrorString(derr));
        (void)cudaGetLastError();
        return NULL;
    }
    derr = cudaSetDevice(ctx->device_id);
    if (derr != cudaSuccess) {
        fprintf(stderr,
                "ds4: cudaSetDevice(%d) failed before scratch alloc on tier %d (what=%s): %s\n",
                ctx->device_id, logical_tier, what ? what : "scratch",
                cudaGetErrorString(derr));
        (void)cudaGetLastError();
        if (prev >= 0) (void)cudaSetDevice(prev);
        return NULL;
    }
    if (ctx->scratch) {
        (void)cudaFree(ctx->scratch);
        ctx->scratch = NULL;
        ctx->scratch_bytes = 0;
    }
    void *p = NULL;
    cudaError_t err = cudaMalloc(&p, (size_t)bytes);
    if (prev >= 0) (void)cudaSetDevice(prev);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA scratch alloc on tier %d (dev=%d) failed for %s (%.2f MiB): %s\n",
                logical_tier, ctx->device_id, what ? what : "scratch",
                (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    ctx->scratch = p;
    ctx->scratch_bytes = (size_t)bytes;
    return p;
}

static int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP;
}

static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

static const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    if (bytes == 0) return cuda_model_ptr(model_map, offset);
    if (g_model_device_owned || g_model_registered) return cuda_model_ptr(model_map, offset);
    if (g_model_hmm_direct &&
        getenv("DS4_CUDA_WEIGHT_CACHE") == NULL &&
        getenv("DS4_CUDA_WEIGHT_PRELOAD") == NULL) {
        return cuda_model_ptr(model_map, offset);
    }
    const char *direct_env = getenv("DS4_CUDA_DIRECT_MODEL");
    if (direct_env && direct_env[0]) return cuda_model_ptr(model_map, offset);

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

    if (getenv("DS4_CUDA_NO_FD_CACHE") == NULL) {
        const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
        if (fd_ptr) return fd_ptr;
    }

    cudaError_t err = cudaSuccess;
    if (g_model_range_mapping_supported) {
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
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA mapped %s %.2f MiB\n",
                            what ? what : "weights",
                            (double)bytes / 1048576.0);
                }
                return dev_ptr;
            }
            fprintf(stderr, "ds4: CUDA model range map pointer failed for %s: %s\n",
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
        fprintf(stderr, "ds4: CUDA model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f/%.2f MiB: %s\n",
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
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

/* Per-tier cuBLAS handle. Used by kernel-dispatch wrappers; returns the
 * cuBLAS handle for the logical tier. The wrapper is expected to have
 * cudaSetDevice'd to that tier's physical device already (kernels and
 * cuBLAS calls ride the default stream and are naturally serialized).
 *
 * Added for multi-GPU execution (multi-GPU execution), sub-area 1. */
static inline cublasHandle_t cuda_cublas_for_tier(int logical_tier) {
    if (g_n_gpus <= 1) {
        return (cublasHandle_t)g_gpu[0].cublas;
    }
    /* The executing device is authoritative: GLM per-layer switching runs
     * generic launchers whose out tensors live on device 0 while the layer
     * executes elsewhere. On DS4 paths the current device always equals the
     * requested tier, so this is behavior-preserving there. */
    int cur_dev = -1;
    if (cudaGetDevice(&cur_dev) == cudaSuccess) {
        for (int t = 0; t < g_n_gpus; t++) {
            if (g_gpu[t].device_id == cur_dev) {
                return (cublasHandle_t)g_gpu[t].cublas;
            }
        }
    }
    if (logical_tier < 0 || logical_tier >= g_n_gpus) {
        return (cublasHandle_t)g_gpu[0].cublas;
    }
    return (cublasHandle_t)g_gpu[logical_tier].cublas;
}

/* Multi-tier-aware weight pointer resolver.
 *
 * Used by kernel-dispatch wrappers in the per-layer execution path to
 * obtain a device pointer for a weight slice on the layer's logical
 * tier. Behavior:
 *
 *   - When g_n_gpus <= 1 (single-tier engine), delegates to the existing
 *     cuda_model_range_ptr path. This short-circuit guarantees byte-
 *     identical behavior to pre-multi-tier code for the gpu_cfg == NULL
 *     case.
 *
 *   - When g_n_gpus >= 2 (multi-tier engine), translates the logical
 *     tier index to the corresponding physical CUDA device id via
 *     g_gpu[logical_tier].device_id and looks up the slice strictly
 *     in the per-device selective cache via ds4_gpu_lookup_cache_strict.
 *     On miss, logs a diagnostic and returns NULL (no host-pointer
 *     fallback — a miss here is a placement/install bug).
 *
 * The caller is responsible for cudaSetDevice'ing to the right physical
 * device before launching the kernel that consumes the returned pointer.
 * Wrappers in this file thread `int logical_tier` from the dispatch
 * caller; single-tier callers pass 0, which hits the short-circuit.
 *
 * Added for multi-GPU execution (multi-GPU execution), sub-area 3 of the
 * spec. */
/* Optional second (support) model map for speculative decoding. The strict
 * multi-tier cache is keyed by source offset only, so support tensors are
 * installed and resolved with a large disjoint offset bias. */
static const void *g_support_host_base = NULL;
static uint64_t g_support_host_size = 0;
static uint64_t g_support_offset_bias = 0;

extern "C" uint64_t ds4_gpu_tier_free_vram(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return 0;
    int prev = -1;
    if (cudaGetDevice(&prev) != cudaSuccess) prev = -1;
    if (cudaSetDevice(g_gpu[logical_tier].device_id) != cudaSuccess) return 0;
    size_t free_b = 0, total_b = 0;
    uint64_t out = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) out = (uint64_t)free_b;
    if (prev >= 0) (void)cudaSetDevice(prev);
    return out;
}

extern "C" int ds4_gpu_register_support_map(const void *map, uint64_t size, uint64_t bias) {
    if (!map || size == 0 || bias == 0) return 0;
    g_support_host_base = map;
    g_support_host_size = size;
    g_support_offset_bias = bias;
    return 1;
}

static const char *cuda_resolve_weight_ptr(const void *model_map,
                                            uint64_t offset,
                                            uint64_t bytes,
                                            int logical_tier,
                                            const char *label) {
    if (g_n_gpus <= 1) {
        return cuda_model_range_ptr(model_map, offset, bytes, label);
    }
    if (g_support_host_base && model_map == g_support_host_base) {
        offset += g_support_offset_bias;
    }
    if (logical_tier < 0 || logical_tier >= g_n_gpus) {
        fprintf(stderr,
            "ds4: cuda_resolve_weight_ptr: bad tier %d (n_gpus=%d, label=%s)\n",
            logical_tier, g_n_gpus, label ? label : "?");
        return NULL;
    }
    const int physical_device = g_gpu[logical_tier].device_id;
    void *dev_ptr = NULL;
    if (ds4_gpu_lookup_cache_strict(offset, bytes, physical_device, &dev_ptr)
        && dev_ptr) {
        return (const char *)dev_ptr;
    }
    /* GLM multi-tier: generic launchers resolve by the OUT tensor's tier,
     * but the executing device (set per layer) is where the weights were
     * cached. Retry with the current device before declaring a miss;
     * DS4 paths never reach this (out tier == current device). */
    int cur_dev = -1;
    if (cudaGetDevice(&cur_dev) == cudaSuccess &&
        cur_dev != physical_device &&
        ds4_gpu_lookup_cache_strict(offset, bytes, cur_dev, &dev_ptr) &&
        dev_ptr) {
        return (const char *)dev_ptr;
    }
    fprintf(stderr,
        "ds4: selective-cache miss for offset=%llu bytes=%llu on "
        "logical_tier=%d (physical_device=%d, current_device=%d, "
        "label=%s); this is a placement/cache-install bug\n",
        (unsigned long long)offset, (unsigned long long)bytes,
        logical_tier, physical_device, cur_dev, label ? label : "?");
    return NULL;
}

static int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;
    if (g_model_device_owned || g_model_registered) return 1;

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
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

static uint64_t cuda_parse_mib_env(const char *name, int *present) {
    const char *env = getenv(name);
    if (present) *present = 0;
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    if (present) *present = 1;
    if (v > UINT64_MAX / 1048576ull) return UINT64_MAX;
    return (uint64_t)v * 1048576ull;
}

static uint32_t cuda_parse_u32_env_clamped(const char *name, uint32_t fallback,
                                           uint32_t min_value, uint32_t max_value,
                                           int *present) {
    const char *env = getenv(name);
    if (present) *present = 0;
    if (!env || !env[0]) return fallback;
    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0') return fallback;
    if (present) *present = 1;
    if (v < min_value) return min_value;
    if (v > max_value) return max_value;
    return (uint32_t)v;
}

static int cuda_env_flag_enabled(const char *name, int fallback) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    return strcmp(env, "0") != 0;
}

extern "C" int ds4_gpu_set_decode_fast_attention(int enabled) {
    const int old = g_decode_fast_attention;
    g_decode_fast_attention = enabled != 0;
    return old;
}

extern "C" int ds4_gpu_set_decode_score_vec4(int enabled) {
    const int old = g_decode_score_vec4;
    g_decode_score_vec4 = enabled != 0;
    return old;
}

static bool cuda_splitkv_decode_requested(void) {
    if (cuda_env_flag_enabled("DS4_CUDA_NO_SPLITKV_DECODE", 0)) return false;
    return g_decode_fast_attention ||
           cuda_env_flag_enabled("DS4_CUDA_SPLITKV_DECODE", 0);
}

static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    int present = 0;
    const uint64_t limit = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_MB", &present);
    return present ? limit : UINT64_MAX;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    int present = 0;
    const uint64_t reserve = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", &present);
    if (present) return reserve;

    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* High-VRAM cards (>= 40 GiB, e.g. 48 GiB RTX 6000 Ada): use a small
     * reserve so the selective Q8->F16 cache can actually engage at tight
     * budgets (e.g. --gpu-vram 47,47, where the 81 GB model leaves only ~1.3
     * GiB free and the old 4 GiB floor rejected every cache allocation,
     * forcing the scalar DP4A prefill kernel).
     *
     * NOTE: this 768 MiB value is a *bounded cache-growth guard*, not a hard
     * guarantee that live free VRAM stays >= 768 MiB.  cuda_q8_f16_cache_has_budget
     * only blocks a *cache* allocation when free - request < reserve at that
     * moment; allocations made outside cache accounting (cuda_tmp_alloc_on
     * activation/prequant buffers, cuBLAS internal workspaces) can still dip
     * below it.  768 MiB is chosen to leave headroom above the ~0.5 GiB
     * memory-safety floor for those out-of-cache allocations; actual minimum
     * free VRAM is verified by measurement, and the disable-after-failure path
     * degrades gracefully if cuBLAS/alloc ever fails under pressure.  Set
     * DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=4096 to restore the prior behavior. */
    if (total_bytes >= 40ull * 1024ull * 1024ull * 1024ull) {
        const uint64_t hi_min_reserve = 768ull * 1048576ull;
        const uint64_t hi_pct_reserve = total_bytes / 100u; /* 1% */
        return hi_pct_reserve > hi_min_reserve ? hi_pct_reserve : hi_min_reserve;
    }

    /* Smaller cards (< 40 GiB): keep the conservative reserve.  The expanded
     * Q8->F16 cache is only an acceleration path; on a small card a sub-GiB
     * reserve would be a large fraction of total VRAM, so keep enough free for
     * cuBLAS workspaces, transient graph buffers, and driver bookkeeping. */
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
    if (g_q8_f16_budget_notice_printed && getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") == NULL) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
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
        fprintf(stderr, "ds4: CUDA q8 fp16 cache memory query failed: %s; using q8 kernels\n",
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
                "ds4: CUDA q8 fp16 cache disabled after %s "
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
    if (g_q8_cache_suppressed) return 0;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (getenv("DS4_CUDA_NO_Q8_F16_CACHE") != NULL) return 0;
    if (cuda_q8_f16_cache_limit_bytes() == 0) return 0;
    if (getenv("DS4_CUDA_Q8_F16_ALL") != NULL) return 1;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return getenv("DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE") == NULL;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return getenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE") == NULL;
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
           (getenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE") == NULL &&
            in_dim == 1024u && out_dim == 32768u);
}

static int cuda_q8_label_is_attention_output(const char *label) {
    return label &&
           (strstr(label, "attn_output_a") != NULL ||
            strstr(label, "attn_output_b") != NULL ||
            strstr(label, "attention_output_a") != NULL ||
            strstr(label, "attention_output_b") != NULL);
}

static int cuda_q8_use_dp4a(void) {
    return getenv("DS4_CUDA_NO_Q8_DP4A") == NULL;
}

static unsigned cuda_q8_exact_threads(uint64_t blocks) {
    if (blocks <= 64u) return 64u;
    if (blocks <= 128u) return 128u;
    return 256u;
}

static int cuda_q8_f16_preload_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (cuda_q8_label_is_attention_output(label) &&
        getenv("DS4_CUDA_ATTENTION_OUTPUT_PRELOAD") == NULL &&
        getenv("DS4_CUDA_Q8_F16_ALL") == NULL) {
        return 0;
    }
    return cuda_q8_f16_cache_allowed(label, in_dim, out_dim);
}

static int cuda_q8_f32_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_q8_cache_suppressed) return 0;
    if (getenv("DS4_CUDA_NO_Q8_F32_CACHE") != NULL) return 0;
    if (getenv("DS4_CUDA_Q8_F32_ALL") != NULL) return 1;
    if (label && strstr(label, "attn_q_b") != NULL) {
        return getenv("DS4_CUDA_ATTN_Q_B_F32_CACHE") != NULL;
    }
    return getenv("DS4_CUDA_Q8_F32_LARGE") != NULL &&
           in_dim == 1024u && out_dim == 32768u;
}

/* Look up a per-device dequantized fp16 slice of the Q8_0 weight at
 * (model_map, offset, weight_bytes, in_dim, out_dim). expected_device is a
 * PHYSICAL CUDA device id (0 in single-tier; g_gpu[logical_tier].device_id in
 * multi-tier). On hit returns the cached pointer for that device. On miss
 * cudaSetDevice's to expected_device, allocates + dequants there, stamps the
 * new entry with device_id == expected_device, restores the previous device,
 * and returns the new pointer.
 *
 * Single-tier (g_n_gpus <= 1) uses the offset-keyed map for a fast path —
 * legacy entries were stamped device_id=0, so the map remains authoritative.
 * Multi-tier linear-scans the ranges vector filtering on device_id (the same
 * offset may now legitimately map to multiple entries, one per device).
 */
static const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        int expected_device,
        const char *label) {
    if (g_n_gpus <= 1) {
        auto exact = g_q8_f16_by_offset.find(offset);
        if (exact != g_q8_f16_by_offset.end()) {
            const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
            if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
                r.in_dim == in_dim && r.out_dim == out_dim) {
                return r.device_ptr;
            }
        }
    } else {
        for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
            if (r.host_base == model_map &&
                r.offset == offset &&
                r.weight_bytes == weight_bytes &&
                r.in_dim == in_dim &&
                r.out_dim == out_dim &&
                r.device_id == expected_device) {
                return r.device_ptr;
            }
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    /* Source Q8 bytes:
     *  - Single-tier (g_n_gpus <= 1): cuda_model_range_ptr — preserves the
     *    legacy behavior (FD cache, host-register, or cudaMalloc-and-copy).
     *  - Multi-tier (g_n_gpus > 1): the per-device selective cache must
     *    already contain the weight on expected_device. Use the strict
     *    lookup; on miss this is a placement bug and we hard-fail.
     */
    const char *q8;
    if (g_n_gpus <= 1) {
        q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    } else {
        void *strict_ptr = NULL;
        if (!ds4_gpu_lookup_cache_strict(offset, weight_bytes, expected_device, &strict_ptr) ||
            !strict_ptr) {
            fprintf(stderr,
                "ds4: q8 fp16 cache miss: source bytes not in selective cache for "
                "offset=%llu bytes=%llu device=%d (label=%s); placement bug\n",
                (unsigned long long)offset, (unsigned long long)weight_bytes,
                expected_device, label ? label : "?");
            return NULL;
        }
        q8 = (const char *)strict_ptr;
    }
    if (!q8) return NULL;

    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / sizeof(__half)) return NULL;
    const uint64_t out_bytes = in_dim * out_dim * sizeof(__half);
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;

    int prev = -1;
    if (g_n_gpus > 1) {
        cudaError_t derr = cudaGetDevice(&prev);
        if (derr != cudaSuccess) {
            fprintf(stderr, "ds4: cudaGetDevice failed before q8 fp16 alloc on device %d: %s\n",
                    expected_device, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            return NULL;
        }
        derr = cudaSetDevice(expected_device);
        if (derr != cudaSuccess) {
            fprintf(stderr, "ds4: cudaSetDevice(%d) failed before q8 fp16 alloc: %s\n",
                    expected_device, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            if (prev >= 0) (void)cudaSetDevice(prev);
            return NULL;
        }
    }
    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache alloc failed on device %d (%.2f MiB): %s\n",
                expected_device, (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
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
        if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev, expected_device});
    if (g_n_gpus <= 1) {
        g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    }
    g_q8_f16_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp16 %.2f MiB on device %d (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0, expected_device,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
    return dev;
}

/* Per-device dequantized fp32 cache. Same conventions as cuda_q8_f16_ptr. */
static float *cuda_q8_f32_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        int expected_device,
        const char *label) {
    if (g_n_gpus <= 1) {
        auto exact = g_q8_f32_by_offset.find(offset);
        if (exact != g_q8_f32_by_offset.end()) {
            const cuda_q8_f32_range &r = g_q8_f32_ranges[exact->second];
            if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
                r.in_dim == in_dim && r.out_dim == out_dim) {
                return r.device_ptr;
            }
        }
    } else {
        for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
            if (r.host_base == model_map &&
                r.offset == offset &&
                r.weight_bytes == weight_bytes &&
                r.in_dim == in_dim &&
                r.out_dim == out_dim &&
                r.device_id == expected_device) {
                return r.device_ptr;
            }
        }
    }
    if (!cuda_q8_f32_cache_allowed(label, in_dim, out_dim)) return NULL;

    /* Source Q8 bytes: legacy path in single-tier; strict per-device lookup
     * in multi-tier (same rationale as cuda_q8_f16_ptr). */
    const char *q8;
    if (g_n_gpus <= 1) {
        q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, label ? label : "q8_0");
    } else {
        void *strict_ptr = NULL;
        if (!ds4_gpu_lookup_cache_strict(offset, weight_bytes, expected_device, &strict_ptr) ||
            !strict_ptr) {
            fprintf(stderr,
                "ds4: q8 fp32 cache miss: source bytes not in selective cache for "
                "offset=%llu bytes=%llu device=%d (label=%s); placement bug\n",
                (unsigned long long)offset, (unsigned long long)weight_bytes,
                expected_device, label ? label : "?");
            return NULL;
        }
        q8 = (const char *)strict_ptr;
    }
    if (!q8) return NULL;

    const uint64_t out_bytes = in_dim * out_dim * sizeof(float);
    int prev = -1;
    if (g_n_gpus > 1) {
        cudaError_t derr = cudaGetDevice(&prev);
        if (derr != cudaSuccess) {
            fprintf(stderr, "ds4: cudaGetDevice failed before q8 fp32 alloc on device %d: %s\n",
                    expected_device, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            return NULL;
        }
        derr = cudaSetDevice(expected_device);
        if (derr != cudaSuccess) {
            fprintf(stderr, "ds4: cudaSetDevice(%d) failed before q8 fp32 alloc: %s\n",
                    expected_device, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            if (prev >= 0) (void)cudaSetDevice(prev);
            return NULL;
        }
    }
    float *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp32 cache alloc failed on device %d (%.2f MiB): %s\n",
                expected_device, (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f32_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp32 dequant launch")) {
        (void)cudaFree(dev);
        if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
        return NULL;
    }
    g_q8_f32_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev, expected_device});
    if (g_n_gpus <= 1) {
        g_q8_f32_by_offset[offset] = g_q8_f32_ranges.size() - 1u;
    }
    g_q8_f32_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp32 %.2f MiB on device %d (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0, expected_device,
                (double)g_q8_f32_bytes / 1073741824.0);
    }
    if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
    return dev;
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cuda_model_load_progress_enabled(void) {
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") != NULL) return 0;
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
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache: 0.00 GiB");
        } else {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache\n");
        }
    }

    if (cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (g_model_load_progress_tty ? 2.0 : 10.0)) {
        return;
    }

    if (g_model_load_progress_tty) {
        fprintf(stderr, "\rds4: CUDA loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        fprintf(stderr, "ds4: CUDA loading model tensors %.2f GiB cached\n",
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

static int cuda_model_prefetch_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("DS4_CUDA_NO_MODEL_PREFETCH") != NULL ||
        getenv("DS4_CUDA_COPY_MODEL") != NULL ||
        getenv("DS4_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("DS4_CUDA_WEIGHT_PRELOAD") != NULL) {
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
#if CUDART_VERSION >= 13000
    cudaMemLocation loc;
    memset(&loc, 0, sizeof(loc));
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;
#else
    int loc = device;
#endif

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
        fprintf(stderr, "ds4: CUDA model read-mostly advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetPreferredLocation, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model preferred-location advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    if (!g_model_prefetch_stream) {
        err = cudaStreamCreateWithFlags(&g_model_prefetch_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch stream creation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }

#if CUDART_VERSION >= 13000
    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, 0, g_model_prefetch_stream);
#else
    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, g_model_prefetch_stream);
#endif
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model prefetch skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    if (getenv("DS4_CUDA_MODEL_PREFETCH_SYNC") != NULL) {
        err = cudaStreamSynchronize(g_model_prefetch_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch sync failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA ATS/HMM prefetch queued %.2f GiB of model tensors in %.3fs\n",
            (double)map_size / 1073741824.0,
            t1 - t0);
    g_model_hmm_direct = 1;
    return 1;
}

static uint64_t cuda_model_copy_chunk_bytes(void) {
    uint64_t mb = 64;
    const char *env = getenv("DS4_CUDA_MODEL_COPY_CHUNK_MB");
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
    if (getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL || !model_map || bytes == 0 || offset > model_size) return;
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
    if (g_model_fd < 0 || getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL || bytes == 0) return;
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
            fprintf(stderr, "ds4: CUDA model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging event creation failed: %s\n", cudaGetErrorString(err));
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
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA direct model read disabled: %s\n", strerror(direct_errno));
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

static void cuda_stream_selected_stage_release(void) {
    for (size_t i = 0; i < 4; i++) {
        if (g_stream_selected_stage_event[i]) {
            (void)cudaEventDestroy(g_stream_selected_stage_event[i]);
            g_stream_selected_stage_event[i] = NULL;
        }
        if (g_stream_selected_stage_raw[i]) {
            (void)cudaFreeHost(g_stream_selected_stage_raw[i]);
            g_stream_selected_stage_raw[i] = NULL;
            g_stream_selected_stage[i] = NULL;
        }
    }
    g_stream_selected_stage_bytes = 0;
    if (g_stream_selected_upload_stream) {
        (void)cudaStreamDestroy(g_stream_selected_upload_stream);
        g_stream_selected_upload_stream = NULL;
    }
}

static int cuda_stream_selected_stage_pool_alloc(uint64_t bytes) {
    if (g_stream_selected_stage_bytes >= bytes) return 1;
    cuda_stream_selected_stage_release();
    cudaError_t err = cudaStreamCreateWithFlags(
            &g_stream_selected_upload_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected upload stream creation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    for (size_t i = 0; i < 4; i++) {
        err = cudaMallocHost(&g_stream_selected_stage_raw[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging allocation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_stream_selected_stage_release();
            return 0;
        }
        g_stream_selected_stage[i] = cuda_align_ptr(
                g_stream_selected_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_stream_selected_stage_event[i],
                                       cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging event creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_stream_selected_stage_release();
            return 0;
        }
    }
    g_stream_selected_stage_bytes = bytes;
    return 1;
}

static int cuda_model_copy_to_device_streamed(
        char *dst,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (!dst || !model_map || offset > model_size ||
        bytes > model_size - offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    if (g_model_fd < 0 ||
        (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base)) {
        return cuda_ok(cudaMemcpy(dst,
                                  (const char *)model_map + offset,
                                  (size_t)bytes,
                                  cudaMemcpyHostToDevice),
                       what ? what : "stream selected expert copy");
    }

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes =
        chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_stream_selected_stage_pool_alloc(stage_bytes)) return 0;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = bytes - copied < chunk ? bytes - copied : chunk;
        const uint64_t bi = chunk_idx % 4u;
        cudaError_t err;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_stream_selected_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "ds4: CUDA streaming selected staging wait failed for %s: %s\n",
                        what ? what : "expert", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return 0;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_stream_selected_stage[bi],
                                   g_stream_selected_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected read failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert", (double)copied / 1048576.0,
                    strerror(errno));
            return 0;
        }
        err = cudaMemcpyAsync(dst + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice,
                              g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert", (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        err = cudaEventRecord(g_stream_selected_stage_event[bi],
                              g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging record failed for %s: %s\n",
                    what ? what : "expert", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, model_size,
                                        offset + copied, n);
        copied += n;
        chunk_idx++;
    }

    const cudaError_t err =
        cudaStreamSynchronize(g_stream_selected_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected upload sync failed for %s: %s\n",
                what ? what : "expert", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static uint64_t cuda_model_cache_limit_bytes(void) {
    uint64_t gb = 0;
    const char *env = getenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) gb = (uint64_t)v;
    }
    if (gb == 0) return UINT64_MAX;
    return gb * 1073741824ull;
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t mb = 1792;
    const char *env = getenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 256) mb = 256;
    if (mb > 8192) mb = 8192;
    uint64_t bytes = mb * 1048576ull;
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
        fprintf(stderr, "ds4: CUDA model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        uint64_t arena_bytes = 0;
        for (const cuda_model_arena &a : g_model_arenas) arena_bytes += a.bytes;
        fprintf(stderr, "ds4: CUDA model arena allocated %.2f MiB (arenas %.2f GiB)\n",
                (double)chunk / 1048576.0,
                (double)arena_bytes / 1073741824.0);
    }
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
        if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "ds4: CUDA direct %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return cuda_model_ptr(model_map, offset);
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        if (getenv("DS4_CUDA_STRICT_WEIGHT_CACHE") != NULL) return NULL;
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
                fprintf(stderr, "ds4: CUDA model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, "ds4: CUDA model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging record failed for %s: %s\n",
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
        fprintf(stderr, "ds4: CUDA model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA fd-cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("DS4_CUDA_NO_MODEL_COPY") != NULL ||
        getenv("DS4_CUDA_DIRECT_MODEL") != NULL ||
        getenv("DS4_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("DS4_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }
    if (g_model_device_owned || g_model_registered) return 1;

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, "ds4: CUDA chunk-copying %.2f GiB model image\n",
            (double)model_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    void *stage = NULL;
    err = cudaMallocHost(&stage, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
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
                fprintf(stderr, "ds4: CUDA model header copy failed: %s\n", cudaGetErrorString(err));
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
            fprintf(stderr, "ds4: CUDA model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFreeHost(stage);
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_discard_source_pages(model_map, model_size, off, n);
        copied += n;
        const double now = cuda_wall_sec();
        if (getenv("DS4_CUDA_MODEL_COPY_VERBOSE") != NULL && now - last_report >= 2.0) {
            fprintf(stderr, "ds4: CUDA model chunk copy %.2f/%.2f GiB\n",
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
            "ds4: CUDA model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
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
    cuda_model_load_progress_reset();
}

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: cuBLAS %s failed: status %d\n", what, (int)st);
    return 0;
}

extern "C" int ds4_gpu_init_multi(const ds4_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus < 1 || cfg->n_gpus > DS4_MAX_GPUS) return 0;
    cuda_xdev_env_refresh();
    cuda_decode_dispatch_env_refresh();
    g_current_logical_tier = -1;

    /* g_n_gpus is published incrementally so ds4_gpu_cleanup() can unwind
     * partial state on failure. We publish `i + 1` BEFORE allocating any
     * resources for context `i`, so even if (e.g.) stream creation
     * succeeds but event creation fails, cleanup still walks device `i`
     * and destroys the stream. ds4_gpu_cleanup is null-safe per field —
     * partial state is OK. */
    for (int i = 0; i < cfg->n_gpus; i++) {
        ds4_gpu_ctx *c = &g_gpu[i];
        c->device_id = cfg->device_indices[i];
        if (c->device_id < 0) return 0;
        /* Publish the in-progress device id so cleanup can target it on
         * any later failure. cudaSetDevice is also required before
         * cleanup's cudaEventDestroy / cudaStreamDestroy / cublasDestroy
         * calls hit the right context. */
        g_n_gpus = i + 1;
        if (!cuda_ok(cudaSetDevice(c->device_id), "init set device")) return 0;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, c->device_id) == cudaSuccess) {
            fprintf(stderr, "ds4: CUDA backend initialized on %s (sm_%d%d) dev=%d\n",
                    prop.name, prop.major, prop.minor, c->device_id);
        }
        /* Per-device stream. */
        cudaStream_t s = NULL;
        if (!cuda_ok(cudaStreamCreate(&s), "init stream")) return 0;
        c->stream = (void *)s;
        /* Per-device boundary event (reusable, no timing). */
        cudaEvent_t ev = NULL;
        if (!cuda_ok(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming),
                      "init event")) return 0;
        c->boundary_event = (void *)ev;
        /* Per-device cuBLAS handle. */
        cublasHandle_t h = NULL;
        if (!cublas_ok(cublasCreate(&h), "init cublas")) return 0;
        c->cublas = (void *)h;
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(h, math_mode);
        c->cublas_ready = 1;
        c->budget_bytes = cfg->vram_bytes[i];
        c->used_bytes   = 0;
        c->scratch      = NULL;
        c->scratch_bytes = 0;
    }

    /* NxN peer-access matrix.
     *
     * Driver semantics: cudaDeviceCanAccessPeer + cudaDeviceEnablePeerAccess
     * can both succeed even on hardware/drivers where cudaMemcpyPeerAsync
     * silently delivers wrong data (notably RTX 6000 Ada under recent
     * NVIDIA drivers, per the v0 design doc). The corruption is
     * non-deterministic and can affect either or both directions of a
     * pair. To guard against this, we run a multi-size, multi-iteration
     * validation at init (see the loop below): write distinct known
     * patterns, peer-copy them to the destination, read back, and only
     * set peer_ok[i][j] if every probe round-trips byte-perfect. A
     * single small probe is not sufficient — it can pass while realistic
     * activation-sized copies still corrupt. On any failure the entry
     * stays at 0 and cross-device copies fall back to the pinned-host
     * bounce path automatically. */
    for (int i = 0; i < g_n_gpus; i++) {
        for (int j = 0; j < g_n_gpus; j++) {
            if (i == j) { g_gpu_peer_ok[i][j] = 1; continue; }
            int can = 0;
            (void)cudaDeviceCanAccessPeer(&can, g_gpu[i].device_id,
                                          g_gpu[j].device_id);
            if (!can) { g_gpu_peer_ok[i][j] = 0; continue; }
            (void)cudaSetDevice(g_gpu[i].device_id);
            cudaError_t e = cudaDeviceEnablePeerAccess(g_gpu[j].device_id, 0);
            int enabled = (e == cudaSuccess ||
                           e == cudaErrorPeerAccessAlreadyEnabled);
            (void)cudaGetLastError();
            if (!enabled) { g_gpu_peer_ok[i][j] = 0; continue; }

            /* Runtime validation: peer copies on RTX 6000 Ada under recent
             * NVIDIA drivers silently corrupt at realistic sizes even though
             * the API returns success. cudaDeviceCanAccessPeer and
             * cudaDeviceEnablePeerAccess can both report success while
             * cudaMemcpyPeer delivers wrong data non-deterministically.
             * Probe with multiple sizes and iterations; ALL must round-trip
             * byte-perfect or we disable peer for this pair and silently fall
             * back to the pinned-host bounce path. */
            static const size_t kValidateSizes[] = {
                4u * 1024u,
                256u * 1024u,
                1u * 1024u * 1024u,
                16u * 1024u * 1024u,
            };
            const int kValidateIters    = 4;
            const int kNValidateSizes   = (int)(sizeof(kValidateSizes) /
                                                sizeof(kValidateSizes[0]));
            const size_t kMaxValidate   = kValidateSizes[kNValidateSizes - 1];

            unsigned char *vh_src = (unsigned char *)malloc(kMaxValidate);
            unsigned char *vh_dst = (unsigned char *)malloc(kMaxValidate);
            if (!vh_src || !vh_dst) {
                free(vh_src); free(vh_dst);
                g_gpu_peer_ok[i][j] = 0; continue;
            }

            void *src_dev = NULL; void *dst_dev = NULL;
            (void)cudaSetDevice(g_gpu[i].device_id);
            if (cudaMalloc(&src_dev, kMaxValidate) != cudaSuccess) {
                (void)cudaGetLastError();
                free(vh_src); free(vh_dst);
                g_gpu_peer_ok[i][j] = 0; continue;
            }
            (void)cudaSetDevice(g_gpu[j].device_id);
            if (cudaMalloc(&dst_dev, kMaxValidate) != cudaSuccess) {
                (void)cudaGetLastError();
                (void)cudaSetDevice(g_gpu[i].device_id);
                (void)cudaFree(src_dev);
                free(vh_src); free(vh_dst);
                g_gpu_peer_ok[i][j] = 0; continue;
            }

            int    peer_validated = 1;
            size_t failed_bytes   = 0;
            int    failed_iter    = -1;
            for (int s_idx = 0;
                 s_idx < kNValidateSizes && peer_validated;
                 s_idx++) {
                size_t n = kValidateSizes[s_idx];
                for (int it = 0; it < kValidateIters && peer_validated; it++) {
                    for (size_t k = 0; k < n; k++) {
                        vh_src[k] = (unsigned char)
                            ((k * 31u + (size_t)it * 17u +
                              (size_t)s_idx * 53u + 11u) & 0xffu);
                    }
                    (void)cudaSetDevice(g_gpu[i].device_id);
                    if (cudaMemcpy(src_dev, vh_src, n,
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        peer_validated = 0; failed_bytes = n; failed_iter = it;
                        break;
                    }
                    cudaError_t pc = cudaMemcpyPeer(
                        dst_dev, g_gpu[j].device_id,
                        src_dev, g_gpu[i].device_id, n);
                    if (pc != cudaSuccess) {
                        peer_validated = 0; failed_bytes = n; failed_iter = it;
                        break;
                    }
                    (void)cudaSetDevice(g_gpu[j].device_id);
                    if (cudaMemcpy(vh_dst, dst_dev, n,
                                   cudaMemcpyDeviceToHost) != cudaSuccess) {
                        peer_validated = 0; failed_bytes = n; failed_iter = it;
                        break;
                    }
                    if (memcmp(vh_src, vh_dst, n) != 0) {
                        peer_validated = 0; failed_bytes = n; failed_iter = it;
                        break;
                    }
                }
            }

            (void)cudaSetDevice(g_gpu[j].device_id);
            (void)cudaFree(dst_dev);
            (void)cudaSetDevice(g_gpu[i].device_id);
            (void)cudaFree(src_dev);
            free(vh_src);
            free(vh_dst);

            g_gpu_peer_ok[i][j] = peer_validated;
            if (peer_validated) {
                fprintf(stderr,
                    "ds4: peer access %d->%d validated across %d sizes x %d"
                    " iterations (max %zu MiB)\n",
                    g_gpu[i].device_id, g_gpu[j].device_id,
                    kNValidateSizes, kValidateIters,
                    kMaxValidate / (1024u * 1024u));
            } else {
                fprintf(stderr,
                    "ds4: peer access %d->%d FAILED validation at"
                    " size=%zu iter=%d; falling back to pinned-host bounce\n",
                    g_gpu[i].device_id, g_gpu[j].device_id,
                    failed_bytes, failed_iter);
            }
        }
    }

    g_cublas_ready = 1;
    return 1;
}

extern "C" int ds4_gpu_init(void) {
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.device_indices[0] = 0;
    cfg.n_gpus = 1;
    return ds4_gpu_init_multi(&cfg);
}

extern "C" void ds4_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    g_current_logical_tier = -1;

    /* Multi-GPU teardown: events, streams, cublas handles, scratch
     * slabs, per-pair bounce buffers. */
    for (int i = 0; i < g_n_gpus; i++) {
        ds4_gpu_ctx *c = &g_gpu[i];
        (void)cudaSetDevice(c->device_id);
        attention_decode_score_split_graph_destroy_one(i);
        routed_moe_decode_graph_destroy_one(i);
        if (c->boundary_event) {
            (void)cudaEventDestroy((cudaEvent_t)c->boundary_event);
            c->boundary_event = NULL;
        }
        if (c->stream) {
            (void)cudaStreamDestroy((cudaStream_t)c->stream);
            c->stream = NULL;
        }
        if (c->cublas) {
            (void)cublasDestroy((cublasHandle_t)c->cublas);
            c->cublas = NULL;
            c->cublas_ready = 0;
        }
        if (c->scratch) {
            (void)cudaFree(c->scratch);
            c->scratch = NULL;
            c->scratch_bytes = 0;
        }
    }
    for (int i = 0; i < DS4_MAX_GPUS; i++) {
        for (int j = 0; j < DS4_MAX_GPUS; j++) {
            if (g_xdev_bounce[i][j]) {
                (void)cudaFreeHost(g_xdev_bounce[i][j]);
                g_xdev_bounce[i][j] = NULL;
                g_xdev_bounce_bytes[i][j] = 0;
            }
        }
    }
    cuda_stream_selected_cache_release();
    cuda_stream_selected_stage_release();
    g_n_gpus = 0;
    g_cublas_ready = 0;

    /* Per-device selective cache teardown (selective model cache). */
    for (int d = 0; d < DS4_MAX_GPUS; d++) {
        if (!g_dev_cache[d].present) continue;
        int prev = -1;
        (void)cudaGetDevice(&prev);
        (void)cudaSetDevice(d);
        if (g_dev_cache[d].base) (void)cudaFree(g_dev_cache[d].base);
        g_dev_cache[d].base = NULL;
        g_dev_cache[d].bytes = 0;
        g_dev_cache[d].present = 0;
        if (prev >= 0) (void)cudaSetDevice(prev);
    }
    g_cache_ranges.clear();

    /* Continue with legacy global teardown below. */

    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
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
    if (g_model_prefetch_stream) {
        (void)cudaStreamDestroy(g_model_prefetch_stream);
        g_model_prefetch_stream = NULL;
    }
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int device_id,
                                       uint64_t bytes) {
    if (!t) return 1;
    if (device_id < 0 || device_id >= g_n_gpus) return 2;
    if (bytes == 0) bytes = 1;
    int ok = 0;
    WITH_DEVICE(g_gpu[device_id].device_id) {
        ok = cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc");
    }
    if (!ok) return 3;
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = device_id;
    g_gpu[device_id].used_bytes += bytes;
    return 0;
}

/* Async D2D copy queued on the destination device's default stream —
 * ordering against the producer comes from the caller's fence; the CPU
 * does not block (unlike ds4_gpu_tensor_copy's sync cudaMemcpy). */
extern "C" int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *dst,
                                         const ds4_gpu_tensor *src,
                                         uint64_t bytes) {
    if (!dst || !src || bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0) return 1;
    return cuda_ok(cudaMemcpyAsync(dst->ptr, src->ptr, (size_t)bytes,
                                   cudaMemcpyDeviceToDevice, 0),
                   "tensor copy async");
}

extern "C" void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t) {
    if (!t) return;
    int d = ds4_tensor_device_idx(t);
    if (t->owner && t->ptr) {
        WITH_DEVICE(g_gpu[d].device_id) {
            (void)cudaFree(t->ptr);
        }
    }
    t->ptr = NULL;
    t->bytes = 0;
    t->owner = 0;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (ds4_gpu_tensor_alloc_on(t, 0, bytes) != 0) {
        free(t);
        return NULL;
    }
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    int ok = 0;
    /* Managed memory is not device-bound, but we record device 0 so that
     * subsequent ds4_gpu_tensor_free pairs with WITH_DEVICE(0) safely. */
    WITH_DEVICE(g_gpu[0].device_id) {
        ok = cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes),
                     "managed tensor alloc");
    }
    if (!ok) { free(t); return NULL; }
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = 0;
    return t;
}

/* Heap-allocated tensor on a specific logical tier.
 *
 * Mirrors the legacy ds4_gpu_tensor_alloc ABI (returns ds4_gpu_tensor *)
 * with an explicit tier argument. Internally calls
 * ds4_gpu_tensor_alloc_on on a freshly malloc'd struct.
 *
 * The legacy ds4_gpu_tensor_alloc(bytes) (above) calls
 * ds4_gpu_tensor_alloc_on(t, 0, bytes); ds4_gpu_tensor_alloc_ptr_on(0,
 * bytes) is byte-equivalent. Single-tier callers MAY remain on the
 * legacy 1-arg helper; new multi-tier callers in ds4.c use _ptr_on. */
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier, uint64_t bytes) {
    if (tier < 0 || tier >= g_n_gpus) {
        fprintf(stderr,
            "ds4: ds4_gpu_tensor_alloc_ptr_on: bad tier %d (n_gpus=%d)\n",
            tier, g_n_gpus);
        return NULL;
    }
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (ds4_gpu_tensor_alloc_on(t, tier, bytes) != 0) {
        free(t);
        return NULL;
    }
    return t;
}

/* Heap-allocated managed-memory tensor on a specific logical tier.
 * Differs from ds4_gpu_tensor_alloc_managed only in stamping
 * tier instead of 0. Used by the per-layer KV cache when tier !=0.
 *
 * Managed-memory paging behavior: cudaMallocManaged pages between
 * devices on first-touch. In a single-tier pipeline the page lives on
 * tier 0; in a multi-tier pipeline the layer's kernels run on the
 * layer's tier so the page lives there after first-touch and stays
 * unless another device touches it. Stamping tier matches the home
 * device for free-time accounting. */
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier, uint64_t bytes) {
    if (tier < 0 || tier >= g_n_gpus) {
        fprintf(stderr,
            "ds4: ds4_gpu_tensor_alloc_managed_on: bad tier %d (n_gpus=%d)\n",
            tier, g_n_gpus);
        return NULL;
    }
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    int ok = 0;
    /* Run the cudaMallocManaged call under the home tier's device so the
     * first-touch home matches the stamped device_id; the page itself can
     * migrate freely under managed-memory semantics. */
    WITH_DEVICE(g_gpu[tier].device_id) {
        ok = cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes),
                     "managed tensor alloc (tier)");
    }
    if (!ok) { free(t); return NULL; }
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = tier;
    return t;
}

extern "C" int ds4_gpu_tensor_device(const ds4_gpu_tensor *t) {
    return t ? t->device_id : -1;
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
    t->device_id = base->device_id;  /* inherit owning device */
    return t;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    int d = ds4_tensor_device_idx(tensor);
    if (tensor->owner && tensor->ptr) {
        WITH_DEVICE(g_gpu[d].device_id) {
            (void)cudaFree(tensor->ptr);
        }
    }
    free(tensor);
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    /* Full-device sync preserves legacy semantics. */
    (void)cudaDeviceSynchronize();
    return tensor->ptr;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    int d = ds4_tensor_device_idx(tensor);
    int ok = 0;
    WITH_DEVICE(g_gpu[d].device_id) {
        fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
        ok = cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    int d = ds4_tensor_device_idx(tensor);
    int ok = 0;
    WITH_DEVICE(g_gpu[d].device_id) {
        ok = cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes,
                                cudaMemcpyHostToDevice),
                     "tensor write");
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    int d = ds4_tensor_device_idx(tensor);
    int ok = 0;
    WITH_DEVICE(g_gpu[d].device_id) {
        ok = cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes,
                                cudaMemcpyDeviceToHost),
                     "tensor read");
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    /* Same-device fast path; for cross-device, callers should use
     * ds4_gpu_tensor_copy_xdev. We still tolerate cross-device callers
     * here by routing to D2D copy on the destination's device. */
    int d = ds4_tensor_device_idx(dst);
    int ok = 0;
    WITH_DEVICE(g_gpu[d].device_id) {
        ok = cuda_ok(cudaMemcpy((char *)dst->ptr + dst_offset,
                                (const char *)src->ptr + src_offset,
                                (size_t)bytes,
                                cudaMemcpyDeviceToDevice),
                     "tensor copy");
    }
    return ok;
}

__global__ static void moe_handoff_pack_kernel(
        unsigned char *packed,
        const float *ffn_norm,
        const int32_t *selected,
        const float *weights,
        uint32_t n_embd,
        uint32_t n_expert) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float *packed_norm = (float *)packed;
    int32_t *packed_selected = (int32_t *)(packed + (uint64_t)n_embd * sizeof(float));
    float *packed_weights = (float *)(packed + (uint64_t)n_embd * sizeof(float) +
                                      (uint64_t)n_expert * sizeof(int32_t));
    if (i < n_embd) packed_norm[i] = ffn_norm[i];
    if (i < n_expert) {
        packed_selected[i] = selected[i];
        packed_weights[i] = weights[i];
    }
}

extern "C" int ds4_gpu_moe_handoff_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *ffn_norm,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert) {
    if (!packed || !ffn_norm || !selected || !weights ||
        n_embd == 0 || n_expert == 0) {
        return 0;
    }
    const uint64_t bytes = (uint64_t)n_embd * sizeof(float) +
                           (uint64_t)n_expert * sizeof(int32_t) +
                           (uint64_t)n_expert * sizeof(float);
    if (packed->bytes < bytes ||
        ffn_norm->bytes < (uint64_t)n_embd * sizeof(float) ||
        selected->bytes < (uint64_t)n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_expert * sizeof(float)) {
        return 0;
    }
    const uint32_t n = n_embd > n_expert ? n_embd : n_expert;
    moe_handoff_pack_kernel<<<(n + 255u) / 256u, 256>>>(
            (unsigned char *)packed->ptr,
            (const float *)ffn_norm->ptr,
            (const int32_t *)selected->ptr,
            (const float *)weights->ptr,
            n_embd,
            n_expert);
    return cuda_ok(cudaGetLastError(), "moe handoff pack launch");
}

/* Cross-device copy primitive. Path selection (highest priority first):
 *   DS4_FORCE_HOST_BOUNCE=1  -> always pinned-host bounce
 *   DS4_FORCE_CUDA_PEER=1    -> always cudaMemcpyPeerAsync (manual-testing
 *                               override; bypasses g_gpu_peer_ok)
 *   otherwise                -> peer if validation passed at init, else
 *                               pinned-host bounce. */
static int ds4_gpu_tensor_copy_xdev_impl(ds4_gpu_tensor *dst,
                                         const ds4_gpu_tensor *src,
                                         uint64_t bytes,
                                         bool order_dst_before_write) {
    if (!dst || !src) return 0;
    if (bytes == 0) return 1;
    if (bytes > dst->bytes || bytes > src->bytes) return 0;
    int sd = ds4_tensor_device_idx(src);
    int dd = ds4_tensor_device_idx(dst);

    /* Same-device fast path. */
    if (sd == dd) {
        int ok = 0;
        WITH_DEVICE(g_gpu[sd].device_id) {
            cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
            ok = cuda_ok(cudaMemcpyAsync(dst->ptr, src->ptr, bytes,
                                         cudaMemcpyDeviceToDevice, s),
                          "xdev same-device copy");
            if (ok && g_xdev_sync_debug) {
                ok = cuda_ok(cudaStreamSynchronize(s), "xdev same-device sync");
            }
        }
        return ok;
    }

    int peer = g_gpu_peer_ok[sd][dd];
    if (g_xdev_force_cuda_peer)   peer = 1;
    if (g_xdev_force_host_bounce) peer = 0;

    if (peer) {
        int ok = 0;
        if (order_dst_before_write) {
            WITH_DEVICE(g_gpu[dd].device_id) {
                cudaStream_t s2 = (cudaStream_t)g_gpu[dd].stream;
                cudaEvent_t  e2 = (cudaEvent_t)g_gpu[dd].boundary_event;
                ok = cuda_ok(cudaEventRecord(e2, s2), "peer dst-ready event record");
            }
            if (!ok) return 0;
        } else {
            ok = 1;
        }
        WITH_DEVICE(g_gpu[sd].device_id) {
            cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
            cudaEvent_t  e = (cudaEvent_t)g_gpu[sd].boundary_event;
            if (order_dst_before_write) {
                ok = cuda_ok(cudaStreamWaitEvent(s, (cudaEvent_t)g_gpu[dd].boundary_event, 0),
                             "peer src wait dst-ready");
            }
            if (ok) ok = cuda_ok(cudaMemcpyPeerAsync(
                            dst->ptr, g_gpu[dd].device_id,
                            src->ptr, g_gpu[sd].device_id,
                            bytes, s),
                          "peer copy");
            if (ok) ok = cuda_ok(cudaEventRecord(e, s), "peer event record");
        }
        if (!ok) return 0;
        WITH_DEVICE(g_gpu[dd].device_id) {
            cudaStream_t s2 = (cudaStream_t)g_gpu[dd].stream;
            (void)cudaStreamWaitEvent(s2, (cudaEvent_t)g_gpu[sd].boundary_event, 0);
            if (g_xdev_sync_debug) {
                ok = cuda_ok(cudaStreamSynchronize(s2), "peer dst sync");
            }
        }
        return ok;
    }

    /* Per-pair pinned-host bounce buffer. */
    if (g_xdev_bounce_bytes[sd][dd] < bytes) {
        if (g_xdev_bounce[sd][dd]) (void)cudaFreeHost(g_xdev_bounce[sd][dd]);
        if (!cuda_ok(cudaMallocHost(&g_xdev_bounce[sd][dd], (size_t)bytes),
                     "bounce alloc")) return 0;
        g_xdev_bounce_bytes[sd][dd] = bytes;
    }
    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
        cudaEvent_t  e = (cudaEvent_t)g_gpu[sd].boundary_event;
        ok = cuda_ok(cudaMemcpyAsync(g_xdev_bounce[sd][dd], src->ptr, bytes,
                                     cudaMemcpyDeviceToHost, s),
                      "bounce d2h");
        if (ok) ok = cuda_ok(cudaEventRecord(e, s), "bounce event record");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        cudaStream_t s2 = (cudaStream_t)g_gpu[dd].stream;
        (void)cudaStreamWaitEvent(s2, (cudaEvent_t)g_gpu[sd].boundary_event, 0);
        ok = cuda_ok(cudaMemcpyAsync(dst->ptr, g_xdev_bounce[sd][dd], bytes,
                                     cudaMemcpyHostToDevice, s2),
                      "bounce h2d");
        if (ok) ok = cuda_ok(cudaStreamSynchronize(s2), "bounce dst sync");
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                                        const ds4_gpu_tensor *src,
                                        uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev_impl(dst, src, bytes, false);
}

static int ds4_gpu_tensor_copy_xdev_default_impl(ds4_gpu_tensor *dst,
                                                  const ds4_gpu_tensor *src,
                                                  uint64_t bytes) {
    if (!dst || !src || bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0u) return 1;
    const int sd = ds4_tensor_device_idx(src);
    const int dd = ds4_tensor_device_idx(dst);
    if (sd == dd) {
        int ok = 0;
        WITH_DEVICE(g_gpu[sd].device_id) {
            ok = cuda_ok(cudaMemcpyAsync(dst->ptr, src->ptr, (size_t)bytes,
                                         cudaMemcpyDeviceToDevice, 0),
                         "default-stream same-device copy");
        }
        return ok;
    }

    int peer = g_gpu_peer_ok[sd][dd];
    if (g_xdev_force_cuda_peer) peer = 1;
    if (g_xdev_force_host_bounce) peer = 0;
    if (peer) {
        int ok = 0;
        WITH_DEVICE(g_gpu[sd].device_id) {
            ok = cuda_ok(cudaMemcpyPeerAsync(
                             dst->ptr, g_gpu[dd].device_id,
                             src->ptr, g_gpu[sd].device_id,
                             (size_t)bytes, 0),
                         "default-stream peer copy");
            if (ok) {
                ok = cuda_ok(cudaEventRecord(
                                 (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                             "default-stream peer event record");
            }
        }
        if (ok) {
            WITH_DEVICE(g_gpu[dd].device_id) {
                ok = cuda_ok(cudaStreamWaitEvent(
                                 0,
                                 (cudaEvent_t)g_gpu[sd].boundary_event,
                                 0),
                             "default-stream peer destination wait");
            }
        }
        return ok;
    }

    if (g_xdev_bounce_bytes[sd][dd] < bytes) {
        if (g_xdev_bounce[sd][dd]) (void)cudaFreeHost(g_xdev_bounce[sd][dd]);
        if (!cuda_ok(cudaMallocHost(&g_xdev_bounce[sd][dd], (size_t)bytes),
                     "default-stream bounce alloc")) return 0;
        g_xdev_bounce_bytes[sd][dd] = bytes;
    }
    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        ok = cuda_ok(cudaMemcpy(g_xdev_bounce[sd][dd], src->ptr, (size_t)bytes,
                                cudaMemcpyDeviceToHost),
                     "default-stream bounce d2h");
    }
    if (ok) {
        WITH_DEVICE(g_gpu[dd].device_id) {
            ok = cuda_ok(cudaMemcpy(dst->ptr, g_xdev_bounce[sd][dd],
                                    (size_t)bytes, cudaMemcpyHostToDevice),
                         "default-stream bounce h2d");
        }
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                                 const ds4_gpu_tensor *src,
                                                 uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev_default_impl(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev3_default_dst(
        ds4_gpu_tensor *dst0,
        const ds4_gpu_tensor *src0,
        uint64_t bytes0,
        ds4_gpu_tensor *dst1,
        const ds4_gpu_tensor *src1,
        uint64_t bytes1,
        ds4_gpu_tensor *dst2,
        const ds4_gpu_tensor *src2,
        uint64_t bytes2) {
    ds4_gpu_tensor *dsts[3] = {dst0, dst1, dst2};
    const ds4_gpu_tensor *srcs[3] = {src0, src1, src2};
    const uint64_t sizes[3] = {bytes0, bytes1, bytes2};
    int sd = -1;
    int dd = -1;
    for (int i = 0; i < 3; i++) {
        if (sizes[i] == 0u) continue;
        if (!dsts[i] || !srcs[i] || sizes[i] > dsts[i]->bytes ||
            sizes[i] > srcs[i]->bytes) {
            return 0;
        }
        const int this_sd = ds4_tensor_device_idx(srcs[i]);
        const int this_dd = ds4_tensor_device_idx(dsts[i]);
        if (sd < 0) {
            sd = this_sd;
            dd = this_dd;
        } else if (sd != this_sd || dd != this_dd) {
            return 0;
        }
    }
    if (sd < 0) return 1;
    if (sd == dd) {
        int ok = 1;
        WITH_DEVICE(g_gpu[sd].device_id) {
            for (int i = 0; ok && i < 3; i++) {
                if (sizes[i] == 0u) continue;
                ok = cuda_ok(cudaMemcpyAsync(
                        dsts[i]->ptr, srcs[i]->ptr, (size_t)sizes[i],
                        cudaMemcpyDeviceToDevice, 0),
                        "grouped default same-device copy");
            }
        }
        return ok;
    }

    int peer = g_gpu_peer_ok[dd][sd];
    if (g_xdev_force_cuda_peer) peer = 1;
    if (g_xdev_force_host_bounce) peer = 0;
    if (!peer) {
        for (int i = 0; i < 3; i++) {
            if (sizes[i] != 0u &&
                !ds4_gpu_tensor_copy_xdev_default_impl(
                        dsts[i], srcs[i], sizes[i])) {
                return 0;
            }
        }
        return 1;
    }

    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        ok = cuda_ok(cudaEventRecord(
                         (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                     "grouped default source-ready record");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        ok = cuda_ok(cudaStreamWaitEvent(
                         0, (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                     "grouped default destination wait");
        for (int i = 0; ok && i < 3; i++) {
            if (sizes[i] == 0u) continue;
            ok = cuda_ok(cudaMemcpyPeerAsync(
                             dsts[i]->ptr, g_gpu[dd].device_id,
                             srcs[i]->ptr, g_gpu[sd].device_id,
                             (size_t)sizes[i], 0),
                         "grouped destination-stream peer copy");
        }
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_xdev3(ds4_gpu_tensor       *dst0,
                                         const ds4_gpu_tensor *src0,
                                         uint64_t              bytes0,
                                         ds4_gpu_tensor       *dst1,
                                         const ds4_gpu_tensor *src1,
                                         uint64_t              bytes1,
                                         ds4_gpu_tensor       *dst2,
                                         const ds4_gpu_tensor *src2,
                                         uint64_t              bytes2) {
    ds4_gpu_tensor *dsts[3] = {dst0, dst1, dst2};
    const ds4_gpu_tensor *srcs[3] = {src0, src1, src2};
    uint64_t bytes[3] = {bytes0, bytes1, bytes2};
    int first = -1;
    for (int i = 0; i < 3; i++) {
        if (bytes[i] == 0) continue;
        if (!dsts[i] || !srcs[i] ||
            bytes[i] > dsts[i]->bytes || bytes[i] > srcs[i]->bytes) {
            return 0;
        }
        if (first < 0) first = i;
    }
    if (first < 0) return 1;

    const int sd = ds4_tensor_device_idx(srcs[first]);
    const int dd = ds4_tensor_device_idx(dsts[first]);
    for (int i = first + 1; i < 3; i++) {
        if (bytes[i] == 0) continue;
        if (ds4_tensor_device_idx(srcs[i]) != sd ||
            ds4_tensor_device_idx(dsts[i]) != dd) {
            int ok = 1;
            for (int j = 0; ok && j < 3; j++) {
                if (bytes[j] == 0) continue;
                ok = ds4_gpu_tensor_copy_xdev(dsts[j], srcs[j], bytes[j]);
            }
            return ok;
        }
    }

    if (sd == dd) {
        int ok = 0;
        WITH_DEVICE(g_gpu[sd].device_id) {
            cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
            ok = 1;
            for (int i = 0; ok && i < 3; i++) {
                if (bytes[i] == 0) continue;
                ok = cuda_ok(cudaMemcpyAsync(dsts[i]->ptr, srcs[i]->ptr, bytes[i],
                                             cudaMemcpyDeviceToDevice, s),
                             "xdev3 same-device copy");
            }
            if (ok && g_xdev_sync_debug) {
                ok = cuda_ok(cudaStreamSynchronize(s), "xdev3 same-device sync");
            }
        }
        return ok;
    }

    int peer = g_gpu_peer_ok[sd][dd];
    if (g_xdev_force_cuda_peer)   peer = 1;
    if (g_xdev_force_host_bounce) peer = 0;
    if (!peer) {
        int ok = 1;
        for (int i = 0; ok && i < 3; i++) {
            if (bytes[i] == 0) continue;
            ok = ds4_gpu_tensor_copy_xdev(dsts[i], srcs[i], bytes[i]);
        }
        return ok;
    }

    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
        cudaEvent_t  e = (cudaEvent_t)g_gpu[sd].boundary_event;
        ok = 1;
        for (int i = 0; ok && i < 3; i++) {
            if (bytes[i] == 0) continue;
            ok = cuda_ok(cudaMemcpyPeerAsync(
                            dsts[i]->ptr, g_gpu[dd].device_id,
                            srcs[i]->ptr, g_gpu[sd].device_id,
                            bytes[i], s),
                         "peer copy3");
        }
        if (ok) ok = cuda_ok(cudaEventRecord(e, s), "peer copy3 event record");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        cudaStream_t s2 = (cudaStream_t)g_gpu[dd].stream;
        ok = cuda_ok(cudaStreamWaitEvent(s2, (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                     "peer copy3 dst wait");
        if (ok && g_xdev_sync_debug) {
            ok = cuda_ok(cudaStreamSynchronize(s2), "peer copy3 dst sync");
        }
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                const ds4_gpu_tensor *src,
                                                uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev_impl(dst, src, bytes, true);
}

extern "C" int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src, int dst_tier) {
    if (!src) return 0;
    if (dst_tier < 0 || dst_tier >= g_n_gpus) return 0;
    int sd = ds4_tensor_device_idx(src);
    int dd = dst_tier;
    if (sd == dd) return 1;
    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
        cudaEvent_t  e = (cudaEvent_t)g_gpu[sd].boundary_event;
        ok = cuda_ok(cudaEventRecord(e, s), "xdev wait source event record");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        cudaStream_t s2 = (cudaStream_t)g_gpu[dd].stream;
        ok = cuda_ok(cudaStreamWaitEvent(s2, (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                     "xdev wait destination wait");
        if (ok && g_xdev_sync_debug) {
            ok = cuda_ok(cudaStreamSynchronize(s2), "xdev wait dst sync");
        }
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_wait_xdev_default(
        const ds4_gpu_tensor *src,
        int dst_tier) {
    if (!src || dst_tier < 0 || dst_tier >= g_n_gpus) return 0;
    const int sd = ds4_tensor_device_idx(src);
    const int dd = dst_tier;
    if (sd == dd) return 1;
    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        ok = cuda_ok(cudaEventRecord(
                         (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                     "default xdev wait source event record");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        ok = cuda_ok(cudaStreamWaitEvent(
                         0,
                         (cudaEvent_t)g_gpu[sd].boundary_event,
                         0),
                     "default xdev wait destination wait");
    }
    return ok;
}

extern "C" int ds4_gpu_q8_cache_suppressed(void) {
    return g_q8_cache_suppressed;
}

extern "C" void ds4_gpu_set_q8_cache_suppressed(int suppressed) {
    g_q8_cache_suppressed = suppressed ? 1 : 0;
}

__global__ static void pack_slot_rows_f32_kernel(float *out, const float *slots, uint32_t n_rows, uint32_t width, uint32_t n_slots, uint32_t slot_cap);

extern "C" int ds4_gpu_pack_slot_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap) {
    uint64_t slot_rows = 0;
    uint64_t slot_elems = 0;
    uint64_t out_rows = 0;
    uint64_t out_elems = 0;
    if (!out || !slots || n_rows == 0 || width == 0 || n_slots == 0 ||
        slot_cap == 0 || n_rows > slot_cap ||
        (uint64_t)n_slots > UINT64_MAX / slot_cap ||
        (slot_rows = (uint64_t)n_slots * slot_cap) > UINT64_MAX / width ||
        (slot_elems = slot_rows * width) > UINT64_MAX / sizeof(float) ||
        (uint64_t)n_rows > UINT64_MAX / n_slots ||
        (out_rows = (uint64_t)n_rows * n_slots) > UINT64_MAX / width ||
        (out_elems = out_rows * width) > UINT64_MAX / sizeof(float) ||
        slots->bytes < slot_elems * sizeof(float) ||
        out->bytes < out_elems * sizeof(float)) {
        return 0;
    }
    const uint64_t blocks = (out_elems + 255u) / 256u;
    if (blocks > UINT32_MAX) return 0;
    pack_slot_rows_f32_kernel<<<(unsigned)blocks, 256>>>(
            (float *)out->ptr,
            (const float *)slots->ptr,
            n_rows,
            width,
            n_slots,
            slot_cap);
    return cuda_ok(cudaGetLastError(), "pack_slot_rows_f32 launch");
}

extern "C" int ds4_gpu_begin_commands(void) { return 1; }
extern "C" int ds4_gpu_flush_commands(void) { return cuda_ok(cudaDeviceSynchronize(), "flush"); }
extern "C" int ds4_gpu_end_commands(void) {
    if (g_cuda_end_stream_sync) {
        return cuda_ok(cudaStreamSynchronize(0), "end commands stream");
    }
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}
extern "C" int ds4_gpu_synchronize(void) { return cuda_ok(cudaDeviceSynchronize(), "synchronize"); }

extern "C" int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;
    cuda_stream_selected_cache_release();
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
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
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }

    const char *copy_env = getenv("DS4_CUDA_COPY_MODEL");
    if (copy_env && copy_env[0]) {
        void *dev = NULL;
        const double t0 = clock() / (double)CLOCKS_PER_SEC;
        cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
        if (err == cudaSuccess) {
            fprintf(stderr, "ds4: CUDA copying %.2f GiB model to device memory\n",
                    (double)model_size / 1073741824.0);
            err = cudaMemcpy(dev, model_map, (size_t)model_size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                g_model_device_base = (const char *)dev;
                g_model_device_owned = 1;
                const double t1 = clock() / (double)CLOCKS_PER_SEC;
                fprintf(stderr, "ds4: CUDA model copy complete in %.3fs\n", t1 - t0);
                return 1;
            }
            fprintf(stderr, "ds4: CUDA model copy failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
        } else {
            fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    }

    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size,
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_device_base = (const char *)dev;
            g_model_registered = 1;
            fprintf(stderr, "ds4: CUDA registered %.2f GiB model mapping for device access\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr, "ds4: CUDA host registration pointer lookup failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr, "ds4: CUDA host registration skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!ds4_gpu_register_model_map_no_copy(model_map, model_size)) return 0;
    if (getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL &&
        !cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
        (void)cuda_model_prefetch_range(model_map, model_size, map_offset, map_size);
    }
    return 1;
}

/* Register the mmap'd host model pointer for selective-cache lookups WITHOUT
 * triggering any device-side copy. Used by multi-GPU placement scaffolding's
 * multi-tier path so DS4_CUDA_COPY_MODEL cannot reintroduce a full-model
 * copy that defeats the per-device selective cache.
 *
 * This is the no-copy subset of ds4_gpu_set_model_map: same bookkeeping
 * for the host pointer plus cudaHostRegister, but skipping the
 * DS4_CUDA_COPY_MODEL branch that allocates and copies the entire model. */
extern "C" int ds4_gpu_register_model_map_no_copy(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;

    cuda_stream_selected_cache_release();
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
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
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }

    /* No DS4_CUDA_COPY_MODEL branch — that is the entire point. */

    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size,
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_device_base = (const char *)dev;
            g_model_registered = 1;
            fprintf(stderr,
                    "ds4: CUDA (no-copy) registered %.2f GiB model mapping for multi-tier selective cache\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr,
                    "ds4: CUDA (no-copy) host registration pointer lookup failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr,
                "ds4: CUDA (no-copy) host registration skipped: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
    }
    return 1;
}

/* Set the current CUDA device by LOGICAL tier index (0..g_n_gpus-1).
 * Maps to the physical CUDA device id stored in g_gpu[].device_id.
 * Added for multi-GPU placement scaffolding (multi-GPU CLI); first executed by
 * multi-GPU execution (follow-up). */
extern "C" int ds4_gpu_set_current_device(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return -1;
    if (!g_cuda_no_setdevice_cache && g_current_logical_tier == logical_tier) {
        return 0;
    }
    if (cudaSetDevice(g_gpu[logical_tier].device_id) == cudaSuccess) {
        g_current_logical_tier = logical_tier;
        return 0;
    }
    g_current_logical_tier = -1;
    return -1;
}

/* Fenced device switch for sequential cross-device pipelines (GLM
 * per-layer placement): work queued on the next device's default stream
 * waits for everything queued so far on the previous device's default
 * stream. Async — no host sync. Falls back to a plain switch when the
 * device does not change. */
extern "C" int ds4_gpu_set_current_device_fenced(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return -1;
    static cudaEvent_t fence_ev[DS4_MAX_GPUS];
    /* Resolve the ACTUAL current device: WITH_DEVICE blocks and direct
     * cudaSetDevice calls can leave g_current_logical_tier stale, and a
     * false "already there" here strands work on the wrong device. */
    int cur_dev = -1;
    (void)cudaGetDevice(&cur_dev);
    int prev = -1;
    for (int t = 0; t < g_n_gpus; t++) {
        if (g_gpu[t].device_id == cur_dev) { prev = t; break; }
    }
    if (getenv("DS4_GLM_FENCE_TRACE")) {
        fprintf(stderr, "ds4: fenced switch %d -> %d\n", prev, logical_tier);
    }
    if (prev == logical_tier) {
        g_current_logical_tier = logical_tier;
        return 0;
    }
    if (prev >= 0 && prev < g_n_gpus && prev != logical_tier) {
        if (cudaSetDevice(g_gpu[prev].device_id) != cudaSuccess) return -1;
        if (!fence_ev[prev] &&
            cudaEventCreateWithFlags(&fence_ev[prev],
                                     cudaEventDisableTiming) != cudaSuccess) {
            fence_ev[prev] = NULL;
        }
        if (fence_ev[prev]) {
            (void)cudaEventRecord(fence_ev[prev], 0);
        }
        if (cudaSetDevice(g_gpu[logical_tier].device_id) != cudaSuccess) {
            g_current_logical_tier = -1;
            return -1;
        }
        g_current_logical_tier = logical_tier;
        if (fence_ev[prev]) {
            (void)cudaStreamWaitEvent(0, fence_ev[prev], 0);
        }
        return 0;
    }
    return ds4_gpu_set_current_device(logical_tier);
}

/* =========================================================================
 * Per-device selective model cache (selective model cache).
 *
 * ds4_gpu_device_cache_tensors copies the listed source ranges from the
 * host mmap onto device_id's selective slab and appends sorted lookup
 * entries. The legacy chunked-copy machinery (cuda_model_range_*) is
 * NOT disturbed — it continues to drive all existing callers. New
 * lookups fall back to it when no selective entry covers the range.
 *
 * Caller-context preference for overlap: when the same source range is
 * cached on multiple devices, ds4_gpu_lookup_cache returns the entry
 * whose device matches cudaGetDevice().
 * ========================================================================= */

extern "C" int ds4_gpu_device_cache_tensors(int device_id,
                                            const ds4_tensor_range *ranges,
                                            int n_ranges) {
    if (device_id < 0 || device_id >= DS4_MAX_GPUS) return 1;
    if (n_ranges < 0 || (!ranges && n_ranges > 0)) return 2;
    if (n_ranges == 0) return 0;

    if (!g_model_host_base || g_model_registered_size == 0) return 3;

    /* Validate ranges against the mmap'd model bounds; reject ranges
     * that overflow or extend past the mapped region. Done in a
     * separate pass before any allocation so a bad input doesn't
     * partially grow the slab. */
    uint64_t want_bytes = 0;
    for (int i = 0; i < n_ranges; i++) {
        if (ranges[i].target_device != device_id) continue;
        const uint64_t off = ranges[i].source_offset;
        const uint64_t nb  = ranges[i].bytes;
        /* Overflow-safe upper bound: off + nb must not exceed model
         * size, and the sum must not wrap. */
        if (nb == 0) continue;
        if (off > g_model_registered_size) return 8;
        if (nb > g_model_registered_size - off) return 9;
        /* Accumulate into want_bytes with overflow check. */
        if (want_bytes > UINT64_MAX - nb) return 10;
        want_bytes += nb;
    }
    if (want_bytes == 0) return 0;

    cuda_device_cache &c = g_dev_cache[device_id];

    int prev_device = -1;
    if (cudaGetDevice(&prev_device) != cudaSuccess) prev_device = -1;
    if (cudaSetDevice(device_id) != cudaSuccess) return 4;

    /* Allocate or grow the slab via cudaMalloc + d2d copy. */
    void *new_base = NULL;
    size_t new_bytes = c.bytes + want_bytes;

    /* Refuse cleanly before cudaMalloc if the device clearly cannot hold
     * the slab. The multi-tier packer reserves per-tier runtime scratch
     * before placing tensors, but it cannot predict the cudaMalloc
     * allocator's overhead (alignment, fragmentation after CUDA context
     * init, default driver-side reservations). On a borderline budget
     * that overhead pushes a "fits-by-packer-math" layout past the actual
     * free pool and the cudaMalloc below OOMs after the engine already
     * committed to the layout — same silent-late-OOM failure mode the
     * upfront refusal path was added to eliminate. Catch it here too. */
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            /* free_b already excludes the existing slab (it's still
             * allocated), so the additional cudaMalloc only needs
             * new_bytes free — not new_bytes + c.bytes. The old slab is
             * freed AFTER the d2d copy succeeds. 2 GiB safety covers what
             * the engine will allocate AFTER the cache slab in the same
             * session_create: per-tier graph scratch (the planner can't
             * predict its cumulative cudaMalloc alignment overhead),
             * cuBLAS workspace beyond the 64 MiB the packer already
             * reserves, and driver-side allocator slack. Without this
             * headroom a borderline budget that fits the slab itself can
             * still OOM at the per-tier tensor allocations a few moments
             * later — same silent-late-OOM failure mode, one layer up. */
            const size_t safety = (size_t)2ull * 1024ull * 1024ull * 1024ull;
            const size_t need = new_bytes + safety;
            if (need > free_b) {
                fprintf(stderr,
                        "ds4: device cache slab needs %.2f GiB on device %d "
                        "but only %.2f GiB free (slab=%.2f GiB + %.2f GiB safety). "
                        "Lower --gpu-vram / --ctx-max, or use --gpu-vram auto on "
                        "a host with more free VRAM. Refusing upfront to avoid "
                        "late OOM at cudaMalloc.\n",
                        (double)need / 1073741824.0,
                        device_id,
                        (double)free_b / 1073741824.0,
                        (double)new_bytes / 1073741824.0,
                        (double)safety / 1073741824.0);
                if (prev_device >= 0) (void)cudaSetDevice(prev_device);
                return 5;
            }
        }
        /* If cudaMemGetInfo itself failed, fall through; cudaMalloc's own
         * error path still catches the late case, just with a less helpful
         * message. */
    }

    if (!cuda_ok(cudaMalloc(&new_base, new_bytes), "device cache alloc")) {
        if (prev_device >= 0) (void)cudaSetDevice(prev_device);
        return 5;
    }
    if (c.present && c.bytes > 0) {
        cudaError_t e = cudaMemcpy(new_base, c.base, c.bytes,
                                   cudaMemcpyDeviceToDevice);
        if (e != cudaSuccess) {
            cuda_ok(e, "device cache grow d2d");
            (void)cudaFree(new_base);
            if (prev_device >= 0) (void)cudaSetDevice(prev_device);
            return 6;
        }
        /* Re-base existing entries on this device. */
        char *old_base = (char *)c.base;
        char *grown    = (char *)new_base;
        for (size_t k = 0; k < g_cache_ranges.size(); k++) {
            if (g_cache_ranges[k].device_id == device_id) {
                g_cache_ranges[k].device_ptr =
                    grown + ((char *)g_cache_ranges[k].device_ptr - old_base);
            }
        }
        (void)cudaFree(c.base);
    }
    c.base = new_base;
    c.bytes = new_bytes;
    c.present = 1;

    /* Copy ranges and append entries. */
    const char *host_base = (const char *)g_model_host_base;
    size_t write_off = c.bytes - want_bytes;
    for (int i = 0; i < n_ranges; i++) {
        if (ranges[i].target_device != device_id) continue;
        char *dev_ptr = (char *)c.base + write_off;
        cudaError_t e = cudaMemcpy(dev_ptr,
                                   host_base + ranges[i].source_offset,
                                   (size_t)ranges[i].bytes,
                                   cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            cuda_ok(e, "device cache range h2d");
            if (prev_device >= 0) (void)cudaSetDevice(prev_device);
            return 7;
        }
        cache_range_entry ent;
        ent.source_offset = ranges[i].source_offset;
        ent.bytes         = ranges[i].bytes;
        ent.device_id     = device_id;
        ent.device_ptr    = dev_ptr;
        g_cache_ranges.push_back(ent);
        write_off += ranges[i].bytes;
    }

    /* Keep sorted by source_offset for binary-search lookup. */
    std::sort(g_cache_ranges.begin(), g_cache_ranges.end(),
              [](const cache_range_entry &a, const cache_range_entry &b) {
                  if (a.source_offset != b.source_offset)
                      return a.source_offset < b.source_offset;
                  return a.device_id < b.device_id;
              });

    if (prev_device >= 0) (void)cudaSetDevice(prev_device);
    return 0;
}

/* Install support-model tensor ranges into device_id's strict cache,
 * copying from the registered support map and keying entries at
 * source_offset + bias. Standalone slab (does not touch the main cache
 * slab growth path). */
extern "C" int ds4_gpu_device_cache_support_tensors(int device_id,
                                                    int entry_device_id,
                                                    const ds4_tensor_range *ranges,
                                                    int n_ranges,
                                                    int from_main_map) {
    if (device_id < 0 || device_id >= DS4_MAX_GPUS) return 1;
    if (entry_device_id < 0 || entry_device_id >= DS4_MAX_GPUS) return 1;
    if (n_ranges <= 0 || !ranges) return 2;
    const char *src_base;
    uint64_t src_size;
    uint64_t key_bias;
    if (from_main_map) {
        /* Auxiliary main-model ranges (e.g. the embedding bucket for the
         * DSpark executor tier): standalone slab, unbiased offsets. */
        src_base = (const char *)g_model_host_base;
        src_size = g_model_registered_size;
        key_bias = 0;
    } else {
        src_base = (const char *)g_support_host_base;
        src_size = g_support_host_size;
        key_bias = g_support_offset_bias;
        if (key_bias == 0) return 3;
    }
    if (!src_base || src_size == 0) return 3;
    uint64_t want = 0;
    for (int i = 0; i < n_ranges; i++) {
        const uint64_t off = ranges[i].source_offset;
        const uint64_t nb = ranges[i].bytes;
        if (nb == 0) continue;
        if (off > src_size || nb > src_size - off) return 8;
        if (want > UINT64_MAX - nb) return 9;
        want += nb;
    }
    if (want == 0) return 0;
    int prev_device = -1;
    if (cudaGetDevice(&prev_device) != cudaSuccess) prev_device = -1;
    if (cudaSetDevice(device_id) != cudaSuccess) return 4;
    void *base = NULL;
    if (!cuda_ok(cudaMalloc(&base, (size_t)want), "support cache alloc")) {
        if (prev_device >= 0) (void)cudaSetDevice(prev_device);
        return 5;
    }
    const char *host_base = src_base;
    size_t write_off = 0;
    for (int i = 0; i < n_ranges; i++) {
        if (ranges[i].bytes == 0) continue;
        char *dev_ptr = (char *)base + write_off;
        cudaError_t e = cudaMemcpy(dev_ptr,
                                   host_base + ranges[i].source_offset,
                                   (size_t)ranges[i].bytes,
                                   cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            cuda_ok(e, "support cache range h2d");
            (void)cudaFree(base);
            if (prev_device >= 0) (void)cudaSetDevice(prev_device);
            return 7;
        }
        cache_range_entry ent;
        ent.source_offset = ranges[i].source_offset + key_bias;
        ent.bytes = ranges[i].bytes;
        /* Entries can claim a different (executor) device than the one the
         * slab physically lives on: strict lookups filter by entry device,
         * and peer access lets the executor's kernels dereference the
         * spilled pointer directly. */
        ent.device_id = entry_device_id;
        ent.device_ptr = dev_ptr;
        g_cache_ranges.push_back(ent);
        write_off += ranges[i].bytes;
    }
    std::sort(g_cache_ranges.begin(), g_cache_ranges.end(),
              [](const cache_range_entry &a, const cache_range_entry &b) {
                  if (a.source_offset != b.source_offset)
                      return a.source_offset < b.source_offset;
                  return a.device_id < b.device_id;
              });
    if (getenv("DS4_DSPARK_VERIFY_CACHE") != NULL) {
        /* Read back every installed range and compare with the host copy. */
        int bad = 0;
        write_off = 0;
        for (int i = 0; i < n_ranges; i++) {
            if (ranges[i].bytes == 0) continue;
            char *dev_ptr = (char *)base + write_off;
            std::vector<char> tmp((size_t)ranges[i].bytes);
            if (cudaMemcpy(tmp.data(), dev_ptr, (size_t)ranges[i].bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                memcmp(tmp.data(), host_base + ranges[i].source_offset,
                       (size_t)ranges[i].bytes) != 0) {
                fprintf(stderr,
                        "ds4: support cache VERIFY MISMATCH offset=%llu bytes=%llu dev=%d\n",
                        (unsigned long long)ranges[i].source_offset,
                        (unsigned long long)ranges[i].bytes, device_id);
                bad++;
            }
            write_off += ranges[i].bytes;
        }
        fprintf(stderr, "ds4: support cache verify dev=%d ranges=%d bad=%d\n",
                device_id, n_ranges, bad);
    }
    if (prev_device >= 0) (void)cudaSetDevice(prev_device);
    return 0;
}

extern "C" int ds4_gpu_lookup_cache(uint64_t source_offset, uint64_t bytes,
                                    int *out_device_id, void **out_device_ptr) {
    int active_device = -1;
    (void)cudaGetDevice(&active_device);

    if (!g_cache_ranges.empty()) {
        /* upper_bound: first entry with source_offset > query.
         * Candidates are at strictly earlier positions; scan all of
         * them rather than breaking on the first non-covering entry,
         * because the table allows overlap across devices. */
        auto it = std::upper_bound(
            g_cache_ranges.begin(), g_cache_ranges.end(),
            source_offset,
            [](uint64_t off, const cache_range_entry &e) {
                return off < e.source_offset;
            });
        const cache_range_entry *match_any  = NULL;
        const cache_range_entry *match_pref = NULL;
        while (it != g_cache_ranges.begin()) {
            --it;
            /* Overflow-safe coverage check:
             *   1. source_offset >= it->source_offset
             *   2. bytes <= it->bytes - (source_offset - it->source_offset)
             * The second form computes only the remaining capacity inside
             * the entry, so neither side can overflow even with bytes ==
             * UINT64_MAX. */
            if (source_offset >= it->source_offset) {
                uint64_t into = source_offset - it->source_offset;
                if (into <= it->bytes && bytes <= it->bytes - into) {
                    if (it->device_id == active_device) {
                        match_pref = &*it;
                        break;
                    }
                    if (!match_any) match_any = &*it;
                }
            }
            /* Do NOT break on non-covering: an earlier entry may still
             * cover if its bytes extend far enough. */
        }
        const cache_range_entry *m = match_pref ? match_pref : match_any;
        if (m) {
            if (out_device_id) *out_device_id = m->device_id;
            if (out_device_ptr) {
                *out_device_ptr =
                    (char *)m->device_ptr + (source_offset - m->source_offset);
            }
            return 1;
        }
    }

    /* Legacy chunk-aware fallback (device 0 only). */
    const char *p = cuda_model_range_ptr_from_fd(g_model_host_base,
                                                  source_offset, bytes,
                                                  "lookup_cache");
    if (p) {
        if (out_device_id) *out_device_id = 0;
        if (out_device_ptr) *out_device_ptr = (void *)p;
        return 1;
    }
    return 0;
}

extern "C" int ds4_gpu_lookup_cache_device(uint64_t source_offset, uint64_t bytes) {
    int d = -1;
    if (!ds4_gpu_lookup_cache(source_offset, bytes, &d, NULL)) return -1;
    return d;
}

/* Strict per-device selective-cache lookup.
 *
 * Returns 1 only if a covering entry exists whose device_id matches the
 * caller-supplied expected_device. Otherwise returns 0 with *out_device_ptr
 * untouched. Unlike ds4_gpu_lookup_cache, this variant performs NO host-
 * pointer fallback (no FD-cache, no model_range_ptr_from_fd) and NO
 * different-device fallback. It is the canonical lookup for multi-tier
 * dispatch where consuming a different device's pointer would be a
 * correctness bug. expected_device is a PHYSICAL CUDA device id (the
 * value stored in g_gpu[logical_tier].device_id, not the logical tier
 * index). The caller is expected to have cudaSetDevice'd to
 * expected_device before invoking; the returned pointer is valid to
 * consume from that device's kernel. Added for
 * multi-GPU execution (multi-GPU execution). */
extern "C" int ds4_gpu_lookup_cache_strict(uint64_t source_offset,
                                            uint64_t bytes,
                                            int      expected_device,
                                            void   **out_device_ptr) {
    if (g_cache_ranges.empty()) return 0;

    auto it = std::upper_bound(
        g_cache_ranges.begin(), g_cache_ranges.end(),
        source_offset,
        [](uint64_t off, const cache_range_entry &e) {
            return off < e.source_offset;
        });
    while (it != g_cache_ranges.begin()) {
        --it;
        if (source_offset < it->source_offset) {
            /* Should not happen given upper_bound semantics, but defensive. */
            continue;
        }
        uint64_t into = source_offset - it->source_offset;
        if (into > it->bytes) continue;
        if (bytes > it->bytes - into) continue;
        if (it->device_id != expected_device) continue;
        if (out_device_ptr) {
            *out_device_ptr =
                (char *)it->device_ptr + (source_offset - it->source_offset);
        }
        return 1;
    }
    return 0;
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
        if (getenv("DS4_CUDA_NO_DIRECT_IO") == NULL) {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA model direct I/O enabled (align=%llu)\n",
                            (unsigned long long)g_model_direct_align);
                }
            } else if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "ds4: CUDA model direct I/O unavailable: %s\n", strerror(errno));
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
    /* Preload runs before any multi-tier dispatch. The cache entries it creates
     * are device-0 by construction; multi-tier callers in kernel wrappers will
     * miss the linear scan (device_id filter) and allocate fresh per-device
     * copies the first time they're consulted. */
    if (getenv("DS4_CUDA_Q8_F32_PRELOAD") != NULL &&
        cuda_q8_f32_cache_allowed(cache_label, in_dim, out_dim)) {
        if (cuda_q8_f32_ptr(model_map, offset, bytes, in_dim, out_dim, 0, cache_label)) return 1;
        optional_q8_preload_disabled = 1;
        return 1;
    }
    if (!cuda_q8_f16_preload_allowed(cache_label, in_dim, out_dim)) return 1;
    if (cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, 0, cache_label)) return 1;
    optional_q8_preload_disabled = 1;
    return 1;
}

extern "C" void ds4_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    (void)cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr, "ds4: CUDA memory report %s: free %.2f MiB total %.2f MiB\n",
            label ? label : "", (double)free_b / 1048576.0, (double)total_b / 1048576.0);
}

extern "C" void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
    const cublasMath_t math_mode =
        (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
            ? CUBLAS_DEFAULT_MATH
            : CUBLAS_TF32_TENSOR_OP_MATH;
    /* Walk every initialized per-tier handle. Single-tier (g_n_gpus == 1)
     * walks exactly one entry. On any device-switch failure,
     * skip the tier and continue — the function is void and the math-mode
     * setting is advisory, but log so misconfiguration is visible. */
    for (int i = 0; i < g_n_gpus; i++) {
        if (!g_gpu[i].cublas_ready || !g_gpu[i].cublas) continue;
        int prev = -1;
        cudaError_t derr = cudaGetDevice(&prev);
        if (derr != cudaSuccess) {
            fprintf(stderr,
                "ds4: ds4_gpu_set_quality: cudaGetDevice failed before tier %d "
                "(dev=%d): %s; skipping\n",
                i, g_gpu[i].device_id, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            continue;
        }
        derr = cudaSetDevice(g_gpu[i].device_id);
        if (derr != cudaSuccess) {
            fprintf(stderr,
                "ds4: ds4_gpu_set_quality: cudaSetDevice(%d) failed for tier %d: "
                "%s; skipping\n",
                g_gpu[i].device_id, i, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            if (prev >= 0) (void)cudaSetDevice(prev);
            continue;
        }
        cublasStatus_t st = cublasSetMathMode((cublasHandle_t)g_gpu[i].cublas, math_mode);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                "ds4: ds4_gpu_set_quality: cublasSetMathMode failed on tier %d "
                "(dev=%d): status %d\n",
                i, g_gpu[i].device_id, (int)st);
        }
        if (prev >= 0) (void)cudaSetDevice(prev);
    }
}
