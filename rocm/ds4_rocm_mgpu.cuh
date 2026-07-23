/* rocm/ds4_rocm_mgpu.cuh — native multi-GPU device placement for ROCm.
 *
 * Mirrors the "GPU-only multi-tier" scaffolding ds4_cuda.cu has had for
 * CUDA: per-device streams/hipBLAS/hipBLASLt handles, a per-device
 * selective weight cache, cross-device tensor copy with peer-access
 * validation and a pinned host-bounce fallback, and the
 * ds4_gpu_set_current_device(tier) shim ds4.c calls at layer boundaries in
 * a multi-tier placement (see metal_graph_set_active_tier_* in ds4.c).
 *
 * Scope: this covers layer/pipeline placement across tiers (weights for a
 * given transformer layer resident on one device; activations ferried
 * across tiers at layer boundaries). It does NOT implement
 * --cuda-tensor-parallel (row/col-sharded compute within a single layer
 * split live across two tiers) -- that mode still requires
 * DS4_BACKEND_CUDA explicitly elsewhere in ds4.c and is refused for ROCm
 * exactly as before. Because of that narrower scope, kernel weight
 * resolution below keys off "whichever device is currently active"
 * (hipGetDevice()) rather than requiring every one of the ~200 kernel
 * launchers across the rocm headers to thread an explicit tier argument through
 * -- ds4.c guarantees the right device is already current before it
 * dispatches a layer's kernels, and no single layer's kernels ever need
 * two tiers active at once outside tensor-parallel mode.
 *
 * ds4_gpu_config/ds4_gpu_ctx are declared here with local (differently
 * named) struct types rather than by #include "ds4_gpu_mgpu.h": that
 * header's ds4_gpu_tensor struct body would collide with the one
 * ds4_rocm.cu already defines above this include. The extern "C" symbols
 * this file defines (g_gpu, g_n_gpus, g_gpu_peer_ok, ds4_gpu_init_multi,
 * ...) link against ds4_gpu_mgpu.h's declarations from other translation
 * units (ds4_rocm_compat.cu, ds4.c) purely by symbol name and matching
 * memory layout -- extern "C" linkage does not check parameter/struct
 * identity across TUs, only symbol names. Field order/types below must
 * stay byte-for-byte identical to ds4_gpu_mgpu.h's ds4_gpu_config /
 * ds4_gpu_ctx. This mirrors how struct ds4_gpu_tensor itself is already
 * handled across ds4_rocm.cu and ds4_rocm_compat.cu in this codebase.
 */

/* struct ds4_rocm_gpu_ctx/ds4_rocm_gpu_config and the extern declarations
 * of g_gpu/g_n_gpus/g_gpu_peer_ok live near the top of ds4_rocm.cu (ahead
 * of ds4_rocm_runtime.cuh) -- the tensor-op functions defined there need
 * g_gpu[tier].device_id too. This is the storage definition. */
ds4_rocm_gpu_ctx g_gpu[DS4_MAX_GPUS] = {};
int              g_n_gpus = 1;
int              g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS] = {{1}};

/* g_current_logical_tier is declared earlier, in ds4_rocm_runtime.cuh, so
 * cuda_tmp_alloc there can use it too. */
static int g_rocm_no_setdevice_cache = 0;
static int g_rocm_force_host_bounce = 0;

/* Per-pair pinned-host bounce buffers, indexed [src][dst]. Lazily grown to
 * the largest copy seen for that pair. */
static void   *g_mgpu_xdev_bounce[DS4_MAX_GPUS][DS4_MAX_GPUS];
static size_t  g_mgpu_xdev_bounce_bytes[DS4_MAX_GPUS][DS4_MAX_GPUS];

/* Per-tier hipBLAS / hipBLASLt handles. ds4_gpu_set_current_device swaps
 * these into the plain g_cublas/g_cublas_ready/g_hipblaslt/g_hipblaslt_ready
 * globals (declared earlier in ds4_rocm_runtime.cuh) that every existing
 * matmul kernel launcher already reads -- so none of those launchers need
 * to change. hipBLASLt matmul *plans* (layout descriptors + chosen algo,
 * see ds4_rocm_hipblaslt.cuh) are pure shape/dtype metadata, not bound to a
 * device context, and both tiers are identical hardware here, so the plan
 * cache (g_hipblaslt_gemm_plans) is safely shared across tiers unswapped. */
static cublasHandle_t    g_gpu_cublas[DS4_MAX_GPUS];
static int               g_gpu_cublas_ready[DS4_MAX_GPUS];
#ifdef __HIP_PLATFORM_AMD__
static hipblasLtHandle_t g_gpu_hipblaslt[DS4_MAX_GPUS];
static int               g_gpu_hipblaslt_ready[DS4_MAX_GPUS];
#endif

/* ds4_tensor_device_idx and WITH_DEVICE are defined near the top of
 * ds4_rocm.cu, ahead of ds4_rocm_runtime.cuh. */

/* =========================================================================
 * Per-device selective weight cache.
 *
 * cuda_model_range_ptr (defined earlier in this file) is the single
 * chokepoint every rocm-header kernel launcher already funnels through to
 * resolve model_map+offset to a device pointer. Rather than threading a
 * tier argument through every launcher, ds4_gpu_device_cache_tensors
 * populates a cache indexed by physical device id, and
 * cuda_model_range_ptr is extended (see below) to consult the entry for
 * "whichever device is currently active" before falling back to its
 * existing single-copy behavior. ========================================= */

struct ds4_rocm_cache_range_entry {
    uint64_t source_offset;
    uint64_t bytes;
    int      device_id;
    void    *device_ptr;
};
static std::vector<ds4_rocm_cache_range_entry> g_mgpu_cache_ranges;

struct ds4_rocm_device_cache {
    void   *base;
    size_t  bytes;
    int     present;
};
static ds4_rocm_device_cache g_mgpu_dev_cache[DS4_MAX_GPUS];

/* Strict per-device lookup: returns 1 only if a covering entry exists whose
 * device_id matches expected_device (a PHYSICAL device id). No fallback. */
extern "C" int ds4_gpu_lookup_cache_strict(uint64_t source_offset,
                                            uint64_t bytes,
                                            int      expected_device,
                                            void   **out_device_ptr) {
    for (size_t i = 0; i < g_mgpu_cache_ranges.size(); i++) {
        const ds4_rocm_cache_range_entry &e = g_mgpu_cache_ranges[i];
        if (e.device_id != expected_device) continue;
        if (source_offset < e.source_offset) continue;
        uint64_t into = source_offset - e.source_offset;
        if (into > e.bytes || bytes > e.bytes - into) continue;
        if (out_device_ptr) {
            *out_device_ptr = (char *)e.device_ptr + into;
        }
        return 1;
    }
    return 0;
}

extern "C" int ds4_gpu_lookup_cache(uint64_t source_offset, uint64_t bytes,
                                     int *out_device_id, void **out_device_ptr) {
    int active_device = -1;
    (void)cudaGetDevice(&active_device);
    for (size_t i = 0; i < g_mgpu_cache_ranges.size(); i++) {
        const ds4_rocm_cache_range_entry &e = g_mgpu_cache_ranges[i];
        if (source_offset < e.source_offset) continue;
        uint64_t into = source_offset - e.source_offset;
        if (into > e.bytes || bytes > e.bytes - into) continue;
        if (e.device_id != active_device) continue;
        if (out_device_id) *out_device_id = e.device_id;
        if (out_device_ptr) *out_device_ptr = (char *)e.device_ptr + into;
        return 1;
    }
    /* Legacy single-copy fallback (device 0 / whatever's resident). */
    const char *p = cuda_model_range_ptr(g_model_host_base, source_offset, bytes,
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

/* No-copy variant of ds4_gpu_set_model_map: registers the host mmap for
 * selective-cache lookups without triggering a full-model device copy, so
 * DS4_CUDA_COPY_MODEL-equivalent behavior can't reintroduce a full copy
 * that would defeat the per-device selective cache.
 *
 * Deliberately does NOT hipHostRegister(..., hipHostRegisterMapped) the
 * mapping the way CUDA's reference does. That flag asks the driver for a
 * GPU-visible (SVM) zero-copy pointer to the whole host mmap so uncached
 * fallback lookups can dereference it directly -- but with two GPU agents
 * registered, requesting an SVM mapping across an ~80 GiB host region
 * overruns the driver's resident-SVM working-set limit (seen on this box
 * as a flood of "amdgpu: SVM mapping failed, exceeds resident system
 * memory limit" kernel log lines followed by a hard crash inside
 * libamdhip64 on the next unrelated kernel launch). It also isn't needed:
 * engine_install_per_device_caches (ds4.c) uploads every GPU-placed
 * tensor into the selective per-device cache via
 * ds4_gpu_device_cache_tensors before any layer runs, so the "uncached
 * fallback" path this pointer exists for is not on the hot path here. */
extern "C" int ds4_gpu_register_model_map_no_copy(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;
    cuda_model_range_release_all();
    g_model_host_base = model_map;
    g_model_device_base = (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_range_mapping_supported = 1;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }
    return 1;
}

/* Reusable pinned staging buffer for the weight-cache H2D copies below.
 *
 * hipMemcpy from a plain pageable host pointer (our mmap'd model file)
 * cannot DMA directly -- the HIP runtime pins a bounce buffer per call
 * ("DmaBlitManager::getBuffer") and stages through it. On this box that
 * per-call pin/copy overhead dominates once the memlock ulimit is no
 * longer the bottleneck (see ds4_gpu_register_model_map_no_copy's comment
 * for that earlier issue): copying an already-page-cached 26 GiB tier
 * this way still took well over a minute. hipHostRegister()'ing the whole
 * ~80 GiB model isn't an option either -- pinning has to fit in real RAM,
 * and this box has 62 GiB. Instead, pin ONE modest buffer up front and
 * reuse it for every range: a plain host-to-host memcpy (memory-bandwidth
 * speed, page cache already warmed by the bulk pread above) into the
 * pinned buffer, then a real DMA'd hipMemcpy out of it. */
enum { DS4_ROCM_MGPU_STAGE_BYTES = 128u * 1024u * 1024u };
static void *g_mgpu_stage_buf;

static bool rocm_mgpu_pinned_h2d_copy(void *dst, const void *src, size_t bytes) {
    if (bytes == 0) return true;
    if (!g_mgpu_stage_buf) {
        if (cudaMallocHost(&g_mgpu_stage_buf, DS4_ROCM_MGPU_STAGE_BYTES) != cudaSuccess) {
            g_mgpu_stage_buf = NULL;
        }
    }
    if (!g_mgpu_stage_buf) {
        /* Pinned staging alloc failed (e.g. still memlock-constrained) --
         * fall back to the direct pageable copy rather than hard-failing. */
        return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    size_t done = 0;
    while (done < bytes) {
        const size_t n = (bytes - done) < DS4_ROCM_MGPU_STAGE_BYTES
                              ? (bytes - done) : (size_t)DS4_ROCM_MGPU_STAGE_BYTES;
        memcpy(g_mgpu_stage_buf, (const char *)src + done, n);
        if (cudaMemcpy((char *)dst + done, g_mgpu_stage_buf, n,
                        cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        done += n;
    }
    return true;
}

extern "C" int ds4_gpu_device_cache_tensors(int device_id,
                                             const ds4_tensor_range *ranges,
                                             int n_ranges) {
    if (device_id < 0 || device_id >= DS4_MAX_GPUS) return 1;
    if (n_ranges < 0 || (!ranges && n_ranges > 0)) return 2;
    if (n_ranges == 0) return 0;
    if (!g_model_host_base || g_model_registered_size == 0) return 3;

    uint64_t want_bytes = 0;
    uint64_t span_min = UINT64_MAX;
    uint64_t span_max = 0;
    for (int i = 0; i < n_ranges; i++) {
        if (ranges[i].target_device != device_id) continue;
        const uint64_t off = ranges[i].source_offset;
        const uint64_t nb  = ranges[i].bytes;
        if (nb == 0) continue;
        if (off > g_model_registered_size) return 8;
        if (nb > g_model_registered_size - off) return 9;
        if (want_bytes > UINT64_MAX - nb) return 10;
        want_bytes += nb;
        if (off < span_min) span_min = off;
        if (off + nb > span_max) span_max = off + nb;
    }
    if (want_bytes == 0) return 0;

    /* This device's ranges typically span most of the source mmap (its
     * share of the model, in roughly ascending file-offset order), but the
     * per-range copy loop below issues one small (tens-of-MiB) cudaMemcpy
     * per tensor straight off the mmap. Without warming the page cache
     * first, each copy demand-faults its own pages piecemeal as the HIP
     * runtime's pageable-copy staging touches them -- on a model that
     * doesn't fully fit in RAM this measured at ~15 MiB/s (page-fault-at-a-
     * time), vs. ~2 GiB/s for a plain sequential read of the same file on
     * this box's NVMe RAID1. posix_fadvise/madvise(WILLNEED) hints did not
     * close that gap in practice (async readahead apparently can't stay
     * ahead of the memcpy loop's consumption), so warm the cache the blunt
     * way: a synchronous bulk sequential pread() over the whole span
     * before touching it via cudaMemcpy. Best-effort: a failed/partial
     * warm just means the subsequent memcpy loop falls back to demand-
     * faulting those pages itself, not a correctness issue. */
    if (span_min < span_max && g_model_fd >= 0) {
        const size_t warm_chunk = 64ull * 1024ull * 1024ull;
        char *warm_buf = (char *)malloc(warm_chunk);
        if (warm_buf) {
            uint64_t off = span_min;
            while (off < span_max) {
                const size_t n = (size_t)((span_max - off) < warm_chunk
                                               ? (span_max - off) : warm_chunk);
                const ssize_t got = pread(g_model_fd, warm_buf, n, (off_t)off);
                if (got <= 0) break;
                off += (uint64_t)got;
            }
            free(warm_buf);
        }
    }

    ds4_rocm_device_cache &c = g_mgpu_dev_cache[device_id];
    int prev_device = -1;
    if (cudaGetDevice(&prev_device) != cudaSuccess) prev_device = -1;
    if (cudaSetDevice(device_id) != cudaSuccess) return 4;

    size_t new_bytes = c.bytes + want_bytes;
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            const size_t safety = (size_t)2ull * 1024ull * 1024ull * 1024ull;
            const size_t need = new_bytes + safety;
            if (need > free_b) {
                fprintf(stderr,
                        DS4_GPU_LOG_PREFIX "device cache slab needs %.2f GiB on device %d "
                        "but only %.2f GiB free (slab=%.2f GiB + %.2f GiB safety). Lower "
                        "--gpu-vram / --ctx-max, or use --gpu-vram auto on a host with more "
                        "free VRAM. Refusing upfront to avoid late OOM at cudaMalloc.\n",
                        (double)need / 1073741824.0, device_id,
                        (double)free_b / 1073741824.0,
                        (double)new_bytes / 1073741824.0,
                        (double)safety / 1073741824.0);
                if (prev_device >= 0) (void)cudaSetDevice(prev_device);
                return 5;
            }
        }
    }

    void *new_base = NULL;
    if (!cuda_ok(cudaMalloc(&new_base, new_bytes), "device cache alloc")) {
        if (prev_device >= 0) (void)cudaSetDevice(prev_device);
        return 5;
    }
    if (c.present && c.bytes > 0) {
        cudaError_t e = cudaMemcpy(new_base, c.base, c.bytes, cudaMemcpyDeviceToDevice);
        if (e != cudaSuccess) {
            cuda_ok(e, "device cache grow d2d");
            (void)cudaFree(new_base);
            if (prev_device >= 0) (void)cudaSetDevice(prev_device);
            return 6;
        }
        char *old_base = (char *)c.base;
        char *grown    = (char *)new_base;
        for (size_t k = 0; k < g_mgpu_cache_ranges.size(); k++) {
            if (g_mgpu_cache_ranges[k].device_id == device_id) {
                g_mgpu_cache_ranges[k].device_ptr =
                    grown + ((char *)g_mgpu_cache_ranges[k].device_ptr - old_base);
            }
        }
        (void)cudaFree(c.base);
    }
    c.base = new_base;
    c.bytes = new_bytes;
    c.present = 1;

    const char *host_base = (const char *)g_model_host_base;
    size_t write_off = c.bytes - want_bytes;
    for (int i = 0; i < n_ranges; i++) {
        if (ranges[i].target_device != device_id) continue;
        char *dev_ptr = (char *)c.base + write_off;
        if (!rocm_mgpu_pinned_h2d_copy(dev_ptr, host_base + ranges[i].source_offset,
                                        (size_t)ranges[i].bytes)) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "device cache range h2d failed\n");
            if (prev_device >= 0) (void)cudaSetDevice(prev_device);
            return 7;
        }
        ds4_rocm_cache_range_entry ent;
        ent.source_offset = ranges[i].source_offset;
        ent.bytes = ranges[i].bytes;
        ent.device_id = device_id;
        ent.device_ptr = dev_ptr;
        g_mgpu_cache_ranges.push_back(ent);
        write_off += ranges[i].bytes;
    }

    std::sort(g_mgpu_cache_ranges.begin(), g_mgpu_cache_ranges.end(),
              [](const ds4_rocm_cache_range_entry &a, const ds4_rocm_cache_range_entry &b) {
                  if (a.source_offset != b.source_offset) return a.source_offset < b.source_offset;
                  return a.device_id < b.device_id;
              });

    if (prev_device >= 0) (void)cudaSetDevice(prev_device);
    return 0;
}

/* DSpark/MTP support-model per-device caching is not ported yet (that
 * feature has never been exercised on the ROCm backend). Behavior matches
 * the pre-multi-GPU ROCM_UNAVAILABLE stub: report unavailable rather than
 * silently doing nothing. */
extern "C" int ds4_gpu_device_cache_support_tensors(int device_id,
                                                     int entry_device_id,
                                                     const ds4_tensor_range *ranges,
                                                     int n_ranges,
                                                     int from_main_map) {
    (void)device_id; (void)entry_device_id; (void)ranges; (void)n_ranges; (void)from_main_map;
    return 1;
}

/* =========================================================================
 * Multi-device lifecycle.
 * ========================================================================= */

extern "C" int ds4_gpu_init_multi(const ds4_rocm_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus < 1 || cfg->n_gpus > DS4_MAX_GPUS) return 0;
    g_rocm_no_setdevice_cache = getenv("DS4_ROCM_NO_SETDEVICE_CACHE") != NULL;
    g_rocm_force_host_bounce = getenv("DS4_FORCE_HOST_BOUNCE") != NULL;
    g_current_logical_tier = -1;

    for (int i = 0; i < cfg->n_gpus; i++) {
        ds4_rocm_gpu_ctx *c = &g_gpu[i];
        c->device_id = cfg->device_indices[i];
        if (c->device_id < 0) return 0;
        g_n_gpus = i + 1;
        if (!cuda_ok(cudaSetDevice(c->device_id), "init set device")) return 0;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, c->device_id) == cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "backend initialized on %s (gfx) dev=%d\n",
                    prop.name, c->device_id);
        }
        cudaStream_t s = NULL;
        if (!cuda_ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "init stream")) return 0;
        c->stream = (void *)s;
        cudaEvent_t ev = NULL;
        if (!cuda_ok(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "init event")) return 0;
        c->boundary_event = (void *)ev;

        if (!cublas_ok(cublasCreate(&g_gpu_cublas[i]), "create handle")) return 0;
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_gpu_cublas[i], math_mode);
        g_gpu_cublas_ready[i] = 1;
#ifdef __HIP_PLATFORM_AMD__
        if (hipblaslt_ok(hipblasLtCreate(&g_gpu_hipblaslt[i]), "create handle")) {
            g_gpu_hipblaslt_ready[i] = 1;
        }
#endif
        c->cublas = (void *)g_gpu_cublas[i];
        c->cublas_ready = g_gpu_cublas_ready[i];
        c->budget_bytes = cfg->vram_bytes[i];
        c->used_bytes = 0;
        c->scratch = NULL;
        c->scratch_bytes = 0;
    }

    /* NxN peer-access matrix. cudaDeviceCanAccessPeer/EnablePeerAccess can
     * both report success on hardware where the copy path still misbehaves
     * (seen on other vendors' drivers; treat as a real risk here too, not
     * a CUDA-only concern) -- validate with a real round-trip before
     * trusting a pair, and fall back to the pinned-host bounce path when
     * validation fails or peer access isn't available at all (expected on
     * a pure-PCIe pair with no P2P-capable switch between them). */
    for (int i = 0; i < g_n_gpus; i++) {
        for (int j = 0; j < g_n_gpus; j++) {
            if (i == j) { g_gpu_peer_ok[i][j] = 1; continue; }
            g_gpu_peer_ok[i][j] = 0;
            if (g_rocm_force_host_bounce) continue;
            int can = 0;
            (void)cudaDeviceCanAccessPeer(&can, g_gpu[i].device_id, g_gpu[j].device_id);
            if (!can) continue;
            (void)cudaSetDevice(g_gpu[i].device_id);
            cudaError_t pe = cudaDeviceEnablePeerAccess(g_gpu[j].device_id, 0);
            int enabled = (pe == cudaSuccess || pe == cudaErrorPeerAccessAlreadyEnabled);
            (void)cudaGetLastError();
            if (!enabled) continue;

            static const size_t kValidateSizes[] = {4u * 1024u, 1u * 1024u * 1024u,
                                                      16u * 1024u * 1024u};
            const int kIters = 3;
            const size_t kMax = kValidateSizes[2];
            unsigned char *vh_src = (unsigned char *)malloc(kMax);
            unsigned char *vh_dst = (unsigned char *)malloc(kMax);
            void *src_dev = NULL; void *dst_dev = NULL;
            int validated = vh_src && vh_dst;
            if (validated) {
                (void)cudaSetDevice(g_gpu[i].device_id);
                validated = cudaMalloc(&src_dev, kMax) == cudaSuccess;
            }
            if (validated) {
                (void)cudaSetDevice(g_gpu[j].device_id);
                validated = cudaMalloc(&dst_dev, kMax) == cudaSuccess;
            }
            for (int s_idx = 0; validated && s_idx < 3; s_idx++) {
                size_t n = kValidateSizes[s_idx];
                for (int it = 0; validated && it < kIters; it++) {
                    for (size_t k = 0; k < n; k++) {
                        vh_src[k] = (unsigned char)((k * 31u + (size_t)it * 17u +
                                                      (size_t)s_idx * 53u + 11u) & 0xffu);
                    }
                    (void)cudaSetDevice(g_gpu[i].device_id);
                    if (cudaMemcpy(src_dev, vh_src, n, cudaMemcpyHostToDevice) != cudaSuccess) {
                        validated = 0; break;
                    }
                    if (cudaMemcpyPeer(dst_dev, g_gpu[j].device_id, src_dev,
                                        g_gpu[i].device_id, n) != cudaSuccess) {
                        validated = 0; break;
                    }
                    (void)cudaSetDevice(g_gpu[j].device_id);
                    if (cudaMemcpy(vh_dst, dst_dev, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
                        validated = 0; break;
                    }
                    if (memcmp(vh_src, vh_dst, n) != 0) { validated = 0; break; }
                }
            }
            if (dst_dev) { (void)cudaSetDevice(g_gpu[j].device_id); (void)cudaFree(dst_dev); }
            if (src_dev) { (void)cudaSetDevice(g_gpu[i].device_id); (void)cudaFree(src_dev); }
            free(vh_src); free(vh_dst);

            g_gpu_peer_ok[i][j] = validated;
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "peer access %d->%d %s\n",
                    g_gpu[i].device_id, g_gpu[j].device_id,
                    validated ? "validated" : "unavailable/failed validation; using pinned-host bounce");
        }
    }

    /* Leave device 0's (tier 0) handles active as the plain g_cublas /
     * g_hipblaslt globals every kernel launcher already reads. */
    g_current_logical_tier = 0;
    g_cublas = g_gpu_cublas[0];
    g_cublas_ready = g_gpu_cublas_ready[0];
#ifdef __HIP_PLATFORM_AMD__
    g_hipblaslt = g_gpu_hipblaslt[0];
    g_hipblaslt_ready = g_gpu_hipblaslt_ready[0];
#endif
    (void)cudaSetDevice(g_gpu[0].device_id);
    return 1;
}

extern "C" int ds4_gpu_set_current_device(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return -1;
    if (!g_rocm_no_setdevice_cache && g_current_logical_tier == logical_tier) return 0;
    if (cudaSetDevice(g_gpu[logical_tier].device_id) != cudaSuccess) {
        g_current_logical_tier = -1;
        return -1;
    }
    g_current_logical_tier = logical_tier;
    g_cublas = g_gpu_cublas[logical_tier];
    g_cublas_ready = g_gpu_cublas_ready[logical_tier];
#ifdef __HIP_PLATFORM_AMD__
    g_hipblaslt = g_gpu_hipblaslt[logical_tier];
    g_hipblaslt_ready = g_gpu_hipblaslt_ready[logical_tier];
#endif
    return 0;
}

/* Fenced device switch: work queued on the destination tier's default
 * stream waits for everything queued so far on the previous tier's default
 * stream. Async -- no host sync. */
extern "C" int ds4_gpu_set_current_device_fenced(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return -1;
    static cudaEvent_t fence_ev[DS4_MAX_GPUS];
    int cur_dev = -1;
    (void)cudaGetDevice(&cur_dev);
    int prev = -1;
    for (int t = 0; t < g_n_gpus; t++) {
        if (g_gpu[t].device_id == cur_dev) { prev = t; break; }
    }
    if (prev == logical_tier) return ds4_gpu_set_current_device(logical_tier);
    if (prev >= 0 && prev < g_n_gpus) {
        if (cudaSetDevice(g_gpu[prev].device_id) != cudaSuccess) return -1;
        if (!fence_ev[prev] &&
            cudaEventCreateWithFlags(&fence_ev[prev], cudaEventDisableTiming) != cudaSuccess) {
            fence_ev[prev] = NULL;
        }
        if (fence_ev[prev]) (void)cudaEventRecord(fence_ev[prev], 0);
        if (ds4_gpu_set_current_device(logical_tier) != 0) return -1;
        if (fence_ev[prev]) (void)cudaStreamWaitEvent(0, fence_ev[prev], 0);
        return 0;
    }
    return ds4_gpu_set_current_device(logical_tier);
}

extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int device_id, uint64_t bytes) {
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

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier, uint64_t bytes) {
    if (tier < 0 || tier >= g_n_gpus) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor_alloc_ptr_on: bad tier %d (n_gpus=%d)\n",
                tier, g_n_gpus);
        return NULL;
    }
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (ds4_gpu_tensor_alloc_on(t, tier, bytes) != 0) { free(t); return NULL; }
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier, uint64_t bytes) {
    if (tier < 0 || tier >= g_n_gpus) return NULL;
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    int ok = 0;
    WITH_DEVICE(g_gpu[tier].device_id) {
        ok = cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc");
    }
    if (!ok) { free(t); return NULL; }
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = tier;
    return t;
}

extern "C" void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t) {
    if (!t) return;
    int d = ds4_tensor_device_idx(t);
    if (t->owner && t->ptr) {
        WITH_DEVICE(g_gpu[d].device_id) { (void)cudaFree(t->ptr); }
    }
    memset(t, 0, sizeof(*t));
}

extern "C" int ds4_gpu_tensor_device(const ds4_gpu_tensor *t) {
    return t ? t->device_id : -1;
}

static void *rocm_mgpu_xdev_bounce_get(int src, int dst, size_t bytes) {
    if (src < 0 || src >= DS4_MAX_GPUS || dst < 0 || dst >= DS4_MAX_GPUS) return NULL;
    if (g_mgpu_xdev_bounce_bytes[src][dst] >= bytes && g_mgpu_xdev_bounce[src][dst]) {
        return g_mgpu_xdev_bounce[src][dst];
    }
    if (g_mgpu_xdev_bounce[src][dst]) (void)cudaFreeHost(g_mgpu_xdev_bounce[src][dst]);
    g_mgpu_xdev_bounce[src][dst] = NULL;
    g_mgpu_xdev_bounce_bytes[src][dst] = 0;
    if (cudaMallocHost(&g_mgpu_xdev_bounce[src][dst], bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        return NULL;
    }
    g_mgpu_xdev_bounce_bytes[src][dst] = bytes;
    return g_mgpu_xdev_bounce[src][dst];
}

/* Synchronous cross-device copy. Peer-capable + validated pairs use
 * cudaMemcpyPeer directly; everything else bounces through pinned host
 * memory. Both legs of the bounce are synchronous cudaMemcpy calls, so the
 * whole operation is synchronous relative to the calling host thread
 * (matches the sync contract of the legacy same-device ds4_gpu_tensor_copy
 * this is the cross-device counterpart to). */
extern "C" int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                                         const ds4_gpu_tensor *src,
                                         uint64_t bytes) {
    if (!dst || !src || bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0) return 1;
    const int sd = ds4_tensor_device_idx(src);
    const int dd = ds4_tensor_device_idx(dst);
    if (sd < 0 || sd >= g_n_gpus || dd < 0 || dd >= g_n_gpus) return 0;
    if (sd == dd) {
        int ok = 0;
        WITH_DEVICE(g_gpu[dd].device_id) {
            ok = cuda_ok(cudaMemcpy(dst->ptr, src->ptr, (size_t)bytes, cudaMemcpyDeviceToDevice),
                         "xdev same-device copy");
        }
        return ok;
    }
    if (g_gpu_peer_ok[sd][dd]) {
        int ok = 0;
        WITH_DEVICE(g_gpu[sd].device_id) {
            ok = cuda_ok(cudaMemcpyPeer(dst->ptr, g_gpu[dd].device_id,
                                        src->ptr, g_gpu[sd].device_id, (size_t)bytes),
                         "xdev peer copy");
        }
        return ok;
    }
    void *bounce = rocm_mgpu_xdev_bounce_get(sd, dd, (size_t)bytes);
    if (!bounce) return 0;
    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        ok = cuda_ok(cudaMemcpy(bounce, src->ptr, (size_t)bytes, cudaMemcpyDeviceToHost),
                     "xdev bounce d2h");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        ok = cuda_ok(cudaMemcpy(dst->ptr, bounce, (size_t)bytes, cudaMemcpyHostToDevice),
                     "xdev bounce h2d");
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                                 const ds4_gpu_tensor *src,
                                                 uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                 const ds4_gpu_tensor *src,
                                                 uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev3(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return (bytes0 == 0 || ds4_gpu_tensor_copy_xdev(dst0, src0, bytes0)) &&
           (bytes1 == 0 || ds4_gpu_tensor_copy_xdev(dst1, src1, bytes1)) &&
           (bytes2 == 0 || ds4_gpu_tensor_copy_xdev(dst2, src2, bytes2));
}

extern "C" int ds4_gpu_tensor_copy_xdev3_default_dst(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return ds4_gpu_tensor_copy_xdev3(dst0, src0, bytes0, dst1, src1, bytes1,
                                     dst2, src2, bytes2);
}

extern "C" int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src, int dst_tier) {
    /* No async cross-stream ordering implemented (all xdev copies above are
     * host-synchronous), so "waiting" is trivially satisfied once the copy
     * that produced src has returned. */
    if (dst_tier < 0 || dst_tier >= g_n_gpus) return 0;
    return src != NULL;
}

extern "C" int ds4_gpu_tensor_wait_xdev_default(const ds4_gpu_tensor *src, int dst_tier) {
    return ds4_gpu_tensor_wait_xdev(src, dst_tier);
}

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

static void rocm_mgpu_cleanup(void) {
    if (g_mgpu_stage_buf) {
        (void)cudaFreeHost(g_mgpu_stage_buf);
        g_mgpu_stage_buf = NULL;
    }
    for (int i = 0; i < g_n_gpus; i++) {
        ds4_rocm_gpu_ctx *c = &g_gpu[i];
        (void)cudaSetDevice(c->device_id);
        if (c->boundary_event) { (void)cudaEventDestroy((cudaEvent_t)c->boundary_event); c->boundary_event = NULL; }
        if (c->stream) { (void)cudaStreamDestroy((cudaStream_t)c->stream); c->stream = NULL; }
        if (g_gpu_cublas_ready[i]) {
            (void)cublasDestroy(g_gpu_cublas[i]);
            g_gpu_cublas_ready[i] = 0;
            g_gpu_cublas[i] = NULL;
        }
#ifdef __HIP_PLATFORM_AMD__
        if (g_gpu_hipblaslt_ready[i]) {
            (void)hipblasLtDestroy(g_gpu_hipblaslt[i]);
            g_gpu_hipblaslt_ready[i] = 0;
            g_gpu_hipblaslt[i] = NULL;
        }
#endif
        if (g_mgpu_dev_cache[i].present) {
            (void)cudaSetDevice(c->device_id);
            if (g_mgpu_dev_cache[i].base) (void)cudaFree(g_mgpu_dev_cache[i].base);
            g_mgpu_dev_cache[i].base = NULL;
            g_mgpu_dev_cache[i].bytes = 0;
            g_mgpu_dev_cache[i].present = 0;
        }
    }
    for (int i = 0; i < DS4_MAX_GPUS; i++) {
        for (int j = 0; j < DS4_MAX_GPUS; j++) {
            if (g_mgpu_xdev_bounce[i][j]) {
                (void)cudaFreeHost(g_mgpu_xdev_bounce[i][j]);
                g_mgpu_xdev_bounce[i][j] = NULL;
                g_mgpu_xdev_bounce_bytes[i][j] = 0;
            }
        }
    }
    g_mgpu_cache_ranges.clear();
    g_n_gpus = 0;
    g_current_logical_tier = -1;
    g_cublas_ready = 0;
    g_cublas = NULL;
#ifdef __HIP_PLATFORM_AMD__
    g_hipblaslt_ready = 0;
    g_hipblaslt = NULL;
#endif
}
