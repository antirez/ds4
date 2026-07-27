/* ds4_gpu_mgpu.h — multi-GPU plumbing types and APIs (v0).
 *
 * This header carries the new multi-GPU additions for the multi-GPU plumbing PP work
 * (device-aware CUDA). It is included from ds4_cuda.cu and from
 * downstream tasks that need access to g_gpu[], g_n_gpus, g_gpu_peer_ok[],
 * the ds4_gpu_config struct, and the new tensor APIs.
 *
 * Why not in ds4_gpu.h? The legacy ds4_gpu.h is included from C-only
 * callers (ds4.c, ds4_cli.c, etc.) and from the Metal build, but is NOT
 * included from ds4_cuda.cu historically. That asymmetry hid pre-existing
 * signature mismatches between the legacy header and ds4_cuda.cu. We keep
 * the legacy header opaque and put the new shared types here, so this
 * file is the single source of truth for both ds4_cuda.cu and downstream
 * multi-GPU tasks without disturbing the legacy contract.
 *
 * The struct definitions reference CUDA-specific handle types via void *
 * placeholders so the header is safe to include from C builds, Metal
 * builds, and the CUDA build (where ds4_cuda.cu casts the void * back
 * to cudaStream_t / cublasHandle_t / cudaEvent_t internally).
 */
#ifndef DS4_GPU_MGPU_H
#define DS4_GPU_MGPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_MAX_GPUS 16

/* Complete definition of the previously-opaque ds4_gpu_tensor, plus a
 * typedef so the new API prototypes below can use the bare name
 * `ds4_gpu_tensor *` in both C and C++ without forcing callers to
 * include ds4_gpu.h first. Callers that include this header can
 * stack-allocate or struct-embed tensors and pass to
 * ds4_gpu_tensor_alloc_on. */
struct ds4_gpu_tensor {
    void    *ptr;
    uint64_t bytes;
    int      owner;
    int      device_id;   /* -1 means legacy/untagged → treat as device 0 */
};
#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

#ifndef DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_BATCH_MAX 32u
typedef struct {
    uint64_t raw_kv;
    uint64_t comp_kv;
    uint64_t topk;
    uint32_t pos;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t window;
    uint32_t ratio;
    uint32_t indexed;
} ds4_gpu_attention_decode_row;
#endif

/* Tagged so headers (notably ds4.h) can forward-declare `struct
 * ds4_gpu_config` without dragging in this entire header. */
typedef struct ds4_gpu_config {
    int    device_indices[DS4_MAX_GPUS];   /* CUDA device IDs to use */
    /* Explicit per-device budget in bytes. The engine does NOT auto-fill
     * missing budgets - a value of 0 means "zero bytes of budget for
     * this slot" and (combined with reserves) will push placement to
     * CPU spill. Auto-detection (e.g. mapping --gpu-vram auto to
     * cudaMemGetInfo) is the caller's job; see CLI flag wiring for the
     * canonical CLI path. The engine emits a clear stderr and refuses
     * if n_gpus > 0 and every vram_bytes[] is 0 (almost certainly a
     * caller bug from zero-initializing the struct). */
    size_t vram_bytes[DS4_MAX_GPUS];
    int    n_gpus;
    size_t safety_margin_bytes;            /* per-device reserve */
} ds4_gpu_config;

typedef struct {
    int    device_id;
    void  *stream;             /* cudaStream_t under CUDA */
    void  *cublas;             /* cublasHandle_t under CUDA */
    int    cublas_ready;
    void  *scratch;
    size_t scratch_bytes;
    size_t budget_bytes;
    size_t used_bytes;
    void  *boundary_event;     /* cudaEvent_t under CUDA */
} ds4_gpu_ctx;

extern ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];
extern int         g_n_gpus;
extern int         g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS];

/* Primary multi-device init. The existing ds4_gpu_init (declared in
 * ds4_gpu.h) is a thin shim that builds a single-device config for
 * device 0 and calls this. */
int ds4_gpu_init_multi(const ds4_gpu_config *cfg);

/* Caller-supplied struct alloc on a specific device. Returns 0 on
 * success, nonzero on error. Pair with ds4_gpu_tensor_free_in_place. */
int  ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int device_id, uint64_t bytes);
void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t);

/* Heap-allocated tensor on a specific logical tier; mirrors the legacy
 * ds4_gpu_tensor_alloc ABI (returns ds4_gpu_tensor *) but with a tier
 * parameter. Returns NULL on failure. Used by the multi-tier graph
 * allocations in ds4.c. Single-tier callers can continue using the
 * legacy ds4_gpu_tensor_alloc(bytes) which is equivalent to
 * ds4_gpu_tensor_alloc_ptr_on(0, bytes). */
ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier, uint64_t bytes);
int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t bytes);
void ds4_gpu_enable_q8_dequant_gemm(void);

/* Heap-allocated MANAGED-memory tensor on a specific logical tier. Used
 * for the KV-cache pool. Stamps tier on the tensor so subsequent
 * ds4_gpu_tensor_free runs under the correct device. Note managed memory
 * pages on first-touch and is not strictly device-bound; the stamped
 * tier records the home tier for accounting + free. */
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier, uint64_t bytes);

/* Cross-device tensor copy. Same-device → cudaMemcpyAsync; peer-capable
 * cross-device → cudaMemcpyPeerAsync with event sync; non-peer → pinned
 * host bounce (per src→dst pair). Honors DS4_FORCE_HOST_BOUNCE=1. */
int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                              const ds4_gpu_tensor *src,
                              uint64_t bytes);
int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                      const ds4_gpu_tensor *src,
                                      uint64_t bytes);

/* Grouped default-stream handoff whose copies execute on the destination
 * device. The source default stream records readiness; the destination waits,
 * performs all three copies, then naturally orders its following kernels.
 * This allows source-device work submitted afterward to overlap the handoff. */
int ds4_gpu_tensor_copy_xdev3_default_dst(
        ds4_gpu_tensor       *dst0,
        const ds4_gpu_tensor *src0,
        uint64_t              bytes0,
        ds4_gpu_tensor       *dst1,
        const ds4_gpu_tensor *src1,
        uint64_t              bytes1,
        ds4_gpu_tensor       *dst2,
        const ds4_gpu_tensor *src2,
        uint64_t              bytes2);

/* Cross-device copy of three buffers between the same source/destination
 * tiers, recording a single readiness event for the destination stream. This
 * is meant for tiny grouped activation handoffs where three independent
 * ds4_gpu_tensor_copy_xdev calls would spend more time in event plumbing than
 * in the copies themselves. Falls back internally when a grouped peer copy is
 * not applicable. */
int ds4_gpu_tensor_copy_xdev3(ds4_gpu_tensor       *dst0,
                              const ds4_gpu_tensor *src0,
                              uint64_t              bytes0,
                              ds4_gpu_tensor       *dst1,
                              const ds4_gpu_tensor *src1,
                              uint64_t              bytes1,
                              ds4_gpu_tensor       *dst2,
                              const ds4_gpu_tensor *src2,
                              uint64_t              bytes2);

/* Cross-device copy that also orders against prior work on the destination
 * stream before overwriting dst. This is needed for pipelined prefill slots
 * where the next producer reuses a destination buffer that the destination
 * stage just consumed. */
int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                      const ds4_gpu_tensor *src,
                                      uint64_t bytes);

/* Order a peer read without copying: records readiness on src's stream and
 * makes dst_tier's stream wait for it. */
int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src, int dst_tier);
int ds4_gpu_tensor_wait_xdev_default(const ds4_gpu_tensor *src, int dst_tier);

/* Cross-device float add used by CUDA tensor-parallel reductions.
 * See ds4_gpu.h for the full contract. */
int ds4_gpu_add_xdev_tensor(ds4_gpu_tensor       *out,
                             const ds4_gpu_tensor *local,
                             const ds4_gpu_tensor *remote,
                             ds4_gpu_tensor       *remote_tmp,
                             uint32_t              n);

/* Returns the device_id recorded on the tensor; -1 if untagged. */
int ds4_gpu_tensor_device(const ds4_gpu_tensor *t);

/* Set the current CUDA device by LOGICAL tier index (0..g_n_gpus-1).
 *
 * This is the canonical shim for per-layer device routing in the
 * multi-tier execution path. The caller passes a logical tier index;
 * the shim internally indexes g_gpu[tier].device_id and calls
 * cudaSetDevice. Returns 0 on success, nonzero on error or if the
 * tier index is out of range.
 *
 * Wave-2 multi-GPU placement scaffolding adds this shim but does not
 * exercise the multi-tier execution path; multi-GPU execution
 * (follow-up) is its first caller. */
int ds4_gpu_set_current_device(int logical_tier);
int ds4_gpu_set_current_device_fenced(int logical_tier);

/* Register the mmap'd host model pointer for selective-cache lookups
 * WITHOUT triggering any device-side copy. This bypasses the
 * DS4_CUDA_COPY_MODEL environment-variable branch that
 * ds4_gpu_set_model_map normally honors, which is essential for
 * multi-tier startup: we want only per-device selective tensor caches,
 * never the whole-model copy. Returns 1 on success, 0 on error. */
int ds4_gpu_register_model_map_no_copy(const void *model_map, uint64_t model_size);

/* Strict per-device selective-cache lookup (no fallback).
 *
 * Returns 1 only if a covering entry exists whose device_id matches the
 * caller-supplied expected_device (a PHYSICAL CUDA device id). Otherwise
 * returns 0 — no host-pointer fallback, no different-device match. The
 * caller is expected to have cudaSetDevice'd to expected_device before
 * invoking; the returned pointer is valid to consume from that device's
 * kernel.
 *
 * Used by multi-tier kernel-dispatch resolvers in
 * multi-GPU execution (multi-GPU execution). Single-tier callers should
 * keep using ds4_gpu_lookup_cache for back-compat behavior. */
int ds4_gpu_lookup_cache_strict(uint64_t source_offset,
                                 uint64_t bytes,
                                 int      expected_device,
                                 void   **out_device_ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DS4_GPU_MGPU_H */
