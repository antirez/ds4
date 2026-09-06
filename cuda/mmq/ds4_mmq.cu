// SPDX-License-Identifier: MIT
// ds4_mmq.cu - host wrapper around llama.cpp's vendored mul_mat_q kernels.
//
// The fused target-prefill MoE dispatcher (ds4_mmq_fused_down,
// ds4_swiglu_weighted_f32, the fused_down branches of
// ds4_mmq_moe_pair_impl, and the ds4_mmq_iq2_xxs_q2_K_moe_fused_* entry
// points) is Portions Copyright (c) 2026 Marco Palaferri (MIT), adapted
// from xangel82/DS4-GB10-GX10-DSpark-CUDA commit 910501e (v0.5 inc-9).
//
// Implements the public ds4_mmq_* entry points and explicitly instantiates
// the mul_mat_q_case<T> template for each quant type the caller needs.
//
// Status:
//   Q8_0 dense ............ implemented, parity-tested against CPU reference
//   Q2_K dense ............ pending (Phase 3)
//   IQ2_XXS dense ......... pending (Phase 3)
//   Q8_0 MoE _id .......... pending (Phase 4)
//   Q2_K MoE _id .......... pending (Phase 4)
//   IQ2_XXS MoE _id ....... pending (Phase 4)

#include "ds4_mmq.h"
#include "ds4_q4_mmvq_epilogue.h"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "ds4_mmq_d2r.cuh"
#if !defined(GGML_USE_HIP)
#include "ds4_mmq_q4_16warp.cuh"
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DS4_MMQ_HAS_NVTX 1
#endif
#endif
#ifndef DS4_MMQ_HAS_NVTX
#define DS4_MMQ_HAS_NVTX 0
#endif

static bool ds4_mmq_nvtx_requested() {
    static int enabled = -1;
    if (enabled < 0) {
        const char *nvtx = getenv("DS4_CUDA_NVTX");
        const char *capture = getenv("DS4_CUDA_NSYS_PREFILL_START_POS");
        enabled = (nvtx != nullptr && std::strcmp(nvtx, "1") == 0) ||
                  (capture != nullptr && capture[0] != '\0');
    }
    return enabled != 0;
}

static bool ds4_mmq_gfx1151_flag(const char *name, int cc) {
    const char *env = getenv(name);
    return env ? env[0] != '0' : cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151;
}

static uint64_t ds4_mmq_nvtx_payload(uint32_t first, uint32_t second) {
    return ((uint64_t)first << 32) | second;
}

class ds4_mmq_nvtx_scope {
public:
    ds4_mmq_nvtx_scope(const char *name, uint64_t payload, bool enabled)
        : active_(enabled) {
#if DS4_MMQ_HAS_NVTX
        if (active_) {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
            attr.payload.ullValue = payload;
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = name;
            (void)nvtxRangePushEx(&attr);
        }
#else
        (void)name;
        (void)payload;
        active_ = false;
#endif
    }

    ~ds4_mmq_nvtx_scope() {
#if DS4_MMQ_HAS_NVTX
        if (active_) (void)nvtxRangePop();
#endif
    }

    ds4_mmq_nvtx_scope(const ds4_mmq_nvtx_scope &) = delete;
    ds4_mmq_nvtx_scope &operator=(const ds4_mmq_nvtx_scope &) = delete;

private:
    bool active_;
};

// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------

// Experimental persistent Q8_1 scratch.  The grouped raw prefill path below
// aliases its input and down-Q8 staging in this arena: both ranges have the
// same layout but disjoint lifetimes on the default stream.  The feature stays
// opt-in because a process-global address is only safe for the single-owner
// GB10 dispatch covered by q81_grouped_persistent_acquire().
static void *g_q81_scratch_ptr   = nullptr;
static size_t g_q81_scratch_bytes = 0;
// A failed resize retirement can leave both allocations live.  Keep the
// unpublished replacement owned here so cleanup/reinit can retry its free.
static void *g_q81_unpublished_replacement_ptr = nullptr;
// Older vector wrappers still contain a generic persistent branch.  Keep that
// branch disabled: unlike grouped fused_raw it has no default-stream lease or
// capture exclusion and therefore cannot safely share the owned arena.
static constexpr bool g_q81_scratch_enabled = false;
static bool   g_q81_grouped_enabled = false;
static bool   g_q81_scratch_poisoned = false;
static int    g_q81_scratch_device = -1;
static std::mutex g_q81_state_mutex; // Process-global state, not per-tensor.

static uint64_t g_q81_grouped_candidates;
static uint64_t g_q81_grouped_uses;
static uint64_t g_q81_grouped_hits;
static uint64_t g_q81_grouped_pool_fallbacks;
static uint64_t g_q81_grouped_allocations;
static uint64_t g_q81_grouped_resizes;
static uint64_t g_q81_grouped_owner_rejects;
static uint64_t g_q81_grouped_stream_rejects;
static uint64_t g_q81_grouped_capture_rejects;
static uint64_t g_q81_grouped_device_rejects;
static uint64_t g_q81_grouped_size_rejects;
static size_t   g_q81_grouped_high_water;

static void  *g_aligned_q81_scratch_ptr = nullptr;
static size_t g_aligned_q81_scratch_bytes = 0;
static int    g_aligned_q81_scratch_device = -1;

// The gfx1151 IQ2 pair path is fed in tiles of at most 2048 tokens with six
// routed experts per token. Keep its small routing maps out of the ROCm async
// pool: repeated shape churn can recycle/remap those allocations while the
// following quantize kernel still consumes them. One plain allocation per
// device gives the maps a stable lifetime for the full MMQ session.
static constexpr size_t MMQ_GFX1151_PAIR_MAP_ROWS = 2048u * 6u;
static constexpr size_t MMQ_GFX1151_PAIR_MAP_EXPERTS = 256u;

struct mmq_pair_map_scratch {
    int32_t *base = nullptr;
    int32_t *ids_src1 = nullptr;
    int32_t *ids_dst = nullptr;
    int32_t *expert_bounds = nullptr;
};

static mmq_pair_map_scratch g_mmq_pair_maps[GGML_CUDA_MAX_DEVICES] = {};

// Backend init/teardown owns transitions, but dispatch admission reads this
// flag without taking the Q8_1 arena mutex. Keep those reads race-free while
// retaining the mutex below for arena ownership and cleanup serialization.
static std::atomic<bool> g_gb10_optimizations{false};

static bool gb10_optimizations_enabled() {
    return g_gb10_optimizations.load(std::memory_order_relaxed);
}

static constexpr size_t DS4_MMQ_Q81_ARENA_MIN_BYTES = 4u * 1024u * 1024u;

enum q81_arena_result {
    Q81_ARENA_REJECTED = 0,
    Q81_ARENA_HIT,
    Q81_ARENA_ALLOCATED,
    Q81_ARENA_RESIZED,
};

static bool q81_persistent_requested() {
    const char *value = getenv("DS4_CUDA_MMQ_Q81_PERSISTENT");
    if (!value || value[0] == '\0' || strcmp(value, "0") == 0 ||
        strcmp(value, "off") == 0 || strcmp(value, "OFF") == 0 ||
        strcmp(value, "false") == 0 || strcmp(value, "FALSE") == 0 ||
        strcmp(value, "no") == 0 || strcmp(value, "NO") == 0) {
        return false;
    }
    return strcmp(value, "1") == 0 ||
           strcmp(value, "on") == 0 || strcmp(value, "ON") == 0 ||
           strcmp(value, "true") == 0 || strcmp(value, "TRUE") == 0 ||
           strcmp(value, "yes") == 0 || strcmp(value, "YES") == 0;
}

static bool q81_is_gb10_owner(int device) {
    if (!gb10_optimizations_enabled() || device < 0 ||
        device >= ggml_cuda_info().device_count) {
        return false;
    }
    const auto &info = ggml_cuda_info().devices[device];
    return info.integrated && info.cc == GGML_CUDA_CC_DGX_SPARK;
}

static q81_arena_result q81_arena_ensure_locked(int device, size_t required) {
    if (g_q81_scratch_poisoned || required > SIZE_MAX - 255u ||
        (g_q81_scratch_ptr && g_q81_scratch_device != device)) {
        return Q81_ARENA_REJECTED;
    }
    if (g_q81_scratch_ptr && required <= g_q81_scratch_bytes) {
        g_q81_grouped_enabled = true;
        return Q81_ARENA_HIT;
    }

    const size_t aligned = (required + 255u) & ~(size_t)255u;
    const size_t bytes = aligned > DS4_MMQ_Q81_ARENA_MIN_BYTES
        ? aligned : DS4_MMQ_Q81_ARENA_MIN_BYTES;
    void *replacement = nullptr;
    cudaError_t err = cudaMalloc(&replacement, bytes);
    if (err != cudaSuccess || !replacement) {
        fprintf(stderr,
                "ds4_mmq: cudaMalloc(persistent Q8_1 arena %zu B) failed: %s; "
                "using stream pool\n",
                bytes, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return Q81_ARENA_REJECTED;
    }

    if (g_q81_scratch_ptr) {
        /* The lease excludes another host submission and the persistent path
         * only accepts the legacy default stream.  Drain the device before
         * retiring the old address: allocation succeeds before any state is
         * changed, so OOM leaves the previous arena usable.  A drain/free
         * failure is different -- pointer liveness is ambiguous and the
         * feature stays poisoned until explicit cleanup. */
        err = cudaDeviceSynchronize();
        if (err == cudaSuccess) {
            err = cudaFree(g_q81_scratch_ptr);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4_mmq: persistent Q8_1 arena resize from %zu B to "
                    "%zu B failed while retiring the old arena: %s; "
                    "persistent reuse poisoned\n",
                    g_q81_scratch_bytes, bytes, cudaGetErrorString(err));
            (void)cudaGetLastError();
            const cudaError_t replacement_free_err = cudaFree(replacement);
            if (replacement_free_err != cudaSuccess) {
                fprintf(stderr,
                        "ds4_mmq: freeing unpublished Q8_1 replacement "
                        "failed: %s\n",
                        cudaGetErrorString(replacement_free_err));
                (void)cudaGetLastError();
                g_q81_unpublished_replacement_ptr = replacement;
            }
            g_q81_grouped_enabled = false;
            g_q81_scratch_poisoned = true;
            return Q81_ARENA_REJECTED;
        }
    }

    const bool resized = g_q81_scratch_ptr != nullptr;
    g_q81_scratch_ptr = replacement;
    g_q81_scratch_bytes = bytes;
    g_q81_scratch_device = device;
    g_q81_grouped_enabled = true;
    g_q81_grouped_allocations++;
    if (resized) g_q81_grouped_resizes++;
    fprintf(stderr,
            "ds4_mmq: persistent Q8_1 arena %s (%zu B at %p, device %d)\n",
            resized ? "resized" : "enabled", bytes,
            g_q81_scratch_ptr, device);
    return resized ? Q81_ARENA_RESIZED : Q81_ARENA_ALLOCATED;
}

// On success, lease remains locked until the complete host dispatch has been
// submitted.  A following dispatch therefore cannot interleave its writes;
// default-stream ordering protects the device-side lifetime after unlock.
static char *q81_grouped_persistent_acquire(
        int device, cudaStream_t stream, size_t required,
        std::unique_lock<std::mutex> *lease) {
    if (!lease || !q81_persistent_requested()) return nullptr;
    lease->lock();
    g_q81_grouped_candidates++;
    if (required > g_q81_grouped_high_water) {
        g_q81_grouped_high_water = required;
    }
    if (!q81_is_gb10_owner(device)) {
        g_q81_grouped_device_rejects++;
    } else if (stream != (cudaStream_t)0) {
        g_q81_grouped_stream_rejects++;
    } else {
        int active_device = -1;
        const cudaError_t device_err = cudaGetDevice(&active_device);
        if (device_err != cudaSuccess || active_device != device ||
            (g_q81_scratch_device >= 0 &&
             g_q81_scratch_device != device)) {
            (void)cudaGetLastError();
            g_q81_grouped_owner_rejects++;
        } else {
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            const cudaError_t capture_err =
                cudaStreamIsCapturing(stream, &capture);
            if (capture_err != cudaSuccess ||
                capture != cudaStreamCaptureStatusNone) {
                (void)cudaGetLastError();
                g_q81_grouped_capture_rejects++;
            } else {
                const q81_arena_result arena =
                    q81_arena_ensure_locked(device, required);
                if (arena != Q81_ARENA_REJECTED &&
                    g_q81_grouped_enabled && !g_q81_scratch_poisoned &&
                    required <= g_q81_scratch_bytes) {
                    g_q81_grouped_uses++;
                    if (arena == Q81_ARENA_HIT) g_q81_grouped_hits++;
                    return (char *)g_q81_scratch_ptr;
                }
                if (g_q81_scratch_ptr && required > g_q81_scratch_bytes) {
                    g_q81_grouped_size_rejects++;
                }
            }
        }
    }
    g_q81_grouped_pool_fallbacks++;
    lease->unlock();
    return nullptr;
}

extern "C" void ds4_mmq_set_aligned_q81_scratch(void *ptr, size_t bytes) {
    g_aligned_q81_scratch_ptr = ptr;
    g_aligned_q81_scratch_bytes = ptr ? bytes : 0;
    g_aligned_q81_scratch_device = -1;
    if (ptr) (void)cudaGetDevice(&g_aligned_q81_scratch_device);
}

static void *ds4_mmq_aligned_q81_scratch(int device, size_t bytes) {
    return g_aligned_q81_scratch_ptr &&
           g_aligned_q81_scratch_device == device &&
           g_aligned_q81_scratch_bytes >= bytes
        ? g_aligned_q81_scratch_ptr : nullptr;
}

/* Test-only preflight hook.  It deliberately traverses the production
 * acquire path (including owner/default-stream/capture checks and resize
 * retirement) without enqueueing a synthetic large MMQ fixture. */
extern "C" int ds4_mmq_q81_persistent_preflight_for_test(
        int device, size_t required) {
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess ||
        cudaSetDevice(device) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    char *arena = nullptr;
    {
        std::unique_lock<std::mutex> lease(
            g_q81_state_mutex, std::defer_lock);
        arena = q81_grouped_persistent_acquire(
            device, (cudaStream_t)0, required, &lease);
    }
    const cudaError_t restore_err = previous != device
        ? cudaSetDevice(previous) : cudaSuccess;
    if (restore_err != cudaSuccess) {
        (void)cudaGetLastError();
        return -2;
    }
    return arena ? 0 : -3;
}

static uint64_t g_q8_fold_oracle_byte_calls;
static uint64_t g_q8_fold_oracle_byte_mismatches;
static uint64_t g_q8_fold_oracle_output_calls;
static uint64_t g_q8_fold_oracle_output_mismatches;
static uint64_t g_q8_fold_oracle_raw_moe_calls;
static uint64_t g_q8_fold_oracle_aligned_q8_calls;
static uint64_t g_q8_fold_oracle_aligned_iq2_calls;
static uint64_t g_q8_fold_oracle_skips;
static int g_q8_fold_oracle_report_registered;

static cudaError_t ds4_mmq_q8_fold_oracle_free(
        void *ptr, const char *label, cudaError_t prior_err) {
    if (!ptr) return prior_err;
    const cudaError_t free_err = cudaFree(ptr);
    if (free_err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA Q8_1 fold oracle cudaFree(%s) failed: %s\n",
                label, cudaGetErrorString(free_err));
        if (prior_err == cudaSuccess) return free_err;
    }
    return prior_err;
}

static void ds4_mmq_q8_fold_oracle_report(void) {
    fprintf(stderr,
            "ds4: CUDA Q8_1 fold oracle: byte_calls=%llu "
            "byte_mismatches=%llu output_calls=%llu "
            "output_mismatches=%llu raw_moe_calls=%llu "
            "aligned_q8_calls=%llu aligned_iq2_calls=%llu skips=%llu "
            "(canonical reference retained)\n",
            (unsigned long long)g_q8_fold_oracle_byte_calls,
            (unsigned long long)g_q8_fold_oracle_byte_mismatches,
            (unsigned long long)g_q8_fold_oracle_output_calls,
            (unsigned long long)g_q8_fold_oracle_output_mismatches,
            (unsigned long long)g_q8_fold_oracle_raw_moe_calls,
            (unsigned long long)g_q8_fold_oracle_aligned_q8_calls,
            (unsigned long long)g_q8_fold_oracle_aligned_iq2_calls,
            (unsigned long long)g_q8_fold_oracle_skips);
}

static bool ds4_mmq_q8_fold_oracle_enabled() {
    const char *env = getenv("DS4_CUDA_Q8_FOLD_ORACLE");
    const bool enabled = env && strcmp(env, "1") == 0;
    if (enabled && !g_q8_fold_oracle_report_registered) {
        g_q8_fold_oracle_report_registered = 1;
        (void)atexit(ds4_mmq_q8_fold_oracle_report);
    }
    return enabled;
}

__global__ static void q8_fold_output_compare_kernel(
        uint32_t *mismatch, const float *candidate,
        const float *reference, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && __float_as_uint(candidate[i]) !=
                 __float_as_uint(reference[i])) {
        atomicExch(mismatch, 1u);
    }
}

/* Byte oracle for every consumer.  On mismatch the fresh canonical bytes
 * overwrite the sidecar before it is consumed.  Any setup/capture failure
 * rejects the fold entirely so the established prelude quantizes again. */
static bool ds4_mmq_q8_fold_oracle_bytes(
        const float *X_f32, int64_t K, int64_t ne10_padded,
        char *folded, cudaStream_t stream) {
    if (!ds4_mmq_q8_fold_oracle_enabled()) return true;
    if (!X_f32 || !folded || K <= 0 || ne10_padded != K ||
        (K % QK8_1) != 0) {
        g_q8_fold_oracle_skips++;
        return false;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const cudaError_t capture_err = cudaStreamIsCapturing(stream, &capture);
    if (capture_err != cudaSuccess) {
        fprintf(stderr,
                "ds4_mmq: Q8 fold oracle stream-capture query failed: %s\n",
                cudaGetErrorString(capture_err));
        (void)cudaGetLastError();
        g_q8_fold_oracle_skips++;
        return false;
    }
    if (capture != cudaStreamCaptureStatusNone) {
        fprintf(stderr,
                "ds4_mmq: Q8 fold oracle skipped during stream capture\n");
        g_q8_fold_oracle_skips++;
        return false;
    }
    const size_t bytes = (size_t)ne10_padded * sizeof(block_q8_1) / QK8_1;
    if (bytes == 0u || bytes > 16384u) {
        g_q8_fold_oracle_skips++;
        return false;
    }
    char *fresh = nullptr;
    char *host = (char *)malloc(bytes * 2u);
    if (!host) {
        fprintf(stderr,
                "ds4_mmq: Q8 fold oracle host allocation (%zu B) failed\n",
                bytes * 2u);
        g_q8_fold_oracle_skips++;
        return false;
    }
    const cudaError_t fresh_alloc_err = cudaMalloc((void **)&fresh, bytes);
    if (fresh_alloc_err != cudaSuccess || !fresh) {
        fprintf(stderr,
                "ds4_mmq: Q8 fold oracle device allocation (%zu B) failed: "
                "%s%s\n",
                bytes, cudaGetErrorString(fresh_alloc_err),
                fresh_alloc_err == cudaSuccess ? " (null pointer)" : "");
        free(host);
        (void)cudaGetLastError();
        g_q8_fold_oracle_skips++;
        return false;
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, fresh, GGML_TYPE_Q8_0,
        /*ne00=*/K, /*s11=*/K, /*s12=*/K, /*s13=*/K,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/1, /*ne3=*/1,
        stream);
    bool setup_ok = cudaGetLastError() == cudaSuccess &&
                    cudaStreamSynchronize(stream) == cudaSuccess &&
                    cudaMemcpy(host, folded, bytes,
                               cudaMemcpyDeviceToHost) == cudaSuccess &&
                    cudaMemcpy(host + bytes, fresh, bytes,
                               cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!setup_ok) {
        (void)cudaGetLastError();
        cudaError_t cleanup_err = ds4_mmq_q8_fold_oracle_free(
            fresh, "byte-fresh", cudaSuccess);
        if (cleanup_err != cudaSuccess) (void)cudaGetLastError();
        free(host);
        g_q8_fold_oracle_skips++;
        return false;
    }
    const bool match = memcmp(host, host + bytes, bytes) == 0;
    g_q8_fold_oracle_byte_calls++;
    if (!match) {
        g_q8_fold_oracle_byte_mismatches++;
        if (cudaMemcpyAsync(folded, fresh, bytes,
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            (void)cudaGetLastError();
            cudaError_t cleanup_err = ds4_mmq_q8_fold_oracle_free(
                fresh, "byte-fresh", cudaSuccess);
            if (cleanup_err != cudaSuccess) (void)cudaGetLastError();
            free(host);
            g_q8_fold_oracle_skips++;
            return false;
        }
    }
    cudaError_t cleanup_err = ds4_mmq_q8_fold_oracle_free(
        fresh, "byte-fresh", cudaSuccess);
    free(host);
    if (cleanup_err != cudaSuccess) {
        (void)cudaGetLastError();
        g_q8_fold_oracle_skips++;
        return false;
    }
    return true;
}

static char *ds4_mmq_folded_q81(const float *X_f32, int64_t K, int n_tokens,
                                int64_t ne10_padded, cudaStream_t stream) {
    if (n_tokens != 1 || ne10_padded != K) return nullptr;
    const void *p = nullptr;
    if (!ds4_cuda_q8_fold_take_q81(
            (const void *)X_f32, (uint64_t)K, stream, &p)) return nullptr;
    if (!ds4_mmq_q8_fold_oracle_bytes(
            X_f32, K, ne10_padded, (char *)(uintptr_t)p, stream)) {
        return nullptr;
    }
    static int logged = 0;
    if (!logged) {
        logged = 1;
        fprintf(stderr, "ds4: M2-Inc2a q8_1 activation fold active (mmvq decode)\n");
    }
    return (char *)(uintptr_t)p;
}

// Default ON (2026-07-09 gated increment: same-boot ABBA 427->493 tok/s @12k,
// gsm8k 97.5 / mbpp 90). DS4_MMQ_D2R=0 is the kill switch back to the
// mul_mat_q SoA-tile down path.
static bool d2r_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_D2R");
        cached = (env && env[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool d2r_iq2_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_D2R_IQ2");
        cached = (env && env[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

// Blanket output zeroing on the dense/MoE-down/pair GEMM entries.  Added by
// 82b2622 as belt-and-suspenders while root-causing the cont BOS spam; the
// actual roots were fixed in the same commit (stream-K fixup write_back goes
// dense + tmp_fixup zeroed + ncols_max=ne_get_rows), after which every
// element a consumer reads is stored by the GEMM itself and the zeroing was
// ~1.0 s/12k-admission of pure memset tax.  Default OFF (2026-07-09 gated
// increment: L42 deep tensors BIT-IDENTICAL with/without, same-boot ABBA
// 641.5 -> 678 tok/s @12k, gsm8k 119/120 / mbpp 36/40 / canary=[]).
// DS4_MMQ_OUT_MEMSET=1 restores the zeroing.
static bool out_memset_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_OUT_MEMSET");
        cached = (env && env[0] == '1') ? 1 : 0;
        if (cached) {
            fprintf(stderr, "ds4: DS4_MMQ_OUT_MEMSET=1 - blanket GEMM output zeroing restored\n");
        }
    }
    return cached != 0;
}

/* v0.5 inc-12 slice 2: Y-buffer (q8_1 activation) memset diet.  The S1.1a-era
 * zero of every quantize staging buffer before quantize_mmq_q8_1 cost ~2.3
 * s/180k of stream time (reslice10 MEMSET table: 56.6/28.3/18.9/9.45/4.7 MB
 * classes = the gateup/down/o_proj/dense/shexp Y buffers).  quantize writes
 * every valid column; only the never-written pad/slack tail is at stake, and
 * the mmq write_back masks tail lanes out of the output (the D2R kernels
 * guard their token loops outright).  Modes, same contract as the cublas ws
 * knob:
 *   DS4_MMQ_YBUF_MEMSET unset/0 -> no zero (default; bit-exact IFF no tail
 *     byte can reach an output, proven by the poison gate)
 *   =1      -> S1.1a always-zero (the old behavior)
 *   =poison -> fill 0xFF: the bit-exactness instrument.  Exact twins vs
 *     always-zero across the gate battery prove the masking claim; any
 *     drift means some path DOES leak tail bytes and OFF must not ship. */
static int ybuf_memset_mode() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_YBUF_MEMSET");
        cached = 0;
        if (env && env[0] == '1') cached = 1;
        else if (env && (env[0] == 'p' || env[0] == 'P')) cached = 2;
        if (cached) {
            fprintf(stderr, "ds4: DS4_MMQ_YBUF_MEMSET=%s - q8_1 staging %s\n",
                    cached == 1 ? "1" : "poison",
                    cached == 1 ? "zeroing restored" : "poisoned (0xFF)");
        }
    }
    return cached;
}

static void ybuf_memset(void *ptr, size_t bytes, cudaStream_t stream) {
    const int mode = ybuf_memset_mode();
    if (mode == 0 || ptr == NULL || bytes == 0) return;
    (void)cudaMemsetAsync(ptr, mode == 1 ? 0 : 0xFF, bytes, stream);
}

/* flat-pool p5b: the direct fused gate/up path stages its input Q8 through
 * the ids_src1 column->token map inside the D2R kernel, so the activation
 * quantize runs once per TOKEN (n_tokens rows) instead of once per
 * assignment slot (n_tokens * top-k rows and bytes).  Bit-identical by
 * construction: the quantize is row-local, so a gathered slot for token t
 * holds exactly the bytes of compact row t; the kernel consumes the same
 * blocks in the same tile order through the indirection.  DS4_MMQ_NO_YIND
 * restores the slot-gathered quantize; DS4_MMQ_YIND_VERIFY byte-compares
 * the two buffers in situ (expect bad=0). */
static int moe_yind_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = getenv("DS4_MMQ_NO_YIND") == NULL ? 1 : 0;
        if (!cached) {
            fprintf(stderr, "ds4: DS4_MMQ_NO_YIND - moe gate/up y-indirect staging disabled\n");
        }
    }
    return cached;
}

static int moe_yind_verify_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = getenv("DS4_MMQ_YIND_VERIFY") != NULL ? 1 : 0;
    }
    return cached;
}

static int64_t d2r_min_cols() {
    static int64_t cached = -1;
    if (cached < 0) {
        cached = 1024;
        const char *env = getenv("DS4_MMQ_D2R_MIN_COLS");
        if (env && env[0] != '\0') {
            char *end = nullptr;
            const long v = strtol(env, &end, 10);
            if (end != env && v > 0) {
                cached = (int64_t)v;
            }
        }
    }
    return cached;
}

extern "C" int ds4_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "ds4_mmq_init: invalid device %d\n", device);
        return -1;
    }
    ggml_cuda_set_device(device);
    // Trigger lazy population of the device-info singleton.
    const auto & info = ggml_cuda_info();
    if (info.device_count == 0) {
        fprintf(stderr, "ds4_mmq_init: no CUDA devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "ds4_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }

    if (info.devices[device].cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151 &&
        !g_mmq_pair_maps[device].base) {
        constexpr size_t map_ints =
            2u * MMQ_GFX1151_PAIR_MAP_ROWS + MMQ_GFX1151_PAIR_MAP_EXPERTS + 1u;
        int32_t *base = nullptr;
        const cudaError_t err = cudaMalloc((void **)&base, map_ints * sizeof(int32_t));
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4_mmq_init: cudaMalloc(gfx1151 pair maps %zu B) failed: %s\n",
                    map_ints * sizeof(int32_t), cudaGetErrorString(err));
            return -1;
        }
        auto & maps = g_mmq_pair_maps[device];
        maps.base = base;
        maps.ids_src1 = base;
        maps.ids_dst = base + MMQ_GFX1151_PAIR_MAP_ROWS;
        maps.expert_bounds = base + 2u * MMQ_GFX1151_PAIR_MAP_ROWS;
    }

    // Allocation is intentionally lazy.  The first eligible grouped dispatch
    // knows its exact maximum input/down staging requirement and resolves the
    // arena during preflight, before it submits any device operation.
    {
        std::lock_guard<std::mutex> lock(g_q81_state_mutex);
        const bool requested = q81_persistent_requested();
        g_q81_grouped_enabled = requested && q81_is_gb10_owner(device) &&
            g_q81_scratch_ptr && !g_q81_scratch_poisoned &&
            g_q81_scratch_device == device;
    }
    return 0;
}

extern "C" int ds4_mmq_q81_persistent_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_q81_state_mutex);
    g_q81_grouped_enabled = false;
    if (!g_q81_scratch_ptr && !g_q81_unpublished_replacement_ptr) {
        g_q81_scratch_bytes = 0;
        g_q81_scratch_device = -1;
        g_q81_scratch_poisoned = false;
        return 0;
    }

    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess ||
        g_q81_scratch_device < 0 ||
        cudaSetDevice(g_q81_scratch_device) != cudaSuccess) {
        (void)cudaGetLastError();
        g_q81_scratch_poisoned = true;
        return -1;
    }
    cudaError_t sync_err = cudaDeviceSynchronize();
    cudaError_t arena_free_err = cudaSuccess;
    cudaError_t replacement_free_err = cudaSuccess;
    if (sync_err == cudaSuccess) {
        if (g_q81_scratch_ptr) {
            arena_free_err = cudaFree(g_q81_scratch_ptr);
            if (arena_free_err == cudaSuccess) {
                g_q81_scratch_ptr = nullptr;
                g_q81_scratch_bytes = 0;
            } else {
                (void)cudaGetLastError();
            }
        }
        if (g_q81_unpublished_replacement_ptr) {
            replacement_free_err = cudaFree(
                g_q81_unpublished_replacement_ptr);
            if (replacement_free_err == cudaSuccess) {
                g_q81_unpublished_replacement_ptr = nullptr;
            } else {
                (void)cudaGetLastError();
            }
        }
    } else {
        (void)cudaGetLastError();
    }
    const cudaError_t restore_err = previous != g_q81_scratch_device
        ? cudaSetDevice(previous) : cudaSuccess;
    if (sync_err != cudaSuccess || arena_free_err != cudaSuccess ||
        replacement_free_err != cudaSuccess) {
        fprintf(stderr,
                "ds4_mmq: persistent Q8_1 arena cleanup failed: "
                "sync=%s arena_free=%s replacement_free=%s\n",
                cudaGetErrorString(sync_err),
                cudaGetErrorString(arena_free_err),
                cudaGetErrorString(replacement_free_err));
        g_q81_scratch_poisoned = true;
        return -1;
    }
    // Both owned allocations are now retired; only now may reinit clear the
    // poison and admit a new lazy allocation.
    g_q81_scratch_device = -1;
    g_q81_scratch_poisoned = false;
    if (restore_err != cudaSuccess) {
        fprintf(stderr,
                "ds4_mmq: persistent Q8_1 arena freed, but restoring CUDA "
                "device %d failed: %s\n",
                previous, cudaGetErrorString(restore_err));
        (void)cudaGetLastError();
        return -2;
    }
    return 0;
}

extern "C" void ds4_mmq_q81_persistent_counters(
        uint64_t *candidates, uint64_t *uses, uint64_t *hits,
        uint64_t *pool_fallbacks, uint64_t *allocations, uint64_t *resizes,
        size_t *arena_bytes, size_t *high_water) {
    std::lock_guard<std::mutex> lock(g_q81_state_mutex);
    if (candidates) *candidates = g_q81_grouped_candidates;
    if (uses) *uses = g_q81_grouped_uses;
    if (hits) *hits = g_q81_grouped_hits;
    if (pool_fallbacks) *pool_fallbacks = g_q81_grouped_pool_fallbacks;
    if (allocations) *allocations = g_q81_grouped_allocations;
    if (resizes) *resizes = g_q81_grouped_resizes;
    if (arena_bytes) *arena_bytes = g_q81_scratch_bytes;
    if (high_water) *high_water = g_q81_grouped_high_water;
}

extern "C" void ds4_mmq_q81_persistent_report(void) {
    std::lock_guard<std::mutex> lock(g_q81_state_mutex);
    fprintf(stderr,
            "ds4: CUDA MMQ grouped Q8_1 persistent: candidates=%llu "
            "uses=%llu hits=%llu pool_fallbacks=%llu allocations=%llu "
            "resizes=%llu "
            "arena=%zu high_water=%zu rejects(device/owner/stream/capture/size)="
            "%llu/%llu/%llu/%llu/%llu poisoned=%d\n",
            (unsigned long long)g_q81_grouped_candidates,
            (unsigned long long)g_q81_grouped_uses,
            (unsigned long long)g_q81_grouped_hits,
            (unsigned long long)g_q81_grouped_pool_fallbacks,
            (unsigned long long)g_q81_grouped_allocations,
            (unsigned long long)g_q81_grouped_resizes,
            g_q81_scratch_bytes, g_q81_grouped_high_water,
            (unsigned long long)g_q81_grouped_device_rejects,
            (unsigned long long)g_q81_grouped_owner_rejects,
            (unsigned long long)g_q81_grouped_stream_rejects,
            (unsigned long long)g_q81_grouped_capture_rejects,
            (unsigned long long)g_q81_grouped_size_rejects,
            g_q81_scratch_poisoned ? 1 : 0);
}

// ----------------------------------------------------------------------------
// Gating: when should the caller choose mmq over dequant+cublas?
//
// Body lifted verbatim from llama.cpp's ggml/src/ggml-cuda/mmq.cu:267-372
// (we do not vendor mmq.cu itself, since its other half talks to ggml_tensor
// and ggml_backend internals we don't carry over).
// ----------------------------------------------------------------------------

static bool ds4_should_use_mmq_impl(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(type); GGML_UNUSED(cc); GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return false;
#endif

    bool mmq_supported;
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }
    if (!mmq_supported) return false;

    if (turing_mma_available(cc)) {
        return true;
    }
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return true;
#endif

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }
    if (amd_mfma_available(cc)) {
        if (GGML_CUDA_CC_IS_CDNA3(cc)) return true;
        if (n_experts > 64 || ne11 <= 128) return true;
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
            type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) return true;
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) return true;
        return false;
    }
    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            if (n_experts >= 64) return true;
            switch (type) {
                case GGML_TYPE_Q2_K: return ne11 <= 128;
                case GGML_TYPE_Q6_K: return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default: return true;
            }
        }
        return true;
    }
    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

extern "C" int ds4_mmq_should_use(int type_x, int64_t ne11, int64_t n_experts) {
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    const enum ggml_type t = (enum ggml_type) type_x;
    return ds4_should_use_mmq_impl(t, cc, ne11, n_experts) ? 1 : 0;
}

// ----------------------------------------------------------------------------
// Dense matmul implementation, shared across all three quant types.
//
// Computes  out[col, row] = sum_k W[row, k] * X[k, col]   with W in the
// type-specific block layout and X / out in F32 (X K-innermost row-major,
// out column-major out[col*M + row]).
//
// Mirrors upstream mmq.cu:154-159 (the no-ids branch) but builds mmq_args
// from plain pointers + shape ints instead of ggml_tensor introspection.
// ----------------------------------------------------------------------------

// Per-device singleton context. Owns the pool for stream-K fixup scratch.
// Phase 4 will make this per-stream as well; for now a single context per
// device is sufficient for the dense path.
namespace {

__global__ static void ds4_mmq_sanitize_f32_kernel(float *p, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = p[i];
    if (!isfinite(v)) p[i] = 0.0f;
}

static void ds4_mmq_sanitize_f32(float *p, uint64_t n, cudaStream_t stream) {
    if (!p || n == 0) return;
    ds4_mmq_sanitize_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, stream>>>(p, n);
}

ggml_backend_cuda_context * get_ctx_for_device(int device) {
    static ggml_backend_cuda_context * cached[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = new ggml_backend_cuda_context(device);
    }
    return cached[device];
}

template <ggml_type type>
bool ds4_mmq_k_tile_supported(const char *tag, int K, int cc) {
    if constexpr (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) {
        if (blackwell_mma_available(cc) && K % MMQ_ITER_K_FP4 != 0) {
            fprintf(stderr,
                    "%s: Blackwell FP4 K=%d must be a multiple of %d\n",
                    tag, K, MMQ_ITER_K_FP4);
            return false;
        }
    }
    return true;
}

#if !defined(GGML_USE_HIP)
static bool ds4_q4_test_q8_1_layout(
        int N, int K, size_t *payload_bytes, size_t *total_bytes) {
    if (N <= 0 || K <= 0 || (K % QK_K) != 0) return false;
    const int64_t padded_k = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t blocks_per_column =
        (size_t)padded_k / (4u * (size_t)QK8_1);
    if ((size_t)N > SIZE_MAX / blocks_per_column) return false;
    const size_t blocks = (size_t)N * blocks_per_column;
    if (blocks > SIZE_MAX / sizeof(block_q8_1_mmq)) return false;
    const size_t payload = blocks * sizeof(block_q8_1_mmq);
    const size_t slack = 128u * sizeof(block_q8_1_mmq);
    if (payload > SIZE_MAX - slack) return false;
    if (payload_bytes) *payload_bytes = payload;
    if (total_bytes) *total_bytes = payload + slack;
    return true;
}

/* The candidate and its caller-owned fixup buffer model canonical m128n128.
 * Confirm the real canonical picker chooses that tile: a width limit alone is
 * insufficient because resource constraints or a ceil-division plateau can
 * retain a narrower width and change both partitioning and scratch size. */
static bool ds4_q4_test_reference_uses_m128n128(
        int device, int cc, int N) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || N <= 0 ||
        get_mmq_y_host(cc) != 128) {
        return false;
    }
    const size_t smpbo = ggml_cuda_info().devices[device].smpbo;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const int nwarps = mmq_get_nwarps_host(cc, warp_size);
    const int mmq_x_max = get_mmq_x_max_host(cc);
    int mmq_x_best = 0;
    int64_t ntiles_x_best = INT64_MAX;
    for (int mmq_x = 8;
         mmq_x <= mmq_x_max && ntiles_x_best > 1;
         mmq_x += 8) {
        const int granularity = mmq_get_granularity_host(mmq_x, cc);
        if (mmq_x % granularity != 0 ||
            mmq_get_nbytes_shared<GGML_TYPE_Q4_K>(
                mmq_x, 128, cc, warp_size, nwarps) > smpbo) {
            continue;
        }
        const int64_t ntiles_x =
            ((int64_t)N + mmq_x - 1) / mmq_x;
        if (ntiles_x < ntiles_x_best) {
            mmq_x_best = mmq_x;
            ntiles_x_best = ntiles_x;
        }
    }
    return mmq_x_best == 128;
}

extern "C" int
ds4_mmq_q4_K_dense_preq_reference_m128n128_for_test(int N) {
    const int dev = ggml_cuda_get_device();
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) return 0;
    return ds4_q4_test_reference_uses_m128n128(
        dev, ggml_cuda_info().devices[dev].cc, N) ? 1 : 0;
}

extern "C" size_t ds4_mmq_q4_K_q8_1_scratch_bytes(int N, int K) {
    size_t total = 0;
    return ds4_q4_test_q8_1_layout(N, K, nullptr, &total) ? total : 0;
}

extern "C" int ds4_mmq_q4_K_quantize_q8_1_for_test(
        const float *X_f32, void *q8_ds4, size_t q8_bytes,
        int N, int K, cudaStream_t stream) {
    size_t total = 0;
    if (!X_f32 || !q8_ds4 ||
        !ds4_q4_test_q8_1_layout(N, K, nullptr, &total) ||
        q8_bytes < total) {
        return -1;
    }
    cudaError_t err = cudaMemsetAsync(q8_ds4, 0, total, stream);
    if (err != cudaSuccess) return -2;
    quantize_mmq_q8_1_cuda(
        X_f32, /*ids=*/nullptr, q8_ds4, GGML_TYPE_Q4_K,
        /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
        /*ne0=*/GGML_PAD((int64_t)K, MATRIX_ROW_PADDING),
        /*ne1=*/N, /*ne2=*/1, /*ne3=*/1, stream);
    err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -3;
}

extern "C" int ds4_mmq_q4_K_dense_preq_reference_for_test(
        const void *W_q4_K, const void *q8_ds4, size_t q8_bytes,
        float *out_f32, int M, int N, int K, int use_stream_k,
        void *stream_k_fixup, size_t stream_k_fixup_bytes,
        cudaStream_t stream) {
    size_t payload = 0, total = 0;
    if (!W_q4_K || !q8_ds4 || !out_f32 || M <= 0 ||
        !ds4_q4_test_q8_1_layout(N, K, &payload, &total) ||
        q8_bytes < total) {
        return -1;
    }
    const int dev = ggml_cuda_get_device();
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) return -1;
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_k_tile_supported<GGML_TYPE_Q4_K>(
            "ds4_mmq_q4_K_dense_preq_reference_for_test", K, cc)) {
        return -1;
    }
    ggml_backend_cuda_context *ctx = get_ctx_for_device(dev);
    if (!ctx) return -1;
    if ((stream_k_fixup == nullptr && stream_k_fixup_bytes != 0u) ||
        (stream_k_fixup != nullptr &&
         ((uintptr_t)stream_k_fixup % alignof(float)) != 0) ||
        (stream_k_fixup_bytes % sizeof(float)) != 0u) {
        return -1;
    }
    if (stream_k_fixup != nullptr) {
        if (!use_stream_k) return -1;
        if (!ds4_q4_test_reference_uses_m128n128(dev, cc, N)) {
            return DS4_MMQ_NOT_APPLICABLE;
        }
        const size_t required =
            ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes(
                M, N, ggml_cuda_info().devices[dev].nsm);
        if (required > stream_k_fixup_bytes) return -1;
    }
    ds4_pool_set_stream(stream);
    const int64_t stride_row_x = (int64_t)K / QK_K;
    const int64_t stride_y = (int64_t)(payload / sizeof(int));
    const mmq_args args = {
        /*x=*/(const char *)W_q4_K,
        /*type_x=*/GGML_TYPE_Q4_K,
        /*y=*/(const int *)q8_ds4,
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out_f32,
        /*ncols_x=*/(int64_t)K, /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/(int64_t)N,
        /*stride_row_x=*/stride_row_x, /*ncols_y=*/(int64_t)N,
        /*nrows_dst=*/(int64_t)M,
        /*nchannels_x=*/1, /*nchannels_y=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/stride_y,
        /*stride_channel_dst=*/0,
        /*nsamples_x=*/1, /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/stride_y,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k != 0,
        /*ncols_max=*/(int64_t)N,
        /*x_soa=*/nullptr,
        /*soa_blocks=*/0,
        /*stream_k_fixup=*/static_cast<float *>(stream_k_fixup),
        /*stream_k_fixup_elements=*/stream_k_fixup_bytes / sizeof(float),
    };
    mul_mat_q_case<GGML_TYPE_Q4_K>(*ctx, args, stream);
    const cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -2;
}

static int ds4_q4_16warp_prepare_once(int device);

extern "C" int ds4_mmq_q4_K_dense_preq_16warp_for_test(
        const void *W_q4_K, const void *q8_ds4, size_t q8_bytes,
        float *out_f32, int M, int N, int K, cudaStream_t stream) {
    size_t total = 0;
    if (!W_q4_K || !q8_ds4 || !out_f32 ||
        !ds4_q4_test_q8_1_layout(N, K, nullptr, &total) ||
        q8_bytes < total) {
        return -1;
    }
    // This Stream-K oracle hook deliberately accepts an N tail (the production
    // selector is stricter until NVIDIA measurements justify broadening it),
    // but it must reject shapes that cannot be represented by this kernel
    // before enqueueing any work.
    if (M < 128 || (M % 128) != 0 || N < 512 ||
        K < 1024 || K > 8192 || (K % QK_K) != 0) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_q4_K_dense_16warp_available(cc) ||
        ds4_q4_16warp_prepare_once(dev) != 0) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    ggml_backend_cuda_context *ctx = get_ctx_for_device(dev);
    if (!ctx) return DS4_MMQ_NOT_APPLICABLE;
    ds4_pool_set_stream(stream);
    const int nsm = ggml_cuda_info().devices[dev].nsm;
    const size_t fixup_bytes =
        ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes(M, N, nsm);
    ggml_cuda_pool_alloc<char> fixup(ctx->pool());
    if (fixup_bytes != 0u) fixup.alloc(fixup_bytes);
    return ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
        W_q4_K, q8_ds4, out_f32, fixup.get(), fixup_bytes,
        M, N, K, nsm, stream);
}

enum {
    DS4_Q4_16WARP_REQUEST = 1,
    DS4_Q4_16WARP_REQUIRE = 2,
    DS4_Q4_16WARP_DISABLE = 4,
};

static bool ds4_q4_16warp_env_enabled(const char *name) {
    const char *value = getenv(name);
    return value && value[0] && !(value[0] == '0' && value[1] == '\0');
}

static int ds4_q4_16warp_mode(void) {
    static const int cached = [] {
        int mode = 0;
        if (ds4_q4_16warp_env_enabled("DS4_CUDA_Q4_MMQ_16WARP")) {
            mode |= DS4_Q4_16WARP_REQUEST;
        }
        if (ds4_q4_16warp_env_enabled(
                "DS4_CUDA_REQUIRE_Q4_MMQ_16WARP")) {
            mode |= DS4_Q4_16WARP_REQUEST | DS4_Q4_16WARP_REQUIRE;
        }
        if (ds4_q4_16warp_env_enabled("DS4_CUDA_NO_Q4_MMQ_16WARP")) {
            mode |= DS4_Q4_16WARP_DISABLE;
        }
        return mode;
    }();
    return cached;
}

static int ds4_q4_16warp_prepare_once(int device) {
    static std::mutex mutex;
    static std::atomic<int> state[GGML_CUDA_MAX_DEVICES];
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return -1;
    int cached = state[device].load(std::memory_order_acquire);
    if (cached == 1) return 0;
    if (cached < 0) return cached;
    std::lock_guard<std::mutex> lock(mutex);
    cached = state[device].load(std::memory_order_relaxed);
    if (cached == 1) return 0;
    if (cached < 0) return cached;
    const int rc = ds4_mmq_q4_K_dense_16warp_prepare();
    state[device].store(rc == 0 ? 1 : rc, std::memory_order_release);
    return rc;
}

/* Keep the 16-warp experiment on geometries with enough independent output
 * tiles to occupy the device well.  Below canonical's 90% whole-tile cutoff,
 * the candidate mirrors canonical stream-K partitioning and fixup; this 80%
 * gate is therefore only an admission/performance heuristic, not a numerical
 * shortcut.  On GB10 the Q-A/KV N=4096 shapes score 88%. */
static bool ds4_q4_16warp_grid_efficient(int M, int N, int nsm) {
    if (M <= 0 || N <= 0 || nsm <= 0) return false;
    const int64_t tiles_m = ((int64_t)M + 127) / 128;
    const int64_t tiles_n = ((int64_t)N + 127) / 128;
    if (tiles_m > INT64_MAX / tiles_n) return false;
    const int64_t tiles = tiles_m * tiles_n;
    const int64_t waves = (tiles + nsm - 1) / nsm;
    if (waves <= 0 || (int64_t)nsm > INT64_MAX / waves) return false;
    return (100 * tiles) / ((int64_t)nsm * waves) >= 80;
}

static bool ds4_q4_16warp_pair_leg_shape_supported(
        int cc, int M, int N, int K) {
    return ds4_mmq_q4_K_dense_16warp_available(cc) &&
           M >= 512 && (M % 128) == 0 &&
           N >= 512 && (N % 128) == 0 &&
           K >= 1024 && K <= 4096 && (K % QK_K) == 0;
}

/* Resolve the experiment before allocation or enqueue. The standalone path
 * uses the public M>=1024 gate; a dense-pair leg may go down to M=512 because
 * Q-A/KV pairs contain a 512-row leg. Each candidate grid must retain at least
 * 80% whole-tile SM-wave efficiency; scheduling itself follows canonical
 * stream-K whenever canonical would split K. */
static int ds4_q4_16warp_select(
        const char *tag, int device, int cc, int M, int N, int K,
        bool pair_leg, bool *selected) {
    if (!selected) return DS4_MMQ_NOT_APPLICABLE;
    *selected = false;
    const int mode = ds4_q4_16warp_mode();
    const bool disabled = (mode & DS4_Q4_16WARP_DISABLE) != 0;
    const bool requested = (mode & DS4_Q4_16WARP_REQUEST) != 0;
    const bool required = (mode & DS4_Q4_16WARP_REQUIRE) != 0;
    if (!requested) return 0;

    const bool shape_supported = pair_leg
        ? ds4_q4_16warp_pair_leg_shape_supported(cc, M, N, K)
        : ds4_mmq_q4_K_dense_16warp_supported(cc, M, N, K) != 0;
    // The exact oracle models canonical m128n128 MMQ. Check the selector's
    // actual result, not only its upper bound: resource limits and equal
    // ceil-division plateaus can make it retain a narrower tile.
    const bool canonical_x128 =
        ds4_q4_test_reference_uses_m128n128(device, cc, N);
    const bool grid_efficient = ds4_q4_16warp_grid_efficient(
        M, N, ggml_cuda_info().devices[device].nsm);
    if (required && (disabled || !shape_supported || !canonical_x128 ||
                     !grid_efficient)) {
        fprintf(stderr,
                "%s: required Q4 16-warp path is ineligible "
                "(scope=%s disabled=%d shape=%d x128=%d grid_eff=%d "
                "M=%d N=%d K=%d)\n",
                tag, pair_leg ? "pair-leg" : "dense",
                disabled ? 1 : 0, shape_supported ? 1 : 0,
                canonical_x128 ? 1 : 0, grid_efficient ? 1 : 0, M, N, K);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    if (disabled || !shape_supported || !canonical_x128 || !grid_efficient) {
        return 0;
    }

    const int prep = ds4_q4_16warp_prepare_once(device);
    if (prep == 0) {
        *selected = true;
        return 0;
    }
    if (required) {
        fprintf(stderr, "%s: required Q4 16-warp preflight failed: %d\n",
                tag, prep);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    (void)cudaGetLastError();
    return 0;
}
#endif

template <ggml_type type>
int ds4_mmq_dense_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        cudaStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        // mmq requires K to be a multiple of the largest super-block size
        // it sees during the inner tile loop, which is QK_K=256.
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_k_tile_supported<type>(tag, K, cc)) return -1;

#if !defined(GGML_USE_HIP)
    bool use_q4_16warp = false;
    if constexpr (type == GGML_TYPE_Q4_K) {
        const int select_rc = ds4_q4_16warp_select(
            tag, dev, cc, M, N, K, /*pair_leg=*/false,
            &use_q4_16warp);
        if (select_rc != 0) {
            return select_rc;
        }
    }
#endif

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    /* Task #22 fix: order the pool's cudaMallocAsync/cudaFreeAsync on the SAME
     * stream the kernels below launch on.  The pool defaults to
     * cudaStreamPerThread; with kernels on the legacy stream the RAII free is
     * ordered on an EMPTY stream, so the driver can recycle/remap the scratch
     * while the in-flight quantize/GEMM still reads it -> intermittent illegal
     * access under shape churn (the batched-draft early-step crash).  The vec
     * impls already do this (graph-capture fix); the batched impls were missed. */
    ds4_pool_set_stream(stream);

    // 1. Quantize F32 activations into the format consumed by MMQ. Blackwell
    //    MXFP4 uses native FP4 tensor cores; other paths use MMQ Q8_1.
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = N;
    const int64_t ne12         = 1;
    const int64_t ne13         = 1;

    const bool use_native_fp4 =
        type == GGML_TYPE_MXFP4 && blackwell_mma_available(cc);
    const size_t y_block_size = use_native_fp4
        ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4
        ? QK_FP4_MMQ : 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne13 * ne12 * ne11 * ne10_padded * y_block_size /
            y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1 = nullptr;
    if (void *scratch = ds4_mmq_aligned_q81_scratch(
            dev, nbytes_src1_q8_1)) {
        src1_q8_1 = (char *)scratch;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_src1_q8_1);
        src1_q8_1 = src1_q8_1_pool.get();
    }

    // S1.1a fix: the mmq Y (activation) buffer is over-allocated for the kernel's
    // tail-tile reads (the +mmq_x_max blocks above), and ne11 columns may not fill
    // the final column tile -- but quantize_mmq_q8_1_cuda only writes the ne11 valid
    // columns.  The mmq kernel (mmq.cuh:3528) unconditionally loads the full column
    // tile, reading the never-written tail.  Pool allocs reuse stale device memory,
    // so that tail is non-deterministic: any allocator/stream perturbation (e.g. an
    // MTP draft's cudaMalloc) changes it and flips a near-threshold argmax in the
    // batched forward (confirmed by compute-sanitizer --tool initcheck on a PRO6000
    // / sm_120: 4-byte uninitialized __global__ read in mul_mat_q_process_tile).
    // The tail's dot-products are masked out by write_back, so only their
    // non-determinism matters; zero the buffer so the tail is a deterministic zero
    // (a zero q8_1 block contributes 0 to the dot product).
    ybuf_memset(src1_q8_1, nbytes_src1_q8_1, stream);

    if (use_native_fp4) {
        quantize_mmq_fp4_cuda(
            X_f32, /*ids=*/nullptr, (void *)src1_q8_1,
            type, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
            /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/ne12, /*ne3=*/ne13,
            stream);
    } else {
        quantize_mmq_q8_1_cuda(
            X_f32, /*ids=*/nullptr, (void *)src1_q8_1,
            type, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
            /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/ne12, /*ne3=*/ne13,
            stream);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. Build mmq_args. stride_row_x is in WEIGHT BLOCKS per row, which
    //    is K / blck_size(type). Q8_0 has block size 32; Q2_K and IQ2_XXS
    //    are K-quants with block size 256.
    const int64_t blck   = ggml_blck_size(type);
    const int64_t s01    = (int64_t)K / blck;
    const int64_t s1     = (int64_t)M;
    const int64_t s12    = ne11 * ne10_padded * y_block_size /
                           (y_values_per_block * sizeof(int));
    const int64_t s13    = ne12 * s12;

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    if (out_memset_enabled()) {
        (void)cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)N * sizeof(float), stream);
    }

#if !defined(GGML_USE_HIP)
    if constexpr (type == GGML_TYPE_Q4_K) {
        if (use_q4_16warp) {
            const int nsm = ggml_cuda_info().devices[dev].nsm;
            const size_t fixup_bytes =
                ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes(
                    M, N, nsm);
            ggml_cuda_pool_alloc<char> fixup(ctx->pool());
            if (fixup_bytes != 0u) fixup.alloc(fixup_bytes);
            const int rc = ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
                W, src1_q8_1, out_f32, fixup.get(), fixup_bytes,
                M, N, K, nsm, stream);
            if (rc != 0) {
                fprintf(stderr,
                        "%s: Q4 16-warp stream-K launch failed: %d\n",
                        tag, rc);
                return -3;
            }
            return 0;
        }
    }
#endif

    const mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1,
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,    /*nrows_x=*/(int64_t)M,    /*ncols_dst=*/ne11,
        /*stride_row_x=*/s01,/*ncols_y=*/ne11,          /*nrows_dst=*/s1,
        /*nchannels_x=*/1,   /*nchannels_y=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/s12, /*stride_channel_dst=*/0,
        /*nsamples_x=*/1,    /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s13, /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne11,
    };

    mul_mat_q_case<type>(*ctx, args, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    if constexpr (type != GGML_TYPE_Q4_K) {
        ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)N, stream);
    }
    return 0;
}

/* Batched Q4_K pair for the prefill tier.  The two ordinary dense calls
 * differ only in their weight/output rows; their Q8_1 MMQ activation is
 * byte-identical.  Keep that activation alive across both established MMQ
 * launches so Q-A and KV pay the quantize/tail-clear prelude once. */
int ds4_mmq_q4_K_dense_pair_impl(
        const void  * W0,
        const void  * W1,
        const float * X_f32,
        float       * out0_f32,
        float       * out1_f32,
        int           M0,
        int           M1,
        int           N,
        int           K,
        cudaStream_t  stream) {
    const char *tag = "ds4_mmq_q4_K_dense_pair";
    if (!W0 || !W1 || !X_f32 || !out0_f32 || !out1_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    if (M0 <= 0 || M1 <= 0 || N <= 0 || K <= 0 || K % 256 != 0) {
        fprintf(stderr, "%s: bad shape M0=%d M1=%d N=%d K=%d\n",
                tag, M0, M1, N, K);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    if ((size_t)M0 > SIZE_MAX / (size_t)N / sizeof(float) ||
        (size_t)M1 > SIZE_MAX / (size_t)N / sizeof(float)) {
        fprintf(stderr, "%s: output size overflow\n", tag);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const size_t out0_bytes = (size_t)M0 * (size_t)N * sizeof(float);
    const size_t out1_bytes = (size_t)M1 * (size_t)N * sizeof(float);
    const uintptr_t out0_addr = (uintptr_t)out0_f32;
    const uintptr_t out1_addr = (uintptr_t)out1_f32;
    const bool outputs_overlap = out0_addr <= out1_addr
        ? (size_t)(out1_addr - out0_addr) < out0_bytes
        : (size_t)(out0_addr - out1_addr) < out1_bytes;
    if (outputs_overlap) {
        fprintf(stderr, "%s: output ranges overlap\n", tag);
        return DS4_MMQ_NOT_APPLICABLE;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_k_tile_supported<GGML_TYPE_Q4_K>(tag, K, cc)) {
        return DS4_MMQ_NOT_APPLICABLE;
    }

#if !defined(GGML_USE_HIP)
    bool use_q4_16warp0 = false;
    bool use_q4_16warp1 = false;
    int select_rc = ds4_q4_16warp_select(
        tag, dev, cc, M0, N, K, /*pair_leg=*/true,
        &use_q4_16warp0);
    if (select_rc != 0) return select_rc;
    select_rc = ds4_q4_16warp_select(
        tag, dev, cc, M1, N, K, /*pair_leg=*/true,
        &use_q4_16warp1);
    if (select_rc != 0) return select_rc;
#endif

    ggml_backend_cuda_context *ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n",
                tag, dev);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t blocks_per_col =
        (size_t)ne10_padded / (4u * (size_t)QK8_1);
    const size_t bytes_per_col =
        blocks_per_col * sizeof(block_q8_1_mmq);
    const size_t slack_blocks = (size_t)get_mmq_x_max_host(cc);
    if ((size_t)N > SIZE_MAX / bytes_per_col ||
        slack_blocks > SIZE_MAX / sizeof(block_q8_1_mmq)) {
        fprintf(stderr, "%s: activation scratch size overflow\n", tag);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const size_t payload_bytes = (size_t)N * bytes_per_col;
    const size_t slack_bytes = slack_blocks * sizeof(block_q8_1_mmq);
    if (payload_bytes > SIZE_MAX - slack_bytes) {
        fprintf(stderr, "%s: activation scratch size overflow\n", tag);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const size_t nbytes_q8_1 = payload_bytes + slack_bytes;

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_q8_1);
    ybuf_memset(src1_q8_1.get(), nbytes_q8_1, stream);
    quantize_mmq_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        GGML_TYPE_Q4_K, /*ne00=*/K, /*s11=*/(int64_t)K,
        /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/(int64_t)N,
        /*ne2=*/1, /*ne3=*/1, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t stride_row_x = (int64_t)K / QK_K;
    const int64_t stride_channel_y =
        (int64_t)(payload_bytes / sizeof(int));
    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) &&
         ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

#if !defined(GGML_USE_HIP)
    const int q4_16warp_nsm = ggml_cuda_info().devices[dev].nsm;
    const size_t q4_16warp_fixup0 = use_q4_16warp0
        ? ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes(
              M0, N, q4_16warp_nsm)
        : 0u;
    const size_t q4_16warp_fixup1 = use_q4_16warp1
        ? ds4_mmq_q4_K_dense_16warp_streamk_scratch_bytes(
              M1, N, q4_16warp_nsm)
        : 0u;
    const size_t q4_16warp_fixup_bytes =
        q4_16warp_fixup0 > q4_16warp_fixup1
            ? q4_16warp_fixup0 : q4_16warp_fixup1;
    // Both legs are ordered on the same stream, so one allocation can be
    // cleared and reused after the first leg's fixup has consumed it.
    ggml_cuda_pool_alloc<char> q4_16warp_fixup(ctx->pool());
    if (q4_16warp_fixup_bytes != 0u) {
        q4_16warp_fixup.alloc(q4_16warp_fixup_bytes);
    }
#endif

    if (out_memset_enabled()) {
        cudaMemsetAsync(out0_f32, 0, out0_bytes, stream);
    }
    const mmq_args args0 = {
        /*x=*/(const char *)W0,
        /*type_x=*/GGML_TYPE_Q4_K,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out0_f32,
        /*ncols_x=*/(int64_t)K,
        /*nrows_x=*/(int64_t)M0,
        /*ncols_dst=*/(int64_t)N,
        /*stride_row_x=*/stride_row_x,
        /*ncols_y=*/(int64_t)N,
        /*nrows_dst=*/(int64_t)M0,
        /*nchannels_x=*/1,
        /*nchannels_y=*/1,
        /*stride_channel_x=*/0,
        /*stride_channel_y=*/stride_channel_y,
        /*stride_channel_dst=*/0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/stride_channel_y,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/(int64_t)N,
    };
#if !defined(GGML_USE_HIP)
    if (use_q4_16warp0) {
        const int rc = ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
            W0, src1_q8_1.get(), out0_f32,
            q4_16warp_fixup.get(), q4_16warp_fixup_bytes,
            M0, N, K, q4_16warp_nsm, stream);
        if (rc != 0) {
            fprintf(stderr,
                    "%s: first Q4 16-warp stream-K launch failed: %d\n",
                    tag, rc);
            return -3;
        }
    } else
#endif
    {
        mul_mat_q_case<GGML_TYPE_Q4_K>(*ctx, args0, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: first mul_mat_q_case launch failed: %s\n",
                    tag, cudaGetErrorString(err));
            return -3;
        }
    }

    if (out_memset_enabled()) {
        cudaMemsetAsync(out1_f32, 0, out1_bytes, stream);
    }
    const mmq_args args1 = {
        /*x=*/(const char *)W1,
        /*type_x=*/GGML_TYPE_Q4_K,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out1_f32,
        /*ncols_x=*/(int64_t)K,
        /*nrows_x=*/(int64_t)M1,
        /*ncols_dst=*/(int64_t)N,
        /*stride_row_x=*/stride_row_x,
        /*ncols_y=*/(int64_t)N,
        /*nrows_dst=*/(int64_t)M1,
        /*nchannels_x=*/1,
        /*nchannels_y=*/1,
        /*stride_channel_x=*/0,
        /*stride_channel_y=*/stride_channel_y,
        /*stride_channel_dst=*/0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/stride_channel_y,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/(int64_t)N,
    };
#if !defined(GGML_USE_HIP)
    if (use_q4_16warp1) {
        const int rc = ds4_mmq_q4_K_dense_16warp_streamk_enqueue(
            W1, src1_q8_1.get(), out1_f32,
            q4_16warp_fixup.get(), q4_16warp_fixup_bytes,
            M1, N, K, q4_16warp_nsm, stream);
        if (rc != 0) {
            fprintf(stderr,
                    "%s: second Q4 16-warp stream-K launch failed: %d\n",
                    tag, rc);
            return -4;
        }
    } else
#endif
    {
        mul_mat_q_case<GGML_TYPE_Q4_K>(*ctx, args1, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: second mul_mat_q_case launch failed: %s\n",
                    tag, cudaGetErrorString(err));
            return -4;
        }
    }
    return 0;
}

#if !defined(GGML_USE_HIP)
static bool ds4_q4_grouped_q81_env_enabled(const char *name) {
    const char *value = getenv(name);
    return value && value[0] && !(value[0] == '0' && value[1] == '\0');
}
#endif

/* Token-batched grouped Q4_K projection for attention output-A.  The source
 * is token-major [N][G][K], while MMQ stores directly into token-major
 * [N][G][M].  Quantizing the strided source as G channels removes the old
 * pack/unpack copies and shares one scratch allocation/quantizer launch.
 *
 * single_grid=false retains the established one-MMQ-launch-per-group path.
 * single_grid=true maps groups to grid.z in one launch, but isolates each
 * z-slice's stream-k coordinate space.  Its grid.x, partial-K ownership and
 * fixup order are therefore identical to the former per-group invocation. */
int ds4_mmq_q4_K_grouped_dense_impl(
        const void  *W,
        const float *X,
        float       *out,
        int          M,
        int          N,
        int          K,
        int          n_groups,
        bool         single_grid,
        cudaStream_t stream) {
    const char *tag = single_grid
        ? "ds4_mmq_q4_K_grouped_dense_single_grid"
        : "ds4_mmq_q4_K_grouped_dense";
    const int pre_enqueue_failure = single_grid
        ? DS4_MMQ_NOT_APPLICABLE
        : -1;
    if (!W || !X || !out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return pre_enqueue_failure;
    }
    if (M <= 0 || N <= 0 || K <= 0 || n_groups <= 0 ||
        K % QK_K != 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d groups=%d\n",
                tag, M, N, K, n_groups);
        return pre_enqueue_failure;
    }

    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
#if !defined(GGML_USE_HIP)
    const bool q81_disable =
        getenv("DS4_CUDA_NO_Q4_GROUPED_ATTN_A_Q81") != nullptr ||
        getenv("DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL") != nullptr ||
        getenv("DS4_CUDA_NO_Q4_GROUPED_ATTN_A") != nullptr ||
        getenv("DS4_CUDA_NO_Q4_GB10_FAST") != nullptr;
    const bool q81_require = ds4_q4_grouped_q81_env_enabled(
        "DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_Q81");
    const bool q81_eligible =
        gb10_optimizations_enabled() &&
        cc == GGML_CUDA_CC_DGX_SPARK && M == 1024 && N > 8 &&
        N <= INT32_MAX / (8*4096) && K == 4096 && n_groups == 8 &&
        (((uintptr_t)X & 15u) == 0u);
    if (q81_require && (q81_disable || !q81_eligible)) {
        fprintf(stderr,
                "%s: required grouped K4096/G8 Q8_1 quantizer is not "
                "eligible\n",
                tag);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const bool use_specialized_q81 = q81_eligible && !q81_disable;
#endif
    ggml_backend_cuda_context *ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n",
                tag, dev);
        return pre_enqueue_failure;
    }
    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t blocks_per_col =
        (size_t)ne10_padded / (4u * (size_t)QK8_1);
    const size_t bytes_per_col =
        blocks_per_col * sizeof(block_q8_1_mmq);
    if ((size_t)N > SIZE_MAX / bytes_per_col) return pre_enqueue_failure;
    const size_t channel_bytes = (size_t)N * bytes_per_col;
    if ((size_t)n_groups > SIZE_MAX / channel_bytes) {
        return pre_enqueue_failure;
    }
    const size_t payload_bytes = (size_t)n_groups * channel_bytes;
    const size_t slack_blocks = (size_t)get_mmq_x_max_host(cc);
    if (slack_blocks > SIZE_MAX / sizeof(block_q8_1_mmq)) {
        return pre_enqueue_failure;
    }
    const size_t slack_bytes = slack_blocks * sizeof(block_q8_1_mmq);
    if (payload_bytes > SIZE_MAX - slack_bytes) return pre_enqueue_failure;

    const int64_t row_blocks = (int64_t)K / QK_K;
    if ((size_t)M > SIZE_MAX / (size_t)row_blocks /
                        sizeof(block_q4_K)) return pre_enqueue_failure;
    const size_t group_weight_bytes =
        (size_t)M * (size_t)row_blocks * sizeof(block_q4_K);
    const int64_t group_weight_blocks = (int64_t)M * row_blocks;
    const int64_t low_dim = (int64_t)M * n_groups;
    if (low_dim > INT_MAX ||
        (uint64_t)low_dim > UINT64_MAX / (uint64_t)N) {
        return pre_enqueue_failure;
    }
    /* The grouped kernel ABI narrows strides and weight-block offsets to int.
     * Reject before allocating or enqueueing so an optional caller can safely
     * fall back to the established per-group launch loop. */
    if (single_grid &&
        (n_groups > 65535 || group_weight_blocks > INT_MAX ||
         group_weight_blocks * (int64_t)n_groups > INT_MAX ||
         channel_bytes / sizeof(int) > (size_t)INT_MAX)) {
        return DS4_MMQ_NOT_APPLICABLE;
    }

    ggml_cuda_pool_alloc<char> y_q8_1(
        ctx->pool(), payload_bytes + slack_bytes);
    ybuf_memset(y_q8_1.get(), payload_bytes + slack_bytes, stream);
#if !defined(GGML_USE_HIP)
    if (use_specialized_q81) {
        quantize_mmq_q8_1_q4_grouped_k4096_g8x2_cuda(
            X, (void *)y_q8_1.get(), N, stream);
    } else
#endif
    {
        quantize_mmq_q8_1_cuda(
            X, /*ids=*/nullptr, (void *)y_q8_1.get(), GGML_TYPE_Q4_K,
            /*ne00=*/K,
            /*s01=*/(int64_t)n_groups * K,
            /*s02=*/(int64_t)K,
            /*s03=*/(int64_t)n_groups * N * K,
            /*ne0=*/ne10_padded, /*ne1=*/N,
            /*ne2=*/n_groups, /*ne3=*/1, stream);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    if (out_memset_enabled()) {
        cudaMemsetAsync(out, 0,
                        (size_t)N * (size_t)low_dim * sizeof(float), stream);
    }
    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) &&
         ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);
    if (single_grid) {
        const mmq_args args = {
            /*x=*/(const char *)W,
            /*type_x=*/GGML_TYPE_Q4_K,
            /*y=*/(const int *)y_q8_1.get(),
            /*ids_dst=*/nullptr,
            /*expert_bounds=*/nullptr,
            /*dst=*/out,
            /*ncols_x=*/(int64_t)K,
            /*nrows_x=*/(int64_t)M,
            /*ncols_dst=*/(int64_t)N,
            /*stride_row_x=*/row_blocks,
            /*ncols_y=*/(int64_t)N,
            /*nrows_dst=*/low_dim,
            /*nchannels_x=*/(int64_t)n_groups,
            /*nchannels_y=*/(int64_t)n_groups,
            /*stride_channel_x=*/group_weight_blocks,
            /*stride_channel_y=*/(int64_t)(channel_bytes / sizeof(int)),
            /*stride_channel_dst=*/(int64_t)M,
            /*nsamples_x=*/1,
            /*nsamples_y=*/1,
            /*stride_sample_x=*/0,
            /*stride_sample_y=*/0,
            /*stride_sample_dst=*/0,
            /*use_stream_k=*/use_stream_k,
            /*ncols_max=*/(int64_t)N,
        };
        mul_mat_q_case_grouped_channels<GGML_TYPE_Q4_K>(*ctx, args, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: grouped launch failed: %s\n",
                    tag, cudaGetErrorString(err));
            return -3;
        }
        return 0;
    }
    for (int g = 0; g < n_groups; ++g) {
        const mmq_args args = {
            /*x=*/(const char *)W + (size_t)g * group_weight_bytes,
            /*type_x=*/GGML_TYPE_Q4_K,
            /*y=*/(const int *)(y_q8_1.get() + (size_t)g * channel_bytes),
            /*ids_dst=*/nullptr,
            /*expert_bounds=*/nullptr,
            /*dst=*/out + (int64_t)g * M,
            /*ncols_x=*/(int64_t)K,
            /*nrows_x=*/(int64_t)M,
            /*ncols_dst=*/(int64_t)N,
            /*stride_row_x=*/row_blocks,
            /*ncols_y=*/(int64_t)N,
            /*nrows_dst=*/low_dim,
            /*nchannels_x=*/1,
            /*nchannels_y=*/1,
            /*stride_channel_x=*/0,
            /*stride_channel_y=*/(int64_t)(channel_bytes / sizeof(int)),
            /*stride_channel_dst=*/0,
            /*nsamples_x=*/1,
            /*nsamples_y=*/1,
            /*stride_sample_x=*/0,
            /*stride_sample_y=*/(int64_t)(channel_bytes / sizeof(int)),
            /*stride_sample_dst=*/0,
            /*use_stream_k=*/use_stream_k,
            /*ncols_max=*/(int64_t)N,
        };
        mul_mat_q_case<GGML_TYPE_Q4_K>(*ctx, args, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: group %d launch failed: %s\n",
                    tag, g, cudaGetErrorString(err));
            return -3;
        }
    }
    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q8_0>("ds4_mmq_q8_0_dense", W, X, out, M, N, K, stream);
}

/* p5a verify instrument: run the REFERENCE quantizer (exact dense_impl
 * parameters) into a caller buffer so producers can be diffed
 * byte-for-byte against it (DS4_CUDA_OUTA_Q8EMIT_VERIFY). */
extern "C" int ds4_mmq_q8_0_quantize_ref(
        const float * X, void * y, size_t y_bytes, int N, int K,
        cudaStream_t stream) {
    if (!X || !y || N <= 0 || K <= 0 || K % 256 != 0) return -1;
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) return -1;
    ds4_pool_set_stream(stream);
    const size_t need = (size_t)N * ne10_padded * sizeof(block_q8_1) / QK8_1;
    if (y_bytes < need) return -1;
    (void)cudaMemsetAsync(y, 0, y_bytes, stream);
    quantize_mmq_q8_1_cuda(
        X, /*ids=*/nullptr, y, GGML_TYPE_Q8_0,
        /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/(int64_t)N, /*ne2=*/1, /*ne3=*/1, stream);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

/* v0.5 flat-pool p5a: dense Q8_0 mmq consuming a PRODUCER-EMITTED
 * block_q8_1_mmq activation buffer (the fused own out_a kernel dual-emits
 * the q8_1 of `low` in its epilogue, op-for-op equal to
 * quantize_mmq_q8_1<D4> -- the batched sibling of the M2-Inc2a
 * producer-codes affordance on the vec entry).  The caller owns the
 * buffer: all N*(K/128) blocks written by the producer, PLUS the
 * mmq_x_max tail-tile slack which THIS entry zeroes (S1.1a determinism;
 * the producer's sticky scratch may hold stale bytes).  Requires the
 * padded row width to equal K so the producer's block addressing
 * (ib = kseg*N + row) matches the quantizer's exactly. */
extern "C" int ds4_mmq_q8_0_dense_preq(
        const void * W, const void * Y_q8_mmq, size_t y_bytes, float * out,
        int M, int N, int K, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q8_0_dense_preq";
    if (!W || !Y_q8_mmq || !out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0 || K % 256 != 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    if (ne10_padded != (int64_t)K) return -1;  /* producer layout requires no row padding */

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);

    const size_t data_bytes =
        (size_t)N * (size_t)ne10_padded * sizeof(block_q8_1) / QK8_1;
    const size_t slack_bytes =
        (size_t)get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    if (y_bytes < data_bytes + slack_bytes) {
        fprintf(stderr, "%s: y buffer too small (%zu < %zu)\n", tag,
                y_bytes, data_bytes + slack_bytes);
        return -1;
    }
    /* Tail-tile slack: deterministic zeros (S1.1a).  ~18 KiB, stream-ordered
     * after the producer's emit on the same stream. */
    (void)cudaMemsetAsync((char *)Y_q8_mmq + data_bytes, 0, slack_bytes, stream);

    const int64_t s01 = (int64_t)K / QK8_0;
    const int64_t s12 = (int64_t)N * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    if (out_memset_enabled()) {
        (void)cudaMemsetAsync(out, 0, (size_t)M * (size_t)N * sizeof(float), stream);
    }

    const mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/GGML_TYPE_Q8_0,
        /*y=*/(const int *)Y_q8_mmq,
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out,
        /*ncols_x=*/(int64_t)K, /*nrows_x=*/(int64_t)M, /*ncols_dst=*/(int64_t)N,
        /*stride_row_x=*/s01,   /*ncols_y=*/(int64_t)N, /*nrows_dst=*/(int64_t)M,
        /*nchannels_x=*/1,   /*nchannels_y=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/s12, /*stride_channel_dst=*/0,
        /*nsamples_x=*/1,    /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s12, /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/(int64_t)N,
    };
    mul_mat_q_case<GGML_TYPE_Q8_0>(*ctx, args, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    ds4_mmq_sanitize_f32(out, (uint64_t)M * (uint64_t)N, stream);
    return 0;
}

// Dense Q8_0 D2R entry: same activation quantize + scratch treatment as
// ds4_mmq_dense_impl (incl. the S1.1a zero for the never-written tail), then
// the D2R kernel on the kind-5 aligned artifact instead of mul_mat_q_case.
// No out-memset / trailing sanitize: the D2R epilogue writes every element
// through an isfinite guard.  Caller (ds4_cuda.cu) resolves W_aligned and
// gates on shape (M%128, K%1024, K<=4096) + n_tok.
extern "C" int ds4_mmq_q8_0_dense_d2r(
        const void * W_aligned, const float * X_f32, float * out_f32,
        int M, int N, int K, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q8_0_dense_d2r";
    if (!W_aligned || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || (M % 128) != 0 || N <= 0 || K <= 0 || (K % 1024) != 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_q8_0_dense_d2r_available(cc)) {
        return -1;
    }
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    // Slack: the guarded last col tile reads up to 128 blocks past N*K/128.
    const int64_t slack_blocks = std::max<int64_t>(get_mmq_x_max_host(cc), 128);
    const size_t nbytes_src1_q8_1 =
        (int64_t)N * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        slack_blocks * sizeof(block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1 = nullptr;
    if (void *scratch = ds4_mmq_aligned_q81_scratch(
            dev, nbytes_src1_q8_1)) {
        src1_q8_1 = (char *)scratch;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_src1_q8_1);
        src1_q8_1 = src1_q8_1_pool.get();
    }
    ybuf_memset(src1_q8_1, nbytes_src1_q8_1, stream);

    quantize_mmq_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1,
        GGML_TYPE_Q8_0, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/(int64_t)N, /*ne2=*/1, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }
    return ds4_mmq_q8_0_dense_d2r_launch(W_aligned, src1_q8_1, out_f32,
                                         M, N, K, stream);
}

/* flat-pool p5c: D2R dense entry over a producer-quantized Y (token-major
 * block_q8_1_mmq, ib = kseg*N + row, no row padding).  Same contract as
 * ds4_mmq_q8_0_dense_d2r minus the activation quantize; the caller's
 * buffer must carry the guarded-tail slack (zeroed here each call, S1.1a:
 * the producer rewrites every payload byte, the slack region may hold a
 * previous larger emit's bytes). */
extern "C" int ds4_mmq_q8_0_dense_d2r_preq(
        const void * W_aligned, const void * Y_q8_mmq, size_t y_bytes,
        float * out_f32, int M, int N, int K, cudaStream_t stream) {
    if (!W_aligned || !Y_q8_mmq || !out_f32) {
        return -1;
    }
    if (M <= 0 || (M % 128) != 0 || N <= 0 || K <= 0 || (K % 1024) != 0) {
        return -1;
    }
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_q8_0_dense_d2r_available(cc)) {
        return -1;
    }
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    if (ne10_padded != (int64_t)K) return -1;  /* producer layout requires no row padding */
    const int64_t slack_blocks = std::max<int64_t>(get_mmq_x_max_host(cc), 128);
    const size_t data_bytes =
        (size_t)N * (size_t)ne10_padded * sizeof(block_q8_1) / QK8_1;
    const size_t slack_bytes = (size_t)slack_blocks * sizeof(block_q8_1_mmq);
    if (y_bytes < data_bytes + slack_bytes) {
        return -1;
    }
    (void)cudaMemsetAsync((char *)Y_q8_mmq + data_bytes, 0, slack_bytes, stream);
    return ds4_mmq_q8_0_dense_d2r_launch(W_aligned, Y_q8_mmq, out_f32,
                                         M, N, K, stream);
}

extern "C" int ds4_mmq_q2_K_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_iq2_xxs_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_IQ2_XXS>("ds4_mmq_iq2_xxs_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_q4_K_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q4_K>("ds4_mmq_q4_K_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_q4_K_dense_pair(
        const void * W0, const void * W1, const float * X,
        float * out0, float * out1,
        int M0, int M1, int N, int K, cudaStream_t stream) {
    return ds4_mmq_q4_K_dense_pair_impl(
        W0, W1, X, out0, out1, M0, M1, N, K, stream);
}

extern "C" int ds4_mmq_q4_K_grouped_dense(
        const void *W, const float *X, float *out,
        int M, int N, int K, int n_groups, cudaStream_t stream) {
    return ds4_mmq_q4_K_grouped_dense_impl(
        W, X, out, M, N, K, n_groups, false, stream);
}

extern "C" int ds4_mmq_q4_K_grouped_dense_single_grid(
        const void *W, const float *X, float *out,
        int M, int N, int K, int n_groups, cudaStream_t stream) {
    return ds4_mmq_q4_K_grouped_dense_impl(
        W, X, out, M, N, K, n_groups, true, stream);
}

#if !defined(GGML_USE_HIP)
extern "C" size_t ds4_mmq_q4_K_grouped_q8_1_scratch_bytes_for_test(int N) {
    if (N <= 0 || N > INT32_MAX / (8*4096)) return 0u;
    constexpr size_t blocks_per_token = 8u * 4096u / (4u * QK8_1);
    if ((size_t)N > SIZE_MAX / blocks_per_token /
                        sizeof(block_q8_1_mmq)) {
        return 0u;
    }
    return (size_t)N * blocks_per_token * sizeof(block_q8_1_mmq);
}

extern "C" int ds4_mmq_q4_K_grouped_quantize_q8_1_for_test(
        const float *X, void *q8, size_t q8_bytes, int N,
        int use_specialized, cudaStream_t stream) {
    const size_t required =
        ds4_mmq_q4_K_grouped_q8_1_scratch_bytes_for_test(N);
    if (!X || !q8 || required == 0u || q8_bytes < required ||
        (((uintptr_t)X & 15u) != 0u)) {
        return -1;
    }
    if (use_specialized) {
        quantize_mmq_q8_1_q4_grouped_k4096_g8x2_cuda(
            X, q8, N, stream);
    } else {
        quantize_mmq_q8_1_cuda(
            X, /*ids=*/nullptr, q8, GGML_TYPE_Q4_K,
            /*ne00=*/4096, /*s01=*/8*4096, /*s02=*/4096,
            /*s03=*/(int64_t)N*8*4096,
            /*ne0=*/4096, /*ne1=*/N, /*ne2=*/8, /*ne3=*/1, stream);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
#endif

extern "C" int ds4_mmq_mxfp4_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_MXFP4>("ds4_mmq_mxfp4_dense", W, X, out, M, N, K, stream);
}

// ----------------------------------------------------------------------------
// MoE matmul implementation, shared across all three quant types.
//
// Mirrors upstream mmq.cu:163-222 (the ids != nullptr branch).  Caller
// provides:
//   - per-expert weights stacked contiguously
//   - per-token activations [n_tokens, K]
//   - routing table ids[t, s] = expert id
// The wrapper invokes:
//   1. ggml_cuda_launch_mm_ids_helper to build (ids_src1, ids_dst,
//      expert_bounds) - permutations that sort assignments by expert.
//   2. quantize_mmq_q8_1_cuda with ids_src1 - gathers and quantizes the
//      activation into the expert-major flat layout.
//   3. mul_mat_q_case<type> with ids_dst + expert_bounds - the matmul.
// ----------------------------------------------------------------------------

namespace {

template <ggml_type type>
int ds4_mmq_moe_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream,
        /* ds4 (P4 Inc3): optional aligned-SoA artifact; when non-null the mmq
         * kernel loads tiles from it directly and W is ignored (see mmq_args). */
        const char    * x_soa      = NULL,
        int64_t         soa_blocks = 0,
        /* ds4 (P3): false skips the whole-buffer nonfinite pass; only valid
         * when every consumer sanitizes at read (the routed-MoE swiglu/sum
         * kernels do). */
        bool            sanitize_out = true) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_k_tile_supported<type>(tag, K, cc)) return -1;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);  /* task #22: pool ops must be stream-ordered with the kernels (see ds4_mmq_dense_impl) */

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;             // src1 rows per channel (one per token)
    const int64_t ne12         = n_tokens;      // src1 channels (= tokens)
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;   // per-expert weight stride in blocks

    // 1. Build the expert-major work map.
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx->pool(), n_experts + 1);

    // Task #22 root-cause fix: mm_ids_helper COMPACTS - it only writes ids_src1
    // entries for in-range router ids and drops invalid ones (the router's NaN
    // path emits -1 by design), so with any dropped id the tail of ids_src1
    // stays unwritten pool memory.  quantize_mmq_q8_1's grid covers all
    // ne_get_rows rows and gathers x rows via ids_src1[i1] unconditionally
    // (quantize.cu:304), so a stale/garbage tail entry becomes a wild OOB read
    // (the intermittent batched-draft illegal access; B200 memcheck-convicted).
    // Zero both id maps so unwritten tail slots gather/scatter row 0 instead:
    // those lanes' output is never consumed (the mmq write-back loop is
    // expert_bounds-bounded), the cost is a few KB of memset on-stream.
    (void)cudaMemsetAsync(ids_src1.get(), 0, ne_get_rows * sizeof(int32_t), stream);
    (void)cudaMemsetAsync(ids_dst.get(),  0, ne_get_rows * sizeof(int32_t), stream);

    // si1 = stride between tokens in the ids tensor, in elements. Our ids is
    // contiguous [n_tokens, n_expert_used] so si1 = n_expert_used.
    // sis1 = stride between src1 channels in row-units. With ne11=1, sis1=1
    //        means each "channel" of src1 is one row of K floats.
    const int si1  = n_expert_used;
    const int sis1 = 1;

    // The smem mm_ids_helper uses n_tokens * 4 bytes of dynamic shared memory;
    // the down matmul reaches here with n_tokens = assignments (6x the forward
    // width), so 8192-row prefill chunks pass 48384 "tokens" > cap.  P5: past
    // the cap the launcher dispatches the bit-identical two-pass global
    // variant instead (mmid.cu mm_ids_helper_global) — refusing here used to
    // throw the WHOLE MoE block (including gate/up mmq work) onto the legacy
    // expert-tile fallback, the W8192 prefill cliff.  DS4_MMID_LARGE=0
    // restores the refusal.
    if ((size_t)n_tokens * 4u > ggml_cuda_info().devices[dev].smpbo && !ds4_mmid_large_enabled()) {
        fprintf(stderr, "%s: n_tokens=%d exceeds mm_ids_helper shared-mem cap; falling back\n",
                tag, n_tokens);
        return -1;
    }

    ggml_cuda_launch_mm_ids_helper(
        ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11, si1, sis1, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. Gather + quantize activations. Native Blackwell MXFP4 consumes FP4;
    //    all other MMQ kernels consume Q8_1.
    const bool use_native_fp4 =
        type == GGML_TYPE_MXFP4 && blackwell_mma_available(cc);
    const size_t y_block_size = use_native_fp4
        ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4
        ? QK_FP4_MMQ : 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * y_block_size / y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    // S1.1a fix (same as the dense path): the mmq Y buffer is over-allocated for the
    // kernel's tail-tile reads and ne_get_rows columns need not fill the final mmq
    // column tile, but quantize only writes the valid columns.  The mmq kernel
    // (mmq.cuh:3528) unconditionally loads the full tile, reading the never-written
    // tail from stale pool memory -> allocator-perturbation-dependent garbage in the
    // (write_back-masked) tail lanes -> non-deterministic batched-forward output.
    // Zero it so the masked-out tail is a deterministic zero.
    ybuf_memset(src1_q8_1.get(), nbytes_src1_q8_1, stream);

    // src1 logical [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost, then
    // one row per channel, channels = tokens.
    const int64_t s11_src = (int64_t)K;                                 // stride between rows of a channel
    const int64_t s12_src = (int64_t)K * ne11;                          // stride between channels = K*1
    const int64_t s13_src = (int64_t)K * ne11 * ne12;                   // stride between samples

    if (use_native_fp4) {
        quantize_mmq_fp4_cuda(
            X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
            type, /*ne00=*/K, s11_src, s12_src, s13_src,
            /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
            stream);
    } else {
        quantize_mmq_q8_1_cuda(
            X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
            type, /*ne00=*/K, s11_src, s12_src, s13_src,
            /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
            stream);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: MMQ activation quantize failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }

    // 3. Build mmq_args for the MoE path.
    //
    // dst layout convention matches upstream's MoE branch
    // (mmq.cu:215-220): dst is interpreted as [M, n_expert_used, n_tokens]
    // with M innermost and n_expert_used as the second dim that mmq writes
    // through ids_dst.  s1 = M (the column stride in the flat dst buffer
    // mmq writes into).  The output is column-major: out[col*M + row].
    const int64_t s1            = (int64_t)M;
    // stride_channel_y per upstream: ne11 * ne10_padded * sizeof(block_q8_1)
    //                                     / (QK8_1 * sizeof(int))
    // In MoE mode the kernel zeroes out the channel-stride contribution to
    // offset_y after reading expert_bounds, so the value is permissive -
    // but we set it consistently with upstream.
    const int64_t s12_mmq = ne11 * ne10_padded * y_block_size /
                            (y_values_per_block * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    if (out_memset_enabled()) {
        (void)cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)ne_get_rows * sizeof(float), stream);
    }

    if (type == GGML_TYPE_Q2_K && x_soa != nullptr && d2r_enabled() &&
        K % 256 == 0 && M % 2 == 0 && ne_get_rows >= d2r_min_cols()) {
        static int d2r_avail_cc = -1;
        static int d2r_avail = 0;
        if (d2r_avail_cc != cc) {
            d2r_avail_cc = cc;
            d2r_avail = ds4_mmq_q2_K_moe_d2r_available(cc) ? 1 : 0;
        }
        if (d2r_avail) {
            const size_t d2r_work_bytes =
                ds4_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
            if (d2r_work_bytes != 0) {
                ggml_cuda_pool_alloc<char> d2r_work(ctx->pool(), d2r_work_bytes);
                const int d2r_rc = ds4_mmq_q2_K_moe_d2r_launch(
                    x_soa, soa_blocks, src1_q8_1.get(), ids_dst.get(), expert_bounds.get(),
                    out_f32, M, K, ne_get_rows, n_experts, d2r_work.get(), d2r_work_bytes,
                    stream);
                if (d2r_rc == 0) {
                    return 0;
                }
            }
        }
    }

    const mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/ids_dst.get(),
        /*expert_bounds=*/expert_bounds.get(),
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,
        /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/ne_get_rows,
        /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows,
        /*nrows_dst=*/s1,
        /*nchannels_x=*/(int64_t)n_experts,
        /*nchannels_y=*/(int64_t)n_experts,
        /*stride_channel_x=*/s02,
        /*stride_channel_y=*/s12_mmq,
        /*stride_channel_dst=*/(int64_t)0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/s13_mmq,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne_get_rows,
        /*x_soa=*/x_soa,
        /*soa_blocks=*/soa_blocks,
    };

    mul_mat_q_case<type>(*ctx, args, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (moe) launch failed: %s\n", tag, cudaGetErrorString(err));
        return -4;
    }
    if (sanitize_out && type != GGML_TYPE_Q4_K) {
        ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
}

struct ds4_mmq_fused_down {
    const void  * W;
    const char  * W_soa;
    int64_t       soa_blocks;
    const float * router_weights;
    float       * mid_f32;
    float       * out;
    int           out_dim;
    float         clamp;
    bool          direct_gateup_q8;
    void        * input_q8_scratch;
    size_t        input_q8_scratch_bytes;
    void        * q8_scratch;
    size_t        q8_scratch_bytes;
    void        * work_scratch;
    size_t        work_scratch_bytes;
    /* flat-pool p5c: producer-emitted token-compact q8 of X (see the
     * fused_direct_soa doc in ds4_mmq.h); NULL = quantize internally. */
    const void  * input_q8_ext;
    size_t        input_q8_ext_bytes;
};

static bool ds4_mmq_take_scratch(
        void *base, size_t capacity, size_t *offset,
        size_t bytes, size_t alignment, void **result) {
    if (!base || !offset || !result || alignment == 0 ||
        (alignment & (alignment - 1)) != 0) {
        return false;
    }
    if (*offset > capacity) return false;
    const uintptr_t address = (uintptr_t)base + *offset;
    const size_t padding = (size_t)(-(uintptr_t)address) & (alignment - 1);
    if (padding > capacity - *offset) return false;
    const size_t aligned = *offset + padding;
    if (bytes > capacity - aligned) return false;
    *result = (char *)base + aligned;
    *offset = aligned + bytes;
    return true;
}

static bool ds4_mmq_scratch_overlaps(
        const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const uintptr_t a_addr = (uintptr_t)a;
    const uintptr_t b_addr = (uintptr_t)b;
    return a_addr <= b_addr
        ? b_addr - a_addr < a_bytes
        : a_addr - b_addr < b_bytes;
}

// Produce the weighted SwiGLU rows in their canonical pair-major order. The
// proven upstream quantizer below gathers them through the already available
// ids_dst map, so gate/up and down share one expert-major schedule without a
// second mm_ids_helper.
static __global__ void ds4_swiglu_weighted_f32(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        const float * __restrict__ router_weights,
        float * __restrict__ mid,
        uint64_t n,
        int K,
        float clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t pair = i / (uint64_t)K;
    float g = isfinite(gate[i]) ? gate[i] : 0.0f;
    float u = isfinite(up[i]) ? up[i] : 0.0f;
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    mid[i] = (g / (1.0f + expf(-g))) * u * router_weights[pair];
}

// Paired MoE: one helper + one quantize covers both weights.  See the
// header comment on ds4_mmq_iq2_xxs_moe_pair for motivation.  Internal
// structure mirrors ds4_mmq_moe_impl above; the only differences are the
// two W pointers, the two output pointers, and the second mul_mat_q_case
// launch with a fresh (x, dst) pair.
template <ggml_type type, bool profile_fused_prefill = false>
int ds4_mmq_moe_pair_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_a,
        float         * out_b,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream,
        /* ds4 (P4 Inc3): optional aligned-SoA artifacts for W_a / W_b (same
         * shape, so one block count); see ds4_mmq_moe_impl. */
        const char    * xa_soa     = NULL,
        const char    * xb_soa     = NULL,
        int64_t         soa_blocks = 0,
        /* ds4 (P3): see ds4_mmq_moe_impl. */
        bool            sanitize_out = true,
        const ds4_mmq_fused_down *fused_down = nullptr) {

    const bool direct_gateup_q8 =
        fused_down != nullptr && fused_down->direct_gateup_q8;
    if (!W_a || !W_b || !X_f32 || !ids ||
        (!direct_gateup_q8 && (!out_a || !out_b))) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }
    if (fused_down &&
        (type != GGML_TYPE_IQ2_XXS || !fused_down->W ||
         !fused_down->router_weights ||
         (!direct_gateup_q8 && !fused_down->mid_f32) ||
         (direct_gateup_q8 &&
          (!xa_soa || !xb_soa || !fused_down->W_soa ||
           !fused_down->input_q8_scratch ||
           fused_down->input_q8_scratch_bytes == 0 ||
           !fused_down->q8_scratch || fused_down->q8_scratch_bytes == 0 ||
           !fused_down->work_scratch || fused_down->work_scratch_bytes == 0)) ||
         !fused_down->out || fused_down->out_dim <= 0 || M % 256 != 0)) {
        fprintf(stderr, "%s: invalid fused Q2_K down configuration\n", tag);
        return -1;
    }

    const bool nvtx_prefill = profile_fused_prefill &&
                              fused_down != nullptr &&
                              n_tokens >= 1024 &&
                              ds4_mmq_nvtx_requested();
    ds4_mmq_nvtx_scope fused_scope(
            "ds4/prefill/moe/mmq_fused",
            ds4_mmq_nvtx_payload((uint32_t)n_tokens, (uint32_t)n_expert_used),
            nvtx_prefill);

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_k_tile_supported<type>(tag, K, cc)) return -1;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);  /* task #22: pool ops must be stream-ordered with the kernels (see ds4_mmq_dense_impl) */

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;
    const int64_t ne12         = n_tokens;
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;

    // Past the shared-memory cap the launcher takes the bit-identical global
    // variant unless the authoritative large-map kill switch rejects it.
    if ((size_t)n_tokens * 4u > ggml_cuda_info().devices[dev].smpbo &&
        !ds4_mmid_large_enabled()) {
        fprintf(stderr,
                "%s: n_tokens=%d exceeds mm_ids_helper shared-mem cap; "
                "falling back\n", tag, n_tokens);
        return -1;
    }

    ggml_cuda_pool_alloc<int32_t> ids_src1_alloc;
    ggml_cuda_pool_alloc<int32_t> ids_dst_alloc;
    ggml_cuda_pool_alloc<int32_t> expert_bounds_alloc;
    int32_t *ids_src1 = nullptr;
    int32_t *ids_dst = nullptr;
    int32_t *expert_bounds = nullptr;
    void *direct_work = nullptr;
    size_t direct_work_bytes = 0;

    const bool use_native_fp4 =
        type == GGML_TYPE_MXFP4 && blackwell_mma_available(cc);
    const size_t y_block_size = use_native_fp4
        ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4
        ? QK_FP4_MMQ : 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * y_block_size / y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    size_t direct_down_q8_bytes = 0;
    const bool persistent_pair_maps =
        !direct_gateup_q8 && type == GGML_TYPE_IQ2_XXS &&
        cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151 && stream == nullptr &&
        n_expert_used == 6 && n_experts == (int)MMQ_GFX1151_PAIR_MAP_EXPERTS &&
        ne_get_rows <= (int64_t)MMQ_GFX1151_PAIR_MAP_ROWS &&
        g_mmq_pair_maps[dev].base != nullptr;
    if (direct_gateup_q8) {
        const int64_t down_ne10_padded = GGML_PAD((int64_t)M, MATRIX_ROW_PADDING);
        direct_down_q8_bytes =
            (size_t)ne_get_rows * (size_t)down_ne10_padded *
                sizeof(block_q8_1) / QK8_1 +
            (size_t)get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        const size_t gateup_work_bytes =
            ds4_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(
                ne_get_rows, n_experts);
        const size_t down_work_bytes =
            ds4_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
        if (fused_down->input_q8_scratch_bytes < nbytes_src1_q8_1) return -91;
        if (fused_down->q8_scratch_bytes < direct_down_q8_bytes) return -92;
        if (gateup_work_bytes == 0 || down_work_bytes == 0) return -93;
        if (!d2r_enabled() || !d2r_iq2_enabled()) return -94;
        if (ne_get_rows < d2r_min_cols()) return -95;
        if (!ds4_mmq_iq2_xxs_moe_d2r_available(cc) ||
            !ds4_mmq_q2_K_moe_d2r_available(cc)) {
            return -96;
        }

        size_t offset = 0;
        void *ids_src1_raw = nullptr;
        void *ids_dst_raw = nullptr;
        void *expert_bounds_raw = nullptr;
        direct_work_bytes = gateup_work_bytes > down_work_bytes
            ? gateup_work_bytes : down_work_bytes;
        if (!ds4_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, (size_t)ne_get_rows * sizeof(int32_t), 256,
                &ids_src1_raw) ||
            !ds4_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, (size_t)ne_get_rows * sizeof(int32_t), 256,
                &ids_dst_raw) ||
            !ds4_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, (size_t)(n_experts + 1) * sizeof(int32_t), 256,
                &expert_bounds_raw) ||
            !ds4_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, direct_work_bytes, 256, &direct_work)) {
            return -97;
        }
        ids_src1 = (int32_t *)ids_src1_raw;
        ids_dst = (int32_t *)ids_dst_raw;
        expert_bounds = (int32_t *)expert_bounds_raw;
    }

    /* `fused_raw` is the only grouped path that owns neither aligned weights
     * nor caller scratch.  Its input Q8 is dead after gate/up, before down Q8
     * is produced, so one max-sized range can back both phases.  Resolve the
     * opt-in arena (or allocate its one-block pool fallback) before the first
     * expert-map enqueue; no mid-pipeline allocation failure can strand a
     * partially submitted candidate. */
    const bool grouped_raw_q81 = profile_fused_prefill &&
        type == GGML_TYPE_IQ2_XXS && fused_down != nullptr &&
        !direct_gateup_q8 && xa_soa == nullptr && xb_soa == nullptr &&
        fused_down->W_soa == nullptr;
    size_t grouped_down_q8_bytes = 0;
    size_t grouped_q81_required = 0;
    std::unique_lock<std::mutex> grouped_q81_lease(
        g_q81_state_mutex, std::defer_lock);
    ggml_cuda_pool_alloc<char> grouped_q81_pool;
    char *grouped_q81_scratch = nullptr;
    if (grouped_raw_q81 && q81_persistent_requested()) {
        const int64_t down_padded = GGML_PAD((int64_t)M, MATRIX_ROW_PADDING);
        const size_t tail =
            (size_t)get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        if ((size_t)ne_get_rows > SIZE_MAX / (size_t)down_padded ||
            (size_t)ne_get_rows * (size_t)down_padded >
                (SIZE_MAX - tail) / sizeof(block_q8_1)) {
            return -98;
        }
        grouped_down_q8_bytes =
            (size_t)ne_get_rows * (size_t)down_padded *
                sizeof(block_q8_1) / QK8_1 + tail;
        grouped_q81_required = nbytes_src1_q8_1 > grouped_down_q8_bytes
            ? nbytes_src1_q8_1 : grouped_down_q8_bytes;
        grouped_q81_scratch = q81_grouped_persistent_acquire(
            dev, stream, grouped_q81_required, &grouped_q81_lease);
        if (!grouped_q81_scratch) {
            grouped_q81_scratch = grouped_q81_pool.alloc(
                ctx->pool(), grouped_q81_required);
        }
    }

    if (!direct_gateup_q8) {
        if (persistent_pair_maps) {
            const auto & maps = g_mmq_pair_maps[dev];
            ids_src1 = maps.ids_src1;
            ids_dst = maps.ids_dst;
            expert_bounds = maps.expert_bounds;
        } else {
            ids_src1 = ids_src1_alloc.alloc(ctx->pool(), ne_get_rows);
            ids_dst = ids_dst_alloc.alloc(ctx->pool(), ne_get_rows);
            expert_bounds = expert_bounds_alloc.alloc(
                ctx->pool(), n_experts + 1);
        }
    }

    const int si1  = n_expert_used;
    const int sis1 = 1;

    cudaError_t err = cudaSuccess;
    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/expert_map",
                ds4_mmq_nvtx_payload((uint32_t)n_tokens, (uint32_t)n_experts),
                nvtx_prefill);
        // Task #22 root-cause fix (same as ds4_mmq_moe_impl): zero the id maps
        // so entries dropped by mm_ids_helper never expose stale pool memory.
        (void)cudaMemsetAsync(ids_src1, 0, ne_get_rows * sizeof(int32_t), stream);
        (void)cudaMemsetAsync(ids_dst,  0, ne_get_rows * sizeof(int32_t), stream);
        ggml_cuda_launch_mm_ids_helper(
            ids, ids_src1, ids_dst, expert_bounds,
            n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11,
            si1, sis1, stream);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
            return -2;
        }
    }

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);
    /* The fused target-prefill path receives a true top-k assignment: one
     * token cannot select the same expert twice, so no expert bucket can
     * exceed n_tokens rows. Keep the conservative gathered-row bound for all
     * generic MMQ callers, including DSpark/MTP. */
    /* The IQ2 gate/up route is a true top-k selection: one token contributes
     * at most one row to any expert. The default gathered-row upper bound
     * overlaunches empty expert tiles by top_k; keep this opt-in until the
     * compact expert-tile launch replaces the rectangular grid entirely. */
    const bool tight_iq2_ncols =
        type == GGML_TYPE_IQ2_XXS &&
        ds4_mmq_gfx1151_flag("DS4_ROCM_MMQ_TIGHT_NCOLS", cc);
    const int64_t routed_ncols_max = (fused_down || tight_iq2_ncols)
        ? (int64_t)n_tokens
        : ne_get_rows;

    /* The materialized path stream-frees gate/up Q8_1 before allocating the
     * down Q8_1. The direct path needs both simultaneously, but writes down
     * Q8_1 into caller-owned gate scratch instead of growing the CUDA pool. */
    {
    ggml_cuda_pool_alloc<char> src1_q8_1_alloc;
    char *src1_q8_1 = direct_gateup_q8
        ? (char *)fused_down->input_q8_scratch
        : (grouped_q81_scratch
            ? grouped_q81_scratch
            : src1_q8_1_alloc.alloc(ctx->pool(), nbytes_src1_q8_1));

    // S1.1a fix (same as the dense/moe paths): zero the over-allocated mmq Y buffer
    // so the kernel's unconditional masked-out tail-tile read (mmq.cuh:3528) returns
    // a deterministic zero instead of allocator-perturbation-dependent stale memory.
    const int64_t s11_src = (int64_t)K;
    const int64_t s12_src = (int64_t)K * ne11;
    const int64_t s13_src = (int64_t)K * ne11 * ne12;
    /* p5b: token-compact input quantize on the direct fused path (the
     * materialized paths below keep the slot-gathered form: their pair/mmq
     * consumers address Y by assignment slot).
     * p5c: a producer-emitted token-compact buffer replaces even the
     * compact quantize — same layout by construction (ib = kseg*n_tokens
     * + row), so the p5b indirection consumes it unchanged. */
    const bool moe_yind = direct_gateup_q8 && moe_yind_enabled();
    const void *input_q8_ext = nullptr;
    if (moe_yind && fused_down && fused_down->input_q8_ext) {
        const size_t ext_need =
            (size_t)n_tokens * (size_t)ne10_padded * sizeof(block_q8_1) / QK8_1;
        if (fused_down->input_q8_ext_bytes >= ext_need) {
            input_q8_ext = fused_down->input_q8_ext;
        }
    }
    if (input_q8_ext) {
        src1_q8_1 = (char *)const_cast<void *>(input_q8_ext);
        static bool ext_logged = false;
        if (!ext_logged) {
            ext_logged = true;
            fprintf(stderr, "ds4: moe gateup consuming producer q8 "
                    "(flat-pool p5c, first n_tokens=%d)\n", n_tokens);
        }
    }
    const int64_t quant_rows = moe_yind ? (int64_t)n_tokens : ne_get_rows;
    if (!input_q8_ext) {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/input_quant_q8_1",
                ds4_mmq_nvtx_payload((uint32_t)quant_rows, (uint32_t)K),
                nvtx_prefill);
        ybuf_memset(src1_q8_1, nbytes_src1_q8_1, stream);
        if (use_native_fp4) {
            quantize_mmq_fp4_cuda(
                X_f32, moe_yind ? nullptr : ids_src1, (void *)src1_q8_1,
                type, /*ne00=*/K, s11_src, s12_src, s13_src,
                /*ne0=*/ne10_padded, /*ne1=*/quant_rows, /*ne2=*/1, /*ne3=*/1,
                stream);
        } else {
            quantize_mmq_q8_1_cuda(
                X_f32, moe_yind ? nullptr : ids_src1, (void *)src1_q8_1,
                type, /*ne00=*/K, s11_src, s12_src, s13_src,
                /*ne0=*/ne10_padded, /*ne1=*/quant_rows, /*ne2=*/1, /*ne3=*/1,
                stream);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: MMQ activation quantize failed: %s\n",
                    tag, cudaGetErrorString(err));
            return -3;
        }
    }

    if (direct_gateup_q8) {
        if (moe_yind) {
            static bool yind_logged = false;
            if (!yind_logged) {
                yind_logged = true;
                fprintf(stderr, "ds4: moe gateup y-indirect q8 staging engage "
                        "(flat-pool p5b, first n_tokens=%d n_assign=%lld)\n",
                        n_tokens, (long long)ne_get_rows);
            }
        }
        if (moe_yind && moe_yind_verify_enabled()) {
            /* In-situ byte-diff (p5a VERIFY pattern): run the slot-gathered
             * reference quantize into a temp buffer and compare every
             * assignment slot's blocks against the token-compact buffer
             * through ids_src1.  Instrument only - synchronous. */
            const size_t blk = sizeof(block_q8_1_mmq);
            const int nkseg = (int)(ne10_padded / (4 * QK8_1));
            const size_t ref_payload = (size_t)nkseg * (size_t)ne_get_rows * blk;
            const size_t cmp_payload = (size_t)nkseg * (size_t)n_tokens * blk;
            char *ref = nullptr;
            if (cudaMalloc((void **)&ref, ref_payload) == cudaSuccess) {
                quantize_mmq_q8_1_cuda(
                    X_f32, ids_src1, (void *)ref,
                    type, K, s11_src, s12_src, s13_src,
                    ne10_padded, ne_get_rows, 1, 1, stream);
                char *h_ref = (char *)malloc(ref_payload);
                char *h_cmp = (char *)malloc(cmp_payload);
                int32_t *h_ids = (int32_t *)malloc((size_t)ne_get_rows * sizeof(int32_t));
                if (h_ref && h_cmp && h_ids) {
                    (void)cudaMemcpyAsync(h_ref, ref, ref_payload, cudaMemcpyDeviceToHost, stream);
                    (void)cudaMemcpyAsync(h_cmp, src1_q8_1, cmp_payload, cudaMemcpyDeviceToHost, stream);
                    (void)cudaMemcpyAsync(h_ids, ids_src1, (size_t)ne_get_rows * sizeof(int32_t),
                                          cudaMemcpyDeviceToHost, stream);
                    (void)cudaStreamSynchronize(stream);
                    long long bad = 0;
                    long long first_slot = -1, first_kseg = -1;
                    for (int ks = 0; ks < nkseg; ++ks) {
                        for (int64_t slot = 0; slot < ne_get_rows; ++slot) {
                            const char *a = h_ref + ((size_t)ks * (size_t)ne_get_rows + (size_t)slot) * blk;
                            const char *b = h_cmp + ((size_t)ks * (size_t)n_tokens + (size_t)h_ids[slot]) * blk;
                            if (memcmp(a, b, blk) != 0) {
                                if (first_slot < 0) { first_slot = slot; first_kseg = ks; }
                                ++bad;
                            }
                        }
                    }
                    fprintf(stderr, "ds4: moe yind VERIFY n_tokens=%d n_assign=%lld ksegs=%d "
                            "bad=%lld/%lld first_slot=%lld first_kseg=%lld\n",
                            n_tokens, (long long)ne_get_rows, nkseg,
                            bad, (long long)nkseg * (long long)ne_get_rows,
                            first_slot, first_kseg);
                }
                free(h_ref); free(h_cmp); free(h_ids);
                (void)cudaFree(ref);
            }
        }
        ybuf_memset(fused_down->q8_scratch, direct_down_q8_bytes, stream);
        const size_t gateup_work_bytes =
            ds4_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(
                ne_get_rows, n_experts);
        if (gateup_work_bytes == 0) {
            return -9;
        }
        {
            ds4_mmq_nvtx_scope stage(
                    "ds4/prefill/moe/iq2_gate_up_swiglu_q8_d2r",
                    ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                    nvtx_prefill);
            const int d2r_rc = ds4_mmq_iq2_xxs_moe_d2r_fused_launch(
                    xa_soa, xb_soa, soa_blocks,
                    src1_q8_1,
                    moe_yind ? ids_src1 : nullptr,
                    moe_yind ? n_tokens : 0,
                    ids_dst, expert_bounds,
                    fused_down->router_weights, fused_down->q8_scratch,
                    M, K, ne_get_rows, n_experts, fused_down->clamp,
                    direct_work, gateup_work_bytes, stream);
            if (d2r_rc != 0) {
                return -10;
            }
        }

        if (out_memset_enabled()) {
            (void)cudaMemsetAsync(fused_down->out, 0,
                                  (size_t)fused_down->out_dim * (size_t)ne_get_rows * sizeof(float),
                                  stream);
        }
        const size_t down_work_bytes =
            ds4_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
        if (down_work_bytes == 0) {
            return -11;
        }
        {
            ds4_mmq_nvtx_scope stage(
                    "ds4/prefill/moe/q2_down_d2r",
                    ds4_mmq_nvtx_payload((uint32_t)ne_get_rows,
                                         (uint32_t)fused_down->out_dim),
                    nvtx_prefill);
            const int down_rc = ds4_mmq_q2_K_moe_d2r_launch(
                    fused_down->W_soa,
                    fused_down->soa_blocks,
                    fused_down->q8_scratch,
                    ids_dst, expert_bounds,
                    fused_down->out,
                    fused_down->out_dim, M, ne_get_rows, n_experts,
                    direct_work, down_work_bytes, stream);
            if (down_rc != 0) {
                return -12;
            }
        }
        return 0;
    }

    const int64_t s1      = (int64_t)M;
    const int64_t s12_mmq = ne11 * ne10_padded * y_block_size /
                            (y_values_per_block * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    if (out_memset_enabled()) {
        (void)cudaMemsetAsync(out_a, 0, (size_t)M * (size_t)ne_get_rows * sizeof(float), stream);
        (void)cudaMemsetAsync(out_b, 0, (size_t)M * (size_t)ne_get_rows * sizeof(float), stream);
    }

    bool gate_up_done = false;
    if (type == GGML_TYPE_IQ2_XXS && xa_soa != nullptr && xb_soa != nullptr &&
        d2r_enabled() && d2r_iq2_enabled() && K % 256 == 0 &&
        ne_get_rows >= d2r_min_cols()) {
        static int d2r_iq2_avail_cc = -1;
        static int d2r_iq2_avail = 0;
        if (d2r_iq2_avail_cc != cc) {
            d2r_iq2_avail_cc = cc;
            d2r_iq2_avail = ds4_mmq_iq2_xxs_moe_d2r_available(cc) ? 1 : 0;
        }
        if (d2r_iq2_avail) {
            const size_t d2r_work_bytes =
                ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(ne_get_rows, n_experts);
            if (d2r_work_bytes != 0) {
                ggml_cuda_pool_alloc<char> d2r_work(ctx->pool(), d2r_work_bytes);
                ds4_mmq_nvtx_scope stage(
                        "ds4/prefill/moe/iq2_gate_up_d2r",
                        ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                        nvtx_prefill);
                const int d2r_rc = ds4_mmq_iq2_xxs_moe_d2r_pair_launch(
                        xa_soa, xb_soa, soa_blocks, src1_q8_1, ids_dst,
                        expert_bounds, out_a, out_b, M, K, ne_get_rows, n_experts,
                        d2r_work.get(), d2r_work_bytes, stream);
                if (d2r_rc == 0) {
                    gate_up_done = true;
                }
            }
        }
    }

    if (!gate_up_done) {
    mmq_args args = {
        /*x=*/(const char *)W_a,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1,
        /*ids_dst=*/ids_dst,
        /*expert_bounds=*/expert_bounds,
        /*dst=*/out_a,
        /*ncols_x=*/ne00,
        /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/ne_get_rows,
        /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows,
        /*nrows_dst=*/s1,
        /*nchannels_x=*/(int64_t)n_experts,
        /*nchannels_y=*/(int64_t)n_experts,
        /*stride_channel_x=*/s02,
        /*stride_channel_y=*/s12_mmq,
        /*stride_channel_dst=*/(int64_t)0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/s13_mmq,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/routed_ncols_max,
        /*x_soa=*/xa_soa,
        /*soa_blocks=*/soa_blocks,
    };

    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/iq2_gate",
                ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                nvtx_prefill);
        mul_mat_q_case<type>(*ctx, args, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_q_case (pair a) launch failed: %s\n", tag, cudaGetErrorString(err));
            return -4;
        }
    }

    // Second matmul over the same activation buffer and same routing map.
    args.x     = (const char *)W_b;
    args.dst   = out_b;
    args.x_soa = xb_soa;
    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/iq2_up",
                ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                nvtx_prefill);
        mul_mat_q_case<type>(*ctx, args, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_q_case (pair b) launch failed: %s\n", tag, cudaGetErrorString(err));
            return -5;
        }
    }
    }
    }

    if (fused_down) {
        const int64_t down_ne10_padded = GGML_PAD((int64_t)M, MATRIX_ROW_PADDING);
        const size_t logical_q8_bytes =
            (size_t)ne_get_rows * (size_t)down_ne10_padded * sizeof(block_q8_1) / QK8_1;
        const size_t tail_q8_bytes =
            (size_t)get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        const size_t down_q8_bytes = logical_q8_bytes + tail_q8_bytes;
        ggml_cuda_pool_alloc<char> down_q8_1_pool;
        char *down_q8_1 = grouped_q81_scratch
            ? grouped_q81_scratch
            : down_q8_1_pool.alloc(ctx->pool(), down_q8_bytes);

        const uint64_t mid_values = (uint64_t)ne_get_rows * (uint64_t)M;
        {
            ds4_mmq_nvtx_scope stage(
                    "ds4/prefill/moe/swiglu_down_quant",
                    ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                    nvtx_prefill);
            ybuf_memset(down_q8_1, down_q8_bytes, stream);
            ds4_swiglu_weighted_f32<<<
                (uint32_t)((mid_values + 255u) / 256u), 256, 0, stream>>>(
                    out_a, out_b, fused_down->router_weights,
                    fused_down->mid_f32, mid_values, M, fused_down->clamp);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: weighted SwiGLU launch failed: %s\n",
                        tag, cudaGetErrorString(err));
                return -6;
            }

            quantize_mmq_q8_1_cuda(
                fused_down->mid_f32, ids_dst, (void *)down_q8_1,
                GGML_TYPE_Q2_K, /*ne00=*/M, /*s01=*/M,
                /*s02=*/(int64_t)M, /*s03=*/(int64_t)M * ne_get_rows,
                /*ne0=*/down_ne10_padded, /*ne1=*/ne_get_rows,
                /*ne2=*/1, /*ne3=*/1, stream);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: down quantize_mmq_q8_1_cuda failed: %s\n",
                        tag, cudaGetErrorString(err));
                return -7;
            }
        }

        if (out_memset_enabled()) {
            (void)cudaMemsetAsync(fused_down->out, 0,
                                  (size_t)fused_down->out_dim * (size_t)ne_get_rows * sizeof(float),
                                  stream);
        }
        const int64_t down_s01 = (int64_t)M / ggml_blck_size(GGML_TYPE_Q2_K);
        const int64_t down_s02 = (int64_t)fused_down->out_dim * down_s01;
        const int64_t down_s12 =
            down_ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const mmq_args down_args = {
            /*x=*/(const char *)fused_down->W,
            /*type_x=*/GGML_TYPE_Q2_K,
            /*y=*/(const int *)down_q8_1,
            /*ids_dst=*/ids_dst,
            /*expert_bounds=*/expert_bounds,
            /*dst=*/fused_down->out,
            /*ncols_x=*/(int64_t)M,
            /*nrows_x=*/(int64_t)fused_down->out_dim,
            /*ncols_dst=*/ne_get_rows,
            /*stride_row_x=*/down_s01,
            /*ncols_y=*/ne_get_rows,
            /*nrows_dst=*/(int64_t)fused_down->out_dim,
            /*nchannels_x=*/(int64_t)n_experts,
            /*nchannels_y=*/(int64_t)n_experts,
            /*stride_channel_x=*/down_s02,
            /*stride_channel_y=*/down_s12,
            /*stride_channel_dst=*/(int64_t)0,
            /*nsamples_x=*/1,
            /*nsamples_y=*/1,
            /*stride_sample_x=*/0,
            /*stride_sample_y=*/ne_get_rows * down_s12,
            /*stride_sample_dst=*/0,
            /*use_stream_k=*/use_stream_k,
            /*ncols_max=*/routed_ncols_max,
            /*x_soa=*/fused_down->W_soa,
            /*soa_blocks=*/fused_down->soa_blocks,
        };
        bool down_done = false;
        if (fused_down->W_soa != nullptr && d2r_enabled() &&
            ne_get_rows >= d2r_min_cols() &&
            ds4_mmq_q2_K_moe_d2r_available(cc)) {
            const size_t work_bytes =
                ds4_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
            if (work_bytes != 0u) {
                ggml_cuda_pool_alloc<char> work(ctx->pool(), work_bytes);
                ds4_mmq_nvtx_scope stage(
                        "ds4/prefill/moe/q2_down_d2r",
                        ds4_mmq_nvtx_payload((uint32_t)ne_get_rows,
                                             (uint32_t)fused_down->out_dim),
                        nvtx_prefill);
                down_done = ds4_mmq_q2_K_moe_d2r_launch(
                        fused_down->W_soa,
                        fused_down->soa_blocks,
                        down_q8_1,
                        ids_dst,
                        expert_bounds,
                        fused_down->out,
                        fused_down->out_dim,
                        M,
                        ne_get_rows,
                        n_experts,
                        work.get(),
                        work_bytes,
                        stream) == 0;
            }
        }
        if (!down_done) {
            ds4_mmq_nvtx_scope stage(
                    "ds4/prefill/moe/q2_down",
                    ds4_mmq_nvtx_payload((uint32_t)ne_get_rows,
                                         (uint32_t)fused_down->out_dim),
                    nvtx_prefill);
            mul_mat_q_case<GGML_TYPE_Q2_K>(*ctx, down_args, stream);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: fused Q2_K down launch failed: %s\n",
                        tag, cudaGetErrorString(err));
                return -8;
            }
        }
    }
    if (sanitize_out && type != GGML_TYPE_Q4_K) {
        ds4_mmq_sanitize_f32(out_a, (uint64_t)M * (uint64_t)ne_get_rows, stream);
        ds4_mmq_sanitize_f32(out_b, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q8_0>("ds4_mmq_q8_0_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q2_K_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_IQ2_XXS>("ds4_mmq_iq2_xxs_moe", W, X, ids, out, M, K,
                                               n_tokens, n_experts, n_expert_used, stream);
}

/* ds4 (P4 Inc3): mmq MoE over the aligned row-pair-SoA Q2_K artifact
 * (weight server --repack-q2k-aligned) -- no raw-layout weights and no
 * derepack scratch involved; the mul_mat_q tile loader reads the SoA
 * sections directly (load_tiles_q2_K_soa, bit-identical tiles). */
extern "C" int ds4_mmq_q2_K_moe_soa(
        const void * W_soa, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    if (M <= 0 || M % 2 != 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) {
        fprintf(stderr, "ds4_mmq_q2_K_moe_soa: bad shape M=%d K=%d nexp=%d\n", M, K, n_experts);
        return -1;
    }
    const int64_t npair = (int64_t)n_experts * (int64_t)(M/2) * (int64_t)(K/256);
    /* W_soa doubles as the (unused) raw pointer so the impl's null checks
     * hold.  sanitize_out=false: the routed-MoE consumers (swiglu / moe_sum)
     * sanitize at read, saving the whole-buffer pass (P3). */
    return ds4_mmq_moe_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_moe_soa", W_soa, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream,
                                            (const char *)W_soa, npair,
                                            /*sanitize_out=*/false);
}

extern "C" int ds4_mmq_q4_K_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q4_K>("ds4_mmq_q4_K_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_mxfp4_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_MXFP4>("ds4_mmq_mxfp4_moe", W, X, ids, out, M, K,
                                             n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

/* ds4 (P4 Inc3): paired mmq MoE over the aligned-SoA IQ2_XXS gate/up
 * artifacts (weight server --repack-iq2-aligned); same contract as
 * ds4_mmq_q2_K_moe_soa. */
extern "C" int ds4_mmq_iq2_xxs_moe_pair_soa(
        const void * Wa_soa, const void * Wb_soa,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    if (M <= 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) {
        fprintf(stderr, "ds4_mmq_iq2_xxs_moe_pair_soa: bad shape M=%d K=%d nexp=%d\n", M, K, n_experts);
        return -1;
    }
    const int64_t nblk = (int64_t)n_experts * (int64_t)M * (int64_t)(K/256);
    /* sanitize_out=false: see ds4_mmq_q2_K_moe_soa. */
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_soa", Wa_soa, Wb_soa, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream,
        (const char *)Wa_soa, (const char *)Wb_soa, nblk,
        /*sanitize_out=*/false);
}

/* v0.5 inc-9 (F7, derived from Marco Palaferri's GB10 fork, MIT): fused
 * target-prefill MoE pipeline over the aligned-SoA artifacts.  One
 * mm_ids_helper + one input quantize serve gate/up AND down; clamp + SwiGLU +
 * router weighting run in the pair-major mid buffer, which is gathered and
 * quantized for the Q2_K down MMQ through the same ids_dst/expert_bounds.
 * gate/up/mid/down keep the standard pair-major output layout. */
extern "C" int ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(
        const void * W_gate, const void * W_up, const void * W_down,
        const float * X, const int32_t * ids, const float * router_weights,
        float * gate, float * up, float * mid_f32, float * down,
        int expert_mid_dim, int expert_in_dim, int out_dim,
        int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    if (expert_mid_dim <= 0 || expert_in_dim <= 0 || out_dim <= 0 ||
        n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0 ||
        n_expert_used > n_experts || expert_in_dim % 256 != 0 ||
        expert_mid_dim % 256 != 0 || out_dim % 2 != 0) {
        return -1;
    }
    const int64_t iq2_blocks =
        (int64_t)n_experts * expert_mid_dim * (expert_in_dim / 256);
    const int64_t q2_pairs =
        (int64_t)n_experts * (out_dim / 2) * (expert_mid_dim / 256);
    const ds4_mmq_fused_down fused_down = {
        W_down,
        (const char *)W_down,
        q2_pairs,
        router_weights,
        mid_f32,
        down,
        out_dim,
        clamp,
        false,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
    };
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS, true>(
        "ds4_mmq_iq2_xxs_q2_K_moe_fused_soa",
        W_gate, W_up, X, ids, gate, up,
        expert_mid_dim, expert_in_dim, n_tokens, n_experts, n_expert_used,
        stream,
        (const char *)W_gate, (const char *)W_up, iq2_blocks,
        /*sanitize_out=*/false, &fused_down);
}

extern "C" int ds4_mmq_iq2_xxs_q2_K_moe_fused_direct_scratch_sizes(
        int expert_mid_dim,
        int expert_in_dim,
        int n_tokens,
        int n_experts,
        int n_expert_used,
        size_t *input_q8_bytes,
        size_t *down_q8_bytes,
        size_t *work_bytes) {
    if (expert_mid_dim <= 0 || expert_in_dim <= 0 || n_tokens <= 0 ||
        n_experts <= 0 || n_expert_used <= 0 ||
        n_expert_used > n_experts || !input_q8_bytes || !down_q8_bytes ||
        !work_bytes) {
        return -1;
    }
    auto mul_u64 = [](uint64_t a, uint64_t b, uint64_t *out) -> bool {
        if (a != 0u && b > UINT64_MAX / a) return false;
        *out = a * b;
        return true;
    };
    uint64_t rows = 0;
    if (!mul_u64((uint64_t)n_tokens, (uint64_t)n_expert_used, &rows) ||
        rows > INT_MAX) {
        return -1;
    }
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    const uint64_t x_max = (uint64_t)get_mmq_x_max_host(cc);
    const uint64_t input_padded =
        (uint64_t)GGML_PAD((int64_t)expert_in_dim, MATRIX_ROW_PADDING);
    const uint64_t mid_padded =
        (uint64_t)GGML_PAD((int64_t)expert_mid_dim, MATRIX_ROW_PADDING);
    uint64_t input_values = 0, input_storage = 0, input_slack = 0;
    uint64_t down_values = 0, down_storage = 0, down_slack = 0;
    if (!mul_u64(rows, input_padded, &input_values) ||
        !mul_u64(input_values, sizeof(block_q8_1_mmq), &input_storage) ||
        !mul_u64(x_max, sizeof(block_q8_1_mmq), &input_slack) ||
        !mul_u64(rows, mid_padded, &down_values) ||
        !mul_u64(down_values, sizeof(block_q8_1), &down_storage) ||
        !mul_u64(x_max, sizeof(block_q8_1_mmq), &down_slack)) {
        return -1;
    }
    input_storage /= 4u * QK8_1;
    down_storage /= QK8_1;
    if (input_storage > UINT64_MAX - input_slack ||
        down_storage > UINT64_MAX - down_slack) {
        return -1;
    }
    const uint64_t input_bytes = input_storage + input_slack;
    const uint64_t down_bytes = down_storage + down_slack;
    const size_t gateup_work =
        ds4_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(
            (int64_t)rows, n_experts);
    const size_t down_work = ds4_mmq_q2_K_moe_d2r_scratch_bytes(
        (int64_t)rows, n_experts);
    if (input_bytes > SIZE_MAX || down_bytes > SIZE_MAX ||
        gateup_work == 0 || down_work == 0) {
        return -1;
    }

    size_t offset = 0;
    auto reserve_aligned = [&offset](size_t bytes) -> bool {
        if (offset > SIZE_MAX - 255u) return false;
        offset = (offset + 255u) & ~(size_t)255u;
        if (bytes > SIZE_MAX - offset) return false;
        offset += bytes;
        return true;
    };
    if (!reserve_aligned((size_t)rows * sizeof(int32_t)) ||
        !reserve_aligned((size_t)rows * sizeof(int32_t)) ||
        !reserve_aligned(((size_t)n_experts + 1u) * sizeof(int32_t)) ||
        !reserve_aligned(gateup_work > down_work ? gateup_work : down_work) ||
        offset > SIZE_MAX - 255u) {
        return -1;
    }
    *input_q8_bytes = (size_t)input_bytes;
    *down_q8_bytes = (size_t)down_bytes;
    *work_bytes = offset + 255u;
    return 0;
}

/* Canonical-GGUF/raw counterpart of the materialized aligned-SoA pipeline.
 * SSD streaming compacts the selected experts and remaps ids before this
 * boundary, so n_experts describes the compact table rather than the model's
 * global expert count.  Keep all preflight ahead of the single pair_impl
 * invocation: after that point map/quantize/MMQ work may already be queued and
 * a negative result must be propagated instead of being converted into the
 * retryable NOT_APPLICABLE result. */
extern "C" int ds4_mmq_iq2_xxs_q2_K_moe_fused_raw(
        const void * W_gate, const void * W_up, const void * W_down,
        const float * X, const int32_t * ids, const float * router_weights,
        float * gate, float * up, float * mid_f32, float * down,
        int expert_mid_dim, int expert_in_dim, int out_dim,
        int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    if (!W_gate || !W_up || !W_down || !X || !ids || !router_weights ||
        !gate || !up || !mid_f32 || !down ||
        expert_mid_dim <= 0 || expert_in_dim <= 0 || out_dim <= 0 ||
        n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0 ||
        n_expert_used > n_experts || n_experts == INT_MAX ||
        n_tokens >= (1 << 22) || n_expert_used >= (1 << 10) ||
        expert_in_dim % 256 != 0 || expert_mid_dim % 256 != 0) {
        return DS4_MMQ_NOT_APPLICABLE;
    }

    const size_t nt = (size_t)n_tokens;
    const size_t nu = (size_t)n_expert_used;
    const size_t mid = (size_t)expert_mid_dim;
    const size_t in = (size_t)expert_in_dim;
    const size_t out = (size_t)out_dim;
    if (nt > SIZE_MAX / nu) return DS4_MMQ_NOT_APPLICABLE;
    const size_t assignments = nt * nu;
    if (assignments > SIZE_MAX / mid ||
        assignments * mid > SIZE_MAX / sizeof(float) ||
        assignments > SIZE_MAX / out ||
        assignments * out > SIZE_MAX / sizeof(float) ||
        assignments > SIZE_MAX / in ||
        assignments * in > SIZE_MAX / 512u ||
        assignments * mid > SIZE_MAX / 512u) {
        return DS4_MMQ_NOT_APPLICABLE;
    }

    /* Bound the raw per-expert strides used by both MMQs before any pool or
     * stream operation.  All dimensions enter as int, but their products do
     * not necessarily fit int64_t. */
    const int64_t iq2_k_blocks = expert_in_dim / 256;
    const int64_t q2_k_blocks = expert_mid_dim / 256;
    if ((int64_t)n_experts > INT64_MAX / expert_mid_dim ||
        (int64_t)n_experts * expert_mid_dim > INT64_MAX / iq2_k_blocks ||
        (int64_t)n_experts > INT64_MAX / out_dim ||
        (int64_t)n_experts * out_dim > INT64_MAX / q2_k_blocks) {
        return DS4_MMQ_NOT_APPLICABLE;
    }

    const int dev = ggml_cuda_get_device();
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_k_tile_supported<GGML_TYPE_IQ2_XXS>(
            "ds4_mmq_iq2_xxs_q2_K_moe_fused_raw", expert_in_dim, cc) ||
        !get_ctx_for_device(dev) ||
        ((size_t)n_tokens * 4u > ggml_cuda_info().devices[dev].smpbo &&
         !ds4_mmid_large_enabled())) {
        return DS4_MMQ_NOT_APPLICABLE;
    }

    const ds4_mmq_fused_down fused_down = {
        W_down,
        nullptr,
        0,
        router_weights,
        mid_f32,
        down,
        out_dim,
        clamp,
        false,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
    };
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS, true>(
        "ds4_mmq_iq2_xxs_q2_K_moe_fused_raw",
        W_gate, W_up, X, ids, gate, up,
        expert_mid_dim, expert_in_dim, n_tokens, n_experts, n_expert_used,
        stream,
        nullptr, nullptr, 0,
        /*sanitize_out=*/false, &fused_down);
}

/* Aligned-artifact production fast path: gate/up accumulators stay in
 * registers, weighted SwiGLU is quantized directly into down_q8_scratch by
 * the fused D2R kernel, and only the pair-major down output is materialized.
 * Caller-owned scratch keeps the hot path free of stream-ordered pool
 * allocations; all three ranges must be distinct and sized to their LOGICAL
 * segments (never an owning arena's capacity - the overlap guard would
 * falsely cover adjacent views). */
extern "C" int ds4_mmq_iq2_xxs_q2_K_moe_fused_direct_soa(
        const void * W_gate, const void * W_up, const void * W_down,
        const float * X, const int32_t * ids, const float * router_weights,
        void * input_q8_scratch, size_t input_q8_scratch_bytes,
        void * down_q8_scratch, size_t down_q8_scratch_bytes,
        void * work_scratch, size_t work_scratch_bytes,
        const void * input_q8_ext, size_t input_q8_ext_bytes,
        float * down,
        int expert_mid_dim, int expert_in_dim, int out_dim,
        int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    if (expert_mid_dim <= 0 || expert_in_dim <= 0 || out_dim <= 0 ||
        n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0 ||
        n_expert_used > n_experts || expert_in_dim % 256 != 0 ||
        expert_mid_dim % 256 != 0 || out_dim % 2 != 0 ||
        !input_q8_scratch || input_q8_scratch_bytes == 0 ||
        !down_q8_scratch || down_q8_scratch_bytes == 0 ||
        !work_scratch || work_scratch_bytes == 0 || !down) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const size_t nt = (size_t)n_tokens;
    const size_t nu = (size_t)n_expert_used;
    const size_t od = (size_t)out_dim;
    if (nt > SIZE_MAX / nu || nt * nu > SIZE_MAX / od ||
        nt * nu * od > SIZE_MAX / sizeof(float)) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const size_t assignments = nt * nu;
    const size_t expert_in = (size_t)expert_in_dim;
    const size_t expert_mid = (size_t)expert_mid_dim;
    if (assignments > SIZE_MAX / expert_in ||
        assignments > SIZE_MAX / expert_mid) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    /* The internal MMQ producer sizes multiply these logical element counts
     * by block structs before dividing by their values-per-block. Keep ample
     * headroom for that multiplication and its fixed tail allocation. */
    if (assignments * expert_in > SIZE_MAX / 512u ||
        assignments * expert_mid > SIZE_MAX / 512u) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const size_t down_bytes = assignments * od * sizeof(float);
    if (ds4_mmq_scratch_overlaps(
            input_q8_scratch, input_q8_scratch_bytes,
            down_q8_scratch, down_q8_scratch_bytes) ||
        ds4_mmq_scratch_overlaps(
            input_q8_scratch, input_q8_scratch_bytes,
            work_scratch, work_scratch_bytes) ||
        ds4_mmq_scratch_overlaps(
            down_q8_scratch, down_q8_scratch_bytes,
            work_scratch, work_scratch_bytes) ||
        ds4_mmq_scratch_overlaps(
            input_q8_scratch, input_q8_scratch_bytes, down, down_bytes) ||
        ds4_mmq_scratch_overlaps(
            down_q8_scratch, down_q8_scratch_bytes, down, down_bytes) ||
        ds4_mmq_scratch_overlaps(
            work_scratch, work_scratch_bytes, down, down_bytes)) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const int64_t iq2_k_blocks = expert_in_dim / 256;
    const int64_t q2_k_blocks = expert_mid_dim / 256;
    if ((int64_t)n_experts > INT64_MAX / expert_mid_dim ||
        (int64_t)n_experts * expert_mid_dim > INT64_MAX / iq2_k_blocks ||
        (int64_t)n_experts > INT64_MAX / (out_dim / 2) ||
        (int64_t)n_experts * (out_dim / 2) > INT64_MAX / q2_k_blocks) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const int64_t iq2_blocks =
        (int64_t)n_experts * expert_mid_dim * iq2_k_blocks;
    const int64_t q2_pairs =
        (int64_t)n_experts * (out_dim / 2) * q2_k_blocks;
    const ds4_mmq_fused_down fused_down = {
        W_down,
        (const char *)W_down,
        q2_pairs,
        router_weights,
        nullptr,
        down,
        out_dim,
        clamp,
        true,
        input_q8_scratch,
        input_q8_scratch_bytes,
        down_q8_scratch,
        down_q8_scratch_bytes,
        work_scratch,
        work_scratch_bytes,
        input_q8_ext,
        input_q8_ext_bytes,
    };
    const int rc = ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS, true>(
        "ds4_mmq_iq2_xxs_q2_K_moe_fused_direct_soa",
        W_gate, W_up, X, ids, nullptr, nullptr,
        expert_mid_dim, expert_in_dim, n_tokens, n_experts, n_expert_used,
        stream,
        (const char *)W_gate, (const char *)W_up, iq2_blocks,
        /*sanitize_out=*/false, &fused_down);
    if (rc == -1 || (rc <= -91 && rc >= -97)) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    return rc;
}

extern "C" int ds4_mmq_q4_K_moe_pair(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_mxfp4_moe_pair(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_MXFP4>(
        "ds4_mmq_mxfp4_moe_pair", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

// ----------------------------------------------------------------------------
// mmvq-backed entry points (Step 6 of the optimization plan).
//
// mmvq is upstream's matrix-vector matmul family, optimised for the
// n_tokens <= MMVQ_MAX_BATCH_SIZE=8 regime. Unlike mmq it consumes the
// CANONICAL block_q8_1 layout (via quantize_row_q8_1_cuda), not the
// interleaved block_q8_1_mmq that quantize_mmq_q8_1_cuda produces.
//
// The single-W _moe_vec entries cover:
//   - the down matmul at decode (treating [n_tokens=1, n_expert_used=6]
//     as [n_tokens=6, n_expert_used=1])
//   - dense attention projections at decode (n_tokens=1, no MoE)
//   - any small-batch path where mmvq's per-token grid wins over mmq's
//     tile-based approach
//
// The pair-fused _moe_pair_vec entries cover the gate+up matmuls at
// decode using mmvq's built-in fusion. fusion.gate is the up_w pointer
// and fusion.glu_op is GGML_GLU_OP_SWIGLU - the kernel computes
// silu(gate@x) * (up@x) in a single launch. mmvq's fusion is supported
// only at ncols_dst=1, so n_tokens=1 is the only valid case.
// ----------------------------------------------------------------------------

#include "mmvq.cuh"

namespace {

template <ggml_type type>
int ds4_mmq_moe_vec_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }
    // mmvq's per-arch batch cap. ncols_dst as computed below is
    // max(n_tokens, n_expert_used) depending on which dim we route into.
    // We follow upstream's convention: ne_y = n_tokens, ne_dst = n_expert_used.
    // So ncols_dst = n_tokens and nchannels_dst = n_expert_used.
    // FD Inc2a: n_tokens beyond the per-launch column cap no longer rejects;
    // the launch loop below splits the column dim into capped chunks.

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's cudaMallocAsync / cudaFreeAsync through the same
    // stream the caller uses for kernel launches.  Required for Step 8
    // (CUDA Graph capture): pool allocations on a different stream than
    // the capture stream would invalidate the capture.
    ds4_pool_set_stream(stream);

    // 1. Quantize X into CANONICAL Q8_1 (NOT the MMQ-interleaved variant).
    //    Layout: [ne13=1, ne12=n_tokens, ne11=1, ne10_padded blocks].
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    // Step 7 task #29: experimental persistent Q8_1 scratch. It avoids
    // captured pool alloc/free nodes as a performance experiment. When
    // disabled (default) or too small, the valid same-stream graph-memory
    // pool path remains the fallback. See ds4_mmq_init for setup.
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
        g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    // s11 = stride between rows of an src1 channel in source-float units.
    //       Logical src1 [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost.
    // s12 = stride between channels = K * ne11 = K.
    // s13 = stride between samples = K * ne11 * ne12 = K * n_tokens.
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. mmvq stride setup. Mirror upstream's ggml_cuda_mul_mat_vec_q
    //    dispatch (mmvq.cu:1101-1136).
    //
    //    For MoE (ids != nullptr): per the dispatch math at line 1121-1130,
    //      ncols_dst          = ne2  = n_tokens
    //      nchannels_y        = ne11 = 1
    //      nchannels_dst      = ne1  = n_expert_used
    //      stride_col_y       = s12  = ne11 * (ne10_padded / QK8_1)
    //      stride_col_dst     = s2   = n_expert_used * M (token stride in dst)
    //      stride_channel_y   = s11  = ne10_padded / QK8_1
    //      stride_channel_dst = s1   = M (channel/slot stride in dst)
    //      ids_stride         = stride between rows of ids[] tensor
    //
    //    FD Inc2a stride fix: stride_col_dst was previously M, same as the
    //    channel stride.  That was invisible while every caller degenerated
    //    one dim (gate/up at n_tokens=1: col index always 0; down at
    //    n_expert_used=1: channel index always 0, and 1 * M == M keeps it
    //    bit-identical here).  At n_tokens >= 2 with n_expert_used > 1 the
    //    multi-token MoE kernel writes dst[chan*s1 + col*s2 + row], and
    //    equal strides collide (token=0,slot=1) with (token=1,slot=0).
    //    s2 = n_expert_used * M yields the row-major
    //    [token * n_expert_used + slot, M] layout the swiglu consumer
    //    expects.
    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;            // weight row stride in blocks
    const int64_t s02_chan  = (int64_t)M * s01_row;         // expert-stack stride
    const int64_t s11_y     = ne10_padded / QK8_1;          // src1 channel stride in blocks
    const int64_t s12_y     = (int64_t)1 * s11_y;           // ne11 * s11
    const int64_t s1_dst    = (int64_t)M;                   // dst channel (slot) stride
    const int64_t s2_dst    = (int64_t)n_expert_used * M;   // dst col (token) stride

    // ids_stride: stride between rows of the ids tensor in int32 elements.
    // Caller passes ids[t * n_expert_used + s], so stride between tokens
    // is n_expert_used.
    const int ids_stride = n_expert_used;

    ggml_cuda_mm_fusion_args_device fusion = {};

    (void)cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)n_tokens * (size_t)n_expert_used * sizeof(float), stream);

    // FD Inc2a: one mmvq launch serves at most col_cap columns -- the moe
    // kernel runs one warp per column (block.y = ncols_dst) under
    // __launch_bounds__ baked per COMPILED arch + type
    // (get_mmvq_mmid_max_batch_for_device).  The runtime device cc can
    // exceed the compiled arch (CUDA_ARCH= builds run default-arch PTX on
    // newer GPUs), so the host cap MUST be looked up at the compiled arch:
    // asking the runtime cc says 8 where the compiled bounds say 7 (e.g.
    // Q2_K builds at turing_plus -> 7*warp_size threads) and the launch
    // dies with cudaErrorInvalidValue.  Wider batches run as
    // ceil(n_tokens / col_cap) launches; every per-column stride (vy, ids,
    // dst) is uniform, so a chunk is plain pointer offsets.  The single
    // quantize above already covers all columns.
    const int cc      = ggml_cuda_info().devices[dev].cc;
    const int col_cap = get_mmvq_mmid_max_batch(type, ggml_cuda_highest_compiled_arch(cc));

    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        mul_mat_vec_q_switch_type(
            /*vx=*/W, /*type_x=*/type,
            /*vy=*/(const void *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1)),
            /*ids=*/ids + (size_t)c0 * ids_stride, /*fusion=*/fusion,
            /*dst=*/out_f32 + (int64_t)c0 * s2_dst,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type launch failed: %s (cols %d..%d cap %d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -3;
        }
    }

    ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)n_tokens * (uint64_t)n_expert_used, stream);
    return 0;
}

// ---------------------------------------------------------------------------
// Aligned-SoA IQ2_XXS decode matvec (megakernel program M1-Inc1).
//
// Layout contract (see ds4_mmq.h): W_aligned = [__half dq[nblk]][pad to 64B]
// [uint2 qs[nblk*8]], nblk = n_experts * M * (K/256), block linear order equal
// to the raw tensor byte order.  Per-pair integer math is bit-identical to
// vec_dot_iq2_xxs_q8_1 (vecdotq.cuh); only the float accumulation order
// differs (per-warp-row here vs per-mmvq-tile there).  Proven +12% over the
// raw-layout vec path at the production decode shape
// (cuda/mmq/test/proto_iq2_aligned.cu).
__global__ void iq2_xxs_aligned_moe_vec_kernel(
        float             *out,        // [n_tokens*n_expert_used, M]
        const uint2       *qs,         // 64B-aligned code pairs
        const __half      *dq,         // block scales
        const block_q8_1  *x8,         // [n_tokens][nyb] canonical Q8_1 activations
        const int32_t     *ids,        // [n_tokens*n_expert_used] expert ids
        int                M,
        int                nb,         // IQ2_XXS blocks per row = K/256
        int                nyb,        // Q8_1 blocks per activation row
        int                n_expert_used)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;       // flat assignment = token*n_expert_used+slot
    const int lane = threadIdx.x;      // 32 lanes: lane covers (block b, pair p)
    // The router's NaN path emits -1 expert ids by design (same guard as
    // mul_mat_vec_q_moe): clamp the pointer math to expert 0, skip the dot
    // loop, write a clean 0.
    const int32_t id_raw = ids[slot];
    const bool invalid_id = id_raw < 0;
    const long long rbase = ((long long)(invalid_id ? 0 : id_raw) * M + row) * nb;
    x8 += (long long)(slot / n_expert_used) * nyb;

    float acc = 0.0f;
    // 32 lanes cover 4 blocks x 8 pairs per pass.
    for (int b0 = 0; !invalid_id && b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const uint2 cw   = qs[(rbase + b) * 8 + p];   // aligned 8B load
        const uint32_t q2 = cw.x, aux32 = cw.y;
        const uint8_t *aux8 = (const uint8_t *)&q2;

        int sumi = 0;
        const int q8i = (b * 256 + p * 32) / 32;   // q8_1 block covering these 32 values
        const int *u = (const int *)x8[q8i].qs;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
            sumi = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi);
        }
        const int ls = aux32 >> 27 | 1;
        sumi = sumi * ls / 8;
        const float d = __half2float(dq[rbase + b]) * __low2float(x8[q8i].ds);
        acc += d * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[(long long)slot * M + row] = acc;
}

// M1-Inc2 variant P: one launch covers gate and up (blockIdx.z selects the
// weight stream); nonfinite accs are zeroed in-kernel so no sanitize pass is
// needed.  Same per-warp math as iq2_xxs_aligned_moe_vec_kernel.
__global__ void iq2_xxs_aligned_moe_pair_vec_kernel(
        float             *out_gate,   // [n_tokens*n_expert_used, M]
        float             *out_up,     // [n_tokens*n_expert_used, M]
        const uint2       *qs_gate,
        const __half      *dq_gate,
        const uint2       *qs_up,
        const __half      *dq_up,
        const block_q8_1  *x8,         // [n_tokens][nyb]
        const int32_t     *ids,        // [n_tokens*n_expert_used]
        int                M,
        int                nb,
        int                nyb,
        int                n_expert_used)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;       // flat assignment = token*n_expert_used+slot
    const int lane = threadIdx.x;
    const uint2  *qs = blockIdx.z ? qs_up : qs_gate;
    const __half *dq = blockIdx.z ? dq_up : dq_gate;
    float        *out = blockIdx.z ? out_up : out_gate;
    const int32_t id_raw = ids[slot];
    const bool invalid_id = id_raw < 0;
    const long long rbase = ((long long)(invalid_id ? 0 : id_raw) * M + row) * nb;
    x8 += (long long)(slot / n_expert_used) * nyb;

    float acc = 0.0f;
    for (int b0 = 0; !invalid_id && b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const uint2 cw   = qs[(rbase + b) * 8 + p];
        const uint32_t q2 = cw.x, aux32 = cw.y;
        const uint8_t *aux8 = (const uint8_t *)&q2;

        int sumi = 0;
        const int q8i = (b * 256 + p * 32) / 32;
        const int *u = (const int *)x8[q8i].qs;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
            sumi = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi);
        }
        const int ls = aux32 >> 27 | 1;
        sumi = sumi * ls / 8;
        const float d = __half2float(dq[rbase + b]) * __low2float(x8[q8i].ds);
        acc += d * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        if (!isfinite(acc)) acc = 0.0f;
        out[(long long)slot * M + row] = acc;
    }
}

// M1-Inc2 variant F: gate and up accumulated in the same warp (interleaved so
// each q8 activation block is loaded once), clamp/SwiGLU/router-weight
// epilogue folded in (semantics copied from
// ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel) -> mid directly.  Replaces
// quantize+gate+up+sanitize+swiglu with quantize+one launch.
__global__ void iq2_xxs_aligned_moe_gate_up_mid_kernel(
        float             *mid,        // [n_tokens*n_expert_used, M]
        const uint2       *qs_gate,
        const __half      *dq_gate,
        const uint2       *qs_up,
        const __half      *dq_up,
        const block_q8_1  *x8,         // [n_tokens][nyb]
        const int32_t     *ids,        // [n_tokens*n_expert_used]
        const float       *weights,    // [n_tokens*n_expert_used] router weights
        int                M,
        int                nb,
        int                nyb,
        int                n_expert_used,
        float              clamp)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;       // flat assignment = token*n_expert_used+slot
    const int lane = threadIdx.x;
    const int32_t id_raw = ids[slot];
    const bool invalid_id = id_raw < 0;
    const long long rbase = ((long long)(invalid_id ? 0 : id_raw) * M + row) * nb;
    x8 += (long long)(slot / n_expert_used) * nyb;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int b0 = 0; !invalid_id && b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const int q8i = (b * 256 + p * 32) / 32;
        const int *u = (const int *)x8[q8i].qs;
        const float d8 = __low2float(x8[q8i].ds);

        const uint2 cwg = qs_gate[(rbase + b) * 8 + p];
        const uint2 cwu = qs_up[(rbase + b) * 8 + p];
        const uint8_t *aux8g = (const uint8_t *)&cwg.x;
        const uint8_t *aux8u = (const uint8_t *)&cwu.x;

        int sumi_g = 0;
        int sumi_u = 0;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            {
                const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8g[k0 / 2]];
                const uint32_t signs = unpack_ksigns(cwg.y >> (7 * k0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
                sumi_g = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi_g);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
                sumi_g = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi_g);
            }
            {
                const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8u[k0 / 2]];
                const uint32_t signs = unpack_ksigns(cwu.y >> (7 * k0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
                sumi_u = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi_u);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
                sumi_u = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi_u);
            }
        }
        const int ls_g = cwg.y >> 27 | 1;
        const int ls_u = cwu.y >> 27 | 1;
        acc_g += __half2float(dq_gate[rbase + b]) * d8 * (float)(sumi_g * ls_g / 8);
        acc_u += __half2float(dq_up[rbase + b])   * d8 * (float)(sumi_u * ls_u / 8);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_g += __shfl_down_sync(0xffffffffu, acc_g, off);
        acc_u += __shfl_down_sync(0xffffffffu, acc_u, off);
    }
    if (lane == 0) {
        float gate = acc_g;
        float up = acc_u;
        if (!isfinite(gate)) gate = 0.0f;
        if (!isfinite(up)) up = 0.0f;
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const float silu = gate / (1.0f + expf(-gate));
        mid[(long long)slot * M + row] = silu * up * weights[slot];
    }
}

// v0.4 V6: expert-overlap dedup for the gate_up mid kernel at DSpark
// verify widths (proto_gemm_gateup_iq2xxs_dedup).  A live census measured
// a mean of 18.2 DISTINCT experts per 30 assignment slots at w5 (~40%
// overlap across the verify tokens); the per-slot kernel above re-reads
// every duplicate's weights from DRAM.  First-owner dedup keeps the grid
// at (M, n_slots) -- capture-safe (the decision replays from LIVE ids
// content inside baked MoE graphs), sort-free, no host id knowledge:
// each CTA exits unless it is the first slot bearing its expert id, and
// otherwise accumulates ALL matching slots (<= n_tokens; top-k is
// without replacement) as extra q8_1 columns.  Weight bytes and the iq2
// grid/sign decode collapse to distinct experts.  Per-slot int dots are
// exact and float folds stay block-major => outputs are BITWISE the
// per-slot kernel's (proto: 0 mismatches on every leg incl. invalid-id
// sign-zeros; timing D=18 1.53x, D=12 2.03x, D=30 0.93x -- the
// no-overlap tail is ~9% of live launches and priced).
template <int MAXM>
__global__ void iq2_xxs_aligned_moe_gate_up_mid_dedup_kernel(
        float             *mid,
        const uint2       *qs_gate,
        const __half      *dq_gate,
        const uint2       *qs_up,
        const __half      *dq_up,
        const block_q8_1  *x8,
        const int32_t     *ids,
        const float       *weights,
        int                M,
        int                nb,
        int                nyb,
        int                n_expert_used,
        int                n_slots,
        float              clamp)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;
    const int lane = threadIdx.x;
    const int32_t id_raw = ids[slot];

    if (id_raw < 0) {
        /* The per-slot kernel's zero path runs the epilogue with acc 0:
         * (+0)*(+0)*w = sign(w)*0 -- keep the sign bitwise. */
        if (lane == 0) mid[(long long)slot * M + row] = 0.0f * weights[slot];
        return;
    }
    for (int j = 0; j < slot; j++)
        if (ids[j] == id_raw) return;

    int msl[MAXM];
    const block_q8_1 *xcol[MAXM];
    int nm = 0;
    for (int j = slot; j < n_slots && nm < MAXM; j++)
        if (ids[j] == id_raw) {
            msl[nm] = j;
            xcol[nm] = x8 + (long long)(j / n_expert_used) * nyb;
            nm++;
        }

    const long long rbase = ((long long)id_raw * M + row) * nb;

    float acc_g[MAXM];
    float acc_u[MAXM];
#pragma unroll
    for (int m = 0; m < MAXM; m++) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

    for (int b0 = 0; b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const int q8i = (b * 256 + p * 32) / 32;

        const uint2 cwg = qs_gate[(rbase + b) * 8 + p];
        const uint2 cwu = qs_up[(rbase + b) * 8 + p];
        const uint8_t *aux8g = (const uint8_t *)&cwg.x;
        const uint8_t *aux8u = (const uint8_t *)&cwu.x;

        int sumi_g[MAXM];
        int sumi_u[MAXM];
#pragma unroll
        for (int m = 0; m < MAXM; m++) { sumi_g[m] = 0; sumi_u[m] = 0; }

#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            int g0g, g1g, g0u, g1u;
            {
                const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8g[k0 / 2]];
                const uint32_t signs = unpack_ksigns(cwg.y >> (7 * k0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                g0g = __vsub4(grid_pos.x ^ signs0, signs0);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                g1g = __vsub4(grid_pos.y ^ signs1, signs1);
            }
            {
                const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8u[k0 / 2]];
                const uint32_t signs = unpack_ksigns(cwu.y >> (7 * k0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                g0u = __vsub4(grid_pos.x ^ signs0, signs0);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                g1u = __vsub4(grid_pos.y ^ signs1, signs1);
            }
#pragma unroll
            for (int m = 0; m < MAXM; m++) {
                if (m < nm) {
                    const int *u = (const int *)xcol[m][q8i].qs;
                    sumi_g[m] = ggml_cuda_dp4a(g0g, u[k0 + 0], sumi_g[m]);
                    sumi_g[m] = ggml_cuda_dp4a(g1g, u[k0 + 1], sumi_g[m]);
                    sumi_u[m] = ggml_cuda_dp4a(g0u, u[k0 + 0], sumi_u[m]);
                    sumi_u[m] = ggml_cuda_dp4a(g1u, u[k0 + 1], sumi_u[m]);
                }
            }
        }
        const int ls_g = cwg.y >> 27 | 1;
        const int ls_u = cwu.y >> 27 | 1;
        const float dg = __half2float(dq_gate[rbase + b]);
        const float du = __half2float(dq_up[rbase + b]);
#pragma unroll
        for (int m = 0; m < MAXM; m++) {
            if (m < nm) {
                const float d8 = __low2float(xcol[m][q8i].ds);
                acc_g[m] += dg * d8 * (float)(sumi_g[m] * ls_g / 8);
                acc_u[m] += du * d8 * (float)(sumi_u[m] * ls_u / 8);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAXM; m++) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc_g[m] += __shfl_down_sync(0xffffffffu, acc_g[m], off);
            acc_u[m] += __shfl_down_sync(0xffffffffu, acc_u[m], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAXM; m++) {
            if (m < nm) {
                float gate = acc_g[m];
                float up = acc_u[m];
                if (!isfinite(gate)) gate = 0.0f;
                if (!isfinite(up)) up = 0.0f;
                if (clamp > 1.0e-6f) {
                    if (gate > clamp) gate = clamp;
                    if (up > clamp) up = clamp;
                    if (up < -clamp) up = -clamp;
                }
                const float silu = gate / (1.0f + expf(-gate));
                mid[(long long)msl[m] * M + row] = silu * up * weights[msl[m]];
            }
        }
    }
}

template <ggml_type type>
int ds4_mmq_moe_pair_raw_vec_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_a,
        float         * out_b,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W_a || !W_b || !X_f32 || !ids || !out_a || !out_b) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s02_chan  = (int64_t)M * s01_row;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)1 * s11_y;
    const int64_t s1_dst    = (int64_t)M;
    const int64_t s2_dst    = (int64_t)n_expert_used * M;
    const int ids_stride    = n_expert_used;
    const int cc            = ggml_cuda_info().devices[dev].cc;
    const int col_cap       = get_mmvq_mmid_max_batch(type, ggml_cuda_highest_compiled_arch(cc));
    ggml_cuda_mm_fusion_args_device fusion = {};

    const size_t out_bytes = (size_t)M * (size_t)n_tokens * (size_t)n_expert_used * sizeof(float);
    (void)cudaMemsetAsync(out_a, 0, out_bytes, stream);
    (void)cudaMemsetAsync(out_b, 0, out_bytes, stream);

    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        const void *vy = (const void *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1));
        const int32_t *ids_chunk = ids + (size_t)c0 * ids_stride;
        float *out_a_chunk = out_a + (int64_t)c0 * s2_dst;
        float *out_b_chunk = out_b + (int64_t)c0 * s2_dst;

        mul_mat_vec_q_switch_type(
            /*vx=*/W_a, /*type_x=*/type,
            /*vy=*/vy, /*ids=*/ids_chunk, /*fusion=*/fusion,
            /*dst=*/out_a_chunk,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type (a) failed: %s (cols %d..%d cap %d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -3;
        }

        mul_mat_vec_q_switch_type(
            /*vx=*/W_b, /*type_x=*/type,
            /*vy=*/vy, /*ids=*/ids_chunk, /*fusion=*/fusion,
            /*dst=*/out_b_chunk,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type (b) failed: %s (cols %d..%d cap %d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -4;
        }
    }

    const uint64_t out_count = (uint64_t)M * (uint64_t)n_tokens * (uint64_t)n_expert_used;
    ds4_mmq_sanitize_f32(out_a, out_count, stream);
    ds4_mmq_sanitize_f32(out_b, out_count, stream);
    return 0;
}

template <ggml_type type>
int ds4_mmq_moe_pair_vec_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_silu,
        int             M,
        int             K,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W_a || !W_b || !X_f32 || !ids || !out_silu) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d nexp=%d nused=%d\n",
                tag, M, K, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's cudaMallocAsync through the caller-supplied stream
    // for Step 8 / CUDA Graph compatibility.  See ds4_mmq_moe_vec_impl.
    ds4_pool_set_stream(stream);

    const int n_tokens = 1;  // fusion only supported at ncols_dst=1.

    // Quantize X (single token) into canonical Q8_1.
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_q8_1);

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s02_chan  = (int64_t)M * s01_row;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)1 * s11_y;
    const int64_t s1_dst    = (int64_t)M;
    const int ids_stride    = n_expert_used;

    // Configure fusion: gate=W_b (up weights), glu_op=SWIGLU.
    // mmvq's kernel will compute, for each (channel_dst, row):
    //   a = vec_dot(W_a, x); b = vec_dot(W_b, x);
    //   dst = silu(a) * b
    ggml_cuda_mm_fusion_args_device fusion = {};
    fusion.gate   = W_b;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;

    (void)cudaMemsetAsync(out_silu, 0, (size_t)M * (size_t)n_expert_used * sizeof(float), stream);

    mul_mat_vec_q_switch_type(
        /*vx=*/W_a, /*type_x=*/type,
        /*vy=*/(const void *)src1_q8_1.get(),
        /*ids=*/ids, /*fusion=*/fusion,
        /*dst=*/out_silu,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/n_tokens,
        /*stride_row_x=*/(int)s01_row,
        /*stride_col_y=*/(int)s12_y,
        /*stride_col_dst=*/(int)s1_dst,
        /*nchannels_x=*/n_experts,
        /*nchannels_y=*/1,
        /*nchannels_dst=*/n_expert_used,
        /*stride_channel_x=*/(int)s02_chan,
        /*stride_channel_y=*/(int)s11_y,
        /*stride_channel_dst=*/(int)s1_dst,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/ids_stride, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_vec_q_switch_type (fused) launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    ds4_mmq_sanitize_f32(out_silu, (uint64_t)M * (uint64_t)n_expert_used, stream);
    return 0;
}

/* Diagnostic counters are host-dispatch counters.  CUDA graph replays do not
 * re-enter this wrapper, so they are deliberately not presented as kernel
 * execution counts.  They are still a fail-closed coverage signal: a GB10
 * model run must observe at least one candidate and one use before this path
 * can be promoted from opt-in to default. */
static uint64_t g_q4_k1024_persistent_candidates;
static uint64_t g_q4_k1024_persistent_uses;
static uint64_t g_q4_k1024_persistent_fallbacks;
static uint64_t g_q4_k1024_persistent_require_failures;
static uint64_t g_q4_k1024_persistent_oracle_calls;
static uint64_t g_q4_k1024_persistent_oracle_mismatches;
static uint64_t g_q4_k1024_persistent_oracle_skips;
static int g_q4_k1024_persistent_report_registered;
static int g_q4_k1024_persistent_oracle_mismatch_reported;

static bool q4_k1024_env_flag(const char *name) {
    const char *value = getenv(name);
    return value && value[0] && strcmp(value, "0") != 0;
}

static void q4_k1024_persistent_report(void) {
    fprintf(stderr,
            "ds4: CUDA Q4 K1024 persistent: "
            "candidates=%llu uses=%llu fallbacks=%llu "
            "require_failures=%llu oracle_calls=%llu "
            "oracle_mismatches=%llu oracle_skips=%llu "
            "(host dispatches; graph replays excluded, canonical oracle output retained)\n",
            (unsigned long long)g_q4_k1024_persistent_candidates,
            (unsigned long long)g_q4_k1024_persistent_uses,
            (unsigned long long)g_q4_k1024_persistent_fallbacks,
            (unsigned long long)g_q4_k1024_persistent_require_failures,
            (unsigned long long)g_q4_k1024_persistent_oracle_calls,
            (unsigned long long)g_q4_k1024_persistent_oracle_mismatches,
            (unsigned long long)g_q4_k1024_persistent_oracle_skips);
}

static void q4_k1024_persistent_maybe_register_report(void) {
    if (!g_q4_k1024_persistent_report_registered &&
        (q4_k1024_env_flag("DS4_CUDA_Q4_K1024_PERSISTENT_STATS") ||
         q4_k1024_env_flag("DS4_CUDA_Q4_K1024_PERSISTENT_ORACLE"))) {
        g_q4_k1024_persistent_report_registered = 1;
        (void)atexit(q4_k1024_persistent_report);
    }
}

__global__ static void q4_K_k1024_bitwise_compare_kernel(
        uint32_t    *mismatch,
        const float *candidate,
        const float *reference,
        uint64_t     count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count &&
        __float_as_uint(candidate[i]) != __float_as_uint(reference[i])) {
        atomicExch(mismatch, 1u);
    }
}

/* GB10 AProjQ4 Q-b decode specialization (M=32768, N=1, K=1024).
 *
 * The canonical MMVQ small-K launch uses four warps to evaluate four rows:
 * warp 0 owns Q4_K superblocks 0/1, warp 1 owns 2/3, and warps 2/3
 * contribute +0.0f.  Its reduction first adds the three peer-warp partials
 * lane by lane, then applies warp_reduce_sum's XOR tree.  Two independent
 * four-warp groups below preserve that assignment and arithmetic order while
 * persistent CTAs walk eight-row tiles at a grid stride.  The immutable
 * canonical Q8_1 activation is staged once per CTA; no Q8_K re-quantization
 * or Q4_K weight repack is involved.
 *
 * Keep this kernel paired with the exact M/N/K admission in
 * ds4_mmq_dense_vec_impl.  Generalizing the row-warp mapping would change
 * floating-point association relative to MMVQ. */
static __global__ __launch_bounds__(256, 4) void
q4_K_dense_vec_k1024_persistent_kernel(
        const block_q4_K * __restrict__ W,
        const block_q8_1 * __restrict__ x8,
        float            * __restrict__ out,
        int                              M) {
    constexpr int k_q4_blocks = 4;       /* 1024 / QK_K */
    constexpr int k_q8_blocks = 32;      /* 1024 / QK8_1 */
    constexpr int k_rows_per_group = 4;  /* canonical MMVQ small-K tile */
    constexpr int k_groups = 2;

    /* block_q8_1 is 36 bytes.  A uint32_t backing array both copies it
     * efficiently and preserves the alignment required by vec_dot's int
     * loads from qs. */
    __shared__ __align__(16) uint32_t x8_words[
        (k_q8_blocks * sizeof(block_q8_1)) / sizeof(uint32_t)];
    __shared__ float partial[k_groups][3][k_rows_per_group][32];

    const uint32_t *x8_src = (const uint32_t *)x8;
    for (uint32_t i = threadIdx.x;
         i < (uint32_t)(sizeof(x8_words) / sizeof(x8_words[0]));
         i += blockDim.x) {
        x8_words[i] = x8_src[i];
    }
    __syncthreads();

    const block_q8_1 *x8_shared = (const block_q8_1 *)x8_words;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t group = warp >> 2u;
    const uint32_t warp_in_group = warp & 3u;
    const uint32_t group_tid = warp_in_group * 32u + lane;
    const uint64_t row_tiles = ((uint64_t)(uint32_t)M + 7u) / 8u;

    /* tile, row_tiles, and gridDim.x are block-uniform, and this loop has no
     * divergent exit.  Every thread therefore reaches both barriers below on
     * every iteration; unrolling is unrelated to their correctness. */
    for (uint64_t tile = blockIdx.x; tile < row_tiles; tile += gridDim.x) {
        const uint32_t row0 = (uint32_t)(tile * 8u) +
                              group * k_rows_per_group;
        float tmp[k_rows_per_group] = {0.0f};

        /* This is the canonical N=1, K=1024 MMVQ small-K loop verbatim:
         * qi/vdr = 16 and blocks_per_iter = 8 for Q4_K. */
        const int kqs = VDR_Q4_K_Q8_1_MMVQ * (int)(group_tid % 16u);
        for (int kbx = (int)(group_tid / 16u);
             kbx < k_q4_blocks;
             kbx += 8) {
            const int kby = kbx * (QK_K / QK8_1);
#pragma unroll
            for (int i = 0; i < k_rows_per_group; ++i) {
                tmp[i] += vec_dot_q4_K_q8_1(
                    W, &x8_shared[kby],
                    (int)((uint64_t)(row0 + (uint32_t)i) * k_q4_blocks) + kbx,
                    kqs);
            }
        }

        if (warp_in_group > 0u) {
#pragma unroll
            for (int i = 0; i < k_rows_per_group; ++i) {
                partial[group][warp_in_group - 1u][i][lane] = tmp[i];
            }
        }
        __syncthreads();

        if (warp_in_group == 0u) {
#pragma unroll
            for (int i = 0; i < k_rows_per_group; ++i) {
#pragma unroll
                for (int peer = 0; peer < 3; ++peer) {
                    tmp[i] += partial[group][peer][i][lane];
                }
                tmp[i] = warp_reduce_sum<32>(tmp[i]);
            }
            if (lane < k_rows_per_group) {
                out[row0 + lane] = tmp[lane];
            }
        }
        /* Both groups must finish consuming partial before the next
         * grid-stride tile reuses it. */
        __syncthreads();
    }
}

static bool ds4_mmq_q4_epilogue_eligible(
        ggml_type type, int M, int N, int K, int device) {
#if !defined(GGML_USE_HIP)
    if (type != GGML_TYPE_Q4_K || !ds4_q4_mmvq_epilogue_shape_ok(M, N, K)) return false;
    const int cc = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING &&
           getenv("DS4_CUDA_DISABLE_Q4_MMVQ_EPILOGUE") == nullptr;
#else
    GGML_UNUSED_VARS(type, M, N, K, device);
    return false;
#endif
}

// Single and shared-activation pair projections use the same dense layout.
// Keep backend selection and stride construction in one place; callers own
// output clearing, sanitization, and launch error handling.
template <ggml_type type>
static void ds4_mmq_launch_dense_vec(
        const void *W, const void *x8, float *out,
        int M, int N, int K, int64_t stride_col_y,
        bool fused_epilogue, cudaStream_t stream) {
#if !defined(GGML_USE_HIP)
    if (fused_epilogue) {
        ds4_mmvq_q4_K_dense_sanitized(W, x8, out, M, N, K, (int)stride_col_y, stream);
        return;
    }
#else
    GGML_UNUSED_VARS(fused_epilogue);
#endif
    const ggml_cuda_mm_fusion_args_device fusion = {};
    mul_mat_vec_q_switch_type(
        /*vx=*/W, /*type_x=*/type, /*vy=*/x8,
        /*ids=*/nullptr, /*fusion=*/fusion, /*dst=*/out,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/N,
        /*stride_row_x=*/(int)((int64_t)K / ggml_blck_size(type)),
        /*stride_col_y=*/(int)stride_col_y, /*stride_col_dst=*/M,
        /*nchannels_x=*/1, /*nchannels_y=*/1, /*nchannels_dst=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/(int)((int64_t)N * stride_col_y),
        /*stride_channel_dst=*/0,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/0, stream);
}

template <ggml_type type>
int ds4_mmq_dense_vec_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        int           q4_weight_device_resident,
        cudaStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (N > MMVQ_MAX_BATCH_SIZE) {
        fprintf(stderr, "%s: N=%d exceeds MMVQ_MAX_BATCH_SIZE=%d\n",
                tag, N, MMVQ_MAX_BATCH_SIZE);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    /* Resolve the exact-shape admission before allocating pool storage,
     * quantizing X, clearing output, or launching any kernel.  REQUIRE and
     * the oracle are coverage gates: an ineligible candidate must therefore
     * fail without leaving work queued on the caller's stream. */
    bool q4_k1024_exact = false;
    bool q4_k1024_eligible = false;
    bool q4_k1024_oracle = false;
    unsigned q4_k1024_grid = 0u;
    if constexpr (type == GGML_TYPE_Q4_K) {
        q4_k1024_persistent_maybe_register_report();
        q4_k1024_exact = M == 32768 && N == 1 && K == 1024;
        q4_k1024_oracle = q4_k1024_exact &&
            q4_k1024_env_flag("DS4_CUDA_Q4_K1024_PERSISTENT_ORACLE");
        const bool enable =
            q4_k1024_env_flag("DS4_CUDA_ENABLE_Q4_K1024_PERSISTENT");
        const bool disable =
            q4_k1024_env_flag("DS4_CUDA_NO_Q4_K1024_PERSISTENT") ||
            q4_k1024_env_flag("DS4_CUDA_NO_Q4_GB10_FAST");
        if (q4_k1024_exact) {
            g_q4_k1024_persistent_candidates++;
        }
        if (q4_k1024_exact && gb10_optimizations_enabled() &&
            (enable || q4_k1024_oracle) && !disable &&
            q4_weight_device_resident > 0 &&
            (((uintptr_t)W & 15u) == 0u)) {
            const uint64_t row_tiles = ((uint64_t)(uint32_t)M + 7u) / 8u;
            const int nsm = ggml_cuda_info().devices[dev].nsm;
            const uint64_t resident_blocks =
                nsm > 0 ? (uint64_t)(uint32_t)nsm * 4u : 0u;
            const uint64_t grid64 = row_tiles < resident_blocks
                ? row_tiles : resident_blocks;
            if (grid64 > 0u && grid64 <= UINT32_MAX) {
                q4_k1024_grid = (unsigned)grid64;
                q4_k1024_eligible = true;
            }
        }
        if (q4_k1024_exact && !q4_k1024_eligible) {
            g_q4_k1024_persistent_fallbacks++;
            const bool require =
                q4_k1024_env_flag(
                    "DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT");
            if (require || q4_k1024_oracle) {
                g_q4_k1024_persistent_require_failures++;
                if (q4_k1024_oracle) {
                    g_q4_k1024_persistent_oracle_skips++;
                }
                fprintf(stderr,
                        "%s: required Q4_K K1024 persistent path unavailable "
                        "before enqueue\n",
                        tag);
                return -4;
            }
        }
        if (q4_k1024_eligible && q4_k1024_oracle) {
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            const cudaError_t capture_err =
                cudaStreamIsCapturing(stream, &capture);
            if (capture_err != cudaSuccess ||
                capture != cudaStreamCaptureStatusNone) {
                (void)cudaGetLastError();
                g_q4_k1024_persistent_fallbacks++;
                g_q4_k1024_persistent_require_failures++;
                g_q4_k1024_persistent_oracle_skips++;
                fprintf(stderr,
                        "%s: Q4_K K1024 persistent oracle refuses CUDA "
                        "graph capture before enqueue; run the oracle with "
                        "DS4_CUDA_DECODE_GRAPHS=0\n",
                        tag);
                return -5;
            }
        }
    }

    // Route the pool's cudaMallocAsync through the caller-supplied stream
    // for Step 8 / CUDA Graph compatibility.  See ds4_mmq_moe_vec_impl.
    ds4_pool_set_stream(stream);

    /* Oracle-only storage.  Graph capture was rejected above, so the pool
     * allocations and the host readback below cannot become graph nodes.
     * The persistent candidate writes here; canonical MMVQ always owns the
     * caller-visible output. */
    ggml_cuda_pool_alloc<float> q4_k1024_candidate;
    ggml_cuda_pool_alloc<uint32_t> q4_k1024_mismatch;
    if (q4_k1024_eligible && q4_k1024_oracle) {
        q4_k1024_candidate.alloc(ctx->pool(), (size_t)M);
        q4_k1024_mismatch.alloc(ctx->pool(), 1u);
    }

    // Dense: no MoE, ids=null. Layout [K, N, 1, 1] for src1.
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)N * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1;
    char *x8 = nullptr;
    if constexpr (type == GGML_TYPE_Q4_K) {
        if (gb10_optimizations_enabled() &&
            getenv("DS4_CUDA_NO_Q4_GB10_FAST") == nullptr &&
            getenv("DS4_CUDA_NO_Q4_DENSE_SCRATCH") == nullptr) {
            x8 = (char *)ds4_mmq_aligned_q81_scratch(dev, nbytes_q8_1);
        }
    }
    if (!x8) {
        src1_q8_1.alloc(ctx->pool(), nbytes_q8_1);
        x8 = src1_q8_1.get();
    }

    // Dense src1 layout: K innermost, N next; ne11=N, ne12=1, ne13=1.
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)x8,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K * N, /*s13=*/(int64_t)K * N,
        /*ne0=*/ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    // Dense N=1..8 MMVQ fully overwrites aligned row cohorts. Sanitize in the
    // output-owning lane to avoid both the pre-clear and the post-read kernel.
    // Leave the independent persistent experiment and its oracle untouched.
    const bool fused_epilogue = !q4_k1024_eligible &&
        ds4_mmq_q4_epilogue_eligible(type, M, N, K, dev);
    if (!fused_epilogue)
        (void)cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)N * sizeof(float), stream);

    bool q4_k1024_persistent = false;
    if constexpr (type == GGML_TYPE_Q4_K) {
        if (q4_k1024_eligible) {
            float *candidate_out = q4_k1024_oracle
                ? q4_k1024_candidate.get() : out_f32;
            q4_K_dense_vec_k1024_persistent_kernel<<<
                q4_k1024_grid, 256, 0, stream>>>(
                    (const block_q4_K *)W,
                    (const block_q8_1 *)x8,
                    candidate_out, M);
            g_q4_k1024_persistent_uses++;
            q4_k1024_persistent = !q4_k1024_oracle;
        }
    }

    if (!q4_k1024_persistent) {
        ds4_mmq_launch_dense_vec<type>(
            W, x8, out_f32, M, N, K, ne10_padded / QK8_1, fused_epilogue, stream);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_vec_q_switch_type (dense) launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    const uint64_t out_count = (uint64_t)M * (uint64_t)N;
    if (!fused_epilogue) ds4_mmq_sanitize_f32(out_f32, out_count, stream);
    if (q4_k1024_oracle) {
        ds4_mmq_sanitize_f32(q4_k1024_candidate.get(), out_count, stream);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "%s: Q4_K K1024 oracle sanitize failed\n", tag);
            g_q4_k1024_persistent_oracle_skips++;
            return -6;
        }
        cudaError_t oracle_err = cudaMemsetAsync(
            q4_k1024_mismatch.get(), 0, sizeof(uint32_t), stream);
        if (oracle_err == cudaSuccess) {
            q4_K_k1024_bitwise_compare_kernel<<<
                (unsigned)((out_count + 255u) / 256u), 256, 0, stream>>>(
                    q4_k1024_mismatch.get(), q4_k1024_candidate.get(),
                    out_f32, out_count);
            oracle_err = cudaGetLastError();
        }
        uint32_t mismatch_host = 0u;
        if (oracle_err == cudaSuccess) {
            oracle_err = cudaMemcpyAsync(
                &mismatch_host, q4_k1024_mismatch.get(), sizeof(uint32_t),
                cudaMemcpyDeviceToHost, stream);
        }
        if (oracle_err == cudaSuccess) {
            oracle_err = cudaStreamSynchronize(stream);
        }
        if (oracle_err != cudaSuccess) {
            fprintf(stderr,
                    "%s: Q4_K K1024 persistent oracle failed: %s\n",
                    tag, cudaGetErrorString(oracle_err));
            (void)cudaGetLastError();
            g_q4_k1024_persistent_oracle_skips++;
            return -6;
        }
        g_q4_k1024_persistent_oracle_calls++;
        if (mismatch_host != 0u) {
            g_q4_k1024_persistent_oracle_mismatches++;
            if (!g_q4_k1024_persistent_oracle_mismatch_reported) {
                g_q4_k1024_persistent_oracle_mismatch_reported = 1;
                fprintf(stderr,
                        "%s: Q4_K K1024 persistent oracle found a bitwise "
                        "mismatch; retained canonical MMVQ output\n",
                        tag);
            }
        }
    }
    return 0;
}

template <ggml_type type>
int ds4_mmq_dense_pair_vec_impl(
        const char  * tag,
        const void  * W0,
        const void  * W1,
        const float * X_f32,
        float       * out0_f32,
        float       * out1_f32,
        int           M0,
        int           M1,
        int           N,
        int           K,
        cudaStream_t  stream) {

    if (!W0 || !W1 || !X_f32 || !out0_f32 || !out1_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    if (M0 <= 0 || M1 <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "%s: bad shape M0=%d M1=%d N=%d K=%d\n",
                tag, M0, M1, N, K);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return DS4_MMQ_NOT_APPLICABLE;
    }
    if (N > MMVQ_MAX_BATCH_SIZE) {
        fprintf(stderr, "%s: N=%d exceeds MMVQ_MAX_BATCH_SIZE=%d\n",
                tag, N, MMVQ_MAX_BATCH_SIZE);
        return DS4_MMQ_NOT_APPLICABLE;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n",
                tag, dev);
        return DS4_MMQ_NOT_APPLICABLE;
    }

    ds4_pool_set_stream(stream);

    /* Match ds4_mmq_dense_vec_impl's activation layout and quantizer exactly,
     * but retain the Q8_1 row for both projections. */
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t nbytes_q8_1 = (size_t)N * ne10_padded *
                               sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1;
    char *x8 = nullptr;
    if constexpr (type == GGML_TYPE_Q4_K) {
        if (gb10_optimizations_enabled() &&
            getenv("DS4_CUDA_NO_Q4_GB10_FAST") == nullptr &&
            getenv("DS4_CUDA_NO_Q4_DENSE_SCRATCH") == nullptr) {
            x8 = (char *)ds4_mmq_aligned_q81_scratch(dev, nbytes_q8_1);
        }
    }
    if (!x8) {
        src1_q8_1.alloc(ctx->pool(), nbytes_q8_1);
        x8 = src1_q8_1.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)x8,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K * N,
        /*s13=*/(int64_t)K * N,
        /*ne0=*/ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t s11_y = ne10_padded / QK8_1;

    // Each aligned single-token leg fuses its own sanitizer. Keep both original
    // matvec launches and their arithmetic; only setup/epilogue work is removed.
    const bool fused0 = ds4_mmq_q4_epilogue_eligible(type, M0, N, K, dev);
    const bool fused1 = ds4_mmq_q4_epilogue_eligible(type, M1, N, K, dev);
    if (!fused0) cudaMemsetAsync(out0_f32, 0,
                    (size_t)M0 * (size_t)N * sizeof(float), stream);
    ds4_mmq_launch_dense_vec<type>(
        W0, x8, out0_f32, M0, N, K, s11_y, fused0, stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: first dense MMVQ launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    if (!fused0) ds4_mmq_sanitize_f32(out0_f32, (uint64_t)M0 * (uint64_t)N, stream);

    if (!fused1) cudaMemsetAsync(out1_f32, 0,
                    (size_t)M1 * (size_t)N * sizeof(float), stream);
    ds4_mmq_launch_dense_vec<type>(
        W1, x8, out1_f32, M1, N, K, s11_y, fused1, stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: second dense MMVQ launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -4;
    }
    if (!fused1) ds4_mmq_sanitize_f32(out1_f32, (uint64_t)M1 * (uint64_t)N, stream);
    return 0;
}

__global__ static void ds4_mmq_group_ids_i32_kernel(
        int32_t *ids, int n, int n_groups) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) ids[i] = i % n_groups;
}

/* Grouped AProjQ4 attention-A projection.  Flatten (token, group) into the
 * MMVQ channel dimension (never the column dimension): ncols_dst stays one,
 * so every pair uses exactly the same one-row Q4_K MMVQ specialization, K
 * partition, peer-warp fold, and reduction tree as the canonical nested
 * token/group loop. The legacy repeated ids select W[group]; the fused Q4
 * specialization derives the same group from grid.y and sanitizes at store.
 * Both retain the token-major activation/output indices. */
static int ds4_mmq_q4_K_grouped_batch_vec_impl(
        const void  *W,
        const float *X,
        float       *out,
        int          M,
        int          K,
        int          n_tokens,
        int          n_groups,
        cudaStream_t stream) {
    const char *tag = n_tokens == 1
        ? "ds4_mmq_q4_K_grouped_vec"
        : "ds4_mmq_q4_K_grouped_batch_vec";
    if (!W || !X || !out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (!gb10_optimizations_enabled() ||
        getenv("DS4_CUDA_NO_Q4_GB10_FAST") != nullptr ||
        getenv("DS4_CUDA_NO_Q4_GROUPED_ATTN_A") != nullptr ||
        M <= 0 || K <= 0 || n_tokens <= 0 || n_tokens > 8 ||
        n_groups <= 0 || n_groups > 16 ||
        K % 256 != 0) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    if (n_tokens > 1) {
        const char *enable =
            getenv("DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH");
        if (!enable || !enable[0] || strcmp(enable, "0") == 0 ||
            getenv("DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH") != nullptr) {
            return DS4_MMQ_NOT_APPLICABLE;
        }
    }

    const int flat_channels = n_tokens * n_groups; /* <= 8 * 16 */

    const int64_t row_blocks = (int64_t)K / ggml_blck_size(GGML_TYPE_Q4_K);
    const int64_t weight_channel_stride = (int64_t)M * row_blocks;
    if (row_blocks <= 0 || row_blocks > INT_MAX ||
        weight_channel_stride > INT_MAX) {
        return DS4_MMQ_NOT_APPLICABLE;
    }

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t q8_row_bytes = (size_t)ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    if ((size_t)flat_channels > SIZE_MAX / q8_row_bytes) {
        return DS4_MMQ_NOT_APPLICABLE;
    }
    const size_t nbytes_q8_1 = (size_t)flat_channels * q8_row_bytes;
    if (nbytes_q8_1 > SIZE_MAX - 15u) return DS4_MMQ_NOT_APPLICABLE;
    const size_t ids_offset = (nbytes_q8_1 + 15u) & ~(size_t)15u;
    const size_t ids_bytes = (size_t)flat_channels * sizeof(int32_t);
    if (ids_offset > SIZE_MAX - ids_bytes) {
        return DS4_MMQ_NOT_APPLICABLE;
    }

    const int dev = ggml_cuda_get_device();
    const int64_t y_channel_stride = ne10_padded / QK8_1;
    // Preserve the scratch-capacity contract in both arms so rollback never
    // needs a larger arena, even though the fused arm does not populate ids.
    char *x8 = (char *)ds4_mmq_aligned_q81_scratch(
        dev, ids_offset + ids_bytes);
    if (!x8) return DS4_MMQ_NOT_APPLICABLE;
    bool fused_grouped = false;
#if !defined(GGML_USE_HIP)
    if (ds4_q4_mmvq_grouped_shape_ok(M, K, n_tokens, n_groups, y_channel_stride)) {
        const int cc = ggml_cuda_info().devices[dev].cc;
        fused_grouped = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING &&
            getenv("DS4_CUDA_DISABLE_Q4_GROUPED_MMVQ_FUSION") == nullptr;
    }
#endif
    int32_t *ids = (int32_t *)(x8 + ids_offset);
    cudaError_t err = cudaSuccess;
    if (!fused_grouped) {
        ds4_mmq_group_ids_i32_kernel<<<
            (unsigned)(flat_channels + 31) / 32u, 32, 0, stream>>>(
                ids, flat_channels, n_groups);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: group-id launch failed: %s\n",
                    tag, cudaGetErrorString(err));
            return -2;
        }
    }

    quantize_row_q8_1_cuda(
        X, /*ids=*/nullptr, (void *)x8,
        GGML_TYPE_Q4_K, /*ne00=*/K,
        /*s11=*/(int64_t)K,
        /*s12=*/(int64_t)K * flat_channels,
        /*s13=*/(int64_t)K * flat_channels,
        /*ne0=*/ne10_padded, /*ne1=*/flat_channels, /*ne2=*/1, /*ne3=*/1,
        stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }

#if !defined(GGML_USE_HIP)
    if (fused_grouped) {
        ds4_mmvq_q4_K_grouped_sanitized(
            W, x8, out, M, K, n_tokens, n_groups, (int)y_channel_stride, stream);
    } else
#endif
    {
        ggml_cuda_mm_fusion_args_device fusion = {};
        err = cudaMemsetAsync(
            out, 0, (size_t)flat_channels * (size_t)M * sizeof(float), stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: output clear failed: %s\n",
                    tag, cudaGetErrorString(err));
            return -4;
        }
        mul_mat_vec_q_switch_type(
            /*vx=*/W, /*type_x=*/GGML_TYPE_Q4_K,
            /*vy=*/(const void *)x8,
            /*ids=*/ids, /*fusion=*/fusion,
            /*dst=*/out,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/1,
            /*stride_row_x=*/(int)row_blocks,
            /*stride_col_y=*/(int)y_channel_stride,
            /*stride_col_dst=*/M,
            /*nchannels_x=*/n_groups,
            /*nchannels_y=*/flat_channels,
            /*nchannels_dst=*/flat_channels,
            /*stride_channel_x=*/(int)weight_channel_stride,
            /*stride_channel_y=*/(int)y_channel_stride,
            /*stride_channel_dst=*/M,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0,
            /*stride_sample_dst=*/0,
            /*ids_stride=*/1, stream);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: grouped MMVQ launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -5;
    }
    if (!fused_grouped) {
        ds4_mmq_sanitize_f32(
            out, (uint64_t)(uint32_t)flat_channels * (uint64_t)(uint32_t)M,
            stream);
    }
    return 0;
}

template <ggml_type type> struct ds4_mmq_vdr_mmvq_value;
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_IQ2_XXS> { static constexpr int value = VDR_IQ2_XXS_Q8_1_MMVQ; };
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_Q2_K>    { static constexpr int value = VDR_Q2_K_Q8_1_MMVQ; };
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_Q4_K>    { static constexpr int value = VDR_Q4_K_Q8_1_MMVQ; };
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_MXFP4>   { static constexpr int value = VDR_MXFP4_Q8_1_MMVQ; };

template <ggml_type type>
static __device__ __forceinline__ float ds4_mmq_vec_dot_q8_1(
        const void * __restrict__ W,
        const block_q8_1 * __restrict__ X_q8,
        const int & kbx,
        const int & iqs) {
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return vec_dot_iq2_xxs_q8_1(W, X_q8, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        return vec_dot_q2_K_q8_1(W, X_q8, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        return vec_dot_q4_K_q8_1(W, X_q8, kbx, iqs);
    } else {
        static_assert(type == GGML_TYPE_MXFP4, "unsupported fused vector type");
        return vec_dot_mxfp4_q8_1(W, X_q8, kbx, iqs);
    }
}

static __device__ __forceinline__ float ds4_mmq_half_warp_sum_f32(float v) {
    const uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    return v;
}

template <ggml_type type>
static __global__ void ds4_mmq_moe_down_sum6_q8_1_qwarp32_kernel(
        const void       * __restrict__ W,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        float            * __restrict__ out,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x) {

    constexpr int top_k = 6;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int q8_per_k = qk / QK8_1;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int lanes_per_k = qi / vdr;
    constexpr int blocks_per_iter = vdr * 16 / qi;
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row_lane = threadIdx.x >> 4u;
    const uint32_t tok  = blockIdx.y;
    if (tok >= n_tokens) return;

    const uint32_t blocks_per_row_x = ncols_x / qk;
    const uint32_t kbx0 = lane / lanes_per_k;
    const int kqs = vdr * (lane % lanes_per_k);

#pragma unroll
    for (uint32_t rr = 0; rr < 8u; ++rr) {
        const uint32_t row = blockIdx.x * 64u + row_lane + rr * 8u;
        if (row >= nrows_x) continue;
        float total = 0.0f;
#pragma unroll
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            const uint32_t assignment = tok * top_k + slot;
            const int32_t id_raw = ids[assignment];
            const bool invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
            const uint32_t expert = invalid_id ? 0u : (uint32_t)id_raw;
            const block_q8_1 * xq = X_q8 + (uint64_t)assignment * stride_col_y;
            const int kbx_base = (int)(expert * stride_channel_x + row * stride_row_x);
            float acc = 0.0f;
            for (uint32_t b = kbx0; !invalid_id && b < blocks_per_row_x; b += blocks_per_iter) {
                acc += ds4_mmq_vec_dot_q8_1<type>(
                    W, xq + (uint64_t)b * q8_per_k, kbx_base + (int)b, kqs);
            }
            acc = ds4_mmq_half_warp_sum_f32(acc);
            if (lane == 0) {
                if (!isfinite(acc)) acc = 0.0f;
                total += acc;
            }
        }
        if (lane == 0) {
            out[(uint64_t)tok * nrows_x + row] = total;
        }
    }
}

template <ggml_type type>
static __global__ void ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel(
        const void       * __restrict__ W_gate,
        const void       * __restrict__ W_up,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        const float      * __restrict__ weights,
        float            * __restrict__ mid,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x,
        const float clamp) {

    constexpr int top_k = 6;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int q8_per_k = qk / QK8_1;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int lanes_per_k = qi / vdr;
    constexpr int blocks_per_iter = vdr * 16 / qi;
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row_lane = threadIdx.x >> 4u;
    const uint32_t assignment = blockIdx.y;
    const uint32_t tok = assignment / top_k;
    const uint32_t slot = assignment - tok * top_k;
    if (tok >= n_tokens) return;

    const int32_t id_raw = ids[(uint64_t)tok * top_k + slot];
    const bool invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert = invalid_id ? 0u : (uint32_t)id_raw;
    const block_q8_1 * xq = X_q8 + (uint64_t)tok * stride_col_y;
    const uint32_t blocks_per_row_x = ncols_x / qk;
    const uint32_t kbx0 = lane / lanes_per_k;
    const int kqs = vdr * (lane % lanes_per_k);

#pragma unroll
    for (uint32_t rr = 0; rr < 4u; ++rr) {
        const uint32_t row = blockIdx.x * 64u + row_lane + rr * 16u;
        if (row >= nrows_x) continue;
        const int kbx_base = (int)(expert * stride_channel_x + row * stride_row_x);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = kbx0; !invalid_id && b < blocks_per_row_x; b += blocks_per_iter) {
            const block_q8_1 * xb = xq + (uint64_t)b * q8_per_k;
            const int kbx = kbx_base + (int)b;
            gate += ds4_mmq_vec_dot_q8_1<type>(W_gate, xb, kbx, kqs);
            up   += ds4_mmq_vec_dot_q8_1<type>(W_up,   xb, kbx, kqs);
        }
        gate = ds4_mmq_half_warp_sum_f32(gate);
        up   = ds4_mmq_half_warp_sum_f32(up);
        if (lane == 0) {
            if (!isfinite(gate)) gate = 0.0f;
            if (!isfinite(up)) up = 0.0f;
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const float silu = gate / (1.0f + expf(-gate));
            mid[(uint64_t)assignment * nrows_x + row] = silu * up * weights[(uint64_t)tok * top_k + slot];
        }
    }
}

template <ggml_type type, int c_rows_per_block>
static __global__ void ds4_mmq_moe_down_sum6_vec_kernel(
        const void       * __restrict__ W,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        float            * __restrict__ out,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x) {

    constexpr int top_k = 6;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t slot  = threadIdx.y;
    const uint32_t token = blockIdx.y;
    const uint32_t row0  = c_rows_per_block * blockIdx.x;

    if (slot >= top_k || token >= n_tokens) {
        return;
    }

    const uint32_t assignment = token * top_k + slot;
    const int32_t  id_raw     = ids[assignment];
    const bool     invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert     = invalid_id ? 0u : (uint32_t)id_raw;

    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const block_q8_1 * y = X_q8 + (uint64_t)assignment * stride_col_y;
    const int kbx_offset = (int)(expert * stride_channel_x + row0 * stride_row_x);

    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi / vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += ds4_mmq_vec_dot_q8_1<type>(
                W, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    __shared__ float partial[top_k][c_rows_per_block];
    if (threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        partial[slot][threadIdx.x] = row < nrows_x ? tmp[threadIdx.x] : 0.0f;
    }
    __syncthreads();

    if (slot == 0 && threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        if (row < nrows_x) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < top_k; ++s) {
                sum += partial[s][threadIdx.x];
            }
            out[(uint64_t)token * nrows_x + row] = sum;
        }
    }
}

template <ggml_type type>
int ds4_mmq_moe_down_sum6_vec_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used != 6) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);

    const int n_assignments = n_tokens * n_expert_used;
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t nbytes_q8_1 = (size_t)n_assignments * ne10_padded *
                               sizeof(block_q8_1) / QK8_1;

    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_assignments,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_assignments, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck = ggml_blck_size(type);
    const uint32_t stride_row_x     = (uint32_t)((int64_t)K / blck);
    const uint32_t stride_col_y     = (uint32_t)(ne10_padded / QK8_1);
    const uint32_t stride_channel_x = (uint32_t)((int64_t)M * stride_row_x);

    const dim3 block_nums((M + 63) / 64, n_tokens);
    const dim3 block_dims(128);

    ds4_mmq_moe_down_sum6_q8_1_qwarp32_kernel<type><<<block_nums, block_dims, 0, stream>>>(
        W, (const block_q8_1 *)src1_q8_1_ptr, ids, out_f32,
        (uint32_t)K, (uint32_t)M, (uint32_t)n_tokens, (uint32_t)n_experts,
        stride_row_x, stride_col_y, stride_channel_x);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: fused down+sum launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }

    return 0;
}

template <ggml_type type, int c_rows_per_block>
static __global__ void ds4_mmq_moe_gate_up_mid_vec_kernel(
        const void       * __restrict__ W_gate,
        const void       * __restrict__ W_up,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        const float      * __restrict__ weights,
        float            * __restrict__ mid,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x,
        const float clamp) {

    constexpr int top_k = 6;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t slot  = threadIdx.y;
    const uint32_t token = blockIdx.y;
    const uint32_t row0  = c_rows_per_block * blockIdx.x;

    const uint32_t assignment = token * top_k + slot;
    const int32_t  id_raw     = ids[assignment];
    const bool     invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert     = invalid_id ? 0u : (uint32_t)id_raw;

    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const block_q8_1 * y = X_q8 + (uint64_t)token * stride_col_y;
    const int kbx_offset = (int)(expert * stride_channel_x + row0 * stride_row_x);

    float gate[c_rows_per_block] = {0.0f};
    float up[c_rows_per_block]   = {0.0f};

    for (int kbx = threadIdx.x / (qi / vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            const int row_kbx = kbx_offset + i * stride_row_x + kbx;
            gate[i] += ds4_mmq_vec_dot_q8_1<type>(W_gate, &y[kby], row_kbx, kqs);
            up[i]   += ds4_mmq_vec_dot_q8_1<type>(W_up,   &y[kby], row_kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        gate[i] = warp_reduce_sum<warp_size>(gate[i]);
        up[i]   = warp_reduce_sum<warp_size>(up[i]);
    }

    if (threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        if (row < nrows_x) {
            float g = gate[threadIdx.x];
            float u = up[threadIdx.x];
            if (!isfinite(g)) g = 0.0f;
            if (!isfinite(u)) u = 0.0f;
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const float silu = g / (1.0f + expf(-g));
            mid[(uint64_t)assignment * nrows_x + row] = silu * u * weights[assignment];
        }
    }
}

template <ggml_type type, int c_rows_per_block>
static __global__ void ds4_mmq_moe_gate_up_mid_vec_by_slot_kernel(
        const void       * __restrict__ W_gate,
        const void       * __restrict__ W_up,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        const float      * __restrict__ weights,
        float            * __restrict__ mid,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x,
        const uint32_t token0,
        const float clamp) {

    constexpr int top_k = 6;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t slot  = blockIdx.y;
    const uint32_t token = token0 + threadIdx.y;
    const uint32_t row0  = c_rows_per_block * blockIdx.x;

    const uint32_t assignment = token * top_k + slot;
    const int32_t  id_raw     = ids[assignment];
    const bool     invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert     = invalid_id ? 0u : (uint32_t)id_raw;

    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const block_q8_1 * y = X_q8 + (uint64_t)token * stride_col_y;
    const int kbx_offset = (int)(expert * stride_channel_x + row0 * stride_row_x);

    float gate[c_rows_per_block] = {0.0f};
    float up[c_rows_per_block]   = {0.0f};

    for (int kbx = threadIdx.x / (qi / vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            const int row_kbx = kbx_offset + i * stride_row_x + kbx;
            gate[i] += ds4_mmq_vec_dot_q8_1<type>(W_gate, &y[kby], row_kbx, kqs);
            up[i]   += ds4_mmq_vec_dot_q8_1<type>(W_up,   &y[kby], row_kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        gate[i] = warp_reduce_sum<warp_size>(gate[i]);
        up[i]   = warp_reduce_sum<warp_size>(up[i]);
    }

    if (threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        if (row < nrows_x) {
            float g = gate[threadIdx.x];
            float u = up[threadIdx.x];
            if (!isfinite(g)) g = 0.0f;
            if (!isfinite(u)) u = 0.0f;
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const float silu = g / (1.0f + expf(-g));
            mid[(uint64_t)assignment * nrows_x + row] = silu * u * weights[assignment];
        }
    }
}

template <ggml_type type>
int ds4_mmq_moe_gate_up_mid_vec_impl(
        const char    * tag,
        const void    * W_gate,
        const void    * W_up,
        const float   * X_f32,
        const int32_t * ids,
        const float   * weights,
        float         * mid_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        float           clamp,
        cudaStream_t    stream) {

    if (!W_gate || !W_up || !X_f32 || !ids || !weights || !mid_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used != 6) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                               sizeof(block_q8_1) / QK8_1;

    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    // M2-Inc2a: the fused HC stage may have emitted this activation's q8_1
    // codes already (ffn_norm) -- take them and skip the quantize prelude.
    char *src1_q8_1_ptr = ds4_mmq_folded_q81(
        X_f32, K, n_tokens, ne10_padded, stream);
    const bool folded_hit = src1_q8_1_ptr != nullptr;
    cudaError_t err;
    if (!src1_q8_1_ptr) {
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }
    }

    const int64_t blck = ggml_blck_size(type);
    const uint32_t stride_row_x     = (uint32_t)((int64_t)K / blck);
    const uint32_t stride_col_y     = (uint32_t)(ne10_padded / QK8_1);
    const uint32_t stride_channel_x = (uint32_t)((int64_t)M * stride_row_x);

    const dim3 block_nums((M + 63) / 64, n_tokens * n_expert_used);
    const dim3 block_dims(256);
    if (folded_hit && ds4_mmq_q8_fold_oracle_enabled()) {
        const size_t q8_bytes =
            (size_t)ne10_padded * sizeof(block_q8_1) / QK8_1;
        const uint64_t mid_count =
            (uint64_t)M * (uint64_t)n_tokens * (uint64_t)n_expert_used;
        const size_t mid_bytes = (size_t)mid_count * sizeof(float);
        block_q8_1 *fresh = nullptr;
        float *reference = nullptr;
        uint32_t *mismatch_device = nullptr;
        const bool allocated =
            cudaMalloc((void **)&fresh, q8_bytes) == cudaSuccess &&
            cudaMalloc((void **)&reference, mid_bytes) == cudaSuccess &&
            cudaMalloc((void **)&mismatch_device, sizeof(uint32_t)) == cudaSuccess &&
            fresh && reference && mismatch_device;
        if (!allocated) {
            (void)cudaGetLastError();
            cudaError_t cleanup_err = cudaSuccess;
            cleanup_err = ds4_mmq_q8_fold_oracle_free(
                fresh, "raw-moe-fresh", cleanup_err);
            cleanup_err = ds4_mmq_q8_fold_oracle_free(
                reference, "raw-moe-reference", cleanup_err);
            cleanup_err = ds4_mmq_q8_fold_oracle_free(
                mismatch_device, "raw-moe-mismatch", cleanup_err);
            if (cleanup_err != cudaSuccess) (void)cudaGetLastError();
            g_q8_fold_oracle_skips++;
        } else {
            cudaError_t oracle_err = cudaMemsetAsync(
                mismatch_device, 0, sizeof(uint32_t), stream);
            if (oracle_err == cudaSuccess) {
                quantize_row_q8_1_cuda(
                    X_f32, /*ids=*/nullptr, fresh, type,
                    /*ne00=*/K, /*s11=*/K, /*s12=*/K, /*s13=*/K,
                    /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/1, /*ne3=*/1,
                    stream);
                oracle_err = cudaGetLastError();
            }
            if (oracle_err == cudaSuccess) {
                ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel<type><<<
                    block_nums, block_dims, 0, stream>>>(
                        W_gate, W_up,
                        (const block_q8_1 *)src1_q8_1_ptr,
                        ids, weights, mid_f32,
                        (uint32_t)K, (uint32_t)M,
                        (uint32_t)n_tokens, (uint32_t)n_experts,
                        stride_row_x, stride_col_y, stride_channel_x, clamp);
                oracle_err = cudaGetLastError();
            }
            if (oracle_err == cudaSuccess) {
                ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel<type><<<
                    block_nums, block_dims, 0, stream>>>(
                        W_gate, W_up, fresh, ids, weights, reference,
                        (uint32_t)K, (uint32_t)M,
                        (uint32_t)n_tokens, (uint32_t)n_experts,
                        stride_row_x, stride_col_y, stride_channel_x, clamp);
                oracle_err = cudaGetLastError();
            }
            if (oracle_err == cudaSuccess) {
                q8_fold_output_compare_kernel<<<
                    (unsigned)((mid_count + 255u) / 256u), 256, 0, stream>>>(
                        mismatch_device, mid_f32, reference, mid_count);
                oracle_err = cudaGetLastError();
            }
            if (oracle_err == cudaSuccess) {
                oracle_err = cudaMemcpyAsync(
                    mid_f32, reference, mid_bytes,
                    cudaMemcpyDeviceToDevice, stream);
            }
            uint32_t mismatch_host = 0u;
            if (oracle_err == cudaSuccess) {
                oracle_err = cudaMemcpyAsync(
                    &mismatch_host, mismatch_device, sizeof(mismatch_host),
                    cudaMemcpyDeviceToHost, stream);
            }
            if (oracle_err == cudaSuccess) {
                oracle_err = cudaStreamSynchronize(stream);
            }
            oracle_err = ds4_mmq_q8_fold_oracle_free(
                fresh, "raw-moe-fresh", oracle_err);
            oracle_err = ds4_mmq_q8_fold_oracle_free(
                reference, "raw-moe-reference", oracle_err);
            oracle_err = ds4_mmq_q8_fold_oracle_free(
                mismatch_device, "raw-moe-mismatch", oracle_err);
            if (oracle_err != cudaSuccess) {
                (void)cudaGetLastError();
                g_q8_fold_oracle_skips++;
                fprintf(stderr, "%s: fold consumer oracle failed\n", tag);
                return -3;
            }
            g_q8_fold_oracle_output_calls++;
            g_q8_fold_oracle_raw_moe_calls++;
            if (mismatch_host != 0u) {
                g_q8_fold_oracle_output_mismatches++;
                fprintf(stderr,
                        "ds4: CUDA Q8_1 fold oracle found a raw MoE "
                        "consumer output mismatch; retained canonical "
                        "output\n");
            }
            return 0;
        }
    }
    ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel<type><<<block_nums, block_dims, 0, stream>>>(
        W_gate, W_up, (const block_q8_1 *)src1_q8_1_ptr, ids, weights, mid_f32,
        (uint32_t)K, (uint32_t)M, (uint32_t)n_tokens, (uint32_t)n_experts,
        stride_row_x, stride_col_y, stride_channel_x, clamp);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: fused gate+up qwarp launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }

    return 0;
}

__global__ static void ds4_mmq_q4_K_dense_pair_vec_kernel(
        const void       * __restrict__ W0,
        const void       * __restrict__ W1,
        const block_q8_1 * __restrict__ X_q8,
        float            * __restrict__ out0,
        float            * __restrict__ out1,
        uint32_t                         M,
        uint32_t                         K) {
    constexpr ggml_type type = GGML_TYPE_Q4_K;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int lanes_per_k = qi / vdr;
    constexpr int blocks_per_iter = vdr * 16 / qi;
    constexpr int q8_per_k = qk / QK8_1;

    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row_lane = threadIdx.x >> 4u;
    const uint32_t row0 = blockIdx.x * 64u + row_lane;
    const uint32_t blocks_per_row = K / qk;
    const uint32_t kbx0 = lane / lanes_per_k;
    const int kqs = vdr * (lane % lanes_per_k);

    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = kbx0; b < blocks_per_row; b += blocks_per_iter) {
        const block_q8_1 *xb = X_q8 + (uint64_t)b * q8_per_k;
#pragma unroll
        for (uint32_t rr = 0; rr < 4u; rr++) {
            const uint32_t row = row0 + rr * 16u;
            if (row < M) {
                const int kbx = (int)(row * blocks_per_row + b);
                acc0[rr] += ds4_mmq_vec_dot_q8_1<type>(W0, xb, kbx, kqs);
                acc1[rr] += ds4_mmq_vec_dot_q8_1<type>(W1, xb, kbx, kqs);
            }
        }
    }
#pragma unroll
    for (uint32_t rr = 0; rr < 4u; rr++) {
        acc0[rr] = ds4_mmq_half_warp_sum_f32(acc0[rr]);
        acc1[rr] = ds4_mmq_half_warp_sum_f32(acc1[rr]);
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t rr = 0; rr < 4u; rr++) {
            const uint32_t row = row0 + rr * 16u;
            if (row < M) {
                const float a = acc0[rr];
                const float b = acc1[rr];
                out0[row] = isfinite(a) ? a : 0.0f;
                out1[row] = isfinite(b) ? b : 0.0f;
            }
        }
    }
}

static int ds4_mmq_q4_K_dense_pair_vec_impl(
        const void *W0, const void *W1, const float *X,
        float *out0, float *out1, int M, int K, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q4_K_dense_pair_vec";
    if (!W0 || !W1 || !X || !out0 || !out1 || M <= 0 || K <= 0 ||
        K % QK_K != 0) {
        return -1;
    }
    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context *ctx = get_ctx_for_device(dev);
    if (!ctx) return -1;
    ds4_pool_set_stream(stream);

    const int64_t padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t qbytes =
        (size_t)padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> q8_pool;
    char *x8 = ds4_mmq_folded_q81(X, K, 1, padded, stream);
    if (!x8) {
        if (void *scratch = ds4_mmq_aligned_q81_scratch(dev, qbytes)) {
            x8 = (char *)scratch;
        } else if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
                   g_q81_scratch_bytes >= qbytes) {
            x8 = (char *)g_q81_scratch_ptr;
        } else {
            q8_pool.alloc(ctx->pool(), qbytes);
            x8 = q8_pool.get();
        }
        quantize_row_q8_1_cuda(
            X, nullptr, x8, GGML_TYPE_Q4_K,
            K, K, K, K, padded, 1, 1, 1, stream);
        const cudaError_t quant_err = cudaGetLastError();
        if (quant_err != cudaSuccess) {
            fprintf(stderr, "%s: activation quantize failed: %s\n",
                    tag, cudaGetErrorString(quant_err));
            return -2;
        }
    }

    ds4_mmq_q4_K_dense_pair_vec_kernel
        <<<((unsigned)M + 63u) / 64u, 256, 0, stream>>>(
            W0, W1, (const block_q8_1 *)x8,
            out0, out1, (uint32_t)M, (uint32_t)K);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

} // anonymous namespace

extern "C" void ds4_mmq_q4_K_k1024_persistent_counters(
        uint64_t *candidates,
        uint64_t *uses,
        uint64_t *fallbacks,
        uint64_t *require_failures,
        uint64_t *oracle_calls,
        uint64_t *oracle_mismatches,
        uint64_t *oracle_skips) {
    if (candidates) *candidates = g_q4_k1024_persistent_candidates;
    if (uses) *uses = g_q4_k1024_persistent_uses;
    if (fallbacks) *fallbacks = g_q4_k1024_persistent_fallbacks;
    if (require_failures) {
        *require_failures = g_q4_k1024_persistent_require_failures;
    }
    if (oracle_calls) *oracle_calls = g_q4_k1024_persistent_oracle_calls;
    if (oracle_mismatches) {
        *oracle_mismatches = g_q4_k1024_persistent_oracle_mismatches;
    }
    if (oracle_skips) *oracle_skips = g_q4_k1024_persistent_oracle_skips;
}

extern "C" int ds4_mmq_q8_0_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q8_0>(
        "ds4_mmq_q8_0_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q2_K_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q2_K>(
        "ds4_mmq_q2_K_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_mxfp4_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_MXFP4>(
        "ds4_mmq_mxfp4_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

// M1-Inc2b: exact inverse of the weight-server repack
// (repack_iq2_xxs_aligned_kernel, tools/ds4_weight_server.cu): aligned-SoA
// artifact -> raw block_iq2_xxs byte stream (66B = [half d][8 x uint2
// codes]).  Device->device fill of a raw-layout scratch so the batched/mmq
// consumers keep their layout while the raw spans stay excluded from the
// upload.  One thread per (block, pair); p==0 additionally writes the
// 2-byte scale.  Destination blocks are 66B so stores are byte-granular.
__global__ void iq2_xxs_aligned_derepack_kernel(
        unsigned char     *raw,        // [nblk * 66]
        const uint2       *qs,         // 64B-aligned code pairs
        const __half      *dq,         // block scales
        uint64_t           nblk)
{
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 8ull) return;
    const uint64_t blk = i >> 3;
    const uint32_t p = (uint32_t)(i & 7u);
    unsigned char *dst = raw + blk * 66ull;
    if (p == 0u) {
        const uint16_t h = __half_as_ushort(dq[blk]);
        memcpy(dst, &h, 2u);
    }
    const uint2 v = qs[blk * 8ull + p];
    memcpy(dst + 2u + (uint64_t)p * 8u, &v, 8u);
}

extern "C" int ds4_mmq_iq2_xxs_aligned_derepack(
        const void * W_aligned, void * raw_out,
        int M, int K, int n_experts, cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_derepack";
    if (!W_aligned || !raw_out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_experts <= 0 || K % 256 != 0) return -1;
    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const uint64_t n_threads = nblk * 8ull;
    iq2_xxs_aligned_derepack_kernel<<<(unsigned)((n_threads + 255ull) / 256ull), 256, 0, stream>>>(
        (unsigned char *)raw_out,
        (const uint2 *)((const char *)W_aligned + dq_bytes),
        (const __half *)W_aligned,
        nblk);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" void ds4_mmq_set_gb10_optimizations(int enabled) {
    {
        // Serialize the transition with any persistent Q8_1 host lease.  The
        // atomic also covers GB10 admission reads outside this arena lock.
        std::lock_guard<std::mutex> lock(g_q81_state_mutex);
        g_gb10_optimizations.store(enabled != 0, std::memory_order_relaxed);
    }
    if (!enabled) {
        // Backend teardown/reinit already funnels through this setter.  Keep
        // MMQ's owned arena lifecycle local to this translation unit.
        (void)ds4_mmq_q81_persistent_cleanup();
    }
}

// ---------------------------------------------------------------------------
// Aligned-SoA Q8_0 dense decode matvec (megakernel program M1-Inc3).
//
// block_q8_0 is 34 bytes ([half d][int8 qs[32]]), so the raw code stream is
// only 2-byte aligned — the same misalignment class proto_iq2_aligned proved
// costly.  Artifact layout (weight server --repack-q8-aligned, derived kind
// DERIVED_Q8_0_ALIGNED_DENSE): [__half dq[nblk]][pad to 64B][int8 qs[nblk*32]]
// with nblk = M * (K/32), block order equal to the raw tensor byte order.
// Unlike the IQ2 expert repack, the raw spans stay SERVED (dense tensors are
// ~6 GiB total, affordable to duplicate), so every other consumer is
// unchanged.  proto_q8_aligned.cu A/B (GB10, L2-defeating rotation, double-ref
// parity): attn_q_b 217->235, mid 2048x4096 172->218, out_a 8192x4096
// 199->230, head 224->243 GB/s; the warp-per-row accumulation is also ~1000x
// closer to the double reference than the mmvq tile order at K>=4096.
template <int WARPS_PER_BLOCK>
__global__ void q8_0_aligned_dense_vec_kernel(
        float             *out,        // [M]
        const int4        *qs,         // aligned codes, 2 int4 per block
        const __half      *dq,         // block scales
        const block_q8_1  *x8,         // [K/32] canonical Q8_1 activation
        int                M,
        int                nb)         // blocks per row = K/32
{
    const int warp = threadIdx.x >> 5;
    const int row  = blockIdx.x * WARPS_PER_BLOCK + warp;
    const int lane = threadIdx.x & 31;
    if (row >= M) return;
    const long long rbase = (long long)row * nb;

    float acc = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        const int b = b0 + lane;
        const int4 w0 = qs[(rbase + b) * 2 + 0];   // aligned 16B loads
        const int4 w1 = qs[(rbase + b) * 2 + 1];
        const int *u = (const int *)x8[b].qs;
        int sumi = 0;
        sumi = ggml_cuda_dp4a(w0.x, u[0], sumi);
        sumi = ggml_cuda_dp4a(w0.y, u[1], sumi);
        sumi = ggml_cuda_dp4a(w0.z, u[2], sumi);
        sumi = ggml_cuda_dp4a(w0.w, u[3], sumi);
        sumi = ggml_cuda_dp4a(w1.x, u[4], sumi);
        sumi = ggml_cuda_dp4a(w1.y, u[5], sumi);
        sumi = ggml_cuda_dp4a(w1.z, u[6], sumi);
        sumi = ggml_cuda_dp4a(w1.w, u[7], sumi);
        acc += __half2float(dq[rbase + b]) * __low2float(x8[b].ds) * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

/* K=1024 decode specialization. Eight persistent row warps per CTA hoist the
 * 32 Q8_1 activation blocks into registers and walk output rows at a grid
 * stride. Each lane still owns the same single block term and the warp tree is
 * unchanged, so output bits match q8_0_aligned_dense_vec_kernel. */
__global__ __launch_bounds__(256, 6) void q8_0_aligned_dense_vec_k1024_persistent_kernel(
        float             *out,
        const int4        *qs,
        const __half      *dq,
        const block_q8_1  *x8,
        int                M)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int *u = (const int *)x8[lane].qs;
    const int u0 = u[0];
    const int u1 = u[1];
    const int u2 = u[2];
    const int u3 = u[3];
    const int u4 = u[4];
    const int u5 = u[5];
    const int u6 = u[6];
    const int u7 = u[7];
    const float dx = __low2float(x8[lane].ds);
    const int64_t row0 = (int64_t)blockIdx.x * 8 + warp;
    const int64_t row_stride = (int64_t)gridDim.x * 8;

    for (int64_t row = row0; row < (int64_t)M; row += row_stride) {
        const long long block = (long long)row * 32 + lane;
        const int4 w0 = qs[block * 2 + 0];
        const int4 w1 = qs[block * 2 + 1];
        int s0 = ggml_cuda_dp4a(w0.x, u0, 0);
        s0 = ggml_cuda_dp4a(w0.y, u1, s0);
        int s1 = ggml_cuda_dp4a(w0.z, u2, 0);
        s1 = ggml_cuda_dp4a(w0.w, u3, s1);
        int s2 = ggml_cuda_dp4a(w1.x, u4, 0);
        s2 = ggml_cuda_dp4a(w1.y, u5, s2);
        int s3 = ggml_cuda_dp4a(w1.z, u6, 0);
        s3 = ggml_cuda_dp4a(w1.w, u7, s3);
        const int sumi = (s0 + s1) + (s2 + s3);
        float acc = 0.0f;
        acc += __half2float(dq[block]) * dx * (float)sumi;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[row] = acc;
    }
}

/* Persistent-CTA form for the K=4096 vocabulary projection. It preserves
 * the original lane/block assignment, per-lane term order, and warp tree;
 * grouping eight row warps removes the one-warp CTA occupancy ceiling. */
__global__ __launch_bounds__(256, 6) void q8_0_aligned_dense_vec_persistent_kernel(
        float             *out,
        const int4        *qs,
        const __half      *dq,
        const block_q8_1  *x8,
        int                M,
        int                nb)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t row0 = (int64_t)blockIdx.x * 8 + warp;
    const int64_t row_stride = (int64_t)gridDim.x * 8;
    for (int64_t row = row0; row < (int64_t)M; row += row_stride) {
        const long long rbase = (long long)row * nb;
        float acc = 0.0f;
        for (int b0 = 0; b0 < nb; b0 += 32) {
            const int b = b0 + lane;
            const int4 w0 = qs[(rbase + b) * 2 + 0];
            const int4 w1 = qs[(rbase + b) * 2 + 1];
            const int *u = (const int *)x8[b].qs;
            int sumi = 0;
            sumi = ggml_cuda_dp4a(w0.x, u[0], sumi);
            sumi = ggml_cuda_dp4a(w0.y, u[1], sumi);
            sumi = ggml_cuda_dp4a(w0.z, u[2], sumi);
            sumi = ggml_cuda_dp4a(w0.w, u[3], sumi);
            sumi = ggml_cuda_dp4a(w1.x, u[4], sumi);
            sumi = ggml_cuda_dp4a(w1.y, u[5], sumi);
            sumi = ggml_cuda_dp4a(w1.z, u[6], sumi);
            sumi = ggml_cuda_dp4a(w1.w, u[7], sumi);
            acc += __half2float(dq[rbase + b]) *
                   __low2float(x8[b].ds) * (float)sumi;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[row] = acc;
    }
}

// Verify-width variant (v0.4 dense chase, proto_q8_aligned_nc): same aligned
// weight stream read ONCE per row, NC output columns accumulated per lane
// against col-strided q8_1 activations (which L1/L2-broadcast across rows).
// Bytes identical to the N=1 kernel, so it holds the aligned tier's rate at
// the spec-verify widths where the raw-block mmvq fallback ran 90-200 GB/s
// (proto: +17..+87% per shape, family within 4% of the weight-bytes floor).
// out is column-major [NC][M], the engine's [n_tok, out_dim] flattening.
template <int NC>
__global__ void q8_0_aligned_dense_vec_nc_kernel(
        float             *out,        // [NC * M]
        const int4        *qs,         // aligned codes, 2 int4 per block
        const __half      *dq,         // block scales
        const block_q8_1  *x8,         // [NC * nb], col stride nb
        int                M,
        int                nb)         // blocks per row = K/32
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const long long rbase = (long long)row * nb;

    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) acc[c] = 0.0f;

    for (int b0 = 0; b0 < nb; b0 += 32) {
        const int b = b0 + lane;
        const int4 w0 = qs[(rbase + b) * 2 + 0];   // aligned 16B, read once
        const int4 w1 = qs[(rbase + b) * 2 + 1];
        const float dw = __half2float(dq[rbase + b]);
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const block_q8_1 *xb = &x8[(size_t)c * nb + b];
            const int *u = (const int *)xb->qs;
            int sumi = 0;
            sumi = ggml_cuda_dp4a(w0.x, u[0], sumi);
            sumi = ggml_cuda_dp4a(w0.y, u[1], sumi);
            sumi = ggml_cuda_dp4a(w0.z, u[2], sumi);
            sumi = ggml_cuda_dp4a(w0.w, u[3], sumi);
            sumi = ggml_cuda_dp4a(w1.x, u[4], sumi);
            sumi = ggml_cuda_dp4a(w1.y, u[5], sumi);
            sumi = ggml_cuda_dp4a(w1.z, u[6], sumi);
            sumi = ggml_cuda_dp4a(w1.w, u[7], sumi);
            acc[c] += dw * __low2float(xb->ds) * (float)sumi;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; c++) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[c] += __shfl_down_sync(0xffffffffu, acc[c], off);
        if (lane == 0) out[(size_t)c * M + row] = acc[c];
    }
}

static int ds4_q8_aligned_warps_per_block(int cc);

static cudaError_t q8_0_aligned_dense_vec_launch(
        float *out, const int4 *qs, const __half *dq,
        const block_q8_1 *x8, int M, int N, int K,
        cudaStream_t stream) {
    switch (N) {
    case 1:
        if (gb10_optimizations_enabled() &&
            getenv("DS4_CUDA_NO_Q8_ALIGNED_PERSISTENT") == NULL &&
            (K == 1024 || K == 4096) && M >= 32768) {
            const uint64_t row_blocks = ((uint64_t)(unsigned)M + 7u) / 8u;
            const unsigned persistent_blocks =
                row_blocks < 288u ? (unsigned)row_blocks : 288u;
            if (K == 1024) {
                q8_0_aligned_dense_vec_k1024_persistent_kernel<<<
                    persistent_blocks, 256, 0, stream>>>(
                        out, qs, dq, x8, M);
            } else {
                q8_0_aligned_dense_vec_persistent_kernel<<<
                    persistent_blocks, 256, 0, stream>>>(
                        out, qs, dq, x8, M, K / 32);
            }
        } else {
            switch (ds4_q8_aligned_warps_per_block(
                        ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
            case 16:
                q8_0_aligned_dense_vec_kernel<16>
                    <<<((unsigned)M + 15u) / 16u, 512, 0, stream>>>(
                        out, qs, dq, x8, M, K / 32);
                break;
            case 8:
                q8_0_aligned_dense_vec_kernel<8>
                    <<<((unsigned)M + 7u) / 8u, 256, 0, stream>>>(
                        out, qs, dq, x8, M, K / 32);
                break;
            case 4:
                q8_0_aligned_dense_vec_kernel<4>
                    <<<((unsigned)M + 3u) / 4u, 128, 0, stream>>>(
                        out, qs, dq, x8, M, K / 32);
                break;
            case 2:
                q8_0_aligned_dense_vec_kernel<2>
                    <<<((unsigned)M + 1u) / 2u, 64, 0, stream>>>(
                        out, qs, dq, x8, M, K / 32);
                break;
            default:
                q8_0_aligned_dense_vec_kernel<1>
                    <<<(unsigned)M, 32, 0, stream>>>(
                        out, qs, dq, x8, M, K / 32);
                break;
            }
        }
        break;
    case 2: q8_0_aligned_dense_vec_nc_kernel<2><<<(unsigned)M, 32, 0, stream>>>(out, qs, dq, x8, M, K / 32); break;
    case 3: q8_0_aligned_dense_vec_nc_kernel<3><<<(unsigned)M, 32, 0, stream>>>(out, qs, dq, x8, M, K / 32); break;
    case 4: q8_0_aligned_dense_vec_nc_kernel<4><<<(unsigned)M, 32, 0, stream>>>(out, qs, dq, x8, M, K / 32); break;
    case 5: q8_0_aligned_dense_vec_nc_kernel<5><<<(unsigned)M, 32, 0, stream>>>(out, qs, dq, x8, M, K / 32); break;
    case 6: q8_0_aligned_dense_vec_nc_kernel<6><<<(unsigned)M, 32, 0, stream>>>(out, qs, dq, x8, M, K / 32); break;
    case 7: q8_0_aligned_dense_vec_nc_kernel<7><<<(unsigned)M, 32, 0, stream>>>(out, qs, dq, x8, M, K / 32); break;
    case 8: q8_0_aligned_dense_vec_nc_kernel<8><<<(unsigned)M, 32, 0, stream>>>(out, qs, dq, x8, M, K / 32); break;
    default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

/* Full consumer oracle for the folded single-column Q8_0 aligned entry.
 * It regenerates canonical Q8_1, runs the exact same consumer twice, compares
 * output bits, and always leaves the freshly quantized reference output in
 * the caller buffer.  Return 1 when handled, 0 when diagnostics could not be
 * set up before enqueue, and -1 after a CUDA failure. */
static int q8_fold_q8_aligned_output_oracle(
        const float *X_f32, const block_q8_1 *folded,
        float *out, const int4 *qs, const __half *dq,
        int M, int K, cudaStream_t stream) {
    if (!ds4_mmq_q8_fold_oracle_enabled() || !folded || M <= 0 || K <= 0) {
        return 0;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        (void)cudaGetLastError();
        g_q8_fold_oracle_skips++;
        return 0;
    }
    const size_t q8_bytes = (size_t)K * sizeof(block_q8_1) / QK8_1;
    const size_t out_bytes = (size_t)M * sizeof(float);
    block_q8_1 *fresh = nullptr;
    float *reference = nullptr;
    uint32_t *mismatch_device = nullptr;
    if (cudaMalloc((void **)&fresh, q8_bytes) != cudaSuccess ||
        cudaMalloc((void **)&reference, out_bytes) != cudaSuccess ||
        cudaMalloc((void **)&mismatch_device, sizeof(uint32_t)) != cudaSuccess ||
        !fresh || !reference || !mismatch_device) {
        (void)cudaGetLastError();
        cudaError_t cleanup_err = cudaSuccess;
        cleanup_err = ds4_mmq_q8_fold_oracle_free(
            fresh, "aligned-q8-fresh", cleanup_err);
        cleanup_err = ds4_mmq_q8_fold_oracle_free(
            reference, "aligned-q8-reference", cleanup_err);
        cleanup_err = ds4_mmq_q8_fold_oracle_free(
            mismatch_device, "aligned-q8-mismatch", cleanup_err);
        if (cleanup_err != cudaSuccess) (void)cudaGetLastError();
        g_q8_fold_oracle_skips++;
        return 0;
    }

    cudaError_t err = cudaMemsetAsync(
        mismatch_device, 0, sizeof(uint32_t), stream);
    if (err == cudaSuccess) {
        quantize_row_q8_1_cuda(
            X_f32, /*ids=*/nullptr, fresh, GGML_TYPE_Q8_0,
            /*ne00=*/K, /*s11=*/K, /*s12=*/K, /*s13=*/K,
            /*ne0=*/K, /*ne1=*/1, /*ne2=*/1, /*ne3=*/1, stream);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = q8_0_aligned_dense_vec_launch(
            out, qs, dq, folded, M, 1, K, stream);
    }
    if (err == cudaSuccess) {
        err = q8_0_aligned_dense_vec_launch(
            reference, qs, dq, fresh, M, 1, K, stream);
    }
    if (err == cudaSuccess) {
        q8_fold_output_compare_kernel<<<
            (unsigned)(((uint64_t)M + 255u) / 256u), 256, 0, stream>>>(
                mismatch_device, out, reference, (uint64_t)M);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(out, reference, out_bytes,
                              cudaMemcpyDeviceToDevice, stream);
    }
    uint32_t mismatch_host = 0u;
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(&mismatch_host, mismatch_device,
                              sizeof(mismatch_host),
                              cudaMemcpyDeviceToHost, stream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);

    err = ds4_mmq_q8_fold_oracle_free(
        fresh, "aligned-q8-fresh", err);
    err = ds4_mmq_q8_fold_oracle_free(
        reference, "aligned-q8-reference", err);
    err = ds4_mmq_q8_fold_oracle_free(
        mismatch_device, "aligned-q8-mismatch", err);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        g_q8_fold_oracle_skips++;
        return -1;
    }
    g_q8_fold_oracle_output_calls++;
    g_q8_fold_oracle_aligned_q8_calls++;
    if (mismatch_host != 0u) {
        g_q8_fold_oracle_output_mismatches++;
        fprintf(stderr,
                "ds4: CUDA Q8_1 fold oracle found a Q8 aligned consumer "
                "output mismatch; retained canonical output\n");
    }
    return 1;
}

extern "C" uint64_t ds4_mmq_q8_0_aligned_bytes(int M, int K) {
    if (M <= 0 || K <= 0 || K % 1024 != 0) return 0;
    const uint64_t nblk = (uint64_t)M * (uint64_t)(K / 32);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    return dq_bytes + nblk * 32u;
}

static int ds4_q8_aligned_warps_per_block(int cc) {
    const char *value = getenv("DS4_CUDA_Q8_ALIGNED_WARPS");
    if (value) {
        const int requested = atoi(value);
        if (requested == 1 || requested == 2 || requested == 4 ||
            requested == 8 || requested == 16) {
            return requested;
        }
    }
    return cc == 1210 ? 16 : 1;
}

template <int WARPS_PER_BLOCK>
__global__ void q8_0_aligned_dense_vec_pair_kernel(
        float             *out0,
        float             *out1,
        const int4        *qs0,
        const int4        *qs1,
        const __half      *dq0,
        const __half      *dq1,
        const block_q8_1  *x8,
        int                M0,
        int                M1,
        int                nb) {
    const int warp = threadIdx.x >> 5;
    const int global_row = blockIdx.x * WARPS_PER_BLOCK + warp;
    const int lane = threadIdx.x & 31;
    if (global_row >= M0 + M1) return;
    const bool second = global_row >= M0;
    const int row = second ? global_row - M0 : global_row;
    const int4 *qs = second ? qs1 : qs0;
    const __half *dq = second ? dq1 : dq0;
    const long long rbase = (long long)row * nb;

    float acc = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        const int b = b0 + lane;
        const int4 w0 = qs[(rbase + b) * 2 + 0];
        const int4 w1 = qs[(rbase + b) * 2 + 1];
        const int *u = (const int *)x8[b].qs;
        int sumi = 0;
        sumi = ggml_cuda_dp4a(w0.x, u[0], sumi);
        sumi = ggml_cuda_dp4a(w0.y, u[1], sumi);
        sumi = ggml_cuda_dp4a(w0.z, u[2], sumi);
        sumi = ggml_cuda_dp4a(w0.w, u[3], sumi);
        sumi = ggml_cuda_dp4a(w1.x, u[4], sumi);
        sumi = ggml_cuda_dp4a(w1.y, u[5], sumi);
        sumi = ggml_cuda_dp4a(w1.z, u[6], sumi);
        sumi = ggml_cuda_dp4a(w1.w, u[7], sumi);
        acc += __half2float(dq[rbase + b]) *
               __low2float(x8[b].ds) * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) (second ? out1 : out0)[row] = acc;
}

extern "C" int ds4_mmq_q8_0_aligned_dense_vec_pair(
        const void *W0_aligned, const void *W1_aligned,
        const float *X_f32, float *out0_f32, float *out1_f32,
        int M0, int M1, int K, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q8_0_aligned_dense_vec_pair";
    if (!W0_aligned || !W1_aligned || !X_f32 || !out0_f32 || !out1_f32 ||
        M0 <= 0 || M1 <= 0 || K <= 0 || K % 1024 != 0) {
        return -1;
    }
    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context *ctx = get_ctx_for_device(dev);
    if (!ctx) return -1;
    ds4_pool_set_stream(stream);

    const int64_t padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t qbytes =
        (size_t)padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> q8_pool;
    char *x8 = ds4_mmq_folded_q81(X_f32, K, 1, padded, stream);
    if (!x8) {
        if (void *scratch = ds4_mmq_aligned_q81_scratch(dev, qbytes)) {
            x8 = (char *)scratch;
        } else if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
            g_q81_scratch_bytes >= qbytes) {
            x8 = (char *)g_q81_scratch_ptr;
        } else {
            q8_pool.alloc(ctx->pool(), qbytes);
            x8 = q8_pool.get();
        }
        quantize_row_q8_1_cuda(
            X_f32, nullptr, (void *)x8, GGML_TYPE_Q8_0,
            K, K, K, K, padded, 1, 1, 1, stream);
        const cudaError_t quant_err = cudaGetLastError();
        if (quant_err != cudaSuccess) {
            fprintf(stderr, "%s: activation quantize failed: %s\n",
                    tag, cudaGetErrorString(quant_err));
            return -2;
        }
    }

    const uint64_t nblk0 = (uint64_t)M0 * (uint64_t)(K / 32);
    const uint64_t nblk1 = (uint64_t)M1 * (uint64_t)(K / 32);
    const uint64_t dq0_bytes = (nblk0 * 2u + 63u) & ~63ull;
    const uint64_t dq1_bytes = (nblk1 * 2u + 63u) & ~63ull;
    const int4 *qs0 =
        (const int4 *)((const char *)W0_aligned + dq0_bytes);
    const int4 *qs1 =
        (const int4 *)((const char *)W1_aligned + dq1_bytes);
    const int rows = M0 + M1;
    const int warps = ds4_q8_aligned_warps_per_block(
        ggml_cuda_info().devices[dev].cc);
#define DS4_LAUNCH_Q8_PAIR(W) \
    q8_0_aligned_dense_vec_pair_kernel<W> \
        <<<(unsigned)(rows + W - 1) / W, W * 32, 0, stream>>>( \
            out0_f32, out1_f32, qs0, qs1, \
            (const __half *)W0_aligned, (const __half *)W1_aligned, \
            (const block_q8_1 *)x8, M0, M1, K / 32)
    switch (warps) {
    case 16: DS4_LAUNCH_Q8_PAIR(16); break;
    case 8: DS4_LAUNCH_Q8_PAIR(8); break;
    case 4: DS4_LAUNCH_Q8_PAIR(4); break;
    case 2: DS4_LAUNCH_Q8_PAIR(2); break;
    default: DS4_LAUNCH_Q8_PAIR(1); break;
    }
#undef DS4_LAUNCH_Q8_PAIR
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" int ds4_mmq_q8_0_aligned_dense_vec(
        const void * W_aligned, const float * X_f32, float * out_f32,
        int M, int N, int K, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q8_0_aligned_dense_vec";
    if (!W_aligned || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    // K % 1024: the kernel's 32-blocks-per-pass loop needs nb % 32 == 0.
    // N covers the decode/verify-width envelope (mmvq batch bound); K % 1024
    // also guarantees ne10_padded == K, so the q8_1 col stride is exactly nb.
    if (N < 1 || N > 8 || M <= 0 || K <= 0 || K % 1024 != 0) return -1;

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)N * ne10_padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> q8_pool;
    // M2-Inc2a: producer-emitted q8_1 codes (qr_norm from the qkv-rms
    // kernel) -- take them and skip the quantize prelude.  Single-column
    // producers only; verify widths always quantize.
    char *x8 = N == 1
        ? ds4_mmq_folded_q81(X_f32, K, 1, ne10_padded, stream)
        : NULL;
    const bool folded_hit = x8 != NULL;
    cudaError_t err;
    if (!x8) {
        if (getenv("DS4_CUDA_NO_Q8_ALIGNED_DENSE_SCRATCH") == NULL) {
            x8 = (char *)ds4_mmq_aligned_q81_scratch(dev, nbytes_q8_1);
        }
        if (!x8 && g_q81_scratch_enabled && g_q81_scratch_ptr &&
            g_q81_scratch_bytes >= nbytes_q8_1) {
            x8 = (char *)g_q81_scratch_ptr;
        }
        if (!x8) {
            q8_pool.alloc(ctx->pool(), nbytes_q8_1);
            x8 = q8_pool.get();
        }
        quantize_row_q8_1_cuda(
            X_f32, /*ids=*/nullptr, (void *)x8,
            GGML_TYPE_Q8_0, /*ne00=*/K,
            /*s11=*/(int64_t)K, /*s12=*/(int64_t)K * N,
            /*s13=*/(int64_t)K * N,
            /*ne0=*/ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1,
            stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "%s: quantize_row_q8_1_cuda failed: %s\n",
                    tag, cudaGetErrorString(err));
            return -2;
        }
    }

    const uint64_t nblk = (uint64_t)M * (uint64_t)(K / 32);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const int4   *qsp = (const int4 *)((const char *)W_aligned + dq_bytes);
    const __half *dqp = (const __half *)W_aligned;
    const block_q8_1 *x8p = (const block_q8_1 *)x8;
    if (folded_hit && N == 1 && ds4_mmq_q8_fold_oracle_enabled()) {
        const int oracle_rc = q8_fold_q8_aligned_output_oracle(
            X_f32, x8p, out_f32, qsp, dqp, M, K, stream);
        if (oracle_rc > 0) return 0;
        if (oracle_rc < 0) {
            fprintf(stderr, "%s: fold consumer oracle failed\n", tag);
            return -3;
        }
    }
    err = q8_0_aligned_dense_vec_launch(
        out_f32, qsp, dqp, x8p, M, N, K, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Aligned row-pair-SoA Q2_K routed-expert decode matvec (megakernel program
// M2, moe-down increment).  The production down leg runs
// mul_mat_vec_q_moe<GGML_TYPE_Q2_K, 2> over raw 84-byte block_q2_K stacks at
// ~190 GB/s (per-lane loads: one 4B qs int, four scale BYTES, one 4B half2 --
// 12 load instructions per lane-iteration).  W_aligned is a repacked copy of
// the SAME bytes keyed to that kernel's rows_per_block == 2: for the row pair
// (2p, 2p+1) of an expert, each lane-iteration needs exactly one 8B qs load
// (both rows' int), one 16B scales-window load (both rows' 8B half), one 8B
// dm load.  Layout contract (shared with the weight server
// --repack-q2k-aligned, DERIVED_Q2_K_ALIGNED_MOE, and ds4_mmq.h):
//
//   npair = n_experts * (M/2) * (K/256)      pair-blocks, expert-major then
//                                            row-pair then block (raw order)
//   [ uint2 dm2[npair] ]        {row0 half2(d,dmin), row1 half2}
//   [ pad to 64B ]
//   [ int4  sc4[npair*2] ]      half h: {row0 scales[8h..8h+3], row0 [8h+4..
//                               8h+7], row1 [8h..8h+3], row1 [8h+4..8h+7]}
//   [ pad to 64B ]
//   [ uint2 qs2[npair*16] ]     iqs: {row0 qs int[iqs], row1 qs int[iqs]}
//
// Lane mapping, scale-byte values, q8 side and the float accumulation order
// are copied verbatim from mul_mat_vec_q_moe/vec_dot_q2_K_q8_1 -> outputs are
// bit-identical to the raw path (proto_m2_q2k.cu: 240/240 parity + graph
// capture/replay, and 214 GB/s vs 154 raw on the same rotating rig).
// ---------------------------------------------------------------------------

// Same float chain as vec_dot_q2_K_q8_1_impl_mmvq; the four scale bytes come
// from the two pre-loaded 32-bit window words (byte lo+2i of the 8B window ==
// scales[scale_offset + 2i] of the raw block).
static __device__ __forceinline__ float q2_k_vec_dot_windowed(
        const int v, const int * __restrict__ u, const uint32_t w0, const uint32_t w1,
        const int lo, const half2 dm2, const float * __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int bidx = lo + 2*i;
        const uint32_t w = (bidx < 4) ? w0 : w1;
        const int sc = (int)((w >> ((bidx & 3) * 8)) & 0xFFu);

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));

        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);
    }
    const float2 dm2f = __half22float2(dm2);
    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// Twin of mul_mat_vec_q_moe<GGML_TYPE_Q2_K, 2> at the down-leg call shape
// (nchannels_dst == 1, ids_stride == 1): grid (M/2, 1), block (32, ncols_dst),
// warp per assignment column.  Keeps the -1 router-id guard (task #23).
__launch_bounds__(8*32, 1)   /* MMVQ_MAX_BATCH_SIZE (mmvq.cuh) * warp; not included here */
__global__ static void q2_k_aligned_moe_vec_kernel(
        const uint2 * __restrict__ dm2_soa,
        const int4  * __restrict__ sc4_soa,
        const uint2 * __restrict__ qs2_soa,
        const block_q8_1 * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t ncols_dst) {
    constexpr int qi  = 16;   // QI2_K
    constexpr int vdr = 1;    // VDR_Q2_K_Q8_1_MMVQ
    constexpr int warp_size = 32;

    const uint32_t token_idx = threadIdx.y;
    const int      row0      = 2*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / QK_K;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;   // 2

    if (token_idx >= ncols_dst) {
        return;
    }

    const int32_t  id_raw     = ids[token_idx];
    const bool     invalid_id = id_raw < 0;
    const uint32_t channel_x  = invalid_id ? 0u : (uint32_t)id_raw;

    const block_q8_1 * y = vy + token_idx*stride_col_y;
    const size_t pair_base = ((size_t)channel_x * (nrows_x/2u) + (size_t)blockIdx.x)
                           * (size_t)blocks_per_row_x;

    float tmp[2] = {0.0f, 0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K/QK8_1);
        const int iqs = vdr * (threadIdx.x % (qi/vdr));

        const int bq8_offset = QR2_K * (iqs / QI8_1);
        const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
        const int whalf = iqs / QI8_1;
        const int lo    = scale_offset - 8*whalf;
        const block_q8_1 * bq8_1 = &y[kby];

        int    u[QR2_K];
        float d8[QR2_K];
#pragma unroll
        for (int i = 0; i < QR2_K; ++i) {
            u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
            d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
        }

        const size_t pblk = pair_base + (size_t)kbx;
        const uint2 v2  = qs2_soa[pblk*16u + (unsigned)iqs];
        const uint2 dmw = dm2_soa[pblk];
        const int4  scw = sc4_soa[pblk*2u + (unsigned)whalf];
        const half2 dm0 = *(const half2 *)&dmw.x;
        const half2 dm1 = *(const half2 *)&dmw.y;

        tmp[0] += q2_k_vec_dot_windowed((int)v2.x, u, (uint32_t)scw.x, (uint32_t)scw.y, lo, dm0, d8);
        tmp[1] += q2_k_vec_dot_windowed((int)v2.y, u, (uint32_t)scw.z, (uint32_t)scw.w, lo, dm1, d8);
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (threadIdx.x < 2 && uint32_t(row0 + threadIdx.x) < nrows_x) {
        dst[token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

// Exact inverse of the weight-server repack (repack_q2_k_aligned_kernel,
// tools/ds4_weight_server.cu): pair-SoA -> raw block_q2_K byte stream.  One
// thread per (raw block, qs int); p < 4 additionally restores a scales word,
// p == 0 the dm word.
__global__ static void q2_k_aligned_derepack_kernel(
        unsigned char *raw_out,
        const uint2   * __restrict__ dm2_soa,
        const int4    * __restrict__ sc4_soa,
        const uint2   * __restrict__ qs2_soa,
        uint64_t nblk,       // raw blocks total
        uint32_t nb_row,     // blocks per row = K/256
        uint32_t nrows) {    // rows per expert = M
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 16ull) return;
    const uint64_t g = i >> 4;
    const uint32_t p = (uint32_t)(i & 15u);
    const uint32_t b = (uint32_t)(g % nb_row);
    const uint32_t r = (uint32_t)((g / nb_row) % nrows);
    const uint64_t e = g / ((uint64_t)nb_row * nrows);
    const uint64_t pblk = ((uint64_t)e * (nrows/2u) + r/2u) * nb_row + b;
    const uint32_t parity = r & 1u;
    unsigned char *dst = raw_out + g * 84ull;
    const uint2 q = qs2_soa[pblk*16u + p];
    const uint32_t qw = parity ? q.y : q.x;
    memcpy(dst + 16u + (uint64_t)p * 4u, &qw, 4u);
    if (p < 4u) {
        const int4 s = sc4_soa[pblk*2u + (p >> 1)];
        const uint32_t sw = parity ? ((p & 1u) ? (uint32_t)s.w : (uint32_t)s.z)
                                   : ((p & 1u) ? (uint32_t)s.y : (uint32_t)s.x);
        memcpy(dst + ((p >> 1) * 8u + (p & 1u) * 4u), &sw, 4u);
    }
    if (p == 0u) {
        const uint2 d = dm2_soa[pblk];
        const uint32_t dw = parity ? d.y : d.x;
        memcpy(dst + 80u, &dw, 4u);
    }
}

extern "C" uint64_t ds4_mmq_q2_k_aligned_bytes(int M, int K, int n_experts) {
    if (M <= 0 || K <= 0 || n_experts <= 0 || K % 256 != 0 || M % 2 != 0) return 0;
    const uint64_t npair = (uint64_t)n_experts * (uint64_t)(M/2) * (uint64_t)(K / 256);
    const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
    const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
    return dm_bytes + sc_bytes + npair * 128u;
}

extern "C" int ds4_mmq_q2_K_aligned_moe_vec(
        const void * W_aligned, const float * X_f32, const int32_t * ids,
        float * out_f32, int M, int K, int n_tokens, int n_experts,
        int n_expert_used, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q2_K_aligned_moe_vec";
    if (!W_aligned || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    /* Down-leg call shape only: each (token, slot) assignment arrives as its
     * own "token" with one expert (n_expert_used == 1, ids_stride == 1). */
    if (n_expert_used != 1 || n_tokens < 1 || M <= 0 || M % 2 != 0 || K <= 0 ||
        K % 256 != 0 || n_experts <= 0) {
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);

    /* Quantize verbatim from ds4_mmq_moe_vec_impl<GGML_TYPE_Q2_K> so the q8_1
     * codes feeding the twin are bit-identical to the raw path's. */
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (void *scratch = ds4_mmq_aligned_q81_scratch(dev, nbytes_q8_1)) {
        src1_q8_1_ptr = (char *)scratch;
    } else if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
               g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        GGML_TYPE_Q2_K, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t s12_y  = ne10_padded / QK8_1;
    const int64_t s2_dst = (int64_t)M;   /* n_expert_used == 1 */

    (void)cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)n_tokens * sizeof(float), stream);

    const uint64_t npair = (uint64_t)n_experts * (uint64_t)(M/2) * (uint64_t)(K / 256);
    const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
    const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
    const uint2 *dm2 = (const uint2 *)W_aligned;
    const int4  *sc4 = (const int4 *)((const char *)W_aligned + dm_bytes);
    const uint2 *qs2 = (const uint2 *)((const char *)W_aligned + dm_bytes + sc_bytes);

    const int col_cap = 8;   /* MMVQ_MAX_BATCH_SIZE; matches __launch_bounds__ */
    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        dim3 grid((unsigned)(M/2), 1);
        dim3 block(32, (unsigned)ncols);
        q2_k_aligned_moe_vec_kernel<<<grid, block, 0, stream>>>(
            dm2, sc4, qs2,
            (const block_q8_1 *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1)),
            ids + c0,
            out_f32 + (int64_t)c0 * s2_dst,
            (uint32_t)K, (uint32_t)M,
            (uint32_t)s12_y, (uint32_t)s2_dst,
            (uint32_t)ncols);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: kernel launch failed: %s (cols %d..%d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1);
            return -3;
        }
    }

    ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)n_tokens, stream);
    return 0;
}

extern "C" int ds4_mmq_q2_K_aligned_derepack(
        const void * W_aligned, void * raw_out,
        int M, int K, int n_experts, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q2_K_aligned_derepack";
    if (!W_aligned || !raw_out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || M % 2 != 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) return -1;
    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t npair = nblk / 2u;
    const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
    const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
    const uint64_t n_threads = nblk * 16ull;
    q2_k_aligned_derepack_kernel<<<(unsigned)((n_threads + 255ull) / 256ull), 256, 0, stream>>>(
        (unsigned char *)raw_out,
        (const uint2 *)W_aligned,
        (const int4 *)((const char *)W_aligned + dm_bytes),
        (const uint2 *)((const char *)W_aligned + dm_bytes + sc_bytes),
        nblk, (uint32_t)(K / 256), (uint32_t)M);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" uint64_t ds4_mmq_iq2_xxs_aligned_bytes(int M, int K, int n_experts) {
    if (M <= 0 || K <= 0 || n_experts <= 0 || K % 256 != 0) return 0;
    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    return dq_bytes + nblk * 64u;
}

// Shared single-token canonical-Q8_1 quantize for the aligned IQ2_XXS
// entries.  Returns the device pointer (persistent scratch when enabled,
// pool otherwise) or nullptr on failure; *pool must outlive the launches.
static char *iq2_aligned_quantize_xn(
        const char *tag, const float *X_f32, int K, int n_tokens,
        ggml_cuda_pool_alloc<char> *pool, cudaStream_t stream,
        bool *was_folded) {
    if (was_folded) *was_folded = false;
    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return nullptr;
    }
    ds4_pool_set_stream(stream);
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    // M2-Inc2a: producer-emitted q8_1 codes (ffn_norm from the fused HC
    // stage) -- take them and skip the quantize prelude.
    char *folded = ds4_mmq_folded_q81(
        X_f32, K, n_tokens, ne10_padded, stream);
    if (folded) {
        if (was_folded) *was_folded = true;
        // C3-Inc4 fold twin selftest (DS4_Q8_FOLD_SELFTEST=<call budget>,
        // eager legs only -- syncs the stream): the taken sidecar must be
        // byte-identical to the fresh quantize this prelude would have run.
        // Do NOT combine with DS4_HC_STAGE_BATCH_PARITY (the probe rewrites
        // norm_out after the sidecar was emitted).
        static int fold_st = -1;
        if (fold_st < 0) {
            const char *st = getenv("DS4_Q8_FOLD_SELFTEST");
            fold_st = st && *st ? atoi(st) : 0;
            if (st && *st && fold_st <= 1) fold_st = 512;
        }
        cudaStreamCaptureStatus fold_cs = cudaStreamCaptureStatusNone;
        if (fold_st > 0) (void)cudaStreamIsCapturing(stream, &fold_cs);
        if (fold_st > 0 && fold_cs == cudaStreamCaptureStatusNone &&
            nbytes_q8_1 <= 16384u) {
            fold_st--;
            static char h[2][16384];
            pool->alloc(ctx->pool(), nbytes_q8_1);
            char *fresh = pool->get();
            quantize_row_q8_1_cuda(
                X_f32, /*ids=*/nullptr, (void *)fresh,
                GGML_TYPE_IQ2_XXS, /*ne00=*/K,
                /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
                /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
                stream);
            if (cudaGetLastError() == cudaSuccess &&
                cudaStreamSynchronize(stream) == cudaSuccess &&
                cudaMemcpy(h[0], folded, nbytes_q8_1, cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(h[1], fresh, nbytes_q8_1, cudaMemcpyDeviceToHost) == cudaSuccess) {
                fprintf(stderr, "ds4: Q8F-SELFTEST(q81 moe) K=%d %s\n", K,
                        memcmp(h[0], h[1], nbytes_q8_1) == 0 ? "PASS" : "FAIL");
            } else {
                fprintf(stderr, "ds4: Q8F-SELFTEST(q81 moe) SKIP (setup failed)\n");
            }
        }
        return folded;
    }
    char *ptr = nullptr;
    if (void *scratch = ds4_mmq_aligned_q81_scratch(dev, nbytes_q8_1)) {
        ptr = (char *)scratch;
    } else if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
               g_q81_scratch_bytes >= nbytes_q8_1) {
        ptr = (char *)g_q81_scratch_ptr;
    } else {
        pool->alloc(ctx->pool(), nbytes_q8_1);
        ptr = pool->get();
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)ptr,
        GGML_TYPE_IQ2_XXS, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

extern "C" int ds4_mmq_iq2_xxs_aligned_moe_pair_vec(
        const void * W_gate_aligned, const void * W_up_aligned,
        const float * X_f32, const int32_t * ids,
        float * gate_out, float * up_out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_moe_pair_vec";
    if (!W_gate_aligned || !W_up_aligned || !X_f32 || !ids || !gate_out || !up_out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (n_tokens < 1 || n_tokens > 16 || M <= 0 || K <= 0 || n_experts <= 0 ||
        n_expert_used <= 0 || n_expert_used > n_experts || K % 1024 != 0) {
        return -1;
    }
    ggml_cuda_pool_alloc<char> q8_pool;
    char *x8 = iq2_aligned_quantize_xn(
        tag, X_f32, K, n_tokens, &q8_pool, stream, nullptr);
    if (!x8) return -2;

    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    dim3 grid((unsigned)M, (unsigned)(n_tokens * n_expert_used), 2);
    iq2_xxs_aligned_moe_pair_vec_kernel<<<grid, 32, 0, stream>>>(
        gate_out, up_out,
        (const uint2 *)((const char *)W_gate_aligned + dq_bytes),
        (const __half *)W_gate_aligned,
        (const uint2 *)((const char *)W_up_aligned + dq_bytes),
        (const __half *)W_up_aligned,
        (const block_q8_1 *)x8, ids, M, K / 256, K / 32, n_expert_used);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" int ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec(
        const void * W_gate_aligned, const void * W_up_aligned,
        const float * X_f32, const int32_t * ids, const float * weights,
        float * mid_f32,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec";
    if (!W_gate_aligned || !W_up_aligned || !X_f32 || !ids || !weights || !mid_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (n_tokens < 1 || n_tokens > 16 || M <= 0 || K <= 0 || n_experts <= 0 ||
        n_expert_used <= 0 || n_expert_used > n_experts || K % 1024 != 0) {
        return -1;
    }
    ggml_cuda_pool_alloc<char> q8_pool;
    bool folded_hit = false;
    char *x8 = iq2_aligned_quantize_xn(
        tag, X_f32, K, n_tokens, &q8_pool, stream, &folded_hit);
    if (!x8) return -2;

    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    dim3 grid((unsigned)M, (unsigned)(n_tokens * n_expert_used), 1);
    const uint2  *qs_g = (const uint2 *)((const char *)W_gate_aligned + dq_bytes);
    const __half *dq_g = (const __half *)W_gate_aligned;
    const uint2  *qs_u = (const uint2 *)((const char *)W_up_aligned + dq_bytes);
    const __half *dq_u = (const __half *)W_up_aligned;
    if (folded_hit && n_tokens == 1 &&
        ds4_mmq_q8_fold_oracle_enabled()) {
        const size_t q8_bytes = (size_t)K * sizeof(block_q8_1) / QK8_1;
        const uint64_t mid_count =
            (uint64_t)M * (uint64_t)n_expert_used;
        const size_t mid_bytes = (size_t)mid_count * sizeof(float);
        block_q8_1 *fresh = nullptr;
        float *reference = nullptr;
        uint32_t *mismatch_device = nullptr;
        const bool allocated =
            cudaMalloc((void **)&fresh, q8_bytes) == cudaSuccess &&
            cudaMalloc((void **)&reference, mid_bytes) == cudaSuccess &&
            cudaMalloc((void **)&mismatch_device, sizeof(uint32_t)) == cudaSuccess &&
            fresh && reference && mismatch_device;
        if (!allocated) {
            (void)cudaGetLastError();
            cudaError_t cleanup_err = cudaSuccess;
            cleanup_err = ds4_mmq_q8_fold_oracle_free(
                fresh, "aligned-iq2-fresh", cleanup_err);
            cleanup_err = ds4_mmq_q8_fold_oracle_free(
                reference, "aligned-iq2-reference", cleanup_err);
            cleanup_err = ds4_mmq_q8_fold_oracle_free(
                mismatch_device, "aligned-iq2-mismatch", cleanup_err);
            if (cleanup_err != cudaSuccess) (void)cudaGetLastError();
            g_q8_fold_oracle_skips++;
        } else {
            cudaError_t oracle_err = cudaMemsetAsync(
                mismatch_device, 0, sizeof(uint32_t), stream);
            if (oracle_err == cudaSuccess) {
                quantize_row_q8_1_cuda(
                    X_f32, /*ids=*/nullptr, fresh, GGML_TYPE_IQ2_XXS,
                    /*ne00=*/K, /*s11=*/K, /*s12=*/K, /*s13=*/K,
                    /*ne0=*/K, /*ne1=*/1, /*ne2=*/1, /*ne3=*/1, stream);
                oracle_err = cudaGetLastError();
            }
            if (oracle_err == cudaSuccess) {
                iq2_xxs_aligned_moe_gate_up_mid_kernel<<<grid, 32, 0, stream>>>(
                    mid_f32, qs_g, dq_g, qs_u, dq_u,
                    (const block_q8_1 *)x8, ids, weights,
                    M, K / 256, K / 32, n_expert_used, clamp);
                oracle_err = cudaGetLastError();
            }
            if (oracle_err == cudaSuccess) {
                iq2_xxs_aligned_moe_gate_up_mid_kernel<<<grid, 32, 0, stream>>>(
                    reference, qs_g, dq_g, qs_u, dq_u, fresh,
                    ids, weights, M, K / 256, K / 32,
                    n_expert_used, clamp);
                oracle_err = cudaGetLastError();
            }
            if (oracle_err == cudaSuccess) {
                q8_fold_output_compare_kernel<<<
                    (unsigned)((mid_count + 255u) / 256u), 256, 0, stream>>>(
                        mismatch_device, mid_f32, reference, mid_count);
                oracle_err = cudaGetLastError();
            }
            if (oracle_err == cudaSuccess) {
                oracle_err = cudaMemcpyAsync(
                    mid_f32, reference, mid_bytes,
                    cudaMemcpyDeviceToDevice, stream);
            }
            uint32_t mismatch_host = 0u;
            if (oracle_err == cudaSuccess) {
                oracle_err = cudaMemcpyAsync(
                    &mismatch_host, mismatch_device, sizeof(mismatch_host),
                    cudaMemcpyDeviceToHost, stream);
            }
            if (oracle_err == cudaSuccess) {
                oracle_err = cudaStreamSynchronize(stream);
            }
            oracle_err = ds4_mmq_q8_fold_oracle_free(
                fresh, "aligned-iq2-fresh", oracle_err);
            oracle_err = ds4_mmq_q8_fold_oracle_free(
                reference, "aligned-iq2-reference", oracle_err);
            oracle_err = ds4_mmq_q8_fold_oracle_free(
                mismatch_device, "aligned-iq2-mismatch", oracle_err);
            if (oracle_err != cudaSuccess) {
                (void)cudaGetLastError();
                g_q8_fold_oracle_skips++;
                fprintf(stderr, "%s: fold consumer oracle failed\n", tag);
                return -3;
            }
            g_q8_fold_oracle_output_calls++;
            g_q8_fold_oracle_aligned_iq2_calls++;
            if (mismatch_host != 0u) {
                g_q8_fold_oracle_output_mismatches++;
                fprintf(stderr,
                        "ds4: CUDA Q8_1 fold oracle found an IQ2 MoE "
                        "consumer output mismatch; retained canonical "
                        "output\n");
            }
            return 0;
        }
    }
    /* v0.4 V6: verify widths dedup expert overlap (see the dedup kernel's
     * header comment).  n_tokens==1 has no cross-token overlap and keeps
     * the per-slot kernel; widths beyond the verify envelope likewise.
     * DS4_CUDA_NO_MOE_DEDUP restores the per-slot kernel (diagnostic). */
    static int moe_dedup_en = -1;
    if (moe_dedup_en < 0) moe_dedup_en = getenv("DS4_CUDA_NO_MOE_DEDUP") == NULL;
    if (moe_dedup_en && n_tokens >= 2 && n_tokens <= 8) {
        const int n_slots = n_tokens * n_expert_used;
        switch (n_tokens) {
#define DS4_GATEUP_DEDUP_CASE(NT) \
        case NT: \
            iq2_xxs_aligned_moe_gate_up_mid_dedup_kernel<NT><<<grid, 32, 0, stream>>>( \
                mid_f32, qs_g, dq_g, qs_u, dq_u, \
                (const block_q8_1 *)x8, ids, weights, M, K / 256, K / 32, \
                n_expert_used, n_slots, clamp); \
            break;
        DS4_GATEUP_DEDUP_CASE(2)
        DS4_GATEUP_DEDUP_CASE(3)
        DS4_GATEUP_DEDUP_CASE(4)
        DS4_GATEUP_DEDUP_CASE(5)
        DS4_GATEUP_DEDUP_CASE(6)
        DS4_GATEUP_DEDUP_CASE(7)
        DS4_GATEUP_DEDUP_CASE(8)
#undef DS4_GATEUP_DEDUP_CASE
        }
    } else {
        iq2_xxs_aligned_moe_gate_up_mid_kernel<<<grid, 32, 0, stream>>>(
            mid_f32, qs_g, dq_g, qs_u, dq_u,
            (const block_q8_1 *)x8, ids, weights, M, K / 256, K / 32, n_expert_used, clamp);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" int ds4_mmq_iq2_xxs_aligned_moe_vec(
        const void * W_aligned, const float * X_f32, const int32_t * ids, float * out_f32,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_moe_vec";
    if (!W_aligned || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    // n_tokens 1..16 (the vec-tier envelope; each warp reads one activation
    // row selected by assignment/n_expert_used).
    // K % 1024: the lane->(block,pair) mapping covers 4 blocks per pass.
    if (n_tokens < 1 || n_tokens > 16 || M <= 0 || K <= 0 || n_experts <= 0 ||
        n_expert_used <= 0 || n_expert_used > n_experts || K % 1024 != 0) {
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);

    // Quantize X into canonical Q8_1, exactly as ds4_mmq_moe_vec_impl does, so
    // the aligned path shares its activation numerics (and its persistent
    // scratch when enabled).
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        GGML_TYPE_IQ2_XXS, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const __half *dq = (const __half *)W_aligned;
    const uint2  *qs = (const uint2 *)((const char *)W_aligned + dq_bytes);

    dim3 grid((unsigned)M, (unsigned)(n_tokens * n_expert_used), 1);
    iq2_xxs_aligned_moe_vec_kernel<<<grid, 32, 0, stream>>>(
        out_f32, qs, dq, (const block_q8_1 *)src1_q8_1_ptr, ids, M, K / 256,
        K / 32, n_expert_used);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }

    ds4_mmq_sanitize_f32(out_f32, (uint64_t)n_tokens * (uint64_t)M * (uint64_t)n_expert_used, stream);
    return 0;
}

extern "C" int ds4_mmq_q2_K_moe_down_sum6_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_down_sum6_vec_impl<GGML_TYPE_Q2_K>(
        "ds4_mmq_q2_K_moe_down_sum6_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_down_sum6_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_down_sum6_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_down_sum6_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_mxfp4_moe_down_sum6_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_down_sum6_vec_impl<GGML_TYPE_MXFP4>(
        "ds4_mmq_mxfp4_moe_down_sum6_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_gate_up_mid_vec(
        const void * W_gate, const void * W_up,
        const float * X, const int32_t * ids, const float * weights, float * mid,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    return ds4_mmq_moe_gate_up_mid_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_gate_up_mid_vec", W_gate, W_up, X, ids, weights, mid,
        M, K, n_tokens, n_experts, n_expert_used, clamp, stream);
}

extern "C" int ds4_mmq_q4_K_moe_gate_up_mid_vec(
        const void * W_gate, const void * W_up,
        const float * X, const int32_t * ids, const float * weights, float * mid,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    return ds4_mmq_moe_gate_up_mid_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_gate_up_mid_vec", W_gate, W_up, X, ids, weights, mid,
        M, K, n_tokens, n_experts, n_expert_used, clamp, stream);
}

extern "C" int ds4_mmq_mxfp4_moe_gate_up_mid_vec(
        const void * W_gate, const void * W_up,
        const float * X, const int32_t * ids, const float * weights, float * mid,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    return ds4_mmq_moe_gate_up_mid_vec_impl<GGML_TYPE_MXFP4>(
        "ds4_mmq_mxfp4_moe_gate_up_mid_vec", W_gate, W_up, X, ids, weights, mid,
        M, K, n_tokens, n_experts, n_expert_used, clamp, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_silu,
        int M, int K, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_vec", W_a, W_b, X, ids, out_silu,
        M, K, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_pair_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_silu,
        int M, int K, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair_vec", W_a, W_b, X, ids, out_silu,
        M, K, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair_raw_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_raw_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_raw_vec", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_pair_raw_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_raw_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair_raw_vec", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q8_0_dense_vec(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_vec_impl<GGML_TYPE_Q8_0>(
        "ds4_mmq_q8_0_dense_vec", W, X, out, M, N, K,
        /*q4_weight_device_resident=*/0, stream);
}

static int ds4_mmq_pointer_is_device_resident(
        const void *ptr, int expected_device) {
    if (!ptr || expected_device < 0) return 0;
    cudaPointerAttributes attr = {};
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
#if defined(GGML_USE_HIP) && HIP_VERSION >= 60000000
    return attr.type == cudaMemoryTypeDevice &&
           attr.device == expected_device;
#elif defined(GGML_USE_HIP)
    return attr.memoryType == cudaMemoryTypeDevice &&
           attr.device == expected_device;
#elif CUDART_VERSION >= 10000
    return attr.type == cudaMemoryTypeDevice &&
           attr.device == expected_device;
#else
    return attr.memoryType == cudaMemoryTypeDevice &&
           attr.device == expected_device;
#endif
}

extern "C" int ds4_mmq_q4_K_dense_vec(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    int weight_device_resident = 0;
    /* Keep pointer introspection out of generic Q4 MMVQ traffic. It is only
     * needed when the exact-shape persistent candidate could be considered;
     * the full runtime uses the explicit provenance API below instead. */
    if (M == 32768 && N == 1 && K == 1024) {
        weight_device_resident = ds4_mmq_pointer_is_device_resident(
            W, ggml_cuda_get_device());
    }
    return ds4_mmq_dense_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_dense_vec", W, X, out, M, N, K,
        weight_device_resident, stream);
}

extern "C" int ds4_mmq_q4_K_dense_vec_with_weight_residency(
        const void * W, const float * X, float * out,
        int M, int N, int K, int weight_device_resident,
        cudaStream_t stream) {
    return ds4_mmq_dense_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_dense_vec_with_weight_residency",
        W, X, out, M, N, K, weight_device_resident > 0, stream);
}

extern "C" int ds4_mmq_q4_K_grouped_vec(
        const void *W, const float *X, float *out,
        int M, int K, int n_groups, cudaStream_t stream) {
    return ds4_mmq_q4_K_grouped_batch_vec_impl(
        W, X, out, M, K, 1, n_groups, stream);
}

extern "C" int ds4_mmq_q4_K_grouped_batch_vec(
        const void *W, const float *X, float *out,
        int M, int K, int n_tokens, int n_groups, cudaStream_t stream) {
    return ds4_mmq_q4_K_grouped_batch_vec_impl(
        W, X, out, M, K, n_tokens, n_groups, stream);
}

extern "C" int ds4_mmq_q4_K_dense_pair_vec(
        const void * W0, const void * W1, const float * X,
        float * out0, float * out1,
        int M0, int M1, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_pair_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_dense_pair_vec", W0, W1, X, out0, out1,
        M0, M1, N, K, stream);
}

// Explicit instantiations. One per quant type the public API exposes.
// Each instantiation drags in the load_tiles_<type> + vec_dot_<type>_*
// device functions from mmq.cuh, so the .o objects below contain everything
// needed to link against the public C entries.
template void mul_mat_q_case<GGML_TYPE_Q8_0>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q2_K>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ2_XXS>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_MXFP4>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
