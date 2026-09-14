// Resident Q4_K Q-B transient F16 scratch and lifecycle.
// One layer is expanded at a time; the exact Q8_K path remains the fallback.

#include "../cuda/ds4_q4_dequant_vec.cuh"

enum {
    DS4_ROCM_Q4_ATTN_Q_B_MAX_LAYERS = 80u,
    DS4_ROCM_Q4_K_TYPE = 12u,
    DS4_ROCM_Q4_ATTN_Q_B_IN_DIM = 1024u,
    DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM = 32768u,
};

static uint64_t g_rocm_q4_attn_q_b_f16_generation = 1u;
static int g_rocm_q4_attn_q_b_f16_hard_failure;
static int g_rocm_q4_attn_q_b_f16_pending_evict;
/* The resident default rebuilds one layer at a time into this combined
 * allocation. The first 64 MiB hold W_F16; the suffix holds the largest
 * preflighted X_F16 batch. ROCm submits graph work on stream 0; keep the
 * mutex through the complete dequant/copy/GEMM/epilogue enqueue sequence so two host callers cannot
 * interleave reuse of any region. */
static void *g_rocm_q4_attn_q_b_transient_f16_scratch;
static uint64_t g_rocm_q4_attn_q_b_transient_f16_scratch_bytes;
static uint64_t g_rocm_q4_attn_q_b_transient_f16_weight_bytes;
static pthread_mutex_t g_rocm_q4_attn_q_b_f16_cache_mu =
    PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_rocm_q4_attn_q_b_f16_build_mu =
    PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_rocm_q4_attn_q_b_transient_f16_mu =
    PTHREAD_MUTEX_INITIALIZER;

static int rocm_q4_attn_q_b_env_value_eq(
        const char *value, size_t n, const char *literal) {
    const size_t literal_n = strlen(literal);
    if (n != literal_n) return 0;
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)value[i]) !=
            tolower((unsigned char)literal[i])) {
            return 0;
        }
    }
    return 1;
}

/* Return -1 when unset, 0 for an explicit false value, and 1 otherwise.
 * Keep value-aware rollback and assertion semantics. */
static int rocm_q4_attn_q_b_env_bool(const char *name) {
    const char *value = getenv(name);
    if (!value) return -1;
    while (isspace((unsigned char)*value)) value++;
    size_t n = strlen(value);
    while (n != 0u && isspace((unsigned char)value[n - 1u])) n--;
    if (n == 0u) return 1;
    if (rocm_q4_attn_q_b_env_value_eq(value, n, "1") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "true") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "yes") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "on")) {
        return 1;
    }
    if (rocm_q4_attn_q_b_env_value_eq(value, n, "0") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "false") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "no") ||
        rocm_q4_attn_q_b_env_value_eq(value, n, "off")) {
        return 0;
    }
    /* Preserve the project's traditional presence-enables behavior for
     * unknown non-empty values while still handling conventional booleans. */
    return 1;
}

static uint64_t rocm_q4_attn_q_b_env_u64(
        const char *name,
        uint64_t fallback,
        uint64_t min_value,
        uint64_t max_value) {
    const char *value = getenv(name);
    if (!value) return fallback;
    while (isspace((unsigned char)*value)) value++;
    if (!*value) return fallback;

    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(value, &end, 10);
    if (end == value || errno == ERANGE) return fallback;
    while (isspace((unsigned char)*end)) end++;
    if (*end != '\0' || parsed < min_value) return fallback;
    const uint64_t result = (uint64_t)parsed;
    return result > max_value ? max_value : result;
}

static int rocm_q4_attn_q_b_transient_f16_disabled(void) {
    return rocm_q4_attn_q_b_env_bool(
        "DS4_ROCM_DISABLE_Q4_ATTN_Q_B_TRANSIENT_F16") == 1;
}

static uint64_t rocm_q4_attn_q_b_transient_f16_min_tokens(void) {
    return rocm_q4_attn_q_b_env_u64(
        "DS4_ROCM_Q4_ATTN_Q_B_TRANSIENT_F16_MIN_TOKENS",
        4096u, 32u, UINT32_MAX);
}

static int rocm_q4_attn_q_b_transient_f16_policy_allowed(void) {
    return !rocm_q4_attn_q_b_transient_f16_disabled() &&
           !g_ssd_streaming_mode &&
           !g_quality_mode &&
           !g_q8_f16_disabled_for_multi_model;
}

/* Read-only lookup for the automatic path.  Normal full-model ROCm loading
 * may use either a contiguous device image or hipMalloc-backed range arenas.
 * Accept both, but never mapped/registered host memory and never populate the
 * range cache here: that would move I/O or page migration into prefill.  Model
 * cache construction is complete before session preflight begins. */
static const char *rocm_q4_attn_q_b_device_resident_source(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes) {
    const char *image =
        cuda_model_image_range_ptr(model_map, offset, bytes);
    if (image) return image;
    if (!model_map || bytes == 0u || offset > UINT64_MAX - bytes) {
        return NULL;
    }
    const uint64_t end = offset + bytes;
    const auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end() &&
        exact->second < g_model_ranges.size()) {
        const cuda_model_range &range = g_model_ranges[exact->second];
        if (range.host_base == model_map && !range.host_registered &&
            range.device_ptr && range.offset == offset &&
            bytes <= range.bytes) {
            return range.device_ptr;
        }
    }
    for (const cuda_model_range &range : g_model_ranges) {
        if (range.host_base != model_map || range.host_registered ||
            !range.device_ptr || offset < range.offset ||
            range.offset > UINT64_MAX - range.bytes) {
            continue;
        }
        const uint64_t range_end = range.offset + range.bytes;
        if (end <= range_end) {
            return range.device_ptr + (offset - range.offset);
        }
    }
    return NULL;
}

static int rocm_q4_attn_q_b_f16_try_runtime_evict(void);

static int rocm_q4_attn_q_b_f16_fallback(
        int required, int rejected, int build_failure) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    (void)rejected;
    if (build_failure) {
        /* A submission/launch failure is backend-wide for this optional
         * specialization. Fail closed for later layers instead of retrying
         * the same hipBLAS or kernel error dozens of times per prefill. */
        g_rocm_q4_attn_q_b_f16_hard_failure = 1;
        g_rocm_q4_attn_q_b_f16_pending_evict = 1;
    }
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    /* Runtime callers release the scratch mutex before eviction. A failed
     * synchronization leaves the scratch disabled until a lifecycle boundary
     * can safely retry the release. */
    if (build_failure) (void)rocm_q4_attn_q_b_f16_try_runtime_evict();
    return required ? -1 : 0;
}

static int rocm_q4_attn_q_b_f16_circuit_open(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    const int open = g_rocm_q4_attn_q_b_f16_hard_failure;
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    return open;
}

__device__ __forceinline__ static void
rocm_q4_attn_q_b_get_scale_min(
        uint32_t group,
        const uint8_t *scales,
        uint8_t *scale,
        uint8_t *minimum) {
    if (group < 4u) {
        *scale = scales[group] & 63u;
        *minimum = scales[group + 4u] & 63u;
    } else {
        *scale = (scales[group + 4u] & 0x0fu) |
                 ((scales[group - 4u] >> 6u) << 4u);
        *minimum = (scales[group + 4u] >> 4u) |
                   ((scales[group] >> 6u) << 4u);
    }
}

/* Expand one contiguous 16-value chunk per thread.  Compared with launching a
 * 256-thread workgroup for every Q4_K block, this cuts the logical thread count
 * by 16x while preserving the row-major [out_dim, in_dim] layout consumed as
 * W^T by hipBLAS. */
__global__ static void rocm_dequant_q4_K_attn_q_b_f16_kernel(
        __half *dst,
        const cuda_block_q4_K *src,
        uint64_t in_dim,
        uint64_t out_dim) {
    const uint64_t chunk =
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t chunks_per_row = in_dim / 16u;
    const uint64_t total_chunks = out_dim * chunks_per_row;
    if (chunk >= total_chunks) return;

    // Both callers supply packed rows with K divisible by 256. Flattening
    // the 16-value chunks avoids row division; one chunk has one scale/min.
    const cuda_block_q4_K *xb = src + (chunk >> 4u);
    const uint32_t group = (uint32_t)((chunk >> 1u) & 7u);
    const uint32_t byte0 = (group >> 1u) * 32u +
                           (uint32_t)(chunk & 1u) * 16u;
    const float d = __half2float(
        __ushort_as_half((unsigned short)xb->d));
    const float dmin = __half2float(
        __ushort_as_half((unsigned short)xb->dmin));
    uint8_t scale = 0;
    uint8_t minimum = 0;
    rocm_q4_attn_q_b_get_scale_min(group, xb->scales, &scale, &minimum);

#pragma unroll
    for (uint32_t k = 0; k < 16u; k++) {
        const uint8_t packed = xb->qs[byte0 + k];
        const uint32_t q =
            (group & 1u) ? (packed >> 4u) : (packed & 0x0fu);
        dst[chunk * 16u + k] =
            __float2half(d * (float)scale * (float)q -
                         dmin * (float)minimum);
    }
}

static int rocm_q4_attn_q_b_f16_desc_valid(
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *desc,
        uint64_t model_size,
        uint64_t *f16_bytes) {
    if (!desc || desc->weight_type != DS4_ROCM_Q4_K_TYPE ||
        desc->in_dim != DS4_ROCM_Q4_ATTN_Q_B_IN_DIM ||
        desc->out_dim != DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM) {
        return 0;
    }
    uint64_t row_bytes = 0;
    uint64_t expected_weight_bytes = 0;
    uint64_t elems = 0;
    uint64_t expanded_bytes = 0;
    if (!cuda_u64_mul_checked(desc->in_dim / CUDA_QK_K,
                              sizeof(cuda_block_q4_K), &row_bytes) ||
        !cuda_u64_mul_checked(desc->out_dim, row_bytes,
                              &expected_weight_bytes) ||
        !cuda_u64_mul_checked(desc->in_dim, desc->out_dim, &elems) ||
        !cuda_u64_mul_checked(elems, sizeof(__half), &expanded_bytes)) {
        return 0;
    }
    if (desc->weight_bytes != expected_weight_bytes ||
        !cuda_model_range_fits(model_size, desc->weight_offset,
                               desc->weight_bytes)) {
        return 0;
    }
    if (f16_bytes) *f16_bytes = expanded_bytes;
    return 1;
}

static int rocm_q4_attn_q_b_f16_memory_has_room(
        uint64_t scratch_bytes,
        uint64_t working_set_reserve_bytes,
        uint64_t *free_bytes_out,
        uint64_t *total_bytes_out,
        uint64_t *reserve_bytes_out) {
    size_t free_b = 0;
    size_t total_b = 0;
    const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess || total_b == 0u) {
        (void)cudaGetLastError();
        return 0;
    }
    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes =
        cuda_q8_f16_cache_reserve_bytes(total_bytes);
    uint64_t required_free = 0;
    if (!cuda_u64_add_checked(scratch_bytes, reserve_bytes,
                              &required_free) ||
        !cuda_u64_add_checked(required_free, working_set_reserve_bytes,
                              &required_free)) {
        return 0;
    }
    if (free_bytes_out) *free_bytes_out = free_bytes;
    if (total_bytes_out) *total_bytes_out = total_bytes;
    if (reserve_bytes_out) *reserve_bytes_out = reserve_bytes;
    return required_free <= free_bytes;
}

static int rocm_q4_attn_q_b_transient_f16_layout(
        uint64_t rows,
        uint64_t *weight_bytes_out,
        uint64_t *x_bytes_out,
        uint64_t *total_bytes_out) {
    uint64_t weight_elems = 0;
    uint64_t weight_bytes = 0;
    uint64_t x_elems = 0;
    uint64_t x_bytes = 0;
    uint64_t total_bytes = 0;
    if (rows == 0u ||
        !cuda_u64_mul_checked(DS4_ROCM_Q4_ATTN_Q_B_IN_DIM,
                              DS4_ROCM_Q4_ATTN_Q_B_OUT_DIM,
                              &weight_elems) ||
        !cuda_u64_mul_checked(weight_elems, sizeof(__half),
                              &weight_bytes) ||
        !cuda_u64_mul_checked(rows, DS4_ROCM_Q4_ATTN_Q_B_IN_DIM,
                              &x_elems) ||
        !cuda_u64_mul_checked(x_elems, sizeof(__half), &x_bytes) ||
        !cuda_u64_add_checked(weight_bytes, x_bytes, &total_bytes) ||
        total_bytes > (uint64_t)SIZE_MAX) {
        return 0;
    }
    if (weight_bytes_out) *weight_bytes_out = weight_bytes;
    if (x_bytes_out) *x_bytes_out = x_bytes;
    if (total_bytes_out) *total_bytes_out = total_bytes;
    return 1;
}

/* Caller holds g_rocm_q4_attn_q_b_transient_f16_mu.  Growth is a preflight
 * operation. Allocate the replacement first, then synchronize and retire the
 * old arena: an allocation failure must not silently destroy the capacity
 * already advertised by the current cache generation. */
static int rocm_q4_attn_q_b_transient_f16_ensure_locked(
        uint64_t required_bytes,
        uint64_t weight_bytes,
        uint64_t working_set_reserve_bytes,
        int *allocated_out) {
    if (allocated_out) *allocated_out = 0;
    if (required_bytes == 0u || weight_bytes == 0u ||
        required_bytes > (uint64_t)SIZE_MAX) {
        return 0;
    }
    if (g_rocm_q4_attn_q_b_transient_f16_scratch &&
        g_rocm_q4_attn_q_b_transient_f16_weight_bytes == weight_bytes &&
        g_rocm_q4_attn_q_b_transient_f16_scratch_bytes >= required_bytes) {
        return 1;
    }

    if (!rocm_q4_attn_q_b_f16_memory_has_room(
            required_bytes, working_set_reserve_bytes,
            NULL, NULL, NULL)) {
        return 0;
    }
    void *scratch = NULL;
    cudaError_t err = cudaMalloc(&scratch, (size_t)required_bytes);
    if (err != cudaSuccess || !scratch) {
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "Q4 attn_q_b transient scratch allocation failed "
                "(%.2f MiB): %s\n",
                (double)required_bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    if (g_rocm_q4_attn_q_b_transient_f16_scratch) {
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4 attn_q_b transient scratch growth sync failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            (void)cudaFree(scratch);
            return 0;
        }
        err = cudaFree(g_rocm_q4_attn_q_b_transient_f16_scratch);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4 attn_q_b transient scratch free failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            (void)cudaFree(scratch);
            return 0;
        }
    }
    g_rocm_q4_attn_q_b_transient_f16_scratch = scratch;
    g_rocm_q4_attn_q_b_transient_f16_scratch_bytes = required_bytes;
    g_rocm_q4_attn_q_b_transient_f16_weight_bytes = weight_bytes;
    if (allocated_out) *allocated_out = 1;
    return 1;
}

/* Success leaves the transient mutex held through the caller's complete GPU
 * enqueue sequence.  No allocation or synchronization is permitted here. */
static int rocm_q4_attn_q_b_transient_f16_acquire(
        uint64_t rows,
        __half **weight_f16_out,
        __half **x_f16_out) {
    uint64_t weight_bytes = 0;
    uint64_t x_bytes = 0;
    uint64_t total_bytes = 0;
    if (!weight_f16_out || !x_f16_out ||
        !rocm_q4_attn_q_b_transient_f16_layout(
            rows, &weight_bytes, &x_bytes,
            &total_bytes)) {
        return 0;
    }
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_transient_f16_mu);
    if (!g_rocm_q4_attn_q_b_transient_f16_scratch ||
        g_rocm_q4_attn_q_b_transient_f16_weight_bytes != weight_bytes ||
        g_rocm_q4_attn_q_b_transient_f16_scratch_bytes < total_bytes) {
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_transient_f16_mu);
        return 0;
    }
    *weight_f16_out =
        (__half *)g_rocm_q4_attn_q_b_transient_f16_scratch;
    *x_f16_out = (__half *)(
        (char *)g_rocm_q4_attn_q_b_transient_f16_scratch + weight_bytes);
    return 1;
}

static void rocm_q4_attn_q_b_transient_f16_release_acquired(void) {
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_transient_f16_mu);
}

static void rocm_q4_attn_q_b_f16_clear_locked(int reset_circuit) {
    if (reset_circuit) g_rocm_q4_attn_q_b_f16_hard_failure = 0;
    g_rocm_q4_attn_q_b_f16_pending_evict = 0;
    if (++g_rocm_q4_attn_q_b_f16_generation == 0u)
        g_rocm_q4_attn_q_b_f16_generation = 1u;
}

/* The caller owns build_mu. Keep transient_mu through synchronization and
 * free so an enqueue cannot lose its scratch before the final consumer. */
static int rocm_q4_attn_q_b_f16_release_with_build_lock(
        int reset_circuit) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_transient_f16_mu);
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);

    if (g_rocm_q4_attn_q_b_transient_f16_scratch) {
        const cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4 attn_q_b F16 cache release sync failed: %s\n",
                    cudaGetErrorString(sync_err));
            (void)cudaGetLastError();
            g_rocm_q4_attn_q_b_f16_hard_failure = 1;
            g_rocm_q4_attn_q_b_f16_pending_evict = 1;
            pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
            pthread_mutex_unlock(&g_rocm_q4_attn_q_b_transient_f16_mu);
            return 0;
        }
    }
    int ok = 1;
    if (g_rocm_q4_attn_q_b_transient_f16_scratch) {
        const cudaError_t free_err =
            cudaFree(g_rocm_q4_attn_q_b_transient_f16_scratch);
        if (free_err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "Q4 attn_q_b transient scratch free failed: %s\n",
                    cudaGetErrorString(free_err));
            (void)cudaGetLastError();
            ok = 0;
        } else {
            g_rocm_q4_attn_q_b_transient_f16_scratch = NULL;
            g_rocm_q4_attn_q_b_transient_f16_scratch_bytes = 0;
            g_rocm_q4_attn_q_b_transient_f16_weight_bytes = 0;
        }
    }
    if (!ok) {
        g_rocm_q4_attn_q_b_f16_hard_failure = 1;
        g_rocm_q4_attn_q_b_f16_pending_evict = 1;
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_transient_f16_mu);
        return 0;
    }
    rocm_q4_attn_q_b_f16_clear_locked(reset_circuit);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_transient_f16_mu);
    return 1;
}

static int rocm_q4_attn_q_b_f16_try_runtime_evict(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_build_mu);
    const int ok = rocm_q4_attn_q_b_f16_release_with_build_lock(0);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
    return ok;
}

extern "C" int ds4_gpu_release_q4_attn_q_b_f16_sidecars(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_build_mu);
    const int ok = rocm_q4_attn_q_b_f16_release_with_build_lock(1);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
    return ok;
}

extern "C" uint64_t ds4_gpu_q4_attn_q_b_f16_cache_generation(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    const uint64_t generation = g_rocm_q4_attn_q_b_f16_generation;
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    return generation;
}

extern "C" int ds4_gpu_make_room_for_q4_attn_q_b_f16_session(void) {
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_build_mu);
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_transient_f16_mu);
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    const int needs_reset =
        g_rocm_q4_attn_q_b_transient_f16_scratch != NULL ||
        g_rocm_q4_attn_q_b_f16_hard_failure ||
        g_rocm_q4_attn_q_b_f16_pending_evict;
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_transient_f16_mu);
    if (!needs_reset) {
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return 1;
    }
    const int ok = rocm_q4_attn_q_b_f16_release_with_build_lock(1);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
    return ok;
}

static int rocm_q4_attn_q_b_prepare_transient_f16(
        const void *model_map,
        uint64_t model_size,
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *descs,
        uint32_t count,
        uint32_t max_prefill_rows,
        uint64_t working_set_reserve_bytes,
        uint64_t *prepared_bytes) {
    const uint64_t min_tokens =
        rocm_q4_attn_q_b_transient_f16_min_tokens();
    if ((uint64_t)max_prefill_rows < min_tokens ||
        !rocm_q4_attn_q_b_transient_f16_policy_allowed() ||
        rocm_q4_attn_q_b_f16_circuit_open() ||
        !g_cublas_ready ||
        model_map != g_model_host_base ||
        model_size != g_model_registered_size) {
        return 0;
    }

    /* Serialize preparation with lifecycle release and model-range teardown.
     * Every Q-B source must already belong to device storage; preflight never
     * registers host pages or populates the mutable range cache. */
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_build_mu);
    if (!rocm_q4_attn_q_b_transient_f16_policy_allowed() ||
        rocm_q4_attn_q_b_f16_circuit_open() ||
        !g_cublas_ready ||
        model_map != g_model_host_base ||
        model_size != g_model_registered_size) {
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return 0;
    }

    uint64_t weight_f16_bytes = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t desc_f16_bytes = 0;
        if (!rocm_q4_attn_q_b_f16_desc_valid(
                &descs[i], model_size, &desc_f16_bytes) ||
            (weight_f16_bytes != 0u &&
             weight_f16_bytes != desc_f16_bytes)) {
            pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
            return 0;
        }
        weight_f16_bytes = desc_f16_bytes;

        if (!rocm_q4_attn_q_b_device_resident_source(
                model_map, descs[i].weight_offset,
                descs[i].weight_bytes)) {
            pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
            return 0;
        }
    }

    uint64_t layout_weight_bytes = 0;
    uint64_t total_bytes = 0;
    if (!rocm_q4_attn_q_b_transient_f16_layout(
            max_prefill_rows, &layout_weight_bytes, NULL,
            &total_bytes) ||
        layout_weight_bytes != weight_f16_bytes) {
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
        return 0;
    }

    int allocated = 0;
    pthread_mutex_lock(&g_rocm_q4_attn_q_b_transient_f16_mu);
    const int ready = rocm_q4_attn_q_b_transient_f16_ensure_locked(
        total_bytes, layout_weight_bytes, working_set_reserve_bytes,
        &allocated);
    if (ready && allocated) {
        pthread_mutex_lock(&g_rocm_q4_attn_q_b_f16_cache_mu);
        if (++g_rocm_q4_attn_q_b_f16_generation == 0u) {
            g_rocm_q4_attn_q_b_f16_generation = 1u;
        }
        pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_cache_mu);
    }
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_transient_f16_mu);
    pthread_mutex_unlock(&g_rocm_q4_attn_q_b_f16_build_mu);
    if (!ready) return 0;

    if (allocated) {
        if (prepared_bytes) *prepared_bytes = total_bytes;
        fprintf(stderr,
                DS4_GPU_LOG_PREFIX
                "prepared %.2f MiB Q4 attn_q_b transient F16 scratch "
                "for up to %u rows (min batch %llu tokens)\n",
                (double)total_bytes / 1048576.0,
                max_prefill_rows,
                (unsigned long long)min_tokens);
    }
    return 1;
}

extern "C" int ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
        const void *model_map,
        uint64_t model_size,
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *descs,
        uint32_t count,
        uint32_t max_prefill_rows,
        uint64_t working_set_reserve_bytes,
        uint64_t *prepared_bytes) {
    if (prepared_bytes) *prepared_bytes = 0;
    if (!model_map || !descs || count == 0u ||
        count > DS4_ROCM_Q4_ATTN_Q_B_MAX_LAYERS ||
        max_prefill_rows < 32u) {
        return 0;
    }

    return rocm_q4_attn_q_b_prepare_transient_f16(
        model_map, model_size, descs, count, max_prefill_rows,
        working_set_reserve_bytes, prepared_bytes);
}
