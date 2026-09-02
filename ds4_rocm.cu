#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm.h"
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS4_GPU_BACKEND_NAME "ROCm"
#define DS4_GPU_LOG_PREFIX "ds4: ROCm "
#define DS4_GPU_BLAS_NAME "hipBLAS"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#define FULL_WARP_MASK 0xFFFFFFFFu
#define MASK_T uint32_t
#define DS4_GPU_BACKEND_NAME "CUDA"
#define DS4_GPU_LOG_PREFIX "ds4: CUDA "
#define DS4_GPU_BLAS_NAME "cuBLAS"
#endif

#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "ds4_gpu.h"
#include "ds4_image.h"

static thread_local bool g_dspark_verify_mode;

extern "C" void ds4_gpu_set_dspark_verify_mode(bool enabled) {
    g_dspark_verify_mode = enabled;
}

extern "C" int ds4_mmq_init(int device);
extern "C" int ds4_mmq_iq2_xxs_moe_pair(
    const void *W_a, const void *W_b, const float *X_f32,
    const int32_t *ids, float *out_a, float *out_b,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    cudaStream_t stream);
extern "C" int ds4_mmq_q8_0_dense_vec(
    const void *W, const float *X_f32, float *out_f32,
    int M, int N, int K, cudaStream_t stream);
extern "C" int ds4_mmq_q2_K_moe_down_sum6_vec(
    const void *W, const float *X, const int32_t *ids, float *out,
    int M, int K, int n_tokens, int n_experts, int n_expert_used,
    cudaStream_t stream);
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_ROCM_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_ROCM_ATTENTION_SCORE_CAP = 8192u,
    DS4_ROCM_ATTENTION_RAW_SCORE_CAP = 256u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

/* Defined by the NHI gate service below; the one-token routed-MoE launcher
 * uses this without coupling kernel headers to transport globals. */
static void ds4_rocm_tp_expert_range(uint32_t n_total_expert,
                                     uint32_t *first_expert,
                                     uint32_t *n_expert);

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

typedef struct {
    uint8_t e;
    uint8_t qs[16];
} cuda_block_mxfp4;

static_assert(sizeof(cuda_block_mxfp4) == 17, "cuda_block_mxfp4 must match the GGUF MXFP4 block layout");

/* Twice the MXFP4 values so each 32-value sub-block can use signed-int8
 * dp4a; the factor of 1/2 is folded into the sub-block scale. */
__device__ __constant__ static const int8_t cuda_mxfp4_values_x2[16] = {
     0,  1,  2,  3,  4,  6,  8,  12,
     0, -1, -2, -3, -4, -6, -8, -12,
};

#include "ds4_iq2_tables_cuda.inc"

#include "rocm/ds4_rocm_runtime.cuh"

#include "rocm/ds4_rocm_common.cuh"

extern "C" int ds4_gpu_dspark_gfx1151_fast_path(void) {
    return ds4_rocm_is_gfx1151();
}

#include "rocm/ds4_rocm_q8.cuh"

#include "rocm/ds4_rocm_norm_rope.cuh"

#include "rocm/ds4_rocm_fp8_kv.cuh"

#include "rocm/ds4_rocm_attention.cuh"

#include "rocm/ds4_rocm_hc.cuh"

#include "rocm/ds4_rocm_output.cuh"

#include "rocm/ds4_rocm_indexer.cuh"

#include "rocm/ds4_rocm_embedding_launch.cuh"

#include "rocm/ds4_rocm_matmul.cuh"

#include "rocm/ds4_rocm_fp8_kv_launch.cuh"

#include "rocm/ds4_rocm_compressor.cuh"

#include "rocm/ds4_rocm_attention_launch.cuh"

#include "rocm/ds4_rocm_shared_expert.cuh"

#include "rocm/ds4_rocm_misc_launch.cuh"
#include "rocm/ds4_rocm_router.cuh"

#include "rocm/ds4_rocm_moe.cuh"

#include "rocm/ds4_rocm_moe_launch.cuh"
#include "rocm/ds4_rocm_poc.cuh"

#include "rocm/ds4_rocm_glm.cuh"

#include "rocm/ds4_rocm_hc_output_launch.cuh"

#include "rocm/ds4_rocm_current_api_compat.cuh"

#include "ds4_glm53_vision_gpu.cuh"
#include "ds4_deepseek4_vision_gpu.cuh"
#include "rocm/ds4_rocm_deepseek4_vision.cuh"
#include "rocm/ds4_rocm_tp.cuh"

/* Stage-1 test entries for the TP combine kernels (tests/test_tp_combine_rocm).
 * Loopback only: a writer kernel or the host plays the peer; the NHI pool
 * import path arrives with the transport backend stage. */
extern "C" int ds4_gpu_tp_test_combine(uint32_t n, uint32_t iterations) {
    if (n == 0 || n > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX) return 0;
    const size_t bytes = (size_t)n * sizeof(float);
    float *h_acc = (float *)malloc(bytes);
    float *h_remote = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    float *d_acc = NULL, *d_remote = NULL;
    int pass = 1;
    if (!h_acc || !h_remote || !h_ref || !h_out ||
        hipMalloc(&d_acc, bytes) != hipSuccess ||
        hipMalloc(&d_remote, bytes) != hipSuccess) {
        pass = 0;
        goto done;
    }
    for (uint32_t it = 0; pass && it < iterations; it++) {
        for (uint32_t i = 0; i < n; i++) {
            h_acc[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
            h_remote[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
            h_ref[i] = h_acc[i] + h_remote[i];
        }
        if (hipMemcpy(d_acc, h_acc, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(d_remote, h_remote, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            pass = 0;
            break;
        }
        const uint32_t block = 256;
        const uint32_t grid = (n + block - 1) / block;
        hipLaunchKernelGGL(dsv4_tp_combine_f32_kernel, dim3(grid), dim3(block),
                           0, 0, d_acc, d_remote, n);
        if (hipDeviceSynchronize() != hipSuccess ||
            hipMemcpy(h_out, d_acc, bytes, hipMemcpyDeviceToHost) != hipSuccess ||
            memcmp(h_out, h_ref, bytes) != 0) {
            pass = 0;
            break;
        }
    }
done:
    free(h_acc); free(h_remote); free(h_ref); free(h_out);
    if (d_acc) (void)hipFree(d_acc);
    if (d_remote) (void)hipFree(d_remote);
    return pass;
}

extern "C" int ds4_gpu_tp_test_spin_exchange(int uncached_pool,
                                             int host_stamp,
                                             uint32_t n,
                                             uint32_t seqs) {
    if (n == 0 || n > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX || seqs == 0) return 0;
    const size_t payload_bytes = (size_t)n * sizeof(float);
    const size_t pool_bytes = (size_t)seqs * DS4_ROCM_TP_SLOT_BYTES;
    unsigned char *d_pool = NULL;
    float *d_acc = NULL, *d_src = NULL;
    uint32_t *h_stamp = NULL, *d_stamp_view = NULL;
    int *d_timeout = NULL;
    float *h_acc = (float *)malloc(payload_bytes);
    float *h_src = (float *)malloc(payload_bytes);
    float *h_ref = (float *)malloc(payload_bytes);
    float *h_out = (float *)malloc(payload_bytes);
    hipStream_t spin_stream = NULL, fill_stream = NULL;
    int pass = 1;
    int h_timeout = 0;

    if (!h_acc || !h_src || !h_ref || !h_out) { pass = 0; goto done; }
    if (uncached_pool) {
        if (hipExtMallocWithFlags((void **)&d_pool, pool_bytes,
                                  hipDeviceMallocUncached) != hipSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "TP test: uncached pool allocation unsupported\n");
            pass = 0;
            goto done;
        }
    } else if (hipMalloc(&d_pool, pool_bytes) != hipSuccess) {
        pass = 0;
        goto done;
    }
    if (hipMalloc(&d_acc, payload_bytes) != hipSuccess ||
        hipMalloc(&d_src, payload_bytes) != hipSuccess ||
        hipMalloc(&d_timeout, sizeof(int)) != hipSuccess ||
        hipMemset(d_pool, 0, pool_bytes) != hipSuccess ||
        hipStreamCreate(&spin_stream) != hipSuccess ||
        hipStreamCreate(&fill_stream) != hipSuccess) {
        pass = 0;
        goto done;
    }
    if (host_stamp) {
        if (hipHostMalloc((void **)&h_stamp, sizeof(uint32_t), 0) != hipSuccess ||
            hipHostGetDevicePointer((void **)&d_stamp_view, h_stamp, 0) != hipSuccess) {
            pass = 0;
            goto done;
        }
        *h_stamp = 0;
    }

    for (uint32_t i = 0; i < n; i++) {
        h_acc[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
        h_ref[i] = h_acc[i];
    }
    if (hipMemcpy(d_acc, h_acc, payload_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        pass = 0;
        goto done;
    }

    for (uint32_t seq = 1; pass && seq <= seqs; seq++) {
        unsigned char *slot = d_pool + (size_t)(seq - 1) * DS4_ROCM_TP_SLOT_BYTES;
        for (uint32_t i = 0; i < n; i++) {
            h_src[i] = ((float)(rand() % 1000) - 500.0f) * 0.03125f + (float)seq;
            h_ref[i] += h_src[i];
        }
        if (hipMemcpy(d_src, h_src, payload_bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemset(d_timeout, 0, sizeof(int)) != hipSuccess) {
            pass = 0;
            break;
        }
        /* The spin kernel launches FIRST and must wait; the payload arrives
         * behind it, proving the stamp actually gates the combine. */
        hipLaunchKernelGGL(dsv4_tp_spin_combine_f32_kernel, dim3(1), dim3(256),
                           0, spin_stream, d_acc, slot, n, seq,
                           (unsigned long long)400000000ull,
                           host_stamp ? d_stamp_view : NULL, d_timeout);
        if (host_stamp) {
            if (hipMemcpyAsync(slot, h_src, payload_bytes, hipMemcpyHostToDevice,
                               fill_stream) != hipSuccess ||
                hipStreamSynchronize(fill_stream) != hipSuccess) {
                pass = 0;
            } else {
                __atomic_store_n(h_stamp, seq, __ATOMIC_RELEASE);
            }
        } else {
            hipLaunchKernelGGL(dsv4_tp_slot_fill_kernel, dim3(1), dim3(256),
                               0, fill_stream, slot, d_src, n, seq);
        }
        if (!pass) break;
        if (hipStreamSynchronize(spin_stream) != hipSuccess ||
            hipStreamSynchronize(fill_stream) != hipSuccess) {
            pass = 0;
            break;
        }
        if (hipMemcpy(&h_timeout, d_timeout, sizeof(int),
                      hipMemcpyDeviceToHost) != hipSuccess || h_timeout != 0) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX
                    "TP test: spin timeout at seq %u\n", seq);
            pass = 0;
            break;
        }
    }
    if (pass &&
        (hipMemcpy(h_out, d_acc, payload_bytes, hipMemcpyDeviceToHost) != hipSuccess ||
         memcmp(h_out, h_ref, payload_bytes) != 0)) {
        pass = 0;
    }

done:
    free(h_acc); free(h_src); free(h_ref); free(h_out);
    if (h_stamp) (void)hipHostFree(h_stamp);
    if (d_pool) (void)hipFree(d_pool);
    if (d_acc) (void)hipFree(d_acc);
    if (d_src) (void)hipFree(d_src);
    if (d_timeout) (void)hipFree(d_timeout);
    if (spin_stream) (void)hipStreamDestroy(spin_stream);
    if (fill_stream) (void)hipStreamDestroy(fill_stream);
    return pass;
}

/* Stage-2 GPU helpers for the NHI TP transport (ds4_tp_nhi.c and the
 * two-node live test): uncached pool allocation for wave-resident polling
 * (contract rules 1 and 6), and stream-based fill/spin-combine wrappers
 * whose fill-side stream synchronize is the dispatch-per-send NHI release
 * (rule 8). */
static hipStream_t g_tp_fill_stream;
static hipStream_t g_tp_spin_stream;
static float *g_tp_fill_scratch;
static size_t g_tp_fill_scratch_bytes;
static int *g_tp_spin_timeout_flag;

static int cuda_tp_streams_ready(void) {
    if (!g_tp_fill_stream &&
        hipStreamCreate(&g_tp_fill_stream) != hipSuccess) return 0;
    if (!g_tp_spin_stream &&
        hipStreamCreate(&g_tp_spin_stream) != hipSuccess) return 0;
    if (!g_tp_spin_timeout_flag &&
        hipMalloc(&g_tp_spin_timeout_flag, sizeof(int)) != hipSuccess) return 0;
    return 1;
}

extern "C" int ds4_gpu_tp_pool_alloc_export_uc(uint64_t bytes,
                                               void **dev_ptr,
                                               int *dmabuf_fd) {
    if (dev_ptr) *dev_ptr = NULL;
    if (dmabuf_fd) *dmabuf_fd = -1;
    if (!dev_ptr || !dmabuf_fd || bytes == 0 ||
        (uint64_t)(size_t)bytes != bytes)
        return 0;
    void *ptr = NULL;
    if (!cuda_ok(hipExtMallocWithFlags(&ptr, (size_t)bytes,
                                       hipDeviceMallocUncached),
                 "uncached TP pool allocation"))
        return 0;
    hipDeviceptr_t base = NULL;
    size_t range = 0;
    hipError_t err = hipMemGetAddressRange(&base, &range, (hipDeviceptr_t)ptr);
    if (err != hipSuccess || base != (hipDeviceptr_t)ptr ||
        range != (size_t)bytes) {
        if (err != hipSuccess) (void)cuda_ok(err, "TP pool address range");
        else
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "TP pool is not a dedicated allocation "
                    "(base=%p self=%p range=%zu bytes=%llu)\n",
                    (void *)base, ptr, range, (unsigned long long)bytes);
        (void)hipFree(ptr);
        return 0;
    }
    int fd = -1;
    err = hipMemGetHandleForAddressRange(&fd, (hipDeviceptr_t)ptr,
                                         (size_t)bytes,
                                         hipMemRangeHandleTypeDmaBufFd, 0);
    if (err != hipSuccess || fd < 0) {
        if (err != hipSuccess) (void)cuda_ok(err, "TP pool DMA-BUF export");
        (void)hipFree(ptr);
        return 0;
    }
    *dev_ptr = ptr;
    *dmabuf_fd = fd;
    return 1;
}

extern "C" int ds4_gpu_tp_dev_buf_create(const float *init_host, uint32_t n,
                                         void **dev_ptr) {
    if (dev_ptr) *dev_ptr = NULL;
    if (!dev_ptr || n == 0) return 0;
    float *d = NULL;
    const size_t bytes = (size_t)n * sizeof(float);
    if (hipMalloc(&d, bytes) != hipSuccess) return 0;
    if (init_host && hipMemcpy(d, init_host, bytes,
                               hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(d);
        return 0;
    }
    *dev_ptr = d;
    return 1;
}

extern "C" int ds4_gpu_tp_dev_buf_read(const void *dev_ptr, float *out_host,
                                       uint32_t n) {
    if (!dev_ptr || !out_host || n == 0) return 0;
    return hipMemcpy(out_host, dev_ptr, (size_t)n * sizeof(float),
                     hipMemcpyDeviceToHost) == hipSuccess;
}

extern "C" void ds4_gpu_tp_dev_buf_free(void *dev_ptr) {
    if (dev_ptr) (void)hipFree(dev_ptr);
}

extern "C" int ds4_gpu_tp_fill_release(void *slot_dev, const float *src_host,
                                       uint32_t n, uint32_t stamp) {
    if (!slot_dev || !src_host || n == 0 ||
        n > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX)
        return 0;
    if (!cuda_tp_streams_ready()) return 0;
    const size_t bytes = (size_t)n * sizeof(float);
    if (g_tp_fill_scratch_bytes < bytes) {
        if (g_tp_fill_scratch) (void)hipFree(g_tp_fill_scratch);
        g_tp_fill_scratch = NULL;
        g_tp_fill_scratch_bytes = 0;
        if (hipMalloc(&g_tp_fill_scratch, bytes) != hipSuccess) return 0;
        g_tp_fill_scratch_bytes = bytes;
    }
    if (hipMemcpyAsync(g_tp_fill_scratch, src_host, bytes,
                       hipMemcpyHostToDevice, g_tp_fill_stream) != hipSuccess)
        return 0;
    hipLaunchKernelGGL(dsv4_tp_slot_fill_kernel, dim3(1), dim3(256), 0,
                       g_tp_fill_stream, (unsigned char *)slot_dev,
                       g_tp_fill_scratch, n, stamp);
    /* Dispatch-per-send design: the stream synchronize is the NHI
     * visibility release (contract rule 8). */
    return cuda_ok(hipStreamSynchronize(g_tp_fill_stream),
                   "TP fill release synchronize");
}

extern "C" int ds4_gpu_tp_spin_combine_start(void *acc_dev,
                                             const void *slot_dev,
                                             uint32_t n,
                                             uint32_t expect_stamp,
                                             unsigned long long max_spins) {
    if (!acc_dev || !slot_dev || n == 0 ||
        n > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX)
        return 0;
    if (!cuda_tp_streams_ready()) return 0;
    if (hipMemsetAsync(g_tp_spin_timeout_flag, 0, sizeof(int),
                       g_tp_spin_stream) != hipSuccess)
        return 0;
    hipLaunchKernelGGL(dsv4_tp_spin_combine_f32_kernel, dim3(1), dim3(256), 0,
                       g_tp_spin_stream, (float *)acc_dev,
                       (const unsigned char *)slot_dev, n, expect_stamp,
                       max_spins, (const uint32_t *)NULL,
                       g_tp_spin_timeout_flag);
    return 1;
}

extern "C" int ds4_gpu_tp_spin_combine_wait(int *timed_out) {
    if (timed_out) *timed_out = 1;
    if (!g_tp_spin_stream || !g_tp_spin_timeout_flag) return 0;
    if (hipStreamSynchronize(g_tp_spin_stream) != hipSuccess) return 0;
    int flag = 0;
    if (hipMemcpy(&flag, g_tp_spin_timeout_flag, sizeof(int),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return 0;
    if (timed_out) *timed_out = flag;
    return 1;
}

/* Production ROCm/NHI row gates.  The graph keeps fixed tp_out/tp_in
 * tensors; this bridge copies each local partial into the rotating TX pool,
 * publishes a role-tagged stamp, and eagerly enqueues a wait-copy from the
 * matching RX slot.  A service thread waits only the event recorded before
 * that spin (never the stream tail), submits one host ioctl, then returns RX
 * credit after the wait-copy's final-reader event. */
enum { DS4_ROCM_TP_QUEUE = 512 };

typedef struct {
    uint32_t layer;
    uint32_t gate;
    uint64_t seq;
    hipEvent_t tx_ready;
    hipEvent_t rx_consumed;
    int in_use;
    int rx_queued; /* producer queued the matching RX spin-copy */
} ds4_rocm_tp_request;

static ds4_gpu_tensor *g_tp_engine_slab;
static uint64_t g_tp_engine_out_offset;
static uint64_t g_tp_engine_in_offset;
static uint32_t g_tp_engine_slots;
static uint32_t g_tp_engine_n_embd;
static uint64_t g_tp_engine_seq;
static unsigned long long g_tp_engine_max_spins = 800000000ull;
static int32_t g_tp_split_rank;
static int32_t g_tp_split_world = 1;
static int32_t g_tp_attn_head_split;
static int32_t g_tp_session_batch_mode;

static ds4_gpu_tp_nhi_tx_slot_fn g_tp_nhi_tx_slot_fn;
static ds4_gpu_tp_nhi_rx_slot_fn g_tp_nhi_rx_slot_fn;
static ds4_gpu_tp_nhi_seq_fn g_tp_nhi_acquire_tx_fn;
static ds4_gpu_tp_nhi_seq_fn g_tp_nhi_submit_fn;
static ds4_gpu_tp_nhi_seq_fn g_tp_nhi_consumed_fn;
static ds4_gpu_tp_nhi_fail_fn g_tp_nhi_fail_fn;
static void *g_tp_nhi_ud;

static pthread_t g_tp_engine_thread;
static int g_tp_engine_thread_running;
static int g_tp_engine_shutdown;
static int g_tp_engine_device;
static uint32_t *g_tp_engine_state_host; /* 0=healthy, 1=failed, 2=shutdown */
static uint32_t *g_tp_engine_state_dev;
static int g_tp_engine_failure_reported;
static pthread_mutex_t g_tp_engine_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_tp_engine_cond = PTHREAD_COND_INITIALIZER;
static ds4_rocm_tp_request g_tp_engine_requests[DS4_ROCM_TP_QUEUE];
static uint32_t g_tp_engine_pending[DS4_ROCM_TP_QUEUE];
static uint32_t g_tp_engine_pending_head;
static uint32_t g_tp_engine_pending_count;
static uint32_t g_tp_engine_inflight[DS4_ROCM_TP_QUEUE];
static uint32_t g_tp_engine_inflight_head;
static uint32_t g_tp_engine_inflight_count;
static uint32_t g_tp_engine_next_record;
static uint32_t g_tp_engine_building;
/* Transport ring capacity in messages (ring slot = seq % msgs); the
 * producer never lets more than this many messages stand between their
 * TX fill and their RX spin.  Plumbed from ds4_tp_nhi_msgs() at engine
 * init via ds4_gpu_tp_nhi_set_ring_msgs; the default is a conservative
 * floor below the 4096-frame production ring (4096 / 64 frames = 64). */
static uint32_t g_tp_engine_ring_msgs = 32;

static void ds4_rocm_tp_expert_range(uint32_t n_total_expert,
                                     uint32_t *first_expert,
                                     uint32_t *n_expert) {
    *first_expert = 0;
    *n_expert = n_total_expert;
    if (g_tp_split_world != 2) return;
    const uint32_t low = n_total_expert / 2u;
    if (g_tp_split_rank == 1) {
        *first_expert = low;
        *n_expert = n_total_expert - low;
    } else {
        *n_expert = low;
    }
}

extern "C" void ds4_gpu_tp_test_set_expert_shard(int rank) {
    if (rank == 0 || rank == 1) {
        g_tp_split_rank = rank;
        g_tp_split_world = 2;
    } else {
        g_tp_split_rank = 0;
        g_tp_split_world = 1;
    }
}

static uint32_t ds4_rocm_tp_stamp(uint32_t rank, uint64_t seq) {
    const uint32_t role_magic = rank == 0 ? 0x5ca16e39u : 0xc35a91e7u;
    return role_magic ^ (uint32_t)seq;
}

static void ds4_rocm_tp_fail(const char *what,
                             const ds4_rocm_tp_request *req) {
    if (g_tp_engine_state_host &&
        __atomic_load_n(g_tp_engine_state_host, __ATOMIC_ACQUIRE) == 0) {
        __atomic_store_n(g_tp_engine_state_host, 1u, __ATOMIC_RELEASE);
    }
    if (__atomic_exchange_n(&g_tp_engine_failure_reported, 1,
                            __ATOMIC_ACQ_REL) == 0) {
        if (req) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX
                    "TP gate failed: %s (layer %u gate %u seq %llu)\n",
                    what, req->layer, req->gate,
                    (unsigned long long)req->seq);
        } else {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "TP gate failed: %s\n", what);
        }
        if (g_tp_nhi_fail_fn) g_tp_nhi_fail_fn(g_tp_nhi_ud);
    }
    /* A big prefill gate can have a producer blocked waiting for a free
     * request record.  Failure must wake that waiter even if the service
     * thread is also exiting. */
    pthread_mutex_lock(&g_tp_engine_mutex);
    pthread_cond_broadcast(&g_tp_engine_cond);
    pthread_mutex_unlock(&g_tp_engine_mutex);
}

static void ds4_rocm_tp_cancel_build(ds4_rocm_tp_request *req) {
    pthread_mutex_lock(&g_tp_engine_mutex);
    if (req) req->in_use = 0;
    if (g_tp_engine_building != 0) g_tp_engine_building--;
    pthread_cond_broadcast(&g_tp_engine_cond);
    pthread_mutex_unlock(&g_tp_engine_mutex);
}

static void *ds4_rocm_tp_service_thread(void *arg) {
    (void)arg;
    if (hipSetDevice(g_tp_engine_device) != hipSuccess) {
        ds4_rocm_tp_fail("service-thread hipSetDevice", NULL);
        return NULL;
    }
    for (;;) {
        pthread_mutex_lock(&g_tp_engine_mutex);
        while (g_tp_engine_pending_count == 0 &&
               g_tp_engine_inflight_count == 0 &&
               (!__atomic_load_n(&g_tp_engine_shutdown, __ATOMIC_ACQUIRE) ||
                g_tp_engine_building != 0))
            pthread_cond_wait(&g_tp_engine_cond, &g_tp_engine_mutex);
        if (g_tp_engine_pending_count == 0 &&
            g_tp_engine_inflight_count == 0 &&
            g_tp_engine_building == 0 &&
            __atomic_load_n(&g_tp_engine_shutdown, __ATOMIC_ACQUIRE)) {
            pthread_mutex_unlock(&g_tp_engine_mutex);
            break;
        }
        const int have_pending = g_tp_engine_pending_count != 0;
        uint32_t index = 0;
        if (have_pending) {
            index = g_tp_engine_pending[g_tp_engine_pending_head];
            g_tp_engine_pending_head =
                (g_tp_engine_pending_head + 1u) % DS4_ROCM_TP_QUEUE;
            g_tp_engine_pending_count--;
        }
        pthread_mutex_unlock(&g_tp_engine_mutex);

        /* Submit as soon as this record's TX fill lands; never wait for an
         * older record's RX final-reader event, so the wire stays busy
         * while the peer is still draining earlier messages. */
        if (have_pending) {
            ds4_rocm_tp_request *req = &g_tp_engine_requests[index];
            if (hipEventSynchronize(req->tx_ready) != hipSuccess)
                ds4_rocm_tp_fail("TX-ready event synchronize", req);
            const uint32_t state = g_tp_engine_state_host
                ? __atomic_load_n(g_tp_engine_state_host, __ATOMIC_ACQUIRE)
                : 1u;
            if (state == 0 &&
                !(g_tp_nhi_submit_fn &&
                  g_tp_nhi_submit_fn(g_tp_nhi_ud, req->seq)))
                ds4_rocm_tp_fail("NHI submit", req);

            pthread_mutex_lock(&g_tp_engine_mutex);
            const uint32_t tail =
                (g_tp_engine_inflight_head + g_tp_engine_inflight_count) %
                DS4_ROCM_TP_QUEUE;
            g_tp_engine_inflight[tail] = index;
            g_tp_engine_inflight_count++;
            pthread_mutex_unlock(&g_tp_engine_mutex);
        }

        /* Retire in-flight records in strict sequence order: consumed_fn
         * requires ordered seqs, so only the head is ever examined. */
        for (;;) {
            pthread_mutex_lock(&g_tp_engine_mutex);
            if (g_tp_engine_inflight_count == 0) {
                pthread_mutex_unlock(&g_tp_engine_mutex);
                break;
            }
            ds4_rocm_tp_request *req =
                &g_tp_engine_requests[g_tp_engine_inflight[g_tp_engine_inflight_head]];
            /* rx_consumed is a complete final-reader marker only once the
             * producer queued the matching RX spin.  A producer that
             * aborted mid-window always trips the failure/shutdown state,
             * which also wakes this wait. */
            while (!req->rx_queued &&
                   g_tp_engine_state_host &&
                   __atomic_load_n(g_tp_engine_state_host,
                                   __ATOMIC_ACQUIRE) == 0 &&
                   !__atomic_load_n(&g_tp_engine_shutdown, __ATOMIC_ACQUIRE))
                pthread_cond_wait(&g_tp_engine_cond, &g_tp_engine_mutex);
            pthread_mutex_unlock(&g_tp_engine_mutex);

            const hipError_t query = hipEventQuery(req->rx_consumed);
            if (query == hipErrorNotReady) {
                pthread_mutex_lock(&g_tp_engine_mutex);
                const int more_pending = g_tp_engine_pending_count != 0;
                pthread_mutex_unlock(&g_tp_engine_mutex);
                if (more_pending) break; /* keep submitting; retire later */
                /* Nothing else to submit: block on the oldest reader.  On
                 * transport failure or shutdown the mapped state word
                 * aborts the spin, so teardown cannot wedge here. */
                if (hipEventSynchronize(req->rx_consumed) != hipSuccess)
                    ds4_rocm_tp_fail("RX-consumed event synchronize", req);
            } else if (query != hipSuccess) {
                ds4_rocm_tp_fail("RX-consumed event query", req);
            }
            if (g_tp_engine_state_host &&
                __atomic_load_n(g_tp_engine_state_host, __ATOMIC_ACQUIRE) == 1u)
                ds4_rocm_tp_fail("RX stamp timeout", req);

            if (g_tp_engine_state_host &&
                __atomic_load_n(g_tp_engine_state_host, __ATOMIC_ACQUIRE) == 0 &&
                g_tp_nhi_consumed_fn &&
                !g_tp_nhi_consumed_fn(g_tp_nhi_ud, req->seq))
                ds4_rocm_tp_fail("NHI RX consume/repost", req);

            pthread_mutex_lock(&g_tp_engine_mutex);
            g_tp_engine_inflight_head =
                (g_tp_engine_inflight_head + 1u) % DS4_ROCM_TP_QUEUE;
            g_tp_engine_inflight_count--;
            req->in_use = 0;
            pthread_cond_broadcast(&g_tp_engine_cond);
            pthread_mutex_unlock(&g_tp_engine_mutex);
        }
    }
    return NULL;
}

extern "C" int ds4_gpu_tp_init(uint32_t rank,
                               ds4_gpu_tensor *slab,
                               uint64_t gpu_flags_off,
                               uint64_t out_off,
                               uint64_t vec_bytes,
                               ds4_gpu_tp_exchange_fn fn,
                               void *ud) {
    (void)rank; (void)slab; (void)gpu_flags_off; (void)out_off;
    (void)vec_bytes; (void)fn; (void)ud;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX
            "ROCm network tensor parallelism requires the NHI gate service\n");
    return 0;
}

extern "C" void ds4_gpu_tp_nhi_set_ring_msgs(uint32_t msgs) {
    if (msgs != 0) g_tp_engine_ring_msgs = msgs;
}

extern "C" int ds4_gpu_tp_nhi_init(
        uint32_t rank,
        ds4_gpu_tensor *slab,
        uint64_t out_offset,
        uint64_t in_offset,
        uint32_t n_slots,
        uint32_t n_embd,
        ds4_gpu_tp_nhi_tx_slot_fn tx_slot_fn,
        ds4_gpu_tp_nhi_rx_slot_fn rx_slot_fn,
        ds4_gpu_tp_nhi_seq_fn acquire_tx_fn,
        ds4_gpu_tp_nhi_seq_fn submit_fn,
        ds4_gpu_tp_nhi_seq_fn consumed_fn,
        ds4_gpu_tp_nhi_fail_fn fail_fn,
        void *ud) {
    if (rank > 1 || !slab ||
        !tx_slot_fn || !rx_slot_fn || !acquire_tx_fn || !submit_fn ||
        !consumed_fn ||
        n_slots == 0 || n_embd == 0 ||
        n_embd > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX ||
        g_tp_engine_thread_running)
        return 0;
    const uint64_t vec_bytes = (uint64_t)n_embd * sizeof(float);
    if (out_offset > slab->bytes || in_offset > slab->bytes ||
        (uint64_t)n_slots * vec_bytes > slab->bytes - out_offset ||
        (uint64_t)n_slots * vec_bytes > slab->bytes - in_offset)
        return 0;

    memset(g_tp_engine_requests, 0, sizeof(g_tp_engine_requests));
    g_tp_engine_slab = slab;
    g_tp_engine_out_offset = out_offset;
    g_tp_engine_in_offset = in_offset;
    g_tp_engine_slots = n_slots;
    g_tp_engine_n_embd = n_embd;
    g_tp_engine_seq = 0;
    g_tp_nhi_tx_slot_fn = tx_slot_fn;
    g_tp_nhi_rx_slot_fn = rx_slot_fn;
    g_tp_nhi_acquire_tx_fn = acquire_tx_fn;
    g_tp_nhi_submit_fn = submit_fn;
    g_tp_nhi_consumed_fn = consumed_fn;
    g_tp_nhi_fail_fn = fail_fn;
    g_tp_nhi_ud = ud;
    g_tp_split_rank = (int32_t)rank;
    g_tp_split_world = 2;
    g_tp_engine_pending_head = 0;
    g_tp_engine_pending_count = 0;
    g_tp_engine_inflight_head = 0;
    g_tp_engine_inflight_count = 0;
    g_tp_engine_next_record = 0;
    g_tp_engine_building = 0;
    __atomic_store_n(&g_tp_engine_shutdown, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_tp_engine_failure_reported, 0, __ATOMIC_RELEASE);
    const char *spins = getenv("DS4_TP_SPIN_MAX");
    if (spins && strtoull(spins, NULL, 10) != 0)
        g_tp_engine_max_spins = strtoull(spins, NULL, 10);

    if (hipGetDevice(&g_tp_engine_device) != hipSuccess ||
        hipHostMalloc((void **)&g_tp_engine_state_host, sizeof(uint32_t),
                      hipHostMallocMapped) != hipSuccess ||
        hipHostGetDevicePointer((void **)&g_tp_engine_state_dev,
                                g_tp_engine_state_host, 0) != hipSuccess) {
        ds4_gpu_tp_shutdown();
        return 0;
    }
    __atomic_store_n(g_tp_engine_state_host, 0u, __ATOMIC_RELEASE);
    for (uint32_t i = 0; i < DS4_ROCM_TP_QUEUE; i++) {
        if (hipEventCreateWithFlags(&g_tp_engine_requests[i].tx_ready,
                                    hipEventReleaseToSystem) != hipSuccess ||
            hipEventCreateWithFlags(&g_tp_engine_requests[i].rx_consumed,
                                    hipEventDefault) != hipSuccess) {
            ds4_gpu_tp_shutdown();
            return 0;
        }
    }
    if (pthread_create(&g_tp_engine_thread, NULL,
                       ds4_rocm_tp_service_thread, NULL) != 0) {
        ds4_gpu_tp_shutdown();
        return 0;
    }
    g_tp_engine_thread_running = 1;
    return 1;
}

extern "C" void ds4_gpu_tp_shutdown(void) {
    if (g_tp_engine_thread_running) {
        pthread_mutex_lock(&g_tp_engine_mutex);
        /* Healthy shutdown rejects new gates but drains every queued
         * submit/final-reader/repost.  A previously latched failure already
         * set state=1, which is the only path that aborts RX spins. */
        __atomic_store_n(&g_tp_engine_shutdown, 1, __ATOMIC_RELEASE);
        pthread_cond_broadcast(&g_tp_engine_cond);
        pthread_mutex_unlock(&g_tp_engine_mutex);
        pthread_join(g_tp_engine_thread, NULL);
        g_tp_engine_thread_running = 0;
        (void)hipDeviceSynchronize();
    }
    if (g_tp_engine_state_host) {
        uint32_t expected = 0;
        (void)__atomic_compare_exchange_n(g_tp_engine_state_host, &expected,
                                          2u, 0, __ATOMIC_ACQ_REL,
                                          __ATOMIC_ACQUIRE);
    }
    for (uint32_t i = 0; i < DS4_ROCM_TP_QUEUE; i++) {
        if (g_tp_engine_requests[i].tx_ready)
            (void)hipEventDestroy(g_tp_engine_requests[i].tx_ready);
        if (g_tp_engine_requests[i].rx_consumed)
            (void)hipEventDestroy(g_tp_engine_requests[i].rx_consumed);
    }
    memset(g_tp_engine_requests, 0, sizeof(g_tp_engine_requests));
    if (g_tp_engine_state_host) (void)hipHostFree(g_tp_engine_state_host);
    g_tp_engine_state_host = NULL;
    g_tp_engine_state_dev = NULL;
    g_tp_engine_slab = NULL;
    g_tp_nhi_tx_slot_fn = NULL;
    g_tp_nhi_rx_slot_fn = NULL;
    g_tp_nhi_acquire_tx_fn = NULL;
    g_tp_nhi_submit_fn = NULL;
    g_tp_nhi_consumed_fn = NULL;
    g_tp_nhi_fail_fn = NULL;
    g_tp_nhi_ud = NULL;
    __atomic_store_n(&g_tp_engine_shutdown, 0, __ATOMIC_RELEASE);
    g_tp_engine_building = 0;
    g_tp_split_rank = 0;
    g_tp_split_world = 1;
    g_tp_attn_head_split = 0;
    g_tp_session_batch_mode = 0;
}

#define DS4_ROCM_TP_BIG_GATE_TAG 0xB16u

/* TX phase of one NHI message: allocate a request record, reserve the
 * transport slot, enqueue the fill + stamp-release kernels, and hand the
 * record to the service thread's pending queue.  Returns the record index,
 * or -1 on failure.  The matching ds4_rocm_tp_enqueue_gate_rx must follow
 * (after at most g_tp_engine_ring_msgs outstanding TX phases) so the RX
 * spin-copy is queued before the service thread retires the record. */
static int ds4_rocm_tp_enqueue_gate_tx(uint32_t layer,
                                       uint32_t gate,
                                       const void *src,
                                       uint32_t n_floats) {
    if (!g_tp_engine_thread_running ||
        __atomic_load_n(&g_tp_engine_shutdown, __ATOMIC_ACQUIRE) ||
        !src || n_floats == 0 ||
        n_floats > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX)
        return -1;
    if (__atomic_load_n(g_tp_engine_state_host, __ATOMIC_ACQUIRE) != 0)
        return -1;

    uint32_t index = UINT32_MAX;
    pthread_mutex_lock(&g_tp_engine_mutex);
    for (;;) {
        if (__atomic_load_n(&g_tp_engine_shutdown, __ATOMIC_ACQUIRE) ||
            __atomic_load_n(g_tp_engine_state_host, __ATOMIC_ACQUIRE) != 0) {
            pthread_mutex_unlock(&g_tp_engine_mutex);
            return -1;
        }
        for (uint32_t n = 0; n < DS4_ROCM_TP_QUEUE; n++) {
            const uint32_t candidate =
                (g_tp_engine_next_record + n) % DS4_ROCM_TP_QUEUE;
            if (!g_tp_engine_requests[candidate].in_use) {
                index = candidate;
                g_tp_engine_next_record =
                    (candidate + 1u) % DS4_ROCM_TP_QUEUE;
                g_tp_engine_requests[candidate].in_use = 1;
                break;
            }
        }
        if (index != UINT32_MAX) break;
        /* A large prefill gate can need more than 512 transport messages.
         * Wait for the service thread to retire older chunks instead of
         * failing the graph encode. */
        pthread_cond_wait(&g_tp_engine_cond, &g_tp_engine_mutex);
    }
    if (g_tp_engine_seq > UINT32_MAX) {
        g_tp_engine_requests[index].in_use = 0;
        pthread_cond_broadcast(&g_tp_engine_cond);
        pthread_mutex_unlock(&g_tp_engine_mutex);
        ds4_rocm_tp_fail("32-bit stamp sequence exhausted", NULL);
        return -1;
    }
    ds4_rocm_tp_request *req = &g_tp_engine_requests[index];
    req->layer = layer;
    req->gate = gate;
    req->seq = g_tp_engine_seq++;
    req->rx_queued = 0;
    g_tp_engine_building++;
    pthread_mutex_unlock(&g_tp_engine_mutex);

    if (!g_tp_nhi_acquire_tx_fn(g_tp_nhi_ud, req->seq)) {
        ds4_rocm_tp_cancel_build(req);
        ds4_rocm_tp_fail("NHI TX slot acquire", req);
        return -1;
    }
    void *tx_slot = g_tp_nhi_tx_slot_fn(g_tp_nhi_ud, req->seq);
    if (!tx_slot) {
        ds4_rocm_tp_cancel_build(req);
        ds4_rocm_tp_fail("null transport slot", req);
        return -1;
    }

    const uint32_t block = 256u;
    const uint32_t grid = (n_floats + block - 1u) / block;
    hipLaunchKernelGGL(dsv4_tp_slot_copy_f32_kernel,
                       dim3(grid), dim3(block), 0, 0,
                       (unsigned char *)tx_slot, (const float *)src,
                       n_floats);
    hipLaunchKernelGGL(dsv4_tp_stamp_release_kernel,
                       dim3(1), dim3(1), 0, 0,
                       (unsigned char *)tx_slot,
                       ds4_rocm_tp_stamp((uint32_t)g_tp_split_rank, req->seq));
    if (hipGetLastError() != hipSuccess ||
        hipEventRecord(req->tx_ready, 0) != hipSuccess) {
        ds4_rocm_tp_cancel_build(req);
        ds4_rocm_tp_fail("gate TX kernel/event enqueue", req);
        return -1;
    }

    pthread_mutex_lock(&g_tp_engine_mutex);
    if (g_tp_engine_pending_count >= DS4_ROCM_TP_QUEUE) {
        req->in_use = 0;
        if (g_tp_engine_building != 0) g_tp_engine_building--;
        pthread_cond_broadcast(&g_tp_engine_cond);
        pthread_mutex_unlock(&g_tp_engine_mutex);
        ds4_rocm_tp_fail("service queue overflow", req);
        return -1;
    }
    const uint32_t tail =
        (g_tp_engine_pending_head + g_tp_engine_pending_count) %
        DS4_ROCM_TP_QUEUE;
    g_tp_engine_pending[tail] = index;
    g_tp_engine_pending_count++;
    if (g_tp_engine_building != 0) g_tp_engine_building--;
    pthread_cond_broadcast(&g_tp_engine_cond);
    pthread_mutex_unlock(&g_tp_engine_mutex);
    return (int)index;
}

/* RX phase of one NHI message: enqueue the spin-copy that waits the peer's
 * stamp and lands the RX payload into dst, then flag the record so the
 * service thread may treat rx_consumed as the final-reader marker and
 * retire it. */
static int ds4_rocm_tp_enqueue_gate_rx(uint32_t index,
                                       void *dst,
                                       uint32_t n_floats) {
    if (index >= DS4_ROCM_TP_QUEUE || !dst || n_floats == 0 ||
        n_floats > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX)
        return 0;
    ds4_rocm_tp_request *req = &g_tp_engine_requests[index];
    const void *rx_slot = g_tp_nhi_rx_slot_fn(g_tp_nhi_ud, req->seq);
    int ok = rx_slot != NULL;
    if (ok) {
        hipLaunchKernelGGL(dsv4_tp_spin_copy_f32_kernel,
                           dim3(1), dim3(256), 0, 0,
                           (float *)dst, (const unsigned char *)rx_slot,
                           n_floats,
                           ds4_rocm_tp_stamp((uint32_t)(g_tp_split_rank ^ 1),
                                             req->seq),
                           g_tp_engine_max_spins,
                           g_tp_engine_state_dev);
        ok = hipGetLastError() == hipSuccess &&
             hipEventRecord(req->rx_consumed, 0) == hipSuccess;
    }
    /* Set rx_queued even on failure: the service thread's retire wait must
     * not linger until shutdown, and the latched failure state makes the
     * retire path skip submit/consume for this record. */
    pthread_mutex_lock(&g_tp_engine_mutex);
    req->rx_queued = 1;
    pthread_cond_broadcast(&g_tp_engine_cond);
    pthread_mutex_unlock(&g_tp_engine_mutex);
    if (!ok) {
        ds4_rocm_tp_fail("gate RX kernel/event enqueue", req);
        return 0;
    }
    return 1;
}

/* Queue one NHI message.  The caller's GPU stream owns ordering: the RX
 * spin-copy is enqueued immediately after the TX fill, so later graph work
 * cannot consume the destination until this message's stamp has arrived. */
static int ds4_rocm_tp_enqueue_gate(uint32_t layer,
                                    uint32_t gate,
                                    const void *src,
                                    void *dst,
                                    uint32_t n_floats) {
    if (!dst) return 0;
    const int index =
        ds4_rocm_tp_enqueue_gate_tx(layer, gate, src, n_floats);
    if (index < 0) return 0;
    return ds4_rocm_tp_enqueue_gate_rx((uint32_t)index, dst, n_floats);
}

extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    if (gate >= 2u) return 0;
    const uint32_t slot = layer * 2u + gate;
    if (slot >= g_tp_engine_slots) return 0;
    const uint64_t vec_bytes =
        (uint64_t)g_tp_engine_n_embd * sizeof(float);
    const float *out = (const float *)((const char *)g_tp_engine_slab->ptr +
                                       g_tp_engine_out_offset +
                                       (uint64_t)slot * vec_bytes);
    float *in = (float *)((char *)g_tp_engine_slab->ptr +
                          g_tp_engine_in_offset +
                          (uint64_t)slot * vec_bytes);
    return ds4_rocm_tp_enqueue_gate(layer, gate, out, in,
                                    g_tp_engine_n_embd);
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) {
    (void)fn;
}

extern "C" void ds4_gpu_tp_set_session_batch_mode(int enabled) {
    g_tp_session_batch_mode = enabled ? 1 : 0;
}

extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    if (g_tp_engine_thread_running) g_tp_split_world = suspend ? 1 : 2;
}

extern "C" void ds4_gpu_tp_keepalive_pause(int paused) {
    (void)paused;
}

extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled) {
    g_tp_attn_head_split = enabled ? 1 : 0;
}

extern "C" int ds4_gpu_tp_failed(void) {
    return __atomic_load_n(&g_tp_engine_failure_reported, __ATOMIC_ACQUIRE) ||
           (g_tp_engine_state_host &&
            __atomic_load_n(g_tp_engine_state_host, __ATOMIC_ACQUIRE) == 1u);
}

extern "C" void ds4_gpu_model_residency_skip(int skip) {
    (void)skip;
}

extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) {
    (void)fn;
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                                          const ds4_gpu_tensor *out_t,
                                          ds4_gpu_tensor *in_t,
                                          uint64_t bytes) {
    (void)rows;
    if (!out_t || !in_t || bytes == 0 || (bytes & 3u) != 0 ||
        bytes > out_t->bytes || bytes > in_t->bytes)
        return 0;

    /* Pipeline in windows: queue up to `window` TX fills before queuing
     * their RX spin-copies, so one message's submit/RTT overlaps the next
     * messages' fills instead of serializing per chunk.  The window must
     * not exceed the transport ring capacity: a ring slot is seq % msgs,
     * so more than `msgs` outstanding unconsumed messages would let a
     * later fill target a ring slot the peer has not consumed yet. */
    uint32_t window = g_tp_engine_ring_msgs;
    if (window > DS4_ROCM_TP_QUEUE / 2u) window = DS4_ROCM_TP_QUEUE / 2u;
    if (window == 0) window = 1;
    int tx_records[DS4_ROCM_TP_QUEUE / 2];

    const uint64_t n_floats = bytes / sizeof(float);
    uint64_t tx_done = 0;
    uint64_t rx_done = 0;
    while (rx_done < n_floats) {
        uint32_t n_window = 0;
        while (tx_done < n_floats && n_window < window) {
            uint64_t left = n_floats - tx_done;
            const uint32_t chunk =
                left > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX ?
                    DS4_ROCM_TP_PAYLOAD_FLOATS_MAX : (uint32_t)left;
            const int index = ds4_rocm_tp_enqueue_gate_tx(
                layer,
                DS4_ROCM_TP_BIG_GATE_TAG,
                (const float *)out_t->ptr + tx_done,
                chunk);
            if (index < 0)
                return 0;
            tx_records[n_window++] = index;
            tx_done += chunk;
        }
        for (uint32_t i = 0; i < n_window; i++) {
            uint64_t left = n_floats - rx_done;
            const uint32_t chunk =
                left > DS4_ROCM_TP_PAYLOAD_FLOATS_MAX ?
                    DS4_ROCM_TP_PAYLOAD_FLOATS_MAX : (uint32_t)left;
            if (!ds4_rocm_tp_enqueue_gate_rx(
                    (uint32_t)tx_records[i],
                    (float *)in_t->ptr + rx_done,
                    chunk))
                return 0;
            rx_done += chunk;
        }
    }
    return 1;
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    (void)layer; (void)rows;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX
            "ROCm/NHI tensor-parallel batch gates are not enabled yet\n");
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (!out || !model_map || !x || full_in_dim == 0 || k_cnt == 0 ||
        out_dim == 0 || (full_in_dim & 31u) != 0 || (k_off & 31u) != 0 ||
        (k_cnt & 31u) != 0 || k_off > full_in_dim ||
        k_cnt > full_in_dim - k_off || full_in_dim > UINT32_MAX ||
        out_dim > UINT32_MAX)
        return 0;
    const uint64_t full_blocks = full_in_dim / 32u;
    const uint64_t slice_blocks = k_cnt / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, out_bytes = 0, x_need = 0;
    if (!cuda_u64_mul_checked(full_blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !cuda_u64_mul_checked(out_dim, sizeof(float), &out_bytes) ||
        !cuda_u64_mul_checked(x_elem_off + k_cnt, sizeof(float), &x_need) ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        out->bytes < out_bytes || x->bytes < x_need)
        return 0;
    const unsigned char *w = (const unsigned char *)cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "q8_0_kslice");
    if (!w) return 0;
    matmul_q8_0_f32_kslice_warp8_kernel<<<
        ((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)out->ptr, w, (const float *)x->ptr,
            full_blocks, k_off / 32u, slice_blocks, out_dim, x_elem_off);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 kslice launch");
}

extern "C" int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type,
        uint64_t full_in_dim, uint64_t k_off, uint64_t k_cnt,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t x_elem_off) {
    /* The Stage-3 ROCm surface currently supports only the Q8_0 shared
     * expert down projection. Keep every other dense type fail-closed. */
    if (weight_type != 8u) return 0;
    return ds4_gpu_matmul_q8_0_kslice_tensor(
        out, model_map, model_size, weight_offset,
        full_in_dim, k_off, k_cnt, out_dim, x, x_elem_off);
}

extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups_total,
        uint32_t group0, uint32_t group_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *heads) {
    if (!out || !low || !model_map || !heads || group_dim == 0 ||
        rank == 0 || n_groups_total == 0 || group_cnt == 0 || out_dim == 0 ||
        group0 > n_groups_total || group_cnt > n_groups_total - group0)
        return 0;
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    uint64_t group_weight_bytes = 0, a_shift = 0;
    if (!cuda_u64_mul3_checked(rank, blocks_a, 34u, &group_weight_bytes) ||
        !cuda_u64_mul_checked(group0, group_weight_bytes, &a_shift) ||
        a_shift > UINT64_MAX - out_a_offset)
        return 0;
    const uint64_t local_low = (uint64_t)group_cnt * rank;
    if (local_low > UINT64_MAX / sizeof(float) ||
        low->bytes < local_low * sizeof(float) ||
        heads->bytes < (uint64_t)group_cnt * group_dim * sizeof(float))
        return 0;
    /* The upstream head-split path packs this rank's owned groups at the
     * buffer base. group0 shifts weights/B-k only, never the heads pointer. */
    ds4_gpu_tensor heads_slice = {
        heads->ptr,
        (uint64_t)group_cnt * group_dim * sizeof(float),
        0,
    };
    ds4_gpu_tensor low_slice = {
        low->ptr,
        local_low * sizeof(float),
        0,
    };
    if (!ds4_gpu_attention_output_low_q8_tensor(
            &low_slice, model_map, model_size, out_a_offset + a_shift,
            group_dim, rank, group_cnt, &heads_slice))
        return 0;
    return ds4_gpu_matmul_q8_0_kslice_tensor(
        out, model_map, model_size, out_b_offset,
        (uint64_t)n_groups_total * rank,
        (uint64_t)group0 * rank, local_low, out_dim, &low_slice, 0);

}

extern "C" int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
        uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0;
    uint64_t post_bytes = 0, comb_bytes = 0, comb_stride = 0;
    if (!out_hc || !block_out || !block_add || !residual_hc || !post ||
        !comb || !cuda_hc_hc_token_count(out_hc, n_embd, n_hc,
                                          &n_tokens64) ||
        !cuda_u64_mul3_checked(n_tokens64, n_embd, sizeof(float),
                               &flat_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd,
                               sizeof(float), &hc_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, n_hc, sizeof(float),
                               &post_bytes) ||
        !cuda_u64_mul_checked(n_hc, n_hc, &comb_stride) ||
        comb_stride > UINT32_MAX ||
        !cuda_u64_mul3_checked(n_tokens64, comb_stride, sizeof(float),
                               &comb_bytes) ||
        block_out->bytes < flat_bytes || block_add->bytes < flat_bytes ||
        residual_hc->bytes < hc_bytes || post->bytes < post_bytes ||
        comb->bytes < comb_bytes)
        return 0;
    const uint32_t n_tokens = (uint32_t)n_tokens64;
    const uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255u) / 256u, 256>>>(
        (float *)out_hc->ptr,
        (const float *)block_out->ptr,
        (const float *)block_add->ptr,
        (const float *)residual_hc->ptr,
        (const float *)post->ptr,
        (const float *)comb->ptr,
        n_embd, n_hc, n_tokens,
        n_hc, (uint32_t)comb_stride, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add launch");
}
