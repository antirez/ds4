#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_CUDA_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_CUDA_ATTENTION_SCORE_CAP = 8192u,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    DS4_CUDA_TOPK_MERGE_GROUP = 8u,
    /* perf-02 split-KV: fixed logical rows per chunk (shared scores = 2KB),
     * grid S = ceil(n_score / CHUNK) clamped, so block count grows with ctx. */
    DS4_CUDA_SPLITKV_CHUNK = 512u,
    DS4_CUDA_SPLITKV_SCORE_CAP = 512u,
    DS4_CUDA_SPLITKV_S_MAX = 16u,
    DS4_CUDA_SPLITKV_S_FLOOR = 4u
};

/* struct ds4_gpu_tensor is defined in ds4_gpu.h (no longer opaque as of
 * the device-aware CUDA PR). Field layout includes the new device_id
 * tag and is read by the WITH_DEVICE-wrapped tensor APIs below. */

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

#include "ds4_gpu_mgpu.h"
#include "ds4_iq2_tables_cuda.inc"

typedef struct {
    ds4_gpu_attention_decode_row row[DS4_GPU_ATTENTION_DECODE_BATCH_MAX];
} cuda_attention_decode_row_table;

static_assert(sizeof(cuda_attention_decode_row_table) <= 3072u,
              "attention row table must fit in CUDA kernel parameters");

static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered;
static thread_local bool g_glm_mtp_verify_mode;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_hmm_direct;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static cudaStream_t g_model_prefetch_stream;
static cudaStream_t g_model_upload_stream;
static int g_cublas_ready;
static int g_quality_mode;
static int g_decode_fast_attention;
static int g_decode_score_vec4;
static int g_xdev_sync_debug;
static int g_xdev_force_cuda_peer;
static int g_xdev_force_host_bounce;
static int g_cuda_disable_qkv_rms_fused;
static int g_cuda_no_window_attention;
static int g_cuda_decode_heads8_online;
static int g_cuda_decode_score4;
static int g_cuda_decode_score8;
static int g_cuda_no_decode_value512;
static int g_cuda_no_top1;
static int g_cuda_end_stream_sync;
static int g_cuda_no_setdevice_cache;
static int g_cuda_exact_score_split_graph;
static int g_cuda_exact_score_split_ldg;
static int g_cuda_exact_score_split_vec4;
static int g_cuda_exact_score_split_vec4_plain;
static int g_cuda_exact_score_split_dim2;
static int g_cuda_exact_score_split_fuse_inv_rope;
static int g_cuda_moe_decode_graph;
static int g_current_logical_tier = -1;
static int g_ssd_streaming_mode;

typedef struct {
    int valid;
    int logical_tier;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t slot_count;
    uint32_t compact_count;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t gate_capacity;
    uint64_t up_capacity;
    uint64_t down_capacity;
    int32_t *slot_selected_ptr;
    uint64_t slot_selected_capacity;
    ds4_gpu_tensor slot_selected_tensor;
} cuda_stream_selected_cache;

static cuda_stream_selected_cache g_stream_selected_cache;

static void cuda_stream_selected_cache_invalidate(void) {
    g_stream_selected_cache.valid = 0;
}

static void cuda_stream_selected_cache_release(void) {
    const int tier = g_stream_selected_cache.logical_tier;
    if (tier >= 0 && tier < g_n_gpus) {
        (void)ds4_gpu_set_current_device(tier);
    }
    if (g_stream_selected_cache.gate_ptr) {
        (void)cudaFree(g_stream_selected_cache.gate_ptr);
    }
    if (g_stream_selected_cache.up_ptr) {
        (void)cudaFree(g_stream_selected_cache.up_ptr);
    }
    if (g_stream_selected_cache.down_ptr) {
        (void)cudaFree(g_stream_selected_cache.down_ptr);
    }
    if (g_stream_selected_cache.slot_selected_ptr) {
        (void)cudaFree(g_stream_selected_cache.slot_selected_ptr);
    }
    memset(&g_stream_selected_cache, 0, sizeof(g_stream_selected_cache));
    g_stream_selected_cache.logical_tier = -1;
}

typedef struct {
    cudaGraph_t     graph;
    cudaGraphExec_t exec;
    cudaGraphNode_t score_node;
    cudaGraphNode_t final_node;
    cudaGraphNode_t rope_node;
    uint32_t        n_head;
    uint32_t        head_dim;
    uint32_t        S;
    uint32_t        final_threads;
    uint32_t        n_rot;
    int             fuses_inv_rope;
    int             valid;
} cuda_score_split_graph_cache;

static cuda_score_split_graph_cache g_score_split_graph[DS4_MAX_GPUS];
static void attention_decode_score_split_graph_destroy_one(int logical_tier);

typedef struct {
    cudaGraph_t     graph;
    cudaGraphExec_t exec;
    cudaGraphNode_t xq_node;
    cudaGraphNode_t gate_node;
    cudaGraphNode_t midq_node;
    cudaGraphNode_t down_node;
    uint32_t        n_expert;
    uint32_t        expert_in_dim;
    uint32_t        expert_mid_dim;
    uint32_t        out_dim;
    int             valid;
} cuda_moe_decode_graph_cache;

static cuda_moe_decode_graph_cache g_moe_decode_graph[DS4_MAX_GPUS];

static int cuda_q4_mma_ok(void) {
    /* Cached once: all tiers on this host are the same GPU model. */
    static int cached = -1;
    if (cached < 0) {
        if (getenv("DS4_CUDA_MOE_NO_Q4_MMA") != NULL) {
            cached = 0;
        } else {
            int dev = 0, major = 0, minor = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
            cached = (major > 7 || (major == 7 && minor >= 5)) ? 1 : 0;
        }
    }
    return cached;
}





static int cuda_q4_mma_tile16_shmem_ok(int which_down);


static void routed_moe_decode_graph_destroy_one(int logical_tier);

#include "cuda/runtime.inc"
#include "models/deepseek/cuda/dense_attention.inc"
#include "models/deepseek/cuda/control.inc"
#include "cuda/common_dispatch.inc"
#include "models/deepseek/cuda/moe.inc"
#include "models/deepseek/cuda/hc.inc"
#include "cuda/runtime_services.inc"
#include "models/glm/cuda/kernels.inc"
