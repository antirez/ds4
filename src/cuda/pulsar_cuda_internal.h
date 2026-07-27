/* pulsar_cuda_internal.h — internal shared declarations for the cuda/ translation units.
 * Produced by the multi-TU split of pulsar_cuda.cu; edit freely (the
 * generator is not part of the build).
 *
 * No -rdc: __global__/__device__/__constant__ symbols never cross TU
 * boundaries. The shared __device__ helpers below are static
 * __forceinline__, so each TU gets its own copy. */
#ifndef PULSAR_CUDA_INTERNAL_H
#define PULSAR_CUDA_INTERNAL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cublasLt.h>
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
#include <unordered_set>
#include <vector>

#include "pulsar_gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define PULSAR_CUDA_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context.  11712 fits under the
     * GB10 48 KB shared-memory limit. */
    PULSAR_CUDA_ATTENTION_SCORE_CAP = 11712u,
    PULSAR_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    PULSAR_CUDA_TOPK_MERGE_GROUP = 8u
};

#define PULSAR_FP8_KV_BLOCK 64u
#define PULSAR_FP8_KV_NBLK(HD) (((HD) + PULSAR_FP8_KV_BLOCK - 1u) / PULSAR_FP8_KV_BLOCK)
#define PULSAR_FP8_KV_ROWBYTES(HD) ((HD) + PULSAR_FP8_KV_NBLK(HD) * sizeof(float))

/*
 * Microscaling (MX / OCP) compressed-KV storage.  One E8M0 (power-of-two)
 * scale byte per BLOCK=32 elements, laid out per row as [data ...][scales ...]:
 *   MXFP8 (E4M3 data): HD data bytes  + NBLK scale bytes  (HD=512 -> 528 B/row)
 *   MXFP4 (E2M1 data): HD/2 nibble bytes + NBLK scale bytes (HD=512 -> 272 B/row)
 * This is the CUTLASS-consumable layout (float_ue8m0_t scales, block 32); the
 * GEMM path re-tiles the scales into CUTLASS's swizzled SF layout at use time.
 */
#define PULSAR_MXKV_BLOCK 32u
#define PULSAR_MXKV_NBLK(HD) (((HD) + PULSAR_MXKV_BLOCK - 1u) / PULSAR_MXKV_BLOCK)
#define PULSAR_MXKV_FP8_ROWBYTES(HD) ((HD) + PULSAR_MXKV_NBLK(HD))
#define PULSAR_MXKV_FP4_ROWBYTES(HD) (((HD) + 1u) / 2u + PULSAR_MXKV_NBLK(HD))
/* KV cache storage format selector (compile/runtime). */
#define PULSAR_MXKV_FMT_NONE 0u
#define PULSAR_MXKV_FMT_FP8  1u
#define PULSAR_MXKV_FMT_FP4  2u
#define PULSAR_MXKV_ROWBYTES(FMT, HD) \
    ((FMT) == PULSAR_MXKV_FMT_FP4 ? PULSAR_MXKV_FP4_ROWBYTES(HD) \
   : (FMT) == PULSAR_MXKV_FMT_FP8 ? PULSAR_MXKV_FP8_ROWBYTES(HD) \
   : (HD) * sizeof(float))

/*
 * PULSAR_ATTN_PACK compressed-KV storage (value-preserving).  A comp row today is
 * f32 with the nope dims already fp8-roundtripped in place (fp8_kv_quantize:
 * per-64 block scale = exp2f(ceilf(log2f(fmaxf(amax,1e-4)/448))), e4m3
 * clamp-roundtrip) and the n_rot rope tail untouched f32.  The packed row
 * stores exactly those values:
 *   [n_nope e4m3 bytes][n_nope/64 E8M0 scale bytes][pad to 4B][n_rot f32]
 * head_dim 512 / n_rot 64 -> 448 + 7 + 1 + 256 = 712 B/row (vs 2048 f32).
 * The E8M0 byte is the scale exponent + 127 (power-of-two by construction),
 * so decode (e4m3 value * 2^(e8-127); rope read f32) is bit-identical to the
 * f32 cache.  Must stay in sync with PULSAR_ENGINE_ATTN_PACK_ROWBYTES.
 */
#define PULSAR_ATTN_PACK_NROT 64u
#define PULSAR_ATTN_PACK_NOPE(HD) ((HD) - PULSAR_ATTN_PACK_NROT)
#define PULSAR_ATTN_PACK_SCALES_PAD(HD) \
    ((PULSAR_ATTN_PACK_NOPE(HD) / PULSAR_FP8_KV_BLOCK + 3u) & ~3u)
#define PULSAR_ATTN_PACK_ROWBYTES(HD) \
    ((uint64_t)PULSAR_ATTN_PACK_NOPE(HD) + PULSAR_ATTN_PACK_SCALES_PAD(HD) + \
     (uint64_t)PULSAR_ATTN_PACK_NROT * 4u)

struct pulsar_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

/* Hyper-connection residual-stream stored sample type (task #62; see the
 * PULSAR_HC_ELT_SIZE note in pulsar_gpu.h). pulsar_hc_t is the on-device STORAGE type of
 * the six swap-coupled HC residual carriers; loads promote to f32 and stores
 * round from f32 so all in-kernel accumulation stays f32 (torch semantics:
 * bf16 storage, f32 math). Only the carriers use this — flat_hc (RMSNorm out)
 * and hc_mix/hc_split (Sinkhorn control weights) stay f32. sizeof(pulsar_hc_t)
 * MUST equal PULSAR_HC_ELT_SIZE. */
#ifdef PULSAR_HC_F32
typedef float pulsar_hc_t;
__device__ __forceinline__ static float pulsar_hc_load(const pulsar_hc_t *p, uint64_t i) { return p[i]; }
__device__ __forceinline__ static void  pulsar_hc_store(pulsar_hc_t *p, uint64_t i, float v) { p[i] = v; }
#else
typedef __nv_bfloat16 pulsar_hc_t;
__device__ __forceinline__ static float pulsar_hc_load(const pulsar_hc_t *p, uint64_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ static void  pulsar_hc_store(pulsar_hc_t *p, uint64_t i, float v) { p[i] = __float2bfloat16(v); }
#endif
static_assert(sizeof(pulsar_hc_t) == PULSAR_HC_ELT_SIZE, "pulsar_hc_t size must match PULSAR_HC_ELT_SIZE");

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

/* ---- shared types ---- */

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

struct fp8_mx_weight { const void *host_base; uint64_t offset, in_dim, out_dim; __nv_fp8_e4m3 *data; unsigned char *scale; };

/* ---- shared host globals ---- */

extern cublasHandle_t g_cublas;
extern int g_cublas_ready;
extern int g_quality_mode;
extern cublasLtHandle_t g_cublaslt;
extern std::unordered_set<uint64_t> g_fp8_offsets;

/* ---- shared host functions ---- */

void *cuda_tmp_alloc(uint64_t bytes, const char *what);
void cuda_fp8_weight_cache_clear(void);
int cuda_attention_score_buffer_fits(uint32_t n_comp);
const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what);
int cuda_ok(cudaError_t err, const char *what);
int cublas_ok(cublasStatus_t st, const char *what);
int cuda_matmul_fp8_hc_expand_tensor_labeled(
        pulsar_gpu_tensor       *out_hc,
        pulsar_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        const pulsar_gpu_tensor *block_add,
        const pulsar_gpu_tensor *residual_hc,
        const pulsar_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label);

/* ---- shared __device__ inline helpers (per-TU copies; no -rdc) ---- */

__device__ static __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static __forceinline__ float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static __forceinline__ float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

#endif /* PULSAR_CUDA_INTERNAL_H */
