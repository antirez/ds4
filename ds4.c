/* =========================================================================
 * ds4.c - DeepSeek V4 Flash inference engine.
 * =========================================================================
 *
 * This file is deliberately vertical: it owns GGUF loading, the fixed
 * DeepSeek V4 Flash tensor layout, CPU reference kernels, the whole-model
 * Metal graph driver, and tokenizer wiring.  The model shape is not
 * configurable here; every validation step is meant to fail early if a GGUF
 * does not match the one layout this engine implements.
 *
 * Loading is mmap based.  The loader parses only the GGUF header, metadata
 * table, and tensor directory.  Tensor data stays in the kernel page cache
 * until inference touches it, or until Metal wraps slices of the mapping as
 * no-copy MTLBuffers.
 */

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

#include "ds4.h"

#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DS4_NEG_INF (-1.0e30f)
#define DS4_POS_INF ( 1.0e30f)
#define DS4_RMS_EPS ( 1.0e-6f)
#define DS4_HC_EPS  ( 1.0e-6f)
#define DS4_EXPERT_WEIGHT_SCALE (1.5f)
#define DS4_SWIGLU_CLAMP_EXP    (10.0f)
#define DS4_ROPE_FREQ_BASE      (10000.0f)
#define DS4_ROPE_SCALE_FACTOR   (16.0f)
#define DS4_ROPE_YARN_BETA_FAST (32.0f)
#define DS4_ROPE_YARN_BETA_SLOW (1.0f)
#define DS4_COMPRESS_ROPE_FREQ_BASE (160000.0f)
#define DS4_ROPE_ORIG_CTX       UINT64_C(65536)

static const char DS4_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/* DeepSeek recommends Think Max only with at least a 384K-token context window.
 * Below that size we keep ordinary thinking to avoid injecting a prompt that
 * asks for a reasoning budget the allocated context is not meant to hold. */
#define DS4_THINK_MAX_MIN_CONTEXT 393216u

static bool ds4_backend_uses_graph(ds4_backend backend) {
    return backend == DS4_BACKEND_METAL || backend == DS4_BACKEND_CUDA;
}

/* =========================================================================
 * Fixed DeepSeek V4 Flash Shape.
 * =========================================================================
 *
 * These constants define the single model family this program accepts.  The
 * weight binder and metadata validator below check the GGUF against the same
 * numbers so the rest of the inference code can use simple fixed-size paths.
 */

enum {
    DS4_N_LAYER            = 43,
    DS4_N_EMBD             = 4096,
    DS4_N_VOCAB            = 129280,
    DS4_N_HEAD             = 64,
    DS4_N_HEAD_KV          = 1,
    DS4_N_HEAD_DIM         = 512,
    DS4_N_VALUE_DIM        = 512,
    DS4_N_ROT              = 64,
    DS4_N_OUT_GROUP        = 8,
    DS4_N_LORA_Q           = 1024,
    DS4_N_LORA_O           = 1024,
    DS4_N_EXPERT           = 256,
    DS4_N_EXPERT_USED      = 6,
    DS4_N_EXPERT_SHARED    = 1,
    DS4_N_FF_EXP           = 2048,
    DS4_N_HASH_LAYER       = 3,
    DS4_N_SWA              = 128,
    DS4_N_INDEXER_HEAD     = 64,
    DS4_N_INDEXER_HEAD_DIM = 128,
    DS4_N_INDEXER_TOP_K    = 512,
    DS4_N_HC               = 4,
    DS4_N_HC_SINKHORN_ITER = 20,
};

static int g_ds4_lock_fd = -1;

#if defined(__GNUC__) || defined(__clang__)
#define DS4_MAYBE_UNUSED __attribute__((unused))
#else
#define DS4_MAYBE_UNUSED
#endif

/* =========================================================================
 * GGUF Quant Block Formats.
 * =========================================================================
 *
 * These layouts and IQ2 tables match the GGUF quantized tensor format,
 * reduced to only the formats ds4.c currently reads:
 *   - Q2_K routed down experts
 *   - Q4_K routed experts in the high-memory variant
 *   - IQ2_XXS routed gate/up experts
 *   - Q8_K temporary activation blocks for dot products
 */
#define QK_K 256

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;

#define DS4_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]
DS4_STATIC_ASSERT(ds4_block_q2_k_size, sizeof(block_q2_K) == 84);
DS4_STATIC_ASSERT(ds4_block_q4_k_size, sizeof(block_q4_K) == 144);
DS4_STATIC_ASSERT(ds4_block_q8_k_size, sizeof(block_q8_K) == 292);
DS4_STATIC_ASSERT(ds4_block_iq2_xxs_size, sizeof(block_iq2_xxs) == 66);

typedef struct {
    uint32_t ctx_size;
    uint32_t comp_cap;
    uint32_t attn_score_cap;
    uint32_t q8_cap;

    float *plain;
    float *cur;
    float *next;

    float *attn_cur;
    float *attn_norm;
    float *attn_residual;
    float *q;
    float *qr;
    float *qr_norm;
    float *kv_raw;
    float *kv;
    float *heads;
    float *attn_low;
    float *attn_out;
    float *after_attn_hc;
    float *attn_score;

    float *comp;
    float *index_comp;
    float *comp_kv_cur;
    float *comp_sc_cur;
    float *comp_pooled;

    bool *index_allowed;
    float *index_q;
    float *index_weights;
    float *index_scores;

    float *ffn_cur;
    float *ffn_norm;
    float *ffn_moe;
    float *ffn_shared;
    float *ffn_out;
    float *shared_gate;
    float *shared_up;
    float *shared_mid;
    float *routed_mid_all;
    block_q8_K *routed_xq;
    block_q8_K *routed_midq;

    int8_t *q8_xq;
    float *q8_xscale;

    float *hc_flat;
    float *output_flat;
    float *output_pre;
    float *output_weights;
    float *output_embd;
    float *output_norm;
} ds4_cpu_decode_scratch;

static const uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static int8_t iq2xxs_signed_grid[256][128][8];
static int8_t iq2xxs_signs[128][8];
static pthread_once_t iq2xxs_signed_grid_once = PTHREAD_ONCE_INIT;

static void iq2xxs_signed_grid_init(void) {
    for (uint32_t s = 0; s < 128; s++) {
        const uint8_t signs = ksigns_iq2xs[s];
        for (uint32_t j = 0; j < 8; j++) {
            iq2xxs_signs[s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -1 : 1);
        }
    }

    for (uint32_t g = 0; g < 256; g++) {
        const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + g);
        for (uint32_t s = 0; s < 128; s++) {
            const uint8_t signs = ksigns_iq2xs[s];
            for (uint32_t j = 0; j < 8; j++) {
                const int v = (int)grid[j];
                iq2xxs_signed_grid[g][s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -v : v);
            }
        }
    }
}

static inline DS4_MAYBE_UNUSED int32_t dot_iq2_pair_16(const int8_t *grid0, const int8_t *grid1, const int8_t *q8) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t gv = vcombine_s8(vld1_s8(grid0), vld1_s8(grid1));
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), gv, vld1q_s8(q8));
    return vaddvq_s32(acc);
#elif defined(__ARM_NEON)
    const int8x16_t gv = vcombine_s8(vld1_s8(grid0), vld1_s8(grid1));
    const int8x16_t qv = vld1q_s8(q8);
    const int16x8_t p0 = vmull_s8(vget_low_s8(gv), vget_low_s8(qv));
    const int16x8_t p1 = vmull_s8(vget_high_s8(gv), vget_high_s8(qv));
    return vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#else
    int32_t sum = 0;
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid0[i] * (int32_t)q8[i];
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid1[i] * (int32_t)q8[8 + i];
    return sum;
#endif
}

static inline DS4_MAYBE_UNUSED int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t packed = vld1q_u8(q2);
    uint8x16_t shifted;
    switch (shift) {
    case 0: shifted = packed; break;
    case 2: shifted = vshrq_n_u8(packed, 2); break;
    case 4: shifted = vshrq_n_u8(packed, 4); break;
    default: shifted = vshrq_n_u8(packed, 6); break;
    }
    const uint8x16_t vals_u = vandq_u8(shifted, vdupq_n_u8(3));
    const int8x16_t vals = vreinterpretq_s8_u8(vals_u);
    const int8x16_t q8v = vld1q_s8(q8);
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), q8v, vals);
    return vaddvq_s32(acc);
#elif defined(__ARM_NEON)
    uint8_t vals_tmp[16];
    for (uint32_t i = 0; i < 16; i++) vals_tmp[i] = (q2[i] >> shift) & 3;
    const int8x16_t vals = vreinterpretq_s8_u8(vld1q_u8(vals_tmp));
    const int8x16_t q8v = vld1q_s8(q8);
    const int16x8_t p0 = vmull_s8(vget_low_s8(q8v), vget_low_s8(vals));
    const int16x8_t p1 = vmull_s8(vget_high_s8(q8v), vget_high_s8(vals));
    const int32x4_t s0 = vpaddlq_s16(p0);
    const int32x4_t s1 = vpaddlq_s16(p1);
    return vaddvq_s32(vaddq_s32(s0, s1));
#else
    int32_t sum = 0;
    for (uint32_t i = 0; i < 16; i++) sum += (int32_t)q8[i] * (int32_t)((q2[i] >> shift) & 3);
    return sum;
#endif
}

/* =========================================================================
 * Shared Helpers, Allocation Guards, Threads, and Cursor Reads.
 * =========================================================================
 *
 * This section holds process-wide utilities used by all later stages:
 * fatal-error helpers, allocation wrappers, the persistent CPU worker pool,
 * and the small byte cursor used to parse GGUF metadata.
 */

#define DS4_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */
#define DS4_MAX_DIMS   8

typedef struct {
    const char *ptr;
    uint64_t len;
} ds4_str;

typedef ds4_tokens token_vec;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} ds4_cursor;

static void ds4_die(const char *msg) {
    fprintf(stderr, "ds4: %s\n", msg);
    exit(1);
}

/* Attention compression alternates after layer 1: dense early layers, then
 * ratio-4 layers with an indexer and ratio-128 layers without one. */
static uint32_t ds4_layer_compress_ratio(uint32_t il) {
    if (il >= DS4_N_LAYER) ds4_die("DeepSeek4 layer index is outside the fixed model layout");
    if (il < 2) return 0;
    return (il & 1u) == 0 ? 4u : 128u;
}

static void ds4_die_errno(const char *what, const char *path) {
    fprintf(stderr, "ds4: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static bool ds4_streq(ds4_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static bool ds4_str_eq(ds4_str a, ds4_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

static uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool g_alloc_guard_enabled;
static const char *g_alloc_guard_phase;

static void ds4_alloc_guard_begin(const char *phase) {
    g_alloc_guard_phase = phase;
    g_alloc_guard_enabled = true;
}

static void ds4_alloc_guard_end(void) {
    g_alloc_guard_enabled = false;
    g_alloc_guard_phase = NULL;
}

static void ds4_alloc_guard_check(const char *op, size_t size) {
    if (!g_alloc_guard_enabled) return;
    fprintf(stderr,
            "ds4: internal allocation during %s: %s(%zu). "
            "CPU decode is expected to reuse preallocated scratch buffers.\n",
            g_alloc_guard_phase ? g_alloc_guard_phase : "guarded phase",
            op,
            size);
    exit(1);
}

static void *xcalloc(size_t n, size_t size) {
    ds4_alloc_guard_check("calloc", n * size);
    void *p = calloc(n, size);
    if (!p) ds4_die("out of memory");
    return p;
}

static void *xmalloc(size_t size) {
    ds4_alloc_guard_check("malloc", size);
    void *p = malloc(size);
    if (!p) ds4_die("out of memory");
    return p;
}

static char *ds4_strdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static void *xrealloc(void *ptr, size_t size) {
    ds4_alloc_guard_check("realloc", size);
    void *p = realloc(ptr, size);
    if (!p) ds4_die("out of memory");
    return p;
}

static void *xmalloc_zeroed(size_t n, size_t size) {
    if (size != 0 && n > SIZE_MAX / size) ds4_die("allocation size overflow");
    const size_t total = n * size;
    void *p = xmalloc(total ? total : 1);
    /*
     * This is intentionally not calloc(). Large untouched calloc ranges may be
     * represented by the VM through shared zero-page bookkeeping. The CPU decode
     * KV cache grows one token at a time, so using calloc here can move thousands
     * of first-touch faults into generation. On Darwin we have observed this end
     * in a kernel cpt_mapcnt_inc overflow panic instead of a user-space error.
     *
     * Explicitly writing the zeroes while the cache is allocated keeps those VM
     * faults out of the token loop and gives the cache private resident pages.
     */
    memset(p, 0, total);
    return p;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static const char *ds4_log_color_code(ds4_log_type type) {
    switch (type) {
    case DS4_LOG_PREFILL:
    case DS4_LOG_TIMING:
        return "\x1b[36m";
    case DS4_LOG_GENERATION:
    case DS4_LOG_OK:
        return "\x1b[32m";
    case DS4_LOG_KVCACHE:
        return "\x1b[33m";
    case DS4_LOG_TOOL:
        return "\x1b[90m";
    case DS4_LOG_WARNING:
        return "\x1b[38;5;208m";
    case DS4_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

bool ds4_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

static void ds4_vlog(FILE *fp, ds4_log_type type, const char *fmt, va_list ap) {
    const bool colorize = type != DS4_LOG_DEFAULT && ds4_log_is_tty(fp);
    if (colorize) fputs(ds4_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ds4_vlog(fp, type, fmt, ap);
    va_end(ap);
}

static bool write_f32_binary_file(const char *path, const float *data, uint64_t n) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open %s for writing: %s\n", path, strerror(errno));
        return false;
    }
    const size_t nw = fwrite(data, sizeof(float), (size_t)n, fp);
    const bool ok = nw == (size_t)n && fclose(fp) == 0;
    if (!ok) {
        fprintf(stderr, "ds4: failed to write %s\n", path);
        return false;
    }
    return true;
}

static bool read_f32_binary_file(const char *path, float *data, uint64_t n) {
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ds4: failed to stat %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size != n * sizeof(float)) {
        fprintf(stderr,
                "ds4: %s has size %llu bytes, expected %llu bytes\n",
                path,
                (unsigned long long)st.st_size,
                (unsigned long long)(n * sizeof(float)));
        return false;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open %s for reading: %s\n", path, strerror(errno));
        return false;
    }
    const size_t nr = fread(data, sizeof(float), (size_t)n, fp);
    const bool ok = nr == (size_t)n && fclose(fp) == 0;
    if (!ok) {
        fprintf(stderr, "ds4: failed to read %s\n", path);
        return false;
    }
    return true;
}

static bool cpu_directional_steering_enabled(
        const float *dirs,
        float        scale);

static void cpu_directional_steering_project_rows(
        float       *x,
        const float *dirs,
        uint32_t     il,
        uint32_t     rows,
        float        scale);

typedef void (*ds4_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);

#define DS4_MAX_THREADS 32

typedef struct {
    pthread_t threads[DS4_MAX_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;
    pthread_cond_t done_cond;
    uint32_t n_threads;
    uint32_t n_workers;
    uint32_t generation;
    uint32_t done;
    bool initialized;
    bool shutdown;
    ds4_parallel_fn fn;
    void *ctx;
    uint64_t n_rows;
} ds4_thread_pool;

static ds4_thread_pool g_pool;
static __thread int g_parallel_depth;
static uint32_t g_requested_threads;

static void *ds4_worker_main(void *arg) {
    const uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&g_pool.mutex);
        while (seen_generation == g_pool.generation && !g_pool.shutdown) {
            pthread_cond_wait(&g_pool.work_cond, &g_pool.mutex);
        }
        if (g_pool.shutdown) {
            pthread_mutex_unlock(&g_pool.mutex);
            return NULL;
        }

        seen_generation = g_pool.generation;
        ds4_parallel_fn fn = g_pool.fn;
        void *ctx = g_pool.ctx;
        const uint64_t n_rows = g_pool.n_rows;
        const uint32_t n_threads = g_pool.n_threads;
        pthread_mutex_unlock(&g_pool.mutex);

        const uint64_t rows_per_thread = (n_rows + n_threads - 1) / n_threads;
        const uint64_t row0 = (uint64_t)tid * rows_per_thread;
        uint64_t row1 = row0 + rows_per_thread;
        if (row1 > n_rows) row1 = n_rows;
        if (row0 < row1) {
            g_parallel_depth++;
            fn(ctx, row0, row1);
            g_parallel_depth--;
        }

        pthread_mutex_lock(&g_pool.mutex);
        g_pool.done++;
        if (g_pool.done == g_pool.n_workers) {
            pthread_cond_signal(&g_pool.done_cond);
        }
        pthread_mutex_unlock(&g_pool.mutex);
    }
}

/* Create the persistent CPU worker pool.  Decode reuses these threads instead
 * of creating pthreads in the token loop. */
static void ds4_threads_init(void) {
    if (g_pool.initialized) return;

    pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);

    uint32_t n_threads = 12;
    const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (online_cpus > 0) {
        n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
    }

    const char *env = getenv("DS4_THREADS");
    if (env && env[0]) {
        long v = strtol(env, NULL, 10);
        if (v > 0) n_threads = (uint32_t)v;
    }
    if (g_requested_threads > 0) n_threads = g_requested_threads;
    if (n_threads > DS4_MAX_THREADS) n_threads = DS4_MAX_THREADS;
    if (n_threads == 0) n_threads = 1;

    pthread_mutex_init(&g_pool.mutex, NULL);
    pthread_cond_init(&g_pool.work_cond, NULL);
    pthread_cond_init(&g_pool.done_cond, NULL);
    g_pool.n_threads = n_threads;
    g_pool.n_workers = n_threads > 0 ? n_threads - 1 : 0;
    g_pool.generation = 0;
    g_pool.done = 0;
    g_pool.shutdown = false;
    g_pool.initialized = true;

    for (uint32_t i = 1; i < n_threads; i++) {
        if (pthread_create(&g_pool.threads[i], NULL, ds4_worker_main, (void *)(uintptr_t)i) != 0) {
            ds4_die("failed to create worker thread");
        }
    }
}

static void ds4_threads_shutdown(void) {
    if (!g_pool.initialized) return;

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.shutdown = true;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);
    pthread_mutex_unlock(&g_pool.mutex);

    for (uint32_t i = 1; i < g_pool.n_threads; i++) {
        pthread_join(g_pool.threads[i], NULL);
    }

    pthread_cond_destroy(&g_pool.done_cond);
    pthread_cond_destroy(&g_pool.work_cond);
    pthread_mutex_destroy(&g_pool.mutex);
    memset(&g_pool, 0, sizeof(g_pool));
}

/* Run a row-parallel CPU kernel, falling back to serial execution for small
 * jobs or nested calls where spawning more work would only add latency. */
static void ds4_parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx, uint64_t min_parallel_rows) {
    ds4_threads_init();

    if (g_parallel_depth > 0 || g_pool.n_threads <= 1 || n_rows < min_parallel_rows) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.n_rows = n_rows;
    g_pool.done = 0;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);

    const uint64_t rows_per_thread = (n_rows + g_pool.n_threads - 1) / g_pool.n_threads;
    uint64_t main_row1 = rows_per_thread;
    if (main_row1 > n_rows) main_row1 = n_rows;
    pthread_mutex_unlock(&g_pool.mutex);

    if (main_row1 > 0) {
        g_parallel_depth++;
        fn(ctx, 0, main_row1);
        g_parallel_depth--;
    }

    pthread_mutex_lock(&g_pool.mutex);
    while (g_pool.done < g_pool.n_workers) {
        pthread_cond_wait(&g_pool.done_cond, &g_pool.mutex);
    }
    pthread_mutex_unlock(&g_pool.mutex);
}

static void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx) {
    ds4_parallel_for_min_rows(n_rows, fn, ctx, 512);
}

static void cursor_error(ds4_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(ds4_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(ds4_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(ds4_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(ds4_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(ds4_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(ds4_cursor *c, ds4_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

/* =========================================================================
 * GGUF Parsing and Model Mapping.
 * =========================================================================
 *
 * The loader maps the model once, records metadata/tensor descriptors, and
 * leaves tensor bytes in place.  Inference code accesses weights by adding
 * tensor offsets to the mapping instead of copying the GGUF into private
 * structures.
 */

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [15] = {"q8_k",   256, 292},
    [16] = {"iq2_xxs",256,  66},
    [17] = {"iq2_xs", 256,  74},
    [18] = {"iq3_xxs",256,  98},
    [19] = {"iq1_s",  256, 110},
    [20] = {"iq4_nl", 256,  50},
    [21] = {"iq3_s",  256, 110},
    [22] = {"iq2_s",  256,  82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [29] = {"iq1_m",  256,  56},
    [30] = {"bf16",     1,   2},
};

enum {
    DS4_TENSOR_F32      = 0,
    DS4_TENSOR_F16      = 1,
    DS4_TENSOR_Q8_0     = 8,
    DS4_TENSOR_Q2_K     = 10,
    DS4_TENSOR_Q4_K     = 12,
    DS4_TENSOR_IQ2_XXS  = 16,
    DS4_TENSOR_I32      = 26,
};

typedef struct {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
} ds4_kv;

typedef struct {
    ds4_str name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} ds4_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;

    ds4_kv *kv;
    ds4_tensor *tensors;
} ds4_model;

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(ds4_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        ds4_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;

        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

static const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

static const char *tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static ds4_cursor cursor_at(const ds4_model *m, uint64_t pos) {
    ds4_cursor c = {
        .base = m->map,
        .size = m->size,
        .pos = pos,
        .error = {0},
    };
    return c;
}

static ds4_kv *model_find_kv(const ds4_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (ds4_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

static bool model_get_string(const ds4_model *m, const char *key, ds4_str *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

static bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static bool model_get_u64(const ds4_model *m, const char *key, uint64_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT64) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u64(&c, out);
}

static bool model_get_bool(const ds4_model *m, const char *key, bool *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_BOOL) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    uint8_t v = 0;
    if (!cursor_read(&c, &v, sizeof(v))) return false;
    *out = v != 0;
    return true;
}

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} ds4_array_ref;

static bool model_get_array(const ds4_model *m, const char *key, ds4_array_ref *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

static void model_close(ds4_model *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static void model_prefetch_cpu_mapping(const ds4_model *m) {
    if (!m || !m->map || m->size == 0) return;

    /*
     * CPU generation touches expert weights according to router decisions, so a
     * long decode can fault in model pages that the prompt never touched. On
     * current Darwin kernels we have seen those late file-backed faults trigger
     * an OS-level VM panic in map-count accounting. This hint does not copy or
     * pin the GGUF; it just asks the kernel to start bringing the read-only
     * mapping into the page cache before token generation reaches it.
     */
#if defined(POSIX_MADV_WILLNEED)
    const int rc = posix_madvise((void *)m->map, (size_t)m->size, POSIX_MADV_WILLNEED);
    if (rc != 0) {
        ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: POSIX_MADV_WILLNEED failed for CPU model mapping: %s\n",
                strerror(rc));
    }
#else
    (void)m;
#endif
}

/* Read the GGUF metadata table.  Values stay in the mmap; we store offsets so
 * later validation can decode only the keys it needs. */
static void parse_metadata(ds4_model *m, ds4_cursor *c) {
    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) ds4_die("out of memory while allocating metadata table");

    m->alignment = 32;

    for (uint64_t i = 0; i < m->n_kv; i++) {
        ds4_kv *kv = &m->kv[i];

        if (!cursor_string(c, &kv->key)) ds4_die(c->error);
        if (!cursor_u32(c, &kv->type)) ds4_die(c->error);

        kv->value_pos = c->pos;

        if (ds4_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32)
        {
            ds4_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) ds4_die(c->error);
    }
}

/* Read the tensor directory and convert relative GGUF offsets to absolute
 * mmap offsets.  Tensor bytes are still never copied here. */
static void parse_tensors(ds4_model *m, ds4_cursor *c) {
    m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if (!m->tensors) ds4_die("out of memory while allocating tensor table");

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];

        if (!cursor_string(c, &t->name)) ds4_die(c->error);
        if (!cursor_u32(c, &t->ndim)) ds4_die(c->error);
        if (t->ndim == 0 || t->ndim > DS4_MAX_DIMS) {
            ds4_die("tensor has an unsupported number of dimensions");
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) ds4_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                ds4_die("tensor element count overflow");
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) ds4_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) ds4_die(c->error);

        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: tensor %.*s has unsupported GGUF type %u\n",
                (int)t->name.len, t->name.ptr, t->type);
        }
    }

    m->tensor_data_pos = align_up(c->pos, m->alignment);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) {
            ds4_die("tensor offset overflow");
        }
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset))
        {
            ds4_die("tensor points outside GGUF file");
        }
    }
}

/* Open and map the GGUF once.  Metal needs a shared mapping for no-copy
 * MTLBuffers; CPU uses a private read-only mapping to avoid Darwin VM stress.
 * Tokenizer-only callers pass prefetch_cpu=false so inspecting tokens never
 * walks the huge tensor payload. */
static void model_open(ds4_model *m, const char *path, bool metal_mapping,
                       bool prefetch_cpu) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) ds4_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
    if (st.st_size < 32) ds4_die("model file is too small to be GGUF");

    /*
     * Metal wraps slices of this mapping as no-copy MTLBuffers, so the Metal
     * path keeps the file-backed shared mapping. The CPU path only reads the
     * weights through normal pointers and should not inherit Metal's VM policy:
     * use a private read-only mapping there.
     *
     * This is deliberately defensive against an OS-level Darwin VM bug observed
     * while the CPU backend streams the very large GGUF through a shared mmap:
     * the kernel can panic in VM map-count accounting instead of returning a
     * normal user-space failure. Keeping CPU inference off the shared mapping
     * avoids that VM accounting path while preserving normal file-backed reads.
     */
    const int mmap_flags = metal_mapping ? MAP_SHARED : MAP_PRIVATE;
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, fd, 0);
    if (map == MAP_FAILED) ds4_die_errno("cannot mmap model", path);

    m->fd = fd;
    m->map = map;
    m->size = (uint64_t)st.st_size;

    ds4_cursor c = cursor_at(m, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) ds4_die(c.error);
    if (magic != DS4_GGUF_MAGIC) ds4_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) ds4_die(c.error);

    if (m->version != 3) ds4_die("only GGUF v3 is supported");

    parse_metadata(m, &c);
    parse_tensors(m, &c);

    if (!metal_mapping && prefetch_cpu) model_prefetch_cpu_mapping(m);
}

static void print_size(uint64_t bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    printf("%.2f GiB", (double)bytes / gib);
}

static void model_summary(const ds4_model *m) {
    ds4_str name = {0};
    ds4_str arch = {0};
    uint32_t layers = 0;
    uint64_t ctx_train = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_swa = 0;
    uint32_t indexer_heads = 0;
    uint32_t indexer_head_dim = 0;
    uint32_t indexer_top_k = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    uint64_t tensor_bytes = 0;
    uint64_t params = 0;

    model_get_string(m, "general.name", &name);
    model_get_string(m, "general.architecture", &arch);
    model_get_u32(m, "deepseek4.block_count", &layers);
    model_get_u64(m, "deepseek4.context_length", &ctx_train);
    model_get_u32(m, "deepseek4.attention.head_count", &n_head);
    model_get_u32(m, "deepseek4.attention.head_count_kv", &n_head_kv);
    model_get_u32(m, "deepseek4.attention.key_length", &head_dim);
    model_get_u32(m, "deepseek4.attention.sliding_window", &n_swa);
    model_get_u32(m, "deepseek4.attention.indexer.head_count", &indexer_heads);
    model_get_u32(m, "deepseek4.attention.indexer.key_length", &indexer_head_dim);
    model_get_u32(m, "deepseek4.attention.indexer.top_k", &indexer_top_k);
    model_get_u32(m, "deepseek4.expert_count", &n_expert);
    model_get_u32(m, "deepseek4.expert_used_count", &n_expert_used);
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        tensor_bytes += m->tensors[i].bytes;
        params += m->tensors[i].elements;
    }

    printf("model: %.*s\n", (int)name.len, name.ptr);
    printf("arch:  %.*s\n", (int)arch.len, arch.ptr);
    printf("gguf:  v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
        m->version, m->n_kv, m->n_tensors);
    if (layers) printf("layers: %u\n", layers);
    if (ctx_train) printf("train context: %" PRIu64 "\n", ctx_train);
    if (n_head || n_head_kv || head_dim || n_swa) {
        printf("attention: heads=%u kv_heads=%u head_dim=%u swa=%u\n",
               n_head, n_head_kv, head_dim, n_swa);
    }
    if (indexer_heads || indexer_head_dim || indexer_top_k) {
        printf("indexer: heads=%u head_dim=%u top_k=%u\n",
               indexer_heads, indexer_head_dim, indexer_top_k);
    }
    if (n_expert || n_expert_used || n_expert_groups || n_group_used) {
        printf("experts: count=%u used=%u groups=%u groups_used=%u\n",
               n_expert, n_expert_used, n_expert_groups, n_group_used);
    }
    printf("file size: ");
    print_size(m->size);
    printf("\n");
    printf("tensor bytes described by GGUF: ");
    print_size(tensor_bytes);
    printf("\n");
    printf("logical parameters: %.2f B\n", (double)params / 1000000000.0);

    printf("tensor types:\n");
    for (uint32_t type = 0; type < sizeof(gguf_types)/sizeof(gguf_types[0]); type++) {
        uint64_t count = 0;
        uint64_t bytes = 0;
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            if (m->tensors[i].type == type) {
                count++;
                bytes += m->tensors[i].bytes;
            }
        }
        if (count != 0) {
            printf("  %-8s %5" PRIu64 " tensors, ", tensor_type_name(type), count);
            print_size(bytes);
            printf("\n");
        }
    }

}

static ds4_tensor *model_find_tensor(const ds4_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len &&
            memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

#ifndef DS4_NO_GPU
#ifndef __APPLE__
typedef struct {
    uint64_t off;
    uint64_t end;
} accelerator_tensor_span;

static int accelerator_tensor_span_cmp(const void *a, const void *b) {
    const accelerator_tensor_span *sa = a;
    const accelerator_tensor_span *sb = b;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}

static uint64_t accelerator_cuda_preload_span_bytes(void) {
    uint64_t mb = 1024;
    const char *env = getenv("DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 64) mb = 64;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}

static bool accelerator_cache_model_tensor_spans(const ds4_model *m, uint64_t *cached_out) {
    accelerator_tensor_span *spans = xmalloc((size_t)m->n_tensors * sizeof(spans[0]));
    uint64_t nspan = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        if (t->bytes == 0) continue;
        if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) {
            free(spans);
            return false;
        }
        spans[nspan++] = (accelerator_tensor_span){
            .off = t->abs_offset,
            .end = t->abs_offset + t->bytes,
        };
    }
    qsort(spans, (size_t)nspan, sizeof(spans[0]), accelerator_tensor_span_cmp);

    const uint64_t max_span = accelerator_cuda_preload_span_bytes();
    uint64_t cached = 0;
    uint64_t merged = 0;
    for (uint64_t i = 0; i < nspan;) {
        uint64_t off = spans[i].off;
        uint64_t end = spans[i].end;
        i++;
        while (i < nspan && spans[i].off <= end + 65536u && spans[i].end - off <= max_span) {
            if (spans[i].end > end) end = spans[i].end;
            i++;
        }
        while (off < end) {
            uint64_t chunk_end = end;
            if (chunk_end - off > max_span) chunk_end = off + max_span;
            char label[96];
            snprintf(label, sizeof(label), "tensor-span:%" PRIu64, merged);
            if (ds4_gpu_cache_model_range(m->map, m->size, off, chunk_end - off, label) == 0) {
                fprintf(stderr,
                        "ds4: accelerator failed to cache model tensor span %" PRIu64
                        " at offset %" PRIu64 "\n",
                        merged, off);
                free(spans);
                return false;
            }
            cached += chunk_end - off;
            merged++;
            off = chunk_end;
        }
    }
    free(spans);
    if (cached_out) *cached_out = cached;
    return true;
}

static bool accelerator_cache_model_tensors(ds4_backend backend, const ds4_model *m) {
    if (backend != DS4_BACKEND_CUDA) return true;
    if (!m || !m->map || m->size == 0) return false;
    if (getenv("DS4_CUDA_DIRECT_MODEL") != NULL) {
        return true;
    }

    const double t0 = now_sec();
    uint64_t cached = 0;
    if (!accelerator_cache_model_tensor_spans(m, &cached)) return false;
    if (getenv("DS4_CUDA_Q8_F16_PRELOAD") != NULL ||
        getenv("DS4_CUDA_Q8_F32_PRELOAD") != NULL) {
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            const ds4_tensor *t = &m->tensors[i];
            if (t->bytes == 0) continue;
            if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) return false;
            char label[128];
            snprintf(label, sizeof(label), "tensor:%.*s", (int)t->name.len, t->name.ptr);
            if (t->type == DS4_TENSOR_Q8_0 && t->ndim == 2 &&
                ds4_gpu_cache_q8_f16_range(m->map, m->size, t->abs_offset, t->bytes, t->dim[0], t->dim[1], label) == 0) {
                fprintf(stderr, "ds4: accelerator failed to cache dequantized Q8 tensor %.*s\n",
                        (int)t->name.len, t->name.ptr);
                return false;
            }
        }
    }
    if (cached != 0) {
        const double t1 = now_sec();
        if (ds4_log_is_tty(stderr)) fputc('\n', stderr);
        fprintf(stderr,
                "ds4: CUDA startup model cache prepared %.2f GiB of tensor spans in %.3fs\n",
                (double)cached / 1073741824.0,
                t1 - t0);
    }
    return true;
}
#else
static bool accelerator_cache_model_tensors(ds4_backend backend, const ds4_model *m) {
    (void)backend;
    (void)m;
    return true;
}
#endif
#endif

/* Return the in-place tensor payload inside the mapped GGUF. */
static const void *tensor_data(const ds4_model *m, const ds4_tensor *t) {
    return m->map + t->abs_offset;
}

/* Optional startup pass that touches tensor pages before timing generation. */
static void model_warm_weights(const ds4_model *m) {
    const uint64_t start = m->tensor_data_pos;
    const uint64_t end = m->size;
    if (start >= end) return;

    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint8_t *p = m->map;
    volatile uint64_t checksum = 0;
    const double t0 = now_sec();

    fprintf(stderr, "ds4: warming mapped tensor pages: %.2f GiB\n",
            (double)(end - start) / (1024.0 * 1024.0 * 1024.0));

#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((void *)(p + start), (size_t)(end - start), POSIX_MADV_WILLNEED);
#endif

    for (uint64_t off = start; off < end; off += page) {
        checksum += p[off];
    }
    checksum += p[end - 1];

    const double t1 = now_sec();
    fprintf(stderr, "ds4: warmed tensor pages in %.3fs (checksum=%llu)\n",
            t1 - t0, (unsigned long long)checksum);
}

/* =========================================================================
 * Scalar Conversion and Quantized Tensor Kernels.
 * =========================================================================
 *
 * These functions are the CPU reference math used by the C backend and by
 * Metal diagnostics.  They implement only the tensor formats present in the
 * DeepSeek V4 Flash GGUF: F16, F32, Q8_0, Q2_K, IQ2_XXS, and Q8_K activation
 * blocks used for expert dot products.
 */

static inline float f16_to_f32(uint16_t h) {
#if defined(__ARM_NEON)
    const float16x4_t hv = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(hv), 0);
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

static inline uint16_t f32_to_f16(float f) {
#if defined(__ARM_NEON)
    const float32x4_t fv = vdupq_n_f32(f);
    const float16x4_t hv = vcvt_f16_f32(fv);
    return vget_lane_u16(vreinterpret_u16_f16(hv), 0);
#else
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
#endif
}

static void f16_round_inplace_cpu(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) x[i] = f16_to_f32(f32_to_f16(x[i]));
}

static float dsv4_e4m3fn_value_cpu(int i) {
    static const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };

    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

static float dsv4_e4m3fn_dequant_cpu(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_cpu(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - dsv4_e4m3fn_value_cpu(best));
        const float next_diff = fabsf(ax - dsv4_e4m3fn_value_cpu(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }

    return sign * dsv4_e4m3fn_value_cpu(best);
}

/* DeepSeek V4 stores the non-RoPE part of compressed KV through an E4M3-style
 * round trip.  Keeping this in the CPU reference makes cache values comparable
 * to the Metal graph's compressed-cache behavior. */
static void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 64; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }

        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 64; i++) {
            float v = x[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x[off + i] = dsv4_e4m3fn_dequant_cpu(v) * scale;
        }
    }
}

/* Quantize a float activation into Q8_K blocks so GGUF Q2_K/IQ2_XXS expert
 * kernels can reuse the same activation for many expert rows. */
static void ds4_quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k) {
    if (k % QK_K != 0) ds4_die("Q8_K quantization length is not QK_K aligned");
    const int64_t nb = k / QK_K;

    for (int64_t b = 0; b < nb; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            const float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }

        if (amax == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            x += QK_K;
            continue;
        }

        const float iscale = -127.0f / max;
        for (int j = 0; j < QK_K; j++) {
            int v = (int)lrintf(iscale * x[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
        x += QK_K;
    }
}

static void ds4_vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m3 = vdupq_n_u8(0x03);
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * f16_to_f32(x[i].d);
        const float dmin = -y[i].d * f16_to_f32(x[i].dmin);

        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
        uint8_t scale_lanes[16];
        vst1q_u8(scale_lanes, scales);

        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
        const int16x8x2_t mins16 = {{
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins))),
        }};
        const int32x4_t s0 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16.val[0]), vget_low_s16(q8sums.val[0])),
            vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16.val[1]), vget_low_s16(q8sums.val[1])),
            vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
        sum += dmin * (float)vaddvq_s32(vaddq_s32(s0, s1));

        int isum = 0;
        int is = 0;
        for (int j = 0; j < QK_K / 128; j++) {
            const uint8x16x2_t q2bits = vld1q_u8_x2(q2);
            q2 += 32;

#define DS4_Q2_DOT_NOSHIFT(scale_index) do {                                           \
                const int8x16x2_t q8bytes = vld1q_s8_x2(q8);                           \
                q8 += 32;                                                              \
                const int8x16_t q2lo = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));\
                const int8x16_t q2hi = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));\
                isum += vaddvq_s32(vdotq_s32(zero, q2lo, q8bytes.val[0])) *            \
                        scale_lanes[is + (scale_index)];                               \
                isum += vaddvq_s32(vdotq_s32(zero, q2hi, q8bytes.val[1])) *            \
                        scale_lanes[is + 1 + (scale_index)];                           \
            } while (0)

#define DS4_Q2_DOT_SHIFT(shift, scale_index) do {                                      \
                const int8x16x2_t q8bytes = vld1q_s8_x2(q8);                           \
                q8 += 32;                                                              \
                const int8x16_t q2lo = vreinterpretq_s8_u8(                            \
                    vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3));                 \
                const int8x16_t q2hi = vreinterpretq_s8_u8(                            \
                    vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3));                 \
                isum += vaddvq_s32(vdotq_s32(zero, q2lo, q8bytes.val[0])) *            \
                        scale_lanes[is + (scale_index)];                               \
                isum += vaddvq_s32(vdotq_s32(zero, q2hi, q8bytes.val[1])) *            \
                        scale_lanes[is + 1 + (scale_index)];                           \
            } while (0)

            DS4_Q2_DOT_NOSHIFT(0);
            DS4_Q2_DOT_SHIFT(2, 2);
            DS4_Q2_DOT_SHIFT(4, 4);
            DS4_Q2_DOT_SHIFT(6, 6);
            is += 8;

#undef DS4_Q2_DOT_NOSHIFT
#undef DS4_Q2_DOT_SHIFT
        }

        sum += d * (float)isum;
    }

    *s = sum;
#else
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; j++) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * f16_to_f32(x[i].d);
        const float dmin = y[i].d * f16_to_f32(x[i].dmin);

        int isum = 0;
        int is = 0;
        for (int k = 0; k < QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                int isuml = dot_q2_16(q2, q8, shift);
                isum += d * isuml;

                d = sc[is++] & 0x0f;
                isuml = dot_q2_16(q2 + 16, q8 + 16, shift);
                isum += d * isuml;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    *s = sumf;
#endif
}

static DS4_MAYBE_UNUSED void ds4_vec_dot_iq2_xxs_q8_K(int n, float *s, const block_iq2_xxs *x, const block_q8_K *y) {
    const int nb = n / QK_K;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        float sumf1 = 0.0f;
        float sumf2 = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            uint32_t aux32[4];
            memcpy(aux32, q2, sizeof(aux32));
            q2 += 8;
            const uint8_t *aux8 = (const uint8_t *)aux32;

            int8x16_t q2u0 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[0])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[1])));
            int8x16_t q2u1 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[2])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[3])));
            int8x16_t q2u2 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[8])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[9])));
            int8x16_t q2u3 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[10])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[11])));

            const int8x16_t q2s0 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[1] >>  0) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[1] >>  7) & 127]));
            const int8x16_t q2s1 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[1] >> 14) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[1] >> 21) & 127]));
            const int8x16_t q2s2 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[3] >>  0) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[3] >>  7) & 127]));
            const int8x16_t q2s3 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[3] >> 14) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[3] >> 21) & 127]));

            q2u0 = vmulq_s8(q2u0, q2s0);
            q2u1 = vmulq_s8(q2u1, q2s1);
            q2u2 = vmulq_s8(q2u2, q2s2);
            q2u3 = vmulq_s8(q2u3, q2s3);

            const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u0, q8b.val[0]), q2u1, q8b.val[1]);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u2, q8b.val[2]), q2u3, q8b.val[3]);

            sumf1 += (float)vaddvq_s32(p1) * (0.5f + (float)(aux32[1] >> 28));
            sumf2 += (float)vaddvq_s32(p2) * (0.5f + (float)(aux32[3] >> 28));
        }

        sumf += d * (sumf1 + sumf2);
    }

    *s = 0.25f * sumf;
#else
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;

            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;
                sumi += dot_iq2_pair_16(iq2xxs_signed_grid[aux8[l]][sign_idx0],
                                        iq2xxs_signed_grid[aux8[l + 1]][sign_idx1],
                                        q8);
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    *s = 0.125f * sumf;
#endif
}

static void ds4_vec_dot_iq2_xxs_pair_q8_K(
        int n,
        float *s0,
        float *s1,
        const block_iq2_xxs *x0,
        const block_iq2_xxs *x1,
        const block_q8_K *y) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int nb = n / QK_K;
    float total0 = 0.0f;
    float total1 = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d0 = f16_to_f32(x0[i].d) * y[i].d;
        const float d1 = f16_to_f32(x1[i].d) * y[i].d;
        const uint16_t *q20 = x0[i].qs;
        const uint16_t *q21 = x1[i].qs;
        const int8_t *q8 = y[i].qs;
        float sum01 = 0.0f;
        float sum02 = 0.0f;
        float sum11 = 0.0f;
        float sum12 = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            uint32_t aux0[4];
            uint32_t aux1[4];
            memcpy(aux0, q20, sizeof(aux0));
            memcpy(aux1, q21, sizeof(aux1));
            q20 += 8;
            q21 += 8;
            const uint8_t *a0 = (const uint8_t *)aux0;
            const uint8_t *a1 = (const uint8_t *)aux1;

#define DS4_IQ2_PAIR_DOT(aux, aux8, accum_a, accum_b) do {                                              \
                int8x16_t u0 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[0])),          \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[1])));          \
                int8x16_t u1 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[2])),          \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[3])));          \
                int8x16_t u2 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[8])),          \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[9])));          \
                int8x16_t u3 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[10])),         \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[11])));         \
                const int8x16_t sgn0 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[1] >>  0) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[1] >>  7) & 127]));      \
                const int8x16_t sgn1 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[1] >> 14) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[1] >> 21) & 127]));      \
                const int8x16_t sgn2 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[3] >>  0) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[3] >>  7) & 127]));      \
                const int8x16_t sgn3 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[3] >> 14) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[3] >> 21) & 127]));      \
                u0 = vmulq_s8(u0, sgn0);                                                               \
                u1 = vmulq_s8(u1, sgn1);                                                               \
                u2 = vmulq_s8(u2, sgn2);                                                               \
                u3 = vmulq_s8(u3, sgn3);                                                               \
                const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), u0, q8b.val[0]), u1, q8b.val[1]); \
                const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), u2, q8b.val[2]), u3, q8b.val[3]); \
                (accum_a) += (float)vaddvq_s32(p1) * (0.5f + (float)((aux)[1] >> 28));                  \
                (accum_b) += (float)vaddvq_s32(p2) * (0.5f + (float)((aux)[3] >> 28));                  \
            } while (0)

            DS4_IQ2_PAIR_DOT(aux0, a0, sum01, sum02);
            DS4_IQ2_PAIR_DOT(aux1, a1, sum11, sum12);

#undef DS4_IQ2_PAIR_DOT
        }

        total0 += d0 * (sum01 + sum02);
        total1 += d1 * (sum11 + sum12);
    }

    *s0 = 0.25f * total0;
    *s1 = 0.25f * total1;
#else
    ds4_vec_dot_iq2_xxs_q8_K(n, s0, x0, y);
    ds4_vec_dot_iq2_xxs_q8_K(n, s1, x1, y);
#endif
}

typedef struct {
    ds4_tensor *hc_attn_fn;
    ds4_tensor *hc_attn_scale;
    ds4_tensor *hc_attn_base;
    ds4_tensor *attn_norm;
    ds4_tensor *attn_q_a;
    ds4_tensor *attn_q_a_norm;
    ds4_tensor *attn_q_b;
    ds4_tensor *attn_kv;
    ds4_tensor *attn_kv_a_norm;
    ds4_tensor *attn_sinks;
    ds4_tensor *attn_output_a;
    ds4_tensor *attn_output_b;
    ds4_tensor *attn_compressor_ape;
    ds4_tensor *attn_compressor_kv;
    ds4_tensor *attn_compressor_gate;
    ds4_tensor *attn_compressor_norm;
    ds4_tensor *indexer_attn_q_b;
    ds4_tensor *indexer_proj;
    ds4_tensor *indexer_compressor_ape;
    ds4_tensor *indexer_compressor_kv;
    ds4_tensor *indexer_compressor_gate;
    ds4_tensor *indexer_compressor_norm;
    ds4_tensor *hc_ffn_fn;
    ds4_tensor *hc_ffn_scale;
    ds4_tensor *hc_ffn_base;
    ds4_tensor *ffn_norm;
    ds4_tensor *ffn_gate_tid2eid;
    ds4_tensor *ffn_gate_inp;
    ds4_tensor *ffn_exp_probs_b;
    ds4_tensor *ffn_gate_exps;
    ds4_tensor *ffn_up_exps;
    ds4_tensor *ffn_down_exps;
    ds4_tensor *ffn_gate_shexp;
    ds4_tensor *ffn_up_shexp;
    ds4_tensor *ffn_down_shexp;
} ds4_layer_weights;

typedef struct {
    ds4_tensor *token_embd;
    ds4_tensor *output_hc_base;
    ds4_tensor *output_hc_fn;
    ds4_tensor *output_hc_scale;
    ds4_tensor *output_norm;
    ds4_tensor *output;
    ds4_layer_weights layer[DS4_N_LAYER];
} ds4_weights;

typedef struct {
    ds4_tensor *e_proj;
    ds4_tensor *h_proj;
    ds4_tensor *enorm;
    ds4_tensor *hnorm;
    ds4_tensor *norm;
    ds4_tensor *hc_head_base;
    ds4_tensor *hc_head_fn;
    ds4_tensor *hc_head_scale;
    ds4_layer_weights block;
} ds4_mtp_weights;

/* =========================================================================
 * Fixed Weight Binding and Model Validation.
 * =========================================================================
 *
 * The GGUF tensor directory is converted into a DS4-specific pointer table.
 * After this section, the rest of the program addresses tensors by semantic
 * fields such as layer->attn_q_a or layer->ffn_gate_exps rather than by string
 * lookup.  Shape validation is intentionally strict.
 */

static uint32_t required_u32(const ds4_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static uint64_t required_u64(const ds4_model *m, const char *key) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) {
        uint64_t v = 0;
        if (!cursor_u64(&c, &v)) ds4_die(c.error);
        return v;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) ds4_die(c.error);
        return v;
    }

    fprintf(stderr, "ds4: metadata key has a non-integer type: %s\n", key);
    exit(1);
}

static float required_f32(const ds4_model *m, const char *key) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) {
        float v = 0.0f;
        if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
        return v;
    }
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0.0;
        if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
        return (float)v;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) ds4_die(c.error);
        return (float)v;
    }
    if (kv->type == GGUF_VALUE_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
        return (float)v;
    }

    fprintf(stderr, "ds4: metadata key has a non-float type %u: %s\n", kv->type, key);
    exit(1);
}

static bool required_bool(const ds4_model *m, const char *key) {
    bool v = false;
    if (!model_get_bool(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static ds4_tensor *required_tensor(const ds4_model *m, const char *name) {
    ds4_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "ds4: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

static ds4_tensor *tensor_by_namef(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static ds4_tensor *required_tensorf(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return required_tensor(m, name);
}

static void tensor_expect_layout(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type),
                tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

static void tensor_expect_optional(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (t) tensor_expect_layout(t, type, ndim, d0, d1, d2);
}

static void tensor_expect_plain_layout(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (t->type != DS4_TENSOR_F16 && t->type != DS4_TENSOR_F32) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected F16 or F32\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static bool tensor_is_routed_expert_type(uint32_t type) {
    return type == DS4_TENSOR_IQ2_XXS ||
           type == DS4_TENSOR_Q2_K ||
           type == DS4_TENSOR_Q4_K;
}

static DS4_MAYBE_UNUSED uint64_t routed_expert_block_bytes(uint32_t type) {
    switch (type) {
    case DS4_TENSOR_IQ2_XXS: return sizeof(block_iq2_xxs);
    case DS4_TENSOR_Q2_K:    return sizeof(block_q2_K);
    case DS4_TENSOR_Q4_K:    return sizeof(block_q4_K);
    default:                 ds4_die("unsupported routed expert tensor type");
    }
    return 0;
}

static DS4_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const ds4_tensor *t) {
    if ((t->dim[0] % QK_K) != 0) ds4_die("routed expert row is not QK_K aligned");
    return (t->dim[0] / QK_K) * routed_expert_block_bytes(t->type);
}

static void tensor_expect_routed_expert(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!tensor_is_routed_expert_type(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %u (%s), expected a routed expert quant type\n",
                (int)t->name.len,
                t->name.ptr,
                t->type,
                tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

/* Verify every tensor type and dimension used by the specialized pipeline.
 * After this succeeds, inference code can rely on fixed DS4 constants. */
static void weights_validate_layout(const ds4_weights *w) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;

    tensor_expect_layout(w->token_embd,      DS4_TENSOR_F16,  2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    tensor_expect_layout(w->output_hc_base,  DS4_TENSOR_F32,  1, DS4_N_HC, 0, 0);
    tensor_expect_layout(w->output_hc_fn,    DS4_TENSOR_F16,  2, hc_dim, DS4_N_HC, 0);
    tensor_expect_layout(w->output_hc_scale, DS4_TENSOR_F32,  1, 1, 0, 0);
    tensor_expect_layout(w->output_norm,     DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->output,          DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);

        tensor_expect_layout(l->hc_attn_fn,     DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_attn_scale,  DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_attn_base,   DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->attn_norm,      DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->attn_q_a,       DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
        tensor_expect_layout(l->attn_q_a_norm,  DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
        tensor_expect_layout(l->attn_q_b,       DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, q_dim, 0);
        tensor_expect_layout(l->attn_kv,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
        tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32,  1, DS4_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks,     DS4_TENSOR_F32,  1, DS4_N_HEAD, 0, 0);
        tensor_expect_layout(l->attn_output_a,  DS4_TENSOR_Q8_0, 2, DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_layout(l->attn_output_b,  DS4_TENSOR_Q8_0, 2, out_low_dim, DS4_N_EMBD, 0);

        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t comp_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            tensor_expect_layout(l->attn_compressor_ape,  DS4_TENSOR_F16, 2, comp_width, ratio, 0);
            tensor_expect_layout(l->attn_compressor_kv,   DS4_TENSOR_F16, 2, DS4_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_gate, DS4_TENSOR_F16, 2, DS4_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
        }
        if (ratio == 4) {
            const uint64_t index_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_width = 2u * DS4_N_INDEXER_HEAD_DIM;
            tensor_expect_layout(l->indexer_attn_q_b,          DS4_TENSOR_F16, 2, DS4_N_LORA_Q, index_q_dim, 0);
            tensor_expect_layout(l->indexer_proj,              DS4_TENSOR_F16, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD, 0);
            tensor_expect_layout(l->indexer_compressor_ape,    DS4_TENSOR_F16, 2, index_width, ratio, 0);
            tensor_expect_layout(l->indexer_compressor_kv,     DS4_TENSOR_F16, 2, DS4_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_gate,   DS4_TENSOR_F16, 2, DS4_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_norm,   DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        }

        tensor_expect_layout(l->hc_ffn_fn,      DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_ffn_scale,   DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_ffn_base,    DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->ffn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->ffn_gate_inp,   DS4_TENSOR_F16,  2, DS4_N_EMBD, DS4_N_EXPERT, 0);
        tensor_expect_optional(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
        tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
        tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
        tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
        if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            fprintf(stderr, "ds4: routed gate/up experts use different quant types in layer %u\n", il);
            exit(1);
        }
        tensor_expect_layout(l->ffn_gate_shexp, DS4_TENSOR_Q8_0,    2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_layout(l->ffn_up_shexp,   DS4_TENSOR_Q8_0,    2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_layout(l->ffn_down_shexp, DS4_TENSOR_Q8_0,    2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
        if (il < DS4_N_HASH_LAYER) {
            tensor_expect_layout(l->ffn_gate_tid2eid, DS4_TENSOR_I32, 2, DS4_N_EXPERT_USED, DS4_N_VOCAB, 0);
        }
    }
}

static void mtp_weights_validate_layout(const ds4_mtp_weights *w) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const ds4_layer_weights *l = &w->block;

    tensor_expect_layout(w->hc_head_base,  DS4_TENSOR_F32,  1, DS4_N_HC, 0, 0);
    tensor_expect_plain_layout(w->hc_head_fn, 2, hc_dim, DS4_N_HC, 0);
    tensor_expect_layout(w->hc_head_scale, DS4_TENSOR_F32,  1, 1, 0, 0);
    tensor_expect_layout(w->e_proj,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_EMBD, 0);
    tensor_expect_layout(w->h_proj,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_EMBD, 0);
    tensor_expect_layout(w->enorm,         DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->hnorm,         DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->norm,          DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);

    tensor_expect_plain_layout(l->hc_attn_fn, 2, hc_dim, hc_mix_dim, 0);
    tensor_expect_layout(l->hc_attn_scale,  DS4_TENSOR_F32,  1, 3, 0, 0);
    tensor_expect_layout(l->hc_attn_base,   DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
    tensor_expect_layout(l->attn_norm,      DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(l->attn_q_a,       DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
    tensor_expect_layout(l->attn_q_a_norm,  DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
    tensor_expect_layout(l->attn_q_b,       DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, q_dim, 0);
    tensor_expect_layout(l->attn_kv,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
    tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32,  1, DS4_N_HEAD_DIM, 0, 0);
    tensor_expect_layout(l->attn_sinks,     DS4_TENSOR_F32,  1, DS4_N_HEAD, 0, 0);
    tensor_expect_layout(l->attn_output_a,  DS4_TENSOR_Q8_0, 2, DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
    tensor_expect_layout(l->attn_output_b,  DS4_TENSOR_Q8_0, 2, out_low_dim, DS4_N_EMBD, 0);

    tensor_expect_plain_layout(l->hc_ffn_fn, 2, hc_dim, hc_mix_dim, 0);
    tensor_expect_layout(l->hc_ffn_scale,   DS4_TENSOR_F32,  1, 3, 0, 0);
    tensor_expect_layout(l->hc_ffn_base,    DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
    tensor_expect_layout(l->ffn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_plain_layout(l->ffn_gate_inp, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
    tensor_expect_layout(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
    tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
    if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
        ds4_die("MTP routed gate/up experts use different quant types");
    }
    tensor_expect_layout(l->ffn_gate_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
    tensor_expect_layout(l->ffn_up_shexp,   DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
    tensor_expect_layout(l->ffn_down_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
}

static void validate_compress_ratio_metadata(const ds4_model *m) {
    const char *key = "deepseek4.attention.compress_ratios";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "ds4: required int32/uint32 array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.attention.compress_ratios is shorter than the layer count");
    }

    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) ds4_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if (v < 0) ds4_die("metadata array contains a negative value");
            got = (uint32_t)v;
        }

        const uint32_t expected = ds4_layer_compress_ratio(il);
        if (got != expected) {
            fprintf(stderr,
                    "ds4: unexpected DeepSeek4 compression ratio at layer %u: got %u, expected %u\n",
                    il, got, expected);
            exit(1);
        }
    }
}

static void config_expect_f32(const char *name, float got, float expected);

static void validate_swiglu_clamp_metadata(const ds4_model *m) {
    const char *key = "deepseek4.swiglu_clamp_exp";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_FLOAT32 && arr.type != GGUF_VALUE_FLOAT64)) {
        fprintf(stderr, "ds4: required float array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.swiglu_clamp_exp is shorter than the layer count");
    }

    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < DS4_N_LAYER; i++) {
        float got = 0.0f;
        if (arr.type == GGUF_VALUE_FLOAT32) {
            if (!cursor_read(&c, &got, sizeof(got))) ds4_die(c.error);
        } else {
            double v = 0.0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            got = (float)v;
        }
        config_expect_f32("swiglu_clamp_exp", got, DS4_SWIGLU_CLAMP_EXP);
    }
}

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%u for DeepSeek4 Flash, got %u\n",
            name, expected, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "ds4: expected %s=%.9g for DeepSeek4 Flash, got %.9g\n",
            name, (double)expected, (double)got);
    exit(1);
}

static void config_expect_bool(const char *name, bool got, bool expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%s for DeepSeek4 Flash, got %s\n",
            name, expected ? "true" : "false", got ? "true" : "false");
    exit(1);
}

static void config_validate_fixed_shape(uint32_t n_layer) {
    config_expect_u32("block_count",                  n_layer,                 DS4_N_LAYER);
}

/* Validate metadata values that affect semantics: attention shape, HC count,
 * expert routing, RoPE scaling, compression ratios, and SwiGLU clamp. */
static void config_validate_model(const ds4_model *m) {
    const uint32_t n_layer = required_u32(m, "deepseek4.block_count");
    const uint32_t n_embd = required_u32(m, "deepseek4.embedding_length");
    const uint32_t n_vocab = required_u32(m, "deepseek4.vocab_size");
    const uint32_t n_head = required_u32(m, "deepseek4.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "deepseek4.attention.head_count_kv");
    const uint32_t n_head_dim = required_u32(m, "deepseek4.attention.key_length");
    const uint32_t n_value_dim = required_u32(m, "deepseek4.attention.value_length");
    const uint32_t n_rot = required_u32(m, "deepseek4.rope.dimension_count");
    const uint32_t n_lora_q = required_u32(m, "deepseek4.attention.q_lora_rank");
    const uint32_t n_lora_o = required_u32(m, "deepseek4.attention.output_lora_rank");
    const uint32_t n_out_group = required_u32(m, "deepseek4.attention.output_group_count");
    const uint32_t n_expert = required_u32(m, "deepseek4.expert_count");
    const uint32_t n_expert_used = required_u32(m, "deepseek4.expert_used_count");
    const uint32_t n_ff_exp = required_u32(m, "deepseek4.expert_feed_forward_length");
    const uint32_t n_expert_shared = required_u32(m, "deepseek4.expert_shared_count");
    const uint32_t n_hash_layer = required_u32(m, "deepseek4.hash_layer_count");
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);
    config_expect_u32("embedding_length",            n_embd,         DS4_N_EMBD);
    config_expect_u32("vocab_size",                  n_vocab,        DS4_N_VOCAB);
    config_expect_u32("attention.head_count",        n_head,         DS4_N_HEAD);
    config_expect_u32("attention.key_length",        n_head_dim,     DS4_N_HEAD_DIM);
    config_expect_u32("attention.head_count_kv",     n_head_kv,      DS4_N_HEAD_KV);
    config_expect_u32("attention.value_length",      n_value_dim,    DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count",        n_rot,          DS4_N_ROT);
    config_expect_u32("attention.output_group_count", n_out_group,    DS4_N_OUT_GROUP);
    config_expect_u32("attention.q_lora_rank",       n_lora_q,        DS4_N_LORA_Q);
    config_expect_u32("attention.output_lora_rank",  n_lora_o,        DS4_N_LORA_O);
    config_expect_u32("expert_count",               n_expert,        DS4_N_EXPERT);
    config_expect_u32("expert_used_count",          n_expert_used,   DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", n_ff_exp,        DS4_N_FF_EXP);
    config_expect_u32("expert_shared_count",         n_expert_shared, DS4_N_EXPERT_SHARED);
    config_expect_u32("hash_layer_count",            n_hash_layer,    DS4_N_HASH_LAYER);
    config_expect_u32("expert_group_count",         n_expert_groups, 0);
    config_expect_u32("expert_group_used_count",    n_group_used,    0);

    const uint32_t n_swa = required_u32(m, "deepseek4.attention.sliding_window");
    config_expect_u32("attention.sliding_window",     n_swa,                   DS4_N_SWA);
    const uint32_t n_indexer_head = required_u32(m, "deepseek4.attention.indexer.head_count");
    const uint32_t n_indexer_head_dim = required_u32(m, "deepseek4.attention.indexer.key_length");
    const uint32_t n_indexer_top_k = required_u32(m, "deepseek4.attention.indexer.top_k");
    config_expect_u32("attention.indexer.head_count", n_indexer_head,     DS4_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length", n_indexer_head_dim, DS4_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k",      n_indexer_top_k,    DS4_N_INDEXER_TOP_K);
    const uint32_t n_hc = required_u32(m, "deepseek4.hyper_connection.count");
    config_expect_u32("hyper_connection.count", n_hc, DS4_N_HC);
    const uint32_t n_hc_sinkhorn_iter = required_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations");
    config_expect_u32("hyper_connection.sinkhorn_iterations", n_hc_sinkhorn_iter, DS4_N_HC_SINKHORN_ITER);

    config_validate_fixed_shape(n_layer);
    validate_compress_ratio_metadata(m);

    validate_swiglu_clamp_metadata(m);

    const uint64_t rope_orig_ctx = required_u64(m, "deepseek4.rope.scaling.original_context_length");
    if (rope_orig_ctx != DS4_ROPE_ORIG_CTX) {
        fprintf(stderr, "ds4: expected rope.scaling.original_context_length=%" PRIu64
                " for DeepSeek4 Flash, got %" PRIu64 "\n",
                (uint64_t)DS4_ROPE_ORIG_CTX, rope_orig_ctx);
        exit(1);
    }
    const float rope_freq_base = required_f32(m, "deepseek4.rope.freq_base");
    config_expect_f32("rope.freq_base", rope_freq_base, DS4_ROPE_FREQ_BASE);
    const float rope_scale_factor = required_f32(m, "deepseek4.rope.scaling.factor");
    config_expect_f32("rope.scaling.factor", rope_scale_factor, DS4_ROPE_SCALE_FACTOR);
    const float rope_yarn_beta_fast = required_f32(m, "deepseek4.rope.scaling.yarn_beta_fast");
    config_expect_f32("rope.scaling.yarn_beta_fast", rope_yarn_beta_fast, DS4_ROPE_YARN_BETA_FAST);
    const float rope_yarn_beta_slow = required_f32(m, "deepseek4.rope.scaling.yarn_beta_slow");
    config_expect_f32("rope.scaling.yarn_beta_slow", rope_yarn_beta_slow, DS4_ROPE_YARN_BETA_SLOW);
    const float compress_rope_freq_base = required_f32(m, "deepseek4.attention.compress_rope_freq_base");
    config_expect_f32("attention.compress_rope_freq_base", compress_rope_freq_base, DS4_COMPRESS_ROPE_FREQ_BASE);
    const float expert_weight_scale = required_f32(m, "deepseek4.expert_weights_scale");
    config_expect_f32("expert_weights_scale", expert_weight_scale, DS4_EXPERT_WEIGHT_SCALE);
    const float rms_eps = required_f32(m, "deepseek4.attention.layer_norm_rms_epsilon");
    config_expect_f32("attention.layer_norm_rms_epsilon", rms_eps, DS4_RMS_EPS);
    const float hc_eps = required_f32(m, "deepseek4.hyper_connection.epsilon");
    config_expect_f32("hyper_connection.epsilon", hc_eps, DS4_HC_EPS);
    const bool expert_weight_norm = required_bool(m, "deepseek4.expert_weights_norm");
    config_expect_bool("expert_weights_norm", expert_weight_norm, true);
}

/* Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers. */
static void weights_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    w->token_embd       = required_tensor(m, "token_embd.weight");
    w->output_hc_base   = required_tensor(m, "output_hc_base.weight");
    w->output_hc_fn     = required_tensor(m, "output_hc_fn.weight");
    w->output_hc_scale  = required_tensor(m, "output_hc_scale.weight");
    w->output_norm      = required_tensor(m, "output_norm.weight");
    w->output           = required_tensor(m, "output.weight");

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_weights *l = &w->layer[il];
        const uint32_t compress_ratio = ds4_layer_compress_ratio(il);

        l->hc_attn_fn      = required_tensorf(m, "blk.%u.hc_attn_fn.weight", il);
        l->hc_attn_scale   = required_tensorf(m, "blk.%u.hc_attn_scale.weight", il);
        l->hc_attn_base    = required_tensorf(m, "blk.%u.hc_attn_base.weight", il);
        l->attn_norm       = required_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->attn_q_a        = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
        l->attn_q_a_norm   = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
        l->attn_q_b        = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
        l->attn_kv         = required_tensorf(m, "blk.%u.attn_kv.weight", il);
        l->attn_kv_a_norm  = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
        l->attn_sinks      = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
        l->attn_output_a   = required_tensorf(m, "blk.%u.attn_output_a.weight", il);
        l->attn_output_b   = required_tensorf(m, "blk.%u.attn_output_b.weight", il);
        if (compress_ratio != 0) {
            l->attn_compressor_ape  = required_tensorf(m, "blk.%u.attn_compressor_ape.weight", il);
            l->attn_compressor_kv   = required_tensorf(m, "blk.%u.attn_compressor_kv.weight", il);
            l->attn_compressor_gate = required_tensorf(m, "blk.%u.attn_compressor_gate.weight", il);
            l->attn_compressor_norm = required_tensorf(m, "blk.%u.attn_compressor_norm.weight", il);
        }
        if (compress_ratio == 4) {
            l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);
            l->indexer_proj     = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
            l->indexer_compressor_ape  = required_tensorf(m, "blk.%u.indexer_compressor_ape.weight", il);
            l->indexer_compressor_kv   = required_tensorf(m, "blk.%u.indexer_compressor_kv.weight", il);
            l->indexer_compressor_gate = required_tensorf(m, "blk.%u.indexer_compressor_gate.weight", il);
            l->indexer_compressor_norm = required_tensorf(m, "blk.%u.indexer_compressor_norm.weight", il);
        }
        l->hc_ffn_fn       = required_tensorf(m, "blk.%u.hc_ffn_fn.weight", il);
        l->hc_ffn_scale    = required_tensorf(m, "blk.%u.hc_ffn_scale.weight", il);
        l->hc_ffn_base     = required_tensorf(m, "blk.%u.hc_ffn_base.weight", il);
        l->ffn_norm        = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
        l->ffn_gate_inp    = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
        l->ffn_exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b.bias", il);
        l->ffn_gate_exps   = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
        l->ffn_up_exps     = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
        l->ffn_down_exps   = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
        l->ffn_gate_shexp  = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
        l->ffn_up_shexp    = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
        l->ffn_down_shexp  = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);

        if (il < DS4_N_HASH_LAYER) {
            l->ffn_gate_tid2eid = required_tensorf(m, "blk.%u.ffn_gate_tid2eid.weight", il);
        }
    }

    weights_validate_layout(w);
}

static void mtp_weights_bind(ds4_mtp_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));

    w->hc_head_base  = required_tensor(m, "mtp.0.hc_head_base.weight");
    w->hc_head_fn    = required_tensor(m, "mtp.0.hc_head_fn.weight");
    w->hc_head_scale = required_tensor(m, "mtp.0.hc_head_scale.weight");
    w->e_proj        = required_tensor(m, "mtp.0.e_proj.weight");
    w->h_proj        = required_tensor(m, "mtp.0.h_proj.weight");
    w->enorm         = required_tensor(m, "mtp.0.enorm.weight");
    w->hnorm         = required_tensor(m, "mtp.0.hnorm.weight");
    w->norm          = required_tensor(m, "mtp.0.norm.weight");

    ds4_layer_weights *l = &w->block;
    l->hc_attn_fn      = required_tensor(m, "mtp.0.hc_attn_fn.weight");
    l->hc_attn_scale   = required_tensor(m, "mtp.0.hc_attn_scale.weight");
    l->hc_attn_base    = required_tensor(m, "mtp.0.hc_attn_base.weight");
    l->attn_norm       = required_tensor(m, "mtp.0.attn_norm.weight");
    l->attn_q_a        = required_tensor(m, "mtp.0.attn_q_a.weight");
    l->attn_q_a_norm   = required_tensor(m, "mtp.0.attn_q_a_norm.weight");
    l->attn_q_b        = required_tensor(m, "mtp.0.attn_q_b.weight");
    l->attn_kv         = required_tensor(m, "mtp.0.attn_kv.weight");
    l->attn_kv_a_norm  = required_tensor(m, "mtp.0.attn_kv_a_norm.weight");
    l->attn_sinks      = required_tensor(m, "mtp.0.attn_sinks.weight");
    l->attn_output_a   = required_tensor(m, "mtp.0.attn_output_a.weight");
    l->attn_output_b   = required_tensor(m, "mtp.0.attn_output_b.weight");
    l->hc_ffn_fn       = required_tensor(m, "mtp.0.hc_ffn_fn.weight");
    l->hc_ffn_scale    = required_tensor(m, "mtp.0.hc_ffn_scale.weight");
    l->hc_ffn_base     = required_tensor(m, "mtp.0.hc_ffn_base.weight");
    l->ffn_norm        = required_tensor(m, "mtp.0.ffn_norm.weight");
    l->ffn_gate_inp    = required_tensor(m, "mtp.0.ffn_gate_inp.weight");
    l->ffn_exp_probs_b = required_tensor(m, "mtp.0.exp_probs_b.bias");
    l->ffn_gate_exps   = required_tensor(m, "mtp.0.ffn_gate_exps.weight");
    l->ffn_up_exps     = required_tensor(m, "mtp.0.ffn_up_exps.weight");
    l->ffn_down_exps   = required_tensor(m, "mtp.0.ffn_down_exps.weight");
    l->ffn_gate_shexp  = required_tensor(m, "mtp.0.ffn_gate_shexp.weight");
    l->ffn_up_shexp    = required_tensor(m, "mtp.0.ffn_up_shexp.weight");
    l->ffn_down_shexp  = required_tensor(m, "mtp.0.ffn_down_shexp.weight");

    mtp_weights_validate_layout(w);
}

static void weights_free(ds4_weights *w) {
    memset(w, 0, sizeof(*w));
}

/* Load one token embedding row and expand it to float activations. */
static void embed_token_f16(const ds4_model *m, const ds4_weights *w, int token, float *out) {
    ds4_tensor *te = w->token_embd;
    if (token < 0 || (uint64_t)token >= te->dim[1]) {
        ds4_die("token id is outside the embedding table");
    }

    const uint16_t *base = tensor_data(m, te);
    const uint64_t stride = te->dim[0];
    const uint16_t *row = base + (uint64_t)token * stride;

    for (uint64_t i = 0; i < stride; i++) {
        out[i] = f16_to_f32(row[i]);
    }
}

/* RMSNorm without a learned scale, used by hyper-connection control vectors. */
static void rms_norm_no_weight(float *out, const float *x, uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];

    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale;
}

/* Standard DS4 RMSNorm with learned per-channel scale. */
static void rms_norm_weight(float *out, const float *x, const float *weight, uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];

    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

/* Normalize each attention head independently after Q projection. */
static void head_rms_norm_inplace(float *x, uint32_t n_head, uint32_t head_dim, float eps) {
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (uint64_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) ss += (double)head[i] * head[i];

        const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (uint32_t i = 0; i < head_dim; i++) head[i] *= scale;
    }
}

typedef struct {
    float *out;
    const uint16_t *data;
    const float *x;
    uint64_t in_dim;
} matvec_f16_ctx;

static inline float dot_f16_row(const uint16_t *row, const float *x, uint64_t n) {
#if defined(__ARM_NEON)
    uint64_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        const float16x8_t hv = vreinterpretq_f16_u16(vld1q_u16(row + i));
        const float32x4_t h0 = vcvt_f32_f16(vget_low_f16(hv));
        const float32x4_t h1 = vcvt_f32_f16(vget_high_f16(hv));
        acc0 = vfmaq_f32(acc0, h0, vld1q_f32(x + i));
        acc1 = vfmaq_f32(acc1, h1, vld1q_f32(x + i + 4));
    }

    float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) acc += f16_to_f32(row[i]) * x[i];
    return acc;
#else
    float acc = 0.0f;
    for (uint64_t i = 0; i < n; i++) acc += f16_to_f32(row[i]) * x[i];
    return acc;
#endif
}

static void matvec_f16_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_f16_ctx *ctx = vctx;

    for (uint64_t o = row0; o < row1; o++) {
        const uint16_t *row = ctx->data + o * ctx->in_dim;
        ctx->out[o] = dot_f16_row(row, ctx->x, ctx->in_dim);
    }
}

/* Dense F16 matvec for small control projections such as HC and router heads. */
static void matvec_f16(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (w->type != 1 || w->ndim != 2) ds4_die("expected a 2D F16 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    matvec_f16_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .x = x,
        .in_dim = in_dim,
    };

    const uint64_t ops = in_dim * out_dim;
    const uint64_t min_rows = ops >= 262144 ? 1 : 512;
    ds4_parallel_for_min_rows(out_dim, matvec_f16_worker, &ctx, min_rows);
}

static void matvec_f16_serial(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (w->type != 1 || w->ndim != 2) ds4_die("expected a 2D F16 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    const uint16_t *data = tensor_data(m, w);
    for (uint64_t o = 0; o < out_dim; o++) {
        out[o] = dot_f16_row(data + o * in_dim, x, in_dim);
    }
}

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t row0;
    uint64_t blocks;
} matvec_q8_0_ctx;

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *data0;
    const uint8_t *data1;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
} matvec_q8_0_pair_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
    uint64_t rank;
} matvec_q8_0_grouped_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t n_groups;
    uint64_t group_dim;
    uint64_t blocks;
    uint64_t rank;
} matmul_q8_0_grouped_batch_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
} matmul_q8_0_batch_ctx;

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *data0;
    const uint8_t *data1;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
} matmul_q8_0_pair_batch_ctx;

typedef struct {
    const float *x;
    int8_t *xq;
    float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
} quantize_q8_0_batch_ctx;

static inline int32_t dot_i8_32(const int8_t *a, const int8_t *b, uint64_t n) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (n == 32) {
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, vld1q_s8(a),      vld1q_s8(b));
        acc = vdotq_s32(acc, vld1q_s8(a + 16), vld1q_s8(b + 16));
        return vaddvq_s32(acc);
    }
#endif
    int32_t sum = 0;
    for (uint64_t i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static inline float dot_q8_0_row(
        const uint8_t *row,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t accv0 = vdupq_n_f32(0.0f);
        float32x4_t accv1 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t scale_bits0;
            uint16_t scale_bits1;
            memcpy(&scale_bits0, row + b * 34, sizeof(scale_bits0));
            memcpy(&scale_bits1, row + (b + 1) * 34, sizeof(scale_bits1));

            const int8_t *qs0 = (const int8_t *)(row + b * 34 + 2);
            const int8_t *qs1 = (const int8_t *)(row + (b + 1) * 34 + 2);
            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;

            int32x4_t dot0 = vdupq_n_s32(0);
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0),      vld1q_s8(xq0));
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0 + 16), vld1q_s8(xq0 + 16));

            int32x4_t dot1 = vdupq_n_s32(0);
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1),      vld1q_s8(xq1));
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1 + 16), vld1q_s8(xq1 + 16));

            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot0), f16_to_f32(scale_bits0) * xscale[b]);
            accv1 = vfmaq_n_f32(accv1, vcvtq_f32_s32(dot1), f16_to_f32(scale_bits1) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
            const int8_t *qs = (const int8_t *)(row + b * 34 + 2);
            const int8_t *xqb = xq + b * 32;
            int32x4_t dot = vdupq_n_s32(0);
            dot = vdotq_s32(dot, vld1q_s8(qs),      vld1q_s8(xqb));
            dot = vdotq_s32(dot, vld1q_s8(qs + 16), vld1q_s8(xqb + 16));
            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot), f16_to_f32(scale_bits) * xscale[b]);
        }

        return vaddvq_f32(vaddq_f32(accv0, accv1));
    }
#endif

    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc += f16_to_f32(scale_bits) * xscale[b] * (float)dot_i8_32(qs, xq + i0, n);
    }
    return acc;
}

static inline void dot_q8_0_row_2(
        const uint8_t *row,
        const int8_t  *xq0,
        const float   *xscale0,
        const int8_t  *xq1,
        const float   *xscale1,
        uint64_t       in_dim,
        uint64_t       blocks,
        float         *out0,
        float         *out1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t acc00 = vdupq_n_f32(0.0f);
        float32x4_t acc01 = vdupq_n_f32(0.0f);
        float32x4_t acc10 = vdupq_n_f32(0.0f);
        float32x4_t acc11 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t scale_bits0;
            uint16_t scale_bits1;
            memcpy(&scale_bits0, row + b * 34, sizeof(scale_bits0));
            memcpy(&scale_bits1, row + (b + 1) * 34, sizeof(scale_bits1));

            const int8_t *qs0 = (const int8_t *)(row + b * 34 + 2);
            const int8_t *qs1 = (const int8_t *)(row + (b + 1) * 34 + 2);

            int32x4_t d00 = vdupq_n_s32(0);
            d00 = vdotq_s32(d00, vld1q_s8(qs0),      vld1q_s8(xq0 + b * 32));
            d00 = vdotq_s32(d00, vld1q_s8(qs0 + 16), vld1q_s8(xq0 + b * 32 + 16));
            int32x4_t d01 = vdupq_n_s32(0);
            d01 = vdotq_s32(d01, vld1q_s8(qs1),      vld1q_s8(xq0 + (b + 1) * 32));
            d01 = vdotq_s32(d01, vld1q_s8(qs1 + 16), vld1q_s8(xq0 + (b + 1) * 32 + 16));

            int32x4_t d10 = vdupq_n_s32(0);
            d10 = vdotq_s32(d10, vld1q_s8(qs0),      vld1q_s8(xq1 + b * 32));
            d10 = vdotq_s32(d10, vld1q_s8(qs0 + 16), vld1q_s8(xq1 + b * 32 + 16));
            int32x4_t d11 = vdupq_n_s32(0);
            d11 = vdotq_s32(d11, vld1q_s8(qs1),      vld1q_s8(xq1 + (b + 1) * 32));
            d11 = vdotq_s32(d11, vld1q_s8(qs1 + 16), vld1q_s8(xq1 + (b + 1) * 32 + 16));

            const float s0 = f16_to_f32(scale_bits0);
            const float s1 = f16_to_f32(scale_bits1);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d00), s0 * xscale0[b]);
            acc01 = vfmaq_n_f32(acc01, vcvtq_f32_s32(d01), s1 * xscale0[b + 1]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d10), s0 * xscale1[b]);
            acc11 = vfmaq_n_f32(acc11, vcvtq_f32_s32(d11), s1 * xscale1[b + 1]);
        }

        if (b < blocks) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
            const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

            int32x4_t d0 = vdupq_n_s32(0);
            d0 = vdotq_s32(d0, vld1q_s8(qs),      vld1q_s8(xq0 + b * 32));
            d0 = vdotq_s32(d0, vld1q_s8(qs + 16), vld1q_s8(xq0 + b * 32 + 16));
            int32x4_t d1 = vdupq_n_s32(0);
            d1 = vdotq_s32(d1, vld1q_s8(qs),      vld1q_s8(xq1 + b * 32));
            d1 = vdotq_s32(d1, vld1q_s8(qs + 16), vld1q_s8(xq1 + b * 32 + 16));

            const float s0 = f16_to_f32(scale_bits);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d0), s0 * xscale0[b]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d1), s0 * xscale1[b]);
        }

        *out0 = vaddvq_f32(vaddq_f32(acc00, acc01));
        *out1 = vaddvq_f32(vaddq_f32(acc10, acc11));
        return;
    }
#endif

    *out0 = dot_q8_0_row(row, xq0, xscale0, in_dim, blocks);
    *out1 = dot_q8_0_row(row, xq1, xscale1, in_dim, blocks);
}

static inline DS4_MAYBE_UNUSED void dot_q8_0_row_pair(
        const uint8_t *row0,
        const uint8_t *row1,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks,
        float         *out0,
        float         *out1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t acc00 = vdupq_n_f32(0.0f);
        float32x4_t acc01 = vdupq_n_f32(0.0f);
        float32x4_t acc10 = vdupq_n_f32(0.0f);
        float32x4_t acc11 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t s00, s01, s10, s11;
            memcpy(&s00, row0 + b * 34, sizeof(s00));
            memcpy(&s01, row0 + (b + 1) * 34, sizeof(s01));
            memcpy(&s10, row1 + b * 34, sizeof(s10));
            memcpy(&s11, row1 + (b + 1) * 34, sizeof(s11));

            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;
            const int8x16_t xv00 = vld1q_s8(xq0);
            const int8x16_t xv01 = vld1q_s8(xq0 + 16);
            const int8x16_t xv10 = vld1q_s8(xq1);
            const int8x16_t xv11 = vld1q_s8(xq1 + 16);

            const int8_t *q00 = (const int8_t *)(row0 + b * 34 + 2);
            const int8_t *q01 = (const int8_t *)(row0 + (b + 1) * 34 + 2);
            const int8_t *q10 = (const int8_t *)(row1 + b * 34 + 2);
            const int8_t *q11 = (const int8_t *)(row1 + (b + 1) * 34 + 2);

            int32x4_t d00 = vdupq_n_s32(0);
            d00 = vdotq_s32(d00, vld1q_s8(q00),      xv00);
            d00 = vdotq_s32(d00, vld1q_s8(q00 + 16), xv01);
            int32x4_t d01 = vdupq_n_s32(0);
            d01 = vdotq_s32(d01, vld1q_s8(q01),      xv10);
            d01 = vdotq_s32(d01, vld1q_s8(q01 + 16), xv11);
            int32x4_t d10 = vdupq_n_s32(0);
            d10 = vdotq_s32(d10, vld1q_s8(q10),      xv00);
            d10 = vdotq_s32(d10, vld1q_s8(q10 + 16), xv01);
            int32x4_t d11 = vdupq_n_s32(0);
            d11 = vdotq_s32(d11, vld1q_s8(q11),      xv10);
            d11 = vdotq_s32(d11, vld1q_s8(q11 + 16), xv11);

            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d00), f16_to_f32(s00) * xscale[b]);
            acc01 = vfmaq_n_f32(acc01, vcvtq_f32_s32(d01), f16_to_f32(s01) * xscale[b + 1]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d10), f16_to_f32(s10) * xscale[b]);
            acc11 = vfmaq_n_f32(acc11, vcvtq_f32_s32(d11), f16_to_f32(s11) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t s0, s1;
            memcpy(&s0, row0 + b * 34, sizeof(s0));
            memcpy(&s1, row1 + b * 34, sizeof(s1));
            const int8_t *xqb = xq + b * 32;
            const int8x16_t xv0 = vld1q_s8(xqb);
            const int8x16_t xv1 = vld1q_s8(xqb + 16);
            const int8_t *q0 = (const int8_t *)(row0 + b * 34 + 2);
            const int8_t *q1 = (const int8_t *)(row1 + b * 34 + 2);
            int32x4_t d0 = vdupq_n_s32(0);
            d0 = vdotq_s32(d0, vld1q_s8(q0),      xv0);
            d0 = vdotq_s32(d0, vld1q_s8(q0 + 16), xv1);
            int32x4_t d1 = vdupq_n_s32(0);
            d1 = vdotq_s32(d1, vld1q_s8(q1),      xv0);
            d1 = vdotq_s32(d1, vld1q_s8(q1 + 16), xv1);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d0), f16_to_f32(s0) * xscale[b]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d1), f16_to_f32(s1) * xscale[b]);
        }

        *out0 = vaddvq_f32(vaddq_f32(acc00, acc01));
        *out1 = vaddvq_f32(vaddq_f32(acc10, acc11));
        return;
    }
#endif

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t s0_bits;
        uint16_t s1_bits;
        memcpy(&s0_bits, row0 + b * 34, sizeof(s0_bits));
        memcpy(&s1_bits, row1 + b * 34, sizeof(s1_bits));
        const int8_t *q0 = (const int8_t *)(row0 + b * 34 + 2);
        const int8_t *q1 = (const int8_t *)(row1 + b * 34 + 2);
        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc0 += f16_to_f32(s0_bits) * xscale[b] * (float)dot_i8_32(q0, xq + i0, n);
        acc1 += f16_to_f32(s1_bits) * xscale[b] * (float)dot_i8_32(q1, xq + i0, n);
    }
    *out0 = acc0;
    *out1 = acc1;
}

static void quantize_q8_0_activation(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31) / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32 && i0 + i < blocks * 32; i++) {
            xq[i0 + i] = 0;
        }
    }
}

static void quantize_q8_0_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    quantize_q8_0_batch_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        quantize_q8_0_activation(ctx->x + t * ctx->in_dim,
                                 ctx->xq + t * ctx->blocks * 32,
                                 ctx->xscale + t * ctx->blocks,
                                 ctx->in_dim);
    }
}

static void quantize_q8_0_activation_batch(
        const float *x,
        int8_t      *xq,
        float       *xscale,
        uint64_t     n_tok,
        uint64_t     in_dim) {
    quantize_q8_0_batch_ctx ctx = {
        .x = x,
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .blocks = (in_dim + 31) / 32,
    };
    ds4_parallel_for(n_tok, quantize_q8_0_batch_worker, &ctx);
}

static void matvec_q8_0_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint64_t o = ctx->row0 + r;
        const uint8_t *row = ctx->data + o * ctx->blocks * 34;
        ctx->out[r] = dot_q8_0_row(row, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks);
    }
}

static void matvec_q8_0_pair_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_pair_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row0 = ctx->data0 + r * ctx->blocks * 34;
        const uint8_t *row1 = ctx->data1 + r * ctx->blocks * 34;
        dot_q8_0_row_pair(row0, row1, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks,
                          ctx->out0 + r, ctx->out1 + r);
    }
}

static void matvec_q8_0_grouped_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_grouped_ctx *ctx = vctx;

    for (uint64_t idx = r0; idx < r1; idx++) {
        const uint64_t group = idx / ctx->rank;
        const uint64_t row_in_group = idx - group * ctx->rank;
        const uint64_t tensor_row = group * ctx->rank + row_in_group;
        const uint8_t *row = ctx->data + tensor_row * ctx->blocks * 34;
        const int8_t *xq = ctx->xq + group * ctx->blocks * 32;
        const float *xscale = ctx->xscale + group * ctx->blocks;
        ctx->out[idx] = dot_q8_0_row(row, xq, xscale, ctx->in_dim, ctx->blocks);
    }
}

static void matmul_q8_0_grouped_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_grouped_batch_ctx *ctx = vctx;

    for (uint64_t idx = r0; idx < r1; idx++) {
        const uint64_t group = idx / ctx->rank;
        const uint64_t row_in_group = idx - group * ctx->rank;
        const uint64_t tensor_row = group * ctx->rank + row_in_group;
        const uint8_t *row = ctx->data + tensor_row * ctx->blocks * 34;

        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            const uint64_t xbase0 = (t * ctx->n_groups + group) * ctx->blocks;
            const uint64_t xbase1 = ((t + 1) * ctx->n_groups + group) * ctx->blocks;
            dot_q8_0_row_2(row,
                           ctx->xq + xbase0 * 32,
                           ctx->xscale + xbase0,
                           ctx->xq + xbase1 * 32,
                           ctx->xscale + xbase1,
                           ctx->group_dim,
                           ctx->blocks,
                           ctx->out + t * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group,
                           ctx->out + (t + 1) * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group);
        }
        for (; t < ctx->n_tok; t++) {
            const uint64_t xbase = (t * ctx->n_groups + group) * ctx->blocks;
            ctx->out[t * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group] =
                dot_q8_0_row(row,
                             ctx->xq + xbase * 32,
                             ctx->xscale + xbase,
                             ctx->group_dim,
                             ctx->blocks);
        }
    }
}

static void matmul_q8_0_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_batch_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row = ctx->data + r * ctx->blocks * 34;
        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            dot_q8_0_row_2(row,
                           ctx->xq + t * ctx->blocks * 32,
                           ctx->xscale + t * ctx->blocks,
                           ctx->xq + (t + 1) * ctx->blocks * 32,
                           ctx->xscale + (t + 1) * ctx->blocks,
                           ctx->in_dim,
                           ctx->blocks,
                           ctx->out + t * ctx->out_dim + r,
                           ctx->out + (t + 1) * ctx->out_dim + r);
        }
        for (; t < ctx->n_tok; t++) {
            ctx->out[t * ctx->out_dim + r] =
                dot_q8_0_row(row,
                             ctx->xq + t * ctx->blocks * 32,
                             ctx->xscale + t * ctx->blocks,
                             ctx->in_dim,
                             ctx->blocks);
        }
    }
}

static void matmul_q8_0_pair_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_pair_batch_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row0 = ctx->data0 + r * ctx->blocks * 34;
        const uint8_t *row1 = ctx->data1 + r * ctx->blocks * 34;
        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            const int8_t *xq0 = ctx->xq + t * ctx->blocks * 32;
            const float *xscale0 = ctx->xscale + t * ctx->blocks;
            const int8_t *xq1 = ctx->xq + (t + 1) * ctx->blocks * 32;
            const float *xscale1 = ctx->xscale + (t + 1) * ctx->blocks;
            dot_q8_0_row_2(row0, xq0, xscale0, xq1, xscale1, ctx->in_dim, ctx->blocks,
                           ctx->out0 + t * ctx->out_dim + r,
                           ctx->out0 + (t + 1) * ctx->out_dim + r);
            dot_q8_0_row_2(row1, xq0, xscale0, xq1, xscale1, ctx->in_dim, ctx->blocks,
                           ctx->out1 + t * ctx->out_dim + r,
                           ctx->out1 + (t + 1) * ctx->out_dim + r);
        }
        for (; t < ctx->n_tok; t++) {
            const int8_t *xq = ctx->xq + t * ctx->blocks * 32;
            const float *xscale = ctx->xscale + t * ctx->blocks;
            dot_q8_0_row_pair(row0, row1, xq, xscale, ctx->in_dim, ctx->blocks,
                              ctx->out0 + t * ctx->out_dim + r,
                              ctx->out1 + t * ctx->out_dim + r);
        }
    }
}

/* Multiply selected Q8_0 rows by an activation that has already been quantized
 * once.  This avoids repeated activation quantization for paired projections. */
static void matvec_q8_0_rows_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          row0,
        uint64_t          n_rows) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    if (row0 > out_dim || n_rows > out_dim - row0) ds4_die("Q8_0 row range is outside tensor");
    const uint64_t ctx_blocks = (in_dim + 31) / 32;

    matvec_q8_0_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .row0 = row0,
        .blocks = ctx_blocks,
    };
    ds4_parallel_for(n_rows, matvec_q8_0_worker, &ctx);
}

static DS4_MAYBE_UNUSED void matvec_q8_0_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale) {
    matvec_q8_0_rows_prequant(out, m, w, xq, xscale, 0, w->dim[1]);
}

/* Compute two Q8_0 projections from the same input, used by gate/up and
 * compressor kv/score pairs. */
static void matvec_q8_0_pair_prequant(
        float           * out0,
        float           * out1,
        const ds4_model * m,
        const ds4_tensor * w0,
        const ds4_tensor * w1,
        const int8_t    * xq,
        const float     * xscale) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        ds4_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        ds4_die("paired Q8_0 tensors do not have the same shape");
    }

    const uint64_t in_dim = w0->dim[0];
    matvec_q8_0_pair_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .data0 = tensor_data(m, w0),
        .data1 = tensor_data(m, w1),
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .blocks = (in_dim + 31) / 32,
    };
    ds4_parallel_for(w0->dim[1], matvec_q8_0_pair_worker, &ctx);
}

static void matmul_q8_0_batch_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          n_tok) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    matmul_q8_0_batch_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .in_dim = w->dim[0],
        .out_dim = w->dim[1],
        .blocks = (w->dim[0] + 31) / 32,
    };
    ds4_parallel_for(ctx.out_dim, matmul_q8_0_batch_worker, &ctx);
}

static void matmul_q8_0_pair_batch_prequant(
        float           * out0,
        float           * out1,
        const ds4_model * m,
        const ds4_tensor * w0,
        const ds4_tensor * w1,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          n_tok) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        ds4_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        ds4_die("paired Q8_0 tensors do not have the same shape");
    }

    matmul_q8_0_pair_batch_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .data0 = tensor_data(m, w0),
        .data1 = tensor_data(m, w1),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .in_dim = w0->dim[0],
        .out_dim = w0->dim[1],
        .blocks = (w0->dim[0] + 31) / 32,
    };
    ds4_parallel_for(ctx.out_dim, matmul_q8_0_pair_batch_worker, &ctx);
}

/* Batched Q8_0 matmul for prefill: quantize all token activations, then scan
 * weight rows once per output channel. */
static void matmul_q8_0_batch(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          n_tok) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_tok * blocks * 32);
    float *xscale = xmalloc((size_t)n_tok * blocks * sizeof(xscale[0]));

    quantize_q8_0_activation_batch(x, xq, xscale, n_tok, in_dim);
    matmul_q8_0_batch_prequant(out, m, w, xq, xscale, n_tok);

    free(xscale);
    free(xq);
}

static void matmul_q8_0_pair_batch(
        float           * out0,
        float           * out1,
        const ds4_model * m,
        const ds4_tensor * w0,
        const ds4_tensor * w1,
        const float     * x,
        uint64_t          n_tok) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        ds4_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        ds4_die("paired Q8_0 tensors do not have the same shape");
    }

    const uint64_t in_dim = w0->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_tok * blocks * 32);
    float *xscale = xmalloc((size_t)n_tok * blocks * sizeof(xscale[0]));

    quantize_q8_0_activation_batch(x, xq, xscale, n_tok, in_dim);
    matmul_q8_0_pair_batch_prequant(out0, out1, m, w0, w1, xq, xscale, n_tok);

    free(xscale);
    free(xq);
}

static void matvec_q8_0_rows(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          row0,
        uint64_t          n_rows) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t ctx_blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)ctx_blocks * 32);
    float *xscale = xmalloc((size_t)ctx_blocks * sizeof(xscale[0]));

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_rows_prequant(out, m, w, xq, xscale, row0, n_rows);

    free(xscale);
    free(xq);
}

/* Single-token Q8_0 matvec, used heavily in decode. */
static void matvec_q8_0(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    matvec_q8_0_rows(out, m, w, x, 0, w->dim[1]);
}

static void matvec_any(float *out, const ds4_model *m, const ds4_tensor *w, const float *x);

/* Decode scratch owns this temporary activation quantization so generation
 * can assert that the hot path performs no malloc. */
static void cpu_decode_quantize_q8_0(
        ds4_cpu_decode_scratch * scratch,
        const float            * x,
        uint64_t                 in_dim) {
    if (in_dim > scratch->q8_cap) ds4_die("CPU decode Q8_0 scratch buffer is too small");
    quantize_q8_0_activation(x, scratch->q8_xq, scratch->q8_xscale, in_dim);
}

static void matvec_q8_0_decode_scratch(
        float                  * out,
        const ds4_model        * m,
        const ds4_tensor       * w,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    cpu_decode_quantize_q8_0(scratch, x, w->dim[0]);
    matvec_q8_0_prequant(out, m, w, scratch->q8_xq, scratch->q8_xscale);
}

static void matvec_q8_0_pair_decode_scratch(
        float                  * out0,
        float                  * out1,
        const ds4_model        * m,
        const ds4_tensor       * w0,
        const ds4_tensor       * w1,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    cpu_decode_quantize_q8_0(scratch, x, w0->dim[0]);
    matvec_q8_0_pair_prequant(out0, out1, m, w0, w1, scratch->q8_xq, scratch->q8_xscale);
}

static void matvec_any_decode_scratch(
        float                  * out,
        const ds4_model        * m,
        const ds4_tensor       * w,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    if (w->type == 8) {
        matvec_q8_0_decode_scratch(out, m, w, x, scratch);
    } else {
        matvec_any(out, m, w, x);
    }
}

static void matvec_q8_0_grouped_rows(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint32_t          n_groups,
        uint64_t          group_dim,
        uint64_t          rank) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        ds4_die("grouped Q8_0 tensor has an unexpected layout");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_groups * blocks * 32);
    float *xscale = xmalloc((size_t)n_groups * blocks * sizeof(xscale[0]));

    for (uint32_t g = 0; g < n_groups; g++) {
        quantize_q8_0_activation(x + (uint64_t)g * group_dim,
                                 xq + (uint64_t)g * blocks * 32,
                                 xscale + (uint64_t)g * blocks,
                                 group_dim);
    }

    matvec_q8_0_grouped_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .in_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    ds4_parallel_for((uint64_t)n_groups * rank, matvec_q8_0_grouped_worker, &ctx);

    free(xscale);
    free(xq);
}

static void matvec_q8_0_grouped_rows_decode_scratch(
        float                  * out,
        const ds4_model        * m,
        const ds4_tensor       * w,
        const float            * x,
        uint32_t                 n_groups,
        uint64_t                 group_dim,
        uint64_t                 rank,
        ds4_cpu_decode_scratch * scratch) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        ds4_die("grouped Q8_0 tensor has an unexpected layout");
    }
    if ((uint64_t)n_groups * group_dim > scratch->q8_cap) {
        ds4_die("CPU decode grouped Q8_0 scratch buffer is too small");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    for (uint32_t g = 0; g < n_groups; g++) {
        quantize_q8_0_activation(x + (uint64_t)g * group_dim,
                                 scratch->q8_xq + (uint64_t)g * blocks * 32,
                                 scratch->q8_xscale + (uint64_t)g * blocks,
                                 group_dim);
    }

    matvec_q8_0_grouped_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = scratch->q8_xq,
        .xscale = scratch->q8_xscale,
        .in_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    ds4_parallel_for((uint64_t)n_groups * rank, matvec_q8_0_grouped_worker, &ctx);
}

static void matmul_q8_0_grouped_batch(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          n_tok,
        uint32_t          n_groups,
        uint64_t          group_dim,
        uint64_t          rank) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        ds4_die("grouped Q8_0 tensor has an unexpected layout");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_tok * n_groups * blocks * 32);
    float *xscale = xmalloc((size_t)n_tok * n_groups * blocks * sizeof(xscale[0]));

    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint32_t g = 0; g < n_groups; g++) {
            const uint64_t xbase = (t * n_groups + g) * blocks;
            quantize_q8_0_activation(x + t * n_groups * group_dim + (uint64_t)g * group_dim,
                                     xq + xbase * 32,
                                     xscale + xbase,
                                     group_dim);
        }
    }

    matmul_q8_0_grouped_batch_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .n_groups = n_groups,
        .group_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    ds4_parallel_for((uint64_t)n_groups * rank, matmul_q8_0_grouped_batch_worker, &ctx);

    free(xscale);
    free(xq);
}

typedef struct {
    float *out;
    const float *data;
    const float *x;
    uint64_t in_dim;
} matvec_f32_ctx;

static void matvec_f32_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_f32_ctx *ctx = vctx;

    for (uint64_t o = row0; o < row1; o++) {
        double acc = 0.0;
        const float *row = ctx->data + o * ctx->in_dim;
        for (uint64_t i = 0; i < ctx->in_dim; i++) {
            acc += (double)row[i] * ctx->x[i];
        }
        ctx->out[o] = (float)acc;
    }
}

static void matvec_f32(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (w->type != 0 || w->ndim != 2) ds4_die("expected a 2D F32 tensor");

    matvec_f32_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .x = x,
        .in_dim = w->dim[0],
    };
    ds4_parallel_for(w->dim[1], matvec_f32_worker, &ctx);
}

/* Dispatch for dense F32/F16/Q8_0 tensors used by auxiliary projections. */
static void matvec_any(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    switch (w->type) {
    case 0: matvec_f32(out, m, w, x); break;
    case 1: matvec_f16(out, m, w, x); break;
    case 8: matvec_q8_0(out, m, w, x); break;
    default:
        ds4_die("unsupported tensor type for dense matvec");
    }
}

static float tensor_1d_value(const ds4_model *m, const ds4_tensor *t, uint64_t i) {
    if (i >= t->elements) ds4_die("tensor scalar index is out of bounds");
    if (t->type == 0) {
        const float *p = tensor_data(m, t);
        return p[i];
    }
    if (t->type == 1) {
        const uint16_t *p = tensor_data(m, t);
        return f16_to_f32(p[i]);
    }
    ds4_die("unsupported tensor scalar type");
    return 0.0f;
}

static float tensor_2d_value(const ds4_model *m, const ds4_tensor *t, uint64_t x, uint64_t y) {
    if (t->ndim != 2 || x >= t->dim[0] || y >= t->dim[1]) {
        ds4_die("tensor 2D index is out of bounds");
    }
    return tensor_1d_value(m, t, y * t->dim[0] + x);
}

/* Locate one expert's 2D matrix inside a 3D GGUF expert tensor. */
static const uint8_t *tensor_expert_bytes(
        const ds4_model  *m,
        const ds4_tensor *w,
        uint32_t          expert,
        uint64_t         *in_dim,
        uint64_t         *out_dim,
        uint64_t         *row_bytes) {
    if (w->ndim != 3) ds4_die("expected a 3D expert tensor");
    if (expert >= w->dim[2]) ds4_die("expert id is outside expert tensor");

    *in_dim = w->dim[0];
    *out_dim = w->dim[1];

    const gguf_type_info *info = tensor_type(w->type);
    if (!info || info->block_elems == 0) ds4_die("unsupported expert tensor type");
    const uint64_t blocks = (*in_dim + info->block_elems - 1) / info->block_elems;
    *row_bytes = blocks * info->block_bytes;

    const uint64_t expert_bytes = *out_dim * *row_bytes;
    return (const uint8_t *)tensor_data(m, w) + (uint64_t)expert * expert_bytes;
}

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *base0;
    const uint8_t *base1;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes0;
    uint64_t row_bytes1;
} matvec_iq2_xxs_pair_ctx;

static void matvec_iq2_xxs_pair_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_iq2_xxs_pair_ctx *ctx = vctx;
    for (uint64_t row = row0; row < row1; row++) {
        const block_iq2_xxs *br0 = (const block_iq2_xxs *)(ctx->base0 + row * ctx->row_bytes0);
        const block_iq2_xxs *br1 = (const block_iq2_xxs *)(ctx->base1 + row * ctx->row_bytes1);
        ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &ctx->out0[row], &ctx->out1[row], br0, br1, ctx->xq);
    }
}

/* Project one routed expert's gate and up matrices.  Both are IQ2_XXS and
 * share the same Q8_K activation. */
static void matvec_iq2_xxs_expert_pair_prequant(
        float            *out0,
        float            *out1,
        const ds4_model  *m,
        const ds4_tensor *w0,
        const ds4_tensor *w1,
        const block_q8_K *xq,
        uint32_t          expert) {
    if (w0->type != 16 || w1->type != 16) ds4_die("expected IQ2_XXS expert tensors");

    uint64_t in_dim0, out_dim0, row_bytes0;
    uint64_t in_dim1, out_dim1, row_bytes1;
    const uint8_t *base0 = tensor_expert_bytes(m, w0, expert, &in_dim0, &out_dim0, &row_bytes0);
    const uint8_t *base1 = tensor_expert_bytes(m, w1, expert, &in_dim1, &out_dim1, &row_bytes1);
    if (in_dim0 != in_dim1 || out_dim0 != out_dim1) ds4_die("paired IQ2_XXS expert tensors do not match");
    if (in_dim0 % QK_K != 0) ds4_die("IQ2_XXS expert row is not QK_K aligned");

    matvec_iq2_xxs_pair_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .base0 = base0,
        .base1 = base1,
        .xq = xq,
        .in_dim = in_dim0,
        .row_bytes0 = row_bytes0,
        .row_bytes1 = row_bytes1,
    };
    ds4_parallel_for(out_dim0, matvec_iq2_xxs_pair_worker, &ctx);
}

static float silu(float x);

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_N_EXPERT_USED];
    const uint8_t *up_base[DS4_N_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_N_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_N_EXPERT_USED];
    uint64_t up_row_bytes[DS4_N_EXPERT_USED];
    int n_expert;
} matvec_iq2_xxs_mid_ctx;

static void matvec_iq2_xxs_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_iq2_xxs_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const block_iq2_xxs *gate_row = (const block_iq2_xxs *)(ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot]);
        const block_iq2_xxs *up_row = (const block_iq2_xxs *)(ctx->up_base[slot] + row * ctx->up_row_bytes[slot]);
        ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, ctx->xq);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

/* Build all selected expert hidden vectors: IQ2_XXS gate/up, clamp, SwiGLU,
 * and router weight.  The down projection runs later on the quantized mids. */
static void matvec_iq2_xxs_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != 16 || up_w->type != 16) ds4_die("expected IQ2_XXS expert tensors");
    if (n_expert < 1 || n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_iq2_xxs_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired IQ2_XXS expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("IQ2_XXS expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if (in_dim0 % QK_K != 0) ds4_die("IQ2_XXS expert row is not QK_K aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_iq2_xxs_mid_worker, &ctx);
}

typedef struct {
    float *out;
    const uint8_t *base;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes;
} matvec_q2_k_ctx;

static void matvec_q2_k_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_ctx *ctx = vctx;
    for (uint64_t row = row0; row < row1; row++) {
        const block_q2_K *br = (const block_q2_K *)(ctx->base + row * ctx->row_bytes);
        ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &ctx->out[row], br, ctx->xq);
    }
}

/* Single expert Q2_K down projection, kept mostly for tracing and diagnostics. */
static void matvec_q2_k_expert(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const float      *x,
        uint32_t          expert) {
    if (w->type != 10) ds4_die("expected a Q2_K expert tensor");

    uint64_t in_dim, out_dim, row_bytes;
    const uint8_t *base = tensor_expert_bytes(m, w, expert, &in_dim, &out_dim, &row_bytes);
    if (in_dim % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");

    block_q8_K *xq = xmalloc((size_t)(in_dim / QK_K) * sizeof(xq[0]));
    ds4_quantize_row_q8_K(x, xq, (int64_t)in_dim);

    matvec_q2_k_ctx ctx = {
        .out = out,
        .base = base,
        .xq = xq,
        .in_dim = in_dim,
        .row_bytes = row_bytes,
    };
    ds4_parallel_for(out_dim, matvec_q2_k_worker, &ctx);

    free(xq);
}

typedef struct {
    float *out;
    const uint8_t *base[DS4_N_EXPERT_USED];
    const block_q8_K *xq[DS4_N_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_N_EXPERT_USED];
    int n_expert;
} matvec_q2_k_accum_ctx;

static void matvec_q2_k_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_accum_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            float v = 0.0f;
            const block_q2_K *br = (const block_q2_K *)(ctx->base[i] + row * ctx->row_bytes[i]);
            ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &v, br, ctx->xq[i]);
            acc += v;
        }
        ctx->out[row] = acc;
    }
}

/* Accumulate all selected experts' Q2_K down projections directly into the
 * 4096-wide MoE output. */
static void matvec_q2_k_experts_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type != 10) ds4_die("expected a Q2_K expert tensor");
    if (n_expert < 1 || n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    const uint8_t *base[DS4_N_EXPERT_USED];
    uint64_t row_bytes[DS4_N_EXPERT_USED];

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i], &in_dim, &out_dim, &row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            ds4_die("Q2_K expert tensors do not share a layout");
        }
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");

    const uint64_t n_blocks = in_dim0 / QK_K;
    matvec_q2_k_accum_ctx ctx = {
        .out = out,
        .in_dim = in_dim0,
        .n_expert = n_expert,
    };
    for (int i = 0; i < n_expert; i++) {
        ctx.base[i] = base[i];
        ctx.row_bytes[i] = row_bytes[i];
        ctx.xq[i] = xq + (uint64_t)i * n_blocks;
    }

    ds4_parallel_for(out_dim0, matvec_q2_k_accum_worker, &ctx);
}

typedef struct {
    uint32_t token;
    uint32_t slot;
} ds4_expert_pair;

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_N_EXPERT];
    const uint8_t *up_base[DS4_N_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_N_EXPERT];
    uint64_t up_row_bytes[DS4_N_EXPERT];
    uint64_t xq_blocks;
} matvec_iq2_xxs_batch_mid_ctx;

static void matvec_iq2_xxs_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_iq2_xxs_batch_mid_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];

        const block_iq2_xxs *gate_row = (const block_iq2_xxs *)(ctx->gate_base[expert] + row * ctx->gate_row_bytes[expert]);
        const block_iq2_xxs *up_row = (const block_iq2_xxs *)(ctx->up_base[expert] + row * ctx->up_row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const ds4_expert_pair pair = ctx->pairs[pair_id];
            const block_q8_K *xq = ctx->xq + (uint64_t)pair.token * ctx->xq_blocks;
            float gate = 0.0f;
            float up = 0.0f;

            ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, xq);

            if (ctx->clamp > 1.0e-6f) {
                if (gate > ctx->clamp) gate = ctx->clamp;
                if (up > ctx->clamp) up = ctx->clamp;
                if (up < -ctx->clamp) up = -ctx->clamp;
            }

            ctx->mid[(uint64_t)pair_id * ctx->out_dim + row] = silu(gate) * up * ctx->pair_weight[pair_id];
        }
    }
}

typedef struct {
    const float *mid;
    block_q8_K *midq;
    uint64_t down_in_dim;
    uint64_t down_blocks;
} quantize_mid_pairs_ctx;

static void quantize_mid_pairs_worker(void *vctx, uint64_t p0, uint64_t p1) {
    quantize_mid_pairs_ctx *ctx = vctx;
    for (uint64_t p = p0; p < p1; p++) {
        ds4_quantize_row_q8_K(ctx->mid + p * ctx->down_in_dim,
                              ctx->midq + p * ctx->down_blocks,
                              (int64_t)ctx->down_in_dim);
    }
}

typedef struct {
    float *down_pair;
    const uint8_t *base[DS4_N_EXPERT];
    const block_q8_K *midq;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_N_EXPERT];
    uint64_t midq_blocks;
} matvec_q2_k_batch_down_ctx;

static DS4_MAYBE_UNUSED void matvec_q2_k_batch_down_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q2_k_batch_down_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];
        const block_q2_K *br = (const block_q2_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
            ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim,
                                  ctx->down_pair + (uint64_t)pair_id * ctx->out_dim + row,
                                  br, xq);
        }
    }
}

typedef struct {
    float *moe;
    const uint8_t *base[DS4_N_EXPERT];
    const block_q8_K *midq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_N_EXPERT];
    uint64_t midq_blocks;
} matvec_q2_k_batch_accum_rows_ctx;

static void matvec_q2_k_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_batch_accum_rows_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            ctx->moe[(uint64_t)t * ctx->out_dim + row] = 0.0f;
        }

        for (uint32_t ai = 0; ai < ctx->n_active; ai++) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];
            const block_q2_K *br = (const block_q2_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

            for (uint32_t i = begin; i < end; i++) {
                const uint32_t pair_id = ctx->pair_ids[i];
                const ds4_expert_pair pair = ctx->pairs[pair_id];
                const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
                float v = 0.0f;

                ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &v, br, xq);
                ctx->moe[(uint64_t)pair.token * ctx->out_dim + row] += v;
            }
        }
    }
}

typedef struct {
    float *moe;
    const float *down_pair;
    uint32_t n_tok;
    uint64_t out_dim;
} sum_down_pairs_ctx;

static DS4_MAYBE_UNUSED void sum_down_pairs_worker(void *vctx, uint64_t row0, uint64_t row1) {
    sum_down_pairs_ctx *ctx = vctx;
    for (uint64_t idx = row0; idx < row1; idx++) {
        const uint32_t token = (uint32_t)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)token * ctx->out_dim;
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const uint64_t pair_id = (uint64_t)token * DS4_N_EXPERT_USED + slot;
            acc += ctx->down_pair[pair_id * ctx->out_dim + row];
        }
        ctx->moe[idx] = acc;
    }
}

/* =========================================================================
 * Hyper-Connection Transforms.
 * =========================================================================
 *
 * DeepSeek V4 Flash keeps four hyper-connection streams per token.  Before
 * attention or FFN, a learned small projection chooses how to reduce the HC
 * state into the 4096-wide sublayer input.  After the sublayer, the post and
 * combine weights expand the result back into the four-stream HC state.
 */

/* Decode the HC control projection.  The output contains pre weights, post
 * gates, and a small doubly-normalized combine matrix. */
static void hc_split_sinkhorn_one(
        float       * out,
        const float * mix,
        const float * scale,
        const float * base,
        int           n_hc,
        int           iters,
        float         eps) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    for (int i = 0; i < n_hc; i++) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[16 * 16];

    for (int dst = 0; dst < n_hc; dst++) {
        float row_max = DS4_NEG_INF;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }

        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    for (int src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];

        const float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }

    for (int iter = 1; iter < iters; iter++) {
        for (int dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];

            const float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }

        for (int src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];

            const float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }

    for (int i = 0; i < n_hc * n_hc; i++) out[2 * n_hc + i] = c[i];
}

/* Reduce the four HC streams into the plain embedding vector consumed by a
 * normal attention or FFN sublayer. */
static void hc_weighted_sum_one(
        float       * out,
        const float * x,
        const float * weights,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    for (uint32_t d = 0; d < n_embd; d++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) {
            acc += x[(uint64_t)h * n_embd + d] * weights[h];
        }
        out[d] = acc;
    }
}

/* HC pre step for one token.  It normalizes the HC state, projects the control
 * vector, runs the Sinkhorn split, and emits the sublayer input plus post data. */
static void hc_pre_from_state_one_scratch(
        const ds4_model   * model,
        const ds4_tensor  * fn,
        const ds4_tensor  * scale_tensor,
        const ds4_tensor  * base_tensor,
        const float       * residual_hc,
        float             * out,
        float             * post,
        float             * comb,
        float             * flat,
        bool                serial_fn) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * n_hc;

    float mix[24];
    float split[24];

    rms_norm_no_weight(flat, residual_hc, hc_dim, DS4_RMS_EPS);
    if (serial_fn) {
        matvec_f16_serial(mix, model, fn, flat);
    } else {
        matvec_f16(mix, model, fn, flat);
    }

    const float *scale = tensor_data(model, scale_tensor);
    const float *base = tensor_data(model, base_tensor);
    hc_split_sinkhorn_one(split, mix, scale, base, (int)n_hc, DS4_N_HC_SINKHORN_ITER, 1.0e-6f);
    hc_weighted_sum_one(out, residual_hc, split, DS4_N_EMBD, n_hc);

    memcpy(post, split + n_hc, n_hc * sizeof(post[0]));
    memcpy(comb, split + 2 * n_hc, n_hc * n_hc * sizeof(comb[0]));
}

static void hc_pre_from_state_one(
        const ds4_model   * model,
        const ds4_tensor  * fn,
        const ds4_tensor  * scale_tensor,
        const ds4_tensor  * base_tensor,
        const float       * residual_hc,
        float             * out,
        float             * post,
        float             * comb) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    float *flat = xmalloc((size_t)hc_dim * sizeof(flat[0]));

    hc_pre_from_state_one_scratch(model,
                                  fn, scale_tensor, base_tensor,
                                  residual_hc, out, post, comb,
                                  flat, false);
    free(flat);
}

static void layer_attn_pre_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * token_embd,
        float             * out,
        float             * residual_hc,
        float             * post,
        float             * comb) {
    const uint32_t n_hc = DS4_N_HC;

    for (uint32_t h = 0; h < n_hc; h++) {
        memcpy(residual_hc + (uint64_t)h * DS4_N_EMBD, token_embd, (size_t)DS4_N_EMBD * sizeof(token_embd[0]));
    }

    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          residual_hc, out, post, comb);
}

/* The input embedding starts all HC streams with the same token vector. */
static void hc_from_plain_embedding(float *out_hc, const float *x, uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t h = 0; h < n_hc; h++) {
        memcpy(out_hc + (uint64_t)h * n_embd, x, (size_t)n_embd * sizeof(x[0]));
    }
}

/* HC post step for one sublayer output.  It injects the new block output and
 * mixes the previous HC streams through the learned combine matrix. */
static void hc_post_one(
        float       * out_hc,
        const float * block_out,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = block_out[d] * post[dst];

            for (uint32_t src = 0; src < n_hc; src++) {
                /* The HC combine matrix is addressed as [dst_hc, src_hc]. */
                acc += comb[dst + src * n_hc] * residual_hc[(uint64_t)src * n_embd + d];
            }

            out_hc[(uint64_t)dst * n_embd + d] = acc;
        }
    }
}

typedef struct {
    float       *out_hc;
    const float *block_out;
    const float *residual_hc;
    const float *post;
    const float *comb;
    uint64_t     hc_dim;
    uint32_t     n_embd;
    uint32_t     n_hc;
} hc_post_batch_ctx;

static void hc_post_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    hc_post_batch_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        hc_post_one(ctx->out_hc + t * ctx->hc_dim,
                    ctx->block_out + t * ctx->n_embd,
                    ctx->residual_hc + t * ctx->hc_dim,
                    ctx->post + t * ctx->n_hc,
                    ctx->comb + t * ctx->n_hc * ctx->n_hc,
                    ctx->n_embd,
                    ctx->n_hc);
    }
}

static void hc_post_batch(
        float       * out_hc,
        const float * block_out,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_tok,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    hc_post_batch_ctx ctx = {
        .out_hc = out_hc,
        .block_out = block_out,
        .residual_hc = residual_hc,
        .post = post,
        .comb = comb,
        .hc_dim = (uint64_t)n_hc * n_embd,
        .n_embd = n_embd,
        .n_hc = n_hc,
    };
    ds4_parallel_for_min_rows(n_tok, hc_post_batch_worker, &ctx, 1);
}

typedef struct {
    float       *out_hc;
    const float *moe;
    const float *shared;
    const float *residual_hc;
    const float *post;
    const float *comb;
    uint64_t     hc_dim;
    uint32_t     n_embd;
    uint32_t     n_hc;
} hc_post_sum_batch_ctx;

static void hc_post_sum_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    hc_post_sum_batch_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        const float *moe = ctx->moe + t * ctx->n_embd;
        const float *shared = ctx->shared + t * ctx->n_embd;
        const float *residual = ctx->residual_hc + t * ctx->hc_dim;
        const float *post = ctx->post + t * ctx->n_hc;
        const float *comb = ctx->comb + t * ctx->n_hc * ctx->n_hc;
        float *out = ctx->out_hc + t * ctx->hc_dim;

        for (uint32_t dst = 0; dst < ctx->n_hc; dst++) {
            for (uint32_t d = 0; d < ctx->n_embd; d++) {
                float acc = (moe[d] + shared[d]) * post[dst];
                for (uint32_t src = 0; src < ctx->n_hc; src++) {
                    acc += comb[dst + src * ctx->n_hc] *
                        residual[(uint64_t)src * ctx->n_embd + d];
                }
                out[(uint64_t)dst * ctx->n_embd + d] = acc;
            }
        }
    }
}

static void hc_post_sum_batch(
        float       * out_hc,
        const float * moe,
        const float * shared,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_tok,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    hc_post_sum_batch_ctx ctx = {
        .out_hc = out_hc,
        .moe = moe,
        .shared = shared,
        .residual_hc = residual_hc,
        .post = post,
        .comb = comb,
        .hc_dim = (uint64_t)n_hc * n_embd,
        .n_embd = n_embd,
        .n_hc = n_hc,
    };
    ds4_parallel_for_min_rows(n_tok, hc_post_sum_batch_worker, &ctx, 1);
}

typedef struct {
    const ds4_model *model;
    const ds4_tensor *fn;
    const ds4_tensor *scale;
    const ds4_tensor *base;
    const ds4_tensor *norm_w;
    const float *inp_hc;
    float *residual_hc;
    float *cur;
    float *norm;
    float *post;
    float *comb;
    uint64_t hc_dim;
    uint32_t n_hc;
} hc_pre_norm_batch_ctx;

static void hc_pre_norm_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    hc_pre_norm_batch_ctx *ctx = vctx;
    const float *norm_w = tensor_data(ctx->model, ctx->norm_w);
    float *flat = xmalloc((size_t)ctx->hc_dim * sizeof(flat[0]));

    for (uint64_t t = t0; t < t1; t++) {
        const float *residual = ctx->inp_hc + t * ctx->hc_dim;
        if (ctx->residual_hc) {
            float *dst = ctx->residual_hc + t * ctx->hc_dim;
            memcpy(dst, residual, (size_t)ctx->hc_dim * sizeof(dst[0]));
            residual = dst;
        }

        hc_pre_from_state_one_scratch(ctx->model,
                                      ctx->fn,
                                      ctx->scale,
                                      ctx->base,
                                      residual,
                                      ctx->cur + t * DS4_N_EMBD,
                                      ctx->post + t * ctx->n_hc,
                                      ctx->comb + t * ctx->n_hc * ctx->n_hc,
                                      flat,
                                      true);
        rms_norm_weight(ctx->norm + t * DS4_N_EMBD,
                        ctx->cur + t * DS4_N_EMBD,
                        norm_w,
                        DS4_N_EMBD,
                        DS4_RMS_EPS);
    }

    free(flat);
}

/* Batched HC pre plus RMSNorm.  Prefill uses this to keep the layer-major
 * token batch in contiguous arrays. */
static void hc_pre_norm_batch(
        const ds4_model  * model,
        const ds4_tensor * fn,
        const ds4_tensor * scale,
        const ds4_tensor * base,
        const ds4_tensor * norm_w,
        const float      * inp_hc,
        float            * residual_hc,
        float            * cur,
        float            * norm,
        float            * post,
        float            * comb,
        uint32_t           n_tok) {
    hc_pre_norm_batch_ctx ctx = {
        .model = model,
        .fn = fn,
        .scale = scale,
        .base = base,
        .norm_w = norm_w,
        .inp_hc = inp_hc,
        .residual_hc = residual_hc,
        .cur = cur,
        .norm = norm,
        .post = post,
        .comb = comb,
        .hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD,
        .n_hc = DS4_N_HC,
    };
    ds4_parallel_for_min_rows(n_tok, hc_pre_norm_batch_worker, &ctx, 1);
}

static void layer_attn_norm_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x) {
    const float *attn_norm = tensor_data(model, layer->attn_norm);
    rms_norm_weight(out, x, attn_norm, DS4_N_EMBD, DS4_RMS_EPS);
}

/* =========================================================================
 * Attention Projections, RoPE, and Attention Output.
 * =========================================================================
 *
 * This block performs the attention half of a transformer layer: HC pre,
 * attention RMSNorm, Q and KV projections, layer-specific RoPE, sink-aware
 * attention over raw and compressed KV rows, and the grouped LoRA output
 * projection back to embedding width.
 */

/* Q projection is low-rank: Q8_0 into a 1024 vector, RMSNorm, then Q8_0 back
 * to 64 heads of width 512. */
static void layer_q_projection_normed_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        float             * q) {
    float *qr = xmalloc(1024 * sizeof(qr[0]));
    float *qr_norm = xmalloc(1024 * sizeof(qr_norm[0]));

    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);

    matvec_q8_0(qr, model, layer->attn_q_a, norm);
    rms_norm_weight(qr_norm, qr, q_a_norm, 1024, DS4_RMS_EPS);
    matvec_q8_0(q, model, layer->attn_q_b, qr_norm);
    head_rms_norm_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS);

    free(qr_norm);
    free(qr);
}

static void layer_q_projection_with_lora_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        float             * q,
        float             * qr_norm) {
    float *qr = xmalloc(1024 * sizeof(qr[0]));
    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);

    matvec_q8_0(qr, model, layer->attn_q_a, norm);
    rms_norm_weight(qr_norm, qr, q_a_norm, 1024, DS4_RMS_EPS);
    matvec_q8_0(q, model, layer->attn_q_b, qr_norm);
    head_rms_norm_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS);

    free(qr);
}

/* KV projection has one KV head of width 512, followed by a learned RMSNorm. */
static void layer_kv_projection_normed_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * normed,
        float             * kv) {
    float *raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(raw[0]));

    const float *kv_norm = tensor_data(model, layer->attn_kv_a_norm);

    matvec_q8_0(raw, model, layer->attn_kv, normed);
    rms_norm_weight(kv, raw, kv_norm, DS4_N_HEAD_DIM, DS4_RMS_EPS);

    free(raw);
}

static void layer_q_projection_with_lora_one_decode_scratch(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * norm,
        float                   * q,
        float                   * qr_norm,
        ds4_cpu_decode_scratch  * scratch) {
    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);

    matvec_q8_0_decode_scratch(scratch->qr, model, layer->attn_q_a, norm, scratch);
    rms_norm_weight(qr_norm, scratch->qr, q_a_norm, 1024, DS4_RMS_EPS);
    matvec_q8_0_decode_scratch(q, model, layer->attn_q_b, qr_norm, scratch);
    head_rms_norm_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS);
}

static void layer_kv_projection_normed_one_decode_scratch(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * normed,
        float                   * kv,
        ds4_cpu_decode_scratch  * scratch) {
    const float *kv_norm = tensor_data(model, layer->attn_kv_a_norm);

    matvec_q8_0_decode_scratch(scratch->kv_raw, model, layer->attn_kv, normed, scratch);
    rms_norm_weight(kv, scratch->kv_raw, kv_norm, DS4_N_HEAD_DIM, DS4_RMS_EPS);
}

static float rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static float rope_yarn_corr_dim(int n_dims, uint64_t n_ctx_orig, float n_rot, float base) {
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * (float)M_PI)) / (2.0f * logf(base));
}

static void rope_yarn_corr_dims(int n_dims, uint64_t n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    const float start = floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const float end = ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = fmaxf(0.0f, start);
    dims[1] = fminf((float)(n_dims - 1), end);
}

/* Apply DS4 RoPE only to the tail of each head.  Compressed layers use the
 * long-context frequency base and scale; inverse mode rotates attention output
 * back before the grouped output projection. */
static void rope_tail_ext_inplace(
        float    * x,
        uint32_t   n_head,
        uint32_t   head_dim,
        uint32_t   n_rot,
        uint32_t   pos,
        uint64_t   n_ctx_orig,
        float      freq_base,
        float      freq_scale,
        float      ext_factor,
        float      attn_factor,
        float      beta_fast,
        float      beta_slow,
        bool       inverse) {
    const uint32_t n_nope = head_dim - n_rot;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    float corr_dims[2] = { 0.0f, 0.0f };
    if (ext_factor != 0.0f) {
        rope_yarn_corr_dims((int)n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    }

    for (uint32_t h = 0; h < n_head; h++) {
        float *tail = x + (uint64_t)h * head_dim + n_nope;
        float theta_extrap = (float)pos;

        for (uint32_t i = 0; i < n_rot; i += 2) {
            const float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;

            if (ext_factor != 0.0f) {
                const float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }

            const float c = cosf(theta) * mscale;
            const float s = sin_sign * sinf(theta) * mscale;
            const float x0 = tail[i + 0];
            const float x1 = tail[i + 1];

            tail[i + 0] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;

            theta_extrap *= theta_scale;
        }
    }
}

/* Dense layers and compressed layers use different RoPE bases. */
static float layer_rope_freq_base(uint32_t il) {
    return ds4_layer_compress_ratio(il) != 0 && DS4_COMPRESS_ROPE_FREQ_BASE > 0.0f
        ? DS4_COMPRESS_ROPE_FREQ_BASE
        : DS4_ROPE_FREQ_BASE;
}

static float layer_rope_freq_scale(uint32_t il) {
    if (ds4_layer_compress_ratio(il) == 0 || DS4_ROPE_SCALE_FACTOR <= 0.0f) {
        return 1.0f;
    }
    return 1.0f / DS4_ROPE_SCALE_FACTOR;
}

static void rope_tail_layer_inplace(
        float            * x,
        uint32_t           n_head,
        uint32_t           head_dim,
        uint32_t           n_rot,
        uint32_t           pos,
        uint32_t           il,
        bool               inverse) {
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        /*
         * This YaRN helper applies magnitude scaling internally. DeepSeek V4
         * reference RoPE uses interpolation without that magnitude change, so
         * pass the inverse factor here and let the helper cancel itself out.
         */
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    rope_tail_ext_inplace(x, n_head, head_dim, n_rot, pos,
                          compressed ? DS4_ROPE_ORIG_CTX : 0,
                          freq_base,
                          freq_scale,
                          ext_factor,
                          attn_factor,
                          DS4_ROPE_YARN_BETA_FAST,
                          DS4_ROPE_YARN_BETA_SLOW,
                          inverse);
}

typedef struct {
    float            *x;
    uint64_t          stride;
    uint32_t          n_head;
    uint32_t          head_dim;
    uint32_t          n_rot;
    uint32_t          pos0;
    uint32_t          il;
    bool              inverse;
} rope_tail_batch_ctx;

static void rope_tail_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    rope_tail_batch_ctx *ctx = vctx;
    for (uint64_t tt = t0; tt < t1; tt++) {
        rope_tail_layer_inplace(ctx->x + tt * ctx->stride,
                                ctx->n_head,
                                ctx->head_dim,
                                ctx->n_rot,
                                ctx->pos0 + (uint32_t)tt,
                                ctx->il,
                                ctx->inverse);
    }
}

static void rope_tail_layer_batch_inplace(
        float            *x,
        uint64_t          stride,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          il,
        bool              inverse,
        uint32_t          n_tok) {
    rope_tail_batch_ctx ctx = {
        .x = x,
        .stride = stride,
        .n_head = n_head,
        .head_dim = head_dim,
        .n_rot = n_rot,
        .pos0 = pos0,
        .il = il,
        .inverse = inverse,
    };
    ds4_parallel_for_min_rows(n_tok, rope_tail_batch_worker, &ctx, 1);
}

static inline float dot_f32(const float *a, const float *b, uint32_t n) {
#if defined(__ARM_NEON)
    uint32_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
#else
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
#endif
}

static inline void axpy_f32(float *y, const float *x, float a, uint32_t n) {
#if defined(__ARM_NEON)
    uint32_t i = 0;
    const float32x4_t av = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i,     vfmaq_f32(vld1q_f32(y + i),     av, vld1q_f32(x + i)));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), av, vld1q_f32(x + i + 4)));
    }
    for (; i < n; i++) y[i] += a * x[i];
#else
    for (uint32_t i = 0; i < n; i++) y[i] += a * x[i];
#endif
}

static inline void scale_f32(float *x, float a, uint32_t n) {
#if defined(__ARM_NEON)
    uint32_t i = 0;
    const float32x4_t av = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(x + i,     vmulq_f32(vld1q_f32(x + i),     av));
        vst1q_f32(x + i + 4, vmulq_f32(vld1q_f32(x + i + 4), av));
    }
    for (; i < n; i++) x[i] *= a;
#else
    for (uint32_t i = 0; i < n; i++) x[i] *= a;
#endif
}

static float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        return e / (1.0f + e);
    }
}

/* Sink-aware attention over a set of KV rows.  The learned sink logit is part
 * of the softmax denominator but contributes no value vector. */
static void layer_attention_rows_one(
        float             * out_heads,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * q,
        const float       * kv_rows,
        uint32_t            n_kv) {
    const float *sinks = tensor_data(model, layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    float score_stack[512];
    float *score = n_kv <= 512 ? score_stack : xmalloc((size_t)n_kv * sizeof(score[0]));

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_N_HEAD_DIM;

        float max_score = sinks[h];
        for (uint32_t r = 0; r < n_kv; r++) {
            const float *kv = kv_rows + (uint64_t)r * DS4_N_HEAD_DIM;
            score[r] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[r] > max_score) max_score = score[r];
        }

        float *oh = out_heads + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        for (uint32_t r = 0; r < n_kv; r++) {
            const float weight = expf(score[r] - max_score);
            const float *kv = kv_rows + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        const float inv = 1.0f / denom;
        scale_f32(oh, inv, DS4_N_HEAD_DIM);
    }

    if (score != score_stack) free(score);
}

static void layer_attention_one(
        float             * out_heads,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * q,
        const float       * kv) {
    layer_attention_rows_one(out_heads, model, layer, q, kv, 1);
}

/* Attention output projection is grouped: each group first maps its heads to
 * a 1024-rank low vector, then all groups are projected back to 4096. */
static void layer_grouped_out_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * heads) {
    const uint32_t n_groups = 8;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = 1024;

    float *low = xcalloc((size_t)n_groups * rank, sizeof(low[0]));

    matvec_q8_0_grouped_rows(low, model, layer->attn_output_a, heads, n_groups, group_dim, rank);

    matvec_q8_0(out, model, layer->attn_output_b, low);
    free(low);
}

static void layer_grouped_out_one_decode_scratch(
        float                  * out,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * heads,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t n_groups = 8;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = 1024;

    memset(scratch->attn_low, 0, (size_t)n_groups * rank * sizeof(scratch->attn_low[0]));
    matvec_q8_0_grouped_rows_decode_scratch(scratch->attn_low, model, layer->attn_output_a,
                                            heads, n_groups, group_dim, rank, scratch);
    matvec_q8_0_decode_scratch(out, model, layer->attn_output_b, scratch->attn_low, scratch);
}

static void layer_grouped_out_batch(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * heads,
        uint32_t            n_tok) {
    const uint32_t n_groups = 8;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = 1024;

    float *low = xcalloc((size_t)n_tok * n_groups * rank, sizeof(low[0]));

    matmul_q8_0_grouped_batch(low, model, layer->attn_output_a, heads,
                              n_tok, n_groups, group_dim, rank);
    matmul_q8_0_batch(out, model, layer->attn_output_b, low, n_tok);

    free(low);
}

/* =========================================================================
 * Mixture-of-Experts FFN.
 * =========================================================================
 *
 * This is the FFN half of each layer.  It includes the shared expert, routed
 * expert selection, IQ2_XXS gate/up projections, SwiGLU, Q2_K down projection,
 * and the HC post step that returns the result to four-stream state.
 */

static float silu(float x) {
    return x * sigmoid_stable(x);
}

static float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void swiglu(float *out, const float *gate, const float *up, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        out[i] = silu(gate[i]) * up[i];
    }
}

/* The shared expert is a normal Q8_0 SwiGLU MLP that runs for every token. */
static void layer_shared_ffn_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x) {
    float *gate = xmalloc((size_t)DS4_N_FF_EXP * sizeof(gate[0]));
    float *up = xmalloc((size_t)DS4_N_FF_EXP * sizeof(up[0]));
    float *mid = xmalloc((size_t)DS4_N_FF_EXP * sizeof(mid[0]));
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)blocks * 32);
    float *xscale = xmalloc((size_t)blocks * sizeof(xscale[0]));

    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim) {
        ds4_die("shared expert gate/up tensors do not share a Q8_0 input layout");
    }

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_pair_prequant(gate, up, model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              xq, xscale);
    swiglu(mid, gate, up, DS4_N_FF_EXP);
    matvec_q8_0(out, model, layer->ffn_down_shexp, mid);

    free(xscale);
    free(xq);
    free(mid);
    free(up);
    free(gate);
}

static void layer_shared_ffn_one_decode_scratch(
        float                  * out,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim) {
        ds4_die("shared expert gate/up tensors do not share a Q8_0 input layout");
    }

    matvec_q8_0_pair_decode_scratch(scratch->shared_gate,
                                    scratch->shared_up,
                                    model,
                                    layer->ffn_gate_shexp,
                                    layer->ffn_up_shexp,
                                    x,
                                    scratch);
    swiglu(scratch->shared_mid, scratch->shared_gate, scratch->shared_up, DS4_N_FF_EXP);
    matvec_q8_0_decode_scratch(out, model, layer->ffn_down_shexp, scratch->shared_mid, scratch);
}

typedef struct {
    float *mid;
    const float *gate;
    const float *up;
    uint64_t n;
} swiglu_batch_ctx;

static void swiglu_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    swiglu_batch_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        swiglu(ctx->mid + t * ctx->n,
               ctx->gate + t * ctx->n,
               ctx->up + t * ctx->n,
               ctx->n);
    }
}

static void layer_shared_ffn_batch(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        uint32_t            n_tok) {
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t hidden = layer->ffn_gate_shexp->dim[1];

    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_down_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim ||
        layer->ffn_up_shexp->dim[1] != hidden ||
        layer->ffn_down_shexp->dim[0] != hidden) {
        ds4_die("shared expert tensors do not share the expected Q8_0 layout");
    }

    float *gate = xmalloc((size_t)n_tok * hidden * sizeof(gate[0]));
    float *up = xmalloc((size_t)n_tok * hidden * sizeof(up[0]));
    float *mid = xmalloc((size_t)n_tok * hidden * sizeof(mid[0]));

    matmul_q8_0_pair_batch(gate, up, model,
                           layer->ffn_gate_shexp,
                           layer->ffn_up_shexp,
                           x,
                           n_tok);

    swiglu_batch_ctx swiglu_ctx = {
        .mid = mid,
        .gate = gate,
        .up = up,
        .n = hidden,
    };
    ds4_parallel_for(n_tok, swiglu_batch_worker, &swiglu_ctx);

    matmul_q8_0_batch(out, model, layer->ffn_down_shexp, mid, n_tok);

    free(mid);
    free(up);
    free(gate);
}

/* Early DS4 layers use token-id hash routing instead of top-k routing. */
static void layer_hash_selected_experts(
        int                    selected[DS4_N_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        int                    token) {
    ds4_tensor *t = layer->ffn_gate_tid2eid;
    if (!t) ds4_die("hash routing table is missing for this layer");
    if (t->type != 26 || t->ndim != 2 || t->dim[0] != DS4_N_EXPERT_USED) {
        ds4_die("ffn_gate_tid2eid.weight has an unexpected layout");
    }
    if (token < 0 || (uint64_t)token >= t->dim[1]) {
        ds4_die("token id is outside the hash routing table");
    }

    const int32_t *table = tensor_data(model, t);
    const int32_t *row = table + (uint64_t)token * DS4_N_EXPERT_USED;
    for (int i = 0; i < DS4_N_EXPERT_USED; i++) selected[i] = row[i];
}

/* Router scores use sqrt(softplus(logit)); normalization happens only after
 * the six selected experts are known. */
static void layer_router_probs_one(
        float             probs[DS4_N_EXPERT],
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x) {
    float logits[DS4_N_EXPERT];

    matvec_f16(logits, model, layer->ffn_gate_inp, x);
    for (int i = 0; i < DS4_N_EXPERT; i++) {
        probs[i] = sqrtf(softplus_stable(logits[i]));
    }
}

static void layer_hash_router_weights_from_probs(
        float             weights_out[DS4_N_EXPERT_USED],
        const float       probs[DS4_N_EXPERT],
        const int          selected[DS4_N_EXPERT_USED]) {
    float sum = 0.0f;
    for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
        if (selected[i] < 0 || selected[i] >= DS4_N_EXPERT) ds4_die("hash-selected expert is outside router range");
        weights_out[i] = probs[selected[i]];
        sum += weights_out[i];
    }

    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
        weights_out[i] = weights_out[i] / sum * DS4_EXPERT_WEIGHT_SCALE;
    }
}

static void layer_hash_router_weights_one(
        float             weights_out[DS4_N_EXPERT_USED],
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        const int          selected[DS4_N_EXPERT_USED]) {
    float probs[DS4_N_EXPERT];

    layer_router_probs_one(probs, model, layer, x);
    layer_hash_router_weights_from_probs(weights_out, probs, selected);
}

static void topk_desc(const float *score, int n, int k, int *idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (idx[j] < 0 || score[i] > score[idx[j]]) {
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = i;
                break;
            }
        }
    }
}

/* Later layers choose the six experts by biased top-k, but weight them using
 * the unbiased router probabilities. */
static void layer_topk_selected_experts_from_probs(
        int                    selected[DS4_N_EXPERT_USED],
        float                  expert_weight[DS4_N_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        const float           probs[DS4_N_EXPERT]);

static void layer_topk_selected_experts(
        int                    selected[DS4_N_EXPERT_USED],
        float                  expert_weight[DS4_N_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        const float           *x) {
    float probs[DS4_N_EXPERT];

    layer_router_probs_one(probs, model, layer, x);
    layer_topk_selected_experts_from_probs(selected, expert_weight, model, layer, probs);
}

static void layer_topk_selected_experts_from_probs(
        int                    selected[DS4_N_EXPERT_USED],
        float                  expert_weight[DS4_N_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        const float           probs[DS4_N_EXPERT]) {
    float selection[DS4_N_EXPERT];

    memcpy(selection, probs, sizeof(selection));

    if (layer->ffn_exp_probs_b) {
        const float *bias = tensor_data(model, layer->ffn_exp_probs_b);
        for (int i = 0; i < DS4_N_EXPERT; i++) selection[i] += bias[i];
    }

    topk_desc(selection, DS4_N_EXPERT, DS4_N_EXPERT_USED, selected);

    float sum = 0.0f;
    for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
        expert_weight[i] = probs[selected[i]];
        sum += expert_weight[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
        expert_weight[i] = expert_weight[i] / sum * DS4_EXPERT_WEIGHT_SCALE;
    }
}

static void print_vec_stats(const char *name, const float *x, uint64_t n);

/* Single-token routed MoE.  It selects six experts, runs IQ2_XXS gate/up,
 * applies SwiGLU and router weights, then accumulates Q2_K down projections. */
static void layer_routed_moe_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        uint32_t            il,
        int                 token,
        float               clamp,
        bool                trace) {
    int selected[DS4_N_EXPERT_USED];
    float expert_weight[DS4_N_EXPERT_USED];
    float *gate = trace ? xmalloc((size_t)DS4_N_FF_EXP * sizeof(gate[0])) : NULL;
    float *up = trace ? xmalloc((size_t)DS4_N_FF_EXP * sizeof(up[0])) : NULL;
    float *mid = trace ? xmalloc((size_t)DS4_N_FF_EXP * sizeof(mid[0])) : NULL;
    float *mid_all = trace ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(mid_all[0]));
    float *down = trace ? xmalloc((size_t)DS4_N_EMBD * sizeof(down[0])) : NULL;
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    if (expert_in_dim % QK_K != 0) ds4_die("IQ2_XXS expert input is not QK_K aligned");
    if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) ds4_die("Q2_K expert input has an unexpected layout");
    block_q8_K *xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(xq[0]));
    block_q8_K *midq = trace ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(midq[0]));

    memset(out, 0, (size_t)DS4_N_EMBD * sizeof(out[0]));
    ds4_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);

    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, x, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, x);
    }

    if (!trace) {
        matvec_iq2_xxs_experts_mid_prequant(mid_all, model,
                                            layer->ffn_gate_exps,
                                            layer->ffn_up_exps,
                                            xq,
                                            selected,
                                            expert_weight,
                                            DS4_N_EXPERT_USED,
                                            clamp);
        for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
            ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                                  midq + (uint64_t)i * (down_in_dim / QK_K),
                                  (int64_t)down_in_dim);
        }
        matvec_q2_k_experts_accum_prequant(out, model, layer->ffn_down_exps, midq, selected, DS4_N_EXPERT_USED);
    } else {
        for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
            const uint32_t expert = (uint32_t)selected[i];

            matvec_iq2_xxs_expert_pair_prequant(gate, up, model,
                                                 layer->ffn_gate_exps,
                                                 layer->ffn_up_exps,
                                                 xq,
                                                 expert);
            char name[64];
            snprintf(name, sizeof(name), "blk.%u expert %u gate", il, expert);
            print_vec_stats(name, gate, DS4_N_FF_EXP);
            snprintf(name, sizeof(name), "blk.%u expert %u up", il, expert);
            print_vec_stats(name, up, DS4_N_FF_EXP);

            /*
             * DeepSeek V4 clamps routed expert gate/up values before SwiGLU and
             * applies the router weight before the down projection.
             */
            const float limit = clamp;
            for (int j = 0; j < DS4_N_FF_EXP; j++) {
                if (limit > 1.0e-6f) {
                    if (gate[j] > limit) gate[j] = limit;
                    if (up[j] > limit) up[j] = limit;
                    if (up[j] < -limit) up[j] = -limit;
                }
                mid[j] = silu(gate[j]) * up[j] * expert_weight[i];
            }

            snprintf(name, sizeof(name), "blk.%u expert %u mid", il, expert);
            print_vec_stats(name, mid, DS4_N_FF_EXP);

            matvec_q2_k_expert(down, model, layer->ffn_down_exps, mid, expert);
            snprintf(name, sizeof(name), "blk.%u expert %u down", il, expert);
            print_vec_stats(name, down, DS4_N_EMBD);
            for (int j = 0; j < DS4_N_EMBD; j++) out[j] += down[j];
        }
    }

    free(midq);
    free(xq);
    free(down);
    free(mid_all);
    free(mid);
    free(up);
    free(gate);
}

/* Decode version of routed MoE: same math as layer_routed_moe_one(), but all
 * large temporaries come from the persistent scratch arena. */
static void layer_routed_moe_one_prealloc(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        uint32_t            il,
        int                 token,
        float               clamp,
        float              * mid_all,
        block_q8_K         * xq,
        block_q8_K         * midq) {
    int selected[DS4_N_EXPERT_USED];
    float expert_weight[DS4_N_EXPERT_USED];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];

    if (expert_in_dim % QK_K != 0) ds4_die("IQ2_XXS expert input is not QK_K aligned");
    if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) ds4_die("Q2_K expert input has an unexpected layout");

    memset(out, 0, (size_t)DS4_N_EMBD * sizeof(out[0]));
    ds4_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);

    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, x, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, x);
    }

    matvec_iq2_xxs_experts_mid_prequant(mid_all, model,
                                        layer->ffn_gate_exps,
                                        layer->ffn_up_exps,
                                        xq,
                                        selected,
                                        expert_weight,
                                        DS4_N_EXPERT_USED,
                                        clamp);

    for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
        ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                              midq + (uint64_t)i * (down_in_dim / QK_K),
                              (int64_t)down_in_dim);
    }
    matvec_q2_k_experts_accum_prequant(out, model, layer->ffn_down_exps, midq, selected, DS4_N_EXPERT_USED);

    (void)il;
}

/* Prefill MoE groups token/expert pairs by expert so each active expert's
 * rows are scanned once for the whole token batch. */
static void layer_routed_moe_batch(
        float             * moe,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        float               clamp) {
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_out_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t down_out_dim = layer->ffn_down_exps->dim[1];
    if (expert_in_dim % QK_K != 0) ds4_die("IQ2_XXS expert input is not QK_K aligned");
    if (down_in_dim % QK_K != 0) ds4_die("Q2_K expert input is not QK_K aligned");
    if (expert_out_dim != down_in_dim || down_out_dim != DS4_N_EMBD) {
        ds4_die("routed expert tensor layout is unexpected");
    }

    const uint32_t total_pairs = n_tok * DS4_N_EXPERT_USED;
    uint32_t counts[DS4_N_EXPERT + 1] = {0};
    uint32_t cursor[DS4_N_EXPERT] = {0};
    uint32_t active_expert[DS4_N_EXPERT];
    uint32_t n_active = 0;

    int *selected = xmalloc((size_t)total_pairs * sizeof(selected[0]));
    float *pair_weight = xmalloc((size_t)total_pairs * sizeof(pair_weight[0]));
    ds4_expert_pair *pairs = xmalloc((size_t)total_pairs * sizeof(pairs[0]));

    const uint64_t xq_blocks = expert_in_dim / QK_K;
    block_q8_K *xq = xmalloc((size_t)n_tok * xq_blocks * sizeof(xq[0]));
    for (uint32_t t = 0; t < n_tok; t++) {
        ds4_quantize_row_q8_K(norm + (uint64_t)t * expert_in_dim,
                              xq + (uint64_t)t * xq_blocks,
                              (int64_t)expert_in_dim);

        int sel[DS4_N_EXPERT_USED];
        float weights[DS4_N_EXPERT_USED];
        if (layer->ffn_gate_tid2eid) {
            layer_hash_selected_experts(sel, model, layer, token_ids[t]);
            layer_hash_router_weights_one(weights, model, layer, norm + (uint64_t)t * expert_in_dim, sel);
        } else {
            layer_topk_selected_experts(sel, weights, model, layer, norm + (uint64_t)t * expert_in_dim);
        }

        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const uint32_t pair_id = t * DS4_N_EXPERT_USED + slot;
            selected[pair_id] = sel[slot];
            pair_weight[pair_id] = weights[slot];
            pairs[pair_id] = (ds4_expert_pair){ .token = t, .slot = slot };
            if (sel[slot] < 0 || sel[slot] >= DS4_N_EXPERT) ds4_die("selected expert is outside range");
            counts[(uint32_t)sel[slot] + 1]++;
        }
    }

    for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
        counts[e + 1] += counts[e];
        cursor[e] = counts[e];
        if (counts[e + 1] != counts[e]) active_expert[n_active++] = e;
    }

    uint32_t *pair_ids = xmalloc((size_t)total_pairs * sizeof(pair_ids[0]));
    for (uint32_t p = 0; p < total_pairs; p++) {
        const uint32_t e = (uint32_t)selected[p];
        pair_ids[cursor[e]++] = p;
    }

    float *mid = xmalloc((size_t)total_pairs * expert_out_dim * sizeof(mid[0]));

    matvec_iq2_xxs_batch_mid_ctx mid_ctx = {
        .mid = mid,
        .xq = xq,
        .pairs = pairs,
        .pair_ids = pair_ids,
        .expert_offset = counts,
        .active_expert = active_expert,
        .pair_weight = pair_weight,
        .clamp = clamp,
        .in_dim = expert_in_dim,
        .out_dim = expert_out_dim,
        .xq_blocks = xq_blocks,
    };

    for (uint32_t ai = 0; ai < n_active; ai++) {
        const uint32_t e = active_expert[ai];
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        mid_ctx.gate_base[e] = tensor_expert_bytes(model, layer->ffn_gate_exps, e,
                                                   &gate_in_dim, &gate_out_dim, &mid_ctx.gate_row_bytes[e]);
        mid_ctx.up_base[e] = tensor_expert_bytes(model, layer->ffn_up_exps, e,
                                                 &up_in_dim, &up_out_dim, &mid_ctx.up_row_bytes[e]);
        if (gate_in_dim != expert_in_dim || up_in_dim != expert_in_dim ||
            gate_out_dim != expert_out_dim || up_out_dim != expert_out_dim) {
            ds4_die("IQ2_XXS batch expert tensor layout mismatch");
        }
    }

    ds4_parallel_for((uint64_t)n_active * expert_out_dim, matvec_iq2_xxs_batch_mid_worker, &mid_ctx);

    const uint64_t midq_blocks = down_in_dim / QK_K;
    block_q8_K *midq = xmalloc((size_t)total_pairs * midq_blocks * sizeof(midq[0]));
    quantize_mid_pairs_ctx quant_ctx = {
        .mid = mid,
        .midq = midq,
        .down_in_dim = down_in_dim,
        .down_blocks = midq_blocks,
    };
    ds4_parallel_for(total_pairs, quantize_mid_pairs_worker, &quant_ctx);
    free(mid);

    matvec_q2_k_batch_accum_rows_ctx down_ctx = {
        .moe = moe,
        .midq = midq,
        .pairs = pairs,
        .pair_ids = pair_ids,
        .expert_offset = counts,
        .active_expert = active_expert,
        .n_active = n_active,
        .n_tok = n_tok,
        .in_dim = down_in_dim,
        .out_dim = down_out_dim,
        .midq_blocks = midq_blocks,
    };

    for (uint32_t ai = 0; ai < n_active; ai++) {
        const uint32_t e = active_expert[ai];
        uint64_t in_dim, out_dim;
        down_ctx.base[e] = tensor_expert_bytes(model, layer->ffn_down_exps, e,
                                               &in_dim, &out_dim, &down_ctx.row_bytes[e]);
        if (in_dim != down_in_dim || out_dim != down_out_dim) {
            ds4_die("Q2_K batch expert tensor layout mismatch");
        }
    }

    ds4_parallel_for(down_out_dim, matvec_q2_k_batch_accum_rows_worker, &down_ctx);

    free(midq);
    free(pair_ids);
    free(xq);
    free(pairs);
    free(pair_weight);
    free(selected);

    (void)il;
}

static void print_vec_stats(const char *name, const float *x, uint64_t n);

/* Full FFN sublayer for one token: HC pre, RMSNorm, routed MoE, shared expert,
 * sum, and HC post. */
static void layer_ffn_one(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * inp_hc,
        uint32_t            il,
        int                 token,
        const float       * steering_dirs,
        float               steering_scale,
        bool                trace) {
    const uint32_t n_hc = DS4_N_HC;
    const bool profile = getenv("DS4_DECODE_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc = 0.0;
    double t_norm = 0.0;
    double t_routed = 0.0;
    double t_shared = 0.0;
    double t_post = 0.0;
    float *ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));
    float *moe = xmalloc((size_t)DS4_N_EMBD * sizeof(moe[0]));
    float *shared = xmalloc((size_t)DS4_N_EMBD * sizeof(shared[0]));
    float *ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(ffn_out[0]));
    float post[4];
    float comb[16];

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          inp_hc, ffn_cur, post, comb);
    if (profile) t_hc = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_cur", il);
        print_vec_stats(name, ffn_cur, DS4_N_EMBD);
    }

    t0 = profile ? now_sec() : 0.0;
    const float *ffn_norm = tensor_data(model, layer->ffn_norm);
    rms_norm_weight(norm, ffn_cur, ffn_norm, DS4_N_EMBD, DS4_RMS_EPS);
    if (profile) t_norm = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_norm", il);
        print_vec_stats(name, norm, DS4_N_EMBD);
    }

    t0 = profile ? now_sec() : 0.0;
    layer_routed_moe_one(moe, model, layer, norm, il, token, DS4_SWIGLU_CLAMP_EXP, trace);
    if (profile) t_routed = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u routed_moe", il);
        print_vec_stats(name, moe, DS4_N_EMBD);
    }
    t0 = profile ? now_sec() : 0.0;
    layer_shared_ffn_one(shared, model, layer, norm);
    if (profile) t_shared = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u shared_ffn", il);
        print_vec_stats(name, shared, DS4_N_EMBD);
    }

    t0 = profile ? now_sec() : 0.0;
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
        ffn_out[i] = moe[i] + shared[i];
    }
    cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, 1, steering_scale);
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_out", il);
        print_vec_stats(name, ffn_out, DS4_N_EMBD);
    }

    hc_post_one(out_hc, ffn_out, inp_hc, post, comb, DS4_N_EMBD, n_hc);
    if (profile) t_post = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_post_hc", il);
        print_vec_stats(name, out_hc, (uint64_t)n_hc * DS4_N_EMBD);
    }

    if (profile) {
        fprintf(stderr,
                "ds4: decode detail layer %u ffn hc=%.3f norm=%.3f routed=%.3f shared=%.3f post=%.3f total=%.3f ms\n",
                il,
                t_hc * 1000.0,
                t_norm * 1000.0,
                t_routed * 1000.0,
                t_shared * 1000.0,
                t_post * 1000.0,
                (now_sec() - t_start) * 1000.0);
    }

    free(ffn_out);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}

/* Allocation-free decode FFN using the persistent CPU scratch buffers. */
static void layer_ffn_one_decode_scratch(
        float                  * out_hc,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * inp_hc,
        uint32_t                 il,
        int                      token,
        const float            * steering_dirs,
        float                    steering_scale,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t n_hc = DS4_N_HC;
    const bool profile = getenv("DS4_DECODE_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc = 0.0;
    double t_norm = 0.0;
    double t_routed = 0.0;
    double t_shared = 0.0;
    double t_post = 0.0;
    float post[4];
    float comb[16];

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_from_state_one_scratch(model,
                                  layer->hc_ffn_fn,
                                  layer->hc_ffn_scale,
                                  layer->hc_ffn_base,
                                  inp_hc, scratch->ffn_cur, post, comb,
                                  scratch->hc_flat,
                                  false);
    if (profile) t_hc = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    const float *ffn_norm = tensor_data(model, layer->ffn_norm);
    rms_norm_weight(scratch->ffn_norm, scratch->ffn_cur, ffn_norm, DS4_N_EMBD, DS4_RMS_EPS);
    if (profile) t_norm = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_routed_moe_one_prealloc(scratch->ffn_moe,
                                  model,
                                  layer,
                                  scratch->ffn_norm,
                                  il,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  scratch->routed_mid_all,
                                  scratch->routed_xq,
                                  scratch->routed_midq);
    if (profile) t_routed = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_shared_ffn_one_decode_scratch(scratch->ffn_shared, model, layer, scratch->ffn_norm, scratch);
    if (profile) t_shared = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
        scratch->ffn_out[i] = scratch->ffn_moe[i] + scratch->ffn_shared[i];
    }
    cpu_directional_steering_project_rows(scratch->ffn_out, steering_dirs, il, 1, steering_scale);
    hc_post_one(out_hc, scratch->ffn_out, inp_hc, post, comb, DS4_N_EMBD, n_hc);
    if (profile) t_post = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "ds4: decode detail layer %u ffn hc=%.3f norm=%.3f routed=%.3f shared=%.3f post=%.3f total=%.3f ms\n",
                il,
                t_hc * 1000.0,
                t_norm * 1000.0,
                t_routed * 1000.0,
                t_shared * 1000.0,
                t_post * 1000.0,
                (now_sec() - t_start) * 1000.0);
    }
}

static void layer_ffn_batch(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale) {
    if (n_tok == 0) return;
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)n_hc * DS4_N_EMBD;
    float *ffn_cur = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(norm[0]));
    float *moe = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(moe[0]));
    float *shared = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(shared[0]));
    float *post = xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));
    const float *ffn_norm = tensor_data(model, layer->ffn_norm);

    for (uint32_t t = 0; t < n_tok; t++) {
        hc_pre_from_state_one(model,
                              layer->hc_ffn_fn,
                              layer->hc_ffn_scale,
                              layer->hc_ffn_base,
                              inp_hc + (uint64_t)t * hc_dim,
                              ffn_cur + (uint64_t)t * DS4_N_EMBD,
                              post + (uint64_t)t * n_hc,
                              comb + (uint64_t)t * n_hc * n_hc);
        rms_norm_weight(norm + (uint64_t)t * DS4_N_EMBD,
                        ffn_cur + (uint64_t)t * DS4_N_EMBD,
                        ffn_norm,
                        DS4_N_EMBD,
                        DS4_RMS_EPS);
    }

    layer_routed_moe_batch(moe, model, layer, norm, token_ids, n_tok, il, DS4_SWIGLU_CLAMP_EXP);
    layer_shared_ffn_batch(shared, model, layer, norm, n_tok);

    if (cpu_directional_steering_enabled(steering_dirs, steering_scale)) {
        float *ffn_out = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(ffn_out[0]));
        for (uint64_t i = 0; i < (uint64_t)n_tok * DS4_N_EMBD; i++) {
            ffn_out[i] = moe[i] + shared[i];
        }
        cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, n_tok, steering_scale);
        hc_post_batch(out_hc,
                      ffn_out,
                      inp_hc,
                      post,
                      comb,
                      n_tok,
                      DS4_N_EMBD,
                      n_hc);
        free(ffn_out);
    } else {
        hc_post_sum_batch(out_hc,
                          moe,
                          shared,
                          inp_hc,
                          post,
                          comb,
                          n_tok,
                          DS4_N_EMBD,
                          n_hc);
    }

    free(comb);
    free(post);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}

typedef struct {
    float *moe;
    const ds4_model *model;
    const ds4_layer_weights *layer;
    const float *norm;
    const int *token_ids;
    uint64_t expert_in_dim;
    uint64_t down_in_dim;
    uint32_t il;
} routed_moe_tokens_ctx;

static void routed_moe_tokens_worker(void *vctx, uint64_t t0, uint64_t t1) {
    routed_moe_tokens_ctx *ctx = vctx;
    float *routed_mid = xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(routed_mid[0]));
    block_q8_K *routed_xq = xmalloc((size_t)(ctx->expert_in_dim / QK_K) * sizeof(routed_xq[0]));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (ctx->down_in_dim / QK_K) * sizeof(routed_midq[0]));

    for (uint64_t t = t0; t < t1; t++) {
        layer_routed_moe_one_prealloc(ctx->moe + t * DS4_N_EMBD,
                                      ctx->model,
                                      ctx->layer,
                                      ctx->norm + t * DS4_N_EMBD,
                                      ctx->il,
                                      ctx->token_ids[t],
                                      DS4_SWIGLU_CLAMP_EXP,
                                      routed_mid,
                                      routed_xq,
                                      routed_midq);
    }

    free(routed_midq);
    free(routed_xq);
    free(routed_mid);
}

static void layer_routed_moe_tokens_parallel(
        float             * moe,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il) {
    routed_moe_tokens_ctx ctx = {
        .moe = moe,
        .model = model,
        .layer = layer,
        .norm = norm,
        .token_ids = token_ids,
        .expert_in_dim = layer->ffn_gate_exps->dim[0],
        .down_in_dim = layer->ffn_down_exps->dim[0],
        .il = il,
    };
    ds4_parallel_for_min_rows(n_tok, routed_moe_tokens_worker, &ctx, 1);
}

/* Default prefill FFN path.  HC and shared expert are batched, while routed
 * experts can run either token-parallel or expert-grouped depending on size. */
static void layer_ffn_shared_batch(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale) {
    const bool profile = getenv("DS4_PREFILL_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc_norm = 0.0;
    double t_routed = 0.0;
    double t_shared = 0.0;
    double t_post = 0.0;
    const uint32_t n_hc = DS4_N_HC;
    float *ffn_cur = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(norm[0]));
    float *moe = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(moe[0]));
    float *shared = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(shared[0]));
    float *post = xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const bool routed_token_parallel =
        getenv("DS4_ROUTED_TOKEN_PARALLEL") != NULL ||
        (getenv("DS4_NO_ROUTED_TOKEN_PARALLEL") == NULL && n_tok >= 64);
    float *routed_mid = routed_token_parallel ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(routed_mid[0]));
    block_q8_K *routed_xq = routed_token_parallel ? NULL : xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(routed_xq[0]));
    block_q8_K *routed_midq = routed_token_parallel ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(routed_midq[0]));

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_norm_batch(model,
                      layer->hc_ffn_fn,
                      layer->hc_ffn_scale,
                      layer->hc_ffn_base,
                      layer->ffn_norm,
                      inp_hc,
                      NULL,
                      ffn_cur,
                      norm,
                      post,
                      comb,
                      n_tok);
    if (profile) t_hc_norm = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    if (routed_token_parallel) {
        layer_routed_moe_tokens_parallel(moe, model, layer, norm, token_ids, n_tok, il);
    } else {
        for (uint32_t t = 0; t < n_tok; t++) {
            layer_routed_moe_one_prealloc(moe + (uint64_t)t * DS4_N_EMBD,
                                          model,
                                          layer,
                                          norm + (uint64_t)t * DS4_N_EMBD,
                                          il,
                                          token_ids[t],
                                          DS4_SWIGLU_CLAMP_EXP,
                                          routed_mid,
                                          routed_xq,
                                          routed_midq);
        }
    }
    if (profile) t_routed = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_shared_ffn_batch(shared, model, layer, norm, n_tok);
    if (profile) t_shared = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    if (cpu_directional_steering_enabled(steering_dirs, steering_scale)) {
        float *ffn_out = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(ffn_out[0]));
        for (uint64_t i = 0; i < (uint64_t)n_tok * DS4_N_EMBD; i++) {
            ffn_out[i] = moe[i] + shared[i];
        }
        cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, n_tok, steering_scale);
        hc_post_batch(out_hc,
                      ffn_out,
                      inp_hc,
                      post,
                      comb,
                      n_tok,
                      DS4_N_EMBD,
                      n_hc);
        free(ffn_out);
    } else {
        hc_post_sum_batch(out_hc,
                          moe,
                          shared,
                          inp_hc,
                          post,
                          comb,
                          n_tok,
                          DS4_N_EMBD,
                          n_hc);
    }
    if (profile) t_post = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "ds4: prefill detail layer %u ffn hc_norm=%.3f routed=%.3f shared=%.3f post=%.3f total=%.3f\n",
                il, t_hc_norm, t_routed, t_shared, t_post, now_sec() - t_start);
    }

    free(comb);
    free(post);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}

typedef struct {
    float *out_hc;
    const ds4_model *model;
    const ds4_layer_weights *layer;
    const float *inp_hc;
    const int *token_ids;
    const float *steering_dirs;
    float steering_scale;
    uint64_t hc_dim;
    uint32_t il;
} layer_ffn_tokens_ctx;

static void layer_ffn_tokens_worker(void *vctx, uint64_t t0, uint64_t t1) {
    layer_ffn_tokens_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        layer_ffn_one(ctx->out_hc + t * ctx->hc_dim,
                      ctx->model,
                      ctx->layer,
                      ctx->inp_hc + t * ctx->hc_dim,
                      ctx->il,
                      ctx->token_ids[t],
                      ctx->steering_dirs,
                      ctx->steering_scale,
                      false);
    }
}

static void layer_ffn_tokens_parallel(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale) {
    layer_ffn_tokens_ctx ctx = {
        .out_hc = out_hc,
        .model = model,
        .layer = layer,
        .inp_hc = inp_hc,
        .token_ids = token_ids,
        .steering_dirs = steering_dirs,
        .steering_scale = steering_scale,
        .hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD,
        .il = il,
    };
    ds4_parallel_for(n_tok, layer_ffn_tokens_worker, &ctx);
}

static void output_logits_one(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        const float       * inp_hc);

/* =========================================================================
 * KV Cache, Compressors, and CPU Layer Execution.
 * =========================================================================
 *
 * The CPU path is the correctness reference.  It maintains raw SWA KV rows,
 * optional compressed KV rows, the indexer mask for ratio-4 layers, and a
 * reusable decode scratch arena so token generation does not allocate in the
 * hot loop.
 */

typedef struct {
    float *raw_kv;
    uint32_t n_raw;
    uint32_t cap_raw;

    uint32_t compress_ratio;
    uint32_t comp_cap;
    uint32_t n_comp;
    float *attn_comp_kv;
    float *attn_state_kv;
    float *attn_state_score;

    uint32_t n_index_comp;
    float *index_comp_kv;
    float *index_state_kv;
    float *index_state_score;
} ds4_layer_cache;

typedef struct {
    ds4_layer_cache layer[DS4_N_LAYER];
    uint32_t head_dim;
} ds4_kv_cache;

static uint32_t ds4_default_raw_cap(uint32_t ctx_size) {
    uint32_t raw_cap = DS4_N_SWA;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    return raw_cap;
}

static uint32_t ds4_default_prefill_cap_for_prompt(int prompt_len) {
    if (prompt_len <= 0) return 1;
    uint32_t cap = (uint32_t)prompt_len;

    const char *env = getenv("DS4_METAL_PREFILL_CHUNK");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return cap;
            cap = (uint32_t)v;
        }
    } else if (prompt_len > 2048) {
        cap = 2048u;
    }

    if (cap == 0) cap = 1;
    if (cap > (uint32_t)prompt_len) cap = (uint32_t)prompt_len;
    return cap;
}

/* Allocate all CPU decode temporaries once.  This keeps generation deterministic
 * from the VM's point of view and makes accidental hot-loop malloc visible. */
static void cpu_decode_scratch_init(ds4_cpu_decode_scratch *scratch, uint32_t ctx_size) {
    memset(scratch, 0, sizeof(*scratch));
    if (ctx_size == 0) ctx_size = 1;
    const uint32_t raw_cap = ds4_default_raw_cap(ctx_size);
    const uint32_t comp_cap = ctx_size / 4 + 2;
    const uint32_t attn_score_cap = raw_cap + comp_cap;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t q8_cap = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t q8_blocks = (q8_cap + 31u) / 32u;

    /*
     * The CPU decode path used to malloc/free dozens of medium-sized buffers
     * for every layer of every generated token. On macOS this can drive the VM
     * system through repeated map/unmap bookkeeping while the huge model mmap is
     * also being streamed, and we have observed kernel panics in VM accounting.
     * Keep decode scratch resident for the whole generation instead.
     */
    scratch->ctx_size = ctx_size;
    scratch->comp_cap = comp_cap;
    scratch->attn_score_cap = attn_score_cap;
    scratch->q8_cap = (uint32_t)q8_cap;

    scratch->plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->cur = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->next = xmalloc((size_t)hc_dim * sizeof(float));

    scratch->attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->attn_residual = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->q = xmalloc((size_t)q_dim * sizeof(float));
    scratch->qr = xmalloc(1024 * sizeof(float));
    scratch->qr_norm = xmalloc(1024 * sizeof(float));
    scratch->kv_raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    scratch->kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    scratch->heads = xmalloc((size_t)q_dim * sizeof(float));
    scratch->attn_low = xmalloc((size_t)8u * 1024u * sizeof(float));
    scratch->attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->attn_score = xmalloc((size_t)attn_score_cap * sizeof(float));

    scratch->comp = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    scratch->index_comp = xmalloc((size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    scratch->comp_kv_cur = xmalloc((size_t)2u * DS4_N_HEAD_DIM * sizeof(float));
    scratch->comp_sc_cur = xmalloc((size_t)2u * DS4_N_HEAD_DIM * sizeof(float));
    scratch->comp_pooled = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));

    scratch->index_allowed = xmalloc((size_t)comp_cap * sizeof(bool));
    scratch->index_q = xmalloc((size_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    scratch->index_weights = xmalloc((size_t)DS4_N_INDEXER_HEAD * sizeof(float));
    scratch->index_scores = xmalloc((size_t)comp_cap * sizeof(float));

    scratch->ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_moe = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->shared_gate = xmalloc((size_t)DS4_N_FF_EXP * sizeof(float));
    scratch->shared_up = xmalloc((size_t)DS4_N_FF_EXP * sizeof(float));
    scratch->shared_mid = xmalloc((size_t)DS4_N_FF_EXP * sizeof(float));
    scratch->routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(float));
    scratch->routed_xq = xmalloc((size_t)(DS4_N_EMBD / QK_K) * sizeof(block_q8_K));
    scratch->routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (DS4_N_FF_EXP / QK_K) * sizeof(block_q8_K));

    scratch->q8_xq = xmalloc((size_t)q8_blocks * 32u);
    scratch->q8_xscale = xmalloc((size_t)q8_blocks * sizeof(float));

    scratch->hc_flat = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->output_flat = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->output_pre = xmalloc((size_t)DS4_N_HC * sizeof(float));
    scratch->output_weights = xmalloc((size_t)DS4_N_HC * sizeof(float));
    scratch->output_embd = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->output_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
}

static void cpu_decode_scratch_free(ds4_cpu_decode_scratch *scratch) {
    if (!scratch) return;
    free(scratch->output_norm);
    free(scratch->output_embd);
    free(scratch->output_weights);
    free(scratch->output_pre);
    free(scratch->output_flat);
    free(scratch->hc_flat);
    free(scratch->q8_xscale);
    free(scratch->q8_xq);
    free(scratch->routed_midq);
    free(scratch->routed_xq);
    free(scratch->routed_mid_all);
    free(scratch->shared_mid);
    free(scratch->shared_up);
    free(scratch->shared_gate);
    free(scratch->ffn_out);
    free(scratch->ffn_shared);
    free(scratch->ffn_moe);
    free(scratch->ffn_norm);
    free(scratch->ffn_cur);
    free(scratch->index_scores);
    free(scratch->index_weights);
    free(scratch->index_q);
    free(scratch->index_allowed);
    free(scratch->comp_pooled);
    free(scratch->comp_sc_cur);
    free(scratch->comp_kv_cur);
    free(scratch->index_comp);
    free(scratch->comp);
    free(scratch->attn_score);
    free(scratch->after_attn_hc);
    free(scratch->attn_out);
    free(scratch->attn_low);
    free(scratch->heads);
    free(scratch->kv);
    free(scratch->kv_raw);
    free(scratch->qr_norm);
    free(scratch->qr);
    free(scratch->q);
    free(scratch->attn_residual);
    free(scratch->attn_norm);
    free(scratch->attn_cur);
    free(scratch->next);
    free(scratch->cur);
    free(scratch->plain);
    memset(scratch, 0, sizeof(*scratch));
}

/* Allocate per-layer KV state: a raw sliding window for all layers, plus
 * compressed attention/indexer caches for layers whose ratio is nonzero. */
static void kv_cache_init(ds4_kv_cache *cache, uint32_t ctx_size, uint32_t raw_cap) {
    memset(cache, 0, sizeof(*cache));
    if (raw_cap == 0) raw_cap = ds4_default_raw_cap(ctx_size);
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;

    cache->head_dim = DS4_N_HEAD_DIM;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        cache->layer[il].cap_raw = raw_cap;
        cache->layer[il].raw_kv = xmalloc_zeroed((size_t)raw_cap * DS4_N_HEAD_DIM, sizeof(float));
        cache->layer[il].compress_ratio = ratio;

        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint32_t comp_cap = ctx_size / ratio + 2;
            const uint32_t attn_width = coff * DS4_N_HEAD_DIM;
            const uint32_t attn_rows = coff * ratio;

            cache->layer[il].comp_cap = comp_cap;
            cache->layer[il].attn_comp_kv = xmalloc_zeroed((size_t)comp_cap * DS4_N_HEAD_DIM, sizeof(float));
            cache->layer[il].attn_state_kv = xmalloc_zeroed((size_t)attn_width * attn_rows, sizeof(float));
            cache->layer[il].attn_state_score = xmalloc((size_t)attn_width * attn_rows * sizeof(float));
            for (uint64_t i = 0; i < (uint64_t)attn_width * attn_rows; i++) {
                cache->layer[il].attn_state_score[i] = DS4_NEG_INF;
            }

            if (ratio == 4) {
                const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
                const uint32_t index_rows = coff * ratio;
                cache->layer[il].index_comp_kv = xmalloc_zeroed((size_t)comp_cap * DS4_N_INDEXER_HEAD_DIM, sizeof(float));
                cache->layer[il].index_state_kv = xmalloc_zeroed((size_t)index_width * index_rows, sizeof(float));
                cache->layer[il].index_state_score = xmalloc((size_t)index_width * index_rows * sizeof(float));
                for (uint64_t i = 0; i < (uint64_t)index_width * index_rows; i++) {
                    cache->layer[il].index_state_score[i] = DS4_NEG_INF;
                }
            }
        }
    }
}

static void kv_cache_free(ds4_kv_cache *cache) {
    if (!cache) return;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        free(cache->layer[il].raw_kv);
        free(cache->layer[il].attn_comp_kv);
        free(cache->layer[il].attn_state_kv);
        free(cache->layer[il].attn_state_score);
        free(cache->layer[il].index_comp_kv);
        free(cache->layer[il].index_state_kv);
        free(cache->layer[il].index_state_score);
    }
    memset(cache, 0, sizeof(*cache));
}

/* Append to the raw SWA cache.  Once full, it slides by one row. */
static void kv_cache_push_raw(ds4_layer_cache *cache, const float *kv) {
    if (cache->n_raw < cache->cap_raw) {
        float *dst = cache->raw_kv + (uint64_t)cache->n_raw * DS4_N_HEAD_DIM;
        for (uint32_t i = 0; i < DS4_N_HEAD_DIM; i++) dst[i] = f16_to_f32(f32_to_f16(kv[i]));
        cache->n_raw++;
        return;
    }

    memmove(cache->raw_kv,
            cache->raw_kv + DS4_N_HEAD_DIM,
            (size_t)(cache->cap_raw - 1) * DS4_N_HEAD_DIM * sizeof(cache->raw_kv[0]));
    float *dst = cache->raw_kv + (uint64_t)(cache->cap_raw - 1) * DS4_N_HEAD_DIM;
    for (uint32_t i = 0; i < DS4_N_HEAD_DIM; i++) dst[i] = f16_to_f32(f32_to_f16(kv[i]));
}

static void kv_cache_push_comp(float *rows, uint32_t *n_rows, uint32_t cap_rows, uint32_t row_dim, const float *kv) {
    if (*n_rows >= cap_rows) ds4_die("compressed KV cache capacity exceeded");
    float *dst = rows + (uint64_t)(*n_rows) * row_dim;
    for (uint32_t i = 0; i < row_dim; i++) dst[i] = f16_to_f32(f32_to_f16(kv[i]));
    (*n_rows)++;
}

/* After prefill, clear unused compressor state rows so decode starts from the
 * same partial-window state the streaming path would have produced. */
static void compressor_finish_prefill_state_cpu(
        float    * state_kv,
        float    * state_score,
        uint32_t   head_dim,
        uint32_t   compress_ratio,
        uint32_t   n_tokens) {
    if (!state_kv || !state_score || head_dim == 0 || compress_ratio == 0) return;

    const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t rem = n_tokens % compress_ratio;
    const uint32_t clear_start = compress_ratio == 4 ? compress_ratio + rem : rem;
    const uint32_t clear_end = compress_ratio == 4 ? 2u * compress_ratio : compress_ratio;

    for (uint32_t row = clear_start; row < clear_end; row++) {
        float *kv = state_kv + (uint64_t)row * width;
        float *score = state_score + (uint64_t)row * width;
        memset(kv, 0, (size_t)width * sizeof(kv[0]));
        for (uint32_t i = 0; i < width; i++) score[i] = DS4_NEG_INF;
    }
}

static void kv_cache_finish_prefill_states(ds4_kv_cache *cache, uint32_t n_tokens) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_cache *layer = &cache->layer[il];
        const uint32_t ratio = layer->compress_ratio;
        if (ratio == 0) continue;

        compressor_finish_prefill_state_cpu(layer->attn_state_kv,
                                            layer->attn_state_score,
                                            DS4_N_HEAD_DIM,
                                            ratio,
                                            n_tokens);
        if (ratio == 4) {
            compressor_finish_prefill_state_cpu(layer->index_state_kv,
                                                layer->index_state_score,
                                                DS4_N_INDEXER_HEAD_DIM,
                                                ratio,
                                                n_tokens);
        }
    }
}

/* Pool the current compression window with a softmax over per-dimension scores.
 * Ratio-4 layers keep two lanes: attention compression and indexer compression. */
static void compressor_pool_decode_state(
        float    * out,
        float    * state_kv,
        float    * state_score,
        uint32_t   head_dim,
        uint32_t   compress_ratio) {
    const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
    const uint32_t width = coff * head_dim;

    for (uint32_t j = 0; j < head_dim; j++) {
        float max_score = DS4_NEG_INF;

        if (compress_ratio == 4) {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float sp = state_score[(uint64_t)r * width + j];
                const float sc = state_score[(uint64_t)(compress_ratio + r) * width + head_dim + j];
                if (sp > max_score) max_score = sp;
                if (sc > max_score) max_score = sc;
            }
        } else {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float s = state_score[(uint64_t)r * width + j];
                if (s > max_score) max_score = s;
            }
        }

        if (max_score <= DS4_NEG_INF * 0.5f) {
            out[j] = 0.0f;
            continue;
        }

        float denom = 0.0f;
        float sum = 0.0f;
        if (compress_ratio == 4) {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float wp = expf(state_score[(uint64_t)r * width + j] - max_score);
                const float wc = expf(state_score[(uint64_t)(compress_ratio + r) * width + head_dim + j] - max_score);
                denom += wp + wc;
                sum += wp * state_kv[(uint64_t)r * width + j];
                sum += wc * state_kv[(uint64_t)(compress_ratio + r) * width + head_dim + j];
            }
        } else {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float w = expf(state_score[(uint64_t)r * width + j] - max_score);
                denom += w;
                sum += w * state_kv[(uint64_t)r * width + j];
            }
        }

        out[j] = denom > 0.0f ? sum / denom : 0.0f;
    }
}

/* Streaming compressor update for one token.  It projects kv/score rows,
 * updates the rolling state, and emits a compressed KV row on ratio boundaries. */
static bool compressor_decode_one(
        float                   * out_comp,
        const ds4_model         * model,
        const ds4_tensor        * wkv,
        const ds4_tensor        * wgate,
        const ds4_tensor        * ape,
        const ds4_tensor        * norm,
        const float             * x,
        float                   * state_kv,
        float                   * state_score,
        uint32_t                  head_dim,
        uint32_t                  compress_ratio,
        uint32_t                  il,
        uint32_t                  pos) {
    const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t pos_mod = pos % compress_ratio;
    const uint32_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = ((pos + 1) % compress_ratio) == 0;

    float *kv_cur = xmalloc((size_t)width * sizeof(kv_cur[0]));
    float *sc_cur = xmalloc((size_t)width * sizeof(sc_cur[0]));
    if (wkv->type == 8 &&
        wgate->type == 8 &&
        wkv->ndim == 2 &&
        wgate->ndim == 2 &&
        wkv->dim[0] == wgate->dim[0]) {
        const uint64_t in_dim = wkv->dim[0];
        const uint64_t blocks = (in_dim + 31) / 32;
        int8_t *xq = xmalloc((size_t)blocks * 32);
        float *xscale = xmalloc((size_t)blocks * sizeof(xscale[0]));

        quantize_q8_0_activation(x, xq, xscale, in_dim);
        matvec_q8_0_pair_prequant(kv_cur, sc_cur, model, wkv, wgate, xq, xscale);

        free(xscale);
        free(xq);
    } else {
        matvec_any(kv_cur, model, wkv, x);
        matvec_any(sc_cur, model, wgate, x);
    }

    for (uint32_t j = 0; j < width; j++) {
        sc_cur[j] += tensor_2d_value(model, ape, j, pos_mod);
    }

    memcpy(state_kv + (uint64_t)row * width, kv_cur, (size_t)width * sizeof(kv_cur[0]));
    memcpy(state_score + (uint64_t)row * width, sc_cur, (size_t)width * sizeof(sc_cur[0]));

    free(sc_cur);
    free(kv_cur);

    if (!should_compress) {
        return false;
    }

    float *pooled = xmalloc((size_t)head_dim * sizeof(pooled[0]));
    compressor_pool_decode_state(pooled, state_kv, state_score, head_dim, compress_ratio);

    double ss = 0.0;
    for (uint32_t i = 0; i < head_dim; i++) ss += (double)pooled[i] * pooled[i];
    const float rms = 1.0f / sqrtf((float)(ss / (double)head_dim) + DS4_RMS_EPS);
    for (uint32_t i = 0; i < head_dim; i++) {
        out_comp[i] = pooled[i] * rms * tensor_1d_value(model, norm, i);
    }

    const uint32_t comp_pos = pos + 1 - compress_ratio;
    rope_tail_layer_inplace(out_comp, 1, head_dim, DS4_N_ROT, comp_pos, il, false);
    if (head_dim == DS4_N_HEAD_DIM) {
        dsv4_fp8_kv_quantize_row_inplace_cpu(out_comp, head_dim, DS4_N_ROT);
    }

    if (compress_ratio == 4) {
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)r * width,
                   state_kv + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)r * width,
                   state_score + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_score[0]));
        }
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)(compress_ratio + r) * width,
                   state_kv + (uint64_t)r * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)(compress_ratio + r) * width,
                   state_score + (uint64_t)r * width,
                   (size_t)width * sizeof(state_score[0]));
        }
    }

    free(pooled);
    return true;
}

static bool compressor_decode_one_decode_scratch(
        float                  * out_comp,
        const ds4_model        * model,
        const ds4_tensor       * wkv,
        const ds4_tensor       * wgate,
        const ds4_tensor       * ape,
        const ds4_tensor       * norm,
        const float            * x,
        float                  * state_kv,
        float                  * state_score,
        uint32_t                 head_dim,
        uint32_t                 compress_ratio,
        uint32_t                 il,
        uint32_t                 pos,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t pos_mod = pos % compress_ratio;
    const uint32_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = ((pos + 1) % compress_ratio) == 0;

    if (width > 2u * DS4_N_HEAD_DIM) ds4_die("compressor scratch width is outside the fixed model layout");
    float *kv_cur = scratch->comp_kv_cur;
    float *sc_cur = scratch->comp_sc_cur;

    if (wkv->type == 8 &&
        wgate->type == 8 &&
        wkv->ndim == 2 &&
        wgate->ndim == 2 &&
        wkv->dim[0] == wgate->dim[0]) {
        matvec_q8_0_pair_decode_scratch(kv_cur, sc_cur, model, wkv, wgate, x, scratch);
    } else {
        matvec_any_decode_scratch(kv_cur, model, wkv, x, scratch);
        matvec_any_decode_scratch(sc_cur, model, wgate, x, scratch);
    }

    for (uint32_t j = 0; j < width; j++) {
        sc_cur[j] += tensor_2d_value(model, ape, j, pos_mod);
    }

    memcpy(state_kv + (uint64_t)row * width, kv_cur, (size_t)width * sizeof(kv_cur[0]));
    memcpy(state_score + (uint64_t)row * width, sc_cur, (size_t)width * sizeof(sc_cur[0]));

    if (!should_compress) {
        return false;
    }

    float *pooled = scratch->comp_pooled;
    compressor_pool_decode_state(pooled, state_kv, state_score, head_dim, compress_ratio);

    double ss = 0.0;
    for (uint32_t i = 0; i < head_dim; i++) ss += (double)pooled[i] * pooled[i];
    const float rms = 1.0f / sqrtf((float)(ss / (double)head_dim) + DS4_RMS_EPS);
    for (uint32_t i = 0; i < head_dim; i++) {
        out_comp[i] = pooled[i] * rms * tensor_1d_value(model, norm, i);
    }

    const uint32_t comp_pos = pos + 1 - compress_ratio;
    rope_tail_layer_inplace(out_comp, 1, head_dim, DS4_N_ROT, comp_pos, il, false);
    if (head_dim == DS4_N_HEAD_DIM) {
        dsv4_fp8_kv_quantize_row_inplace_cpu(out_comp, head_dim, DS4_N_ROT);
    }

    if (compress_ratio == 4) {
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)r * width,
                   state_kv + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)r * width,
                   state_score + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_score[0]));
        }
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)(compress_ratio + r) * width,
                   state_kv + (uint64_t)r * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)(compress_ratio + r) * width,
                   state_score + (uint64_t)r * width,
                   (size_t)width * sizeof(state_score[0]));
        }
    }

    return true;
}

/* Attention over raw SWA rows plus optional compressed rows.  Ratio-4 layers
 * pass an indexer mask to hide compressed rows not selected for this token. */
static void layer_attention_mixed_one(
        float             * out_heads,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * q,
        const float       * raw_kv,
        uint32_t            n_raw,
        const float       * comp_kv,
        uint32_t            n_comp,
        const bool        * comp_allowed) {
    const float *sinks = tensor_data(model, layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const uint32_t n_total = n_raw + n_comp;
    float score_stack[512];
    float *score = n_total <= 512 ? score_stack : xmalloc((size_t)n_total * sizeof(score[0]));

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_N_HEAD_DIM;
        float max_score = sinks[h];
        uint32_t idx = 0;

        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (comp_allowed && !comp_allowed[r]) {
                score[idx] = DS4_NEG_INF;
                continue;
            }
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }

        float *oh = out_heads + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        idx = 0;
        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float weight = expf(score[idx] - max_score);
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (score[idx] <= DS4_NEG_INF * 0.5f) continue;
            const float weight = expf(score[idx] - max_score);
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        const float inv = 1.0f / denom;
        scale_f32(oh, inv, DS4_N_HEAD_DIM);
    }

    if (score != score_stack) free(score);
}

static void layer_attention_mixed_one_decode_scratch(
        float                  * out_heads,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * q,
        const float            * raw_kv,
        uint32_t                 n_raw,
        const float            * comp_kv,
        uint32_t                 n_comp,
        const bool             * comp_allowed,
        ds4_cpu_decode_scratch * scratch) {
    const float *sinks = tensor_data(model, layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const uint32_t n_total = n_raw + n_comp;
    if (n_total > scratch->attn_score_cap) ds4_die("CPU decode attention score scratch buffer is too small");
    float *score = scratch->attn_score;

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_N_HEAD_DIM;
        float max_score = sinks[h];
        uint32_t idx = 0;

        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (comp_allowed && !comp_allowed[r]) {
                score[idx] = DS4_NEG_INF;
                continue;
            }
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }

        float *oh = out_heads + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        idx = 0;
        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float weight = expf(score[idx] - max_score);
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (score[idx] <= DS4_NEG_INF * 0.5f) continue;
            const float weight = expf(score[idx] - max_score);
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        const float inv = 1.0f / denom;
        scale_f32(oh, inv, DS4_N_HEAD_DIM);
    }
}

typedef struct {
    float             * out_heads;
    const ds4_model   * model;
    const ds4_layer_weights * layer;
    const float       * q;
    const float       * raw_kv;
    const float       * comp_kv;
    const uint32_t    * comp_counts;
    const uint8_t     * allowed_mask;
    const uint8_t     * allowed_bits;
    uint64_t            allowed_stride;
    uint32_t            n_tok;
    uint32_t            raw_cap;
} layer_attention_prefix_batch_ctx;

static inline bool attention_prefix_comp_allowed(
        const layer_attention_prefix_batch_ctx *ctx,
        uint32_t                                t,
        uint32_t                                c) {
    if (!ctx->allowed_bits || !ctx->allowed_mask || !ctx->allowed_mask[t]) return true;
    const uint8_t *bits = ctx->allowed_bits + (uint64_t)t * ctx->allowed_stride;
    return (bits[c >> 3] & (uint8_t)(1u << (c & 7u))) != 0;
}

static void layer_attention_prefix_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    layer_attention_prefix_batch_ctx *ctx = vctx;
    const float *sinks = tensor_data(ctx->model, ctx->layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const uint32_t max_comp = ctx->comp_counts ? ctx->comp_counts[ctx->n_tok - 1] : 0;
    const uint32_t max_total = ctx->raw_cap + max_comp;
    float score_stack[2048];
    float *score = max_total <= 2048 ? score_stack : xmalloc((size_t)max_total * sizeof(score[0]));

    for (uint64_t idx = r0; idx < r1; idx++) {
        const uint32_t t = (uint32_t)(idx / DS4_N_HEAD);
        const uint32_t h = (uint32_t)(idx - (uint64_t)t * DS4_N_HEAD);
        const uint32_t raw_count = t + 1 < ctx->raw_cap ? t + 1 : ctx->raw_cap;
        const uint32_t raw_start = t + 1 - raw_count;
        const uint32_t comp_count = ctx->comp_counts ? ctx->comp_counts[t] : 0;
        const float *qh = ctx->q + (uint64_t)t * DS4_N_HEAD * DS4_N_HEAD_DIM + (uint64_t)h * DS4_N_HEAD_DIM;

        float max_score = sinks[h];
        uint32_t sidx = 0;
        for (uint32_t r = 0; r < raw_count; r++, sidx++) {
            const float *kv = ctx->raw_kv + (uint64_t)(raw_start + r) * DS4_N_HEAD_DIM;
            score[sidx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[sidx] > max_score) max_score = score[sidx];
        }
        for (uint32_t c = 0; c < comp_count; c++, sidx++) {
            if (!attention_prefix_comp_allowed(ctx, t, c)) {
                score[sidx] = DS4_NEG_INF;
                continue;
            }
            const float *kv = ctx->comp_kv + (uint64_t)c * DS4_N_HEAD_DIM;
            score[sidx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[sidx] > max_score) max_score = score[sidx];
        }

        float *oh = ctx->out_heads + (uint64_t)t * DS4_N_HEAD * DS4_N_HEAD_DIM + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        sidx = 0;
        for (uint32_t r = 0; r < raw_count; r++, sidx++) {
            const float weight = expf(score[sidx] - max_score);
            const float *kv = ctx->raw_kv + (uint64_t)(raw_start + r) * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }
        for (uint32_t c = 0; c < comp_count; c++, sidx++) {
            if (score[sidx] <= DS4_NEG_INF * 0.5f) continue;
            const float weight = expf(score[sidx] - max_score);
            const float *kv = ctx->comp_kv + (uint64_t)c * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        scale_f32(oh, 1.0f / denom, DS4_N_HEAD_DIM);
    }

    if (score != score_stack) free(score);
}

/* Prefix prefill attention for a fresh prompt.  It computes each token's view
 * of the raw window and compressed rows without running the decode loop. */
static void layer_attention_prefix_batch(
        float                   * out_heads,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * q,
        const float             * raw_kv,
        const float             * comp_kv,
        const uint32_t          * comp_counts,
        const uint8_t           * allowed_mask,
        const uint8_t           * allowed_bits,
        uint64_t                  allowed_stride,
        uint32_t                  n_tok,
        uint32_t                  raw_cap) {
    layer_attention_prefix_batch_ctx ctx = {
        .out_heads = out_heads,
        .model = model,
        .layer = layer,
        .q = q,
        .raw_kv = raw_kv,
        .comp_kv = comp_kv,
        .comp_counts = comp_counts,
        .allowed_mask = allowed_mask,
        .allowed_bits = allowed_bits,
        .allowed_stride = allowed_stride,
        .n_tok = n_tok,
        .raw_cap = raw_cap,
    };
    ds4_parallel_for_min_rows((uint64_t)n_tok * DS4_N_HEAD,
                              layer_attention_prefix_batch_worker,
                              &ctx,
                              1);
}

/* Ratio-4 layers use an auxiliary indexer to select which compressed rows are
 * visible to attention.  This is the CPU allocation-owning helper. */
static bool *indexer_allowed_decode_one(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * cur,
        const float             * qr_norm,
        const float             * index_comp,
        uint32_t                  n_comp,
        uint32_t                  il,
        uint32_t                  pos) {
    if (n_comp == 0) return NULL;

    bool *allowed = xcalloc(n_comp, sizeof(allowed[0]));
    const uint32_t top_k = DS4_N_INDEXER_TOP_K < n_comp ? DS4_N_INDEXER_TOP_K : n_comp;
    if (top_k == n_comp) {
        for (uint32_t i = 0; i < n_comp; i++) allowed[i] = true;
        return allowed;
    }

    const uint32_t head_dim = DS4_N_INDEXER_HEAD_DIM;
    const uint32_t n_head = DS4_N_INDEXER_HEAD;
    float *q = xmalloc((size_t)head_dim * n_head * sizeof(q[0]));
    float *weights = xmalloc((size_t)n_head * sizeof(weights[0]));
    float *scores = xmalloc((size_t)n_comp * sizeof(scores[0]));

    matvec_any(q, model, layer->indexer_attn_q_b, qr_norm);
    rope_tail_layer_inplace(q, n_head, head_dim, DS4_N_ROT, pos, il, false);

    matvec_any(weights, model, layer->indexer_proj, cur);
    const float scale = 1.0f / sqrtf((float)(head_dim * n_head));
    for (uint32_t h = 0; h < n_head; h++) weights[h] *= scale;

    for (uint32_t c = 0; c < n_comp; c++) {
        const float *kv = index_comp + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + (uint64_t)h * head_dim;
            float dot = dot_f32(kv, qh, head_dim);
            if (dot < 0.0f) dot = 0.0f;
            s += dot * weights[h];
        }
        scores[c] = s;
    }

    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t best = 0;
        float best_score = DS4_NEG_INF;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (!allowed[c] && scores[c] > best_score) {
                best = c;
                best_score = scores[c];
            }
        }
        allowed[best] = true;
    }

    free(scores);
    free(weights);
    free(q);
    return allowed;
}

/* Scratch-backed indexer selection for decode. */
static bool *indexer_allowed_decode_one_decode_scratch(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * cur,
        const float             * qr_norm,
        const float             * index_comp,
        uint32_t                  n_comp,
        uint32_t                  il,
        uint32_t                  pos,
        ds4_cpu_decode_scratch  * scratch) {
    if (n_comp == 0) return NULL;
    if (n_comp > scratch->comp_cap) ds4_die("CPU decode indexer scratch buffer is too small");

    bool *allowed = scratch->index_allowed;
    memset(allowed, 0, (size_t)n_comp * sizeof(allowed[0]));
    const uint32_t top_k = DS4_N_INDEXER_TOP_K < n_comp ? DS4_N_INDEXER_TOP_K : n_comp;
    if (top_k == n_comp) {
        for (uint32_t i = 0; i < n_comp; i++) allowed[i] = true;
        return allowed;
    }

    const uint32_t head_dim = DS4_N_INDEXER_HEAD_DIM;
    const uint32_t n_head = DS4_N_INDEXER_HEAD;
    float *q = scratch->index_q;
    float *weights = scratch->index_weights;
    float *scores = scratch->index_scores;

    matvec_any_decode_scratch(q, model, layer->indexer_attn_q_b, qr_norm, scratch);
    rope_tail_layer_inplace(q, n_head, head_dim, DS4_N_ROT, pos, il, false);

    matvec_any_decode_scratch(weights, model, layer->indexer_proj, cur, scratch);
    const float scale = 1.0f / sqrtf((float)(head_dim * n_head));
    for (uint32_t h = 0; h < n_head; h++) weights[h] *= scale;

    for (uint32_t c = 0; c < n_comp; c++) {
        const float *kv = index_comp + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + (uint64_t)h * head_dim;
            float dot = dot_f32(kv, qh, head_dim);
            if (dot < 0.0f) dot = 0.0f;
            s += dot * weights[h];
        }
        scores[c] = s;
    }

    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t best = 0;
        float best_score = DS4_NEG_INF;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (!allowed[c] && scores[c] > best_score) {
                best = c;
                best_score = scores[c];
            }
        }
        allowed[best] = true;
    }

    return allowed;
}

/* Single-token attention sublayer with raw SWA cache and DS4 compression. */
static void layer_attention_raw_swa_one(
        float                   * after_attn_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        const float             * steering_dirs,
        float                     steering_scale) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    float *attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(attn_residual[0]));
    float *q = xmalloc((size_t)q_dim * sizeof(q[0]));
    float *qr_norm = xmalloc(1024 * sizeof(qr_norm[0]));
    float *kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = xmalloc((size_t)q_dim * sizeof(heads[0]));
    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    bool *comp_allowed = NULL;
    float post[4];
    float comb[16];

    memcpy(attn_residual, inp_hc, (size_t)n_hc * DS4_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          attn_residual, attn_cur, post, comb);

    layer_attn_norm_one(attn_norm, model, layer, attn_cur);
    layer_q_projection_with_lora_one(model, layer, attn_norm, q, qr_norm);
    layer_kv_projection_normed_one(model, layer, attn_norm, kv);

    rope_tail_layer_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    rope_tail_layer_inplace(kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv, DS4_N_HEAD_DIM, DS4_N_ROT);

    kv_cache_push_raw(cache, kv);

    const uint32_t ratio = cache->compress_ratio;
    if (ratio != 0) {
        float *comp = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(comp[0]));
        if (compressor_decode_one(comp, model,
                                  layer->attn_compressor_kv,
                                  layer->attn_compressor_gate,
                                  layer->attn_compressor_ape,
                                  layer->attn_compressor_norm,
                                  attn_norm,
                                  cache->attn_state_kv,
                                  cache->attn_state_score,
                                  DS4_N_HEAD_DIM,
                                  ratio,
                                  il,
                                  pos)) {
            kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, comp);
        }
        free(comp);

        if (ratio == 4) {
            float *index_comp = xmalloc((size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(index_comp[0]));
            if (compressor_decode_one(index_comp, model,
                                      layer->indexer_compressor_kv,
                                      layer->indexer_compressor_gate,
                                      layer->indexer_compressor_ape,
                                      layer->indexer_compressor_norm,
                                      attn_norm,
                                      cache->index_state_kv,
                                      cache->index_state_score,
                                      DS4_N_INDEXER_HEAD_DIM,
                                      ratio,
                                      il,
                                      pos)) {
                kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap, DS4_N_INDEXER_HEAD_DIM, index_comp);
            }
            free(index_comp);

            comp_allowed = indexer_allowed_decode_one(model, layer,
                                                      attn_norm, qr_norm,
                                                      cache->index_comp_kv,
                                                      cache->n_index_comp,
                                                      il, pos);
        }

        layer_attention_mixed_one(heads, model, layer, q,
                                  cache->raw_kv, cache->n_raw,
                                  cache->attn_comp_kv, cache->n_comp,
                                  comp_allowed);
    } else {
        layer_attention_rows_one(heads, model, layer, q, cache->raw_kv, cache->n_raw);
    }

    rope_tail_layer_inplace(heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);
    layer_grouped_out_one(attn_out, model, layer, heads);
    cpu_directional_steering_project_rows(attn_out, steering_dirs, il, 1, steering_scale);
    hc_post_one(after_attn_hc, attn_out, attn_residual, post, comb, DS4_N_EMBD, n_hc);

    free(comp_allowed);
    free(attn_out);
    free(heads);
    free(kv);
    free(qr_norm);
    free(q);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}

/* Batched prefill attention.  It projects Q/KV for all tokens, streams them
 * through the same raw/compressed cache updates, then runs prefix attention. */
static void layer_attention_raw_swa_batch(
        float                   * after_attn_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  n_tok,
        uint32_t                  il,
        uint32_t                  pos0,
        const float             * steering_dirs,
        float                     steering_scale) {
    const bool profile = getenv("DS4_PREFILL_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc_norm = 0.0;
    double t_q = 0.0;
    double t_kv = 0.0;
    double t_token_loop = 0.0;
    double t_tl_rope_cache = 0.0;
    double t_tl_compress = 0.0;
    double t_tl_indexer = 0.0;
    double t_tl_attn_rows = 0.0;
    double t_tl_inv_rope = 0.0;
    double t_out = 0.0;
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)n_hc * DS4_N_EMBD;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    float *attn_cur = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = xmalloc((size_t)n_tok * hc_dim * sizeof(attn_residual[0]));
    float *qr = xmalloc((size_t)n_tok * 1024 * sizeof(qr[0]));
    float *qr_norm = xmalloc((size_t)n_tok * 1024 * sizeof(qr_norm[0]));
    float *q = xmalloc((size_t)n_tok * q_dim * sizeof(q[0]));
    float *kv_raw = xmalloc((size_t)n_tok * DS4_N_HEAD_DIM * sizeof(kv_raw[0]));
    float *kv = xmalloc((size_t)n_tok * DS4_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = NULL;
    float *attn_out = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_out[0]));
    float *post = xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));

    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);
    const float *kv_norm = tensor_data(model, layer->attn_kv_a_norm);

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_norm_batch(model,
                      layer->hc_attn_fn,
                      layer->hc_attn_scale,
                      layer->hc_attn_base,
                      layer->attn_norm,
                      inp_hc,
                      attn_residual,
                      attn_cur,
                      attn_norm,
                      post,
                      comb,
                      n_tok);
    if (profile) t_hc_norm = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    matmul_q8_0_batch(qr, model, layer->attn_q_a, attn_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        rms_norm_weight(qr_norm + (uint64_t)t * 1024,
                        qr + (uint64_t)t * 1024,
                        q_a_norm,
                        1024,
                        DS4_RMS_EPS);
    }
    matmul_q8_0_batch(q, model, layer->attn_q_b, qr_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        head_rms_norm_inplace(q + (uint64_t)t * q_dim,
                              DS4_N_HEAD,
                              DS4_N_HEAD_DIM,
                              DS4_RMS_EPS);
    }
    if (profile) t_q = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    matmul_q8_0_batch(kv_raw, model, layer->attn_kv, attn_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        rms_norm_weight(kv + (uint64_t)t * DS4_N_HEAD_DIM,
                        kv_raw + (uint64_t)t * DS4_N_HEAD_DIM,
                        kv_norm,
                        DS4_N_HEAD_DIM,
                        DS4_RMS_EPS);
    }
    if (profile) t_kv = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    const uint32_t ratio = cache->compress_ratio;
    const bool prefer_parallel_attn = getenv("DS4_PARALLEL_ATTN_ROWS") != NULL;
    const bool prefix_batch_attn =
        prefer_parallel_attn &&
        getenv("DS4_NO_PARALLEL_ATTN_ROWS") == NULL &&
        cache->n_raw == 0 &&
        pos0 == 0;
    if (!prefix_batch_attn) {
        heads = xmalloc((size_t)n_tok * q_dim * sizeof(heads[0]));
    }
    uint32_t batch_rope_max = 4096;
    const char *batch_rope_max_env = getenv("DS4_BATCHED_ROPE_MAX");
    if (batch_rope_max_env && batch_rope_max_env[0]) {
        long v = strtol(batch_rope_max_env, NULL, 10);
        if (v >= 0 && v <= 65536) batch_rope_max = (uint32_t)v;
    }
    const bool batch_prefix_rope =
        prefix_batch_attn &&
        getenv("DS4_NO_BATCHED_ROPE") == NULL &&
        n_tok <= batch_rope_max;
    uint32_t *comp_counts = prefix_batch_attn ?
        xcalloc((size_t)n_tok, sizeof(comp_counts[0])) : NULL;
    uint8_t *allowed_mask = prefix_batch_attn && ratio == 4 ?
        xcalloc((size_t)n_tok, sizeof(allowed_mask[0])) : NULL;
    uint8_t *allowed_bits = NULL;
    const uint64_t allowed_stride = ratio == 4 ? ((uint64_t)cache->comp_cap + 7u) / 8u : 0;
    float *comp_scratch = NULL;
    float *index_comp_scratch = NULL;

    if (ratio != 0) {
        comp_scratch = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(comp_scratch[0]));

        if (ratio == 4) {
            index_comp_scratch = xmalloc((size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(index_comp_scratch[0]));
        }
    }

    if (batch_prefix_rope) {
        double tx = profile ? now_sec() : 0.0;
        rope_tail_layer_batch_inplace(q,
                                      q_dim,
                                      DS4_N_HEAD,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos0,
                                      il,
                                      false,
                                      n_tok);
        rope_tail_layer_batch_inplace(kv,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_HEAD_KV,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos0,
                                      il,
                                      false,
                                      n_tok);
        if (profile) t_tl_rope_cache += now_sec() - tx;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t pos = pos0 + t;
        float *q_t = q + (uint64_t)t * q_dim;
        float *kv_t = kv + (uint64_t)t * DS4_N_HEAD_DIM;
        bool *comp_allowed = NULL;

        double tx = profile ? now_sec() : 0.0;
        if (!batch_prefix_rope) {
            rope_tail_layer_inplace(q_t, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
            rope_tail_layer_inplace(kv_t, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
        }
        dsv4_fp8_kv_quantize_row_inplace_cpu(kv_t, DS4_N_HEAD_DIM, DS4_N_ROT);

        kv_cache_push_raw(cache, kv_t);
        if (profile) t_tl_rope_cache += now_sec() - tx;

        if (ratio != 0) {
            tx = profile ? now_sec() : 0.0;
            float *comp = comp_scratch;
            const bool have_comp = compressor_decode_one(comp, model,
                                                         layer->attn_compressor_kv,
                                                         layer->attn_compressor_gate,
                                                         layer->attn_compressor_ape,
                                                         layer->attn_compressor_norm,
                                                         attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                         cache->attn_state_kv,
                                                         cache->attn_state_score,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         il,
                                                         pos);
            if (have_comp) {
                kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, comp);
            }

            if (ratio == 4) {
                float *index_comp = index_comp_scratch;
                const bool have_index_comp = compressor_decode_one(index_comp, model,
                                                                   layer->indexer_compressor_kv,
                                                                   layer->indexer_compressor_gate,
                                                                   layer->indexer_compressor_ape,
                                                                   layer->indexer_compressor_norm,
                                                                   attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                                   cache->index_state_kv,
                                                                   cache->index_state_score,
                                                                   DS4_N_INDEXER_HEAD_DIM,
                                                                   ratio,
                                                                   il,
                                                                   pos);
                if (have_index_comp) {
                    kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap, DS4_N_INDEXER_HEAD_DIM, index_comp);
                }
                if (profile) t_tl_compress += now_sec() - tx;

                tx = profile ? now_sec() : 0.0;
                comp_allowed = indexer_allowed_decode_one(model, layer,
                                                          attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                          qr_norm + (uint64_t)t * 1024,
                                                          cache->index_comp_kv,
                                                          cache->n_index_comp,
                                                          il, pos);
                if (profile) t_tl_indexer += now_sec() - tx;
            } else {
                if (profile) t_tl_compress += now_sec() - tx;
            }

            if (comp_counts) comp_counts[t] = cache->n_comp;
            if (prefix_batch_attn && comp_allowed) {
                if (!allowed_bits) {
                    allowed_bits = xcalloc((size_t)n_tok * allowed_stride, sizeof(allowed_bits[0]));
                }
                allowed_mask[t] = 1;
                uint8_t *bits = allowed_bits + (uint64_t)t * allowed_stride;
                for (uint32_t c = 0; c < cache->n_comp; c++) {
                    if (comp_allowed[c]) bits[c >> 3] |= (uint8_t)(1u << (c & 7u));
                }
            }

            if (!prefix_batch_attn) {
                tx = profile ? now_sec() : 0.0;
                layer_attention_mixed_one(heads + (uint64_t)t * q_dim, model, layer, q_t,
                                          cache->raw_kv, cache->n_raw,
                                          cache->attn_comp_kv, cache->n_comp,
                                          comp_allowed);
                if (profile) t_tl_attn_rows += now_sec() - tx;
            }
        } else {
            if (!prefix_batch_attn) {
                tx = profile ? now_sec() : 0.0;
                layer_attention_rows_one(heads + (uint64_t)t * q_dim, model, layer, q_t, cache->raw_kv, cache->n_raw);
                if (profile) t_tl_attn_rows += now_sec() - tx;
            }
        }

        if (!prefix_batch_attn) {
            tx = profile ? now_sec() : 0.0;
            rope_tail_layer_inplace(heads + (uint64_t)t * q_dim,
                                    DS4_N_HEAD,
                                    DS4_N_HEAD_DIM,
                                    DS4_N_ROT,
                                    pos,
                                    il,
                                    true);
            if (profile) t_tl_inv_rope += now_sec() - tx;
        }

        free(comp_allowed);
    }

    if (prefix_batch_attn) {
        double tx = profile ? now_sec() : 0.0;
        const float *comp_kv_for_prefix = cache->attn_comp_kv ? cache->attn_comp_kv : kv;
        if (!heads) {
            heads = xmalloc((size_t)n_tok * q_dim * sizeof(heads[0]));
        }
        layer_attention_prefix_batch(heads, model, layer,
                                     q,
                                     kv,
                                     comp_kv_for_prefix,
                                     comp_counts,
                                     allowed_mask,
                                     allowed_bits,
                                     allowed_stride,
                                     n_tok,
                                     cache->cap_raw);
        if (profile) t_tl_attn_rows += now_sec() - tx;
        tx = profile ? now_sec() : 0.0;
        if (batch_prefix_rope) {
            rope_tail_layer_batch_inplace(heads,
                                          q_dim,
                                          DS4_N_HEAD,
                                          DS4_N_HEAD_DIM,
                                          DS4_N_ROT,
                                          pos0,
                                          il,
                                          true,
                                          n_tok);
        } else {
            for (uint32_t t = 0; t < n_tok; t++) {
                rope_tail_layer_inplace(heads + (uint64_t)t * q_dim,
                                        DS4_N_HEAD,
                                        DS4_N_HEAD_DIM,
                                        DS4_N_ROT,
                                        pos0 + t,
                                        il,
                                        true);
            }
        }
        if (profile) t_tl_inv_rope += now_sec() - tx;
    }
    if (profile) t_token_loop = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_grouped_out_batch(attn_out, model, layer, heads, n_tok);
    cpu_directional_steering_project_rows(attn_out, steering_dirs, il, n_tok, steering_scale);

    hc_post_batch(after_attn_hc,
                  attn_out,
                  attn_residual,
                  post,
                  comb,
                  n_tok,
                  DS4_N_EMBD,
                  n_hc);
    if (profile) t_out = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "ds4: prefill detail layer %u attn hc_norm=%.3f q=%.3f kv=%.3f token_loop=%.3f out=%.3f total=%.3f\n",
                il, t_hc_norm, t_q, t_kv, t_token_loop, t_out, now_sec() - t_start);
        if (getenv("DS4_PREFILL_PROFILE_TOKEN") != NULL) {
            fprintf(stderr,
                    "ds4: prefill token detail layer %u rope_cache=%.3f compress=%.3f indexer=%.3f attn_rows=%.3f inv_rope=%.3f\n",
                    il, t_tl_rope_cache, t_tl_compress, t_tl_indexer, t_tl_attn_rows, t_tl_inv_rope);
        }
    }

    free(allowed_bits);
    free(allowed_mask);
    free(comp_counts);
    free(index_comp_scratch);
    free(comp_scratch);
    free(comb);
    free(post);
    free(attn_out);
    free(heads);
    free(kv);
    free(kv_raw);
    free(q);
    free(qr_norm);
    free(qr);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}

/* Full transformer layer for one decode token: attention sublayer followed by
 * FFN sublayer, both operating on the HC state. */
static void layer_forward_raw_swa_one(
        float                   * out_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        int                       token,
        const float             * steering_dirs,
        float                     steering_attn_scale,
        float                     steering_ffn_scale,
        ds4_cpu_decode_scratch  * scratch) {
    const uint32_t n_hc = DS4_N_HC;
    const bool profile = getenv("DS4_DECODE_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc = 0.0;
    double t_q = 0.0;
    double t_kv = 0.0;
    double t_rope_cache = 0.0;
    double t_compress = 0.0;
    double t_indexer = 0.0;
    double t_attn_rows = 0.0;
    double t_inv_rope = 0.0;
    double t_out = 0.0;
    double t_post = 0.0;
    double t_ffn = 0.0;

    bool *comp_allowed = NULL;
    float post[4];
    float comb[16];

    double t0 = profile ? now_sec() : 0.0;
    memcpy(scratch->attn_residual, inp_hc, (size_t)n_hc * DS4_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one_scratch(model,
                                  layer->hc_attn_fn,
                                  layer->hc_attn_scale,
                                  layer->hc_attn_base,
                                  scratch->attn_residual, scratch->attn_cur, post, comb,
                                  scratch->hc_flat,
                                  false);
    if (profile) t_hc = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_attn_norm_one(scratch->attn_norm, model, layer, scratch->attn_cur);
    const uint32_t ratio = cache->compress_ratio;
    layer_q_projection_with_lora_one_decode_scratch(model, layer,
                                                    scratch->attn_norm,
                                                    scratch->q,
                                                    scratch->qr_norm,
                                                    scratch);
    if (profile) t_q = now_sec() - t0;
    t0 = profile ? now_sec() : 0.0;
    layer_kv_projection_normed_one_decode_scratch(model, layer,
                                                  scratch->attn_norm,
                                                  scratch->kv,
                                                  scratch);
    if (profile) t_kv = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    rope_tail_layer_inplace(scratch->q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    rope_tail_layer_inplace(scratch->kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(scratch->kv, DS4_N_HEAD_DIM, DS4_N_ROT);

    kv_cache_push_raw(cache, scratch->kv);
    if (profile) t_rope_cache = now_sec() - t0;

    if (ratio != 0) {
        t0 = profile ? now_sec() : 0.0;
        if (compressor_decode_one_decode_scratch(scratch->comp, model,
                                                 layer->attn_compressor_kv,
                                                 layer->attn_compressor_gate,
                                                 layer->attn_compressor_ape,
                                                 layer->attn_compressor_norm,
                                                 scratch->attn_norm,
                                                 cache->attn_state_kv,
                                                 cache->attn_state_score,
                                                 DS4_N_HEAD_DIM,
                                                 ratio,
                                                 il,
                                                 pos,
                                                 scratch)) {
            kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, scratch->comp);
        }

        if (ratio == 4) {
            if (compressor_decode_one_decode_scratch(scratch->index_comp, model,
                                                     layer->indexer_compressor_kv,
                                                     layer->indexer_compressor_gate,
                                                     layer->indexer_compressor_ape,
                                                     layer->indexer_compressor_norm,
                                                     scratch->attn_norm,
                                                     cache->index_state_kv,
                                                     cache->index_state_score,
                                                     DS4_N_INDEXER_HEAD_DIM,
                                                     ratio,
                                                     il,
                                                     pos,
                                                     scratch)) {
                kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap,
                                   DS4_N_INDEXER_HEAD_DIM, scratch->index_comp);
            }
            if (profile) t_compress = now_sec() - t0;
        } else if (profile) {
            t_compress = now_sec() - t0;
        }
    }
    if (ratio == 4) {
        t0 = profile ? now_sec() : 0.0;
        comp_allowed = indexer_allowed_decode_one_decode_scratch(model, layer,
                                                                 scratch->attn_norm,
                                                                 scratch->qr_norm,
                                                                 cache->index_comp_kv,
                                                                 cache->n_index_comp,
                                                                 il, pos,
                                                                 scratch);
        if (profile) t_indexer = now_sec() - t0;
    }

    t0 = profile ? now_sec() : 0.0;
    if (ratio != 0) {
        layer_attention_mixed_one_decode_scratch(scratch->heads, model, layer, scratch->q,
                                                 cache->raw_kv, cache->n_raw,
                                                 cache->attn_comp_kv, cache->n_comp,
                                                 comp_allowed,
                                                 scratch);
    } else {
        layer_attention_rows_one(scratch->heads, model, layer, scratch->q, cache->raw_kv, cache->n_raw);
    }
    if (profile) t_attn_rows = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    rope_tail_layer_inplace(scratch->heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);
    if (profile) t_inv_rope = now_sec() - t0;
    t0 = profile ? now_sec() : 0.0;
    layer_grouped_out_one_decode_scratch(scratch->attn_out, model, layer, scratch->heads, scratch);
    cpu_directional_steering_project_rows(scratch->attn_out, steering_dirs, il, 1, steering_attn_scale);
    if (profile) t_out = now_sec() - t0;
    t0 = profile ? now_sec() : 0.0;
    hc_post_one(scratch->after_attn_hc, scratch->attn_out, scratch->attn_residual, post, comb, DS4_N_EMBD, n_hc);
    if (profile) t_post = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_ffn_one_decode_scratch(out_hc, model, layer, scratch->after_attn_hc, il, token,
                                 steering_dirs, steering_ffn_scale, scratch);
    if (profile) t_ffn = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "ds4: decode detail layer %u attn hc=%.3f q=%.3f kv=%.3f rope=%.3f compress=%.3f indexer=%.3f attn_rows=%.3f inv_rope=%.3f out=%.3f post=%.3f ffn=%.3f total=%.3f ms\n",
                il,
                t_hc * 1000.0,
                t_q * 1000.0,
                t_kv * 1000.0,
                t_rope_cache * 1000.0,
                t_compress * 1000.0,
                t_indexer * 1000.0,
                t_attn_rows * 1000.0,
                t_inv_rope * 1000.0,
                t_out * 1000.0,
                t_post * 1000.0,
                t_ffn * 1000.0,
                (now_sec() - t_start) * 1000.0);
    }

}

static void output_logits_one_decode_scratch(
        float                  * logits,
        const ds4_model        * model,
        const ds4_weights      * weights,
        const float            * inp_hc,
        ds4_cpu_decode_scratch * scratch);

/* CPU decode for one token through all 43 layers.  The caller owns scratch and
 * cache lifetimes so no per-token allocations are needed. */
static void forward_token_raw_swa_cpu_decode_scratch(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        int                 token,
        uint32_t            pos,
        const float       * steering_dirs,
        float               steering_attn_scale,
        float               steering_ffn_scale,
        ds4_cpu_decode_scratch * scratch) {
    float *cur = scratch->cur;
    float *next = scratch->next;

    embed_token_f16(model, weights, token, scratch->plain);
    hc_from_plain_embedding(cur, scratch->plain, DS4_N_EMBD, DS4_N_HC);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        layer_forward_raw_swa_one(next, model, &weights->layer[il], &cache->layer[il],
                                  cur, il, pos, token,
                                  steering_dirs,
                                  steering_attn_scale,
                                  steering_ffn_scale,
                                  scratch);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    if (logits) {
        output_logits_one_decode_scratch(logits, model, weights, cur, scratch);
    }
}

#ifndef DS4_NO_GPU
static void forward_token_raw_swa_cpu(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        int                 token,
        uint32_t            pos) {
    ds4_cpu_decode_scratch scratch;
    uint32_t ctx_guess = pos + 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = cache->layer[il].compress_ratio;
        if (ratio != 0 && cache->layer[il].comp_cap > 2) {
            const uint32_t ctx_from_comp = (cache->layer[il].comp_cap - 2u) * ratio;
            if (ctx_guess < ctx_from_comp) ctx_guess = ctx_from_comp;
        }
    }
    cpu_decode_scratch_init(&scratch, ctx_guess);
    forward_token_raw_swa_cpu_decode_scratch(logits, model, weights, cache, token, pos,
                                             NULL, 0.0f, 0.0f, &scratch);
    cpu_decode_scratch_free(&scratch);
}
#endif

/* CPU prefill in layer-major order.  All prompt tokens pass through layer 0,
 * then layer 1, etc., which exposes batch matmul opportunities. */
static void prefill_layer_major_cpu(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        const token_vec   * prompt,
        const float       * steering_dirs,
        float               steering_attn_scale,
        float               steering_ffn_scale) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t n_tok = (uint64_t)prompt->len;
    float *cur = xmalloc((size_t)n_tok * hc_dim * sizeof(cur[0]));
    float *next = xmalloc((size_t)n_tok * hc_dim * sizeof(next[0]));
    float *attn = xmalloc((size_t)n_tok * hc_dim * sizeof(attn[0]));
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));
    uint32_t ffn_batch = 128;
    const bool batched_attn = getenv("DS4_NO_BATCHED_ATTN") == NULL;
    const bool batched_ffn = getenv("DS4_BATCHED_FFN") != NULL;
    const bool parallel_ffn = getenv("DS4_PARALLEL_FFN") != NULL;
    const bool shared_batch_ffn = getenv("DS4_NO_SHARED_BATCH_FFN") == NULL;
    const char *batch_env = getenv("DS4_PREFILL_BATCH");
    ds4_cpu_decode_scratch decode_scratch;
    bool decode_scratch_ready = false;
    if (batch_env && batch_env[0]) {
        long v = strtol(batch_env, NULL, 10);
        if (v > 0 && v < 4096) ffn_batch = (uint32_t)v;
    }

    for (uint64_t t = 0; t < n_tok; t++) {
        embed_token_f16(model, weights, prompt->v[t], plain);
        hc_from_plain_embedding(cur + t * hc_dim, plain, DS4_N_EMBD, DS4_N_HC);
    }

    free(plain);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        fprintf(stderr, "ds4: prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
        fflush(stderr);

        if (batched_attn) {
            layer_attention_raw_swa_batch(attn,
                                          model,
                                          &weights->layer[il],
                                          &cache->layer[il],
                                          cur,
                                          (uint32_t)n_tok,
                                          il,
                                          0,
                                          steering_dirs,
                                          steering_attn_scale);

            if (batched_ffn) {
                for (uint64_t t = 0; t < n_tok; t += ffn_batch) {
                    uint32_t nb = (uint32_t)((n_tok - t) < ffn_batch ? (n_tok - t) : ffn_batch);
                    layer_ffn_batch(next + t * hc_dim,
                                    model,
                                    &weights->layer[il],
                                    attn + t * hc_dim,
                                    prompt->v + t,
                                    nb,
                                    il,
                                    steering_dirs,
                                    steering_ffn_scale);
                }
            } else if (shared_batch_ffn) {
                layer_ffn_shared_batch(next,
                                       model,
                                       &weights->layer[il],
                                       attn,
                                       prompt->v,
                                       (uint32_t)n_tok,
                                       il,
                                       steering_dirs,
                                       steering_ffn_scale);
            } else if (parallel_ffn) {
                layer_ffn_tokens_parallel(next,
                                          model,
                                          &weights->layer[il],
                                          attn,
                                          prompt->v,
                                          (uint32_t)n_tok,
                                          il,
                                          steering_dirs,
                                          steering_ffn_scale);
            } else {
                for (uint64_t t = 0; t < n_tok; t++) {
                    layer_ffn_one(next + t * hc_dim,
                                  model,
                                  &weights->layer[il],
                                  attn + t * hc_dim,
                                  il,
                                  prompt->v[t],
                                  steering_dirs,
                                  steering_ffn_scale,
                                  false);
                }
            }
        } else if (batched_ffn) {
            for (uint64_t t = 0; t < n_tok; t++) {
                layer_attention_raw_swa_one(attn + t * hc_dim,
                                            model,
                                            &weights->layer[il],
                                            &cache->layer[il],
                                            cur + t * hc_dim,
                                            il,
                                            (uint32_t)t,
                                            steering_dirs,
                                            steering_attn_scale);
            }

            for (uint64_t t = 0; t < n_tok; t += ffn_batch) {
                uint32_t nb = (uint32_t)((n_tok - t) < ffn_batch ? (n_tok - t) : ffn_batch);
                layer_ffn_batch(next + t * hc_dim,
                                model,
                                &weights->layer[il],
                                attn + t * hc_dim,
                                prompt->v + t,
                                nb,
                                il,
                                steering_dirs,
                                steering_ffn_scale);
            }
        } else {
            if (!decode_scratch_ready) {
                cpu_decode_scratch_init(&decode_scratch, (uint32_t)n_tok);
                decode_scratch_ready = true;
            }
            for (uint64_t t = 0; t < n_tok; t++) {
                layer_forward_raw_swa_one(next + t * hc_dim,
                                          model,
                                          &weights->layer[il],
                                          &cache->layer[il],
                                          cur + t * hc_dim,
                                          il,
                                          (uint32_t)t,
                                          prompt->v[t],
                                          steering_dirs,
                                          steering_attn_scale,
                                          steering_ffn_scale,
                                          &decode_scratch);
            }
        }

        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    kv_cache_finish_prefill_states(cache, (uint32_t)n_tok);

    if (logits) {
        output_logits_one(logits, model, weights, cur + (n_tok - 1) * hc_dim);
    }

    if (decode_scratch_ready) cpu_decode_scratch_free(&decode_scratch);
    free(next);
    free(cur);
    free(attn);
}

/* Diagnostic first-token layer without cache history: the token attends only
 * to itself, useful for checking a minimal end-to-end slice. */
static void layer_forward_self_one(
        float                   * out_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        int                       token) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    float *attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(attn_residual[0]));
    float *q = xmalloc((size_t)q_dim * sizeof(q[0]));
    float *kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = xmalloc((size_t)q_dim * sizeof(heads[0]));
    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    float *after_attn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_attn_hc[0]));
    float post[4];
    float comb[16];

    memcpy(attn_residual, inp_hc, (size_t)n_hc * DS4_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          attn_residual, attn_cur, post, comb);

    layer_attn_norm_one(attn_norm, model, layer, attn_cur);
    layer_q_projection_normed_one(model, layer, attn_norm, q);
    layer_kv_projection_normed_one(model, layer, attn_norm, kv);
    rope_tail_layer_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    rope_tail_layer_inplace(kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(kv, DS4_N_HEAD_DIM);

    layer_attention_one(heads, model, layer, q, kv);
    rope_tail_layer_inplace(heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);
    layer_grouped_out_one(attn_out, model, layer, heads);
    hc_post_one(after_attn_hc, attn_out, attn_residual, post, comb, DS4_N_EMBD, n_hc);

    layer_ffn_one(out_hc, model, layer, after_attn_hc, il, token,
                  NULL, 0.0f, false);

    free(after_attn_hc);
    free(attn_out);
    free(heads);
    free(kv);
    free(q);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}

static void forward_first_token_cpu(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_weights * weights,
        int                 token) {
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));
    float *cur = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(cur[0]));
    float *next = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(next[0]));

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cur, plain, DS4_N_EMBD, DS4_N_HC);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        layer_forward_self_one(next, model, &weights->layer[il], cur, il, 0, token);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    memcpy(out_hc, cur, (size_t)DS4_N_HC * DS4_N_EMBD * sizeof(out_hc[0]));

    free(next);
    free(cur);
    free(plain);
}

/* Collapse final HC streams into the ordinary embedding vector before the
 * output norm and vocabulary projection. */
static void output_hc_head_one(
        float             * out,
        const ds4_model   * model,
        const ds4_weights * weights,
        const float       * inp_hc) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * n_hc;
    float *flat = xmalloc((size_t)hc_dim * sizeof(flat[0]));
    float *pre = xmalloc((size_t)n_hc * sizeof(pre[0]));
    float *w = xmalloc((size_t)n_hc * sizeof(w[0]));

    rms_norm_no_weight(flat, inp_hc, hc_dim, DS4_RMS_EPS);
    matvec_f16(pre, model, weights->output_hc_fn, flat);

    const float *scale = tensor_data(model, weights->output_hc_scale);
    const float *base = tensor_data(model, weights->output_hc_base);
    for (uint32_t i = 0; i < n_hc; i++) {
        w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + DS4_HC_EPS;
    }

    hc_weighted_sum_one(out, inp_hc, w, DS4_N_EMBD, n_hc);

    free(w);
    free(pre);
    free(flat);
}

/* Final language-model head: HC collapse, RMSNorm, and Q8_0 vocab projection. */
static void output_logits_one(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        const float       * inp_hc) {
    float *embd = xmalloc((size_t)DS4_N_EMBD * sizeof(embd[0]));
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));

    output_hc_head_one(embd, model, weights, inp_hc);
    rms_norm_weight(norm, embd, tensor_data(model, weights->output_norm), DS4_N_EMBD, DS4_RMS_EPS);

    matvec_q8_0(logits, model, weights->output, norm);

    free(norm);
    free(embd);
}

/* Allocation-free logits head for CPU decode. */
static void output_logits_one_decode_scratch(
        float                  * logits,
        const ds4_model        * model,
        const ds4_weights      * weights,
        const float            * inp_hc,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * n_hc;

    rms_norm_no_weight(scratch->output_flat, inp_hc, hc_dim, DS4_RMS_EPS);
    matvec_f16(scratch->output_pre, model, weights->output_hc_fn, scratch->output_flat);

    const float *scale = tensor_data(model, weights->output_hc_scale);
    const float *base = tensor_data(model, weights->output_hc_base);
    for (uint32_t i = 0; i < n_hc; i++) {
        scratch->output_weights[i] = sigmoid_stable(scratch->output_pre[i] * scale[0] + base[i]) + DS4_HC_EPS;
    }

    hc_weighted_sum_one(scratch->output_embd, inp_hc, scratch->output_weights, DS4_N_EMBD, n_hc);
    rms_norm_weight(scratch->output_norm, scratch->output_embd,
                    tensor_data(model, weights->output_norm),
                    DS4_N_EMBD, DS4_RMS_EPS);
    matvec_q8_0_decode_scratch(logits, model, weights->output, scratch->output_norm, scratch);
}

#ifndef DS4_NO_GPU
static int sample_argmax(const float *logits, uint32_t n_vocab);

/* =========================================================================
 * Metal Reference Comparison Helpers.
 * =========================================================================
 *
 * These small scalar helpers are used only by diagnostics that compare the C
 * reference path with the Metal executor.
 */

static float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float max_diff = 0.0f;
    for (uint64_t i = 0; i < n; i++) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

static float rms_abs_diff(const float *a, const float *b, uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        ss += d * d;
    }
    return n ? (float)sqrt(ss / (double)n) : 0.0f;
}

static uint64_t argmax_f32(const float *x, uint64_t n) {
    uint64_t best = 0;
    for (uint64_t i = 1; i < n; i++) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}

#endif

static void print_vec_stats(const char *name, const float *x, uint64_t n) {
    float minv = DS4_POS_INF;
    float maxv = DS4_NEG_INF;
    double ss = 0.0;

    for (uint64_t i = 0; i < n; i++) {
        const float v = x[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        ss += (double)v * v;
    }

    printf("%s: min=%g max=%g rms=%g\n",
        name, minv, maxv, sqrt(ss / (double)n));
}

#ifndef DS4_NO_GPU
/* =========================================================================
 * Metal Release Graph State.
 * =========================================================================
 *
 * The release Metal executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

typedef struct {
    /* One-token decode tensors.  These stay allocated for the life of a
     * session; a generated token enters as an embedding in cur_hc and leaves as
     * logits after all 43 layers update their raw/compressed/indexer caches. */
    ds4_gpu_tensor *cur_hc;
    ds4_gpu_tensor *flat_hc;
    ds4_gpu_tensor *hc_mix;
    ds4_gpu_tensor *hc_split;
    ds4_gpu_tensor *hc_pre;
    ds4_gpu_tensor *hc_post;
    ds4_gpu_tensor *hc_comb;
    ds4_gpu_tensor *attn_cur;
    ds4_gpu_tensor *attn_norm;
    ds4_gpu_tensor *qr;
    ds4_gpu_tensor *qr_norm;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv_raw;
    ds4_gpu_tensor *kv;

    /* Persistent KV state.  Raw KV is a sliding-window ring per layer.  Ratio-4
     * layers also keep an indexer-compressed cache; ratio-128 layers keep only
     * the attention-compressed cache.  The small state tensors are compressor
     * frontiers for the next compressed row, so they must be snapshotted with
     * the row counters whenever a checkpoint is saved or partially rewound. */
    ds4_gpu_tensor *layer_raw_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *layer_attn_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *layer_index_comp_cache[DS4_N_LAYER];
    ds4_gpu_tensor *layer_index_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *layer_index_state_score[DS4_N_LAYER];

    /* Speculative decoding scratch.  MTP is allowed to mutate graph state only
     * if the target verifier can either commit it or restore the saved
     * frontiers.  The prefix1 buffers are the cheap partial-accept state for the
     * common N=2 case. */
    ds4_gpu_tensor *spec_attn_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *spec_attn_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *spec_index_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *spec_index_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_kv[DS4_N_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_score[DS4_N_LAYER];
    ds4_gpu_tensor *spec_logits;
    uint32_t layer_n_comp[DS4_N_LAYER];
    uint32_t layer_n_index_comp[DS4_N_LAYER];
    uint32_t spec_prefix1_n_comp[DS4_N_LAYER];
    uint32_t spec_prefix1_n_index_comp[DS4_N_LAYER];
    bool spec_capture_prefix1;
    uint32_t raw_cap;
    /* Maximum compressed-row capacity across layers.  Shared work buffers use
     * this worst-case size because ratio-4 indexer layers can still reach it. */
    uint32_t comp_cap;
    /* Persistent compressed caches are per layer, so size them from the actual
     * layer compression ratio instead of pessimistically using the ratio-4 cap
     * for every ratio-128 layer. */
    uint32_t layer_comp_cap[DS4_N_LAYER];

    /* Per-layer work tensors.  They are reused in place by every layer instead
     * of allocating a generic graph arena.  This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    ds4_gpu_tensor *comp_kv_cur;
    ds4_gpu_tensor *comp_sc_cur;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor *indexer_scores;
    ds4_gpu_tensor *comp_mask;
    ds4_gpu_tensor *comp_selected;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_low;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn_hc;
    ds4_gpu_tensor *ffn_cur;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared_out;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_mid;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *routed_out;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *after_ffn_hc;
    ds4_gpu_tensor *output_pre;
    ds4_gpu_tensor *output_weights;
    ds4_gpu_tensor *output_embd;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;

    /* Optional MTP model state.  It has its own raw cache because the drafter
     * runs on speculative future tokens; target KV state is updated only after
     * verification accepts draft tokens. */
    ds4_gpu_tensor *mtp_embed;
    ds4_gpu_tensor *mtp_enorm;
    ds4_gpu_tensor *mtp_eproj;
    ds4_gpu_tensor *mtp_eproj_hc;
    ds4_gpu_tensor *mtp_hnorm_hc;
    ds4_gpu_tensor *mtp_hproj_hc;
    ds4_gpu_tensor *mtp_input_hc;
    ds4_gpu_tensor *mtp_state_hc;
    ds4_gpu_tensor *mtp_next_hc;
    ds4_gpu_tensor *mtp_raw_cache;
    uint32_t mtp_n_raw;
    uint32_t prefill_cap;
    uint32_t raw_window;

    /* Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    ds4_gpu_tensor *prefill_tokens;
    ds4_gpu_tensor *batch_cur_hc;
    ds4_gpu_tensor *batch_next_hc;
    ds4_gpu_tensor *batch_flat_hc;
    ds4_gpu_tensor *batch_hc_mix;
    ds4_gpu_tensor *batch_hc_split;
    ds4_gpu_tensor *batch_attn_cur;
    ds4_gpu_tensor *batch_attn_norm;
    ds4_gpu_tensor *batch_qr;
    ds4_gpu_tensor *batch_qr_norm;
    ds4_gpu_tensor *batch_q;
    ds4_gpu_tensor *batch_kv_raw;
    ds4_gpu_tensor *batch_kv;
    ds4_gpu_tensor *batch_comp_kv;
    ds4_gpu_tensor *batch_comp_sc;
    ds4_gpu_tensor *batch_indexer_q;
    ds4_gpu_tensor *batch_indexer_weights;
    ds4_gpu_tensor *batch_heads;
    ds4_gpu_tensor *batch_attn_low;
    ds4_gpu_tensor *batch_attn_out;
    ds4_gpu_tensor *batch_group_tmp;
    ds4_gpu_tensor *batch_low_tmp;
    ds4_gpu_tensor *batch_after_attn_hc;
    ds4_gpu_tensor *batch_ffn_cur;
    ds4_gpu_tensor *batch_ffn_norm;
    ds4_gpu_tensor *batch_shared_gate;
    ds4_gpu_tensor *batch_shared_up;
    ds4_gpu_tensor *batch_shared_mid;
    ds4_gpu_tensor *batch_shared_out;
    ds4_gpu_tensor *batch_router_logits;
    ds4_gpu_tensor *batch_router_probs;
    ds4_gpu_tensor *batch_router_selected;
    ds4_gpu_tensor *batch_router_weights;
    ds4_gpu_tensor *batch_routed_gate;
    ds4_gpu_tensor *batch_routed_up;
    ds4_gpu_tensor *batch_routed_mid;
    ds4_gpu_tensor *batch_routed_down;
    ds4_gpu_tensor *batch_routed_out;
    bool batch_routed_mid_is_f16;
    ds4_gpu_tensor *batch_ffn_out;
    bool materialize_ffn_out;
    ds4_gpu_tensor *directional_steering_dirs;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    bool quality;
    bool mtp_enabled;
} ds4_gpu_graph;

/* Release every Metal tensor owned by the whole-model graph runtime. */
static void metal_graph_free(ds4_gpu_graph *g) {
    ds4_gpu_tensor_free(g->directional_steering_dirs);
    ds4_gpu_tensor_free(g->batch_ffn_out);
    ds4_gpu_tensor_free(g->batch_routed_out);
    ds4_gpu_tensor_free(g->batch_routed_down);
    ds4_gpu_tensor_free(g->batch_routed_mid);
    ds4_gpu_tensor_free(g->batch_routed_up);
    ds4_gpu_tensor_free(g->batch_routed_gate);
    ds4_gpu_tensor_free(g->batch_router_weights);
    ds4_gpu_tensor_free(g->batch_router_selected);
    ds4_gpu_tensor_free(g->batch_router_probs);
    ds4_gpu_tensor_free(g->batch_router_logits);
    ds4_gpu_tensor_free(g->batch_shared_out);
    ds4_gpu_tensor_free(g->batch_shared_mid);
    ds4_gpu_tensor_free(g->batch_shared_up);
    ds4_gpu_tensor_free(g->batch_shared_gate);
    ds4_gpu_tensor_free(g->batch_ffn_norm);
    ds4_gpu_tensor_free(g->batch_ffn_cur);
    ds4_gpu_tensor_free(g->batch_after_attn_hc);
    ds4_gpu_tensor_free(g->batch_low_tmp);
    ds4_gpu_tensor_free(g->batch_group_tmp);
    ds4_gpu_tensor_free(g->batch_attn_out);
    ds4_gpu_tensor_free(g->batch_attn_low);
    ds4_gpu_tensor_free(g->batch_heads);
    ds4_gpu_tensor_free(g->batch_indexer_weights);
    ds4_gpu_tensor_free(g->batch_indexer_q);
    ds4_gpu_tensor_free(g->batch_comp_sc);
    ds4_gpu_tensor_free(g->batch_comp_kv);
    ds4_gpu_tensor_free(g->batch_kv);
    ds4_gpu_tensor_free(g->batch_kv_raw);
    ds4_gpu_tensor_free(g->batch_q);
    ds4_gpu_tensor_free(g->batch_qr_norm);
    ds4_gpu_tensor_free(g->batch_qr);
    ds4_gpu_tensor_free(g->batch_attn_norm);
    ds4_gpu_tensor_free(g->batch_attn_cur);
    ds4_gpu_tensor_free(g->batch_hc_split);
    ds4_gpu_tensor_free(g->batch_hc_mix);
    ds4_gpu_tensor_free(g->batch_flat_hc);
    ds4_gpu_tensor_free(g->batch_next_hc);
    ds4_gpu_tensor_free(g->batch_cur_hc);
    ds4_gpu_tensor_free(g->prefill_tokens);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->mtp_raw_cache);
    ds4_gpu_tensor_free(g->mtp_next_hc);
    ds4_gpu_tensor_free(g->mtp_state_hc);
    ds4_gpu_tensor_free(g->mtp_input_hc);
    ds4_gpu_tensor_free(g->mtp_hproj_hc);
    ds4_gpu_tensor_free(g->mtp_hnorm_hc);
    ds4_gpu_tensor_free(g->mtp_eproj_hc);
    ds4_gpu_tensor_free(g->mtp_eproj);
    ds4_gpu_tensor_free(g->mtp_enorm);
    ds4_gpu_tensor_free(g->mtp_embed);
    ds4_gpu_tensor_free(g->spec_logits);
    ds4_gpu_tensor_free(g->output_norm);
    ds4_gpu_tensor_free(g->output_embd);
    ds4_gpu_tensor_free(g->output_weights);
    ds4_gpu_tensor_free(g->output_pre);
    ds4_gpu_tensor_free(g->after_ffn_hc);
    ds4_gpu_tensor_free(g->ffn_out);
    ds4_gpu_tensor_free(g->routed_out);
    ds4_gpu_tensor_free(g->routed_down);
    ds4_gpu_tensor_free(g->routed_mid);
    ds4_gpu_tensor_free(g->routed_up);
    ds4_gpu_tensor_free(g->routed_gate);
    ds4_gpu_tensor_free(g->router_weights);
    ds4_gpu_tensor_free(g->router_selected);
    ds4_gpu_tensor_free(g->router_probs);
    ds4_gpu_tensor_free(g->router_logits);
    ds4_gpu_tensor_free(g->shared_out);
    ds4_gpu_tensor_free(g->shared_mid);
    ds4_gpu_tensor_free(g->shared_up);
    ds4_gpu_tensor_free(g->shared_gate);
    ds4_gpu_tensor_free(g->ffn_norm);
    ds4_gpu_tensor_free(g->ffn_cur);
    ds4_gpu_tensor_free(g->after_attn_hc);
    ds4_gpu_tensor_free(g->attn_out);
    ds4_gpu_tensor_free(g->attn_low);
    ds4_gpu_tensor_free(g->heads);
    ds4_gpu_tensor_free(g->comp_sc_cur);
    ds4_gpu_tensor_free(g->comp_kv_cur);
    ds4_gpu_tensor_free(g->comp_mask);
    ds4_gpu_tensor_free(g->comp_selected);
    ds4_gpu_tensor_free(g->indexer_scores);
    ds4_gpu_tensor_free(g->indexer_weights);
    ds4_gpu_tensor_free(g->indexer_q);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_index_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_index_state_score[il]);
    }
    ds4_gpu_tensor_free(g->kv);
    ds4_gpu_tensor_free(g->kv_raw);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->qr_norm);
    ds4_gpu_tensor_free(g->qr);
    ds4_gpu_tensor_free(g->attn_norm);
    ds4_gpu_tensor_free(g->attn_cur);
    ds4_gpu_tensor_free(g->hc_comb);
    ds4_gpu_tensor_free(g->hc_post);
    ds4_gpu_tensor_free(g->hc_pre);
    ds4_gpu_tensor_free(g->hc_split);
    ds4_gpu_tensor_free(g->hc_mix);
    ds4_gpu_tensor_free(g->flat_hc);
    ds4_gpu_tensor_free(g->cur_hc);
    memset(g, 0, sizeof(*g));
}

static bool metal_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n) {
    return ds4_gpu_tensor_fill_f32(t, v, n) != 0;
}

/* =========================================================================
 * Directional Steering.
 * =========================================================================
 *
 * A steering file contains one normalized 4096-wide direction per layer.  When
 * enabled, the Metal graph edits selected block outputs in-place:
 *
 *     y = y - scale * v * dot(v, y)
 *
 * Positive scales remove the represented direction from the activation.
 * Negative scales add it.  This is deliberately explicit and opt-in; with zero
 * scales, the release graph does not allocate the direction tensor and follows
 * the normal inference path.
 */

static bool metal_graph_load_directional_steering(
        ds4_gpu_graph *g,
        const char      *path,
        float            attn_scale,
        float            ffn_scale) {
    if (attn_scale == 0.0f && ffn_scale == 0.0f) return true;

    if (!path || !path[0]) {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        return false;
    }

    const uint64_t n = (uint64_t)DS4_N_LAYER * DS4_N_EMBD;
    float *dirs = xmalloc((size_t)n * sizeof(dirs[0]));
    bool ok = read_f32_binary_file(path, dirs, n);
    if (ok) {
        g->directional_steering_dirs = ds4_gpu_tensor_alloc(n * sizeof(dirs[0]));
        ok = g->directional_steering_dirs != NULL &&
             ds4_gpu_tensor_write(g->directional_steering_dirs, 0, dirs, n * sizeof(dirs[0])) != 0;
    }
    free(dirs);

    if (!ok) {
        fprintf(stderr, "ds4: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    g->directional_steering_attn_scale = attn_scale;
    g->directional_steering_ffn_scale = ffn_scale;
    fprintf(stderr, "ds4: directional steering enabled: %s attn=%g ffn=%g\n",
            path, (double)attn_scale, (double)ffn_scale);
    return true;
}

static bool metal_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_attn_scale != 0.0f;
}

static bool metal_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_ffn_scale != 0.0f;
}

static bool metal_graph_apply_directional_steering(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows,
        float             scale) {
    if (!g || !g->directional_steering_dirs || scale == 0.0f) return true;
    return ds4_gpu_directional_steering_project_tensor(x,
                                            g->directional_steering_dirs,
                                            il,
                                            DS4_N_EMBD,
                                            rows,
                                            scale) != 0;
}

static bool metal_graph_apply_directional_steering_attn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return metal_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_attn_scale : 0.0f);
}

static bool metal_graph_apply_directional_steering_ffn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return metal_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_ffn_scale : 0.0f);
}

/* =========================================================================
 * Metal Diagnostic Dump Hooks.
 * =========================================================================
 *
 * The release path calls these after important stages, but they are no-ops
 * unless DS4_METAL_GRAPH_DUMP_PREFIX is set.  Dumping synchronizes and restarts
 * the command batch, so it is intentionally isolated here.
 */

static bool metal_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!prefix || !prefix[0]) return false;

    const char *name_env = getenv("DS4_METAL_GRAPH_DUMP_NAME");
    if (name_env && name_env[0] && strstr(name_env, name) == NULL) return false;

    const char *layer_env = getenv("DS4_METAL_GRAPH_DUMP_LAYER");
    if (layer_env && layer_env[0] && strcmp(layer_env, "all") != 0 &&
        (uint32_t)strtoul(layer_env, NULL, 10) != il) return false;

    const char *pos_env = getenv("DS4_METAL_GRAPH_DUMP_POS");
    if (pos_env && pos_env[0] && (uint32_t)strtoul(pos_env, NULL, 10) != pos) return false;

    return true;
}

static void metal_graph_debug_dump_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!t || n_f32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = xmalloc((size_t)n_f32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static void metal_graph_debug_dump_i32_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!t || n_i32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.i32", prefix, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static bool metal_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    return metal_graph_directional_steering_ffn_enabled(g) ||
           g->materialize_ffn_out ||
           metal_graph_debug_wants("ffn_out", il, pos);
}

static bool metal_graph_ensure_ffn_out(ds4_gpu_graph *g) {
    if (!g->ffn_out) {
        g->ffn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    }
    return g->ffn_out != NULL;
}

static bool metal_graph_ensure_batch_ffn_out(ds4_gpu_graph *g) {
    if (!g->batch_ffn_out) {
        g->batch_ffn_out = ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out != NULL;
}

/* =========================================================================
 * Metal Release Graph Allocation.
 * ========================================================================= */

/* Allocate the Metal graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
static bool metal_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_mtp) {
    memset(g, 0, sizeof(*g));
    g->mtp_enabled = enable_mtp;
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    g->raw_cap = raw_cap;
    g->raw_window = raw_window;
    g->prefill_cap = prefill_cap;
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    g->comp_cap = ctx_size / min_ratio + 2u;
    if (g->comp_cap < 2u) g->comp_cap = 2u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            g->layer_comp_cap[il] = 0;
        } else {
            g->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (g->layer_comp_cap[il] < 2u) g->layer_comp_cap[il] = 2u;
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t vocab_dim = weights->output->dim[1];
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = prefill_cap;

    g->cur_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->flat_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->hc_mix = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_pre = ds4_gpu_tensor_view(g->hc_split, 0, (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_post = ds4_gpu_tensor_view(g->hc_split,
                                       (uint64_t)DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_comb = ds4_gpu_tensor_view(g->hc_split,
                                       2ull * DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    g->attn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->qr = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->q = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->kv_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_raw_cache[il] = ds4_gpu_tensor_alloc((uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            g->layer_attn_comp_cache[il] = ds4_gpu_tensor_alloc((uint64_t)g->layer_comp_cap[il] * DS4_N_HEAD_DIM * sizeof(float));
            g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            if (enable_mtp) {
                g->spec_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_prefix1_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_prefix1_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                g->layer_index_comp_cache[il] = ds4_gpu_tensor_alloc((uint64_t)g->layer_comp_cap[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                if (enable_mtp) {
                    g->spec_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_prefix1_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_prefix1_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    g->comp_kv_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->indexer_q = ds4_gpu_tensor_alloc(indexer_q_dim * sizeof(float));
    g->indexer_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
    g->indexer_scores = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_mask = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * pc * sizeof(float));
    g->comp_selected = ds4_gpu_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    g->heads = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->attn_low = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    g->attn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_attn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->ffn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->shared_gate = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_up = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->router_logits = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_probs = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_selected = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(int));
    g->router_weights = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(float));
    g->routed_gate = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_up = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_mid = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_down = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->routed_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_ffn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->output_pre = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc(vocab_dim * sizeof(float));
    /*
     * MTP is deliberately outside the normal graph footprint.  A session that
     * does not opt in with --mtp must allocate and execute exactly the same
     * buffers as the plain decoder: no support-model mapping, no draft logits,
     * and no MTP scratch hidden behind otherwise unused tensors.
     */
    if (enable_mtp) {
        g->mtp_embed = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_enorm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hnorm_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_input_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_state_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_next_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_raw_cache = ds4_gpu_tensor_alloc((uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        g->spec_logits = ds4_gpu_tensor_alloc((uint64_t)16 * DS4_N_VOCAB * sizeof(float));
        g->mtp_n_raw = 0;
    }

    g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_flat_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_attn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_attn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_q = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_kv_raw = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_kv = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_comp_kv = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_comp_sc = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_indexer_q = ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
    g->batch_indexer_weights = ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
    g->batch_heads = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_attn_low = ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
    g->batch_attn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_group_tmp = ds4_gpu_tensor_alloc(pc * group_dim * sizeof(float));
    g->batch_low_tmp = ds4_gpu_tensor_alloc(pc * DS4_N_LORA_O * sizeof(float));
    g->batch_after_attn_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_ffn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_shared_gate = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_up = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_mid = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_router_logits = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_probs = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_selected = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
    g->batch_router_weights = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_gate = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_up = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->batch_routed_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_mtp ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL &&
                               g->spec_prefix1_attn_state_kv[il] != NULL &&
                               g->spec_prefix1_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_index_comp_cache[il] != NULL &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_mtp ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL &&
                               g->spec_prefix1_index_state_kv[il] != NULL &&
                               g->spec_prefix1_index_state_score[il] != NULL));
        }
    }

    const bool ok = state_init_ok && layer_cache_ok &&
                    g->cur_hc && g->flat_hc && g->hc_mix && g->hc_split &&
                    g->hc_pre && g->hc_post && g->hc_comb &&
                    g->attn_cur && g->attn_norm && g->qr && g->qr_norm &&
                    g->q && g->kv_raw && g->kv &&
                    g->comp_kv_cur && g->comp_sc_cur &&
                    g->indexer_q && g->indexer_weights && g->indexer_scores &&
                    g->comp_mask && g->comp_selected &&
                    g->heads && g->attn_low && g->attn_out &&
                    g->after_attn_hc && g->ffn_cur && g->ffn_norm &&
                    g->shared_gate && g->shared_up && g->shared_mid &&
                    g->shared_out &&
                    g->router_logits && g->router_probs && g->router_selected && g->router_weights &&
                    g->routed_gate && g->routed_up && g->routed_mid &&
                    g->routed_down && g->routed_out &&
                    g->after_ffn_hc &&
                    g->output_pre && g->output_weights && g->output_embd &&
                    g->output_norm && g->logits &&
                    (!enable_mtp ||
                     (g->mtp_embed && g->mtp_enorm && g->mtp_eproj &&
                      g->mtp_eproj_hc && g->mtp_hnorm_hc && g->mtp_hproj_hc &&
                      g->mtp_input_hc && g->mtp_state_hc && g->mtp_next_hc &&
                      g->mtp_raw_cache && g->spec_logits)) &&
                    g->prefill_tokens &&
                    g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
                    g->batch_hc_mix && g->batch_hc_split &&
                    g->batch_attn_cur && g->batch_attn_norm &&
                    g->batch_qr && g->batch_qr_norm && g->batch_q &&
                    g->batch_kv_raw && g->batch_kv &&
                    g->batch_comp_kv && g->batch_comp_sc &&
                    g->batch_indexer_q && g->batch_indexer_weights &&
                    g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
                    g->batch_group_tmp && g->batch_low_tmp && g->batch_after_attn_hc &&
                    g->batch_ffn_cur && g->batch_ffn_norm &&
                    g->batch_shared_gate && g->batch_shared_up &&
                    g->batch_shared_mid && g->batch_shared_out &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_gate && g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out;
    if (!ok) metal_graph_free(g);
    return ok;
}

static bool metal_graph_alloc(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer) {
    return metal_graph_alloc_raw_cap(g, weights, layer, DS4_N_SWA, DS4_N_SWA, 1, false);
}

static uint32_t metal_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}

static uint32_t metal_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}

/* Capture the verifier prefix after the first speculative token.
 *
 * Exact MTP speculation is only profitable if partial accepts are cheap.  The
 * target verifier computes two draft tokens together; if only the first token
 * is accepted, replaying a one-token verifier throws away most of the gain.
 * For compressed-attention layers the mutable frontier is just the small
 * compressor state plus append counters, so we save that prefix-1 state while
 * the N=2 verifier is already stepping the compressor token by token.
 *
 * Raw SWA rows are not captured here.  This graph uses a raw ring larger than
 * the 128-token logical SWA window, so writing speculative future rows does
 * not evict visible raw rows.  If the raw cache is ever reduced to a strict
 * 128-row ring, speculative raw rows must become shadow rows and be copied
 * into the ring only on commit. */
static bool metal_graph_capture_prefix1_attn_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_attn_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
    g->spec_prefix1_n_comp[il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_kv[il], 0,
                                 g->layer_attn_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_score[il], 0,
                                 g->layer_attn_state_score[il], 0, bytes) != 0;
}

static bool metal_graph_capture_prefix1_index_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_index_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
    g->spec_prefix1_n_index_comp[il] = g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_index_state_kv[il], 0,
                                 g->layer_index_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_index_state_score[il], 0,
                                 g->layer_index_state_score[il], 0, bytes) != 0;
}

static uint32_t metal_graph_decode_indexer_top_k(const ds4_gpu_graph *g) {
    (void)g;
    return DS4_N_INDEXER_TOP_K;
}

/* =========================================================================
 * Metal Decode Release Helpers and Reference Fallbacks.
 * =========================================================================
 *
 * The normal generation path uses the fused helpers below.  The older unfused
 * kernels remain available as diagnostic reference paths selected only by the
 * DS4_METAL_DISABLE_*_FUSION environment switches.
 */

static bool metal_graph_env_flag(const char *name, int *cache) {
    if (*cache == -1) {
        const char *env = getenv(name);
        *cache = env && env[0] && strcmp(env, "0") != 0;
    }
    return *cache != 0;
}

static bool metal_graph_use_reference_hc_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_FUSION", &cache);
}

static bool metal_graph_use_reference_kv_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_KV_FUSION", &cache);
}

static bool metal_graph_use_reference_qkv_norm(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_QKV_NORM_FUSION", &cache);
}

static bool metal_graph_use_reference_compressor_pair_proj(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ", &cache);
}

static bool metal_graph_use_reference_hc_norm_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_NORM_FUSION", &cache);
}

static bool metal_graph_use_reference_shared_down_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION", &cache);
}

static bool metal_graph_use_reference_attn_out_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_ATTN_OUT_HC_FUSION", &cache);
}

static bool metal_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    if (metal_graph_use_reference_hc_decode()) {
        return ds4_gpu_hc_split_sinkhorn_tensor(split,
                                                  mix,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
               ds4_gpu_hc_weighted_sum_tensor(out,
                                                 residual_hc,
                                                 split,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    }

    return ds4_gpu_hc_split_weighted_sum_tensor(out,
                                                  split,
                                                  mix,
                                                  residual_hc,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0;
}

static bool metal_graph_decode_kv_store(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row) {
    if (metal_graph_use_reference_kv_decode()) {
        return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0 &&
               ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, DS4_N_HEAD_DIM) != 0;
    }

    return ds4_gpu_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             DS4_N_HEAD_DIM,
                                             DS4_N_ROT) != 0;
}

/* Encode one DS4 decode layer on Metal.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
static bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);
static bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);
static bool metal_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

static bool metal_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const bool qkv_rms_fused = !metal_graph_use_reference_qkv_norm();

    bool ok = true;
    const bool decode_stage_profile = getenv("DS4_METAL_DECODE_STAGE_PROFILE") != NULL;
    double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_attn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    const bool fuse_hc_norm =
        !metal_graph_use_reference_hc_decode() &&
        !metal_graph_use_reference_hc_norm_decode();
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->attn_cur,
                                                         g->attn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->cur_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_attn_scale->abs_offset,
                                                         layer->hc_attn_base->abs_offset,
                                                         layer->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
    } else if (ok) {
        ok = metal_graph_decode_hc_pre(g->attn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->cur_hc,
                                       model,
                                       layer->hc_attn_scale->abs_offset,
                                       layer->hc_attn_base->abs_offset);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre", g->attn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm, g->attn_cur,
                                                                   model->map, model->size,
                                                                   layer->attn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("attn_norm", g->attn_norm, DS4_N_EMBD, il, pos);
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->qr, model->map, model->size,
                                              layer->attn_q_a->abs_offset,
                                              DS4_N_EMBD, q_rank,
                                              g->attn_norm, 1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora", g->qr, q_rank, il, pos);
    }
    if (qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->qr_norm,
                                                             g->qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->kv,
                                                             g->kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             1,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->qr_norm, g->qr,
                                                      model->map, model->size,
                                                      layer->attn_q_a_norm->abs_offset,
                                                      (uint32_t)q_rank, DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora_norm", g->qr_norm, q_rank, il, pos);
    }
    if (qkv_rms_fused && ok) {
        metal_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->q, model->map, model->size,
                                              layer->attn_q_b->abs_offset,
                                              q_rank, q_dim,
                                              g->qr_norm, 1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("Qraw", g->q, q_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("Qnorm", g->q, q_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("q_path");
    if (ok) {
        metal_graph_debug_dump_tensor("Qcur", g->q, q_dim, il, pos);
    }
    if (!qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->kv, g->kv_raw,
                                                      model->map, model->size,
                                                      layer->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->kv, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("KVrope", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    /* RoPE stays as the exact standalone kernel above.  The decode fusion
     * starts after that, where FP8 KV quantization and raw-cache storage can
     * share one pass without changing the trigonometric path. */
    if (ok) ok = metal_graph_decode_kv_store(g->kv, raw_cache, raw_cap, raw_row);
    DS4_METAL_PROFILE_DECODE_STAGE("kv_path");
    if (ok) {
        metal_graph_debug_dump_tensor("KVcur", g->kv, DS4_N_HEAD_DIM, il, pos);
    }

    uint32_t n_comp = 0;
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    double decode_index_stage_t0 = 0.0;
    const bool decode_index_stage_profile = getenv("DS4_METAL_INDEXER_STAGE_PROFILE") != NULL;
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !layer->attn_compressor_ape || !layer->attn_compressor_norm ||
            layer->attn_compressor_kv->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_gate->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != comp_width ||
            layer->attn_compressor_gate->dim[1] != comp_width) {
            fprintf(stderr, "ds4: Metal graph compressor expects paired F16 compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        if (ok && !metal_graph_use_reference_compressor_pair_proj()) {
            ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                  g->comp_sc_cur,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_compressor_kv->abs_offset,
                                                  layer->attn_compressor_gate->abs_offset,
                                                  DS4_N_EMBD,
                                                  comp_width,
                                                  g->attn_norm,
                                                  1) != 0;
        } else {
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_kv_cur, model->map, model->size,
                                                     layer->attn_compressor_kv->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     g->attn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_sc_cur, model->map, model->size,
                                                     layer->attn_compressor_gate->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     g->attn_norm, 1) != 0;
        }
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                        g->comp_sc_cur,
                                                        g->layer_attn_state_kv[il],
                                                        g->layer_attn_state_score[il],
                                                        g->layer_attn_comp_cache[il],
                                                        model->map,
                                                        model->size,
                                                        layer->attn_compressor_ape->abs_offset,
                                                        layer->attn_compressor_ape->type,
                                                        layer->attn_compressor_norm->abs_offset,
                                                        layer->attn_compressor_norm->type,
                                                        DS4_N_HEAD_DIM,
                                                        ratio,
                                                        pos,
                                                        comp_row,
                                                        DS4_N_ROT,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW,
                                                        DS4_RMS_EPS) != 0;
        if (ok && emit) {
            ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
                    g->layer_attn_comp_cache[il],
                    (uint64_t)comp_row * DS4_N_HEAD_DIM * sizeof(float),
                    (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
            if (!comp_row_view) {
                ok = false;
            } else {
                ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
                if (ok) {
                    metal_graph_debug_dump_tensor("KVcompress", comp_row_view, DS4_N_HEAD_DIM, il, pos);
                }
                ds4_gpu_tensor_free(comp_row_view);
            }
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                layer->indexer_compressor_kv->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_gate->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr, "ds4: Metal graph indexer compressor expects paired F16 projections\n");
                ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && !metal_graph_use_reference_compressor_pair_proj()) {
                ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                      g->comp_sc_cur,
                                                      model->map,
                                                      model->size,
                                                      layer->indexer_compressor_kv->abs_offset,
                                                      layer->indexer_compressor_gate->abs_offset,
                                                      DS4_N_EMBD,
                                                      index_width,
                                                      g->attn_norm,
                                                      1) != 0;
            } else {
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_kv_cur, model->map, model->size,
                                                         layer->indexer_compressor_kv->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         g->attn_norm, 1) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->comp_sc_cur, model->map, model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         g->attn_norm, 1) != 0;
            }
            const uint32_t index_row = g->layer_n_index_comp[il];
            if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                            g->comp_sc_cur,
                                                            g->layer_index_state_kv[il],
                                                            g->layer_index_state_score[il],
                                                            g->layer_index_comp_cache[il],
                                                            model->map,
                                                            model->size,
                                                            layer->indexer_compressor_ape->abs_offset,
                                                            layer->indexer_compressor_ape->type,
                                                            layer->indexer_compressor_norm->abs_offset,
                                                            layer->indexer_compressor_norm->type,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            index_row,
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS) != 0;
            if (ok && emit) g->layer_n_index_comp[il]++;
            const uint32_t decode_top_k = metal_graph_decode_indexer_top_k(g);
            if (ok && g->layer_n_comp[il] > decode_top_k) {
                const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    layer->indexer_attn_q_b->type != DS4_TENSOR_F16 ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                    fprintf(stderr, "ds4: Metal graph indexer q projection expects F16 weights\n");
                    ok = false;
                }
                if (ok && (!layer->indexer_proj ||
                           layer->indexer_proj->type != DS4_TENSOR_F16 ||
                           layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                           layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                    fprintf(stderr, "ds4: Metal graph indexer weight projection expects F16 weights\n");
                    ok = false;
                }
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->indexer_q, model->map, model->size,
                                                         layer->indexer_attn_q_b->abs_offset,
                                                         q_rank, indexer_q_dim,
                                                         g->qr_norm, 1) != 0;
                if (ok) ok = ds4_gpu_rope_tail_tensor(g->indexer_q, 1,
                                                        DS4_N_INDEXER_HEAD,
                                                        DS4_N_INDEXER_HEAD_DIM,
                                                        DS4_N_ROT,
                                                        pos,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        false,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(g->indexer_weights, model->map, model->size,
                                                         layer->indexer_proj->abs_offset,
                                                         DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                                                         g->attn_norm, 1) != 0;
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                                g->indexer_q,
                                                                g->indexer_weights,
                                                                g->layer_index_comp_cache[il],
                                                                g->layer_n_index_comp[il],
                                                                DS4_N_INDEXER_HEAD,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                index_scale) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("decode_score",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                           g->indexer_scores,
                                                           g->layer_n_index_comp[il],
                                                           1,
                                                           decode_top_k) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("decode_topk",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                /* Decode used to materialize a dense compressed-row mask and
                 * call the generic gathered FlashAttention wrapper below.
                 * That wrapper scans every compressed row and rejects long
                 * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                 * attention is sparse after indexer top-k, so use the private
                 * indexed attention kernel instead: it scans only SWA raw rows
                 * plus the selected compressed rows, matching prefill and
                 * avoiding the long-context decode failure. */
                if (ok) {
                    comp_selected = g->comp_selected;
                    n_selected = decode_top_k < g->layer_n_index_comp[il]
                        ? decode_top_k
                        : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache = g->layer_attn_comp_cache[il];
    }
    DS4_METAL_PROFILE_DECODE_STAGE("compressor_indexer");

    if (ok) {
        const uint32_t raw_start = metal_graph_raw_start_for_span(g, pos, n_raw);
        if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    g->heads,
                    model->map,
                    model->size,
                    layer->attn_sinks->abs_offset,
                    g->q,
                    raw_cache,
                    comp_cache,
                    comp_selected,
                    1,
                    pos,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    g->raw_window,
                    ds4_layer_compress_ratio(il),
                    DS4_N_HEAD,
                    DS4_N_HEAD_DIM) != 0;
            if (ok && decode_index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else {
            ok = ds4_gpu_attention_decode_heads_tensor(g->heads,
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset,
                                                         g->q, raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         n_comp,
                                                         NULL,
                                                         0,
                                                         DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        }
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attention");
    if (ok) {
        metal_graph_debug_dump_tensor("kqv_out", g->heads, q_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->heads,
                                            1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("kqv_back", g->heads, q_dim, il, pos);
    }
    const bool fuse_attn_out_hc =
        !metal_graph_directional_steering_attn_enabled(g) &&
        !metal_graph_use_reference_attn_out_hc();
    if (ok && fuse_attn_out_hc) {
        ok = ds4_gpu_attention_output_low_q8_tensor(g->attn_low,
                                                      model->map,
                                                      model->size,
                                                      layer->attn_output_a->abs_offset,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      g->heads) != 0;
        if (ok) {
            ok = ds4_gpu_matmul_q8_0_hc_expand_tensor(g->after_attn_hc,
                                                        g->attn_out,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_b->abs_offset,
                                                        (uint64_t)n_groups * rank,
                                                        DS4_N_EMBD,
                                                        g->attn_low,
                                                        g->cur_hc,
                                                        g->hc_split,
                                                        DS4_N_EMBD,
                                                        DS4_N_HC) != 0;
        }
    } else if (ok) {
        ok = ds4_gpu_attention_output_q8_batch_tensor(g->attn_out,
                                                        g->attn_low,
                                                        g->batch_group_tmp,
                                                        g->batch_low_tmp,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_a->abs_offset,
                                                        layer->attn_output_b->abs_offset,
                                                        group_dim, rank,
                                                        n_groups, DS4_N_EMBD,
                                                        g->heads, 1) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_output");
    if (ok) {
        metal_graph_debug_dump_tensor("attn_low", g->attn_low, (uint64_t)n_groups * rank, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("attn_out", g->attn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && metal_graph_directional_steering_attn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_attn(g, g->attn_out, il, 1);
    }
    if (ok && !fuse_attn_out_hc) {
        ok = ds4_gpu_hc_expand_tensor(g->after_attn_hc, g->attn_out, g->cur_hc,
                                        g->hc_post, g->hc_comb, DS4_N_EMBD, DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_post", g->after_attn_hc, hc_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->after_attn_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_ffn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->ffn_cur,
                                                         g->ffn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->after_attn_hc,
                                                         model->map,
                                                         model->size,
                                                         layer->hc_ffn_scale->abs_offset,
                                                         layer->hc_ffn_base->abs_offset,
                                                         layer->ffn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC,
                                                         DS4_N_HC_SINKHORN_ITER,
                                                         DS4_HC_EPS,
                                                         DS4_RMS_EPS) != 0;
    } else if (ok) {
        ok = metal_graph_decode_hc_pre(g->ffn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->after_attn_hc,
                                       model,
                                       layer->hc_ffn_scale->abs_offset,
                                       layer->hc_ffn_base->abs_offset);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", g->ffn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->ffn_norm, g->ffn_cur,
                                                                   model->map, model->size,
                                                                   layer->ffn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", g->ffn_norm, DS4_N_EMBD, il, pos);
    }
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,
                                                 DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);
    if (ok) ok = ds4_gpu_router_select_tensor(g->router_selected, g->router_weights, g->router_probs,
                                                model->map, model->size,
                                                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                (uint32_t)token,
                                                0,
                                                0,
                                                layer->ffn_exp_probs_b != NULL,
                                                layer->ffn_gate_tid2eid != NULL,
                                                g->router_logits) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("router");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", g->router_logits, DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_probs", g->router_probs, DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", g->router_selected, DS4_N_EXPERT_USED, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->router_weights, DS4_N_EXPERT_USED, il, pos);
    }
    if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,
                                                 g->routed_gate,
                                                 g->routed_up,
                                                 g->routed_mid,
                                                 g->routed_down,
                                                 model->map, model->size,
                                                 layer->ffn_gate_exps->abs_offset,
                                                 layer->ffn_up_exps->abs_offset,
                                                 layer->ffn_down_exps->abs_offset,
                                                 layer->ffn_gate_exps->type,
                                                 layer->ffn_down_exps->type,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 (uint32_t)expert_in_dim,
                                                 (uint32_t)down_in_dim,
                                                 (uint32_t)routed_out_dim,
                                                 g->router_selected, g->router_weights,
                                                 DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->ffn_norm) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->routed_gate,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", g->routed_up,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->routed_mid,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", g->routed_down,
                                      (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", g->routed_out, DS4_N_EMBD, il, pos);
    }
    const bool fuse_shared_gate_up =
        !g->quality &&
        getenv("DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION") == NULL;
    if (ok && fuse_shared_gate_up) {
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(g->shared_gate,
                                                         g->shared_up,
                                                         g->shared_mid,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_gate_shexp->abs_offset,
                                                         layer->ffn_up_shexp->abs_offset,
                                                         DS4_N_EMBD,
                                                         shared_dim,
                                                         g->ffn_norm) != 0;
    } else {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->shared_gate, model->map, model->size,
                                                  layer->ffn_gate_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->shared_up, model->map, model->size,
                                                  layer->ffn_up_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_gpu_swiglu_tensor(g->shared_mid, g->shared_gate, g->shared_up, shared_dim, 0.0f, 1.0f) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_gate_up");
    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos);
    const bool fuse_shared_down_hc =
        !keep_ffn_out && !metal_graph_use_reference_shared_down_hc();
    if (ok && fuse_shared_down_hc) {
        ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(g->after_ffn_hc,
                                                         g->shared_out,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_down_shexp->abs_offset,
                                                         shared_dim,
                                                         DS4_N_EMBD,
                                                         g->shared_mid,
                                                         g->routed_out,
                                                         g->after_attn_hc,
                                                         g->hc_split,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(g->shared_out, model->map, model->size,
                                          layer->ffn_down_shexp->abs_offset,
                                          shared_dim, DS4_N_EMBD,
                                          g->shared_mid, 1) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_shexp", g->shared_out, DS4_N_EMBD, il, pos);
    }
    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_ffn_out(g) &&
             ds4_gpu_add_tensor(g->ffn_out, g->shared_out, g->routed_out, DS4_N_EMBD) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", g->ffn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(g, g->ffn_out, il, 1);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_tensor(g->after_ffn_hc,
                                        g->ffn_out,
                                        g->after_attn_hc,
                                        g->hc_post,
                                        g->hc_comb,
                                        DS4_N_EMBD,
                                        DS4_N_HC) != 0;
    } else if (ok && !fuse_shared_down_hc) {
        ok = ds4_gpu_hc_expand_add_split_tensor(g->after_ffn_hc,
                                                  g->routed_out,
                                                  g->shared_out,
                                                  g->after_attn_hc,
                                                  g->hc_split,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef DS4_METAL_PROFILE_DECODE_STAGE
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", g->after_ffn_hc, hc_dim, il, pos);
    }
    return ok;
}

/* Encode the final HC collapse, output norm, and vocab projection on Metal. */
static bool metal_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(g->output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->flat_hc,
                                             1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc_pre", g->output_pre, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc_weights", g->output_weights, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc", g->output_embd, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_norm", g->output_norm, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->logits,
                                              model->map,
                                              model->size,
                                              weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              g->output_norm,
                                              1) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("result_output", g->logits, vocab_dim, DS4_N_LAYER, 0);
    }
    return ok;
}

/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and Q8 vocab projection.  For
 * tiny MTP suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
static bool metal_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre = NULL;
    ds4_gpu_tensor *output_weights = NULL;
    ds4_gpu_tensor *output_embd = NULL;
    ds4_gpu_tensor *output_norm = NULL;
    ds4_gpu_tensor *logits = NULL;

    bool ok = true;
    output_pre = ds4_gpu_tensor_view(g->batch_hc_mix,
                                       0,
                                       (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_weights = ds4_gpu_tensor_view(g->batch_hc_split,
                                           0,
                                           (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_embd = ds4_gpu_tensor_view(g->batch_ffn_cur,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    output_norm = ds4_gpu_tensor_view(g->batch_ffn_norm,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    logits = ds4_gpu_tensor_view(g->spec_logits,
                                   0,
                                   (uint64_t)n_tokens * vocab_dim * sizeof(float));
    ok = output_pre && output_weights && output_embd && output_norm && logits;

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                    output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                  g->batch_cur_hc,
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(logits,
                                              model->map,
                                              model->size,
                                              weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              output_norm,
                                              n_tokens) != 0;

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

static bool metal_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    fprintf(stderr, "ds4: Metal plain matmul does not support %s\n", tensor_type_name(w->type));
    return false;
}

static bool metal_graph_encode_output_head_mtp(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(g->output_pre, mtp_model, mtp->hc_head_fn,
                                                 hc_dim, DS4_N_HC, g->flat_hc, 1);
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    mtp_model->map,
                                                    mtp_model->size,
                                                    mtp->hc_head_scale->abs_offset,
                                                    mtp->hc_head_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  mtp_model->map,
                                                  mtp_model->size,
                                                  mtp->norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->logits,
                                              base_model->map,
                                              base_model->size,
                                              base_weights->output->abs_offset,
                                              DS4_N_EMBD,
                                              vocab_dim,
                                              g->output_norm,
                                              1) != 0;
    return ok;
}

/* =========================================================================
 * Metal Diagnostic Comparisons.
 * =========================================================================
 *
 * These routines deliberately allocate CPU-side reference buffers and read
 * Metal tensors back.  They are not part of generation; command-line tests use
 * them to localize drift against the C reference pipeline.
 */

static void metal_graph_trace_layer_stages(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        const float            *cpu_in_hc,
        uint32_t                il,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t shared_in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];

    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float post[4];
    float comb[16];
    float ffn_post[4];
    float ffn_comb[16];
    int selected[DS4_N_EXPERT_USED];
    float expert_weight[DS4_N_EXPERT_USED];
    const uint64_t shared_blocks = (shared_in_dim + 31) / 32;
    int8_t *shared_xq = xmalloc((size_t)shared_blocks * 32);
    float *shared_xscale = xmalloc((size_t)shared_blocks * sizeof(float));
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));

    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_in_hc, cpu_attn_cur, post, comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_one(cpu_heads, model, layer, cpu_q, cpu_kv);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_in_hc, post, comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, ffn_post, ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    quantize_q8_0_activation(cpu_ffn_norm, shared_xq, shared_xscale, shared_in_dim);
    matvec_q8_0_pair_prequant(cpu_shared_gate,
                              cpu_shared_up,
                              model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              shared_xq,
                              shared_xscale);
    swiglu(cpu_shared_mid, cpu_shared_gate, cpu_shared_up, shared_dim);
    matvec_q8_0(cpu_shared, model, layer->ffn_down_shexp, cpu_shared_mid);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  il,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc, cpu_ffn_out, cpu_after_attn_hc, ffn_post, ffn_comb, DS4_N_EMBD, DS4_N_HC);

    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    int gpu_selected[DS4_N_EXPERT_USED];
    float gpu_expert_weight[DS4_N_EXPERT_USED];

    bool ok = ds4_gpu_tensor_read(g->attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->after_attn_hc, 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_gate, 0, gpu_shared_gate, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_up, 0, gpu_shared_up, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_mid, 0, gpu_shared_mid, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
              ds4_gpu_tensor_read(g->router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
              ds4_gpu_tensor_read(g->routed_mid, 0, gpu_routed_mid_all, (uint64_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->cur_hc, 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0;

    if (ok) {
        fprintf(stderr,
                "ds4: Metal stage layer %u attn_cur=%g/%g attn_norm=%g/%g q=%g/%g kv=%g/%g attn_out=%g/%g after_attn_hc=%g/%g ffn_cur=%g/%g ffn_norm=%g/%g shared=%g/%g router_w=%g routed=%g/%g ffn_out=%g/%g after_ffn_hc=%g/%g\n",
                il,
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD), rms_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD), rms_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim), rms_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM), rms_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD), rms_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim), rms_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD), rms_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD), rms_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD), rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD), rms_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim), rms_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim));
        fprintf(stderr,
                "ds4: Metal shared layer %u gate=%g/%g up=%g/%g mid=%g/%g down=%g/%g\n",
                il,
                max_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim), rms_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim),
                max_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim), rms_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim),
                max_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim), rms_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD));
        fprintf(stderr,
                "ds4: Metal routed layer %u mid=%g/%g out=%g/%g\n",
                il,
                max_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                rms_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal stage layer %u router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    il,
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
    }

    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_routed_mid_all);
    free(gpu_shared);
    free(gpu_shared_mid);
    free(gpu_shared_up);
    free(gpu_shared_gate);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(shared_xscale);
    free(shared_xq);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_routed);
    free(cpu_shared);
    free(cpu_shared_mid);
    free(cpu_shared_up);
    free(cpu_shared_gate);
    free(cpu_ffn_norm);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_kv);
    free(cpu_qr_norm);
    free(cpu_q);
    free(cpu_attn_norm);
    free(cpu_attn_cur);
}

static int metal_graph_decode_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const ds4_layer_weights *layer = &weights->layer[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t vocab_dim = weights->output->dim[1];

    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_ffn_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    int gpu_selected[DS4_N_EXPERT_USED];
    float gpu_expert_weight[DS4_N_EXPERT_USED];
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));
    int selected[DS4_N_EXPERT_USED];
    float expert_weight[DS4_N_EXPERT_USED];

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cpu_hc, plain, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_hc, cpu_attn_cur, cpu_post, cpu_comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_rows_one(cpu_heads, model, layer, cpu_q, cpu_kv, 1);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_hc, cpu_post, cpu_comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, cpu_ffn_post, cpu_ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    layer_shared_ffn_one(cpu_shared, model, layer, cpu_ffn_norm);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  0,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc,
                cpu_ffn_out,
                cpu_after_attn_hc,
                cpu_ffn_post,
                cpu_ffn_comb,
                DS4_N_EMBD,
                DS4_N_HC);
    output_logits_one(cpu_logits, model, weights, cpu_after_ffn_hc);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc(&g, weights, layer);
    g.materialize_ffn_out = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                 model->map,
                                                 model->size,
                                                 weights->token_embd->abs_offset,
                                                 (uint32_t)weights->token_embd->dim[1],
                                                 (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
    if (ok) ok = metal_graph_encode_decode_layer(&g,
                                               model,
                                               layer,
                                               0,
                                               0,
                                               g.layer_raw_cache[0],
                                               g.raw_cap,
                                               0,
                                               1,
                                               token);
    if (ok) {
        ds4_gpu_tensor *embedded_hc = g.cur_hc;
        g.cur_hc = g.after_ffn_hc;
        g.after_ffn_hc = embedded_hc;
    }
    if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
    if (ok) ok = ds4_gpu_end_commands() != 0;

    if (ok) {
        ok = ds4_gpu_tensor_read(g.after_ffn_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.layer_raw_cache[0], 0, gpu_raw, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.after_attn_hc, 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
             ds4_gpu_tensor_read(g.router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
             ds4_gpu_tensor_read(g.routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.cur_hc, 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.logits, 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        fprintf(stderr,
                "ds4: Metal graph test layer0 diffs: embed_hc=%g hc_pre=%g attn_norm=%g q_rope=%g kv_rope=%g raw_cache=%g attn_out=%g after_attn_hc=%g ffn_cur=%g ffn_norm=%g shared=%g router_w=%g routed=%g ffn_out=%g after_ffn_hc=%g logits=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_kv, gpu_raw, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal graph router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
        print_vec_stats("metal graph q", gpu_q, q_dim);
        print_vec_stats("metal graph kv", gpu_kv, DS4_N_HEAD_DIM);
        print_vec_stats("metal graph routed", gpu_routed, DS4_N_EMBD);
    } else {
        fprintf(stderr, "ds4: Metal graph test failed while encoding first decode stages\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph test failure also failed\n");
        }
    }

    metal_graph_free(&g);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(gpu_logits);
    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_shared);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_raw);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(gpu_hc);
    free(cpu_kv);
    free(cpu_q);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_ffn_norm);
    free(cpu_routed);
    free(cpu_logits);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_shared);
    free(cpu_ffn_comb);
    free(cpu_ffn_post);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_qr_norm);
    free(cpu_attn_norm);
    free(cpu_comb);
    free(cpu_post);
    free(cpu_attn_cur);
    free(cpu_hc);
    free(plain);
    return ok ? 0 : 1;
}

static int metal_graph_first_token_full_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: full Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t vocab_dim = weights->output->dim[1];
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));

    forward_first_token_cpu(cpu_hc, model, weights, token);
    output_logits_one(cpu_logits, model, weights, cpu_hc);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc(&g, weights, &weights->layer[0]);
    const bool trace_layers = getenv("DS4_METAL_GRAPH_TRACE_LAYERS") != NULL;
    if (trace_layers && ok) {
        g.materialize_ffn_out = true;
        const bool teacher_force = getenv("DS4_METAL_GRAPH_TEACHER_FORCE") != NULL;
        const char *stage_layer_env = getenv("DS4_METAL_GRAPH_TRACE_STAGE_LAYER");
        const long stage_layer = stage_layer_env ? strtol(stage_layer_env, NULL, 10) : -1;
        float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
        float *cpu_cur = xmalloc((size_t)hc_dim * sizeof(float));
        float *cpu_next = xmalloc((size_t)hc_dim * sizeof(float));

        embed_token_f16(model, weights, token, plain);
        hc_from_plain_embedding(cpu_cur, plain, DS4_N_EMBD, DS4_N_HC);
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            if (teacher_force) {
                ok = ds4_gpu_tensor_write(g.cur_hc, 0, cpu_cur, hc_dim * sizeof(float)) != 0;
            }
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                       il, 0, g.layer_raw_cache[il], g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = g.cur_hc;
            g.cur_hc = g.after_ffn_hc;
            g.after_ffn_hc = tmp;
            if (ok) ok = ds4_gpu_end_commands() != 0;

            layer_forward_self_one(cpu_next, model, &weights->layer[il], cpu_cur, il, 0, token);
            if (ok) ok = ds4_gpu_tensor_read(g.cur_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0;
            if (ok) {
                fprintf(stderr,
                        "ds4: Metal full graph layer %u%s hc_max=%g hc_rms=%g\n",
                        il,
                        teacher_force ? " teacher" : "",
                        max_abs_diff(cpu_next, gpu_hc, hc_dim),
                        rms_abs_diff(cpu_next, gpu_hc, hc_dim));
                if (stage_layer == (long)il) {
                    metal_graph_trace_layer_stages(&g, model, &weights->layer[il], cpu_cur, il, token);
                }
            }
            float *ctmp = cpu_cur;
            cpu_cur = cpu_next;
            cpu_next = ctmp;
        }

        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;

        free(cpu_next);
        free(cpu_cur);
        free(plain);
    } else {
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                 il, 0, g.layer_raw_cache[il],
                                                 g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = g.cur_hc;
            g.cur_hc = g.after_ffn_hc;
            g.after_ffn_hc = tmp;
        }

        if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;
    }

    if (ok) {
        ok = ds4_gpu_tensor_read(g.cur_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.logits, 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        const uint64_t cpu_top = argmax_f32(cpu_logits, vocab_dim);
        const uint64_t gpu_top = argmax_f32(gpu_logits, vocab_dim);
        fprintf(stderr,
                "ds4: Metal full first-token graph diffs: final_hc_max=%g final_hc_rms=%g logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                rms_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                rms_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
    } else {
        fprintf(stderr, "ds4: Metal full first-token graph test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after full graph failure also failed\n");
        }
    }

    metal_graph_free(&g);
    free(gpu_logits);
    free(cpu_logits);
    free(gpu_hc);
    free(cpu_hc);
    return ok ? 0 : 1;
}

/* =========================================================================
 * Metal Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing Metal backend.  It uses the same layer
 * encoder as diagnostics, but diagnostics are not required for normal command
 * flow and their CPU reads stay outside these generation entry points.
 */

/* Encode a full single-token decode step on Metal.  This is the generation
 * hot path: update caches, run all layers, then produce logits. */
static bool metal_graph_encode_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        bool                   need_logits,
        bool                   allow_split_flush) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: Metal graph raw KV cache is not allocated\n");
        return false;
    }
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);

    bool ok = ds4_gpu_embed_token_hc_tensor(g->cur_hc,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              (uint32_t)weights->token_embd->dim[1],
                                              (uint32_t)token,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;

    /*
     * Start executing the prefix of the decode graph while the CPU is still
     * encoding the rest. The split point is layer-based because this executor is
     * a fixed DS4 tape, not a dynamic node graph; four layers is the measured
     * point where the prefix is large enough to hide useful work without
     * starving the second command buffer.
     */
    uint32_t split_after_layers = 4;
    const char *split_env = getenv("DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS");
    if (split_env && split_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(split_env, &end, 10);
        if (end != split_env && v <= DS4_N_LAYER) split_after_layers = (uint32_t)v;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
        ds4_gpu_tensor *tmp = g->cur_hc;
        g->cur_hc = g->after_ffn_hc;
        g->after_ffn_hc = tmp;
        if (ok && allow_split_flush && split_after_layers != 0 && il + 1u == split_after_layers) {
            ok = ds4_gpu_flush_commands() != 0;
        }
    }

    if (ok && need_logits) {
        ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    }
    return ok;
}

static ds4_gpu_tensor *metal_graph_tensor_row_view(
        ds4_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return ds4_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
}

/* Upload prompt token ids for kernels that need token-aware hash routing. */
static bool metal_graph_upload_prompt_tokens(
        ds4_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!out_tokens || pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) {
        return false;
    }

    int32_t *tokens = xmalloc((size_t)n_tokens * sizeof(tokens[0]));
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = prompt->v[pos0 + i];

    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                           0,
                                           tokens,
                                           (uint64_t)n_tokens * sizeof(tokens[0])) != 0;
    free(tokens);
    return ok;
}

/* Rebuild ratio-4 compressor state after chunked prefill so a following decode
 * token sees the same rolling compression window. */
static bool metal_graph_refresh_ratio4_compressor_state(
        ds4_gpu_graph  *g,
        const ds4_model  *model,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_tensor *kv_weight,
        const ds4_tensor *score_weight,
        const ds4_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!g || !model || !state_kv || !state_score || !kv_weight || !score_weight || !ape ||
        head_dim == 0 || width == 0 || n_tokens < 4) {
        return false;
    }

    /*
     * The recurrent ratio-4 state is intentionally rebuilt from the last
     * four tokens using the small-batch projection kernel. The full-chunk
     * projection is already available, but it uses the matrix-matrix path;
     * mixing those two accumulation orders changes a few FP8 rounding
     * decisions in later chunks.
     */
    ds4_gpu_tensor *tail_hc = ds4_gpu_tensor_view(
            g->batch_attn_norm,
            (uint64_t)(n_tokens - 4u) * DS4_N_EMBD * sizeof(float),
            4ull * DS4_N_EMBD * sizeof(float));
    bool ok = tail_hc != NULL;
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                         model->map,
                                         model->size,
                                         kv_weight->abs_offset,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                         model->map,
                                         model->size,
                                         score_weight->abs_offset,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = ds4_gpu_compressor_prefill_state_ratio4_tensor(state_kv,
                                                              state_score,
                                                              g->batch_comp_kv,
                                                              g->batch_comp_sc,
                                                              model->map,
                                                              model->size,
                                                              ape->abs_offset,
                                                              ape->type,
                                                              head_dim,
                                                              pos0 + n_tokens - 4u) != 0;
    }
    ds4_gpu_tensor_free(tail_hc);
    return ok;
}

/* CPU fallback for seeding batched HC state from token embeddings.  It is still
 * useful for tiny speculative verifier batches where a separate GPU embedding
 * command buffer costs more than the small host write. */
static bool metal_graph_upload_prompt_embeddings_hc_cpu(
        ds4_gpu_tensor   *out_hc,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t total = (uint64_t)n_tokens * hc_dim;
    float *hc = xmalloc((size_t)total * sizeof(hc[0]));
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));

    for (uint32_t t = 0; t < n_tokens; t++) {
        embed_token_f16(model, weights, prompt->v[pos0 + t], plain);
        float *dst = hc + (uint64_t)t * hc_dim;
        for (uint32_t h = 0; h < DS4_N_HC; h++) {
            memcpy(dst + (uint64_t)h * DS4_N_EMBD,
                   plain,
                   (size_t)DS4_N_EMBD * sizeof(plain[0]));
        }
    }

    const bool ok = ds4_gpu_tensor_write(out_hc, 0, hc, total * sizeof(hc[0])) != 0;
    free(plain);
    free(hc);
    return ok;
}

/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding.  Long prefill chunks use the Metal get-rows/repeat
 * kernel so the CPU does not build and upload a large [token, HC, dim] tensor. */
static bool metal_graph_upload_prompt_embeddings_hc(
        ds4_gpu_tensor   *out_hc,
        ds4_gpu_tensor   *tokens,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;

    uint32_t gpu_min = 512;
    const char *gpu_min_env = getenv("DS4_METAL_GPU_BATCH_EMBED_MIN");
    if (gpu_min_env && gpu_min_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(gpu_min_env, &end, 10);
        if (end != gpu_min_env && v <= UINT32_MAX) gpu_min = (uint32_t)v;
    }

    if (tokens && n_tokens >= gpu_min) {
        return ds4_gpu_embed_tokens_hc_tensor(out_hc,
                                                tokens,
                                                model->map,
                                                model->size,
                                                weights->token_embd->abs_offset,
                                                (uint32_t)weights->token_embd->dim[1],
                                                n_tokens,
                                                DS4_N_EMBD,
                                                DS4_N_HC) != 0;
    }

    return metal_graph_upload_prompt_embeddings_hc_cpu(out_hc,
                                                       model,
                                                       weights,
                                                       prompt,
                                                       pos0,
                                                       n_tokens);
}

static bool metal_graph_warmup_prefill_kernels(
        ds4_gpu_graph   *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           n_tokens) {
    static bool warmed = false;
    if (warmed || getenv("DS4_METAL_NO_PREFILL_KERNEL_WARMUP") != NULL) return true;

    /*
     * The first batched F16 matmul can pay Metal's one-time pipeline execution
     * cost. Run the same HC attention projection on scratch storage before the
     * measured prefill. The output is overwritten by the real graph.
     */
    if (n_tokens <= 8) return true;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(g->batch_hc_mix,
                                         model->map,
                                         model->size,
                                         weights->layer[0].hc_attn_fn->abs_offset,
                                         hc_dim,
                                         mix_hc,
                                         g->batch_flat_hc,
                                         n_tokens) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) {
        fprintf(stderr, "ds4: Metal prefill kernel warmup failed\n");
        return false;
    }

    warmed = true;
    return true;
}

/* Encode the batched prefill attention half for one layer.  It mirrors the CPU
 * layer-major path: HC pre/norm, Q/KV, cache/compression, prefix attention. */
static bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    if (stage != NULL) {
        fprintf(stderr,
                "ds4: metal indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
                il,
                pos0,
                n_tokens,
                n_comp,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

/* Optional prefill stage profiler. It intentionally ends the current Metal
 * command buffer and waits, so the printed number includes encoding plus GPU
 * execution for the stage just emitted. This is disabled by default because it
 * adds synchronization points and changes scheduling. */
static bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: metal layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            part,
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool metal_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: metal Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool metal_graph_encode_layer_attention_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const bool compressed = ratio != 0;
    const bool zero_prefix = pos0 == 0;
    const bool index_stage_profile = getenv("DS4_METAL_INDEXER_STAGE_PROFILE") != NULL;
    const bool layer_stage_profile = getenv("DS4_METAL_LAYER_STAGE_PROFILE") != NULL;
    const bool q_stage_profile = getenv("DS4_METAL_Q_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
    double q_stage_t0 = q_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_ATTN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("attn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)
#define DS4_METAL_PROFILE_Q_STAGE(name) do { \
        if (ok && q_stage_profile) { \
            ok = metal_graph_q_stage_profile_boundary((name), il, pos0, n_tokens, &q_stage_t0); \
        } \
    } while (0)
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    uint32_t *comp_counts = compressed ? xcalloc(n_tokens, sizeof(comp_counts[0])) : NULL;
    uint32_t *index_counts = ratio == 4 ? xcalloc(n_tokens, sizeof(index_counts[0])) : NULL;
    const bool qkv_rms_fused = !metal_graph_use_reference_qkv_norm();
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_attn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (metal_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_attn_scale->abs_offset,
                                                        layer->hc_attn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(attn_cur_view,
                                                            g->batch_cur_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_cur_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_attn_scale->abs_offset,
                                                            layer->hc_attn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre", g->batch_attn_cur,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("hc_pre");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                       g->batch_attn_cur,
                                                       model->map,
                                                       model->size,
                                                       layer->attn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("attn_norm", g->batch_attn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("norm");
    DS4_METAL_PROFILE_Q_STAGE("pre_q");
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_qr,
                                              model->map,
                                              model->size,
                                              layer->attn_q_a->abs_offset,
                                              DS4_N_EMBD,
                                              q_rank,
                                              g->batch_attn_norm,
                                              n_tokens) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora", g->batch_qr,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("q_a");
    if (qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_kv_raw,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HEAD_DIM,
                                                  g->batch_attn_norm,
                                                  n_tokens) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->batch_qr_norm,
                                                             g->batch_qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->batch_kv,
                                                             g->batch_kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             n_tokens,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_qr_norm,
                                                           g->batch_qr,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_q_a_norm->abs_offset,
                                                           (uint32_t)q_rank,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora_norm", g->batch_qr_norm,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    if (qkv_rms_fused && ok) {
        metal_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("q_a_norm");
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_q,
                                              model->map,
                                              model->size,
                                              layer->attn_q_b->abs_offset,
                                              q_rank,
                                              q_dim,
                                              g->batch_qr_norm,
                                              n_tokens) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("Qraw", g->batch_q,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("q_b");
    if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->batch_q,
                                                n_tokens,
                                                DS4_N_HEAD,
                                                DS4_N_HEAD_DIM,
                                                DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("Qnorm", g->batch_q,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("head_norm");
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                            n_tokens,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("Qcur", g->batch_q,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("rope");
    DS4_METAL_PROFILE_ATTN_STAGE("q_path");
    if (!qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_kv_raw,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HEAD_DIM,
                                                  g->batch_attn_norm,
                                                  n_tokens) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_kv,
                                                           g->batch_kv_raw,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_kv_a_norm->abs_offset,
                                                           DS4_N_HEAD_DIM,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_kv,
                                            n_tokens,
                                            DS4_N_HEAD_KV,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("KVrope", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv,
                                                       n_tokens,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("KVcur", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("kv_path");
    /*
     * Static graph order is q, kv, cpy_k(raw SWA), then attention. For a
     * zero-prefix batch it is safe to store the whole batch at once: attention
     * reads the contiguous batch KV, and the ring only has to end with the last
     * SWA rows for later chunks/decode. For nonzero chunks the physical ring is
     * sized to hold the current chunk plus the previous SWA window, while the
     * attention mask still enforces the 128-token logical window.
     */
    if (ok && zero_prefix) ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                                    g->batch_kv,
                                                                    g->raw_cap,
                                                                    pos0,
                                                                    n_tokens,
                                                                    DS4_N_HEAD_DIM) != 0;
    const bool raw_batch_attention = zero_prefix && ratio == 0;
    bool batch_attention_done = false;

    if (ok && raw_batch_attention) {
        ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                          model->map,
                                                          model->size,
                                                          layer->attn_sinks->abs_offset,
                                                          g->batch_q,
                                                          g->batch_kv,
                                                          n_tokens,
                                                          g->raw_window,
                                                          DS4_N_HEAD,
                                                          DS4_N_HEAD_DIM) != 0;
        if (ok) batch_attention_done = true;
    } else if (ok && !zero_prefix && ratio == 0 && n_tokens <= g->raw_cap) {
        /*
         * The ubatch path stores the whole batch in the SWA cache, then runs
         * one batched attention kernel with an absolute-position causal/window
         * mask.  This avoids mixing prefill with the different single-token
         * attention path.
         */
        const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, n_tokens);
        /* Nonzero prompt chunks read the SWA cache as a ring.  FlashAttention
         * receives a linearized window starting at raw_start, not physical row
         * zero; otherwise wrapped chunks silently miss recent raw keys. */
        const uint32_t raw_start = metal_graph_raw_start_for_span(g,
                                                                  pos0 + n_tokens - 1u,
                                                                  n_raw);
        ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                 g->batch_kv,
                                                 g->raw_cap,
                                                 pos0,
                                                 n_tokens,
                                                 DS4_N_HEAD_DIM) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("raw_cache",
                                          g->layer_raw_cache[il],
                                          (uint64_t)n_raw * DS4_N_HEAD_DIM,
                                          il,
                                          pos0);
        }
        if (ok) {
            ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(g->batch_heads,
                                                                   model->map,
                                                                   model->size,
                                                                   layer->attn_sinks->abs_offset,
                                                                   g->batch_q,
                                                                   g->layer_raw_cache[il],
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   g->raw_cap,
                                                                   raw_start,
                                                                   g->raw_window,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_HEAD_DIM) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && ratio != 0) {
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_gate &&
                                    layer->attn_compressor_ape && layer->attn_compressor_norm;
        if (!have_attn_comp) {
            fprintf(stderr, "ds4: Metal layer-major prefill needs attention compressor weights\n");
            ok = false;
        }
        if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                                 model->map,
                                                 model->size,
                                                 layer->attn_compressor_kv->abs_offset,
                                                 DS4_N_EMBD,
                                                 comp_width,
                                                 g->batch_attn_norm,
                                                 n_tokens) != 0;
        if (ok) metal_graph_debug_dump_tensor("attn_comp_kv_raw",
                                              g->batch_comp_kv,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                                 model->map,
                                                 model->size,
                                                 layer->attn_compressor_gate->abs_offset,
                                                 DS4_N_EMBD,
                                                 comp_width,
                                                 g->batch_attn_norm,
                                                 n_tokens) != 0;
        if (ok) metal_graph_debug_dump_tensor("attn_comp_score_raw",
                                              g->batch_comp_sc,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        uint32_t n_comp = g->layer_n_comp[il];
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok) {
                ok = ds4_gpu_compressor_prefill_tensor(g->layer_attn_comp_cache[il],
                                                         g->layer_attn_state_kv[il],
                                                         g->layer_attn_state_score[il],
                                                         g->batch_comp_kv,
                                                         g->batch_comp_sc,
                                                         model->map,
                                                         model->size,
                                                         layer->attn_compressor_ape->abs_offset,
                                                         layer->attn_compressor_ape->type,
                                                         layer->attn_compressor_norm->abs_offset,
                                                         layer->attn_compressor_norm->type,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         pos0,
                                                         n_tokens,
                                                         DS4_N_ROT,
                                                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                         true,
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         DS4_ROPE_YARN_BETA_FAST,
                                                         DS4_ROPE_YARN_BETA_SLOW,
                                                         DS4_RMS_EPS) != 0;
                if (ok && ratio == 4) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
            }
            if (ok) {
                g->layer_n_comp[il] = n_comp;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
                if (n_comp != 0) {
                    metal_graph_debug_dump_tensor("KVcompress",
                                                  g->layer_attn_comp_cache[il],
                                                  (uint64_t)n_comp * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                }
                metal_graph_debug_dump_tensor("attn_state_kv",
                                              g->layer_attn_state_kv[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
                metal_graph_debug_dump_tensor("attn_state_score",
                                              g->layer_attn_state_score[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
            }
        } else {
            const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            if (aligned_chunk) {
                const uint32_t comp_before = g->layer_n_comp[il];
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                ds4_gpu_tensor *comp_view = NULL;
                if (ok) {
                    comp_view = ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                                                      (uint64_t)comp_before * DS4_N_HEAD_DIM * sizeof(float),
                                                      (uint64_t)comp_chunk * DS4_N_HEAD_DIM * sizeof(float));
                    ok = comp_view != NULL;
                }
                if (ok && ratio == 4) {
                    ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                            comp_view,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                } else if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(
                            comp_view,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            ratio,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                }
                if (ok && ratio == 4) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_comp[il] = comp_before + comp_chunk;
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                    metal_graph_debug_dump_tensor("KVcompress",
                                                  comp_view,
                                                  (uint64_t)comp_chunk * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("attn_state_kv",
                                                  g->layer_attn_state_kv[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("attn_state_score",
                                                  g->layer_attn_state_score[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
                ds4_gpu_tensor_free(comp_view);
            } else {
                for (uint32_t t = 0; ok && t < n_tokens; t++) {
                    const uint32_t pos = pos0 + t;
                    const bool emit = ((pos + 1u) % ratio) == 0u;
                    if (emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                        break;
                    }
                    ds4_gpu_tensor *kv_view = metal_graph_tensor_row_view(g->batch_comp_kv, t, comp_width);
                    ds4_gpu_tensor *sc_view = metal_graph_tensor_row_view(g->batch_comp_sc, t, comp_width);
                    const uint32_t comp_row = g->layer_n_comp[il];
                    ok = kv_view && sc_view &&
                         ds4_gpu_compressor_update_tensor(kv_view,
                                                            sc_view,
                                                            g->layer_attn_state_kv[il],
                                                            g->layer_attn_state_score[il],
                                                            g->layer_attn_comp_cache[il],
                                                            model->map,
                                                            model->size,
                                                            layer->attn_compressor_ape->abs_offset,
                                                            layer->attn_compressor_ape->type,
                                                            layer->attn_compressor_norm->abs_offset,
                                                            layer->attn_compressor_norm->type,
                                                            DS4_N_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            comp_row,
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS) != 0;
                    if (ok && emit) {
                        ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
                                g->layer_attn_comp_cache[il],
                                (uint64_t)comp_row * DS4_N_HEAD_DIM * sizeof(float),
                                (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
                        ok = comp_row_view &&
                             ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view,
                                                                   1,
                                                                   DS4_N_HEAD_DIM,
                                                                   DS4_N_ROT) != 0;
                        if (ok) {
                            metal_graph_debug_dump_tensor("KVcompress",
                                                          comp_row_view,
                                                          DS4_N_HEAD_DIM,
                                                          il,
                                                          pos);
                        }
                        ds4_gpu_tensor_free(comp_row_view);
                    }
                    if (ok && emit) g->layer_n_comp[il]++;
                    if (comp_counts) comp_counts[t] = g->layer_n_comp[il];
                    if (ok && t == 0) ok = metal_graph_capture_prefix1_attn_state(g, il);
                    ds4_gpu_tensor_free(sc_view);
                    ds4_gpu_tensor_free(kv_view);
                }
            }
            n_comp = g->layer_n_comp[il];
        }
        DS4_METAL_PROFILE_ATTN_STAGE("compressor");

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "ds4: Metal layer-major prefill needs indexer weights\n");
                ok = false;
            }
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_kv,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_compressor_kv->abs_offset,
                                                     DS4_N_EMBD,
                                                     index_width,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (ok) metal_graph_debug_dump_tensor("indexer_comp_kv_raw",
                                                  g->batch_comp_kv,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_comp_sc,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_compressor_gate->abs_offset,
                                                     DS4_N_EMBD,
                                                     index_width,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (ok) metal_graph_debug_dump_tensor("indexer_comp_score_raw",
                                                  g->batch_comp_sc,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_q,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_attn_q_b->abs_offset,
                                                     q_rank,
                                                     (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                     g->batch_qr_norm,
                                                     n_tokens) != 0;
            if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_indexer_q,
                                                    n_tokens,
                                                    DS4_N_INDEXER_HEAD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    DS4_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    DS4_ROPE_YARN_BETA_FAST,
                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_indexer_weights,
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_proj->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_INDEXER_HEAD,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (zero_prefix) {
                if (ok && n_comp > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal layer-major indexer cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(g->layer_index_comp_cache[il],
                                                             g->layer_index_state_kv[il],
                                                             g->layer_index_state_score[il],
                                                             g->batch_comp_kv,
                                                             g->batch_comp_sc,
                                                             model->map,
                                                             model->size,
                                                             layer->indexer_compressor_ape->abs_offset,
                                                             layer->indexer_compressor_ape->type,
                                                             layer->indexer_compressor_norm->abs_offset,
                                                             layer->indexer_compressor_norm->type,
                                                             DS4_N_INDEXER_HEAD_DIM,
                                                             ratio,
                                                             pos0,
                                                             n_tokens,
                                                             DS4_N_ROT,
                                                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                             false,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             DS4_RMS_EPS) != 0;
                }
                if (ok) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_index_state_kv[il],
                                                                     g->layer_index_state_score[il],
                                                                     layer->indexer_compressor_kv,
                                                                     layer->indexer_compressor_gate,
                                                                     layer->indexer_compressor_ape,
                                                                     DS4_N_INDEXER_HEAD_DIM,
                                                                     index_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_index_comp[il] = n_comp;
                    for (uint32_t t = 0; t < n_tokens; t++) {
                        index_counts[t] = (pos0 + t + 1u) / ratio;
                    }
                    if (n_comp != 0) {
                        metal_graph_debug_dump_tensor("indexer_KVcompress",
                                                      g->layer_index_comp_cache[il],
                                                      (uint64_t)n_comp * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                    }
                    metal_graph_debug_dump_tensor("indexer_state_kv",
                                                  g->layer_index_state_kv[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("indexer_state_score",
                                                  g->layer_index_state_score[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
            } else {
                const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                if (aligned_chunk) {
                    const uint32_t index_before = g->layer_n_index_comp[il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    ds4_gpu_tensor *index_view = NULL;
                    if (ok) {
                        index_view = ds4_gpu_tensor_view(
                                g->layer_index_comp_cache[il],
                                (uint64_t)index_before * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
                    if (ok) {
                        ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                                index_view,
                                g->layer_index_state_kv[il],
                                g->layer_index_state_score[il],
                                g->batch_comp_kv,
                                g->batch_comp_sc,
                                model->map,
                                model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                DS4_N_INDEXER_HEAD_DIM,
                                pos0,
                                n_tokens,
                                DS4_N_ROT,
                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                false,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW,
                                DS4_RMS_EPS) != 0;
                    }
                    if (ok) {
                        ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                         model,
                                                                         g->layer_index_state_kv[il],
                                                                         g->layer_index_state_score[il],
                                                                         layer->indexer_compressor_kv,
                                                                         layer->indexer_compressor_gate,
                                                                         layer->indexer_compressor_ape,
                                                                         DS4_N_INDEXER_HEAD_DIM,
                                                                         index_width,
                                                                         pos0,
                                                                         n_tokens);
                    }
                    if (ok) {
                        g->layer_n_index_comp[il] = index_before + index_chunk;
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                        metal_graph_debug_dump_tensor("indexer_KVcompress",
                                                      index_view,
                                                      (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                        metal_graph_debug_dump_tensor("indexer_state_kv",
                                                      g->layer_index_state_kv[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                        metal_graph_debug_dump_tensor("indexer_state_score",
                                                      g->layer_index_state_score[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                    }
                    ds4_gpu_tensor_free(index_view);
                } else {
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t pos = pos0 + t;
                        const bool emit = ((pos + 1u) % ratio) == 0u;
                        if (emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                            fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                            ok = false;
                            break;
                        }
                        ds4_gpu_tensor *kv_view = metal_graph_tensor_row_view(g->batch_comp_kv, t, index_width);
                        ds4_gpu_tensor *sc_view = metal_graph_tensor_row_view(g->batch_comp_sc, t, index_width);
                        const uint32_t index_row = g->layer_n_index_comp[il];
                        ok = kv_view && sc_view &&
                             ds4_gpu_compressor_update_tensor(kv_view,
                                                                sc_view,
                                                                g->layer_index_state_kv[il],
                                                                g->layer_index_state_score[il],
                                                                g->layer_index_comp_cache[il],
                                                                model->map,
                                                                model->size,
                                                                layer->indexer_compressor_ape->abs_offset,
                                                                layer->indexer_compressor_ape->type,
                                                                layer->indexer_compressor_norm->abs_offset,
                                                                layer->indexer_compressor_norm->type,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                ratio,
                                                                pos,
                                                                index_row,
                                                                DS4_N_ROT,
                                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                                freq_base,
                                                                freq_scale,
                                                                ext_factor,
                                                                attn_factor,
                                                                DS4_ROPE_YARN_BETA_FAST,
                                                                DS4_ROPE_YARN_BETA_SLOW,
                                                                DS4_RMS_EPS) != 0;
                        if (ok && emit) g->layer_n_index_comp[il]++;
                        if (index_counts) index_counts[t] = g->layer_n_index_comp[il];
                        if (ok && t == 0) ok = metal_graph_capture_prefix1_index_state(g, il);
                        ds4_gpu_tensor_free(sc_view);
                        ds4_gpu_tensor_free(kv_view);
                    }
                }
            }
        }
        if (ratio == 4) DS4_METAL_PROFILE_ATTN_STAGE("indexer_setup");

        if (ok && !zero_prefix && n_tokens <= g->raw_cap) {
            const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start = metal_graph_raw_start_for_span(g,
                                                                      pos0 + n_tokens - 1u,
                                                                      n_raw);
            uint32_t use_comp_mask = 0;
            bool use_indexed_comp = false;
            double index_stage_t0 = 0.0;

            ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                     g->batch_kv,
                                                     g->raw_cap,
                                                     pos0,
                                                     n_tokens,
                                                     DS4_N_HEAD_DIM) != 0;
            if (ok && ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K) {
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                ok = ds4_gpu_indexer_scores_decode_batch_tensor(g->indexer_scores,
                                                                  g->batch_indexer_q,
                                                                  g->batch_indexer_weights,
                                                                  g->layer_index_comp_cache[il],
                                                                  n_comp,
                                                                  n_tokens,
                                                                  pos0,
                                                                  DS4_N_INDEXER_HEAD,
                                                                  DS4_N_INDEXER_HEAD_DIM,
                                                                  ratio,
                                                                  index_scale) != 0;
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("score",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    metal_graph_debug_dump_tensor("indexer_scores",
                                                  g->indexer_scores,
                                                  (uint64_t)n_comp * n_tokens,
                                                  il,
                                                  pos0);
                }
                if (ok) {
                    ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       n_comp,
                                                       n_tokens,
                                                       DS4_N_INDEXER_TOP_K) != 0;
                    if (ok && index_stage_profile) {
                        ok = metal_graph_indexer_stage_profile_boundary("topk",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                    if (ok) {
                        metal_graph_debug_dump_i32_tensor("indexer_topk",
                                                          g->comp_selected,
                                                          (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                          il,
                                                          pos0);
                    }
                }
                if (ok) {
                    use_indexed_comp = true;
                }
                use_comp_mask = 1;
            }
            if (ok) {
                if (use_indexed_comp) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              g->batch_q,
                                                                              g->layer_raw_cache[il],
                                                                              g->layer_attn_comp_cache[il],
                                                                              g->comp_selected,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              DS4_N_INDEXER_TOP_K,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                    if (ok && index_stage_profile) {
                        ok = metal_graph_indexer_stage_profile_boundary("attention",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                } else {
                    ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(g->batch_heads,
                                                                             model->map,
                                                                             model->size,
                                                                             layer->attn_sinks->abs_offset,
                                                                             g->batch_q,
                                                                             g->layer_raw_cache[il],
                                                                             g->layer_attn_comp_cache[il],
                                                                             use_comp_mask ? g->comp_mask : NULL,
                                                                             use_comp_mask,
                                                                             n_tokens,
                                                                             pos0,
                                                                             n_raw,
                                                                             g->raw_cap,
                                                                             raw_start,
                                                                             n_comp,
                                                                             g->raw_window,
                                                                             ratio,
                                                                             DS4_N_HEAD,
                                                                             DS4_N_HEAD_DIM) != 0;
                }
            }
            if (ok) batch_attention_done = true;
        }

        const bool topk_prefill_needed = ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K;
        if (ok && zero_prefix && topk_prefill_needed && n_comp != 0) {
            const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
            double index_stage_t0 = 0.0;
            if (index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            ok = ds4_gpu_indexer_scores_prefill_tensor(g->indexer_scores,
                                                         g->batch_indexer_q,
                                                         g->batch_indexer_weights,
                                                         g->layer_index_comp_cache[il],
                                                         n_comp,
                                                         n_tokens,
                                                         DS4_N_INDEXER_HEAD,
                                                         DS4_N_INDEXER_HEAD_DIM,
                                                         ratio,
                                                         index_scale) != 0;
            if (ok && index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary("score",
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            if (ok) {
                metal_graph_debug_dump_tensor("indexer_scores",
                                              g->indexer_scores,
                                              (uint64_t)n_comp * n_tokens,
                                              il,
                                              pos0);
            }
            if (ok) {
                ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                   g->indexer_scores,
                                                   n_comp,
                                                   n_tokens,
                                                   DS4_N_INDEXER_TOP_K) != 0;
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("topk",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    metal_graph_debug_dump_i32_tensor("indexer_topk",
                                                      g->comp_selected,
                                                      (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                      il,
                                                      pos0);
                }
            }
            if (ok) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                          model->map,
                                                                          model->size,
                                                                          layer->attn_sinks->abs_offset,
                                                                          g->batch_q,
                                                                          g->layer_raw_cache[il],
                                                                          g->layer_attn_comp_cache[il],
                                                                          g->comp_selected,
                                                                          n_tokens,
                                                                          pos0,
                                                                          n_tokens,
                                                                          g->raw_cap,
                                                                          0,
                                                                          n_comp,
                                                                          DS4_N_INDEXER_TOP_K,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          DS4_N_HEAD,
                                                                          DS4_N_HEAD_DIM) != 0;
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("attention",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
            }
            if (ok) batch_attention_done = true;
        }
        if (ok && zero_prefix && !topk_prefill_needed && n_comp != 0) {
            ok = ds4_gpu_attention_prefill_static_mixed_heads_tensor(g->batch_heads,
                                                                       model->map,
                                                                       model->size,
                                                                       layer->attn_sinks->abs_offset,
                                                                       g->batch_q,
                                                                       g->batch_kv,
                                                                       g->layer_attn_comp_cache[il],
                                                                       n_tokens,
                                                                       n_comp,
                                                                       g->raw_window,
                                                                       ratio,
                                                                       DS4_N_HEAD,
                                                                       DS4_N_HEAD_DIM) != 0;
            if (ok) batch_attention_done = true;
        }
    }

    if (ok && !raw_batch_attention && !batch_attention_done) {
        uint32_t raw_prefix_tokens = 0;
        if (zero_prefix && ratio != 0 && n_tokens <= g->raw_cap && comp_counts != NULL) {
            while (raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens] == 0u) {
                raw_prefix_tokens++;
            }
        }

        if (raw_prefix_tokens != 0) {
            ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                              model->map,
                                                              model->size,
                                                              layer->attn_sinks->abs_offset,
                                                              g->batch_q,
                                                              g->batch_kv,
                                                              raw_prefix_tokens,
                                                              g->raw_window,
                                                              DS4_N_HEAD,
                                                              DS4_N_HEAD_DIM) != 0;
        }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = metal_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp = comp_counts ? comp_counts[t] : 0u;
                const uint32_t cur_index = index_counts ? index_counts[t] : 0u;
                uint32_t n_selected = 0;
                ds4_gpu_tensor *comp_mask = NULL;

                if (ratio == 4 && cur_comp > DS4_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ds4_gpu_tensor *indexer_q_view = metal_graph_tensor_row_view(
                            g->batch_indexer_q, t, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                    ds4_gpu_tensor *indexer_w_view = metal_graph_tensor_row_view(
                            g->batch_indexer_weights, t, DS4_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                            indexer_q_view,
                                                            indexer_w_view,
                                                            g->layer_index_comp_cache[il],
                                                            cur_index,
                                                            DS4_N_INDEXER_HEAD,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            index_scale) != 0 &&
                         ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       cur_index,
                                                       1,
                                                       DS4_N_INDEXER_TOP_K) != 0 &&
                         ds4_gpu_dsv4_topk_mask_tensor(g->comp_mask,
                                                         g->comp_selected,
                                                         cur_index,
                                                         1,
                                                         DS4_N_INDEXER_TOP_K) != 0;
                    ds4_gpu_tensor_free(indexer_w_view);
                    ds4_gpu_tensor_free(indexer_q_view);
                    if (ok) {
                        comp_mask = g->comp_mask;
                        n_selected = DS4_N_INDEXER_TOP_K < cur_index
                            ? DS4_N_INDEXER_TOP_K
                            : cur_index;
                    }
                }

                ds4_gpu_tensor *q_view = metal_graph_tensor_row_view(g->batch_q, t, q_dim);
                ds4_gpu_tensor *kv_cache_view = metal_graph_tensor_row_view(g->batch_kv, t, DS4_N_HEAD_DIM);
                ds4_gpu_tensor *heads_view = metal_graph_tensor_row_view(g->batch_heads, t, q_dim);
                ok = ok && q_view && kv_cache_view && heads_view;
                if (ok && !zero_prefix) {
                    ok = ds4_gpu_store_raw_kv_tensor(g->layer_raw_cache[il],
                                                       kv_cache_view,
                                                       g->raw_cap,
                                                       pos % g->raw_cap,
                                                       DS4_N_HEAD_DIM) != 0;
                }
                if (ok && comp_mask != NULL && n_selected != 0) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads_view,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              q_view,
                                                                              g->layer_raw_cache[il],
                                                                              g->layer_attn_comp_cache[il],
                                                                              g->comp_selected,
                                                                              1,
                                                                              pos,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              cur_comp,
                                                                              n_selected,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                } else if (ok) {
                    ok = ds4_gpu_attention_decode_heads_tensor(heads_view,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->attn_sinks->abs_offset,
                                                                 q_view,
                                                                 g->layer_raw_cache[il],
                                                                 n_raw,
                                                                 g->raw_cap,
                                                                 raw_start,
                                                                 cur_comp ? g->layer_attn_comp_cache[il] : NULL,
                                                                 cur_comp,
                                                                 comp_mask,
                                                                 n_selected,
                                                                 DS4_N_HEAD,
                                                                 DS4_N_HEAD_DIM) != 0;
                }
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_cache_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
    }
    DS4_METAL_PROFILE_ATTN_STAGE("attention");

    if (ok) {
        metal_graph_debug_dump_tensor("kqv_out", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_heads,
                                            n_tokens,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("kqv_back", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("inv_rope");
    if (ok) ok = ds4_gpu_attention_output_q8_batch_tensor(g->batch_attn_out,
                                                            g->batch_attn_low,
                                                            g->batch_group_tmp,
                                                            g->batch_low_tmp,
                                                            model->map,
                                                            model->size,
                                                            layer->attn_output_a->abs_offset,
                                                            layer->attn_output_b->abs_offset,
                                                            group_dim,
                                                            rank,
                                                            n_groups,
                                                            DS4_N_EMBD,
                                                            g->batch_heads,
                                                            n_tokens) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("attn_low", g->batch_attn_low,
                                      (uint64_t)n_tokens * n_groups * rank,
                                      il,
                                      pos0);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("attn_out", g->batch_attn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("output_proj");
    if (ok && metal_graph_directional_steering_attn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_attn(g, g->batch_attn_out, il, n_tokens);
    }
    if (ok) ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                  g->batch_attn_out,
                                                  g->batch_cur_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_post", g->batch_after_attn_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("hc_post");
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    free(index_counts);
    free(comp_counts);
#undef DS4_METAL_PROFILE_ATTN_STAGE
#undef DS4_METAL_PROFILE_Q_STAGE
    return ok;
}

/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post. */
static bool metal_graph_encode_layer_ffn_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const bool layer_stage_profile = getenv("DS4_METAL_LAYER_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *ffn_cur_view = ds4_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *next_hc_view = ds4_gpu_tensor_view(
            g->batch_next_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_after_attn_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(hc_mix_view,
                                             model->map,
                                             model->size,
                                             layer->hc_ffn_fn->abs_offset,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (metal_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_ffn_scale->abs_offset,
                                                        layer->hc_ffn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(ffn_cur_view,
                                                            g->batch_after_attn_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_after_attn_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_ffn_scale->abs_offset,
                                                            layer->hc_ffn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", g->batch_ffn_cur,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_pre");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                       g->batch_ffn_cur,
                                                       model->map,
                                                       model->size,
                                                       layer->ffn_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", g->batch_ffn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("norm");
    if (ok) ok = ds4_gpu_matmul_f16_tensor(g->batch_router_logits,
                                             model->map,
                                             model->size,
                                             layer->ffn_gate_inp->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_N_EXPERT,
                                             g->batch_ffn_norm,
                                             n_tokens) != 0;

    if (ok) ok = ds4_gpu_router_select_batch_tensor(g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      g->batch_router_probs,
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      g->batch_router_logits,
                                                      g->prefill_tokens,
                                                      n_tokens) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", g->batch_router_logits,
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_probs", g->batch_router_probs,
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", g->batch_router_selected,
                                          (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->batch_router_weights,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("router");

    if (ok) ok = ds4_gpu_routed_moe_batch_tensor(g->batch_routed_out,
                                                   g->batch_routed_gate,
                                                   g->batch_routed_up,
                                                   g->batch_routed_mid,
                                                   g->batch_routed_down,
                                                   model->map,
                                                   model->size,
                                                   layer->ffn_gate_exps->abs_offset,
                                                   layer->ffn_up_exps->abs_offset,
                                                   layer->ffn_down_exps->abs_offset,
                                                   layer->ffn_gate_exps->type,
                                                   layer->ffn_down_exps->type,
                                                   gate_expert_bytes,
                                                   gate_row_bytes,
                                                   down_expert_bytes,
                                                   down_row_bytes,
                                                   (uint32_t)expert_in_dim,
                                                   (uint32_t)down_in_dim,
                                                   (uint32_t)routed_out_dim,
                                                   g->batch_router_selected,
                                                   g->batch_router_weights,
                                                   DS4_N_EXPERT_USED,
                                                   DS4_SWIGLU_CLAMP_EXP,
                                                   g->batch_ffn_norm,
                                                   n_tokens,
                                                   &g->batch_routed_mid_is_f16) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->batch_routed_gate,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", g->batch_routed_up,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", g->batch_routed_down,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos0);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", g->batch_routed_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("routed_moe");
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_shared_gate,
                                              model->map,
                                              model->size,
                                              layer->ffn_gate_shexp->abs_offset,
                                              DS4_N_EMBD,
                                              shared_dim,
                                              g->batch_ffn_norm,
                                              n_tokens) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_shared_up,
                                              model->map,
                                              model->size,
                                              layer->ffn_up_shexp->abs_offset,
                                              DS4_N_EMBD,
                                              shared_dim,
                                              g->batch_ffn_norm,
                                              n_tokens) != 0;
    DS4_METAL_PROFILE_FFN_STAGE("shared_gate_up");
    if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid,
                                         g->batch_shared_gate,
                                         g->batch_shared_up,
                                         (uint32_t)((uint64_t)n_tokens * shared_dim),
                                         0.0f,
                                         1.0f) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->batch_shared_out,
                                              model->map,
                                              model->size,
                                              layer->ffn_down_shexp->abs_offset,
                                              shared_dim,
                                              DS4_N_EMBD,
                                              g->batch_shared_mid,
                                              n_tokens) != 0;
    DS4_METAL_PROFILE_FFN_STAGE("shared_down");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_shexp", g->batch_shared_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }

    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos0);
    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_batch_ffn_out(g) &&
             ds4_gpu_add_tensor(g->batch_ffn_out,
                                  g->batch_shared_out,
                                  g->batch_routed_out,
                                  (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", g->batch_ffn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(g, g->batch_ffn_out, il, n_tokens);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                              g->batch_ffn_out,
                                              g->batch_after_attn_hc,
                                              hc_split_view,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  g->batch_routed_out,
                                                  g->batch_shared_out,
                                                  g->batch_after_attn_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", g->batch_next_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_post");
    ds4_gpu_tensor_free(next_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef DS4_METAL_PROFILE_FFN_STAGE
    return ok;
}

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
static bool metal_graph_encode_layer_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    bool ok = metal_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    if (ok) ok = metal_graph_encode_layer_ffn_batch(g, model, layer, il, pos0, n_tokens);
    if (ok) {
        ds4_gpu_tensor *tmp = g->batch_cur_hc;
        g->batch_cur_hc = g->batch_next_hc;
        g->batch_next_hc = tmp;
    }
    return ok;
}

/* Execute one Metal decode token and read back logits. */
static bool metal_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    const bool profile = getenv("DS4_METAL_GRAPH_TOKEN_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL, true);
    const double t_encoded = profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = profile ? now_sec() : 0.0;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        fprintf(stderr,
                "ds4: metal graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph eval failure also failed\n");
        }
    }
    return ok;
}

/* Greedy verifier helper.  Speculative decoding only needs the target model's
 * top token after most accepted draft rows; the full vocabulary row is needed
 * once, for the final committed state that normal sampling will continue from.
 * Keeping intermediate rows device-resident avoids turning verification into a
 * sequence of large CPU readbacks. */
static bool metal_graph_eval_token_raw_swa_top(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        int                   *top_id,
        float                 *logits) {
    if (!top_id) return false;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_token_raw_swa(g, model, weights,
                                                  token, pos, true, true);
    if (ok) {
        ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                           g->logits,
                                           DS4_N_VOCAB,
                                           1,
                                           1) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (ok) ok = ds4_gpu_tensor_read(g->comp_selected, 0, top_id, sizeof(*top_id)) != 0;
    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after top-only graph eval failure also failed\n");
        }
    }
    return ok;
}

static bool metal_graph_eval_mtp_draft_from_hc(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        ds4_gpu_tensor      *prev_hc,
        ds4_gpu_tensor      *out_hc,
        int                    token,
        uint32_t               pos,
        float                 *logits,
        int                   *top_id) {
    if (!mtp || !mtp->block.attn_q_a || !g->mtp_raw_cache || !prev_hc || !out_hc) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint32_t raw_row = pos % g->raw_cap;
    uint32_t n_raw = g->mtp_n_raw + 1u;
    if (n_raw > g->raw_window) n_raw = g->raw_window;
    if (n_raw > g->raw_cap) n_raw = g->raw_cap;

    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *saved_after = g->after_ffn_hc;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(g->mtp_embed,
                                                  base_model->map,
                                                  base_model->size,
                                                  base_weights->token_embd->abs_offset,
                                                  (uint32_t)base_weights->token_embd->dim[1],
                                                  (uint32_t)token,
                                                  DS4_N_EMBD,
                                                  1) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->mtp_enorm,
                                                  g->mtp_embed,
                                                  mtp_model->map,
                                                  mtp_model->size,
                                                  mtp->enorm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->mtp_eproj,
                                              mtp_model->map,
                                              mtp_model->size,
                                              mtp->e_proj->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD,
                                              g->mtp_enorm,
                                              1) != 0;
    if (ok) ok = ds4_gpu_repeat_hc_tensor(g->mtp_eproj_hc,
                                            g->mtp_eproj,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mtp_hnorm_hc,
                                                       prev_hc,
                                                       mtp_model->map,
                                                       mtp_model->size,
                                                       mtp->hnorm->abs_offset,
                                                       DS4_N_EMBD,
                                                       DS4_N_HC,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->mtp_hproj_hc,
                                              mtp_model->map,
                                              mtp_model->size,
                                              mtp->h_proj->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD,
                                              g->mtp_hnorm_hc,
                                              DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_add_tensor(g->mtp_input_hc,
                                      g->mtp_eproj_hc,
                                      g->mtp_hproj_hc,
                                      (uint32_t)hc_dim) != 0;
    if (ok) {
        g->cur_hc = g->mtp_input_hc;
        g->after_ffn_hc = out_hc;
        ok = metal_graph_encode_decode_layer(g,
                                             mtp_model,
                                             &mtp->block,
                                             1,
                                             pos,
                                             g->mtp_raw_cache,
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
    }
    if (ok) g->cur_hc = out_hc;
    if (ok) ok = metal_graph_encode_output_head_mtp(g,
                                                    base_model,
                                                    base_weights,
                                                    mtp_model,
                                                    mtp,
                                                    base_weights->output->dim[1]);
    if (ok && top_id) {
        ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                           g->logits,
                                           DS4_N_VOCAB,
                                           1,
                                           1) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok && top_id) {
        ok = ds4_gpu_tensor_read(g->comp_selected, 0, top_id, sizeof(*top_id)) != 0;
    }
    if (ok && g->mtp_n_raw < g->raw_window) g->mtp_n_raw++;
    if (!ok) {
        (void)ds4_gpu_synchronize();
        g->cur_hc = saved_cur;
        g->after_ffn_hc = saved_after;
    }
    return ok;
}

static bool metal_graph_eval_mtp_draft(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        int                    token,
        uint32_t               pos,
        float                 *logits,
        int                   *top_id) {
    return metal_graph_eval_mtp_draft_from_hc(g,
                                              base_model,
                                              base_weights,
                                              mtp_model,
                                              mtp,
                                              g->cur_hc,
                                              g->mtp_state_hc,
                                              token,
                                              pos,
                                              logits,
                                              top_id);
}

/* =========================================================================
 * Imatrix Collection.
 * =========================================================================
 *
 * The 2-bit DS4 quants care most about routed MoE experts.  For expert gate
 * and up matrices the matmul input is the FFN-normalized activation row.  For
 * expert down matrices the matmul input is the routed SwiGLU row after route
 * weighting.  During Metal prefill those tensors are already materialized as
 * `batch_ffn_norm`, `batch_router_selected`, and `batch_routed_mid`, so the
 * collector observes the exact release graph without changing inference math.
 *
 * The output is llama.cpp's legacy imatrix `.dat` format.  Entries are packed
 * by expert: one tensor entry contains `n_expert * n_columns` floats and the
 * quantizer slices the vector for each expert.
 */
typedef struct {
    float *gate_up_sum2;   /* [layer][expert][4096] */
    float *down_sum2;      /* [layer][expert][2048] */
    uint32_t gate_up_count[DS4_N_LAYER][DS4_N_EXPERT];
    uint32_t down_count[DS4_N_LAYER][DS4_N_EXPERT];
    float *ffn_norm_buf;
    float *routed_mid_buf;
    uint16_t *routed_mid_f16_buf;
    int   *selected_buf;
    float *sq_tmp;
    uint32_t cap_tokens;
    uint64_t observed_tokens;
    uint64_t observed_routes;
    uint32_t chunks;
    const char *dataset_path;
} ds4_imatrix_collector;

static bool imatrix_collector_init(ds4_imatrix_collector *c, uint32_t cap_tokens, const char *dataset_path) {
    memset(c, 0, sizeof(*c));
    c->cap_tokens = cap_tokens ? cap_tokens : 1u;
    c->dataset_path = dataset_path;
    const size_t gate_n = (size_t)DS4_N_LAYER * DS4_N_EXPERT * DS4_N_EMBD;
    const size_t down_n = (size_t)DS4_N_LAYER * DS4_N_EXPERT * DS4_N_FF_EXP;
    c->gate_up_sum2 = xcalloc(gate_n, sizeof(c->gate_up_sum2[0]));
    c->down_sum2 = xcalloc(down_n, sizeof(c->down_sum2[0]));
    c->ffn_norm_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EMBD * sizeof(c->ffn_norm_buf[0]));
    c->routed_mid_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(c->routed_mid_buf[0]));
    c->routed_mid_f16_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(c->routed_mid_f16_buf[0]));
    c->selected_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * sizeof(c->selected_buf[0]));
    c->sq_tmp = xmalloc((size_t)DS4_N_EMBD * sizeof(c->sq_tmp[0]));
    return c->gate_up_sum2 && c->down_sum2 && c->ffn_norm_buf &&
           c->routed_mid_buf && c->routed_mid_f16_buf && c->selected_buf && c->sq_tmp;
}

static void imatrix_collector_free(ds4_imatrix_collector *c) {
    if (!c) return;
    free(c->gate_up_sum2);
    free(c->down_sum2);
    free(c->ffn_norm_buf);
    free(c->routed_mid_buf);
    free(c->routed_mid_f16_buf);
    free(c->selected_buf);
    free(c->sq_tmp);
    memset(c, 0, sizeof(*c));
}

static float *imatrix_gate_up_ptr(ds4_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->gate_up_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_EMBD;
}

static float *imatrix_down_ptr(ds4_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->down_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_FF_EXP;
}

static bool imatrix_collect_layer_batch(
        ds4_imatrix_collector *c,
        ds4_gpu_graph         *g,
        uint32_t               il,
        uint32_t               n_tokens) {
    if (!c || n_tokens == 0) return true;
    if (n_tokens > c->cap_tokens) return false;

    const uint64_t norm_bytes = (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
    const uint64_t mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t mid_bytes = mid_elems * (g->batch_routed_mid_is_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t sel_bytes = (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(int);
    void *mid_dst = g->batch_routed_mid_is_f16
        ? (void *)c->routed_mid_f16_buf
        : (void *)c->routed_mid_buf;
    if (ds4_gpu_tensor_read(g->batch_ffn_norm, 0, c->ffn_norm_buf, norm_bytes) == 0 ||
        ds4_gpu_tensor_read(g->batch_routed_mid, 0, mid_dst, mid_bytes) == 0 ||
        ds4_gpu_tensor_read(g->batch_router_selected, 0, c->selected_buf, sel_bytes) == 0)
    {
        return false;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *x = c->ffn_norm_buf + (size_t)t * DS4_N_EMBD;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) c->sq_tmp[i] = x[i] * x[i];

        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const int expert = c->selected_buf[(size_t)t * DS4_N_EXPERT_USED + slot];
            if (expert < 0 || expert >= DS4_N_EXPERT) continue;

            float *gate_up = imatrix_gate_up_ptr(c, il, (uint32_t)expert);
            for (uint32_t i = 0; i < DS4_N_EMBD; i++) gate_up[i] += c->sq_tmp[i];
            c->gate_up_count[il][expert]++;

            float *down = imatrix_down_ptr(c, il, (uint32_t)expert);
            const size_t mid_off = ((size_t)t * DS4_N_EXPERT_USED + slot) * DS4_N_FF_EXP;
            if (g->batch_routed_mid_is_f16) {
                const uint16_t *mid = c->routed_mid_f16_buf + mid_off;
                for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) {
                    const float v = f16_to_f32(mid[i]);
                    down[i] += v * v;
                }
            } else {
                const float *mid = c->routed_mid_buf + mid_off;
                for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) down[i] += mid[i] * mid[i];
            }
            c->down_count[il][expert]++;
            c->observed_routes++;
        }
    }
    c->observed_tokens += n_tokens;
    c->chunks++;
    return true;
}

static void imatrix_write_i32(FILE *fp, int32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) ds4_die("failed to write imatrix");
}

static void imatrix_write_entry(
        FILE       *fp,
        const char *name,
        const float *sum2,
        const uint32_t *counts,
        uint32_t n_expert,
        uint32_t n_col) {
    const int32_t len = (int32_t)strlen(name);
    const int32_t ncall = 1;
    const int32_t nval = (int32_t)((uint64_t)n_expert * n_col);
    imatrix_write_i32(fp, len);
    if (fwrite(name, 1, (size_t)len, fp) != (size_t)len) ds4_die("failed to write imatrix name");
    imatrix_write_i32(fp, ncall);
    imatrix_write_i32(fp, nval);

    float *tmp = xmalloc((size_t)n_col * sizeof(tmp[0]));
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint32_t count = counts[e];
        const float *src = sum2 + (size_t)e * n_col;
        if (count == 0) {
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = 1.0f;
        } else {
            const float inv = 1.0f / (float)count;
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = src[i] * inv;
        }
        if (fwrite(tmp, sizeof(tmp[0]), n_col, fp) != n_col) ds4_die("failed to write imatrix values");
    }
    free(tmp);
}

static bool imatrix_collector_save(
        const ds4_imatrix_collector *c,
        const ds4_weights           *weights,
        const char                  *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }

    const int32_t entries = (int32_t)(DS4_N_LAYER * 3);
    imatrix_write_i32(fp, entries);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        char name[256];
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_gate_exps->name.len, layer->ffn_gate_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_EMBD,
                            c->gate_up_count[il],
                            DS4_N_EXPERT,
                            DS4_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_up_exps->name.len, layer->ffn_up_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_EMBD,
                            c->gate_up_count[il],
                            DS4_N_EXPERT,
                            DS4_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_down_exps->name.len, layer->ffn_down_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->down_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_FF_EXP,
                            c->down_count[il],
                            DS4_N_EXPERT,
                            DS4_N_FF_EXP);
    }

    const int32_t chunks = (int32_t)c->chunks;
    imatrix_write_i32(fp, chunks);
    const char *dataset = c->dataset_path ? c->dataset_path : "";
    const int32_t dataset_len = (int32_t)strlen(dataset);
    imatrix_write_i32(fp, dataset_len);
    if (dataset_len && fwrite(dataset, 1, (size_t)dataset_len, fp) != (size_t)dataset_len) {
        ds4_die("failed to write imatrix dataset name");
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }
    return true;
}

static bool metal_graph_reset_prefill_state(ds4_gpu_graph *g) {
    memset(g->layer_n_index_comp, 0, sizeof(g->layer_n_index_comp));
    g->mtp_n_raw = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        if (!metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows)) return false;
        if (!metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows)) return false;
        if (ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            if (!metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows)) return false;
            if (!metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows)) return false;
        }
    }
    return true;
}

/* Execute Metal prefill in layer-major order so intermediate activations stay
 * on the GPU and cache state is built exactly once. */
static bool metal_graph_prefill_layer_major(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_imatrix_collector *imatrix) {
    if (n_tokens <= 0 || n_tokens > prompt->len || (uint32_t)n_tokens > g->prefill_cap) return false;

    bool ok = metal_graph_upload_prompt_tokens(g->prefill_tokens, prompt, 0, (uint32_t)n_tokens);
    if (!ok) return false;

    if (!metal_graph_warmup_prefill_kernels(g, model, weights, (uint32_t)n_tokens)) return false;

    const bool split_profile = getenv("DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE") != NULL;
    /*
     * A full long-prompt prefill can keep the GPU busy long enough for macOS
     * to watchdog WindowServer. Keep short prompts in one command buffer for
     * low overhead, but submit long prompts layer by layer so the display
     * server gets regular scheduling points.
     */
    const bool split_commands = split_profile || n_tokens > 2048 || imatrix != NULL;
    const bool profile = getenv("DS4_METAL_GRAPH_PREFILL_PROFILE") != NULL || split_profile;
    const double t0 = profile ? now_sec() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;

    if (!split_commands) {
        ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     model,
                                                     weights,
                                                     prompt,
                                                     0,
                                                     (uint32_t)n_tokens);
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_layer_batch(g,
                                                model,
                                                &weights->layer[il],
                                                il,
                                                0,
                                                (uint32_t)n_tokens);
            if (show_progress) {
                fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
                fflush(stderr);
            }
        }
        if (show_progress) fputc('\n', stderr);

        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
        uint32_t output_row = (uint32_t)n_tokens - 1u;
        const char *output_row_env = getenv("DS4_METAL_GRAPH_OUTPUT_ROW");
        if (output_row_env && output_row_env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(output_row_env, &end, 10);
            if (end != output_row_env && v < (unsigned long)n_tokens) {
                output_row = (uint32_t)v;
            }
        }
        ds4_gpu_tensor *last_hc = NULL;
        ds4_gpu_tensor *saved_cur = g->cur_hc;
        if (ok) {
            last_hc = metal_graph_tensor_row_view(g->batch_cur_hc, output_row, hc_dim);
            ok = last_hc != NULL;
        }
        if (ok) {
            g->cur_hc = last_hc;
            ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            g->cur_hc = saved_cur;
        }

        const double t_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = profile ? now_sec() : 0.0;
        g->cur_hc = saved_cur;
        if (last_hc) ds4_gpu_tensor_free(last_hc);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after whole-prefill graph failure also failed\n");
            }
            return false;
        }

        const double t_before_read = profile ? now_sec() : 0.0;
        if (logits) {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (profile) {
            const double t_read = now_sec();
            fprintf(stderr,
                    "ds4: gpu graph prefill total tokens=%d encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                    n_tokens,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_before_read) * 1000.0,
                    (t_read - t0) * 1000.0);
        }
        return ok;
    }

    double t_layer0 = profile ? now_sec() : 0.0;
    ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                 g->prefill_tokens,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 0,
                                                 (uint32_t)n_tokens);
    const double t_embed_encoded = profile ? now_sec() : 0.0;
    const double t_embed_done = profile ? now_sec() : 0.0;
    if (profile) {
        encode_s += t_embed_encoded - t_layer0;
        execute_s += t_embed_done - t_embed_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: metal layer-major prefill embed encode=%.3f ms execute=%.3f ms\n",
                    (t_embed_encoded - t_layer0) * 1000.0,
                    (t_embed_done - t_embed_encoded) * 1000.0);
        }
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after layer-major prefill embed failure also failed\n");
        }
        return false;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (split_profile) {
            const double t_attn0 = now_sec();
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_attention_batch(g,
                                                                  model,
                                                                  &weights->layer[il],
                                                                  il,
                                                                  0,
                                                                  (uint32_t)n_tokens);
            const double t_attn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_attn_done = now_sec();

            const double t_ffn0 = now_sec();
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_ffn_batch(g,
                                                            model,
                                                            &weights->layer[il],
                                                            il,
                                                            0,
                                                            (uint32_t)n_tokens);
            if (ok) {
                ds4_gpu_tensor *tmp = g->batch_cur_hc;
                g->batch_cur_hc = g->batch_next_hc;
                g->batch_next_hc = tmp;
            }
            const double t_ffn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_ffn_done = now_sec();
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);

            encode_s += (t_attn_encoded - t_attn0) + (t_ffn_encoded - t_ffn0);
            execute_s += (t_attn_done - t_attn_encoded) + (t_ffn_done - t_ffn_encoded);
            fprintf(stderr,
                    "ds4: metal layer-major prefill layer %u attn encode=%.3f execute=%.3f ms ffn encode=%.3f execute=%.3f ms\n",
                    il,
                    (t_attn_encoded - t_attn0) * 1000.0,
                    (t_attn_done - t_attn_encoded) * 1000.0,
                    (t_ffn_encoded - t_ffn0) * 1000.0,
                    (t_ffn_done - t_ffn_encoded) * 1000.0);
        } else {
            const double t_chunk0 = profile ? now_sec() : 0.0;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_batch(g,
                                                        model,
                                                        &weights->layer[il],
                                                        il,
                                                        0,
                                                        (uint32_t)n_tokens);
            const double t_encoded = profile ? now_sec() : 0.0;
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_done = profile ? now_sec() : 0.0;
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            if (profile) {
                encode_s += t_encoded - t_chunk0;
                execute_s += t_done - t_encoded;
                fprintf(stderr,
                        "ds4: gpu layer-major prefill layer %u encode=%.3f ms execute=%.3f ms\n",
                        il,
                        (t_encoded - t_chunk0) * 1000.0,
                        (t_done - t_encoded) * 1000.0);
            }
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after layer-major prefill failure also failed\n");
            }
            return false;
        }
        if (show_progress) {
            fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
            fflush(stderr);
        }
    }
    if (show_progress) fputc('\n', stderr);

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    uint32_t output_row = (uint32_t)n_tokens - 1u;
    const char *output_row_env = getenv("DS4_METAL_GRAPH_OUTPUT_ROW");
    if (output_row_env && output_row_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(output_row_env, &end, 10);
        if (end != output_row_env && v < (unsigned long)n_tokens) {
            output_row = (uint32_t)v;
        }
    }
    ds4_gpu_tensor *last_hc = metal_graph_tensor_row_view(g->batch_cur_hc,
                                                            output_row,
                                                            hc_dim);
    if (!last_hc) return false;
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    g->cur_hc = last_hc;

    const double t_head0 = profile ? now_sec() : 0.0;
    ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_head_done = profile ? now_sec() : 0.0;
    g->cur_hc = saved_cur;
    ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? now_sec() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: gpu layer-major prefill head encode=%.3f ms execute=%.3f ms\n",
                    (t_head_encoded - t_head0) * 1000.0,
                    (t_head_done - t_head_encoded) * 1000.0);
        }
        fprintf(stderr,
                "ds4: gpu layer-major prefill total tokens=%d encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                n_tokens,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0,
                (t_read - t0) * 1000.0);
    }
    return ok;
}

static bool metal_graph_prefill_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress) {
    if (n_tokens <= 0 || n_tokens > prompt->len) return false;
    if ((uint32_t)n_tokens > g->prefill_cap) return false;
    return metal_graph_prefill_layer_major(g, model, weights, prompt, n_tokens, logits, show_progress, NULL);
}

static bool metal_graph_prefill_batch_row_logits(
        ds4_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           batch_row,
        float             *logits) {
    if (!logits) return true;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *last_hc = metal_graph_tensor_row_view(g->batch_cur_hc,
                                                            batch_row,
                                                            hc_dim);
    if (!last_hc) return false;
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    g->cur_hc = last_hc;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->cur_hc = saved_cur;
    ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;
    return ds4_gpu_tensor_read(g->logits, 0, logits,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
}

/* Prefill a contiguous token range in fixed-size chunks.
 *
 * The common case starts at token zero, but server sessions also use this to
 * extend an existing KV cache with a long suffix.  Resumed chunks are aligned
 * to the same absolute prefill-cap boundaries used by a cold full prompt, so
 * compression windows and row finalization follow the same schedule after the
 * cached prefix.
 */
static bool metal_graph_prefill_chunked_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_imatrix_collector *imatrix) {
    if (n_tokens == 0 || g->prefill_cap == 0) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    uint32_t chunk_cap = g->prefill_cap;
    if (start != 0 && chunk_cap > g->raw_cap) chunk_cap = g->raw_cap;
    if (chunk_cap == 0) return false;

    uint32_t first_chunk = n_tokens < chunk_cap ? n_tokens : chunk_cap;
    if (start != 0 && g->prefill_cap != 0) {
        const uint32_t mod = start % g->prefill_cap;
        if (mod != 0) {
            const uint32_t to_boundary = g->prefill_cap - mod;
            if (to_boundary < first_chunk) first_chunk = to_boundary;
        }
    }
    if (!metal_graph_warmup_prefill_kernels(g, model, weights, first_chunk)) return false;

    const bool profile = getenv("DS4_METAL_GRAPH_PREFILL_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;
    uint32_t last_chunk_tokens = 0;
    const uint32_t end = start + n_tokens;

    if (progress) {
        progress(progress_ud, "prefill_chunk", (int)start, prompt->len);
    }

    for (uint32_t pos0 = start; pos0 < end; ) {
        const uint32_t remaining = end - pos0;
        uint32_t local_cap = chunk_cap;
        if (start != 0 && g->prefill_cap != 0) {
            const uint32_t mod = pos0 % g->prefill_cap;
            if (mod != 0) {
                const uint32_t to_boundary = g->prefill_cap - mod;
                if (to_boundary < local_cap) local_cap = to_boundary;
            }
        }
        const uint32_t chunk = remaining < local_cap ? remaining : local_cap;
        last_chunk_tokens = chunk;

        bool ok = metal_graph_upload_prompt_tokens(g->prefill_tokens, prompt, pos0, chunk);
        if (ok) ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                             g->prefill_tokens,
                                                             model,
                                                             weights,
                                                             prompt,
                                                             pos0,
                                                             chunk);
        if (!ok) return false;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            const double t_layer0 = profile ? now_sec() : 0.0;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_batch(g,
                                                        model,
                                                        &weights->layer[il],
                                                        il,
                                                        pos0,
                                                        chunk);
            const double t_encoded = profile ? now_sec() : 0.0;
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_done = profile ? now_sec() : 0.0;
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, chunk);
            if (profile) {
                encode_s += t_encoded - t_layer0;
                execute_s += t_done - t_encoded;
                fprintf(stderr,
                        "ds4: gpu chunked prefill pos=%u tokens=%u layer %u encode=%.3f ms execute=%.3f ms\n",
                        pos0,
                        chunk,
                        il,
                        (t_encoded - t_layer0) * 1000.0,
                        (t_done - t_encoded) * 1000.0);
            }
            if (show_progress) {
                fprintf(stderr,
                        "ds4: gpu prefill token %u/%u layer %u/%u\r",
                        pos0 + chunk,
                        (uint32_t)prompt->len,
                        il + 1,
                        (uint32_t)DS4_N_LAYER);
                fflush(stderr);
            }
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after chunked prefill failure also failed\n");
            }
            return false;
        }
        if (progress && !metal_graph_prefill_batch_row_logits(g, model, weights,
                                                              chunk - 1u,
                                                              logits))
        {
            return false;
        }
        if (progress) {
            progress(progress_ud, "prefill_chunk", (int)(pos0 + chunk), prompt->len);
        }
        pos0 += chunk;
    }
    if (show_progress) fputc('\n', stderr);
    if (last_chunk_tokens == 0) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *last_hc = metal_graph_tensor_row_view(g->batch_cur_hc,
                                                            last_chunk_tokens - 1u,
                                                            hc_dim);
    if (!last_hc) return false;
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    g->cur_hc = last_hc;

    const double t_head0 = profile ? now_sec() : 0.0;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_head_done = profile ? now_sec() : 0.0;
    g->cur_hc = saved_cur;
    ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? now_sec() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        fprintf(stderr,
                "ds4: gpu chunked prefill start=%u tokens=%u chunk=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                start,
                n_tokens,
                chunk_cap,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0,
                (t_read - t0) * 1000.0);
    }
    return ok;
}

/* Long prompts are prefetched in fixed-size chunks.  Chunks bound transient
 * attention buffers while preserving the same final KV/cache state. */
static bool metal_graph_prefill_chunked(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud) {
    if (n_tokens <= 0) return false;
    return metal_graph_prefill_chunked_range(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             (uint32_t)n_tokens,
                                             logits,
                                             show_progress,
                                             progress,
                                             progress_ud,
                                             NULL);
}

/* Layer-major speculative target verifier for tiny MTP suffixes.
 *
 * This is the first production-shaped verifier attempt: unlike repeated decode
 * it runs the target model layer-by-layer for the whole speculative suffix, and
 * unlike the diagnostic path it does not read back full logits for every row.
 * The verifier returns the row top-1 ids needed for acceptance.  The caller
 * then reads exactly one logits row: the row that becomes the new continuation
 * state.  It still reuses the existing batch layer kernels, so it is not yet
 * the final hand-written N=2/N=4 decode microbatch, but it exercises the right
 * verifier contract and removes the obvious diagnostic overheads first. */
static bool metal_graph_verify_suffix_tops(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        bool                   capture_prefix1,
        int                   *row_tops,
        float                 *row_logits) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
    if (start > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - start) return false;
    const uint32_t top_rows = n_tokens > 1 ? n_tokens - 1 : 0;
    if (top_rows && !row_tops) return false;

    bool ok = metal_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens);
    if (ok) ok = metal_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                         g->prefill_tokens,
                                                         model,
                                                         weights,
                                                         prompt,
                                                         start,
                                                         n_tokens);
    if (!ok) return false;

    const bool saved_capture = g->spec_capture_prefix1;
    g->spec_capture_prefix1 = capture_prefix1 && n_tokens == 2;

    ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = metal_graph_encode_layer_batch(g,
                                            model,
                                            &weights->layer[il],
                                            il,
                                            start,
                                            n_tokens);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->spec_capture_prefix1 = saved_capture;
    if (!ok) return false;

    ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_output_head_batch(g,
                                                      model,
                                                      weights,
                                                      n_tokens,
                                                      weights->output->dim[1]);
    if (ok) {
        if (top_rows) {
            ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                               g->spec_logits,
                                               DS4_N_VOCAB,
                                               1,
                                               top_rows) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok && top_rows) {
        ok = ds4_gpu_tensor_read(g->comp_selected,
                                   0,
                                   row_tops,
                                   (uint64_t)top_rows * sizeof(row_tops[0])) != 0;
    }
    if (ok && row_logits) {
        ok = ds4_gpu_tensor_read(g->spec_logits,
                                   0,
                                   row_logits,
                                   (uint64_t)n_tokens * DS4_N_VOCAB * sizeof(row_logits[0])) != 0;
    }
    return ok;
}

static bool metal_graph_read_spec_logits_row(ds4_gpu_graph *g, uint32_t row, float *logits) {
    if (!g || !g->spec_logits || !logits || row >= g->prefill_cap) return false;
    const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    return ds4_gpu_tensor_read(g->spec_logits,
                                 (uint64_t)row * row_bytes,
                                 logits,
                                 row_bytes) != 0;
}

/* Exact N=2 target verifier for MTP.
 *
 * The generic batch prefill path is fast, but it is not a safe substitute for
 * autoregressive decode: small row-wise differences in HC/MoE/output kernels
 * are enough to flip future greedy tokens.  This verifier keeps the exact
 * decode kernels and cache update order, but encodes the two proposed tokens
 * layer-by-layer in one command stream.  It returns the exact target top after
 * token0, and exact logits after token1. */
static bool metal_graph_verify_decode2_exact(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token0,
        int                    token1,
        uint32_t               start,
        int                   *top0,
        float                 *logits0,
        float                 *logits1) {
    if (!g || !top0 || !logits1 || g->raw_cap == 0) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *cur0 = metal_graph_tensor_row_view(g->batch_cur_hc, 0, hc_dim);
    ds4_gpu_tensor *cur1 = metal_graph_tensor_row_view(g->batch_cur_hc, 1, hc_dim);
    ds4_gpu_tensor *next0 = metal_graph_tensor_row_view(g->batch_next_hc, 0, hc_dim);
    ds4_gpu_tensor *next1 = metal_graph_tensor_row_view(g->batch_next_hc, 1, hc_dim);
    bool ok = cur0 && cur1 && next0 && next1;

    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur0,
                                                  model->map,
                                                  model->size,
                                                  weights->token_embd->abs_offset,
                                                  (uint32_t)weights->token_embd->dim[1],
                                                  (uint32_t)token0,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur1,
                                                  model->map,
                                                  model->size,
                                                  weights->token_embd->abs_offset,
                                                  (uint32_t)weights->token_embd->dim[1],
                                                  (uint32_t)token1,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;

    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *saved_after = g->after_ffn_hc;
    const bool saved_capture = g->spec_capture_prefix1;
    g->spec_capture_prefix1 = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t pos0 = start;
        const uint32_t pos1 = start + 1u;

        g->cur_hc = cur0;
        g->after_ffn_hc = next0;
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos0,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos0 % g->raw_cap,
                                             metal_graph_raw_span_for_batch(g, pos0, 1),
                                             token0);
        if (!ok) break;
        ok = metal_graph_capture_prefix1_attn_state(g, il) &&
             metal_graph_capture_prefix1_index_state(g, il);
        if (!ok) break;

        g->cur_hc = cur1;
        g->after_ffn_hc = next1;
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos1,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos1 % g->raw_cap,
                                             metal_graph_raw_span_for_batch(g, pos1, 1),
                                             token1);
        if (!ok) break;

        ds4_gpu_tensor *tmp = cur0; cur0 = next0; next0 = tmp;
        tmp = cur1; cur1 = next1; next1 = tmp;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->spec_capture_prefix1 = saved_capture;
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;

    if (ok) {
        g->cur_hc = cur0;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                   g->logits,
                                                   DS4_N_VOCAB,
                                                   1,
                                                   1) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        g->cur_hc = saved_cur;
        if (ok) ok = ds4_gpu_tensor_read(g->comp_selected, 0, top0, sizeof(*top0)) != 0;
        if (ok && logits0) {
            ok = ds4_gpu_tensor_read(g->logits,
                                       0,
                                       logits0,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits0[0])) != 0;
        }
    }

    if (ok) {
        g->cur_hc = cur1;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        g->cur_hc = saved_cur;
        if (ok) {
            ok = ds4_gpu_tensor_read(g->logits,
                                       0,
                                       logits1,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits1[0])) != 0;
        }
    }
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;
    g->spec_capture_prefix1 = saved_capture;

    ds4_gpu_tensor_free(next1);
    ds4_gpu_tensor_free(next0);
    ds4_gpu_tensor_free(cur1);
    ds4_gpu_tensor_free(cur0);
    return ok;
}

/* Pick a raw SWA cache size for Metal.  During batched prefill it must cover
 * the previous window plus the current ubatch. */
static uint32_t metal_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap) {
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > (uint32_t)ctx_size) raw_window = (uint32_t)ctx_size;
    if (raw_window == 0) raw_window = 1;

    /*
     * During batched prefill the SWA cache must hold the current ubatch plus
     * the previous logical window. The cache is padded to a 256-row multiple
     * so the physical row order and FlashAttention block grouping match the
     * model path we compare against.
     */
    uint64_t wanted = (uint64_t)raw_window + prefill_cap;
    if (wanted > (uint32_t)ctx_size) wanted = (uint32_t)ctx_size;
    if (wanted == 0) wanted = 1;
    wanted = align_up(wanted, 256u);
    if (wanted > 8192u) wanted = 8192u;
    uint32_t raw_cap = (uint32_t)wanted;
    if (raw_cap < raw_window) raw_cap = raw_window;

    const char *env = getenv("DS4_METAL_GRAPH_RAW_CAP");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env && v > 0) {
            raw_cap = (uint32_t)v;
            if (raw_cap > (uint32_t)ctx_size) raw_cap = (uint32_t)ctx_size;
            if (raw_cap > 8192u) raw_cap = 8192u;
            if (raw_cap < raw_window) raw_cap = raw_window;
        }
    }

    return raw_cap;
}

/* Choose the prefill ubatch size.  Whole-batch is fastest for normal prompts;
 * long prompts default to 2048-token chunks. */
static uint32_t metal_graph_prefill_cap_for_prompt(int prompt_len) {
    return ds4_default_prefill_cap_for_prompt(prompt_len);
}

/* When a server request shares a large prefix with the live checkpoint, extend
 * the KV cache with batched prefill instead of single-token decode.  The env
 * knob is useful while tuning the crossover point for different Macs. */
static uint32_t metal_graph_resume_prefill_min_tokens(void) {
    const char *env = getenv("DS4_METAL_RESUME_PREFILL_MIN");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
    return 32u;
}

ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size) {
    ds4_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    if (ds4_backend_uses_graph(backend)) {
        m.prefill_cap = metal_graph_prefill_cap_for_prompt((int)ctx);
        m.raw_cap = metal_graph_raw_cap_for_context((int)ctx, m.prefill_cap);

        uint32_t min_ratio = UINT32_MAX;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
        }
        if (min_ratio == UINT32_MAX) min_ratio = ctx;
        m.comp_cap = ctx / min_ratio + 2u;
        if (m.comp_cap < 2u) m.comp_cap = 2u;

        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t layer_comp_cap = ctx / ratio + 2u;
            m.compressed_bytes += (uint64_t)layer_comp_cap *
                                  DS4_N_HEAD_DIM *
                                  sizeof(float);
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)layer_comp_cap *
                                      DS4_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        m.scratch_bytes = 2ull *
                          m.comp_cap *
                          m.prefill_cap *
                          sizeof(float);
    } else {
        m.raw_cap = ds4_default_raw_cap(ctx);
        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t comp_cap = ctx / ratio + 2u;
            if (ratio == 4) m.comp_cap = comp_cap;
            m.compressed_bytes += (uint64_t)comp_cap *
                                  DS4_N_HEAD_DIM *
                                  sizeof(float);
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)comp_cap *
                                      DS4_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        if (m.comp_cap == 0) m.comp_cap = ctx / 4u + 2u;
        m.scratch_bytes = ((uint64_t)(m.raw_cap + m.comp_cap) * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(bool));
    }

    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

static int metal_graph_prompt_logits_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        int                ctx_size) {
    int n_test = prompt->len;
    const char *n_test_env = getenv("DS4_METAL_GRAPH_PROMPT_TOKENS");
    if (n_test_env && n_test_env[0]) {
        char *endp = NULL;
        const long v = strtol(n_test_env, &endp, 10);
        if (endp != n_test_env && v > 0 && v <= prompt->len) n_test = (int)v;
    }

    if (n_test <= 0 || n_test > ctx_size) {
        fprintf(stderr, "ds4: Metal graph prompt test needs 1..%d prompt tokens\n", ctx_size);
        return 1;
    }

    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, (uint32_t)n_test);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, (uint32_t)n_test, false);
    if (!ok) {
        metal_graph_free(&g);
        fprintf(stderr, "ds4: failed to initialize Metal graph prompt test runtime\n");
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after graph alloc");

    ds4_kv_cache cpu_cache;
    kv_cache_init(&cpu_cache, (uint32_t)ctx_size, raw_cap);
    float *cpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *gpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *oracle_logits = NULL;

    const char *oracle_path = getenv("DS4_ORACLE_LOGITS");
    if (oracle_path && oracle_path[0]) {
        oracle_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (!read_f32_binary_file(oracle_path, oracle_logits, DS4_N_VOCAB)) {
            free(oracle_logits);
            oracle_logits = NULL;
        }
    }

    for (int t = 0; t < n_test; t++) {
        const bool last = t == n_test - 1;
        forward_token_raw_swa_cpu(last ? cpu_logits : NULL,
                                  model,
                                  weights,
                                  &cpu_cache,
                                  prompt->v[t],
                                  (uint32_t)t);
    }
    ok = metal_graph_prefill_raw_swa(&g, model, weights, prompt, n_test, gpu_logits, true);
    if (memory_report) ds4_gpu_print_memory_report("after prompt graph");

    if (ok) {
        const char *dump_gpu = getenv("DS4_METAL_GRAPH_DUMP_LOGITS");
        if (dump_gpu && dump_gpu[0]) {
            if (write_f32_binary_file(dump_gpu, gpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote Metal graph logits to %s\n", dump_gpu);
            }
        }
        const char *dump_cpu = getenv("DS4_CPU_DUMP_LOGITS");
        if (dump_cpu && dump_cpu[0]) {
            if (write_f32_binary_file(dump_cpu, cpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote CPU logits to %s\n", dump_cpu);
            }
        }
        if (getenv("DS4_METAL_GRAPH_TRACE_CACHE") != NULL ||
            getenv("DS4_METAL_GRAPH_TRACE_COMP") != NULL) {
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                const uint32_t n_raw = cpu_cache.layer[il].n_raw;
                if (n_raw != 0) {
                    const uint64_t raw_phys_n = (uint64_t)raw_cap * DS4_N_HEAD_DIM;
                    const uint64_t raw_logical_n = (uint64_t)n_raw * DS4_N_HEAD_DIM;
                    const uint32_t raw_start = n_raw < raw_cap ? 0u : ((uint32_t)n_test % raw_cap);
                    float *gpu_raw_phys = xmalloc((size_t)raw_phys_n * sizeof(float));
                    float *gpu_raw_logical = xmalloc((size_t)raw_logical_n * sizeof(float));
                    if (ds4_gpu_tensor_read(g.layer_raw_cache[il], 0, gpu_raw_phys, raw_phys_n * sizeof(float)) != 0) {
                        for (uint32_t r = 0; r < n_raw; r++) {
                            const uint32_t phys = (raw_start + r) % raw_cap;
                            memcpy(gpu_raw_logical + (uint64_t)r * DS4_N_HEAD_DIM,
                                   gpu_raw_phys + (uint64_t)phys * DS4_N_HEAD_DIM,
                                   (size_t)DS4_N_HEAD_DIM * sizeof(float));
                        }
                        fprintf(stderr,
                                "ds4: cache trace layer %u raw_n=%u raw_start=%u raw_max=%g raw_rms=%g\n",
                                il, n_raw, raw_start,
                                max_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n),
                                rms_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n));
                    }
                    free(gpu_raw_logical);
                    free(gpu_raw_phys);
                }

                const uint32_t n_comp = cpu_cache.layer[il].n_comp;
                if (n_comp == 0) continue;
                const uint64_t n = (uint64_t)n_comp * DS4_N_HEAD_DIM;
                float *gpu_comp = xmalloc((size_t)n * sizeof(float));
                if (ds4_gpu_tensor_read(g.layer_attn_comp_cache[il], 0, gpu_comp, n * sizeof(float)) != 0) {
                    fprintf(stderr,
                            "ds4: comp trace layer %u n=%u attn_max=%g attn_rms=%g\n",
                            il, n_comp,
                            max_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n),
                            rms_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n));
                }
                free(gpu_comp);

                const uint32_t n_index = cpu_cache.layer[il].n_index_comp;
                if (n_index != 0 && g.layer_index_comp_cache[il]) {
                    const uint64_t ni = (uint64_t)n_index * DS4_N_INDEXER_HEAD_DIM;
                    float *gpu_index = xmalloc((size_t)ni * sizeof(float));
                    if (ds4_gpu_tensor_read(g.layer_index_comp_cache[il], 0, gpu_index, ni * sizeof(float)) != 0) {
                        fprintf(stderr,
                                "ds4: comp trace layer %u n=%u index_max=%g index_rms=%g\n",
                                il, n_index,
                                max_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni),
                                rms_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni));
                    }
                    free(gpu_index);
                }
            }
        }
        const uint64_t cpu_top = argmax_f32(cpu_logits, DS4_N_VOCAB);
        const uint64_t gpu_top = argmax_f32(gpu_logits, DS4_N_VOCAB);
        fprintf(stderr,
                "ds4: Metal prompt graph logits: tokens=%d logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                n_test,
                max_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                rms_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
        if (oracle_logits) {
            const uint64_t oracle_top = argmax_f32(oracle_logits, DS4_N_VOCAB);
            fprintf(stderr,
                    "ds4: oracle logits: tokens=%d oracle_top=%llu oracle_top_logit=%g cpu_max=%g cpu_rms=%g metal_max=%g metal_rms=%g\n",
                    n_test,
                    (unsigned long long)oracle_top,
                    oracle_logits[oracle_top],
                    max_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    max_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB));
        }
    } else {
        fprintf(stderr, "ds4: Metal prompt graph logits test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after prompt graph failure also failed\n");
        }
    }

    free(gpu_logits);
    free(cpu_logits);
    free(oracle_logits);
    kv_cache_free(&cpu_cache);
    metal_graph_free(&g);
    return ok ? 0 : 1;
}

#endif

typedef struct ds4_vocab ds4_vocab;

static void embed_prompt(
        const ds4_model   * model,
        const ds4_weights * weights,
        const token_vec   * tokens,
        uint32_t            n_embd,
        float             * out) {
    for (int i = 0; i < tokens->len; i++) {
        embed_token_f16(model, weights, tokens->v[i], out + (uint64_t)i * n_embd);
    }
}

/* =========================================================================
 * Tokenizer and Chat Prompt Encoding.
 * =========================================================================
 *
 * DeepSeek V4 Flash stores a GPT-2 style byte-level BPE tokenizer in GGUF.
 * The implementation below is intentionally small.  It loads token strings
 * and merge ranks from the mmaped file, builds two open-addressed hash tables,
 * and applies BPE to user text.  Chat special tokens are inserted directly by
 * ID; user text goes through BPE.
 */

typedef struct {
    ds4_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, ds4_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (ds4_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        ds4_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

static void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

static void token_vec_free(token_vec *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

void ds4_tokens_push(ds4_tokens *tv, int token) {
    token_vec_push(tv, token);
}

void ds4_tokens_free(ds4_tokens *tv) {
    token_vec_free(tv);
}

void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) token_vec_push(dst, src->v[i]);
}

bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

struct ds4_vocab {
    ds4_str *token;
    int n_vocab;
    int bos_id;
    int eos_id;
    int user_id;
    int assistant_id;
    int think_start_id;
    int think_end_id;
    int dsml_id;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
};

struct ds4_engine {
    ds4_model model;
    ds4_model mtp_model;
    ds4_vocab vocab;
    ds4_weights weights;
    ds4_mtp_weights mtp_weights;
    ds4_backend backend;
    int mtp_draft_tokens;
    float mtp_margin;
    char *directional_steering_file;
    float *directional_steering_dirs;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    bool quality;
    bool metal_ready;
    bool mtp_ready;
};

static bool cpu_directional_steering_enabled(
        const float *dirs,
        float        scale) {
    return dirs && scale != 0.0f;
}

static void cpu_directional_steering_project_rows(
        float       *x,
        const float *dirs,
        uint32_t     il,
        uint32_t     rows,
        float        scale) {
    if (!cpu_directional_steering_enabled(dirs, scale) || !x || rows == 0) return;

    const float *dir = dirs + (uint64_t)il * DS4_N_EMBD;
    for (uint32_t row = 0; row < rows; row++) {
        float *xr = x + (uint64_t)row * DS4_N_EMBD;
        float dot = 0.0f;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
            dot += xr[i] * dir[i];
        }
        const float coeff = scale * dot;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
            xr[i] -= coeff * dir[i];
        }
    }
}

static bool cpu_load_directional_steering(ds4_engine *e) {
    if (!e ||
        (e->directional_steering_attn_scale == 0.0f &&
         e->directional_steering_ffn_scale == 0.0f)) {
        return true;
    }

    const char *path = e->directional_steering_file;
    if (!path || !path[0]) {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        return false;
    }

    const uint64_t n = (uint64_t)DS4_N_LAYER * DS4_N_EMBD;
    e->directional_steering_dirs = xmalloc((size_t)n * sizeof(e->directional_steering_dirs[0]));
    if (!read_f32_binary_file(path, e->directional_steering_dirs, n)) {
        free(e->directional_steering_dirs);
        e->directional_steering_dirs = NULL;
        fprintf(stderr, "ds4: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    fprintf(stderr, "ds4: CPU directional steering enabled: %s attn=%g ffn=%g\n",
            path,
            (double)e->directional_steering_attn_scale,
            (double)e->directional_steering_ffn_scale);
    return true;
}

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }

    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

/* GPT-2 byte-level BPE first maps raw bytes to printable Unicode codepoints
 * so merges can operate on UTF-8 strings without losing byte identity. */
static char *byte_encode(ds4_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;

    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

/* Look up the merge rank for two adjacent BPE symbols. */
static int bpe_rank(const ds4_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}

/* Apply byte-level BPE to one regex-like pre-tokenized piece and emit token ids. */
static void bpe_emit_piece(const ds4_vocab *vocab, ds4_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool joyai_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static bool utf8_is_cjk_hira_kata(uint32_t cp) {
    return (cp >= 0x4e00 && cp <= 0x9fa5) ||
           (cp >= 0x3040 && cp <= 0x309f) ||
           (cp >= 0x30a0 && cp <= 0x30ff);
}

static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

static bool joyai_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);

    /*
     * The JoyAI tokenizer maps Unicode letters into a collapsed regex alphabet before
     * applying the JoyAI pre-tokenizer.  The prompts we care about are mostly
     * ASCII, but treating non-ASCII non-control bytes as letters preserves the
     * useful behavior for ordinary UTF-8 text such as Italian accents.  CJK and
     * kana are isolated by the JoyAI pre-tokenizer before the generic letter
     * rule, below.
     */
    return true;
}

static uint64_t joyai_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && joyai_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}

static bool joyai_cjk_at(const char *s, uint64_t len, uint64_t pos) {
    if ((uint8_t)s[pos] < 128) return false;
    uint64_t next = pos;
    uint32_t cp = utf8_peek_one(s, len, pos, &next);
    return utf8_is_cjk_hira_kata(cp);
}

/*
 * DeepSeek V4 Flash declares tokenizer.ggml.pre = "joyai-llm".  The split
 * below mirrors the JoyAI BPE pre-tokenizer for the cases this model
 * uses in normal text and source-code prompts:
 *
 *   \p{N}{1,3}
 *   [CJK/Hiragana/Katakana]+
 *   [P/S][A-Za-z]+
 *   [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
 *    ?[\p{P}\p{S}]+[\r\n]*
 *   \s*[\r\n]+
 *   \s+(?!\S)
 *   \s+
 *
 * The punctuation rule intentionally keeps trailing newlines in the same BPE
 * word (for example ">;\n").  Splitting those newlines separately changes the
 * token stream for code prompts and produces wrong long-context logits.
 */
/* JoyAI/DeepSeek pre-tokenization.  The split shape matters: different pieces
 * lead to different BPE merges even when the final text bytes are identical. */
static void bpe_tokenize_text(const ds4_vocab *vocab, const char *text, token_vec *out) {
    const uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (joyai_cjk_at(text, len, pos)) {
            do {
                pos = next_utf8_char(text, len, pos);
            } while (pos < len && joyai_cjk_at(text, len, pos));
        } else if (joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   ascii_alpha((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
        } else if (joyai_letter_like_at(text, len, pos)) {
            pos = joyai_consume_letters(text, len, pos);
        } else if (!ascii_newline(c) &&
                   !joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   joyai_letter_like_at(text, len, pos + 1)) {
            pos++;
            pos = joyai_consume_letters(text, len, pos);
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   joyai_ascii_punct_symbol((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (joyai_ascii_punct_symbol(c)) {
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (joyai_letter_like_at(text, len, p) ||
                        joyai_ascii_punct_symbol((uint8_t)text[p]))) {
                /*
                 * JoyAI lets a single leading space join the following word or
                 * punctuation run.  For "    int", the pre-tokenizer therefore emits
                 * "   " then " int", not "    " then "int".
                 */
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
    }
}

static int vocab_lookup(const ds4_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "ds4: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}

/* Load token strings, special token ids, and merge ranks from GGUF metadata. */
static void vocab_load(ds4_vocab *vocab, const ds4_model *model) {
    memset(vocab, 0, sizeof(*vocab));

    ds4_array_ref tokens;
    ds4_array_ref merges;
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING ||
        tokens.len > INT32_MAX) {
        ds4_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING) {
        ds4_die("GGUF tokenizer merge table is missing or invalid");
    }

    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);

    ds4_cursor c = cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!cursor_string(&c, &vocab->token[i])) ds4_die(c.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }

    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        ds4_str merge;
        if (!cursor_string(&c, &merge)) ds4_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }

    vocab->bos_id       = vocab_lookup(vocab, "<｜begin▁of▁sentence｜>");
    vocab->eos_id       = vocab_lookup(vocab, "<｜end▁of▁sentence｜>");
    vocab->user_id      = vocab_lookup(vocab, "<｜User｜>");
    vocab->assistant_id = vocab_lookup(vocab, "<｜Assistant｜>");
    vocab->think_start_id = vocab_lookup(vocab, "<think>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");
    vocab->dsml_id = vocab_lookup(vocab, "｜DSML｜");
}

static void vocab_free(ds4_vocab *vocab) {
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

/* Build the DS4 chat prompt: BOS, optional system text, user prompt, assistant
 * marker, and either <think> or </think> depending on the requested mode.  Max
 * thinking is only a prompt prefix: the model still enters through <think>. */
static void encode_chat_prompt(
        const ds4_vocab *vocab,
        const char      *system,
        const char      *prompt,
        ds4_think_mode   think_mode,
        token_vec       *out) {
    token_vec_push(out, vocab->bos_id);
    if (think_mode == DS4_THINK_MAX) {
        bpe_tokenize_text(vocab, DS4_REASONING_EFFORT_MAX_PREFIX, out);
    }
    if (system && system[0]) {
        bpe_tokenize_text(vocab, system, out);
    }
    token_vec_push(out, vocab->user_id);
    bpe_tokenize_text(vocab, prompt, out);
    token_vec_push(out, vocab->assistant_id);
    if (ds4_think_mode_enabled(think_mode)) {
        token_vec_push(out, vocab->think_start_id);
    } else {
        token_vec_push(out, vocab->think_end_id);
    }
}

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    bpe_tokenize_text(&e->vocab, text ? text : "", out);
}

static bool special_token_at(const ds4_vocab *vocab, const char *p, int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<｜begin▁of▁sentence｜>", vocab->bos_id},
        {"<｜end▁of▁sentence｜>",   vocab->eos_id},
        {"<｜User｜>",              vocab->user_id},
        {"<｜Assistant｜>",         vocab->assistant_id},
        {"<think>",                vocab->think_start_id},
        {"</think>",               vocab->think_end_id},
        {"｜DSML｜",                vocab->dsml_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const ds4_vocab *vocab, const char *p, size_t n, token_vec *out) {
    if (!n) return;
    char *tmp = xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}

static void tokenize_rendered_chat_vocab(const ds4_vocab *vocab, const char *text,
                                         token_vec *out) {
    if (!text) text = "";

    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (special_token_at(vocab, p, &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            token_vec_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    tokenize_rendered_chat_vocab(&e->vocab, text, out);
}

void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens) {
    token_vec_push(tokens, e->vocab.bos_id);
}

void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out) {
    encode_chat_prompt(&e->vocab, system, prompt ? prompt : "", think_mode, out);
}

void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens) {
    bpe_tokenize_text(&e->vocab, DS4_REASONING_EFFORT_MAX_PREFIX, tokens);
}

void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content) {
    ds4_vocab *vocab = &e->vocab;
    if (!role) role = "user";
    if (!content) content = "";

    if (!strcmp(role, "system") || !strcmp(role, "developer")) {
        bpe_tokenize_text(vocab, content, tokens);
    } else if (!strcmp(role, "assistant")) {
        token_vec_push(tokens, vocab->assistant_id);
        if (strncmp(content, "<think>", 7) != 0 && strncmp(content, "</think>", 8) != 0) {
            token_vec_push(tokens, vocab->think_end_id);
        }
        bpe_tokenize_text(vocab, content, tokens);
    } else {
        token_vec_push(tokens, vocab->user_id);
        if (!strcmp(role, "tool") || !strcmp(role, "function")) {
            bpe_tokenize_text(vocab, "Tool: ", tokens);
        }
        bpe_tokenize_text(vocab, content, tokens);
    }
}

void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode) {
    token_vec_push(tokens, e->vocab.assistant_id);
    token_vec_push(tokens, ds4_think_mode_enabled(think_mode) ?
                   e->vocab.think_start_id : e->vocab.think_end_id);
}

static void dump_tokens_fp(FILE *fp, const ds4_vocab *vocab, const token_vec *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");

    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%6d  %.*s\n", id, (int)vocab->token[id].len, vocab->token[id].ptr);
        }
    }
}

static void dump_tokens(const ds4_vocab *vocab, const token_vec *tokens) {
    dump_tokens_fp(stdout, vocab, tokens);
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

static bool vocab_token_is_literal_special(ds4_str s) {
    const unsigned char bar[] = {0xef, 0xbd, 0x9c}; /* U+FF5C fullwidth vertical bar. */
    if (s.len < sizeof(bar)) return false;
    for (uint64_t i = 0; i + sizeof(bar) <= s.len; i++) {
        if (!memcmp(s.ptr + i, bar, sizeof(bar))) return true;
    }
    return false;
}

char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    ds4_vocab *vocab = &e->vocab;
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = xmalloc(1);
        out[0] = '\0';
        return out;
    }

    ds4_str s = vocab->token[token];
    char *out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }

    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

int ds4_token_eos(ds4_engine *e) {
    return e->vocab.eos_id;
}

static int sample_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

static DS4_MAYBE_UNUSED void logits_top2(const float *logits, uint32_t n_vocab,
                        int *top0, float *logit0,
                        int *top1, float *logit1) {
    int b0 = -1, b1 = -1;
    float v0 = DS4_NEG_INF, v1 = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > v0) {
            b1 = b0; v1 = v0;
            b0 = (int)i; v0 = v;
        } else if (v > v1) {
            b1 = (int)i; v1 = v;
        }
    }
    if (top0) *top0 = b0;
    if (logit0) *logit0 = v0;
    if (top1) *top1 = b1;
    if (logit1) *logit1 = v1;
}

static uint64_t sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float sample_rng_f32(uint64_t *state) {
    const uint64_t x = sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

typedef struct {
    int id;
    float logit;
    float prob;
} sample_candidate;

static int sample_candidate_cmp_desc(const void *a, const void *b) {
    const sample_candidate *ca = a;
    const sample_candidate *cb = b;
    return (cb->logit > ca->logit) - (cb->logit < ca->logit);
}

static int sample_full_vocab(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    float max_logit = DS4_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    sample_candidate *cand = xmalloc((size_t)finite * sizeof(cand[0]));
    uint32_t n = 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = p};
        sum += p;
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }

    qsort(cand, n, sizeof(cand[0]), sample_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) {
        free(cand);
        return best;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}

static int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) return sample_full_vocab(logits, n_vocab, temperature, top_p, min_p, rng);
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    int ids[1024];
    float vals[1024];
    int n = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == top_k && v <= vals[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = (int)i;
    }
    if (n == 0) return sample_argmax(logits, n_vocab);

    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}

static void print_top_logits(
        FILE          * fp,
        const char    * label,
        const ds4_vocab * vocab,
        const float   * logits,
        uint32_t        n_vocab,
        int             k) {
    int best[16];
    if (k > 16) k = 16;
    for (int i = 0; i < k; i++) best[i] = -1;

    for (uint32_t i = 0; i < n_vocab; i++) {
        for (int j = 0; j < k; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int l = k - 1; l > j; l--) best[l] = best[l - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    fprintf(fp, "ds4: top logits %s:\n", label);
    for (int i = 0; i < k && best[i] >= 0; i++) {
        const int id = best[i];
        fprintf(fp, "  %2d %7d % .9g  ", i, id, logits[id]);
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%.*s", (int)vocab->token[id].len, vocab->token[id].ptr);
        }
        fputc('\n', fp);
    }
}

/* CPU generation entry point.  It runs layer-major prefill once, then decodes
 * one token at a time using the persistent KV cache and scratch arena. */
static int generate_raw_swa_cpu(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        const float       * directional_steering_dirs,
        float               directional_steering_attn,
        float               directional_steering_ffn,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    (void)progress;
    (void)progress_ud;
    fprintf(stderr, "ds4: using CPU generation with layer-major prefill\n");

    ds4_kv_cache cache;
    kv_cache_init(&cache, (uint32_t)ctx_size, 0);
    ds4_cpu_decode_scratch decode_scratch;
    cpu_decode_scratch_init(&decode_scratch, (uint32_t)ctx_size);

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    int pos = prompt->len;
    const bool trace_top = getenv("DS4_TRACE_TOP") != NULL;
    const double t_prefill0 = now_sec();

    if (prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        free(logits);
        cpu_decode_scratch_free(&decode_scratch);
        kv_cache_free(&cache);
        return 1;
    }

    prefill_layer_major_cpu(logits, model, weights, &cache, prompt,
                            directional_steering_dirs,
                            directional_steering_attn,
                            directional_steering_ffn);

    const double t_prefill1 = now_sec();
    fprintf(stderr, "ds4: prefill %d/%d done\n", prompt->len, prompt->len);
    const char *dump_prefill_logits = getenv("DS4_CPU_DUMP_PREFILL_LOGITS");
    if (dump_prefill_logits && dump_prefill_logits[0]) {
        if (!write_f32_binary_file(dump_prefill_logits, logits, DS4_N_VOCAB)) {
            free(logits);
            cpu_decode_scratch_free(&decode_scratch);
            kv_cache_free(&cache);
            return 1;
        }
        fprintf(stderr, "ds4: wrote CPU prefill logits to %s\n", dump_prefill_logits);
    }

    int n_generated = 0;
    int n_decode_eval = 0;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < ctx_size; i++) {
        if (trace_top) {
            char label[64];
            snprintf(label, sizeof(label), "step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }

        int token = sample_argmax(logits, DS4_N_VOCAB);
        if (token == vocab->eos_id) break;

        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1 >= ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        /* The CPU decode step is expected to reuse buffers from
         * cpu_decode_scratch.  Keep the allocation guard tightly scoped to the
         * decode math itself; sampling, token emission, tracing, and callbacks
         * may allocate small temporary strings without invalidating that
         * guarantee. */
        ds4_alloc_guard_begin("CPU token decode");
        forward_token_raw_swa_cpu_decode_scratch(logits, model, weights, &cache, token, (uint32_t)pos,
                                                 directional_steering_dirs,
                                                 directional_steering_attn,
                                                 directional_steering_ffn,
                                                 &decode_scratch);
        ds4_alloc_guard_end();
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr, "ds4: decode eval %d took %.3f ms\n", n_decode_eval + 1, (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
    }
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    free(logits);
    cpu_decode_scratch_free(&decode_scratch);
    kv_cache_free(&cache);
    return 0;
}

#ifndef DS4_NO_GPU
/* Metal generation entry point.  The model runs as one local whole-graph
 * pipeline: chunked/layer-major prefill followed by graph decode steps. */
static int generate_metal_graph_raw_swa(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        bool                quality,
        const char        * directional_steering_file,
        float               directional_steering_attn,
        float               directional_steering_ffn,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    fprintf(stderr, "ds4: using GPU graph generation with layer-major graph prefill\n");

    if (prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }

    const uint32_t prefill_cap = metal_graph_prefill_cap_for_prompt(prompt->len);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, prefill_cap);
    if (prefill_cap < (uint32_t)prompt->len) {
        fprintf(stderr,
                "ds4: using chunked GPU prefill (%u-token chunks for %d prompt tokens)\n",
                prefill_cap,
                prompt->len);
    }
    ds4_gpu_graph g;
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap, false);
    if (!ok) {
        fprintf(stderr, "ds4: failed to allocate GPU graph runtime\n");
        return 1;
    }
    g.quality = quality;
    if (!metal_graph_load_directional_steering(&g,
                                               directional_steering_file,
                                               directional_steering_attn,
                                               directional_steering_ffn)) {
        metal_graph_free(&g);
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after graph alloc");

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    const bool trace_top = getenv("DS4_TRACE_TOP") != NULL;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;

    const double t_prefill0 = now_sec();
    if (prefill_cap < (uint32_t)prompt->len) {
        ok = metal_graph_prefill_chunked(&g, model, weights, prompt, prompt->len, logits, false, progress, progress_ud);
    } else {
        ok = metal_graph_prefill_raw_swa(&g, model, weights, prompt, prompt->len, logits, true);
    }
    const double t_prefill1 = now_sec();
    if (memory_report) ds4_gpu_print_memory_report("after prefill");

    if (!ok) {
        free(logits);
        metal_graph_free(&g);
        return 1;
    }
    const char *dump_prefill_logits = getenv("DS4_METAL_DUMP_PREFILL_LOGITS");
    if (dump_prefill_logits && dump_prefill_logits[0]) {
        if (!write_f32_binary_file(dump_prefill_logits, logits, DS4_N_VOCAB)) {
            free(logits);
            metal_graph_free(&g);
            return 1;
        }
        fprintf(stderr, "ds4: wrote GPU prefill logits to %s\n", dump_prefill_logits);
    }

    int pos = prompt->len;
    int n_generated = 0;
    int n_decode_eval = 0;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < ctx_size; i++) {
        if (trace_top) {
            char label[64];
            snprintf(label, sizeof(label), "step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }

        int token = sample_argmax(logits, DS4_N_VOCAB);
        if (token == vocab->eos_id) break;

        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1 >= ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        ok = metal_graph_eval_token_raw_swa(&g,
                                            model,
                                            weights,
                                            (uint32_t)token,
                                            (uint32_t)pos,
                                            logits);
        if (!ok) break;
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr, "ds4: gpu decode eval %d took %.3f ms\n", n_decode_eval + 1, (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
    }
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    if (memory_report) ds4_gpu_print_memory_report("before graph free");
    free(logits);
    metal_graph_free(&g);
    return ok ? 0 : 1;
}
#endif

#ifdef DS4_NO_GPU
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size) {
    (void)backend;
    ds4_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    m.raw_cap = ds4_default_raw_cap(ctx);
    m.raw_bytes = (uint64_t)DS4_N_LAYER *
                  m.raw_cap *
                  DS4_N_HEAD_DIM *
                  sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t comp_cap = ctx / ratio + 2u;
        if (ratio == 4) m.comp_cap = comp_cap;
        m.compressed_bytes += (uint64_t)comp_cap *
                              DS4_N_HEAD_DIM *
                              sizeof(float);
        if (ratio == 4) {
            m.compressed_bytes += (uint64_t)comp_cap *
                                  DS4_N_INDEXER_HEAD_DIM *
                                  sizeof(float);
        }
    }
    if (m.comp_cap == 0) m.comp_cap = ctx / 4u + 2u;
    m.scratch_bytes = ((uint64_t)(m.raw_cap + m.comp_cap) * sizeof(float)) +
                      ((uint64_t)m.comp_cap * sizeof(float)) +
                      ((uint64_t)m.comp_cap * sizeof(bool));
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}
#endif

/* =========================================================================
 * Engine API and Process Lock.
 * =========================================================================
 *
 * The public entry points acquire the single instance lock, open the GGUF with
 * the backend-appropriate mmap policy, and expose tokenized prompt operations
 * to the CLI and server.
 */

const char *ds4_backend_name(ds4_backend backend) {
    switch (backend) {
    case DS4_BACKEND_METAL: return "metal";
    case DS4_BACKEND_CUDA:  return "cuda";
    case DS4_BACKEND_CPU:   return "cpu";
    }
    return "unknown";
}

bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return mode == DS4_THINK_HIGH || mode == DS4_THINK_MAX;
}

const char *ds4_think_mode_name(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_NONE: return "none";
    case DS4_THINK_HIGH: return "high";
    case DS4_THINK_MAX:  return "max";
    }
    return "unknown";
}

const char *ds4_think_max_prefix(void) {
    return DS4_REASONING_EFFORT_MAX_PREFIX;
}

uint32_t ds4_think_max_min_context(void) {
    return DS4_THINK_MAX_MIN_CONTEXT;
}

ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size) {
    if (mode == DS4_THINK_MAX && (uint32_t)(ctx_size > 0 ? ctx_size : 0) < DS4_THINK_MAX_MIN_CONTEXT) {
        return DS4_THINK_HIGH;
    }
    return mode;
}

static void ds4_release_instance_lock(void) {
    if (g_ds4_lock_fd >= 0) {
        close(g_ds4_lock_fd);
        g_ds4_lock_fd = -1;
    }
}

/* Refuse to start a second ds4 process.  The model can map tens of GiB, so a
 * stale accidental second run is more dangerous than a normal CLI error. */
static void ds4_acquire_instance_lock(void) {
    const char *path = getenv("DS4_LOCK_FILE");
    if (!path || !path[0]) path = "/tmp/ds4.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "ds4: failed to open lock file %s: %s\n", path, strerror(errno));
        exit(2);
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                char *end = NULL;
                owner = strtol(buf, &end, 10);
            }
            if (owner > 0) {
                fprintf(stderr, "ds4: another ds4 process is already running (pid %ld); refusing to start\n", owner);
            } else {
                fprintf(stderr, "ds4: another ds4 process is already running; refusing to start\n");
            }
            close(fd);
            exit(2);
        }
        fprintf(stderr, "ds4: failed to lock %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }

    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "ds4: failed to truncate lock file %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }
    dprintf(fd, "%ld\n", (long)getpid());
    g_ds4_lock_fd = fd;
    atexit(ds4_release_instance_lock);
}

struct ds4_session {
    ds4_engine *engine;
#ifndef DS4_NO_GPU
    ds4_gpu_graph graph;
#endif
    ds4_kv_cache cpu_cache;
    ds4_cpu_decode_scratch cpu_scratch;
    token_vec checkpoint;
    float *logits;
    float *mtp_logits;
    int mtp_draft_token;
    uint64_t mtp_probe_total;
    uint64_t mtp_probe_hit;
    ds4_session_progress_fn progress;
    void *progress_ud;
    uint32_t prefill_cap;
    int ctx_size;
    bool checkpoint_valid;
    bool mtp_draft_valid;
};

/* =========================================================================
 * Session Snapshot Payloads.
 * =========================================================================
 *
 * The server disk cache stores a high-level file header, then delegates the
 * graph-specific payload below to the engine.  This payload is intentionally
 * not mmaped: restoring a checkpoint copies bytes back into the already
 * allocated Metal tensors, preserving the same live graph buffers used by
 * normal prefill/decode.  The raw SWA cache is serialized as the last logical
 * window only; suffix prefill writes its own raw rows before attention.  The
 * compressed caches are serialized up to their live row counts because sparse
 * attention may select rows from the whole prefix.
 *
 * The payload is model-specific rather than self-describing.  The fixed header
 * records enough shape information to reject a file written for a different
 * DS4 runtime, then the body writes: checkpoint tokens, last logits, per-layer
 * compressed row counts, raw SWA rows in logical order, compressed attention
 * rows, and the compressor/indexer frontiers.  That is the minimum state needed
 * for the next token to match a session that had just prefetched the prefix.
 */

#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(1)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_IO_CHUNK (8u * 1024u * 1024u)

static void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

static DS4_MAYBE_UNUSED int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static DS4_MAYBE_UNUSED int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static DS4_MAYBE_UNUSED int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}

static DS4_MAYBE_UNUSED uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
}

static DS4_MAYBE_UNUSED uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}

#ifndef DS4_NO_GPU
/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical Metal tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
static uint32_t session_raw_live_rows(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}

/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
static uint64_t session_payload_live_tensor_bytes(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

/* Metal tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (ds4_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read Metal session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (ds4_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore Metal session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}
#endif

static bool ds4_session_is_cpu(const ds4_session *s) {
    return s && s->engine && s->engine->backend == DS4_BACKEND_CPU;
}

static uint32_t session_cpu_raw_live_rows(const ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    uint32_t rows = ds4_default_raw_cap((uint32_t)s->ctx_size);
    if (rows > (uint32_t)s->checkpoint.len) rows = (uint32_t)s->checkpoint.len;
    return rows;
}

static uint32_t session_cpu_comp_cap(const ds4_session *s) {
    if (!s) return 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        if (layer->compress_ratio == 4) return layer->comp_cap;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        if (layer->compress_ratio != 0) return layer->comp_cap;
    }
    return (uint32_t)s->ctx_size;
}

static uint64_t session_cpu_payload_live_tensor_bytes(const ds4_session *s) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_cpu_raw_live_rows(s);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = layer->compress_ratio;
        if (ratio == 0) continue;
        bytes += (uint64_t)layer->n_comp * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)layer->n_index_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

static void session_cpu_reset_cache(ds4_session *s) {
    kv_cache_free(&s->cpu_cache);
    kv_cache_init(&s->cpu_cache, (uint32_t)s->ctx_size, 0);
}

int ds4_engine_routed_quant_bits(ds4_engine *e) {
    if (!e) return 0;
    const ds4_tensor *gate = e->weights.layer[0].ffn_gate_exps;
    if (!gate) return 0;
    return gate->type == DS4_TENSOR_Q4_K ? 4 : 2;
}

bool ds4_engine_has_mtp(ds4_engine *e) {
    return e && e->backend != DS4_BACKEND_CPU && e->mtp_ready;
}

int ds4_engine_mtp_draft_tokens(ds4_engine *e) {
    return e && e->backend != DS4_BACKEND_CPU && e->mtp_ready ? e->mtp_draft_tokens : 0;
}

const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    return s ? &s->checkpoint : NULL;
}

#ifndef DS4_NO_GPU
typedef struct {
    uint32_t n_comp[DS4_N_LAYER];
    uint32_t n_index_comp[DS4_N_LAYER];
    uint32_t mtp_n_raw;
} ds4_spec_frontier;

static void spec_frontier_free(ds4_spec_frontier *f) {
    if (!f) return;
    memset(f, 0, sizeof(*f));
}

static bool spec_frontier_snapshot(ds4_spec_frontier *f, ds4_session *s) {
    memset(f, 0, sizeof(*f));
    ds4_gpu_graph *g = &s->graph;
    f->mtp_n_raw = g->mtp_n_raw;

    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        f->n_comp[il] = g->layer_n_comp[il];
        f->n_index_comp[il] = g->layer_n_index_comp[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t ab = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_gpu_tensor_copy(g->spec_attn_state_kv[il], 0,
                                   g->layer_attn_state_kv[il], 0, ab) != 0 &&
             ds4_gpu_tensor_copy(g->spec_attn_state_score[il], 0,
                                   g->layer_attn_state_score[il], 0, ab) != 0;
        if (ratio == 4) {
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ok &&
                 ds4_gpu_tensor_copy(g->spec_index_state_kv[il], 0,
                                       g->layer_index_state_kv[il], 0, ib) != 0 &&
                 ds4_gpu_tensor_copy(g->spec_index_state_score[il], 0,
                                       g->layer_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok) return true;

    spec_frontier_free(f);
    return false;
}

static bool spec_frontier_restore(ds4_spec_frontier *f, ds4_session *s) {
    ds4_gpu_graph *g = &s->graph;
    bool ok = ds4_gpu_begin_commands() != 0;
    g->mtp_n_raw = f->mtp_n_raw;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = f->n_comp[il];
        g->layer_n_index_comp[il] = f->n_index_comp[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t ab = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,
                                   g->spec_attn_state_kv[il], 0, ab) != 0 &&
             ds4_gpu_tensor_copy(g->layer_attn_state_score[il], 0,
                                   g->spec_attn_state_score[il], 0, ab) != 0;
        if (ok && ratio == 4) {
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,
                                       g->spec_index_state_kv[il], 0, ib) != 0 &&
                 ds4_gpu_tensor_copy(g->layer_index_state_score[il], 0,
                                       g->spec_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    return ok;
}

/* Commit the prefix-1 state captured by the N=2 speculative verifier.
 *
 * The verifier has already advanced every layer through both draft tokens.  On
 * a one-token accept the append-only compressed caches can keep the second
 * speculative row as invisible garbage, but the compressor frontiers and row
 * counters must be rewound to the exact state after draft[0].  This is the
 * cheap partial-accept path: copy a few small per-layer frontiers instead of
 * restoring the whole prefix and replaying a one-token target decode. */
static bool spec_frontier_commit_prefix1(ds4_session *s) {
    ds4_gpu_graph *g = &s->graph;
    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;

        g->layer_n_comp[il] = g->spec_prefix1_n_comp[il];
        const uint64_t ab = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,
                                   g->spec_prefix1_attn_state_kv[il], 0, ab) != 0 &&
             ds4_gpu_tensor_copy(g->layer_attn_state_score[il], 0,
                                   g->spec_prefix1_attn_state_score[il], 0, ab) != 0;
        if (ok && ratio == 4) {
            g->layer_n_index_comp[il] = g->spec_prefix1_n_index_comp[il];
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,
                                       g->spec_prefix1_index_state_kv[il], 0, ib) != 0 &&
                 ds4_gpu_tensor_copy(g->layer_index_state_score[il], 0,
                                       g->spec_prefix1_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    return ok;
}
#endif

uint64_t ds4_session_payload_bytes(ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    if (ds4_session_is_cpu(s)) {
        uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
        bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
        bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
        bytes += session_cpu_payload_live_tensor_bytes(s);
        return bytes;
    }
#ifdef DS4_NO_GPU
    return 0;
#else
    const ds4_gpu_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(g, (uint32_t)s->checkpoint.len);
    return bytes;
#endif
}

int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        const uint32_t raw_live = session_cpu_raw_live_rows(s);
        const uint32_t raw_cap = ds4_default_raw_cap((uint32_t)s->ctx_size);
        const uint32_t comp_cap = session_cpu_comp_cap(s);
        uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
            DS4_SESSION_PAYLOAD_MAGIC,
            DS4_SESSION_PAYLOAD_VERSION,
            (uint32_t)s->ctx_size,
            s->prefill_cap,
            raw_cap,
            raw_cap,
            comp_cap,
            (uint32_t)s->checkpoint.len,
            DS4_N_LAYER,
            DS4_N_HEAD_DIM,
            DS4_N_INDEXER_HEAD_DIM,
            DS4_N_VOCAB,
            raw_live,
        };
        for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
            if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
        }
        for (int i = 0; i < s->checkpoint.len; i++) {
            if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
        }
        if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_write_u32(fp, s->cpu_cache.layer[il].n_comp, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_write_u32(fp, s->cpu_cache.layer[il].n_index_comp, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
            if (raw_live > layer->n_raw) {
                payload_set_err(err, errlen, "CPU session raw cache has fewer live rows than checkpoint");
                return 1;
            }
            const uint32_t raw_start = layer->n_raw - raw_live;
            if (payload_write_bytes(fp,
                                    layer->raw_kv + (uint64_t)raw_start * DS4_N_HEAD_DIM,
                                    (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float),
                                    err,
                                    errlen) != 0) return 1;
            const uint32_t ratio = layer->compress_ratio;
            if (ratio == 0) continue;
            if (payload_write_bytes(fp,
                                    layer->attn_comp_kv,
                                    (uint64_t)layer->n_comp * DS4_N_HEAD_DIM * sizeof(float),
                                    err,
                                    errlen) != 0) return 1;
            if (payload_write_bytes(fp, layer->attn_state_kv, layer_attn_state_bytes(ratio), err, errlen) != 0) return 1;
            if (payload_write_bytes(fp, layer->attn_state_score, layer_attn_state_bytes(ratio), err, errlen) != 0) return 1;
            if (ratio == 4) {
                if (payload_write_bytes(fp,
                                        layer->index_comp_kv,
                                        (uint64_t)layer->n_index_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                        err,
                                        errlen) != 0) return 1;
                if (payload_write_bytes(fp, layer->index_state_kv, layer_index_state_bytes(ratio), err, errlen) != 0) return 1;
                if (payload_write_bytes(fp, layer->index_state_score, layer_index_state_bytes(ratio), err, errlen) != 0) return 1;
            }
        }
        return 0;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize Metal before snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 live raw rows serialized below.
     */
    uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        DS4_N_VOCAB,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Write the raw ring in logical position order.  The file does not care
         * where the rows happened to live physically in the source graph. */
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_tensor_span(fp,
                                           g->layer_raw_cache[il],
                                           (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                           (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous.  The two compressor state tensors hold the partial window
         * that will become the next compressed row. */
        rc = payload_write_tensor_span(fp,
                                       g->layer_attn_comp_cache[il],
                                       0,
                                       (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                       buf,
                                       DS4_SESSION_IO_CHUNK,
                                       err,
                                       errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(fp,
                                           g->layer_index_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
#endif
}

int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        const uint32_t saved_ctx = h[2];
        const uint32_t saved_prefill_cap = h[3];
        const uint32_t saved_raw_cap = h[4];
        const uint32_t saved_raw_window = h[5];
        const uint32_t saved_comp_cap = h[6];
        const uint32_t saved_tokens = h[7];
        const uint32_t saved_raw_live = h[12];
        const uint32_t cpu_raw_cap = ds4_default_raw_cap((uint32_t)s->ctx_size);
        const uint32_t cpu_comp_cap = session_cpu_comp_cap(s);
        if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
            payload_set_err(err, errlen, "KV checkpoint does not fit current context");
            return 1;
        }
        if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
            h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
        {
            payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
            return 1;
        }
        if (saved_prefill_cap != s->prefill_cap || saved_raw_window != cpu_raw_cap) {
            payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
            return 1;
        }
        const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
        if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
            saved_raw_live > saved_raw_cap || saved_raw_live > cpu_raw_cap)
        {
            payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
            return 1;
        }
        if (saved_comp_cap > cpu_comp_cap) {
            payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
            return 1;
        }

        token_vec new_checkpoint = {0};
        for (uint32_t i = 0; i < saved_tokens; i++) {
            uint32_t tok = 0;
            if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            token_vec_push(&new_checkpoint, (int)tok);
        }
        if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                               &remaining, err, errlen) != 0)
        {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        uint32_t n_comp[DS4_N_LAYER];
        uint32_t n_index_comp[DS4_N_LAYER];
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_comp[il] > saved_comp_cap || n_comp[il] > cpu_comp_cap) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
                return 1;
            }
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > cpu_comp_cap) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
                return 1;
            }
        }

        s->checkpoint_valid = false;
        s->mtp_draft_valid = false;
        session_cpu_reset_cache(s);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            ds4_layer_cache *layer = &s->cpu_cache.layer[il];
            if (payload_read_bytes(fp,
                                   layer->raw_kv,
                                   (uint64_t)saved_raw_live * DS4_N_HEAD_DIM * sizeof(float),
                                   &remaining,
                                   err,
                                   errlen) != 0)
            {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            layer->n_raw = saved_raw_live;
            const uint32_t ratio = layer->compress_ratio;
            if (ratio == 0) continue;
            layer->n_comp = n_comp[il];
            layer->n_index_comp = n_index_comp[il];
            if (payload_read_bytes(fp,
                                   layer->attn_comp_kv,
                                   (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                   &remaining,
                                   err,
                                   errlen) != 0 ||
                payload_read_bytes(fp, layer->attn_state_kv, layer_attn_state_bytes(ratio), &remaining, err, errlen) != 0 ||
                payload_read_bytes(fp, layer->attn_state_score, layer_attn_state_bytes(ratio), &remaining, err, errlen) != 0)
            {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (ratio == 4) {
                if (payload_read_bytes(fp,
                                       layer->index_comp_kv,
                                       (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                       &remaining,
                                       err,
                                       errlen) != 0 ||
                    payload_read_bytes(fp, layer->index_state_kv, layer_index_state_bytes(ratio), &remaining, err, errlen) != 0 ||
                    payload_read_bytes(fp, layer->index_state_score, layer_index_state_bytes(ratio), &remaining, err, errlen) != 0)
                {
                    token_vec_free(&new_checkpoint);
                    return 1;
                }
            }
        }
        if (remaining != 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
            return 1;
        }
        token_vec_free(&s->checkpoint);
        s->checkpoint = new_checkpoint;
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        return 0;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
    {
        payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
        return 1;
    }
    if (saved_prefill_cap != s->prefill_cap || saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
        return 1;
    }
    /* The raw rows in the file are logical rows.  We can restore them into any
     * current ring with enough capacity, but the saved live count must be exactly
     * the last window implied by the saved token count. */
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap)
    {
        payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0)
    {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    uint32_t n_comp[DS4_N_LAYER];
    uint32_t n_index_comp[DS4_N_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_comp[il] > saved_comp_cap || n_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
            return 1;
        }
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Rebuild the physical raw ring expected by the current graph.  This is
         * why the file stores rows in logical order instead of dumping bytes from
         * the old ring layout. */
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          g->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                          (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        rc = payload_read_tensor_span(fp,
                                      g->layer_attn_comp_cache[il],
                                      0,
                                      (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                      buf,
                                      DS4_SESSION_IO_CHUNK,
                                      &remaining,
                                      err,
                                      errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = n_comp[il];
        g->layer_n_index_comp[il] = n_index_comp[il];
    }
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;
    return 0;
#endif
}

int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap) {
        payload_set_err(err, errlen, "invalid session snapshot save");
        return 1;
    }
    const uint64_t bytes = ds4_session_payload_bytes(s);
    if (bytes == 0) {
        payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (bytes > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t *p = realloc(snap->ptr, (size_t)bytes);
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot");
        return 1;
    }
    const int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to finalize memory session snapshot");
        return 1;
    }
    if (rc != 0) return 1;
    snap->len = bytes;
    return 0;
}

int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        payload_set_err(err, errlen, "invalid session snapshot load");
        return 1;
    }
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = ds4_session_load_payload(s, fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens) {
    dump_tokens(&e->vocab, tokens);
}

int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    ds4_model model;
    ds4_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    model_open(&model, model_path, false, false);
    vocab_load(&vocab, &model);
    tokenize_rendered_chat_vocab(&vocab, text ? text : "", &tokens);

    dump_tokens_fp(fp, &vocab, &tokens);
    token_vec_free(&tokens);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}

#ifndef DS4_NO_GPU
static bool imatrix_read_text_file(const char *path, char **out, size_t *len_out) {
    *out = NULL;
    *len_out = 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ds4: failed to stat imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX - 1) {
        fprintf(stderr, "ds4: imatrix dataset is too large: %s\n", path);
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    size_t n = (size_t)st.st_size;
    char *buf = xmalloc(n + 1);
    if (n != 0 && fread(buf, 1, n, fp) != n) {
        fprintf(stderr, "ds4: failed to read imatrix dataset %s\n", path);
        fclose(fp);
        free(buf);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close imatrix dataset %s: %s\n", path, strerror(errno));
        free(buf);
        return false;
    }
    buf[n] = '\0';
    *out = buf;
    *len_out = n;
    return true;
}

static char *imatrix_trim_block(char *p, char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return p;
}
#endif

int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)dataset_path;
    (void)output_path;
    (void)ctx_size;
    (void)max_prompts;
    (void)max_tokens;
    fprintf(stderr, "ds4: imatrix collection requires a graph backend build\n");
    return 1;
#else
    if (!e || !dataset_path || !output_path) return 1;
    if (e->backend != DS4_BACKEND_METAL || !e->metal_ready) {
        fprintf(stderr, "ds4: imatrix collection currently requires --metal\n");
        return 1;
    }
    if (ctx_size <= 0) ctx_size = 32768;

    char *dataset = NULL;
    size_t dataset_len = 0;
    if (!imatrix_read_text_file(dataset_path, &dataset, &dataset_len)) return 1;

    const ds4_model *model = &e->model;
    const ds4_weights *weights = &e->weights;
    const uint32_t prefill_cap = metal_graph_prefill_cap_for_prompt(ctx_size);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, prefill_cap);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap, false);
    if (!ok) {
        fprintf(stderr, "ds4: failed to allocate imatrix Metal graph runtime\n");
        free(dataset);
        return 1;
    }
    g.quality = e->quality;

    ds4_imatrix_collector collector;
    if (!imatrix_collector_init(&collector, prefill_cap, dataset_path)) {
        fprintf(stderr, "ds4: failed to allocate imatrix collector\n");
        metal_graph_free(&g);
        free(dataset);
        return 1;
    }

    fprintf(stderr,
            "ds4: collecting routed-MoE imatrix from %s (ctx=%d, chunk=%u)\n",
            dataset_path, ctx_size, prefill_cap);

    int prompts_done = 0;
    int tokens_done = 0;
    char *cursor = dataset;
    const char *marker_lit = "===== DS4_IMATRIX_PROMPT";
    while (*cursor) {
        char *start = cursor;
        char *marker = strstr(cursor, marker_lit);
        if (marker) {
            char *nl = strchr(marker, '\n');
            if (!nl) break;
            start = nl + 1;
        } else if (prompts_done != 0) {
            break;
        }

        char *next = strstr(start, marker_lit);
        char *end = next ? next : dataset + dataset_len;
        char saved = *end;
        char *prompt_text = imatrix_trim_block(start, end);
        if (prompt_text[0] != '\0') {
            token_vec prompt = {0};
            ds4_tokenize_rendered_chat(e, prompt_text, &prompt);
            if (prompt.len > ctx_size) prompt.len = ctx_size;
            if (max_tokens > 0 && prompt.len > max_tokens - tokens_done) {
                prompt.len = max_tokens - tokens_done;
            }
            if (prompt.len > 0) {
                if (!metal_graph_reset_prefill_state(&g)) {
                    fprintf(stderr, "ds4: failed to reset imatrix graph state\n");
                    ok = false;
                } else if ((uint32_t)prompt.len > prefill_cap) {
                    ok = metal_graph_prefill_chunked_range(&g, model, weights,
                                                           &prompt, 0,
                                                           (uint32_t)prompt.len,
                                                           NULL, false,
                                                           NULL, NULL,
                                                           &collector);
                } else {
                    ok = metal_graph_prefill_layer_major(&g, model, weights,
                                                         &prompt, prompt.len,
                                                         NULL, false,
                                                         &collector);
                }
                if (!ok) {
                    fprintf(stderr, "ds4: imatrix prefill failed at prompt %d\n", prompts_done + 1);
                    token_vec_free(&prompt);
                    *end = saved;
                    break;
                }
                prompts_done++;
                tokens_done += prompt.len;
                if (prompts_done % 10 == 0) {
                    fprintf(stderr,
                            "ds4: imatrix prompts=%d tokens=%d routes=%llu\r",
                            prompts_done,
                            tokens_done,
                            (unsigned long long)collector.observed_routes);
                    fflush(stderr);
                }
            }
            token_vec_free(&prompt);
        }
        *end = saved;
        if (!next) break;
        cursor = next;
        if (max_prompts > 0 && prompts_done >= max_prompts) break;
        if (max_tokens > 0 && tokens_done >= max_tokens) break;
    }
    fputc('\n', stderr);

    if (ok) {
        ok = imatrix_collector_save(&collector, weights, output_path);
        if (ok) {
            fprintf(stderr,
                    "ds4: wrote imatrix %s from %d prompts, %d tokens, %llu routed expert observations\n",
                    output_path,
                    prompts_done,
                    tokens_done,
                    (unsigned long long)collector.observed_routes);
        }
    }

    imatrix_collector_free(&collector);
    metal_graph_free(&g);
    free(dataset);
    return ok ? 0 : 1;
#endif
}

int ds4_engine_generate_argmax(
        ds4_engine        *e,
        const ds4_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    if (ds4_backend_uses_graph(e->backend)) {
#ifndef DS4_NO_GPU
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: %s generation requested but the graph backend is unavailable\n",
                    ds4_backend_name(e->backend));
            return 1;
        }
        return generate_metal_graph_raw_swa(model, vocab, weights, prompt,
                                            n_predict, ctx_size, e->quality,
                                            e->directional_steering_file,
                                            e->directional_steering_attn_scale,
                                            e->directional_steering_ffn_scale,
                                            emit, done, emit_ud,
                                            progress, progress_ud);
#else
        fprintf(stderr, "ds4: %s generation requested but this build has no graph backend support\n",
                ds4_backend_name(e->backend));
        return 1;
#endif
    }

    return generate_raw_swa_cpu(model, vocab, weights, prompt, n_predict,
                                ctx_size,
                                e->directional_steering_dirs,
                                e->directional_steering_attn_scale,
                                e->directional_steering_ffn_scale,
                                emit, done, emit_ud, progress, progress_ud);
}

int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_decode_test(&e->model, &e->weights, prompt);
#else
    (void)e;
    (void)prompt;
    fprintf(stderr, "ds4: Metal graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal full graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_first_token_full_test(&e->model, &e->weights, prompt);
#else
    (void)e;
    (void)prompt;
    fprintf(stderr, "ds4: Metal full graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: Metal prompt graph test requested but Metal is unavailable\n");
        return 1;
    }
    return metal_graph_prompt_logits_test(&e->model, &e->weights, prompt, ctx_size);
#else
    (void)e;
    (void)prompt;
    (void)ctx_size;
    fprintf(stderr, "ds4: Metal prompt graph test requested but this build has no Metal support\n");
    return 1;
#endif
}

int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: head test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;
    const ds4_layer_weights *layer0 = &weights->layer[0];

    float *prompt_embd = xmalloc((size_t)prompt->len * DS4_N_EMBD * sizeof(prompt_embd[0]));
    embed_prompt(model, weights, prompt, DS4_N_EMBD, prompt_embd);

    const uint32_t n_hc = DS4_N_HC;
    float *hc0 = xmalloc((size_t)DS4_N_EMBD * sizeof(hc0[0]));
    float *residual_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(residual_hc[0]));
    float hc_post[4];
    float hc_comb[16];
    layer_attn_pre_one(model, layer0,
        prompt_embd + (uint64_t)(prompt->len - 1) * DS4_N_EMBD,
        hc0, residual_hc, hc_post, hc_comb);
    print_vec_stats("blk.0 attn_pre", hc0, DS4_N_EMBD);

    float *attn_norm0 = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm0[0]));
    layer_attn_norm_one(attn_norm0, model, layer0, hc0);

    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    float *q0 = xmalloc((size_t)q_dim * sizeof(q0[0]));
    layer_q_projection_normed_one(model, layer0, attn_norm0, q0);
    print_vec_stats("blk.0 q", q0, q_dim);

    float *kv0 = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv0[0]));
    layer_kv_projection_normed_one(model, layer0, attn_norm0, kv0);
    print_vec_stats("blk.0 kv", kv0, DS4_N_HEAD_DIM);
    rope_tail_layer_inplace(q0, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    rope_tail_layer_inplace(kv0, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv0, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(kv0, DS4_N_HEAD_DIM);

    float *attn_heads = xmalloc((size_t)q_dim * sizeof(attn_heads[0]));
    layer_attention_one(attn_heads, model, layer0, q0, kv0);
    print_vec_stats("blk.0 attn_heads", attn_heads, q_dim);
    rope_tail_layer_inplace(attn_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, true);

    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    layer_grouped_out_one(attn_out, model, layer0, attn_heads);
    print_vec_stats("blk.0 attn_out", attn_out, DS4_N_EMBD);

    float *after_attn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_attn_hc[0]));
    hc_post_one(after_attn_hc, attn_out, residual_hc, hc_post, hc_comb, DS4_N_EMBD, n_hc);
    print_vec_stats("blk.0 after_attn_hc", after_attn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *after_ffn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_ffn_hc[0]));
    layer_ffn_one(after_ffn_hc, model, layer0, after_attn_hc, 0, prompt->v[prompt->len - 1],
                  NULL, 0.0f, true);
    print_vec_stats("blk.0 after_ffn_hc", after_ffn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    output_logits_one(logits, model, weights, after_ffn_hc);
    print_vec_stats("logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after native blk.0 slice:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(after_ffn_hc);
    free(after_attn_hc);
    free(attn_out);
    free(attn_heads);
    free(kv0);
    free(q0);
    free(attn_norm0);
    free(residual_hc);
    free(hc0);
    free(prompt_embd);
    return 0;
}

int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: first-token test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    float *hc = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(hc[0]));
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    forward_first_token_cpu(hc, model, weights, prompt->v[0]);
    print_vec_stats("first-token final_hc", hc, (uint64_t)DS4_N_HC * DS4_N_EMBD);
    output_logits_one(logits, model, weights, hc);
    print_vec_stats("first-token logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after first-token whole-model CPU pass:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(hc);
    return 0;
}

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->mtp_model.fd = -1;
    e->backend = opt->backend;
    e->quality = opt->quality;
    e->mtp_draft_tokens = opt->mtp_draft_tokens > 0 ? opt->mtp_draft_tokens : 1;
    if (e->mtp_draft_tokens > 16) e->mtp_draft_tokens = 16;
    e->mtp_margin = opt->mtp_margin >= 0.0f ? opt->mtp_margin : 3.0f;
    if ((opt->directional_steering_attn != 0.0f || opt->directional_steering_ffn != 0.0f) &&
        (!opt->directional_steering_file || !opt->directional_steering_file[0]))
    {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = ds4_strdup(opt->directional_steering_file);
        e->directional_steering_attn_scale = opt->directional_steering_attn;
        e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    }
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;
    ds4_acquire_instance_lock();

    const bool graph_backend = ds4_backend_uses_graph(opt->backend);
    model_open(&e->model, opt->model_path, graph_backend, true);
    if (opt->warm_weights) model_warm_weights(&e->model);
    vocab_load(&e->vocab, &e->model);
    config_validate_model(&e->model);
    weights_bind(&e->weights, &e->model);
    if (e->backend == DS4_BACKEND_CPU && !cpu_load_directional_steering(e)) {
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    if (opt->mtp_path && opt->mtp_path[0]) {
        model_open(&e->mtp_model, opt->mtp_path, graph_backend, true);
        mtp_weights_bind(&e->mtp_weights, &e->mtp_model);
        e->mtp_ready = true;
        fprintf(stderr, "ds4: MTP support model loaded: %s (draft=%d)\n",
                opt->mtp_path,
                e->mtp_draft_tokens);
    }

#ifndef DS4_NO_GPU
    if (e->backend == DS4_BACKEND_CUDA) {
#ifdef __APPLE__
        fprintf(stderr, "ds4: CUDA backend requested but this build is linked with Metal, not CUDA\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
#endif
    }
    if (e->backend == DS4_BACKEND_METAL) {
#ifndef __APPLE__
        fprintf(stderr, "ds4: Metal backend requested but this build is linked with CUDA, not Metal\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
#endif
    }
    if (graph_backend) {
        e->metal_ready = ds4_gpu_init() != 0;
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: %s backend unavailable; aborting startup\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        ds4_gpu_set_quality(e->quality);
        (void)ds4_gpu_set_model_fd(e->model.fd);
        if (!ds4_gpu_set_model_map_range(e->model.map,
                                           e->model.size,
                                           e->model.tensor_data_pos,
                                           e->model.size - e->model.tensor_data_pos))
        {
            fprintf(stderr,
                    "ds4: %s failed to map model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (e->mtp_ready &&
            !ds4_gpu_set_model_map_range(e->mtp_model.map,
                                           e->mtp_model.size,
                                           e->mtp_model.tensor_data_pos,
                                           e->mtp_model.size - e->mtp_model.tensor_data_pos))
        {
            fprintf(stderr,
                    "ds4: %s failed to map MTP model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (!e->mtp_ready && !accelerator_cache_model_tensors(e->backend, &e->model)) {
            fprintf(stderr, "ds4: %s failed to prepare startup model cache\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        fprintf(stderr, "ds4: %s backend initialized for graph diagnostics\n",
                ds4_backend_name(e->backend));
    }
#else
    if (graph_backend) {
        fprintf(stderr, "ds4: %s backend requested but this build has no graph backend support; aborting startup\n",
                ds4_backend_name(e->backend));
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
#endif

    *out = e;
    return 0;
}

void ds4_engine_summary(ds4_engine *e) {
    model_summary(&e->model);
}

void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
    weights_free(&e->weights);
    vocab_free(&e->vocab);
    ds4_threads_shutdown();
    if (e->mtp_ready) model_close(&e->mtp_model);
    model_close(&e->model);
#ifndef DS4_NO_GPU
    ds4_gpu_cleanup();
#endif
    ds4_release_instance_lock();
    free(e->directional_steering_dirs);
    free(e->directional_steering_file);
    free(e);
}

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
    if (e->backend == DS4_BACKEND_CPU) {
        ds4_session *s = xcalloc(1, sizeof(*s));
        s->engine = e;
        s->ctx_size = ctx_size;
        s->prefill_cap = ds4_default_prefill_cap_for_prompt(ctx_size);
        kv_cache_init(&s->cpu_cache, (uint32_t)ctx_size, 0);
        cpu_decode_scratch_init(&s->cpu_scratch, (uint32_t)ctx_size);
        s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        *out = s;
        return 0;
    }
#ifdef DS4_NO_GPU
    return 1;
#else
    if (!ds4_backend_uses_graph(e->backend) || !e->metal_ready) return 1;

    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_cap = metal_graph_prefill_cap_for_prompt(ctx_size);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, s->prefill_cap);
    if (!metal_graph_alloc_raw_cap(&s->graph, &e->weights, &e->weights.layer[0],
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap, e->mtp_ready))
    {
        free(s);
        return 1;
    }
    s->graph.quality = e->quality;
    if (!metal_graph_load_directional_steering(&s->graph,
                                               e->directional_steering_file,
                                               e->directional_steering_attn_scale,
                                               e->directional_steering_ffn_scale)) {
        metal_graph_free(&s->graph);
        free(s);
        return 1;
    }
    s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    if (e->mtp_ready) {
        s->mtp_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->mtp_logits[0]));
        s->mtp_draft_token = -1;
    }
    *out = s;
    return 0;
#endif
}

void ds4_session_free(ds4_session *s) {
    if (!s) return;
    if (ds4_session_is_cpu(s)) {
        kv_cache_free(&s->cpu_cache);
        cpu_decode_scratch_free(&s->cpu_scratch);
    }
#ifndef DS4_NO_GPU
    else {
        metal_graph_free(&s->graph);
    }
#endif
    token_vec_free(&s->checkpoint);
    free(s->logits);
    free(s->mtp_logits);
    free(s);
}

void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

#ifndef DS4_NO_GPU
typedef struct {
    ds4_session *session;
    const ds4_tokens *prompt;
    ds4_session_progress_fn user;
    void *user_ud;
} ds4_sync_progress;

static void ds4_session_note_prefill_progress(void *ud, const char *event, int current, int total) {
    ds4_sync_progress *p = ud;
    if (!p || !p->session || !p->prompt) return;
    if (!strcmp(event, "prefill_chunk") && current > 0 && current <= p->prompt->len) {
        p->session->checkpoint.len = 0;
        for (int i = 0; i < current; i++) token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
        p->session->checkpoint_valid = true;
        p->session->mtp_draft_valid = false;
    }
    if (p->user) p->user(p->user_ud, event, current, total);
}
#endif

/* Bring the live backend state to exactly the supplied token prefix.
 *
 * ds4-server and the REPL are stateless at the text/API layer but stateful here:
 * they resend or rebuild the full transcript, and this function decides whether
 * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 * ways:
 *
 *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 *     boundaries so compressor/indexer rows finalize in the same order as a
 *     cold prompt;
 *   - short suffix: ordinary one-token decode, which is faster below the
 *     measured crossover and preserves exact autoregressive semantics.
 *
 * A non-matching prompt discards the checkpoint and prefills from token zero.
 */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        ds4_engine *e = s->engine;
        if (s->checkpoint_valid &&
            prompt->len >= s->checkpoint.len &&
            ds4_tokens_starts_with(prompt, &s->checkpoint))
        {
            s->mtp_draft_valid = false;
            for (int i = s->checkpoint.len; i < prompt->len; i++) {
                forward_token_raw_swa_cpu_decode_scratch(s->logits,
                                                         &e->model,
                                                         &e->weights,
                                                         &s->cpu_cache,
                                                         prompt->v[i],
                                                         (uint32_t)s->checkpoint.len,
                                                         e->directional_steering_dirs,
                                                         e->directional_steering_attn_scale,
                                                         e->directional_steering_ffn_scale,
                                                         &s->cpu_scratch);
                token_vec_push(&s->checkpoint, prompt->v[i]);
                if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i + 1, prompt->len);
            }
            s->checkpoint_valid = true;
            return 0;
        }

        session_cpu_reset_cache(s);
        prefill_layer_major_cpu(s->logits,
                                &e->model,
                                &e->weights,
                                &s->cpu_cache,
                                prompt,
                                e->directional_steering_dirs,
                                e->directional_steering_attn_scale,
                                e->directional_steering_ffn_scale);
        ds4_tokens_copy(&s->checkpoint, prompt);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        if (s->progress) s->progress(s->progress_ud, "prefill_chunk", prompt->len, prompt->len);
        return 0;
    }
#ifdef DS4_NO_GPU
    (void)s;
    (void)prompt;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    const char *backend_name = ds4_backend_name(e->backend);

    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &s->checkpoint))
    {
        s->mtp_draft_valid = false;
        const int suffix = prompt->len - s->checkpoint.len;
        const uint32_t resume_min = metal_graph_resume_prefill_min_tokens();
        if (suffix > 0 && (uint32_t)suffix >= resume_min) {
            ds4_sync_progress progress = {
                .session = s,
                .prompt = prompt,
                .user = s->progress,
                .user_ud = s->progress_ud,
            };
            ds4_session_progress_fn progress_fn =
                s->progress ? ds4_session_note_prefill_progress : NULL;
            bool ok = metal_graph_prefill_chunked_range(&s->graph,
                                                        &e->model,
                                                        &e->weights,
                                                        prompt,
                                                        (uint32_t)s->checkpoint.len,
                                                        (uint32_t)suffix,
                                                        s->logits,
                                                        false,
                                                        progress_fn,
                                                        progress_fn ? &progress : NULL,
                                                        NULL);
            if (!ok) {
                snprintf(err, errlen, "%s resumed prefill failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            ds4_tokens_copy(&s->checkpoint, prompt);
            s->checkpoint_valid = true;
            return 0;
        }

        for (int i = s->checkpoint.len; i < prompt->len; i++) {
            if (!metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                                (uint32_t)prompt->v[i],
                                                (uint32_t)s->checkpoint.len,
                                                s->logits))
            {
                snprintf(err, errlen, "%s decode failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            token_vec_push(&s->checkpoint, prompt->v[i]);
        }
        return 0;
    }

    bool ok;
    if (s->prefill_cap < (uint32_t)prompt->len) {
        ds4_sync_progress progress = {
            .session = s,
            .prompt = prompt,
            .user = s->progress,
            .user_ud = s->progress_ud,
        };
        ds4_session_progress_fn progress_fn =
            s->progress ? ds4_session_note_prefill_progress : NULL;
        ok = metal_graph_prefill_chunked(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         progress_fn, progress_fn ? &progress : NULL);
    } else {
        ok = metal_graph_prefill_raw_swa(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false);
    }
    if (!ok) {
        snprintf(err, errlen, "%s prefill failed", backend_name);
        s->checkpoint_valid = false;
        return 1;
    }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    s->graph.mtp_n_raw = 0;
    return 0;
#endif
}

/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the backend state also
 * contains raw SWA rows, compressed KV rows, indexer rows, and compressor
 * frontiers.  Replacing any part of the live tail requires restoring that whole
 * frontier first.  Extending exactly at the live end is safe; rewriting behind
 * it is not an in-place operation. */
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}

/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the backend still contains raw and compressed rows
 * for the old suffix.  Until we have a real frontier snapshot at the
 * rewrite point, any replacement behind the live end reports that a rebuild is
 * needed without mutating the session.  The server may still find an older disk KV
 * checkpoint before falling back to a full replay. */
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (!s->checkpoint_valid) {
        snprintf(err, errlen, "session has no valid checkpoint");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (common < 0 || common > s->checkpoint.len || common > prompt->len) {
        snprintf(err, errlen, "invalid rewrite prefix");
        return DS4_SESSION_REWRITE_ERROR;
    }
    for (int i = 0; i < common; i++) {
        if (s->checkpoint.v[i] != prompt->v[i]) {
            snprintf(err, errlen, "rewrite prefix does not match live checkpoint");
            return DS4_SESSION_REWRITE_ERROR;
        }
    }

    if (common == s->checkpoint.len) {
        return ds4_session_sync(s, prompt, err, errlen) == 0 ?
            DS4_SESSION_REWRITE_OK : DS4_SESSION_REWRITE_ERROR;
    }

    if (ds4_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
                 common, s->checkpoint.len, prompt->len);
        return DS4_SESSION_REWRITE_REBUILD_NEEDED;
    }

    snprintf(err, errlen, "unexpected canonical rewrite state");
    return DS4_SESSION_REWRITE_ERROR;
}

int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt) {
    if (!s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}

int ds4_session_argmax(ds4_session *s) {
    return sample_argmax(s->logits, DS4_N_VOCAB);
}

int ds4_session_argmax_excluding(ds4_session *s, int excluded_id) {
    if (!s || !s->logits) return -1;
    int best = -1;
    float best_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}

int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    return sample_top_p_min_p(s->logits, DS4_N_VOCAB, temperature, top_k, top_p, min_p, rng);
}

int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    if (!s || !out || k <= 0) return 0;
    if (k > (int)DS4_N_VOCAB) k = (int)DS4_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = DS4_NEG_INF;
        out[i].logprob = DS4_NEG_INF;
    }

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : DS4_NEG_INF;
    }
    return k;
}

int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out) {
    if (!s || !out || token < 0 || token >= (int)DS4_N_VOCAB) return 0;

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v) && v > max_logit) max_logit = v;
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = isfinite(out->logit) ? (float)((double)out->logit - logsum) : DS4_NEG_INF;
    return 1;
}

static int ds4_session_eval_internal(ds4_session *s, int token, bool probe_mtp,
                                     char *err, size_t errlen) {
    if (!s) return 1;
    if (ds4_session_is_cpu(s)) {
        ds4_engine *e = s->engine;
        forward_token_raw_swa_cpu_decode_scratch(s->logits,
                                                 &e->model,
                                                 &e->weights,
                                                 &s->cpu_cache,
                                                 token,
                                                 (uint32_t)s->checkpoint.len,
                                                 e->directional_steering_dirs,
                                                 e->directional_steering_attn_scale,
                                                 e->directional_steering_ffn_scale,
                                                 &s->cpu_scratch);
        token_vec_push(&s->checkpoint, token);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        (void)probe_mtp;
        return 0;
    }
#ifdef DS4_NO_GPU
    (void)s;
    (void)token;
    (void)probe_mtp;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    const bool mtp_probe_log = getenv("DS4_MTP_PROBE") != NULL;
    const bool mtp_should_draft =
        probe_mtp && e->mtp_ready && s->mtp_logits &&
        (e->mtp_draft_tokens > 1 || mtp_probe_log);
    if (probe_mtp && s->mtp_draft_valid) {
        if (mtp_probe_log) {
            s->mtp_probe_total++;
            if (s->mtp_draft_token == token) s->mtp_probe_hit++;
            fprintf(stderr,
                    "ds4: mtp probe token=%d draft=%d hit=%llu/%llu\n",
                    token,
                    s->mtp_draft_token,
                    (unsigned long long)s->mtp_probe_hit,
                    (unsigned long long)s->mtp_probe_total);
        }
        s->mtp_draft_valid = false;
    }
    if (!metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                        (uint32_t)token,
                                        (uint32_t)s->checkpoint.len,
                                        s->logits))
    {
        snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    if (mtp_should_draft) {
        int mtp_top = -1;
        if (metal_graph_eval_mtp_draft(&s->graph,
                                       &e->model,
                                       &e->weights,
                                       &e->mtp_model,
                                       &e->mtp_weights,
                                       token,
                                       (uint32_t)(s->checkpoint.len - 1),
                                       getenv("DS4_MTP_FULL_LOGITS") ? s->mtp_logits : NULL,
                                       &mtp_top)) {
            s->mtp_draft_token = mtp_top >= 0 ? mtp_top : sample_argmax(s->mtp_logits, DS4_N_VOCAB);
            s->mtp_draft_valid = true;
        } else if (getenv("DS4_MTP_PROBE")) {
            fprintf(stderr, "ds4: mtp probe draft failed\n");
        }
    }
    return 0;
#endif
}

int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    return ds4_session_eval_internal(s, token, true, err, errlen);
}

/* Speculative decode state machine:
 * 1. commit the normal target token and use its logits to validate draft[0];
 * 2. let MTP recursively draft a tiny suffix from its own raw-cache frontier;
 * 3. verify the suffix with the target graph, committing only the accepted
 *    prefix and rolling back speculative Metal state on miss;
 * 4. fall back to ordinary one-token decode if the fast verifier cannot prove
 *    the target stream. */
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    if (ds4_session_is_cpu(s)) {
        (void)max_tokens;
        (void)eos_token;
        if (!accepted || accepted_cap <= 0) return 0;
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
#ifdef DS4_NO_GPU
    (void)s; (void)first_token; (void)max_tokens; (void)eos_token;
    (void)accepted; (void)accepted_cap;
    snprintf(err, errlen, "GPU support is not compiled in");
    return -1;
#else
    if (!s || max_tokens <= 0 || accepted_cap <= 0) return 0;
    ds4_engine *e = s->engine;

    /*
     * MTP in DeepSeek V4 is a speculative drafter, not a replacement sampler.
     * The target model still defines the exact output stream.  A cycle starts
     * by accepting one normal target token, then asks the MTP block to propose
     * a short suffix.  The suffix is useful only if the target model can verify
     * several proposed positions together; running ordinary decode once per
     * draft token is correctness-safe but cannot be faster than baseline.
     */
    if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
    int n_accept = 0;
    accepted[n_accept++] = first_token;
    if (first_token == eos_token || max_tokens == 1 || n_accept >= accepted_cap) return n_accept;

    if (!e->mtp_ready || !s->mtp_draft_valid || e->mtp_draft_tokens <= 1) return n_accept;

    int draft_cap = e->mtp_draft_tokens;
    if (draft_cap > max_tokens - n_accept) draft_cap = max_tokens - n_accept;
    if (draft_cap > accepted_cap - n_accept) draft_cap = accepted_cap - n_accept;
    int room = s->ctx_size - s->checkpoint.len;
    if (draft_cap > room - 1) draft_cap = room - 1;
    if (draft_cap <= 0) return n_accept;

    int drafts[16];
    int draft_n = 1;
    drafts[0] = s->mtp_draft_token;
    s->mtp_draft_valid = false;
    const bool strict_mtp = e->quality || getenv("DS4_MTP_STRICT") != NULL;
    float mtp_margin_threshold = e->mtp_margin;
    const char *mtp_margin_env = getenv("DS4_MTP_MIN_MARGIN");
    if (mtp_margin_env && mtp_margin_env[0]) {
        char *end = NULL;
        float v = strtof(mtp_margin_env, &end);
        if (end != mtp_margin_env && v >= 0.0f) mtp_margin_threshold = v;
    }
    const bool mtp_timing = getenv("DS4_MTP_TIMING") != NULL;
    const bool mtp_conf_log = getenv("DS4_MTP_CONF_LOG") != NULL;
    const bool mtp_need_logits = mtp_conf_log ||
        getenv("DS4_MTP_FULL_LOGITS") != NULL ||
        (!strict_mtp && mtp_margin_threshold > 0.0f);
    const double mtp_t0 = mtp_timing ? now_sec() : 0.0;
    double mtp_t_after_draft = mtp_t0;
    float mtp_last_margin = 0.0f;
    int mtp_last_top0 = -1, mtp_last_top1 = -1;

    /*
     * The first proposed token is verified for free: ds4_session_eval() just
     * produced the base logits for the committed prefix.  If MTP disagrees at
     * this point there is no suffix to verify, so the exact behavior is to emit
     * only first_token and skip all speculative work.
     */
    if (sample_argmax(s->logits, DS4_N_VOCAB) != drafts[0]) {
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp spec miss first draft=%d\n", drafts[0]);
        }
        return n_accept;
    }
    if (drafts[0] == eos_token) draft_cap = 1;
    const uint32_t mtp_base_raw = s->graph.mtp_n_raw;
    /*
     * MTP has its own raw SWA cache. Recursive drafting writes speculative
     * future rows into it; after verification, rows beyond the accepted prefix
     * must become invisible.  We do not copy/rollback the cache body because the
     * next draft attempt will overwrite future slots.  A counter is enough.
     */
#define DS4_MTP_KEEP_ACCEPTED(n_) do { \
        uint32_t keep_ = mtp_base_raw + (uint32_t)(n_); \
        if (keep_ > s->graph.raw_window) keep_ = s->graph.raw_window; \
        s->graph.mtp_n_raw = keep_; \
    } while (0)

    for (; draft_n < draft_cap; draft_n++) {
        ds4_gpu_tensor *prev_hc = (draft_n & 1) ? s->graph.mtp_state_hc : s->graph.mtp_next_hc;
        ds4_gpu_tensor *out_hc = (draft_n & 1) ? s->graph.mtp_next_hc : s->graph.mtp_state_hc;
        int mtp_top = -1;
        if (!metal_graph_eval_mtp_draft_from_hc(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                &e->mtp_model,
                                                &e->mtp_weights,
                                                prev_hc,
                                                out_hc,
                                                drafts[draft_n - 1],
                                                (uint32_t)(s->checkpoint.len + draft_n - 1),
                                                mtp_need_logits ? s->mtp_logits : NULL,
                                                &mtp_top))
        {
            return n_accept;
        }
        drafts[draft_n] = mtp_top >= 0 ? mtp_top : sample_argmax(s->mtp_logits, DS4_N_VOCAB);
        if (drafts[draft_n] == eos_token) {
            draft_n++;
            break;
        }
    }
    if (mtp_conf_log && draft_n > 1) {
        float v0 = 0.0f, v1 = 0.0f;
        logits_top2(s->mtp_logits, DS4_N_VOCAB, &mtp_last_top0, &v0, &mtp_last_top1, &v1);
        mtp_last_margin = v0 - v1;
    }
    if (mtp_timing) mtp_t_after_draft = now_sec();

    if (!strict_mtp && draft_n == 2 && mtp_margin_threshold > 0.0f) {
        if (!mtp_conf_log) {
            float v0 = 0.0f, v1 = 0.0f;
            logits_top2(s->mtp_logits, DS4_N_VOCAB, &mtp_last_top0, &v0, &mtp_last_top1, &v1);
            mtp_last_margin = v0 - v1;
        }
        if (mtp_last_margin < mtp_margin_threshold) {
            float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
            const int start = s->checkpoint.len;
            const double verify_t0 = mtp_timing ? now_sec() : 0.0;
            bool ok = metal_graph_eval_token_raw_swa(&s->graph,
                                                     &e->model,
                                                     &e->weights,
                                                     drafts[0],
                                                     (uint32_t)start,
                                                     row_logits);
            if (!ok) {
                free(row_logits);
                snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
                s->checkpoint_valid = false;
                return -1;
            }
            memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            free(row_logits);
            token_vec_push(&s->checkpoint, drafts[0]);
            accepted[n_accept++] = drafts[0];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(1);
            if (mtp_timing) {
                const double done = now_sec();
                fprintf(stderr,
                        "ds4: mtp timing margin-skip drafted=2 committed=1 margin=%.3f threshold=%.3f draft=%.3f ms verify=%.3f ms total=%.3f ms\n",
                        mtp_last_margin,
                        mtp_margin_threshold,
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (done - verify_t0) * 1000.0,
                        (done - mtp_t0) * 1000.0);
            }
            return n_accept;
        }
    }

    /*
     * The useful N=2 verifier is the tiny batch path: it verifies two target
     * positions in one layer-major pass and commits prefix-1 directly on a
     * partial accept.  Like the rest of the non-quality Metal path, it may pick
     * a different greedy token when batched reductions perturb nearly-tied
     * logits.  --quality / DS4_MTP_STRICT selects the exact decode verifier,
     * which preserves the one-token target stream but is not a speed win.
     */
    const bool use_decode2_exact =
        draft_n == 2 && strict_mtp && getenv("DS4_MTP_BATCH_VERIFY") == NULL;
    if (use_decode2_exact) {
        ds4_spec_frontier frontier;
        memset(&frontier, 0, sizeof(frontier));
        float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
        float *row0_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row0_logits[0]));
        const int start = s->checkpoint.len;
        int row0_top = -1;
        const double snapshot_t0 = mtp_timing ? now_sec() : 0.0;
        bool have_frontier = spec_frontier_snapshot(&frontier, s);
        const double snapshot_done = mtp_timing ? now_sec() : 0.0;
        bool ok = have_frontier;
        if (ok) {
            ok = metal_graph_verify_decode2_exact(&s->graph,
                                                  &e->model,
                                                  &e->weights,
                                                  drafts[0],
                                                  drafts[1],
                                                  (uint32_t)start,
                                                  &row0_top,
                                                  row0_logits,
                                                  row_logits);
        }
        const double verify_done = mtp_timing ? now_sec() : 0.0;
        if (ok && row0_top == drafts[1]) {
            memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            token_vec_push(&s->checkpoint, drafts[0]);
            token_vec_push(&s->checkpoint, drafts[1]);
            accepted[n_accept++] = drafts[0];
            if (n_accept < accepted_cap) accepted[n_accept++] = drafts[1];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(2);
            if (mtp_timing) {
                fprintf(stderr,
                        "ds4: mtp timing decode2 drafted=2 committed=2 draft=%.3f ms snapshot=%.3f ms verify=%.3f ms total=%.3f ms\n",
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (snapshot_done - snapshot_t0) * 1000.0,
                        (verify_done - snapshot_done) * 1000.0,
                        (now_sec() - mtp_t0) * 1000.0);
            }
            spec_frontier_free(&frontier);
            free(row0_logits);
            free(row_logits);
            return n_accept;
        }

        if (ok) {
            s->checkpoint.len = start;
            ok = spec_frontier_commit_prefix1(s);
        }
        if (ok) memcpy(s->logits, row0_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        if (ok) {
            token_vec_push(&s->checkpoint, drafts[0]);
            accepted[n_accept++] = drafts[0];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(1);
            if (mtp_timing) {
                const double replay_done = now_sec();
                fprintf(stderr,
                        "ds4: mtp timing decode2 drafted=2 committed=1 draft=%.3f ms snapshot=%.3f ms verify=%.3f ms prefix=%.3f ms total=%.3f ms\n",
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (snapshot_done - snapshot_t0) * 1000.0,
                        (verify_done - snapshot_done) * 1000.0,
                        (replay_done - verify_done) * 1000.0,
                        (replay_done - mtp_t0) * 1000.0);
            }
            spec_frontier_free(&frontier);
            free(row0_logits);
            free(row_logits);
            return n_accept;
        }
        if (have_frontier) {
            s->checkpoint.len = start;
            (void)spec_frontier_restore(&frontier, s);
        }
        spec_frontier_free(&frontier);
        free(row0_logits);
        free(row_logits);
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp decode2 verifier failed, falling back to sequential\n");
        }
    }

    if (!use_decode2_exact)
    {
        ds4_spec_frontier frontier;
        memset(&frontier, 0, sizeof(frontier));
        int *row_tops = xmalloc((size_t)draft_n * sizeof(row_tops[0]));
        float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
        const int start = s->checkpoint.len;
        /*
         * The production MTP depth is two.  Prefix-1 capture makes partial
         * accepts cheap, but it copies per-layer compressor frontiers even when
         * both draft tokens are accepted.  Full accepts are the path that makes
         * MTP worthwhile, so by default we snapshot before the verifier and
         * replay one token on partial accept.  DS4_MTP_CAPTURE_PREFIX1 restores
         * the older no-replay partial path for measurement.
         */
        const bool capture_prefix1 =
            draft_n == 2 && (!strict_mtp || getenv("DS4_MTP_CAPTURE_PREFIX1") != NULL);
        const bool exact_replay_debug = getenv("DS4_MTP_EXACT_REPLAY") != NULL;
        const bool snapshot_required =
            draft_n > 2 ||
            (draft_n == 2 && (!capture_prefix1 || exact_replay_debug)) ||
            getenv("DS4_MTP_FORCE_SNAPSHOT") != NULL;
        bool have_frontier = false;
        bool ok = true;
        bool verifier_may_have_mutated = false;
        const double snapshot_t0 = mtp_timing ? now_sec() : 0.0;
        if (snapshot_required) {
            have_frontier = spec_frontier_snapshot(&frontier, s);
            ok = have_frontier;
        }
        const double snapshot_done = mtp_timing ? now_sec() : 0.0;
        if (ok) {
            for (int i = 0; i < draft_n; i++) token_vec_push(&s->checkpoint, drafts[i]);
            verifier_may_have_mutated = true;
            ok = metal_graph_verify_suffix_tops(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                &s->checkpoint,
                                                (uint32_t)start,
                                                (uint32_t)draft_n,
                                                capture_prefix1,
                                                row_tops,
                                                NULL);
        }
        const double micro_verify_done = mtp_timing ? now_sec() : 0.0;
        if (ok) {
            int commit_drafts = 1;
            for (int i = 1; i < draft_n; i++) {
                if (row_tops[i - 1] != drafts[i]) break;
                commit_drafts++;
            }
            if (mtp_conf_log) {
                fprintf(stderr,
                        "ds4: mtp conf drafted=%d committed=%d mtp_top=%d runner=%d margin=%.6f target_next=%d draft_next=%d\n",
                        draft_n,
                        commit_drafts,
                        mtp_last_top0,
                        mtp_last_top1,
                        mtp_last_margin,
                        draft_n > 1 ? row_tops[0] : -1,
                        draft_n > 1 ? drafts[1] : -1);
            }
            if (exact_replay_debug && have_frontier) {
                s->checkpoint.len = start;
                ok = spec_frontier_restore(&frontier, s);
                if (ok) {
                    int replayed = 0;
                    for (; replayed < commit_drafts && ok; replayed++) {
                        ok = metal_graph_eval_token_raw_swa(&s->graph,
                                                            &e->model,
                                                            &e->weights,
                                                            drafts[replayed],
                                                            (uint32_t)(start + replayed),
                                                            row_logits);
                        if (ok) token_vec_push(&s->checkpoint, drafts[replayed]);
                    }
                    if (ok) {
                        memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                        for (int i = 0; i < replayed && n_accept < accepted_cap; i++) {
                            accepted[n_accept++] = drafts[i];
                            if (drafts[i] == eos_token) break;
                        }
                        s->checkpoint_valid = true;
                        s->mtp_draft_valid = false;
                        DS4_MTP_KEEP_ACCEPTED(replayed);
                        spec_frontier_free(&frontier);
                        free(row_logits);
                        free(row_tops);
                        return n_accept;
                    }
                }
            }

            if (commit_drafts == draft_n) {
                ok = metal_graph_read_spec_logits_row(&s->graph,
                                                      (uint32_t)(draft_n - 1),
                                                      row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
                        accepted[n_accept++] = drafts[i];
                        if (drafts[i] == eos_token) break;
                    }
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(draft_n);
                    if (mtp_timing) {
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms total=%.3f ms\n",
                                draft_n,
                                draft_n,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (now_sec() - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }

            if (draft_n == 2 && commit_drafts == 1 && capture_prefix1) {
                s->checkpoint.len = start;
                const double prefix_t0 = mtp_timing ? now_sec() : 0.0;
                ok = spec_frontier_commit_prefix1(s);
                const double prefix_done = mtp_timing ? now_sec() : 0.0;
                if (ok) ok = metal_graph_read_spec_logits_row(&s->graph, 0, row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    accepted[n_accept++] = drafts[0];
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(1);
                    token_vec_push(&s->checkpoint, drafts[0]);
                    if (mtp_timing) {
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms prefix=%.3f ms total=%.3f ms noreplay=1\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (prefix_done - prefix_t0) * 1000.0,
                                (now_sec() - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            } else {
                s->checkpoint.len = start;
                ok = have_frontier && spec_frontier_restore(&frontier, s);
            }
            if (ok && draft_n == 2 && commit_drafts == 1) {
                ok = metal_graph_eval_token_raw_swa(&s->graph,
                                                    &e->model,
                                                    &e->weights,
                                                    drafts[0],
                                                    (uint32_t)start,
                                                    row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    accepted[n_accept++] = drafts[0];
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(1);
                    token_vec_push(&s->checkpoint, drafts[0]);
                    if (mtp_timing) {
                        const double replay_done = now_sec();
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms exact_replay=%.3f ms total=%.3f ms\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (replay_done - micro_verify_done) * 1000.0,
                                (replay_done - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }
            if (ok) {
                for (int i = 0; i < commit_drafts; i++) token_vec_push(&s->checkpoint, drafts[i]);
                ok = metal_graph_verify_suffix_tops(&s->graph,
                                                    &e->model,
                                                    &e->weights,
                                                    &s->checkpoint,
                                                    (uint32_t)start,
                                                    (uint32_t)commit_drafts,
                                                    false,
                                                    row_tops,
                                                    NULL);
                if (ok) ok = metal_graph_read_spec_logits_row(&s->graph,
                                                              (uint32_t)(commit_drafts - 1),
                                                              row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    for (int i = 0; i < commit_drafts && n_accept < accepted_cap; i++) {
                        accepted[n_accept++] = drafts[i];
                        if (drafts[i] == eos_token) break;
                    }
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(commit_drafts);
                    if (mtp_timing) {
                        const double replay_done = now_sec();
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms replay=%.3f ms total=%.3f ms\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (replay_done - micro_verify_done) * 1000.0,
                                (replay_done - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }
        }
        s->checkpoint.len = start;
        if (have_frontier) {
            (void)spec_frontier_restore(&frontier, s);
        } else if (!verifier_may_have_mutated) {
            /* Snapshot setup failed before the verifier touched Metal state.
             * Fall through to the exact sequential verifier below. */
        } else {
            snprintf(err, errlen, "MTP verifier failed");
            s->checkpoint_valid = false;
            DS4_MTP_KEEP_ACCEPTED(0);
            spec_frontier_free(&frontier);
            free(row_logits);
            free(row_tops);
            return -1;
        }
        spec_frontier_free(&frontier);
        free(row_logits);
        free(row_tops);
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp spec micro verifier failed, falling back to sequential\n");
        }
    }

    /*
     * Safety fallback: if the production microbatch verifier fails, verify
     * drafts with the exact normal one-token decode path instead of returning
     * wrong state.  This path is deliberately slow and should not be selected
     * during normal --mtp operation.
     */
    int verified = 0;
    int target_top = sample_argmax(s->logits, DS4_N_VOCAB);
    bool logits_on_host = true;
    const double seq_t0 = mtp_timing ? now_sec() : 0.0;
    for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
        if (target_top != drafts[i]) {
            if (getenv("DS4_MTP_SPEC_LOG")) {
                fprintf(stderr,
                        "ds4: mtp spec seq miss at=%d draft=%d base=%d drafted=%d accepted=%d\n",
                        i,
                        drafts[i],
                        target_top,
                        draft_n,
                        n_accept);
            }
            break;
        }
        if (!metal_graph_eval_token_raw_swa_top(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                drafts[i],
                                                (uint32_t)s->checkpoint.len,
                                                &target_top,
                                                NULL))
        {
            snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return -1;
        }
        token_vec_push(&s->checkpoint, drafts[i]);
        logits_on_host = false;
        accepted[n_accept++] = drafts[i];
        verified++;
        if (drafts[i] == eos_token) break;
    }
    if (verified > 0 && !logits_on_host) {
        if (ds4_gpu_tensor_read(s->graph.logits,
                                  0,
                                  s->logits,
                                  (uint64_t)DS4_N_VOCAB * sizeof(s->logits[0])) == 0)
        {
            snprintf(err, errlen, "%s logits readback failed", ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return -1;
        }
        logits_on_host = true;
    }
    (void)logits_on_host;
    DS4_MTP_KEEP_ACCEPTED(verified);
#undef DS4_MTP_KEEP_ACCEPTED
    if (mtp_timing) {
        fprintf(stderr,
                "ds4: mtp timing seq drafted=%d verified=%d draft=%.3f ms verify=%.3f ms total=%.3f ms\n",
                draft_n,
                verified,
                (mtp_t_after_draft - mtp_t0) * 1000.0,
                (now_sec() - seq_t0) * 1000.0,
                (now_sec() - mtp_t0) * 1000.0);
    }
    if (getenv("DS4_MTP_SPEC_LOG")) {
        if (verified == draft_n) {
            fprintf(stderr,
                    "ds4: mtp spec seq accept drafted=%d accepted=%d\n",
                    draft_n,
                    n_accept);
        } else {
            fprintf(stderr,
                    "ds4: mtp spec seq partial drafted=%d verified=%d accepted=%d\n",
                    draft_n,
                    verified,
                    n_accept);
        }
    }
    return n_accept;
#endif
}

void ds4_session_invalidate(ds4_session *s) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
}

void ds4_session_rewind(ds4_session *s, int pos) {
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    s->mtp_draft_valid = false;
}

int ds4_session_pos(ds4_session *s) {
    return s->checkpoint.len;
}

int ds4_session_ctx(ds4_session *s) {
    return s->ctx_size;
}
