/* pulsar_engine_internal.h — internal shared declarations for the engine sources.
 * Produced by the multi-TU split of ds4.c; edit freely (the
 * generator is not part of the build). */
#ifndef PULSAR_ENGINE_INTERNAL_H
#define PULSAR_ENGINE_INTERNAL_H

/* =========================================================================
 * ds4.c - DeepSeek V4 inference engine.
 * =========================================================================
 *
 * This file is deliberately vertical: it owns GGUF loading, the fixed
 * DeepSeek V4 tensor layouts, CPU reference kernels, the whole-model GPU
 * graph driver, and tokenizer wiring.  Model shape selection is intentionally
 * narrow: validation accepts the known Flash and Pro layouts and fails early
 * for anything else.
 *
 * Loading is mmap based.  The loader parses only the GGUF header, metadata
 * table, and tensor directory.  Tensor data stays in the kernel page cache
 * until inference touches it, or until GPU wraps slices of the mapping as
 * no-copy MTLBuffers.
 */

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

#include "pulsar.h"

#include "pulsar_gpu.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PULSAR_NEG_INF (-1.0e30f)
#define PULSAR_POS_INF ( 1.0e30f)
#define PULSAR_DEFAULT_RMS_EPS ( 1.0e-6f)
#define PULSAR_DEFAULT_HC_EPS  ( 1.0e-6f)
#define PULSAR_DEFAULT_SWIGLU_CLAMP_EXP    (10.0f)
#define PULSAR_DEFAULT_ROPE_FREQ_BASE      (10000.0f)
#define PULSAR_DEFAULT_ROPE_SCALE_FACTOR   (16.0f)
#define PULSAR_DEFAULT_ROPE_YARN_BETA_FAST (32.0f)
#define PULSAR_DEFAULT_ROPE_YARN_BETA_SLOW (1.0f)
#define PULSAR_DEFAULT_COMPRESS_ROPE_FREQ_BASE (160000.0f)
#define PULSAR_DEFAULT_ROPE_ORIG_CTX       UINT64_C(65536)

static const char PULSAR_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";


/* DeepSeek recommends Think Max only with at least a 384K-token context window.
 * Below that size we keep ordinary thinking to avoid injecting a prompt that
 * asks for a reasoning budget the allocated context is not meant to hold. */
#define PULSAR_THINK_MAX_MIN_CONTEXT 393216u


#if defined(__GNUC__) || defined(__clang__)
#define PULSAR_MAYBE_UNUSED __attribute__((unused))
#else
#define PULSAR_MAYBE_UNUSED
#endif

/* ---- shared macros ---- */



#define PULSAR_MODEL_SHAPE_NAME          (g_pulsar_shape.name)
#define PULSAR_MODEL_VARIANT             (g_pulsar_shape.variant)
#define PULSAR_N_LAYER                   (g_pulsar_shape.n_layer)
#define PULSAR_N_EMBD                    (g_pulsar_shape.n_embd)
#define PULSAR_N_VOCAB                   (g_pulsar_shape.n_vocab)
#define PULSAR_N_HEAD                    (g_pulsar_shape.n_head)
#define PULSAR_N_HEAD_KV                 (g_pulsar_shape.n_head_kv)
#define PULSAR_N_HEAD_DIM                (g_pulsar_shape.n_head_dim)
#define PULSAR_N_VALUE_DIM               (g_pulsar_shape.n_value_dim)
#define PULSAR_N_ROT                     (g_pulsar_shape.n_rot)
#define PULSAR_N_OUT_GROUP               (g_pulsar_shape.n_out_group)
#define PULSAR_N_LORA_Q                  (g_pulsar_shape.n_lora_q)
#define PULSAR_N_LORA_O                  (g_pulsar_shape.n_lora_o)
#define PULSAR_N_EXPERT                  (g_pulsar_shape.n_expert)
#define PULSAR_N_EXPERT_USED             (g_pulsar_shape.n_expert_used)
#define PULSAR_N_EXPERT_SHARED           (g_pulsar_shape.n_expert_shared)
#define PULSAR_N_FF_EXP                  (g_pulsar_shape.n_ff_exp)
#define PULSAR_N_HASH_LAYER              (g_pulsar_shape.n_hash_layer)
#define PULSAR_N_SWA                     (g_pulsar_shape.n_swa)
#define PULSAR_N_INDEXER_HEAD            (g_pulsar_shape.n_indexer_head)
#define PULSAR_N_INDEXER_HEAD_DIM        (g_pulsar_shape.n_indexer_head_dim)
#define PULSAR_N_INDEXER_TOP_K           (g_pulsar_shape.n_indexer_top_k)
#define PULSAR_N_HC                      (g_pulsar_shape.n_hc)
#define PULSAR_N_HC_SINKHORN_ITER        (g_pulsar_shape.n_hc_sinkhorn_iter)
#define PULSAR_RMS_EPS                   (g_pulsar_shape.rms_eps)
#define PULSAR_HC_EPS                    (g_pulsar_shape.hc_eps)
#define PULSAR_EXPERT_WEIGHT_SCALE       (g_pulsar_shape.expert_weight_scale)
#define PULSAR_SWIGLU_CLAMP_EXP          (g_pulsar_shape.swiglu_clamp_exp)
#define PULSAR_ROPE_FREQ_BASE            (g_pulsar_shape.rope_freq_base)
#define PULSAR_ROPE_SCALE_FACTOR         (g_pulsar_shape.rope_scale_factor)
#define PULSAR_ROPE_YARN_BETA_FAST       (g_pulsar_shape.rope_yarn_beta_fast)
#define PULSAR_ROPE_YARN_BETA_SLOW       (g_pulsar_shape.rope_yarn_beta_slow)
#define PULSAR_COMPRESS_ROPE_FREQ_BASE   (g_pulsar_shape.compress_rope_freq_base)
#define PULSAR_ROPE_ORIG_CTX             (g_pulsar_shape.rope_orig_ctx)

/* =========================================================================
 * GGUF Quant Block Formats.
 * =========================================================================
 *
 * These layouts and IQ2 tables match the GGUF quantized tensor format,
 * reduced to only the formats ds4.c currently reads:
 *   - Q2_K routed down experts
 *   - IQ2_XXS routed gate/up experts
 *   - Q8_K temporary activation blocks for dot products
 */
#define QK_K 256


#define PULSAR_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]


/* =========================================================================
 * Shared Helpers, Allocation Guards, Threads, and Cursor Reads.
 * =========================================================================
 *
 * This section holds process-wide utilities used by all later stages:
 * fatal-error helpers, allocation wrappers, the persistent CPU worker pool,
 * and the small byte cursor used to parse GGUF metadata.
 */

#define PULSAR_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */
#define PULSAR_MAX_DIMS   8


#define PULSAR_MAX_THREADS 32


/* MXKV format selector passed to the pulsar_gpu_mxkv_* kernels (must match
 * PULSAR_MXKV_FMT_FP4 in src/cuda/pulsar_cuda_internal.h). */
#define PULSAR_ENGINE_MXKV_FMT_FP4 2u
/* MXKV FP4 row bytes for the indexer compressed cache (head_dim 128):
 * 64 nibble-pair bytes + 4 E8M0 block-32 scale bytes = 68 B (vs 512 f32). */
#define PULSAR_ENGINE_IDXFP4_ROWBYTES \
    (((uint64_t)PULSAR_N_INDEXER_HEAD_DIM + 1u) / 2u + \
     (((uint64_t)PULSAR_N_INDEXER_HEAD_DIM + 31u) / 32u))

/* PULSAR_ATTN_PACK (runtime env, see gpu_graph_attn_pack_enabled()): VALUE-
 * PRESERVING packed attn comp cache.  One row = [448 e4m3 nope bytes][7 E8M0
 * block-64 scale bytes][1 pad][64 f32 rope] = 712 B (vs 2048 f32).  The nope
 * dims store exactly the fp8_kv_quantize roundtrip the f32 pipeline applies
 * in place; the rope tail stays f32 — read-back is bit-identical.  Must stay
 * in sync with PULSAR_ATTN_PACK_ROWBYTES in src/cuda/pulsar_cuda_internal.h. */
#define PULSAR_ENGINE_ATTN_PACK_ROWBYTES \
    ((uint64_t)(PULSAR_N_HEAD_DIM - PULSAR_N_ROT) + \
     ((((uint64_t)(PULSAR_N_HEAD_DIM - PULSAR_N_ROT) / 64u) + 3u) & ~3ull) + \
     (uint64_t)PULSAR_N_ROT * 4u)


/* =========================================================================
 * Session Snapshot Payloads.
 * =========================================================================
 *
 * The server disk cache stores a high-level file header, then delegates the
 * graph-specific payload below to the engine.  This payload is intentionally
 * not mmaped: restoring a checkpoint copies bytes back into the already
 * allocated GPU tensors, preserving the same live graph buffers used by
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

#define PULSAR_SESSION_IO_CHUNK (8u * 1024u * 1024u)
#define PULSAR_DSPARK_DRAFT_WINDOW 128u
#define PULSAR_DSPARK_NOISE_TOKEN_ID 128799

/* ---- shared types ---- */

/* =========================================================================
 * DeepSeek V4 Shape Profiles.
 * =========================================================================
 *
 * The weight binder and metadata validator select one of the known model
 * profiles below.  Arrays reserve the maximum Pro dimensions; hot loops read
 * the active profile after GGUF validation.
 */

enum {
    PULSAR_MAX_LAYER            = 61,
    PULSAR_MAX_EMBD             = 7168,
    PULSAR_MAX_VOCAB            = 129280,
    PULSAR_MAX_HEAD             = 128,
    PULSAR_MAX_HEAD_KV          = 1,
    PULSAR_MAX_HEAD_DIM         = 512,
    PULSAR_MAX_VALUE_DIM        = 512,
    PULSAR_MAX_ROT              = 64,
    PULSAR_MAX_OUT_GROUP        = 16,
    PULSAR_MAX_LORA_Q           = 1536,
    PULSAR_MAX_LORA_O           = 1024,
    PULSAR_MAX_EXPERT           = 384,
    PULSAR_MAX_EXPERT_USED      = 6,
    PULSAR_MAX_EXPERT_SHARED    = 1,
    PULSAR_MAX_FF_EXP           = 3072,
    PULSAR_MAX_HASH_LAYER       = 3,
    PULSAR_MAX_SWA              = 128,
    PULSAR_MAX_INDEXER_HEAD     = 64,
    PULSAR_MAX_INDEXER_HEAD_DIM = 128,
    PULSAR_MAX_INDEXER_TOP_K    = 1024,
    PULSAR_MAX_HC               = 4,
    PULSAR_MAX_HC_SINKHORN_ITER = 20,
};

typedef enum {
    PULSAR_VARIANT_FLASH = 0,
    PULSAR_VARIANT_PRO   = 1,
} pulsar_variant;

typedef struct {
    const char *name;
    pulsar_variant variant;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_value_dim;
    uint32_t n_rot;
    uint32_t n_out_group;
    uint32_t n_lora_q;
    uint32_t n_lora_o;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_shared;
    uint32_t n_ff_exp;
    uint32_t n_hash_layer;
    uint32_t n_swa;
    uint32_t n_indexer_head;
    uint32_t n_indexer_head_dim;
    uint32_t n_indexer_top_k;
    uint32_t n_hc;
    uint32_t n_hc_sinkhorn_iter;
    float rms_eps;
    float hc_eps;
    float expert_weight_scale;
    float swiglu_clamp_exp;
    float rope_freq_base;
    float rope_scale_factor;
    float rope_yarn_beta_fast;
    float rope_yarn_beta_slow;
    float compress_rope_freq_base;
    uint64_t rope_orig_ctx;
} pulsar_shape;

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;


typedef struct {
    const char *ptr;
    uint64_t len;
} pulsar_str;

typedef pulsar_tokens token_vec;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} pulsar_cursor;

typedef void (*pulsar_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);

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

enum {
    PULSAR_TENSOR_F32      = 0,
    PULSAR_TENSOR_F16      = 1,
    PULSAR_TENSOR_Q8_0     = 8,
    PULSAR_TENSOR_Q2_K     = 10,
    PULSAR_TENSOR_IQ2_XXS  = 16,
    PULSAR_TENSOR_I32      = 26,
    PULSAR_TENSOR_BF16     = 30,
    PULSAR_TENSOR_FP8_E4M3 = 38,
    PULSAR_TENSOR_FP4_E2M1 = 39,
    /* CUTLASS block-scaled MXFP4: expert-major ColumnMajor E2M1 data blob
     * followed by a swizzled E8M0 SF blob, per expert. Byte size is NOT a
     * uniform per-element rate (see cutlass_mxfp4_expert_bytes()) -- the
     * gguf_types[] table entry for this type exists only so tensor_type()
     * recognizes it; real per-expert offsets come from that helper, not
     * from the table's block_elems/block_bytes. */
    PULSAR_TENSOR_CUTLASS_MXFP4 = 40,
    /* MXFP8_LT: the pre-store of PULSAR_TENSOR_FP8_E4M3 (type 38). Identical E4M3
     * weights and E8M0 block scales, but stored in the EXACT device-side layout
     * the runtime otherwise builds at first use: de-interleaved [in,out]
     * col-major E4M3 data immediately followed by the mx_sfoff()-swizzled E8M0
     * scale. On detection the FP8 matmul skips the per-weight cudaMalloc+convert
     * and points cuBLASLt straight at the mmap (g_model_device_base+offset),
     * freeing the ~6.4 GiB double-store. For 128-aligned shapes (out%128==0,
     * (in/32)%4==0 -- true for every shipped weight) the total byte size equals
     * the type-38 size exactly, so it shares the {32,33} table entry. */
    PULSAR_TENSOR_MXFP8_LT = 41,
};

typedef struct {
    pulsar_str key;
    uint32_t type;
    uint64_t value_pos;
} pulsar_kv;

typedef struct {
    pulsar_str name;
    uint32_t ndim;
    uint64_t dim[PULSAR_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
    /* Set only when this entry was swapped in from an overlay GGUF
     * (--expert-overlay): the payload lives at ext_map + abs_offset inside
     * the overlay file's mapping instead of the owning model's map. */
    const uint8_t *ext_map;
    uint64_t ext_size;
} pulsar_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    pulsar_kv *kv;
    pulsar_tensor *tensors;
} pulsar_model;

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} pulsar_array_ref;

typedef struct {
    uint64_t off;
    uint64_t end;
} accelerator_tensor_span;

typedef struct {
    pulsar_tensor *hc_attn_fn;
    pulsar_tensor *hc_attn_scale;
    pulsar_tensor *hc_attn_base;
    pulsar_tensor *attn_norm;
    pulsar_tensor *attn_q_a;
    pulsar_tensor *attn_q_a_norm;
    pulsar_tensor *attn_q_b;
    pulsar_tensor *attn_kv;
    pulsar_tensor *attn_kv_a_norm;
    pulsar_tensor *attn_sinks;
    pulsar_tensor *attn_output_a;
    pulsar_tensor *attn_output_b;
    pulsar_tensor *attn_compressor_ape;
    pulsar_tensor *attn_compressor_kv;
    pulsar_tensor *attn_compressor_gate;
    pulsar_tensor *attn_compressor_norm;
    pulsar_tensor *indexer_attn_q_b;
    pulsar_tensor *indexer_proj;
    pulsar_tensor *indexer_compressor_ape;
    pulsar_tensor *indexer_compressor_kv;
    pulsar_tensor *indexer_compressor_gate;
    pulsar_tensor *indexer_compressor_norm;
    pulsar_tensor *hc_ffn_fn;
    pulsar_tensor *hc_ffn_scale;
    pulsar_tensor *hc_ffn_base;
    pulsar_tensor *ffn_norm;
    pulsar_tensor *ffn_gate_tid2eid;
    pulsar_tensor *ffn_gate_inp;
    pulsar_tensor *ffn_exp_probs_b;
    pulsar_tensor *ffn_gate_exps;
    pulsar_tensor *ffn_up_exps;
    pulsar_tensor *ffn_down_exps;
    pulsar_tensor *ffn_gate_shexp;
    pulsar_tensor *ffn_up_shexp;
    pulsar_tensor *ffn_down_shexp;
} pulsar_layer_weights;

typedef struct {
    pulsar_tensor *token_embd;
    pulsar_tensor *output_hc_base;
    pulsar_tensor *output_hc_fn;
    pulsar_tensor *output_hc_scale;
    pulsar_tensor *output_norm;
    pulsar_tensor *output;
    pulsar_layer_weights layer[PULSAR_MAX_LAYER];
} pulsar_weights;

typedef struct {
    pulsar_tensor *main_proj;
    pulsar_tensor *main_norm;
    pulsar_layer_weights layer[3];
    pulsar_tensor *markov_w1;
    pulsar_tensor *markov_w2;
    pulsar_tensor *confidence_proj;
    pulsar_tensor *hc_head_base;
    pulsar_tensor *hc_head_fn;
    pulsar_tensor *hc_head_scale;
    pulsar_tensor *final_norm;
    uint32_t embed_dim;
    uint32_t vocab_size;
    uint32_t target_layer_ids[3];
} pulsar_dspark_weights;

typedef struct {
    float *out;
    const uint16_t *data;
    const float *x;
    uint64_t in_dim;
} matvec_f16_ctx;

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

typedef struct {
    float *out;
    const float *data;
    const float *x;
    uint64_t in_dim;
} matvec_f32_ctx;

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

typedef struct {
    float *mid;
    const uint8_t *gate_base[PULSAR_MAX_EXPERT_USED];
    const uint8_t *up_base[PULSAR_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[PULSAR_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[PULSAR_MAX_EXPERT_USED];
    uint64_t up_row_bytes[PULSAR_MAX_EXPERT_USED];
    int n_expert;
} matvec_iq2_xxs_mid_ctx;

typedef struct {
    float *out;
    const uint8_t *base;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes;
} matvec_q2_k_ctx;

typedef struct {
    float *out;
    const uint8_t *base[PULSAR_MAX_EXPERT_USED];
    const block_q8_K *xq[PULSAR_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[PULSAR_MAX_EXPERT_USED];
    int n_expert;
} matvec_q2_k_accum_ctx;

typedef struct {
    uint32_t token;
    uint32_t slot;
} pulsar_expert_pair;

typedef struct {
    float *mid;
    const uint8_t *gate_base[PULSAR_MAX_EXPERT];
    const uint8_t *up_base[PULSAR_MAX_EXPERT];
    const block_q8_K *xq;
    const pulsar_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[PULSAR_MAX_EXPERT];
    uint64_t up_row_bytes[PULSAR_MAX_EXPERT];
    uint64_t xq_blocks;
} matvec_iq2_xxs_batch_mid_ctx;

typedef struct {
    const float *mid;
    block_q8_K *midq;
    uint64_t down_in_dim;
    uint64_t down_blocks;
} quantize_mid_pairs_ctx;

typedef struct {
    float *down_pair;
    const uint8_t *base[PULSAR_MAX_EXPERT];
    const block_q8_K *midq;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[PULSAR_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q2_k_batch_down_ctx;

typedef struct {
    float *moe;
    const uint8_t *base[PULSAR_MAX_EXPERT];
    const block_q8_K *midq;
    const pulsar_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[PULSAR_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q2_k_batch_accum_rows_ctx;

typedef struct {
    float *moe;
    const float *down_pair;
    uint32_t n_tok;
    uint64_t out_dim;
} sum_down_pairs_ctx;

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

typedef struct {
    const pulsar_model *model;
    const pulsar_tensor *fn;
    const pulsar_tensor *scale;
    const pulsar_tensor *base;
    const pulsar_tensor *norm_w;
    const float *inp_hc;
    float *residual_hc;
    float *cur;
    float *norm;
    float *post;
    float *comb;
    uint64_t hc_dim;
    uint32_t n_hc;
} hc_pre_norm_batch_ctx;

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

typedef struct {
    float *mid;
    const float *gate;
    const float *up;
    uint64_t n;
    float clamp;
} swiglu_batch_ctx;

typedef struct {
    float *moe;
    const pulsar_model *model;
    const pulsar_layer_weights *layer;
    const float *norm;
    const int *token_ids;
    uint64_t expert_in_dim;
    uint64_t down_in_dim;
    uint32_t il;
} routed_moe_tokens_ctx;

typedef struct {
    float *out_hc;
    const pulsar_model *model;
    const pulsar_layer_weights *layer;
    const float *inp_hc;
    const int *token_ids;
    const float *steering_dirs;
    float steering_scale;
    uint64_t hc_dim;
    uint32_t il;
} layer_ffn_tokens_ctx;

/* =========================================================================
 * KV Cache, Compressors, and CPU Layer Execution.
 * =========================================================================
 *
 * The CPU path is the correctness reference.  It maintains raw SWA KV rows,
 * optional compressed KV rows, the indexer mask for ratio-4 layers, and a
 * reusable decode scratch arena so token generation does not allocate in the
 * hot loop.
 */

/* =========================================================================
 * GPU Release Graph State.
 * =========================================================================
 *
 * The release GPU executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

/* Tier-2 multi-session bank pool (compile-time bound on co-scheduled
 * sessions; the runtime co-schedule cap is a later, smaller number).
 *
 * Multi-sequence batched-decode KV banking design (per-row positions[]/
 * seq_id[] descriptors over fixed per-bank KV slabs) adapted from the
 * MIT-licensed Entrpi/ds4 fork (https://github.com/Entrpi/ds4, v0.2,
 * c71a49ac9316db02eaa6322dee2c919e6de1e792).  Reimplemented from scratch
 * against this engine's packed MXFP8/MXFP4 KV layout; no Entrpi code was
 * copied. */
#define PULSAR_MSEQ_MAX 8u

/* Fixed per-bank KV slabs: per layer, one contiguous allocation per cache
 * kind, bank-major, stride = one bank's single-session capacity.  When the
 * pool is enabled (n_banks >= 2), the graph's per-layer cache pointers
 * (layer_raw_cache[il] etc.) are VIEWS into these slabs — a single-bank view
 * means every existing single-session code path (prefill, decode, snapshot,
 * spec) runs unmodified against that bank; the batched decode kernels address
 * other banks with per-row seq_id[t]*cap offsets over the whole slab.
 *
 * The ctx-scaled comp/index slabs are cudaMallocManaged: on GB10 unified
 * memory that is the demand-paged analog of the reference's cuMemAddressReserve
 * VMM scheme (physical pages materialize on first touch, address math is
 * byte-identical to an eager slab), so short sessions do not pay resident
 * memory for worst-case padding.  Raw rings and compressor state lanes are
 * eager (fixed floor).  n_banks == 0 means the pool is disabled and the graph
 * owns plain single-session cache tensors. */
typedef struct {
    uint32_t n_banks;                        /* 0 = pool disabled */
    uint32_t cur_bank;                       /* bank the installed views address */
    uint64_t raw_bank_bytes;                 /* raw_cap * head_dim * raw elem */
    uint64_t comp_bank_bytes[PULSAR_MAX_LAYER];  /* layer_comp_cap * comp row bytes */
    uint64_t index_bank_bytes[PULSAR_MAX_LAYER]; /* layer_comp_cap * indexer row bytes */
    uint64_t astate_bank_bytes[PULSAR_MAX_LAYER];/* attn compressor frontier lane */
    uint64_t istate_bank_bytes[PULSAR_MAX_LAYER];/* indexer compressor frontier lane */
    pulsar_gpu_tensor *raw[PULSAR_MAX_LAYER];
    /* Tier-2 task #55 (increment 2a): the ctx-scaled comp/index caches are now
     * ONE cudaMallocManaged allocation PER BANK (comp[il][bank]) instead of one
     * n_banks*bank_bytes slab — so the increment-2 eviction guard can cudaFree a
     * single idle bank's physical directly (the only reclaim primitive that
     * returns memory on GB10; Step-1). The stride stays UNIFORM 1M (comp_bank_bytes
     * is per-layer, bank-independent) — this is NOT B-full (no variable caps). The
     * seq_id-scattered batched-decode READ kernels can no longer address a bank as
     * base + seq_id*comp_cap across one slab, so they take a per-bank BASE-POINTER
     * table (comp_bases[il]/index_bases[il]): a device array of the n_banks
     * comp[il][*]->ptr, indexed by seq_id[t]. NULL when the pool is disabled
     * (single-session paths use the repointed view). */
    pulsar_gpu_tensor *comp[PULSAR_MAX_LAYER][PULSAR_MSEQ_MAX];
    pulsar_gpu_tensor *index[PULSAR_MAX_LAYER][PULSAR_MSEQ_MAX];
    pulsar_gpu_tensor *comp_bases[PULSAR_MAX_LAYER];  /* [n_banks] device ptr array: comp[il][b]->ptr */
    pulsar_gpu_tensor *index_bases[PULSAR_MAX_LAYER]; /* [n_banks] device ptr array: index[il][b]->ptr */
    pulsar_gpu_tensor *askv[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *assc[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *iskv[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *issc[PULSAR_MAX_LAYER];
    /* Tier-2 Option F: per-bank DSpark drafter context ring, bank-major
     * (~6.75 MB/bank: raw 0.75 + prompt 6).  Allocated in
     * gpu_graph_init_dspark_target only when the pool is enabled AND the
     * drafter is loaded; the graph's dspark_raw_cache[i]/dspark_prompt_h[i]
     * become bank views into these, swapped by gpu_graph_bank_repoint so the
     * spec path transparently uses the active bank's ring.  NULL otherwise. */
    uint64_t dspark_raw_bank_bytes;      /* DRAFT_WINDOW * head_dim * f32 */
    uint64_t dspark_prompt_bank_bytes;   /* DRAFT_WINDOW * n_embd  * f32 */
    pulsar_gpu_tensor *dspark_raw[3];       /* N * dspark_raw_bank_bytes */
    pulsar_gpu_tensor *dspark_prompt[3];    /* N * dspark_prompt_bank_bytes */
} pulsar_bank_slabs;

typedef struct {
    /* One-token decode tensors.  These stay allocated for the life of a
     * session; a generated token enters as an embedding in cur_hc and leaves as
     * logits after all 43 layers update their raw/compressed/indexer caches. */
    pulsar_gpu_tensor *cur_hc;
    pulsar_gpu_tensor *flat_hc;
    pulsar_gpu_tensor *hc_mix;
    pulsar_gpu_tensor *hc_split;
    pulsar_gpu_tensor *hc_pre;
    pulsar_gpu_tensor *hc_post;
    pulsar_gpu_tensor *hc_comb;
    pulsar_gpu_tensor *attn_cur;
    pulsar_gpu_tensor *attn_norm;
    pulsar_gpu_tensor *qr;
    pulsar_gpu_tensor *qr_norm;
    pulsar_gpu_tensor *q;
    pulsar_gpu_tensor *kv_raw;
    pulsar_gpu_tensor *kv;

    /* Persistent KV state.  Raw KV is a sliding-window ring per layer.  Ratio-4
     * layers also keep an indexer-compressed cache; ratio-128 layers keep only
     * the attention-compressed cache.  The small state tensors are compressor
     * frontiers for the next compressed row, so they must be snapshotted with
     * the row counters whenever a checkpoint is saved or partially rewound. */
    pulsar_gpu_tensor *layer_raw_cache[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *layer_attn_comp_cache[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *layer_attn_state_kv[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *layer_attn_state_score[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *layer_index_comp_cache[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *layer_index_state_kv[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *layer_index_state_score[PULSAR_MAX_LAYER];

    /* Speculative decoding scratch.  The drafter is allowed to mutate graph
     * state only if the target verifier can either commit it or restore the
     * saved frontiers. */
    pulsar_gpu_tensor *spec_attn_state_kv[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *spec_attn_state_score[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *spec_index_state_kv[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *spec_index_state_score[PULSAR_MAX_LAYER];
    /* Batched-copy descriptor tables for the frontier snapshot (layer->spec)
     * and restore (spec->layer) copy sets: one kernel launch instead of ~126
     * cudaMemcpy calls per direction. Built lazily on first snapshot; NULL
     * handle falls back to the per-tensor copy loop. */
    void *spec_snap_copies;
    void *spec_restore_copies;
    uint32_t spec_frontier_copy_n;
    uint64_t spec_frontier_copy_max_bytes;
    int spec_frontier_copy_init;
    /* Shared multi-row logits slab (16 rows x n_vocab f32), written by every
     * batched multi-row output head: the DSpark draft/verify passes,
     * gpu_graph_verify_suffix_tops, and the Tier-2 batched multi-session
     * decode driver.  Despite the "spec_" name it is NOT speculation-owned —
     * gpu_graph_alloc_raw_cap allocates it unconditionally so the batched
     * paths work with speculation disabled. */
    pulsar_gpu_tensor *spec_logits;
    uint32_t layer_n_comp[PULSAR_MAX_LAYER];
    uint32_t layer_n_index_comp[PULSAR_MAX_LAYER];
    uint32_t raw_cap;
    /* Maximum compressed-row capacity across layers.  Shared work buffers use
     * this worst-case size because ratio-4 indexer layers can still reach it. */
    uint32_t comp_cap;
    /* Persistent compressed caches are per layer, so size them from the actual
     * layer compression ratio instead of pessimistically using the ratio-4 cap
     * for every ratio-128 layer. */
    uint32_t layer_comp_cap[PULSAR_MAX_LAYER];
    uint32_t attn_comp_stage_cap;

    /* Per-layer work tensors.  They are reused in place by every layer instead
     * of allocating a generic graph arena.  This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    pulsar_gpu_tensor *comp_kv_cur;
    pulsar_gpu_tensor *comp_sc_cur;
    pulsar_gpu_tensor *attn_comp_stage;
    /* f32 shadow used only when PULSAR_ATTN_PACK is on: the prefill attention
     * consumers (and session save) read plain-f32 comp rows, so the persistent
     * packed comp cache is dequantized here (up to max layer_comp_cap rows,
     * bit-exact) before each prefill-attention read.  Decode reads the packed
     * cache natively. */
    pulsar_gpu_tensor *attn_comp_dequant;
    /* f32 staging used only when PULSAR_IDX_FP4 is on: the compressor emits new
     * indexer rows here (comp-cap rows, same row indices as the cache), and
     * the QAT+pack step stores them MXKV-FP4-packed into the persistent
     * layer_index_comp_cache.  Also reused for session-save dequant and
     * session-load repack. */
    pulsar_gpu_tensor *idx_comp_stage;
    pulsar_gpu_tensor *indexer_q;
    pulsar_gpu_tensor *indexer_weights;
    pulsar_gpu_tensor *indexer_scores;
    pulsar_gpu_tensor *comp_mask;
    pulsar_gpu_tensor *comp_selected;
    pulsar_gpu_tensor *heads;
    pulsar_gpu_tensor *attn_low;
    pulsar_gpu_tensor *attn_out;
    pulsar_gpu_tensor *after_attn_hc;
    pulsar_gpu_tensor *ffn_cur;
    pulsar_gpu_tensor *ffn_norm;
    pulsar_gpu_tensor *shared_gate;
    pulsar_gpu_tensor *shared_up;
    pulsar_gpu_tensor *shared_mid;
    pulsar_gpu_tensor *shared_out;
    pulsar_gpu_tensor *router_logits;
    pulsar_gpu_tensor *router_probs;
    pulsar_gpu_tensor *router_selected;
    pulsar_gpu_tensor *router_weights;
    pulsar_gpu_tensor *routed_gate;
    pulsar_gpu_tensor *routed_up;
    pulsar_gpu_tensor *routed_mid;
    pulsar_gpu_tensor *routed_down;
    pulsar_gpu_tensor *routed_out;
    pulsar_gpu_tensor *ffn_out;
    pulsar_gpu_tensor *after_ffn_hc;
    pulsar_gpu_tensor *output_pre;
    pulsar_gpu_tensor *output_weights;
    pulsar_gpu_tensor *output_embd;
    pulsar_gpu_tensor *output_norm;
    pulsar_gpu_tensor *logits;

    /* DSpark target hidden capture buffers */
    pulsar_gpu_tensor *dspark_target_h[3];
    pulsar_gpu_tensor *dspark_main_x;
    uint32_t dspark_target_layer_ids[3];
    /* Bulk prefill anchor-hidden capture for drafter retraining
     * (PULSAR_DSPARK_PREFILL_DUMP): per-chunk [prefill_cap, N_EMBD] buffers, one
     * per anchor layer. dspark_bulk_n is armed to the chunk's token count by
     * the prefill path and cleared by the drain; 0 everywhere else. */
    pulsar_gpu_tensor *dspark_bulk_h[3];
    uint32_t dspark_bulk_n;
    /* Prompt-window capture for drafter seeding: the anchor hiddens of the
     * last <=128 prompt positions, kept as a position%128 ring so the fused
     * loop can seed the drafter's context window at generation start (the
     * reference prefills this window; an empty or stale window collapses
     * drafter acceptance). dspark_prompt_n counts captured prompt positions. */
    pulsar_gpu_tensor *dspark_prompt_h[3];
    uint32_t dspark_prompt_n;    /* positions captured: ring valid for [lo, n) */
    uint32_t dspark_prompt_lo;
    /* Fused spec loop (P2): per-position anchor hiddens captured during the
     * verify batch — [spec cap, N_EMBD] per anchor layer. dspark_capture_batch_n
     * != 0 arms the capture in gpu_graph_encode_layer_batch for that many
     * positions; 0 = off (prefill and plain decode unaffected). */
    pulsar_gpu_tensor *dspark_target_h_batch[3];
    uint32_t dspark_capture_batch_n;
    /* Fused spec loop Stage B (no-replay rollback): per-position compressor
     * projections saved during the verify batch, so a partial accept can roll
     * the recurrent pool state forward from the frontier snapshot WITHOUT
     * replaying the transformer (the pool update kernels re-run from these
     * exact rows -> bit-identical state). [17 rows x width] per compressed
     * layer; the indexer compressor reuses batch_comp_kv/sc so it needs its
     * own save. spec_comp_save_n arms the save (0 = off). */
    pulsar_gpu_tensor *spec_comp_kv_save[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *spec_comp_sc_save[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *spec_icomp_kv_save[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *spec_icomp_sc_save[PULSAR_MAX_LAYER];
    pulsar_gpu_tensor *spec_comp_scratch_row;   /* emit sink during roll-forward */
    uint32_t spec_comp_save_n;
    /* Persistent drafter scratch (was per-call cudaMalloc/cudaFree churn --
     * cudaFree device-syncs, and the fused loop projects/seeds up to 5x/step). */
    pulsar_gpu_tensor *dspark_concat;       /* [3*N_EMBD] target_h concat */
    pulsar_gpu_tensor *dspark_proj_out;     /* [N_EMBD] pre-norm projection */
    pulsar_gpu_tensor *dspark_seed_kv;      /* [HEAD_DIM] seed kv scratch */
    pulsar_gpu_tensor *dspark_seed_norm;    /* [HEAD_DIM] */
    pulsar_gpu_tensor *dspark_seed_rot;     /* [HEAD_DIM] */
    pulsar_gpu_tensor *dspark_markov_logits; /* [N_VOCAB] markov refine scratch */

    /* DSpark draft KV raw caches (one per draft layer, window=128) */
    pulsar_gpu_tensor *dspark_raw_cache[3];
    uint32_t dspark_n_raw[3];

    /* Override compression ratio for DSpark draft layers (set to 0 before
     * calling gpu_graph_encode_decode_layer for draft model forwarding). */
    int comp_ratio_override;

    uint32_t prefill_cap;
    uint32_t raw_window;

    /* Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    pulsar_gpu_tensor *prefill_tokens;
    pulsar_gpu_tensor *batch_cur_hc;
    pulsar_gpu_tensor *batch_next_hc;
    pulsar_gpu_tensor *batch_flat_hc;
    pulsar_gpu_tensor *batch_hc_mix;
    pulsar_gpu_tensor *batch_hc_split;
    pulsar_gpu_tensor *batch_attn_cur;
    pulsar_gpu_tensor *batch_attn_norm;
    pulsar_gpu_tensor *batch_qr;
    pulsar_gpu_tensor *batch_qr_norm;
    pulsar_gpu_tensor *batch_q;
    pulsar_gpu_tensor *batch_kv_raw;
    pulsar_gpu_tensor *batch_kv;
    pulsar_gpu_tensor *batch_comp_kv;
    pulsar_gpu_tensor *batch_comp_sc;
    pulsar_gpu_tensor *batch_indexer_q;
    pulsar_gpu_tensor *batch_indexer_weights;
    pulsar_gpu_tensor *batch_heads;
    pulsar_gpu_tensor *batch_attn_low;
    pulsar_gpu_tensor *batch_attn_out;
    pulsar_gpu_tensor *batch_after_attn_hc;
    pulsar_gpu_tensor *batch_ffn_cur;
    pulsar_gpu_tensor *batch_ffn_norm;
    pulsar_gpu_tensor *batch_shared_gate;
    pulsar_gpu_tensor *batch_shared_up;
    pulsar_gpu_tensor *batch_shared_mid;
    pulsar_gpu_tensor *batch_shared_out;
    pulsar_gpu_tensor *batch_router_logits;
    pulsar_gpu_tensor *batch_router_probs;
    pulsar_gpu_tensor *batch_router_selected;
    pulsar_gpu_tensor *batch_router_weights;
    pulsar_gpu_tensor *batch_routed_gate;
    pulsar_gpu_tensor *batch_routed_up;
    pulsar_gpu_tensor *batch_routed_mid;
    pulsar_gpu_tensor *batch_routed_down;
    pulsar_gpu_tensor *batch_routed_out;
    bool batch_routed_mid_is_f16;
    pulsar_gpu_tensor *batch_ffn_out;
    bool materialize_ffn_out;
    pulsar_gpu_tensor *directional_steering_dirs;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    bool quality;

    /* Tier-2 bank pool (see pulsar_bank_slabs above).  banks.n_banks == 0 keeps
     * the classic single-session layout; >= 2 makes the per-layer cache
     * pointers bank views into the slabs. */
    pulsar_bank_slabs banks;

    /* PULSAR_DECODE_DESCR diagnostic (gpu_decode.c): 1-row device descriptor
     * arrays (positions=[pos], seq_id=[0]) for routing single-token decode
     * attention through the banked entry points as an n_banks=1 pool over the
     * bank-0 cache views.  Lazily allocated on first diagnostic step; NULL
     * in production (the env flag is read once per process, never per
     * token). */
    pulsar_gpu_tensor *descr_diag_pos;
    pulsar_gpu_tensor *descr_diag_seq;

    /* Tier-2 banked multiseq step state (increment 2 — per-bank compressor
     * frontiers).  The authoritative per-bank compressed-row counters are
     * ms_n_comp / ms_n_index_comp (indexed by TRUE bank id, never a packed
     * row ordinal); they are HOST bookkeeping owned by the multiseq driver:
     * gpu_graph_bank_repoint swaps device views only, and classic
     * single-session work against a repointed bank runs on the scalar
     * layer_n_comp counters — use gpu_graph_bank_counters_install /
     * _capture at the boundary.
     *
     * During a multiseq step (batch_multiseq armed by
     * gpu_graph_multiseq_step_begin), the scalar layer_n_comp /
     * layer_n_index_comp become CROSS-BANK SUPERSETS, written exactly once
     * at step top — the step's emit-inclusive visibility bound,
     * max over rows of (pos+1)/ratio — and never mutated mid-forward (the
     * structural avoidance of the reference fork's context-killing race:
     * cross-bank maxima are launch/scratch bounds only, never bank
     * addresses or extents).  The batched emit loop writes each emitted row
     * into seq_id[t]'s bank at that bank's frontier and bumps ONLY that
     * bank's ms counter; per-row raw-ring state needs no bookkeeping at all
     * (the ring is position-indexed: slot = pos % raw_cap per bank).
     *
     * ms_positions/ms_seq_id are the host mirrors the emit loop reads;
     * batch_positions/batch_seq_id the device arrays the kernels read.
     * All four are lazily allocated (prefill_cap entries) on the first
     * multiseq step; NULL in production single-session serving. */
    uint32_t ms_n_comp[PULSAR_MSEQ_MAX][PULSAR_MAX_LAYER];
    uint32_t ms_n_index_comp[PULSAR_MSEQ_MAX][PULSAR_MAX_LAYER];
    /* Tier-2 Option F: per-bank DSpark drafter-ring frontier counters (the
     * device rings themselves are banked slabs, pulsar_bank_slabs.dspark_*).
     * Captured/installed alongside ms_n_comp so each bank keeps a WARM drafter
     * window under N=2 spec-time-slice — the whole point of Option F. */
    uint32_t ms_dspark_n_raw[PULSAR_MSEQ_MAX][3];
    uint32_t ms_dspark_prompt_n[PULSAR_MSEQ_MAX];
    uint32_t ms_dspark_prompt_lo[PULSAR_MSEQ_MAX];
    /* Tier-2 PATH-A partial-prefix KV-reuse (plan-33). Net-new. ms_emit_keep[bank]
     * is the ratio-4 boundary-row restore threshold: 0 = inactive (increment A
     * full-prefix fork clears it; increment C's partial cut sets R/4+1 and the
     * emit hook overwrites the recomputed boundary row with the packed stash while
     * row0 < it). fork_pin[bank] is a transient eviction pin so the guard's victim
     * picker cannot free_physical a source bank mid-clone (plan-33 anti-corruption
     * guarantee). Both zero-initialised with the graph. */
    uint32_t ms_emit_keep[PULSAR_MSEQ_MAX];
    uint8_t  fork_pin[PULSAR_MSEQ_MAX];
    /* Boundary-row stash (inc C): one PACKED row per (bank, layer) — the ratio-4
     * comp row R/4 and index row R/4 copied byte-for-byte at fork_copy_cut, and
     * byte-REPLACED over the replay's recomputed row by gpu_graph_emit_keep_restore
     * (never re-encoded: bit-exact for MXFP8-pack AND the non-idempotent MXFP4 QAT
     * alike). Sized n_banks * PULSAR_N_LAYER * row_bytes at slab alloc; NULL when the
     * pool is disabled. */
    pulsar_gpu_tensor *emit_stash_comp;
    pulsar_gpu_tensor *emit_stash_index;
    int32_t *ms_positions;
    int32_t *ms_seq_id;
    pulsar_gpu_tensor *batch_positions;
    pulsar_gpu_tensor *batch_seq_id;
    bool batch_multiseq;
    uint32_t batch_multiseq_rows;
} pulsar_gpu_graph;

/* =========================================================================
 * Imatrix Collection.
 * =========================================================================
 *
 * The 2-bit DS4 quants care most about routed MoE experts.  For expert gate
 * and up matrices the matmul input is the FFN-normalized activation row.  For
 * expert down matrices the matmul input is the routed SwiGLU row after route
 * weighting.  During GPU prefill those tensors are already materialized as
 * `batch_ffn_norm`, `batch_router_selected`, and `batch_routed_mid`, so the
 * collector observes the exact release graph without changing inference math.
 *
 * The output is llama.cpp's legacy imatrix `.dat` format.  Entries are packed
 * by expert: one tensor entry contains `n_expert * n_columns` floats and the
 * quantizer slices the vector for each expert.
 */
typedef struct {
    float *gate_up_sum2;   /* [active layer][active expert][hidden] */
    float *down_sum2;      /* [active layer][active expert][expert FFN] */
    uint32_t gate_up_count[PULSAR_MAX_LAYER][PULSAR_MAX_EXPERT];
    uint32_t down_count[PULSAR_MAX_LAYER][PULSAR_MAX_EXPERT];
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
} pulsar_imatrix_collector;

typedef struct pulsar_vocab pulsar_vocab;

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
    pulsar_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;

struct owned_str;  /* forward decl: bpe_rank() param; full def appears later */

struct pulsar_vocab {
    pulsar_str *token;
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

    /* ---- methods (C++ port): 1:1 mirror of the vocab verb family in
     * tokenizer.cpp; bodies keep the auto *vocab = this alias, logic verbatim.
     * Names kept as-is (none carry the pulsar_vocab type-name prefix). ---- */
    int bpe_rank(const owned_str *a, const owned_str *b) const;
    void bpe_emit_piece(pulsar_str raw_piece, token_vec *out) const;
    void bpe_tokenize_text(const char *text, token_vec *out) const;
    int vocab_lookup(const char *text) const;
    void vocab_load(const pulsar_model *model);
    void vocab_free();
    bool special_token_at(const char *p, int *token, size_t *len) const;
    void tokenize_span(const char *p, size_t n, token_vec *out) const;
    void tokenize_rendered_chat_vocab(const char *text, token_vec *out) const;
    void bpe_tokenize_tool_result_text(const char *content, token_vec *out);
    void dump_tokens(const token_vec *tokens) const;
};

struct pulsar_engine {
    pulsar_model model;
    pulsar_model dspark_model;
    pulsar_vocab vocab;
    pulsar_weights weights;
    pulsar_dspark_weights dspark_weights;
    pulsar_backend backend;
    int dspark_draft_tokens;
    float dspark_confidence;
    char *directional_steering_file;
    float *directional_steering_dirs;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    uint32_t prefill_chunk;
    bool quality;
    bool gpu_ready;
    bool dspark_ready;
    bool dspark_external;   /* drafter opened from its own GGUF (own map/fd) */
    pulsar_model overlay_model;
    bool overlay_ready;
    /* Prometheus /metrics spec-decode counters (server /metrics endpoint via
     * pulsar_engine_spec_metrics). Incremented from the DSpark fused verify loop;
     * monotonic. GPU decode submission is single-threaded, so plain uint64 is
     * adequate for these monitoring counters. */
    uint64_t spec_accepted_tokens;         /* accepted draft tokens */
    uint64_t spec_draft_tokens;            /* proposed/verified draft tokens */
    uint64_t spec_num_drafts;              /* draft rounds (verify steps w/ drafts) */
    uint64_t spec_gen_tokens;              /* tokens emitted by the spec loop */
    uint64_t spec_accepted_per_pos[16];    /* accepted count per draft position */

    /* ---- methods (C++ port): 1:1 mirror of the pulsar_engine_* verb family.
     * The public API in pulsar.h stays the free-function facade (defined in
     * engine_api.cpp); engine internals call these members directly.  Members
     * stay public and the struct stays trivially constructible: lifetime is
     * managed exactly as before via open()/destroy() (xcalloc/free), NOT
     * constructors/destructors.
     * NOTE: pulsar_engine_dspark_draft_tokens stays a free function — a member
     * would collide with the data member of the same name. */
    static int open(pulsar_engine **out, const pulsar_engine_options *opt);
    void destroy();                        /* was pulsar_engine_close */
    void summary();
    int vocab_size();
    int logits_width() const;
    const char *model_name();
    int layer_count();
    void spec_metrics(pulsar_spec_metrics *out);
    uint32_t layer_compress_ratio(uint32_t layer);
    uint64_t hidden_f32_values();
    int model_id();
    bool is_pruned() const;
    uint64_t session_cost_bytes(int ctx_size);
    uint64_t session_cost_bytes_banked(int ctx_size, int n_banks);
    uint64_t demand_paged_bytes_per_bank(int ctx_size);
    uint64_t weights_resident_bytes();
    int generate_argmax(const pulsar_tokens *prompt,
                        int n_predict, int ctx_size,
                        pulsar_token_emit_fn emit,
                        pulsar_generation_done_fn done,
                        void *emit_ud,
                        pulsar_session_progress_fn progress,
                        void *progress_ud);
    int collect_imatrix(const char *dataset_path, const char *output_path,
                        int ctx_size, int max_prompts, int max_tokens);
    void dump_tokens(const pulsar_tokens *tokens);
    int head_test(const pulsar_tokens *prompt);
    int gpu_graph_test(const pulsar_tokens *prompt);
    int routed_quant_bits();
    bool has_output_head();
    bool has_dspark();
};

typedef struct owned_str {
    char *ptr;
    uint64_t len;
} owned_str;

typedef struct {
    int id;
    float logit;
    float prob;
} sample_candidate;

/* Reusable working set for pulsar_sample_dist_build's full-vocab (top_k <= 0)
 * path, which sorts all n_vocab candidates. Zero-initialize before first use;
 * grows on demand and is reused across calls, so the sampled speculative walk
 * does not malloc/free ~1.5 MB per accepted position. Free with
 * pulsar_sample_scratch_free.
 *
 * Caller-owned and NOT shared: each concurrent session runs its own sampled
 * acceptance walk, so this lives on pulsar_session, never on pulsar_engine. */
typedef struct {
    sample_candidate *cand;   /* sorted candidates              */
    uint64_t *keys;           /* packed (sort key << 32 | id)   */
    uint64_t *tmp;            /* radix ping-pong buffer         */
    /* Gather target for the min-p prefilter path: survivors are collected
     * into `cand` in ascending-id order (probs computed once, alongside the
     * full-vocab sum), then gathered here in descending sort order. A second
     * buffer because the gather cannot run in place and the degenerate
     * all-equal-logits case keeps every candidate (m == cap). */
    sample_candidate *cand2;
    uint32_t cap;             /* elements reserved in each      */
    /* Dense token->q(prob) map for pulsar_sample_dist_draw_residual, which needs
     * q(x) for each x in p's support: the linear pulsar_sample_dist_prob scan
     * would make that O(|p|*|q|) — 1.6e10 at full vocab. Sized by token id
     * (NOT by `cap`, which is top_k on the preselect path while ids still run
     * to n_vocab), and INVARIANT: all-zero on entry and on exit. The residual
     * draw scatters q in, reads, then re-zeros only q's own ids, so the clear
     * is O(|q|) rather than a full-vocab memset. */
    float *qmap;
    uint32_t qmap_cap;
} pulsar_sample_scratch;

void pulsar_sample_scratch_free(pulsar_sample_scratch *s);


/* The per-conversation SPECULATIVE / DSpark host shadow, factored into one
 * named aggregate so it can be saved and restored WHOLESALE.  It is embedded
 * BY VALUE in both pulsar_session (the live state) and pulsar_bank_carry (the
 * per-bank Tier-2 shadow), and pulsar_session_bank_state_save/_restore copy it
 * with a single struct assignment.  That is the whole point: the old
 * field-by-field mirror meant every new field had to be added at three sites
 * (struct, save, restore) and forgetting one silently handed a bank ANOTHER
 * conversation's speculative state.  Add new spec/DSpark scalars HERE and
 * both paths pick them up for free.
 *
 * Deliberately EXCLUDED: heap-backed state (checkpoint, logits,
 * dspark_pending_qrows) — those need deep copies with per-side ownership, so
 * they stay as explicit members of each struct. */
typedef struct pulsar_spec_carry_state {
    /* Fused DSpark loop (P2): drafts produced LAST step from the last-accepted
     * position's hidden, pending verification in THIS step's single batched
     * forward (EAGLE pipeline inversion). 0 pending = next step is a plain
     * n=1 forward. Invalidated on rewind/invalidate. */
    int32_t dspark_pending[16];
    uint32_t dspark_n_pending;
    /* The base token the pending drafts continue from (predicted greedy next).
     * If the caller's next first_token differs (non-greedy interruption, tool
     * injection), the pending drafts are stale and dropped. */
    int32_t dspark_pending_base;
    /* checkpoint.len the drafts were produced at — an ACCEPTANCE guard, not an
     * exactness guard. The base-token check above is a VALUE check, not an
     * identity check: a plain pulsar_session_eval (tool injection, think-tag
     * recovery) advances the session and clears the carry but leaves the
     * pendings, so a later first_token that merely COLLIDES with
     * dspark_pending_base would resurrect drafts conditioned on a different
     * position.
     *
     * Do NOT read this as an exactness guard and do NOT delete it on the grounds
     * that it isn't one. Both accept rules are PROPOSAL-AGNOSTIC: they are exact
     * for an arbitrary proposal and never read where it came from. q_oldpos IS
     * the distribution the stale draft was actually drawn from, so p_newpos/q_oldpos
     * is a perfectly valid ratio and the rule still emits exactly p_newpos — the
     * draft simply won't get accepted very often. That was true of the
     * deterministic rule (which is why staleness was benign before Item 1) and it
     * stays true under p/q. The cost of staleness is throughput; we drop stale
     * pendings because a draft conditioned on the wrong position is a wasted
     * verify row. Mirrors spec_carry_pos. */
    int32_t dspark_pending_pos;
    /* Speculative-sampling carry: the next base token, already drawn from the
     * request's filtered distribution (bonus draw on full accept, residual
     * draw on rejection) but NOT yet forwarded through the target. The next
     * generate_speculative call forwards it as batch position 0. Invalidated
     * with the pendings on rewind/invalidate/sync. */
    int32_t spec_carry_token;
    bool spec_carry_valid;
    /* checkpoint.len the carry was drawn at; any session advance outside the
     * speculative path (sync, plain eval) moves it and voids the carry */
    int32_t spec_carry_pos;
    /* sampling params the carry was drawn under; a param change between calls
     * drops the carry and redraws from s->logits (exact: the carry was never
     * emitted or forwarded) */
    float spec_carry_temp, spec_carry_top_p, spec_carry_min_p;
    int spec_carry_top_k;
    /* DTree Phase 0 (PULSAR_DTREE_STATS): the drafter #2 token and confidence
     * score for each pending draft, carried from the drafting step to the
     * verify step so a rejection can be scored p2 = P(target correction ==
     * drafter #2 | #1 rejected), bucketed by conf. Measurement-only. */
    int32_t dspark_pending_alt[16];
    float   dspark_pending_conf[16];
    /* --- Temperature-matched draft sampling (spec-decode Item 1) ---
     * At temperature > 0 the drafts above are DRAWN from a temperature-matched
     * proposal q (the drafter's refined logits filtered at the request's
     * params) rather than taken as the drafter's argmax. Verification then uses
     * the standard sampled-proposal rule — accept w.p. min(1, p/q), else draw
     * the residual (p-q)+ — which is not capped at p(mode) the way the
     * deterministic-proposal rule is.
     *
     * `dspark_pending_sampled` records which rule the pendings were PROPOSED
     * under, so verify applies the matching rule: false => argmax proposal =>
     * the deterministic rule (accept w.p. p(x), residual p-excluding). The two
     * rules are not interchangeable; applying p/q to an argmax proposal (or
     * vice versa) silently breaks exactness. */
    bool dspark_pending_sampled;
    /* q(pend[i]) at draft time — the accept denominator. */
    float dspark_pending_q[16];
    /* The sampling params the pendings were sampled under. TWO consumers, and
     * they are different in kind:
     *   1) EXACTNESS (load-bearing): the verify walk rebuilds the rejecting
     *      position's q from dspark_pending_qrows using THESE params, so the
     *      stored accept denominator dspark_pending_q[i] and the residual's q
     *      name the same proposal q_X by construction. This is why the rebuild
     *      must never be fed the live request params.
     *   2) THROUGHPUT (the guard in the verify path): drafts sampled under X and
     *      verified under Y are still exact — the rule is proposal-agnostic and
     *      returns exactly p_Y for any q — but q_X is a badly-matched proposal
     *      for p_Y, so acceptance craters. The guard drops them to avoid wasting
     *      verify rows, not to avoid bias.
     * Mirrors the spec_carry_* params guard. Greedy never needed this. */
    float dspark_pending_temp, dspark_pending_top_p, dspark_pending_min_p;
    int   dspark_pending_top_k;
    /* --- Terminal yield-quench controller (spec-decode Item 4) ---
     * Per-request cumulative-regret gate: each fused spec step charges
     * debt += guard(n_batch) - tokens_committed (the breakeven yield minus the
     * realized yield, in plain-token equivalents), and once the request has
     * demonstrably lost more than PULSAR_QUENCH_BUDGET plain tokens to
     * speculation with its recent yield still below breakeven, spec_quenched
     * latches and the request decodes PLAIN for its remainder (terminal;
     * re-armed at the request boundaries: sync/invalidate/rewind/load — the
     * same sites that drop the carry and pendings). All-zero is the armed
     * state, so xcalloc'd sessions start armed and spec_quench_reset is a
     * plain zeroing. Controller design after Entrpi ds4 v0.1.1 (MIT).
     *
     * The controller reads only (commit, n_batch) — counts, never wall-clock —
     * so the quench decision is deterministic run-to-run for a fixed stream.
     * Multiseq note: this state belongs to the classic single-request flow;
     * the dormant multi-bank driver would need per-bank copies (not wired —
     * generate_speculative already refuses when mseq_dirty). */
    float spec_quench_debt;    /* cumulative plain-token-equivalents lost */
    float spec_quench_ewma;    /* EWMA of per-step margin (yield - guard) */
    uint32_t spec_quench_steps;/* fused spec steps this request */
    bool spec_quenched;        /* latched: plain decode for the remainder */
    /* Per-SESSION mirror of the engine's cumulative DSpark counters. The engine
     * copies are global (Prometheus /metrics, cross-request); these let the
     * server compute a per-RESPONSE accept-rate/tokens-per-step by snapshotting
     * at decode start and diffing at finish, which the global copies cannot give
     * because decode quanta from concurrent sessions interleave on the single
     * worker. Incremented alongside the engine counters in the fused verify
     * loop; monotonic since session open (never reset per request). */
    uint64_t spec_accepted_tokens;
    uint64_t spec_draft_tokens;
    uint64_t spec_num_drafts;
    uint64_t spec_gen_tokens;
} pulsar_spec_carry_state;

/* Tier-2 PATH A: per-bank host carry for the unified bank model.  The shared
 * pool-session's HOST per-conversation state (checkpoint token history, host
 * logits, and the whole DSpark fused-loop / spec-carry shadow) is single-
 * instance on pulsar_session, so time-slicing classic/spec work across banks in
 * one graph must save the leaving bank's carry and restore the entering bank's.
 * gpu_graph_bank_repoint swaps the DEVICE views; this covers the HOST half the
 * engine header (gpu_graph_bank_repoint contract) delegates to the caller.
 * The three heap members are owned deep copies; every other field is a scalar
 * mirror of the identically-named pulsar_session field; the speculative/DSpark
 * scalars are carried wholesale as one embedded pulsar_spec_carry_state. */
typedef struct pulsar_bank_carry {
    bool      valid;                 /* has this bank's state ever been saved */
    /* heap-backed (owned): */
    token_vec checkpoint;            /* deep copy of s->checkpoint */
    float    *logits;                /* PULSAR_N_VOCAB floats, owned */
    float    *dspark_pending_qrows;  /* dspark_pending_qrows_cap floats, owned */
    uint32_t  dspark_pending_qrows_cap;
    /* scalar mirrors: */
    bool      checkpoint_valid;
    /* Whole speculative/DSpark shadow, mirrored by value (single assignment in
     * save/restore).  NOTE: pulsar_session.mseq_dirty is deliberately NOT carried:
     * it is a property of the GRAPH's scalar frontier counters, not of a bank's
     * conversation, and _restore re-establishes per-bank frontier truth via
     * gpu_graph_bank_counters_install and then clears it unconditionally.  The
     * old mirror field was write-only (saved, never read) and has been dropped. */
    pulsar_spec_carry_state spec;
    void copy(const pulsar_bank_carry *sc);   /* was bank_carry_copy */
    void free_one();                          /* was bank_carry_free_one */
} pulsar_bank_carry;


struct pulsar_session {
    pulsar_engine *engine;
    pulsar_gpu_graph graph;
    token_vec checkpoint;
    float *logits;
    /* Reused working set for the sampled speculative acceptance walk's
     * full-vocab distribution builds (one per accepted position). Per-session,
     * never shared: concurrent sessions each run their own walk. */
    pulsar_sample_scratch sample_scratch;
    /* Reusable PULSAR_N_VOCAB-float staging row for the speculative paths'
     * spec_logits readbacks.  ~517 KB, i.e. above glibc's mmap threshold, so a
     * per-step malloc/free would mmap/munmap and re-fault it every accepted
     * position — allocate once per session instead.  Deliberately NOT drawn
     * from sample_scratch: sample_scratch_reserve frees the whole struct when
     * it grows, which would dangle a row held across a pulsar_sample_dist_build.
     * Only ever live inside one speculative eval call (the fused and block
     * paths never overlap: block tail-calls fused). */
    float *spec_row_scratch;
    pulsar_session_progress_fn progress;
    void *progress_ud;
    pulsar_session_progress_fn display_progress;
    void *display_progress_ud;
    pulsar_session_cancel_fn cancel;
    void *cancel_ud;
    uint32_t prefill_cap;
    int ctx_size;
    bool checkpoint_valid;
    /* A multiseq step has run and the graph's CLASSIC per-bank state is no
     * longer re-establishable by bookkeeping alone: the scalar frontier
     * counters (layer_n_comp / layer_n_index_comp) hold a cross-bank
     * SUPERSET, not any single bank's truth.  Any classic entry that decodes
     * against those scalars (pulsar_session_eval) would emit its compressor row
     * at the superset index and attend over the rows below it — a previous
     * tenant's bytes — producing wrong logits SILENTLY.  checkpoint_valid
     * does NOT cover this: pulsar_session_eval never reads it.  Set on every
     * decode_multiseq path that armed a step; cleared only where per-bank
     * device state is legitimately re-established (pulsar_session_sync's rebuild
     * path, via gpu_graph_reset_prefill_state zeroing the counters). */
    bool mseq_dirty;
    /* GPU bytes this session's create actually allocated (tensor-allocator
     * delta across pulsar_session_create); the server ledger commits this. */
    uint64_t resident_bytes;
    /* Live speculative/DSpark shadow; see pulsar_spec_carry_state. */
    pulsar_spec_carry_state spec;
    /* The refined-logits row each pending draft was sampled from, 16 rows of
     * PULSAR_N_VOCAB floats, allocated lazily on first sampled draft. The residual
     * needs the FULL q, but only for the single rejected position — unknown
     * until verify — so every position's row is persisted and the one that
     * rejects rebuilds its q via pulsar_sample_dist_build.
     *
     * Rebuilding from these PERSISTED logits is bit-identical to the draft-time
     * q: dist_build is a pure function of (logits, params), and the rebuild is
     * handed BOTH persisted halves — this row and dspark_pending_temp/top_k/
     * top_p/min_p below. Feeding it either half from live state is the trap: the
     * live drafter state has advanced, and the live REQUEST params may differ
     * from the draft-time ones, which would leave the stored accept denominator
     * dspark_pending_q[i] (computed under the draft-time params) and the
     * residual's q describing two different proposals inside one rule.
     *
     * Device-first note (Item 2): storing the logits ROW + params rather than a
     * materialized nucleus is deliberate — the GPU accept kernel wants exactly
     * this, so it swaps this host pool for a resident device buffer instead of
     * reshaping the format. */
    float *dspark_pending_qrows;
    uint32_t dspark_pending_qrows_cap;   /* floats reserved */
    /* Tier-2 PATH A: per-bank host carry, one entry per pool bank.  Lazily
     * allocated on the first pulsar_session_bank_state_save; NULL / bank_carry_n==0
     * when the pool is disabled (single-session use never touches it). */
    pulsar_bank_carry *bank_carry;
    uint32_t bank_carry_n;

    /* ---- methods (C++ port): 1:1 mirror of the pulsar_session_* verb family.
     * The public API in pulsar.h stays the free-function facade (defined in
     * engine_api.cpp); engine internals call these members directly.  Members
     * stay public and the struct stays trivially constructible: lifetime is
     * managed exactly as before via create()/destroy() (xcalloc/free), NOT
     * constructors/destructors.
     * NOTE: pulsar_session_prefill_cap and pulsar_session_resident_bytes stay
     * free functions — members would collide with the same-named data members.
     * pulsar_session_rewrite_requires_rebuild / _write_staged_payload /
     * _payload_file_free / _snapshot_free do not take a session and stay free. */
    static int create(pulsar_session **out, pulsar_engine *e, int ctx_size);
    void destroy();                        /* was pulsar_session_free */
    void set_progress(pulsar_session_progress_fn fn, void *ud);
    void set_display_progress(pulsar_session_progress_fn fn, void *ud);
    void set_cancel(pulsar_session_cancel_fn fn, void *ud);
    void report_progress(const char *event, int current, int total);
    void spec_metrics(pulsar_spec_metrics *out) const;
    uint64_t touched_kv_bytes() const;
    bool bank_free_physical(uint32_t bank);
    bool bank_alloc_physical(uint32_t bank);
    bool bank_is_evicted(uint32_t bank) const;
    uint64_t bank_touched_kv_bytes(uint32_t bank);
    int bank_kv_save(uint32_t bank, FILE *fp, char *err, size_t errlen);
    int bank_kv_load(uint32_t bank, FILE *fp, char *err, size_t errlen);
    uint64_t quantum_growth_bytes_per_bank(uint32_t q);
    int bank_fork(uint32_t src, uint32_t dst, const int *tokens, int n_cached);
    bool bank_fork_pinned(uint32_t bank) const;
    int bank_fork_partial(uint32_t src, uint32_t dst, const int *tokens, int n_cached);
    int sync(const pulsar_tokens *prompt, char *err, size_t errlen);
    pulsar_session_rewrite_result rewrite_from_common(const pulsar_tokens *prompt, int common,
                                                      char *err, size_t errlen);
    int common_prefix(const pulsar_tokens *prompt);
    int argmax();
    int argmax_excluding(int excluded_id);
    int sample(float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
    int top_logprobs(pulsar_token_score *out, int k);
    int token_logprob(int token, pulsar_token_score *out);
    int copy_logits(float *out, int cap);
    int set_logits(const float *logits, int n);
    int eval(int token, char *err, size_t errlen);
    int decode_multiseq(const pulsar_multiseq_req *reqs, uint32_t n,
                        float *logits, int logits_cap, char *err, size_t errlen);
    int decode_mixed(const pulsar_multiseq_req *reqs, uint32_t n_rows,
                     float *logits, int logits_cap, uint32_t *out_n_rows,
                     uint32_t max_head_runs, char *err, size_t errlen);
    void bank_carry_free();
    int bank_count();
    int bank_repoint(uint32_t bank);
    void bank_state_save(uint32_t bank);
    bool bank_state_restore(uint32_t bank);
    int bank_pos(uint32_t bank);
    const pulsar_tokens *bank_tokens(uint32_t bank);
    int bank_common_prefix(uint32_t bank, const pulsar_tokens *prompt);
    void note_committed_tokens(const int *toks, int n);
    int generate_speculative(float temperature, int top_k, float top_p, float min_p,
                             uint64_t *rng, int max_tokens, int eos_token,
                             int *accepted, int accepted_cap, char *err, size_t errlen);
    int eval_speculative_block(int first_token, int max_tokens, int eos_token,
                               int *accepted, int accepted_cap, char *err, size_t errlen);
    void invalidate();
    void rewind(int pos);
    int pos();
    int ctx();
    uint32_t prefill_quantum_min_suffix() const;
    const pulsar_tokens *tokens();
    uint64_t payload_bytes();
    int stage_payload(pulsar_session_payload_file *out, char *err, size_t errlen);
    int save_payload(FILE *fp, char *err, size_t errlen);
    int load_payload(FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
    int save_snapshot(pulsar_session_snapshot *snap, char *err, size_t errlen);
    int load_snapshot(const pulsar_session_snapshot *snap, char *err, size_t errlen);
};

typedef struct {
    uint32_t n_comp[PULSAR_MAX_LAYER];
    uint32_t n_index_comp[PULSAR_MAX_LAYER];
} pulsar_spec_frontier;

typedef struct {
    pulsar_session *session;
    const pulsar_tokens *prompt;
    pulsar_session_progress_fn user;
    void *user_ud;
} pulsar_sync_progress;

/* ---- helpers shared across the session_*.cpp TUs ----
 * payload_set_err (session_payload.cpp) is the payload/bank-KV error stamper;
 * spec_quench_reset (session_spec.cpp) re-arms the terminal yield quench at
 * request boundaries (sync/invalidate/rewind/load_payload). */
void payload_set_err(char *err, size_t errlen, const char *msg);
void spec_quench_reset(pulsar_session *s);

/* ---- shared globals ---- */

extern const pulsar_shape PULSAR_SHAPE_FLASH;
extern const pulsar_shape PULSAR_SHAPE_PRO;
extern pulsar_shape g_pulsar_shape;
extern uint32_t g_pulsar_compress_ratios[PULSAR_MAX_LAYER];
/* REAP ds4-compact-v1: per-layer count of physically-present routed experts.
 * 0 means "not set" -> falls back to n_expert (the un-pruned default). The
 * router/bias tensors stay padded to n_expert (256); only the expert weight
 * tensors are dense-trimmed to this count. Read from reap.layer.keep_count. */
extern uint32_t g_pulsar_layer_expert_count[PULSAR_MAX_LAYER];
extern int g_pulsar_lock_fd;
extern const uint64_t iq2xxs_grid[256];
extern int8_t iq2xxs_signed_grid[256][128][8];
extern int8_t iq2xxs_signs[128][8];
extern pthread_once_t iq2xxs_signed_grid_once;
extern uint32_t g_requested_threads;

/* ---- shared functions ---- */

bool pulsar_backend_uses_graph(pulsar_backend backend);
void iq2xxs_signed_grid_init(void);
void pulsar_die(const char *msg);
uint32_t pulsar_layer_compress_ratio(uint32_t il);
uint32_t pulsar_layer_n_expert(uint32_t il);
uint32_t pulsar_expected_layer_compress_ratio(uint32_t il);
void pulsar_die_errno(const char *what, const char *path);
bool pulsar_streq(pulsar_str s, const char *z);
bool pulsar_str_eq(pulsar_str a, pulsar_str b);
uint64_t hash_bytes(const void *ptr, uint64_t len);
void pulsar_alloc_guard_begin(const char *phase);
void pulsar_alloc_guard_end(void);
void *xcalloc(size_t n, size_t size);
void *xmalloc(size_t size);
char *pulsar_strdup(const char *s);
void *xrealloc(void *ptr, size_t size);
void *xmalloc_zeroed(size_t n, size_t size);
double now_sec(void);
void sleep_sec(double sec);
bool write_f32_binary_file(const char *path, const float *data, uint64_t n);
bool read_f32_binary_file(const char *path, float *data, uint64_t n);
void pulsar_threads_shutdown(void);
void pulsar_parallel_for_min_rows(uint64_t n_rows, pulsar_parallel_fn fn, void *ctx, uint64_t min_parallel_rows);
void pulsar_parallel_for(uint64_t n_rows, pulsar_parallel_fn fn, void *ctx);
void cursor_error(pulsar_cursor *c, const char *msg);
bool cursor_read(pulsar_cursor *c, void *dst, uint64_t n);
bool cursor_skip(pulsar_cursor *c, uint64_t n);
bool cursor_u32(pulsar_cursor *c, uint32_t *v);
bool cursor_u64(pulsar_cursor *c, uint64_t *v);
bool cursor_string(pulsar_cursor *c, pulsar_str *s);
uint64_t align_up(uint64_t value, uint64_t alignment);
const gguf_type_info *tensor_type(uint32_t type);
const char *tensor_type_name(uint32_t type);
void cutlass_mxfp4_expert_layout(uint64_t k, uint64_t n,
                                  uint64_t *data_bytes, uint64_t *sf_bytes,
                                  uint64_t *stride);
pulsar_cursor cursor_at(const pulsar_model *m, uint64_t pos);
bool model_get_u32(const pulsar_model *m, const char *key, uint32_t *out);
bool model_get_u64_compat(const pulsar_model *m, const char *key, uint64_t *out);
bool model_get_f32_compat(const pulsar_model *m, const char *key, float *out);
bool model_get_bool(const pulsar_model *m, const char *key, bool *out);
bool model_get_array(const pulsar_model *m, const char *key, pulsar_array_ref *out);
void model_close(pulsar_model *m);
void model_open(pulsar_model *m, const char *path, bool gpu_mapping,
                       bool prefetch_cpu);
void model_summary(const pulsar_model *m);
pulsar_tensor *model_find_tensor(const pulsar_model *m, const char *name);
bool accelerator_cache_model_tensors(pulsar_backend backend,
                                            const pulsar_model *m,
                                            const uint64_t *span_offsets,
                                            const uint64_t *span_sizes,
                                            uint32_t span_count,
                                            const char *skip_prefix);
const void *tensor_data(const pulsar_model *m, const pulsar_tensor *t);
uint32_t model_apply_expert_overlay(pulsar_model *base, const pulsar_model *overlay,
                                    const char *prefix);
bool accelerator_prepare_expert_overlay(pulsar_backend backend,
                                        const pulsar_model *base,
                                        const pulsar_model *overlay);
void model_warm_weights(const pulsar_model *m);

/* Mapping that owns a tensor's payload: the overlay file's map for
 * --expert-overlay swapped tensors, the base model's map otherwise. */
static inline const void *tensor_map_base(const pulsar_model *m, const pulsar_tensor *t) {
    return t->ext_map ? (const void *)t->ext_map : (const void *)m->map;
}
static inline uint64_t tensor_map_size(const pulsar_model *m, const pulsar_tensor *t) {
    return t->ext_map ? t->ext_size : m->size;
}
void f16_round_inplace_cpu(float *x, uint32_t n);
void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x, uint32_t head_dim, uint32_t n_rot);
void dsv4_indexer_qat_row_inplace_cpu(float *x, uint32_t head_dim);
void dsv4_indexer_qat_rows_inplace_cpu(float *x, uint32_t rows, uint32_t head_dim);
void pulsar_quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k);
void pulsar_vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y);
void pulsar_vec_dot_iq2_xxs_pair_q8_K(
        int n,
        float *s0,
        float *s1,
        const block_iq2_xxs *x0,
        const block_iq2_xxs *x1,
        const block_q8_K *y);
uint32_t required_u32(const pulsar_model *m, const char *key);
PULSAR_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const pulsar_tensor *t);
bool routed_expert_gate_down_layout(
        const pulsar_tensor *gate,
        const pulsar_tensor *down,
        uint64_t         *gate_expert_bytes,
        uint64_t         *gate_row_bytes,
        uint64_t         *down_expert_bytes,
        uint64_t         *down_row_bytes);
bool weights_have_output_head(const pulsar_weights *w);
const pulsar_layer_weights *weights_first_bound_layer(const pulsar_weights *w);
void config_validate_model(const pulsar_model *m);
void weights_bind(pulsar_weights *w, const pulsar_model *m);
void dspark_weights_bind(pulsar_dspark_weights *w, const pulsar_model *m);
void weights_free(pulsar_weights *w);
void embed_token_f16(const pulsar_model *m, const pulsar_weights *w, int token, float *out);
void rms_norm_no_weight(float *out, const float *x, uint64_t n, float eps);
void rms_norm_weight(float *out, const float *x, const float *weight, uint64_t n, float eps);
void head_rms_norm_inplace(float *x, uint32_t n_head, uint32_t head_dim, float eps);
void matvec_f16(float *out, const pulsar_model *m, const pulsar_tensor *w, const float *x);
void matvec_f16_serial(float *out, const pulsar_model *m, const pulsar_tensor *w, const float *x);
void quantize_q8_0_activation(const float *x, int8_t *xq, float *scale, uint64_t n);
void matvec_q8_0_pair_prequant(
        float           * out0,
        float           * out1,
        const pulsar_model * m,
        const pulsar_tensor * w0,
        const pulsar_tensor * w1,
        const int8_t    * xq,
        const float     * xscale);
void matmul_q8_0_batch(
        float           * out,
        const pulsar_model * m,
        const pulsar_tensor * w,
        const float     * x,
        uint64_t          n_tok);
void matmul_q8_0_pair_batch(
        float           * out0,
        float           * out1,
        const pulsar_model * m,
        const pulsar_tensor * w0,
        const pulsar_tensor * w1,
        const float     * x,
        uint64_t          n_tok);
void matvec_q8_0(float *out, const pulsar_model *m, const pulsar_tensor *w, const float *x);
void matvec_q8_0_grouped_rows(
        float           * out,
        const pulsar_model * m,
        const pulsar_tensor * w,
        const float     * x,
        uint32_t          n_groups,
        uint64_t          group_dim,
        uint64_t          rank);
void matmul_q8_0_grouped_batch(
        float           * out,
        const pulsar_model * m,
        const pulsar_tensor * w,
        const float     * x,
        uint64_t          n_tok,
        uint32_t          n_groups,
        uint64_t          group_dim,
        uint64_t          rank);
void matvec_any(float *out, const pulsar_model *m, const pulsar_tensor *w, const float *x);
float tensor_1d_value(const pulsar_model *m, const pulsar_tensor *t, uint64_t i);
float tensor_2d_value(const pulsar_model *m, const pulsar_tensor *t, uint64_t x, uint64_t y);
const uint8_t *tensor_expert_bytes(
        const pulsar_model  *m,
        const pulsar_tensor *w,
        uint32_t          expert,
        uint64_t         *in_dim,
        uint64_t         *out_dim,
        uint64_t         *row_bytes);
void matvec_iq2_xxs_expert_pair_prequant(
        float            *out0,
        float            *out1,
        const pulsar_model  *m,
        const pulsar_tensor *w0,
        const pulsar_tensor *w1,
        const block_q8_K *xq,
        uint32_t          expert);
void matvec_iq2_xxs_experts_mid_prequant(
        float            *mid,
        const pulsar_model  *m,
        const pulsar_tensor *gate_w,
        const pulsar_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp);
void matvec_q2_k_expert(
        float            *out,
        const pulsar_model  *m,
        const pulsar_tensor *w,
        const float      *x,
        uint32_t          expert);
void matvec_q2_k_experts_accum_prequant(
        float            *out,
        const pulsar_model  *m,
        const pulsar_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert);
void matvec_iq2_xxs_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1);
void quantize_mid_pairs_worker(void *vctx, uint64_t p0, uint64_t p1);
void matvec_q2_k_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1);
void matvec_experts_mid_prequant(
        float            *mid,
        const pulsar_model  *m,
        const pulsar_tensor *gate_w,
        const pulsar_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp);
void matvec_experts_down_accum_prequant(
        float            *out,
        const pulsar_model  *m,
        const pulsar_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert);
void matvec_expert_pair_prequant(
        float            *out0,
        float            *out1,
        const pulsar_model  *m,
        const pulsar_tensor *w0,
        const pulsar_tensor *w1,
        const block_q8_K *xq,
        uint32_t          expert);
void matvec_expert_down(
        float            *out,
        const pulsar_model  *m,
        const pulsar_tensor *w,
        const float      *x,
        uint32_t          expert);
void hc_split_sinkhorn_one(
        float       * out,
        const float * mix,
        const float * scale,
        const float * base,
        int           n_hc,
        int           iters,
        float         eps);
void hc_weighted_sum_one(
        float       * out,
        const float * x,
        const float * weights,
        uint32_t      n_embd,
        uint32_t      n_hc);
void hc_pre_from_state_one_scratch(
        const pulsar_model   * model,
        const pulsar_tensor  * fn,
        const pulsar_tensor  * scale_tensor,
        const pulsar_tensor  * base_tensor,
        const float       * residual_hc,
        float             * out,
        float             * post,
        float             * comb,
        float             * flat,
        bool                serial_fn);
void hc_pre_from_state_one(
        const pulsar_model   * model,
        const pulsar_tensor  * fn,
        const pulsar_tensor  * scale_tensor,
        const pulsar_tensor  * base_tensor,
        const float       * residual_hc,
        float             * out,
        float             * post,
        float             * comb);
void layer_attn_pre_one(
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * token_embd,
        float             * out,
        float             * residual_hc,
        float             * post,
        float             * comb);
void hc_from_plain_embedding(float *out_hc, const float *x, uint32_t n_embd, uint32_t n_hc);
void hc_post_one(
        float       * out_hc,
        const float * block_out,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_embd,
        uint32_t      n_hc);
void hc_post_batch(
        float       * out_hc,
        const float * block_out,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_tok,
        uint32_t      n_embd,
        uint32_t      n_hc);
void hc_post_sum_batch(
        float       * out_hc,
        const float * moe,
        const float * shared,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_tok,
        uint32_t      n_embd,
        uint32_t      n_hc);
void hc_pre_norm_batch(
        const pulsar_model  * model,
        const pulsar_tensor * fn,
        const pulsar_tensor * scale,
        const pulsar_tensor * base,
        const pulsar_tensor * norm_w,
        const float      * inp_hc,
        float            * residual_hc,
        float            * cur,
        float            * norm,
        float            * post,
        float            * comb,
        uint32_t           n_tok);
void layer_attn_norm_one(
        float             * out,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * x);
void layer_q_projection_normed_one(
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * norm,
        float             * q);
void layer_q_projection_with_lora_one(
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * norm,
        float             * q,
        float             * qr_norm);
void layer_kv_projection_normed_one(
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * normed,
        float             * kv);
float layer_rope_freq_base(uint32_t il);
float layer_rope_freq_scale(uint32_t il);
void rope_tail_layer_inplace(
        float            * x,
        uint32_t           n_head,
        uint32_t           head_dim,
        uint32_t           n_rot,
        uint32_t           pos,
        uint32_t           il,
        bool               inverse);
void rope_tail_layer_batch_inplace(
        float            *x,
        uint64_t          stride,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          il,
        bool              inverse,
        uint32_t          n_tok);
float sigmoid_stable(float x);
void layer_attention_rows_one(
        float             * out_heads,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * q,
        const float       * kv_rows,
        uint32_t            n_kv);
void layer_attention_one(
        float             * out_heads,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * q,
        const float       * kv);
void layer_grouped_out_one(
        float             * out,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * heads);
void layer_grouped_out_batch(
        float             * out,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * heads,
        uint32_t            n_tok);
float silu(float x);
float softplus_stable(float x);
void swiglu(float *out, const float *gate, const float *up, uint64_t n, float clamp);
void layer_shared_ffn_one(
        float             * out,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * x);
void layer_hash_selected_experts(
        int                    selected[PULSAR_MAX_EXPERT_USED],
        const pulsar_model       *model,
        const pulsar_layer_weights *layer,
        int                    token);
void layer_hash_router_weights_from_probs(
        float             weights_out[PULSAR_MAX_EXPERT_USED],
        const float       probs[PULSAR_MAX_EXPERT],
        const int          selected[PULSAR_MAX_EXPERT_USED]);
void layer_hash_router_weights_one(
        float             weights_out[PULSAR_MAX_EXPERT_USED],
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * x,
        const int          selected[PULSAR_MAX_EXPERT_USED]);
void layer_topk_selected_experts(
        int                    selected[PULSAR_MAX_EXPERT_USED],
        float                  expert_weight[PULSAR_MAX_EXPERT_USED],
        const pulsar_model       *model,
        const pulsar_layer_weights *layer,
        const float           *x);
void layer_topk_selected_experts_from_probs(
        int                    selected[PULSAR_MAX_EXPERT_USED],
        float                  expert_weight[PULSAR_MAX_EXPERT_USED],
        const pulsar_model       *model,
        const pulsar_layer_weights *layer,
        const float           probs[PULSAR_MAX_EXPERT]);
void layer_routed_moe_one_prealloc(
        float             * out,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * x,
        uint32_t            il,
        int                 token,
        float               clamp,
        float              * mid_all,
        block_q8_K         * xq,
        block_q8_K         * midq);
void layer_ffn_one(
        float             * out_hc,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * inp_hc,
        uint32_t            il,
        int                 token,
        const float       * steering_dirs,
        float               steering_scale,
        bool                trace);
void layer_ffn_batch(
        float             * out_hc,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale);
void layer_ffn_shared_batch(
        float             * out_hc,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale);
void layer_ffn_tokens_parallel(
        float             * out_hc,
        const pulsar_model   * model,
        const pulsar_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale);
uint32_t pulsar_default_raw_cap(uint32_t ctx_size);
uint32_t pulsar_prefill_cap_for_prompt(int prompt_len,
                                           uint32_t requested_chunk);
void layer_forward_self_one(
        float                   * out_hc,
        const pulsar_model         * model,
        const pulsar_layer_weights * layer,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        int                       token);
void output_logits_one(
        float             * logits,
        const pulsar_model   * model,
        const pulsar_weights * weights,
        const float       * inp_hc);
float max_abs_diff(const float *a, const float *b, uint64_t n);
float rms_abs_diff(const float *a, const float *b, uint64_t n);
uint64_t argmax_f32(const float *x, uint64_t n);
void print_vec_stats(const char *name, const float *x, uint64_t n);
void gpu_graph_free(pulsar_gpu_graph *g);
bool gpu_tensor_fill_f32(pulsar_gpu_tensor *t, float v, uint64_t n);
bool gpu_graph_load_directional_steering(
        pulsar_gpu_graph *g,
        const char      *path,
        float            attn_scale,
        float            ffn_scale);
bool gpu_graph_directional_steering_attn_enabled(const pulsar_gpu_graph *g);
bool gpu_graph_directional_steering_ffn_enabled(const pulsar_gpu_graph *g);
bool gpu_graph_apply_directional_steering_attn(
        pulsar_gpu_graph  *g,
        pulsar_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows);
bool gpu_graph_apply_directional_steering_ffn(
        pulsar_gpu_graph  *g,
        pulsar_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows);
uint64_t gpu_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out);
pulsar_gpu_tensor *gpu_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes);
/* True when PULSAR_CUDA_GRAPH_DUMP_PREFIX is set (cached). Graph allocation
 * uses this to skip buffers that exist only to be dumped. */
bool gpu_graph_debug_dump_enabled(void);
bool gpu_graph_debug_wants(const char *name, uint32_t il, uint32_t pos);
void gpu_graph_debug_dump_hc_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_elems,
        uint32_t          il,
        uint32_t          pos);
void gpu_graph_debug_dump_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos);
void gpu_graph_debug_dump_f16_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_f16,
        uint32_t          il,
        uint32_t          pos);
void gpu_graph_debug_dump_i32_tensor(
        const char       *name,
        pulsar_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos);
bool gpu_graph_needs_ffn_out(const pulsar_gpu_graph *g, uint32_t il, uint32_t pos);
bool gpu_graph_ensure_ffn_out(pulsar_gpu_graph *g);
bool gpu_graph_ensure_batch_ffn_out(pulsar_gpu_graph *g);
bool gpu_graph_alloc_raw_cap(
        pulsar_gpu_graph *g,
        const pulsar_weights     *weights,
        const pulsar_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_spec);
/* Bank-pool size the next gpu_graph_alloc_raw_cap will use (PULSAR_MSEQ_BANKS,
 * read once, clamped to [1, PULSAR_MSEQ_MAX]; 1 = pool disabled).  Interim
 * wiring: later increments make the server pass the pool size explicitly. */
uint32_t gpu_graph_bank_pool_n(void);
/* Re-install the graph's per-layer cache views onto `bank` (pool mode only).
 * Contract: call only between fully synchronized forwards — the previous
 * bank's enqueued work must be complete, because the graph pointers change
 * under every subsequent launch.  This swaps DEVICE views only: the host
 * per-session state (layer_n_comp/layer_n_index_comp, ring fill, positions,
 * spec-shadow contents) is the caller's to save/restore per bank.  On
 * failure the views may be mixed-bank — treat the graph as dead. */
bool gpu_graph_bank_repoint(pulsar_gpu_graph *g, uint32_t bank);
/* Effective pool size for banked kernel launches: banks.n_banks, or 1 when
 * the pool is disabled (the classic tensors act as bank 0). */
uint32_t gpu_graph_bank_pool_count(const pulsar_gpu_graph *g);
/* Tier-2 overcommit (task #55): demand-paged comp+index VA bytes for ONE bank at
 * a context (the overcommit-reserved, physical-on-touch part); and the EXACT
 * touched (physically resident) demand-paged KV summed over the whole pool from
 * the per-bank compressor frontier. See the definitions in gpu_diag.c. */
uint64_t gpu_graph_demand_paged_bytes_per_bank(uint32_t ctx_size);
uint64_t gpu_graph_touched_kv_bytes(const pulsar_gpu_graph *g);
uint64_t gpu_graph_bank_touched_kv_bytes(const pulsar_gpu_graph *g, uint32_t bank);
uint64_t gpu_graph_quantum_growth_bytes_per_bank(uint32_t q);
/* Tier-2 task #55 increment 2b — per-bank physical evict/restore reclaim
 * primitives (direct cudaFree / cudaMallocManaged of one bank's split comp/index
 * + base-table rebuild). See gpu_diag.c. */
bool gpu_graph_bank_free_physical(pulsar_gpu_graph *g, uint32_t bank);
bool gpu_graph_bank_alloc_physical(pulsar_gpu_graph *g, uint32_t bank);
bool gpu_graph_bank_is_evicted(const pulsar_gpu_graph *g, uint32_t bank);
/* Tier-2 PATH-A full-prefix fork (plan-33 inc A): D2D clone src bank's committed
 * KV into dst + mirror frontier counters. Caller validates + pins src first. */
bool gpu_graph_bank_fork_copy(pulsar_gpu_graph *g, uint32_t src, uint32_t dst);
/* plan-33 inc C: partial-cut fork + boundary machinery (gpu_diag.c). */
uint32_t pulsar_partial_fork_base_align(void);
bool gpu_graph_bank_fork_copy_cut(pulsar_gpu_graph *g, uint32_t src, uint32_t dst,
                                  uint32_t R, uint32_t src_len);
bool gpu_graph_emit_keep_restore(pulsar_gpu_graph *g, uint32_t il, uint32_t bank,
                                 uint32_t row0, uint32_t rows, bool indexer);
/* Whole-pool cache tensors for banked kernel operands: the bank slab when
 * the pool is enabled, else the classic single-session tensor (== bank 0).
 * NULL for layers without that cache kind. */
pulsar_gpu_tensor *gpu_graph_bank_raw_pool(pulsar_gpu_graph *g, uint32_t il);
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_pool(pulsar_gpu_graph *g, uint32_t il);
pulsar_gpu_tensor *gpu_graph_bank_index_comp_pool(pulsar_gpu_graph *g, uint32_t il);
/* Per-bank comp/index base-pointer tables (device arrays of n_banks pointers,
 * indexed by seq_id) the batched READ kernels use in place of base +
 * seq_id*comp_cap over one slab. NULL when the pool is disabled. */
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_bases(pulsar_gpu_graph *g, uint32_t il);
pulsar_gpu_tensor *gpu_graph_bank_index_comp_bases(pulsar_gpu_graph *g, uint32_t il);
/* Fresh single-bank views for the batched emit path (caller frees; when the
 * pool is disabled, bank must be 0 and the view wraps the classic tensor).
 * kind: the per-(bank,layer) comp caches and compressor state lanes. */
pulsar_gpu_tensor *gpu_graph_bank_attn_comp_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_index_comp_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_attn_state_kv_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_attn_state_score_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_index_state_kv_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
pulsar_gpu_tensor *gpu_graph_bank_index_state_score_view(pulsar_gpu_graph *g, uint32_t il, uint32_t bank);
/* Host counter hand-off between classic single-session work (scalar
 * layer_n_comp/layer_n_index_comp) and the per-bank ms counters.  Capture
 * after classic per-bank work (admission prefill, replay) so the ms arrays
 * reflect that bank's committed frontier; install before classic per-bank
 * work so the scalars are that bank's counts again. */
void gpu_graph_bank_counters_capture(pulsar_gpu_graph *g, uint32_t bank);
void gpu_graph_bank_counters_install(pulsar_gpu_graph *g, uint32_t bank);
/* Tier-2 PATH A host-carry primitive (see pulsar_bank_carry).  save copies the
 * session's live HOST per-conversation state into bank's shadow AND captures
 * the graph frontier counters (gpu_graph_bank_counters_capture).  restore
 * repoints the device views to bank, installs its frontier counters, copies
 * the shadow back into the session, and clears mseq_dirty (the cheap
 * no-re-prefill resume: counters_install re-establishes per-bank truth, which
 * is exactly what mseq_dirty guards).  Call save on the leaving bank before,
 * and restore on the entering bank after, a bank switch — both only between
 * fully synchronized forwards (the gpu_graph_bank_repoint contract).  restore
 * returns false only on a bad bank id or OOM growing an owned buffer. */
/* pulsar_session_bank_state_save/restore + pulsar_session_bank_count/repoint are the
 * public server-facing API (declared in pulsar.h). */
/* pulsar_session::bank_carry_free() is the member form of the old
 * pulsar_session_bank_carry_free free function (internal-only, no facade). */
/* Arm one banked multiseq batched step over n_rows packed rows: pos[t] is
 * row t's absolute position, seq[t] its TRUE bank id.  Writes the host
 * mirrors + device descriptor arrays (lazily allocated), verifies the
 * DRIVER CONTRACT (each batched bank's ms frontier is position-true —
 * ms_n_comp == first_pos/ratio — i.e. no mid-prefill bank is co-scheduled),
 * and refreshes the scalar superset counters ONCE (the step's emit-inclusive
 * bound, max over rows of (pos+1)/ratio).  capture_cur first captures the
 * current bank's scalars into its ms row (single-session diagnostic use).
 * Constraint (fail-loud): each bank's rows form ONE contiguous run with
 * consecutive positions inside the run (pos ascending by 1), and every run
 * starts at a position > 0 (position-0 rows rejected — admission prefill is
 * classic single-bank in v1).  Banks may sit at unrelated positions: every
 * upstream batch stage is per-row-position driven (RoPE variants take the
 * batch_positions device array; NULL degenerates to pos0+t).  Every rejection
 * prints the reason.  Disarm + self-check with gpu_graph_multiseq_step_end
 * after the layer sweep (it validates every batched bank's frontier advanced
 * to its position-derived value and the superset equals max over banks). */
bool gpu_graph_multiseq_step_begin(pulsar_gpu_graph *g, const int32_t *pos,
                                   const int32_t *seq, uint32_t n_rows,
                                   bool capture_cur);
bool gpu_graph_multiseq_step_end(pulsar_gpu_graph *g);
/* Tier-2 batched multi-session decode: one token per live bank through ONE
 * weight sweep (see the definition comment in imatrix.c for the full driver
 * contract).  logits out = [n_active * PULSAR_N_VOCAB], row k = bank[k].
 * Returns 1 ok / 0 recoverable rejection (nothing mutated) / -1 fatal (armed
 * sweep or head failed — session state untrusted). */
int gpu_graph_decode_multiseq_batch(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const int             *tokens,
        const int32_t         *pos,
        const int32_t         *bank,
        uint32_t               n_active,
        float                 *logits,
        uint32_t              *out_n_rows,
        uint32_t               max_head_runs);
/* PULSAR_DECODE_DESCR=1 (env, read once): Tier-2 descriptor-vs-classic byte
 * diagnostic — see gpu_decode.c. */
int gpu_graph_decode_descr_enabled(void);
/* TRUE per-session GPU byte cost of gpu_graph_alloc_raw_cap (+ the DSpark
 * graph state when enable_spec); the sizing side of the admission-control
 * single source of truth (see gpu_diag.c).  Includes the whole bank pool
 * when PULSAR_MSEQ_BANKS >= 2 (same knob the allocator reads). */
uint64_t gpu_graph_session_bytes(
        const pulsar_weights       *weights,
        const pulsar_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap,
        bool                     enable_spec);
/* Same, but priced for an EXPLICIT bank-pool size instead of reading
 * PULSAR_MSEQ_BANKS — the fit-table / auto-sizing path (cli_main) evaluates many
 * (n_banks, ctx) candidates before the env is committed. n_banks == 0 or 1 is
 * the classic single-session layout. */
uint64_t gpu_graph_session_bytes_banked(
        const pulsar_weights       *weights,
        const pulsar_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap,
        bool                     enable_spec,
        uint32_t                 n_banks);
bool gpu_graph_init_dspark_target(pulsar_gpu_graph *g, const uint32_t target_layer_ids[3]);
bool gpu_graph_alloc(
        pulsar_gpu_graph *g,
        const pulsar_weights     *weights,
        const pulsar_layer_weights *layer);
uint32_t gpu_graph_raw_span_for_batch(
        const pulsar_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens);
uint32_t gpu_graph_raw_start_for_span(
        const pulsar_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw);
uint32_t gpu_graph_decode_indexer_sparse_threshold(const pulsar_gpu_graph *g);
bool gpu_graph_env_flag(const char *name, int *cache);
bool gpu_graph_use_reference_hc_decode(void);
bool gpu_graph_use_reference_qkv_norm(void);
bool gpu_graph_enable_batch_hc_norm_fusion(void);
uint32_t gpu_graph_attn_comp_cache_is_pack(void);
int gpu_graph_attn_pack_enabled(void);
uint32_t gpu_graph_prefill_slice(void);
/* True when PULSAR_IDX_FP4 is set (cached). When on, the ratio-4 indexer
 * compressed cache is stored MXKV-FP4-packed (PULSAR_ENGINE_IDXFP4_ROWBYTES/row,
 * 7.5x smaller than f32) and the indexer score kernels read it packed.  The
 * cache rows are QAT-roundtripped to exactly these fp4 values in both modes,
 * so scores and outputs are bit-identical; only storage and traffic change. */
int gpu_graph_idx_fp4_enabled(void);
/* True when PULSAR_RAW_F16 is set (cached). When on, the raw KV ring stores
 * __half rows (half the bytes) instead of f32 containers.  The stored values
 * are ALREADY f16-rounded in both modes (the store kernels roundtrip through
 * __float2half, and the fp8-roundtripped nope dims are exactly
 * f16-representable), so reads are bit-identical; only storage changes. */
int gpu_graph_raw_f16_enabled(void);
/* Comp-cache row stride in bytes for the active storage format (pack-aware). */
uint64_t gpu_graph_attn_comp_cache_row_bytes(void);
/* Returns the comp-cache tensor to hand to the f32 prefill attention consumers
 * for `n_rows` rows: the cache itself normally, or the f32 shadow
 * (attn_comp_dequant) after a bit-exact dequant when PULSAR_ATTN_PACK storage is
 * on.  NULL on error. */
pulsar_gpu_tensor *gpu_graph_attn_comp_read_cache(
        pulsar_gpu_graph *g,
        uint32_t       il,
        uint32_t       n_rows);
pulsar_gpu_tensor *gpu_graph_attn_comp_update_target(
        pulsar_gpu_graph *g,
        uint32_t       il);
uint32_t gpu_graph_attn_comp_update_row(uint32_t row);
bool gpu_graph_commit_attn_comp_stage(
        pulsar_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows);
/* Bank-aware commit for the batched multiseq emit path: quantize+pack the
 * staged f32 rows into BANK's comp cache at bank-local first_row.  Equals
 * the classic commit when the pool is disabled (bank must be 0). */
bool gpu_graph_commit_attn_comp_stage_bank(
        pulsar_gpu_graph *g,
        uint32_t       il,
        uint32_t       bank,
        uint32_t       first_row,
        uint32_t       rows);
pulsar_gpu_tensor *gpu_graph_attn_comp_row_view(
        pulsar_gpu_graph *g,
        uint32_t       il,
        uint32_t       row);
pulsar_gpu_tensor *gpu_graph_attn_comp_prefill_target(
        pulsar_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows);
void gpu_graph_attn_comp_prefill_target_free(pulsar_gpu_tensor *t);
bool gpu_graph_encode_decode_layer(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        pulsar_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token);
void gpu_graph_capture_dspark_target_hc(pulsar_gpu_graph *g, uint32_t il);
bool gpu_graph_encode_output_head(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint64_t               vocab_dim);
bool gpu_graph_encode_output_head_batch(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim);
bool gpu_graph_encode_dspark_output_head_batch(
        pulsar_gpu_graph            *g,
        const pulsar_model          *dspark_model,
        const pulsar_dspark_weights *dw,
        const pulsar_model          *base_model,
        const pulsar_weights        *bw,
        uint32_t                  n_tokens,
        uint64_t                  vocab_dim);
bool gpu_graph_dspark_project_main_x(
        pulsar_gpu_graph          *g,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w);
void gpu_graph_dspark_seed_draft_kv(
        pulsar_gpu_graph          *g,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w,
        uint32_t                 n_rows);
bool gpu_graph_dspark_draft_forward(
        pulsar_gpu_graph          *g,
        const pulsar_model         *base_model,
        const pulsar_weights       *base_weights,
        const pulsar_model         *dspark_model,
        const pulsar_dspark_weights *w,
        pulsar_gpu_tensor         *base_logits_out,
        const int32_t            draft_ids[],
        uint32_t                n_draft);
bool gpu_graph_matmul_plain_tensor(
        pulsar_gpu_tensor       *out,
        const pulsar_model        *model,
        const pulsar_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok);
bool gpu_graph_matmul_mxfp8_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        pulsar_gpu_tensor       *out,
        const pulsar_model        *model,
        const pulsar_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const pulsar_gpu_tensor *x,
        uint64_t                n_tok);
int gpu_graph_decode_test(
        const pulsar_model   *model,
        const pulsar_weights *weights,
        const token_vec   *prompt,
        bool               quality);
uint32_t gpu_graph_token_split_after_layers(void);
pulsar_gpu_tensor *gpu_graph_tensor_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values);
pulsar_gpu_tensor *gpu_graph_hc_row_view(
        pulsar_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values);
/* Read an HC residual carrier (BF16 storage; task #62) into an f32 host buffer,
 * expanding each sample. Dev-only (parity self-test + env-gated DSpark dumps). */
int pulsar_read_hc_carrier_f32(const pulsar_gpu_tensor *t, uint64_t off_elems,
                            float *out, uint64_t n);
/* f32 -> HC carrier bytes (RNE, matches the GPU __float2bfloat16 store). */
void pulsar_store_hc_carrier_f32(void *dst, const float *src, uint64_t n);
bool gpu_graph_upload_prompt_tokens(
        pulsar_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens);
bool gpu_graph_upload_prompt_embeddings_hc(
        pulsar_gpu_tensor   *out_hc,
        pulsar_gpu_tensor   *tokens,
        const pulsar_model    *model,
        const pulsar_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens);
bool gpu_graph_warmup_prefill_kernels(
        pulsar_gpu_graph   *g,
        const pulsar_model   *model,
        const pulsar_weights *weights,
        uint32_t           n_tokens);
bool gpu_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);
bool gpu_graph_decode_stage_profile_enabled(uint32_t il);
bool gpu_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);
bool gpu_graph_encode_layer_attention_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens);
bool gpu_graph_encode_layer_ffn_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens);
bool gpu_graph_encode_layer_batch(
        pulsar_gpu_graph  *g,
        const pulsar_model        *model,
        const pulsar_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens);
bool gpu_graph_eval_token_raw_swa(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits);
bool gpu_graph_eval_token_raw_swa_top(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        int                    token,
        uint32_t               pos,
        int                   *top_id,
        float                 *logits);
bool gpu_graph_dspark_compressor_rollforward(
        pulsar_gpu_graph  *g,
        const pulsar_model  *model,
        const pulsar_weights *weights,
        uint32_t          pos0,
        uint32_t          n_positions);
bool imatrix_collector_init(pulsar_imatrix_collector *c, uint32_t cap_tokens, const char *dataset_path);
void imatrix_collector_free(pulsar_imatrix_collector *c);
bool imatrix_collector_save(
        const pulsar_imatrix_collector *c,
        const pulsar_weights           *weights,
        const char                  *path);
bool gpu_graph_reset_prefill_state(pulsar_gpu_graph *g);
bool gpu_graph_prefill_layer_major(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_imatrix_collector *imatrix,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud);
bool gpu_graph_prefill_raw_swa(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud,
        pulsar_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled);
bool gpu_graph_prefill_chunked_range(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_session_progress_fn progress,
        void                  *progress_ud,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud,
        pulsar_imatrix_collector *imatrix,
        pulsar_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled);
bool gpu_graph_prefill_chunked(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        pulsar_session_progress_fn progress,
        void                  *progress_ud,
        pulsar_session_progress_fn display_progress,
        void                  *display_progress_ud,
        pulsar_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled);
bool gpu_graph_verify_suffix_tops(
        pulsar_gpu_graph *g,
        const pulsar_model       *model,
        const pulsar_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        int                   *row_tops,
        float                 *row_logits);
bool gpu_graph_read_spec_logits_row(pulsar_gpu_graph *g, uint32_t row, float *logits);
uint32_t gpu_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap);
uint32_t gpu_graph_prefill_cap_for_prompt(int prompt_len,
                                                   uint32_t prefill_chunk);
uint32_t gpu_graph_resume_prefill_min_tokens(void);
void embed_prompt(
        const pulsar_model   * model,
        const pulsar_weights * weights,
        const token_vec   * tokens,
        uint32_t            n_embd,
        float             * out);
void token_vec_push(token_vec *tv, int token);
void token_vec_free(token_vec *tv);
bool cpu_directional_steering_enabled(
        const float *dirs,
        float        scale);
void cpu_directional_steering_project_rows(
        float       *x,
        const float *dirs,
        uint32_t     il,
        uint32_t     rows,
        float        scale);
void dump_tokens_fp(FILE *fp, const pulsar_vocab *vocab, const token_vec *tokens);
int sample_argmax(const float *logits, uint32_t n_vocab);
typedef struct {
    int *ids;
    float *probs;   /* renormalized over the filtered nucleus */
    uint32_t n;
} pulsar_sample_dist;

/* `scratch` is required (non-NULL) and must outlive nothing: it is pure
 * working memory, reusable across calls and independent of `out`. */
int pulsar_sample_dist_build(const float *logits, uint32_t n_vocab,
                          float temperature, int top_k, float top_p, float min_p,
                          pulsar_sample_scratch *scratch, pulsar_sample_dist *out);
void pulsar_sample_dist_free(pulsar_sample_dist *d);
float pulsar_sample_dist_prob(const pulsar_sample_dist *d, int token);
int pulsar_sample_dist_accept(const pulsar_sample_dist *d, int token, uint64_t *rng);
int pulsar_sample_dist_draw(const pulsar_sample_dist *d, uint64_t *rng);
int pulsar_sample_dist_draw_excluding(const pulsar_sample_dist *d, int excluded, uint64_t *rng);
/* Sampled-proposal speculative rule (the deterministic-proposal pair above is
 * pulsar_sample_dist_accept / _draw_excluding). `token` was drawn from a proposal
 * q; `q` is q(token). Accepts with probability min(1, p(token)/q(token)).
 * Never accepts a token with p(token) <= 0. Consumes no rng when the outcome
 * is certain (p >= q), matching pulsar_sample_dist_accept's p >= 1 fast path —
 * which is what keeps the temperature<=0 path byte-identical. */
int pulsar_sample_dist_accept_pq(const pulsar_sample_dist *p, int token, float q, uint64_t *rng);
/* The matching residual: draw from (p-q)+ normalized. Every token it can
 * return has p(token) > 0 AND strictly positive residual mass; if the total
 * residual mass is <= 0 it falls back to a plain draw from p. `scratch` is
 * working memory (see pulsar_sample_scratch::qmap); it must not alias p or q. */
int pulsar_sample_dist_draw_residual(const pulsar_sample_dist *p, const pulsar_sample_dist *q,
                                  pulsar_sample_scratch *scratch, uint64_t *rng);

int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng);
int generate_gpu_graph_raw_swa(
        const pulsar_model   * model,
        const pulsar_vocab   * vocab,
        const pulsar_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        bool                quality,
        uint32_t            prefill_chunk,
        const char        * directional_steering_file,
        float               directional_steering_attn,
        float               directional_steering_ffn,
        pulsar_token_emit_fn   emit,
        pulsar_generation_done_fn done,
        void              * emit_ud,
        pulsar_session_progress_fn progress,
        void              * progress_ud);
void pulsar_linux_graph_backend_set_oom_score(pulsar_backend backend);
void pulsar_release_instance_lock(void);
void pulsar_acquire_instance_lock(void);

/* ---- shared inline helpers ---- */

static inline PULSAR_MAYBE_UNUSED int32_t dot_iq2_pair_16(const int8_t *grid0, const int8_t *grid1, const int8_t *q8) {
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

static inline PULSAR_MAYBE_UNUSED int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
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
 * Scalar Conversion and Quantized Tensor Kernels.
 * =========================================================================
 *
 * These functions are the CPU reference math used by the C backend and by
 * GPU diagnostics.  They implement only the tensor formats present in the
 * DeepSeek V4 Flash GGUF: F16, F32, Q2_K, IQ2_XXS, and Q8_K activation
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


#endif /* PULSAR_ENGINE_INTERNAL_H */
