#ifndef DS4_FLINT_GPU_H
#define DS4_FLINT_GPU_H

/*
 * GPU entry points for the Recursive Flint family (fork-owned).
 *
 * The decode step is one persistent Metal kernel (metal/flint.metal, ported
 * from rec).  ds4_metal_flint.m compiles that file into its own library with
 * strict IEEE math, wraps the mapped GGUF as the kernel's single weight
 * buffer, and owns the per-session KV blocks and scratch.  The C side
 * (ds4_flint.inc) builds the 50 x 64 descriptor table from the bound tensors
 * and the expert pack offsets, and drives sessions.
 *
 * All sizes are rec's: 49 layers, 150144 logits, a 50 x 2048 bf16 residual
 * trace (embedding output and every layer's output).
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_FLINT_GPU_N_LAYER      49u
#define DS4_FLINT_GPU_N_EXPERT     256u
#define DS4_FLINT_GPU_N_VOCAB      150144u
#define DS4_FLINT_GPU_N_EMBD       2048u
#define DS4_FLINT_GPU_TRACE_ELEMS  (50u * 2048u)
#define DS4_FLINT_GPU_META_WORDS   (50u * 64u)
#define DS4_FLINT_GPU_SILU_ENTRIES 65536u
#define DS4_FLINT_GPU_SWA          513u
#define DS4_FLINT_GPU_BLOCK_TOKENS 65536u
#define DS4_FLINT_GPU_MAX_BLOCKS   4u
#define DS4_FLINT_GPU_MAX_CTX      (DS4_FLINT_GPU_BLOCK_TOKENS * DS4_FLINT_GPU_MAX_BLOCKS)

/* The device name rec's profile table knows ("M4 Pro", ...), or NULL when
 * this Mac has no profile; sets *why on failure. */
const char *ds4_flint_gpu_device_profile(const char **why);

/* Compiles the kernel, wraps the model map, uploads the descriptor table
 * (50 x 64 words; KV words 61/62 are filled per session), the per-layer
 * expert pack byte offsets into the map (49 x 256; zero for dense layers)
 * and the silu table.  Returns 1 on success. */
int ds4_flint_gpu_engine_open(const void *model_map, uint64_t model_size,
                              const uint32_t *metadata,
                              const uint64_t *expert_pack_offsets,
                              const uint16_t *silu_bf16, const char **why);
void ds4_flint_gpu_engine_release(void);

typedef struct ds4_flint_gpu_session ds4_flint_gpu_session;

/* Bytes a session with this context capacity allocates (KV blocks + scratch). */
uint64_t ds4_flint_gpu_session_bytes(uint32_t ctx_cap);
ds4_flint_gpu_session *ds4_flint_gpu_session_create(uint32_t ctx_cap);
void ds4_flint_gpu_session_free(ds4_flint_gpu_session *s);
uint32_t ds4_flint_gpu_session_ctx_cap(const ds4_flint_gpu_session *s);

/* KV rows of one layer and head: n_rows consecutive slots from row0, each
 * 128 bf16.  Global layers index slots by position; sliding layers by ring
 * slot (0..512).  key selects K (1) or V (0). */
int ds4_flint_gpu_kv_write(ds4_flint_gpu_session *s, uint32_t layer, int key, uint32_t head,
                           uint32_t row0, uint32_t n_rows, const uint16_t *rows);
int ds4_flint_gpu_kv_read(const ds4_flint_gpu_session *s, uint32_t layer, int key, uint32_t head,
                          uint32_t row0, uint32_t n_rows, uint16_t *rows);

/* One decode step: the token at `position` with the KV of positions below
 * it already written; `initial_position` is the prefill length (it fixes the
 * sliding ring's slot mapping, as in rec).  Writes 150144 bf16 logits and,
 * when trace is not NULL, the 50 x 2048 bf16 trace.  Returns 1 on success;
 * on a GPU synchronization failure the session is unusable and err says
 * which group timed out in which phase. */
int ds4_flint_gpu_decode(ds4_flint_gpu_session *s, uint32_t token, uint32_t position,
                         uint32_t initial_position, uint16_t *logits_bf16, uint16_t *trace_bf16,
                         char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
