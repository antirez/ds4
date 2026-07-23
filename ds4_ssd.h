#ifndef DS4_SSD_H
#define DS4_SSD_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    void *ptr;
    uint64_t bytes;
} ds4_ssd_memory_lock;

typedef struct {
    uint64_t model_target_bytes;
    uint64_t cache_bytes;
    uint64_t effective_cache_bytes;
    uint32_t cache_experts;
} ds4_ssd_cache_plan;

/*
 * Host-I/O policy for data that can either use the OS page cache or bypass it
 * (F_NOCACHE on macOS).  recommended_bytes may be zero when the platform does
 * not expose a recommended working-set limit.  host_bytes and hot_bytes must
 * be known and non-zero; UINT64_MAX is reserved as the caller's
 * unknown/overflow sentinel.  Invalid inputs always produce a conservative
 * direct-I/O plan.
 */
typedef struct {
    uint64_t hot_bytes;
    uint64_t budget_bytes;
    uint64_t reserve_bytes;
    bool     inputs_valid;
    bool     use_direct_io;
} ds4_ssd_io_plan;

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes);
bool ds4_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes);

ds4_ssd_io_plan ds4_ssd_io_plan_compute(uint64_t host_bytes,
                                        uint64_t recommended_bytes,
                                        uint64_t hot_bytes);

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes);
bool ds4_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             ds4_ssd_cache_plan *out);

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock,
                                 uint64_t             bytes);
void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock);

/*
 * SSD streaming expert-bundle sidecar ("DSEB" version 1).
 *
 * The sidecar is a one-time repack of the routed experts: each expert's
 * gate|up|down slabs are stored contiguously (4 KiB aligned records, ordered
 * by layer then expert id), so a streaming-cache miss becomes one sequential
 * burst instead of three reads scattered across the GGUF. Same bytes, same
 * numerics: only the on-disk layout changes. The sidecar duplicates the
 * expert region of the model on disk. Metal SSD streaming enables it by
 * default, skips creation when free space is short, and accepts
 * DS4_EXPERT_BUNDLE=0 as a diagnostic rollback.
 *
 * The record layout is shared with the DwarfStar Swift port. C readers accept
 * both canonical and original Swift v1 fingerprints:
 *
 *   u32 magic "DSEB"    u32 version=1    u64 model_size
 *   u32 layer_lo        u32 layer_count  u32 n_expert     u32 pad=0
 *   u64 gate_bytes      u64 up_bytes     u64 down_bytes
 *   u64 fnv1a[layer_count]   (first 4 KiB of each layer's gate tensor)
 *   ...optional DSEI identity/full-tensor-hash extension, then pad to 4 KiB...
 *   records: layer-major, expert-minor, gate|up|down zero-padded to the
 *   record stride (gate+up+down rounded up to 4 KiB).
 *
 * Writers emit standard FNV-1a. Readers also accept the fingerprint table
 * produced by the original Swift v1 writer's incorrect FNV multiplier, so
 * existing multi-gigabyte sidecars do not need to be rebuilt. Original Swift
 * readers ignore the aligned header padding, so DSEI preserves version 1,
 * data_base, and every record offset. When the extension fits, a legacy file
 * is compared byte-for-byte with the GGUF once, then upgraded in that padding
 * best-effort; matching identity avoids later full payload scans. Very large
 * layer tables that fill the page retain the exact legacy v1 behavior.
 */
typedef struct {
    uint64_t gate_offset;   /* absolute model-file offsets of the layer's   */
    uint64_t up_offset;     /* ffn_{gate,up,down}_exps tensors              */
    uint64_t down_offset;
} ds4_expert_bundle_layer;

typedef struct {
    int      fd;            /* -1 when the bundle is not open               */
    uint32_t layer_lo;      /* first bundled layer (absolute index)         */
    uint32_t layer_count;
    uint32_t n_expert;
    uint64_t gate_bytes;    /* per-expert slab bytes                        */
    uint64_t up_bytes;
    uint64_t down_bytes;
    uint64_t data_base;     /* aligned file offset of the first record      */
    uint64_t record;        /* aligned gate+up+down record stride           */
    char     path[1024];    /* file actually opened, for the startup log    */
} ds4_expert_bundle;

/*
 * Open a valid sidecar for this model, or build it (one-time: it copies the
 * whole expert region next to the model, or into $DS4_BUNDLE_DIR when set).
 * Returns false on setup or validation failure so callers can retain the
 * plain GGUF read path. Runtime read failures are handled separately by the
 * backend's GGUF fallback. layers[] carries layer_count entries for layers
 * layer_lo..layer_lo+layer_count-1. On macOS, use_direct_io selects
 * F_NOCACHE; false leaves the sidecar eligible for normal page-cache reuse.
 */
bool ds4_expert_bundle_open_or_build(ds4_expert_bundle             *b,
                                     const char                    *model_path,
                                     int                            model_fd,
                                     uint64_t                       model_size,
                                     uint32_t                       layer_lo,
                                     uint32_t                       layer_count,
                                     uint32_t                       n_expert,
                                     uint64_t                       gate_bytes,
                                     uint64_t                       up_bytes,
                                     uint64_t                       down_bytes,
                                     const ds4_expert_bundle_layer *layers,
                                     bool                            use_direct_io);
void ds4_expert_bundle_close(ds4_expert_bundle *b);

/* File offset of expert (layer, id)'s record; layer is an absolute index. */
uint64_t ds4_expert_bundle_record_offset(const ds4_expert_bundle *b,
                                         uint32_t                 layer,
                                         uint32_t                 expert);

#endif
