#ifndef K3_STATIC_CACHE_H
#define K3_STATIC_CACHE_H
#include <stddef.h>
#include <stdint.h>

typedef struct k3_static_cache k3_static_cache;
typedef struct {
    uint32_t layer;
    uint64_t offset;
    uint64_t bytes;
} k3_static_cache_span;

/* Metadata-only open. The source fd is duplicated and its stat identity must
 * match the sidecar. Expected spans may be a sorted subset of sidecar layers;
 * the complete index is always validated, including layers outside the subset.
 * No decode arenas/threads are allocated until prepare, after admission. */
int k3_static_cache_open(const char *path, int model_fd,
    const k3_static_cache_span *spans, uint32_t span_count, unsigned workers,
    k3_static_cache **out, char *err, size_t errcap);

/* Conservative memory reservation after prepare: metadata rounded to pages,
 * exact fixed worker arenas/stacks, and 64KiB/worker for pthread control/TLS.
 * Charge once per shared cache before prepare. Read creates no extra decode
 * buffers and uses static zstd contexts. */
uint64_t k3_static_cache_workspace_bytes(const k3_static_cache *cache);
int k3_static_cache_prepare(k3_static_cache *cache, char *err, size_t errcap);

/* Synchronous, internally serialized across retained users. Layer/offset/bytes
 * must match the expected original aligned cover exactly. dst retains that
 * coordinate system. physical_bytes reports actual padded sidecar pread bytes,
 * not decoded bytes. On failure dst is partial/invalid and MUST NOT be used. */
int k3_static_cache_read(k3_static_cache *cache, uint32_t layer,
    uint64_t offset, uint64_t bytes, void *dst, uint64_t dst_capacity,
    uint64_t *physical_bytes, char *err, size_t errcap);

void k3_static_cache_retain(k3_static_cache *cache);
void k3_static_cache_release(k3_static_cache *cache);

/* Bounded serial CPU builder; never modifies the model or overwrites a cache.
 * Publishes via exclusive atomic link of a completed temporary file (rename
 * with no-replace semantics where supported), then removes the temp name.
 * Every block is roundtrip-verified before writing, source stat checked before
 * and after the pass. Caller supplies model metadata-derived aligned spans. */
int k3_static_cache_build(const char *path, int model_fd,
    const k3_static_cache_span *spans, uint32_t span_count,
    char *err, size_t errcap);

#endif
