#ifndef DS4_FLINT_STREAM_H
#define DS4_FLINT_STREAM_H

/*
 * SSD streaming of Recursive Flint's routed experts (fork-owned;
 * docs/RECURSIVE_FLINT.md step 4).  Plain C, no Metal: the GPU memory for
 * the expert slots comes from a callback, and the module writes GPU
 * addresses into the kernel's expert table.
 *
 * Decode: the megakernel looks each selected expert up in the address
 * table; a zero entry aborts the launch at that layer and publishes the
 * layer's six ids.  The host calls ds4_flint_stream_fetch for them, which
 * reads the missing packs from the GGUF into free or evicted slots (least
 * recently used first, the layer's own selection pinned), fills the table
 * and returns; the kernel is relaunched from that layer.  Hits cost
 * nothing and read the same bytes the resident path reads, so streamed and
 * resident logits are identical.
 *
 * Prefill: a chunk touches many experts of a layer.  Few distinct experts
 * (short appends) go through the cache; otherwise the layer is staged in
 * one of two layer-sized buffers, read in the background one layer ahead
 * while the previous layer computes, and the per-layer table given to the
 * prefill kernels points into the staging buffer.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_FLINT_STREAM_N_LAYER   49u
#define DS4_FLINT_STREAM_N_EXPERT  256u
#define DS4_FLINT_STREAM_PACK_BYTES 1769472ull
#define DS4_FLINT_STREAM_MIN_SLOTS 64u

typedef struct ds4_flint_stream ds4_flint_stream;

typedef struct ds4_flint_stream_region {
    uint8_t *ptr;      /* host pointer of a shared GPU buffer */
    uint64_t gpu;      /* its GPU address */
} ds4_flint_stream_region;

/* Allocates `bytes` of shared GPU memory reachable through its GPU address
 * (the caller keeps the buffer resident for every encoder).  Returns 1. */
typedef int (*ds4_flint_stream_alloc_fn)(void *ud, uint64_t bytes, const char *label, ds4_flint_stream_region *out);

typedef struct {
    const char *path;                /* the GGUF, opened again for pread */
    const uint64_t *pack_offsets;    /* [49 x 256] byte offsets of each expert pack in the file (0: dense layer) */
    uint32_t n_slots;                /* cache capacity in experts (>= DS4_FLINT_STREAM_MIN_SLOTS) */
    uint64_t *gpu_table;             /* [49 x 256] the kernel's expert address table (shared memory) */
    ds4_flint_stream_alloc_fn alloc;
    void *alloc_ud;
    uint32_t threads;                /* pread threads (0: default) */
    int cached_io;                   /* keep the page cache (default: F_NOCACHE) */
} ds4_flint_stream_config;

ds4_flint_stream *ds4_flint_stream_open(const ds4_flint_stream_config *cfg, char *err, size_t errlen);
void ds4_flint_stream_close(ds4_flint_stream *st);

uint32_t ds4_flint_stream_slots(const ds4_flint_stream *st);
uint64_t ds4_flint_stream_staging_bytes(void);   /* the two layer-sized staging buffers */

/* Decode miss: make the n ids of `layer` resident (misses are read in
 * parallel; the n ids are pinned against eviction).  Returns 1 on success. */
int ds4_flint_stream_fetch(ds4_flint_stream *st, uint32_t layer, const uint32_t *ids, uint32_t n,
                           char *err, size_t errlen);
/* Hits: mark the layer's selection as most recently used (after a launch). */
void ds4_flint_stream_touch(ds4_flint_stream *st, uint32_t layer, const uint32_t *ids, uint32_t n);

/* Prefill of one sparse layer over n_tokens tokens: need[e] != 0 marks the
 * experts the chunk routes to.  Fills table_out[256] with GPU addresses
 * (0 where not needed) and returns 1; the addresses stay valid until the
 * next call for the same chunk parity (layer & 1) or until a decode fetch,
 * so the caller runs the layer's kernels before calling again twice. */
int ds4_flint_stream_prefill_layer(ds4_flint_stream *st, uint32_t layer, const uint8_t *need, uint32_t n_tokens,
                                   uint64_t *table_out, char *err, size_t errlen);
/* Call after a chunk: joins any background layer read. */
void ds4_flint_stream_prefill_end(ds4_flint_stream *st);

typedef struct {
    uint64_t decode_layers;      /* sparse layers run by decode launches */
    uint64_t decode_hits;        /* selections of completed layers (includes the fetched ones: subtract misses) */
    uint64_t decode_misses, decode_relaunches;
    uint64_t decode_bytes;       /* bytes read for decode misses */
    double decode_read_ms;       /* wall time inside fetch */
    uint64_t prefill_layers_cache, prefill_layers_staged, prefill_bytes;
    double prefill_read_ms, prefill_wait_ms;   /* foreground read time; time waiting for background reads */
    uint64_t evictions;
    uint32_t slots, used;
} ds4_flint_stream_stats;

void ds4_flint_stream_get_stats(const ds4_flint_stream *st, ds4_flint_stream_stats *out);
void ds4_flint_stream_reset_stats(ds4_flint_stream *st);

#ifdef __cplusplus
}
#endif

#endif
