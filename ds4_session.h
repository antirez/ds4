#ifndef DS4_SESSION_H
#define DS4_SESSION_H

/* Session pool for concurrent context windows.
 *
 * Metal/CUDA can only run one compute pass at a time, so the pool does not
 * provide parallel execution.  What it does provide is **zero-cost switching**:
 * multiple sessions stay resident in GPU memory, and the worker picks the right
 * one for each request without disk I/O or re-prefill.
 *
 * Routing:
 *   1. If the request carries a session_id, find the slot with that ID.
 *   2. Otherwise, find the slot whose live checkpoint is a prefix of the prompt.
 *   3. If no slot matches, evict the LRU slot (optionally to disk) and init fresh.
 *
 * Disk KV eviction (L2 cache):
 *   When a slot is evicted from the in-memory pool, its KV state is persisted
 *   to disk if it has meaningful state (>= 1000 tokens).  On acquire, if a
 *   matching disk cache file exists for the session_id, it's loaded instead of
 *   forcing a full cold prefill.
 */

#include "ds4.h"
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    ds4_session *session;
    char         session_id[64];  /* client-assigned affinity key, or empty */
    uint64_t     last_used_seq;  /* monotonic counter for LRU */
    bool         active;         /* true if session is allocated */
    int          disk_pos;       /* token count last persisted to disk (avoid overwriting larger files) */
    int          pos_snapshot;   /* last worker-published token position for diagnostics */
} ds4_pool_slot;

/* Pool stats populated on each ds4_pool_acquire() call. */
typedef struct {
    int total_slots;
    int active_slots;
    int hit_type;  /* 0=miss/empty, 1=session_id match, 2=prefix match, 3=eviction */
    int matched_slot;
    int evicted_slot;  /* -1 if no eviction */
} ds4_pool_stats;

typedef struct {
    ds4_pool_slot    *slots;
    int               n_slots;       /* --sessions N */
    int               active_count;  /* currently allocated slots */
    uint64_t          seq;           /* monotonic clock for LRU */
    ds4_engine       *engine;
    int               ctx_size;
    char             *kv_disk_dir;   /* disk L2 cache dir, NULL = disabled */
    ds4_pool_stats    last_acquire;  /* stats from most recent acquire call */
} ds4_pool;

/* Initialize the pool with n_slots session slots. */
int ds4_pool_init(ds4_pool *pool, ds4_engine *engine, int n_slots, int ctx_size);

/* Enable disk KV eviction (L2 cache).  Call after init.
 * dir: filesystem path for .kv files.  Created if it doesn't exist.
 * Pass NULL or don't call to disable disk eviction. */
void ds4_pool_set_kv_disk_dir(ds4_pool *pool, const char *dir);

/* Free all sessions and disk resources in the pool. */
void ds4_pool_free(ds4_pool *pool);

/* Find or allocate a session for the given request.
 *
 * session_id: optional client affinity key (NULL = use prefix matching)
 * prompt: the full tokenized prompt for prefix matching
 *
 * Returns the matched/allocated slot index, or -1 on failure.
 * The returned slot's session is ready for ds4_session_sync(). */
int ds4_pool_acquire(ds4_pool *pool, const char *session_id,
                     const ds4_tokens *prompt);

/* Get the session for a slot. */
ds4_session *ds4_pool_session(ds4_pool *pool, int slot_idx);

/* Mark a slot as recently used (called after generation completes). */
void ds4_pool_touch(ds4_pool *pool, int slot_idx);

/* Persist a slot's KV state to disk L2 cache (by session_id). */
void ds4_pool_persist_slot(ds4_pool *pool, int slot_idx);

/* Pre-warm a slot for a session without generation.  Same routing as acquire
 * but intended for background prefill.  Returns slot index or -1. */
int ds4_pool_warmup(ds4_pool *pool, const char *session_id,
                    const ds4_tokens *prompt);

#endif
