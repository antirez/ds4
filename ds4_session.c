#include "ds4_session.h"
#include "ds4.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* =========================================================================
 * Session Pool Implementation.
 *
 * The pool maintains N GPU sessions in memory simultaneously.  GPU resources
 * (command buffers, KV tensors) are allocated per-session at creation and stay
 * resident.  Switching between sessions is a pointer swap — no GPU work, no
 * disk I/O.
 *
 * Memory cost per session at 100K context (DeepSeek V4 compressed KV):
 *   ~1.5 GB on Metal (dominated by ratio-4 compressed layers + indexer)
 *
 * On a 128GB M3 Max with the 81GB q2 model, 4 sessions = 6GB, leaving ~40GB
 * of headroom.  Even 8 sessions (12GB) is comfortable.
 * ========================================================================= */

#define DISK_EVICT_MIN_TOKENS 1000  /* only persist sessions with meaningful state */

static void *pool_malloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "ds4_pool: out of memory\n");
        exit(1);
    }
    return p;
}

/* =========================================================================
 * Disk KV Cache Helpers.
 *
 * Session KV state is persisted to {kv_disk_dir}/{sanitized_session_id}.kv
 * on eviction and loaded back on acquire.  This gives the pool an L2 cache:
 * evicted sessions can be restored without a full cold prefill.
 * ========================================================================= */

/* Sanitize session_id for filesystem safety: non-alnum chars become '_'. */
static void sanitize_session_id(const char *session_id, char *out, size_t outlen) {
    size_t i = 0;
    while (session_id[i] && i < outlen - 1) {
        out[i] = isalnum((unsigned char)session_id[i]) ? session_id[i] : '_';
        i++;
    }
    out[i] = '\0';
}

/* Build the disk cache path for a session_id. Returns malloc'd string. */
static char *disk_cache_path(const char *dir, const char *session_id) {
    if (!dir || !session_id || !session_id[0]) return NULL;
    char safe_id[64];
    sanitize_session_id(session_id, safe_id, sizeof(safe_id));
    if (!safe_id[0]) return NULL;
    size_t len = strlen(dir) + 1 + strlen(safe_id) + 4; /* dir/id.kv\0 */
    char *path = malloc(len);
    if (!path) return NULL;
    snprintf(path, len, "%s/%s.kv", dir, safe_id);
    return path;
}

/* Save a session's KV state to disk.  Returns true on success. */
static bool disk_save_session(const char *dir, const char *session_id,
                              ds4_session *session) {
    if (!dir || !session_id || !session_id[0]) return false;
    char *path = disk_cache_path(dir, session_id);
    if (!path) return false;

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        free(path);
        return false;
    }

    char err[160];
    int rc = ds4_session_save_payload(session, fp, err, sizeof(err));
    fclose(fp);

    if (rc != 0) {
        unlink(path);  /* remove partial file on failure */
        fprintf(stderr, "ds4_pool: disk save failed for '%s': %s\n", session_id, err);
        free(path);
        return false;
    }

    fprintf(stderr, "ds4_pool: saved '%s' to disk (%d tokens)\n",
            session_id, ds4_session_pos(session));
    free(path);
    return true;
}

/* Try to load a session's KV state from disk.  Returns true on success. */
static bool disk_load_session(const char *dir, const char *session_id,
                              ds4_session *session) {
    if (!dir || !session_id || !session_id[0]) return false;
    char *path = disk_cache_path(dir, session_id);
    if (!path) return false;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(path);
        return false;  /* no cached file — not an error */
    }

    /* Get file size for payload_bytes parameter */
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize <= 0) {
        fclose(fp);
        free(path);
        return false;
    }

    char err[160];
    int rc = ds4_session_load_payload(session, fp, (uint64_t)fsize, err, sizeof(err));
    fclose(fp);

    if (rc != 0) {
        fprintf(stderr, "ds4_pool: disk load failed for '%s': %s\n", session_id, err);
        free(path);
        return false;
    }

    fprintf(stderr, "ds4_pool: restored '%s' from disk (%d tokens)\n",
            session_id, ds4_session_pos(session));
    free(path);
    return true;
}

/* Ensure the disk cache directory exists. */
static bool ensure_disk_dir(const char *dir) {
    if (!dir) return false;
    struct stat st;
    if (stat(dir, &st) == 0 && S_ISDIR(st.st_mode)) return true;
    return mkdir(dir, 0755) == 0;
}

/* =========================================================================
 * Pool Init / Free
 * ========================================================================= */

int ds4_pool_init(ds4_pool *pool, ds4_engine *engine, int n_slots, int ctx_size) {
    memset(pool, 0, sizeof(*pool));
    if (n_slots < 1) n_slots = 1;
    if (n_slots > 64) n_slots = 64;

    pool->engine = engine;
    pool->n_slots = n_slots;
    pool->ctx_size = ctx_size;
    pool->slots = pool_malloc((size_t)n_slots * sizeof(ds4_pool_slot));
    memset(pool->slots, 0, (size_t)n_slots * sizeof(ds4_pool_slot));

    /* Pre-allocate all sessions upfront.  This makes the memory cost
     * deterministic at startup — no surprises during inference. */
    for (int i = 0; i < n_slots; i++) {
        ds4_pool_slot *slot = &pool->slots[i];
        if (ds4_session_create(&slot->session, engine, ctx_size) != 0) {
            fprintf(stderr, "ds4_pool: failed to create session %d/%d\n", i + 1, n_slots);
            for (int j = 0; j < i; j++) {
                ds4_session_free(pool->slots[j].session);
            }
            free(pool->slots);
            return -1;
        }
        slot->active = false;
        slot->last_used_seq = 0;
        slot->pos_snapshot = 0;
        slot->session_id[0] = '\0';
    }

    fprintf(stderr, "ds4_pool: initialized %d session slots (ctx=%d each)\n",
            n_slots, ctx_size);
    return 0;
}

void ds4_pool_set_kv_disk_dir(ds4_pool *pool, const char *dir) {
    if (!pool || !dir || !dir[0]) return;
    free(pool->kv_disk_dir);  /* safe on NULL */
    if (ensure_disk_dir(dir)) {
        pool->kv_disk_dir = strdup(dir);
        fprintf(stderr, "ds4_pool: disk KV cache enabled at '%s'\n", dir);
    } else {
        fprintf(stderr, "ds4_pool: warning: could not create disk dir '%s', "
                "disk eviction disabled\n", dir);
        pool->kv_disk_dir = NULL;
    }
}

void ds4_pool_free(ds4_pool *pool) {
    if (!pool) return;
    for (int i = 0; i < pool->n_slots; i++) {
        if (pool->slots[i].session) {
            ds4_session_free(pool->slots[i].session);
        }
    }
    free(pool->kv_disk_dir);
    free(pool->slots);
    memset(pool, 0, sizeof(*pool));
}

/* =========================================================================
 * Slot Lookup Routines
 * ========================================================================= */

/* Find slot by session_id.  Returns -1 if not found. */
static int find_by_session_id(ds4_pool *pool, const char *session_id) {
    if (!session_id || !session_id[0]) return -1;
    for (int i = 0; i < pool->n_slots; i++) {
        if (pool->slots[i].active &&
            strcmp(pool->slots[i].session_id, session_id) == 0) {
            return i;
        }
    }
    return -1;
}

/* Find the slot whose live checkpoint is the longest prefix of `prompt`.
 * Returns the slot index with the longest prefix match, or -1 if no active
 * slot has any common prefix. */
static int find_best_prefix_match(ds4_pool *pool, const ds4_tokens *prompt) {
    int best_idx = -1;
    int best_len = 0;

    for (int i = 0; i < pool->n_slots; i++) {
        if (!pool->slots[i].active) continue;
        const ds4_tokens *live = ds4_session_tokens(pool->slots[i].session);
        if (!live || live->len == 0) continue;

        int common = ds4_session_common_prefix(pool->slots[i].session, prompt);
        int live_pos = ds4_session_pos(pool->slots[i].session);

        /* A valid prefix match: the live checkpoint IS a prefix of the prompt
         * (common == live_pos && prompt->len >= live_pos). */
        if (common == live_pos && prompt->len >= live_pos && common > best_len) {
            best_len = common;
            best_idx = i;
        }
    }
    return best_idx;
}

/* Find an empty (inactive) slot. */
static int find_empty_slot(ds4_pool *pool) {
    for (int i = 0; i < pool->n_slots; i++) {
        if (!pool->slots[i].active) return i;
    }
    return -1;
}

/* Find the least-recently-used active slot for eviction. */
static int find_lru_slot(ds4_pool *pool) {
    int lru_idx = -1;
    uint64_t lru_seq = UINT64_MAX;
    for (int i = 0; i < pool->n_slots; i++) {
        if (!pool->slots[i].active) continue;
        if (pool->slots[i].last_used_seq < lru_seq) {
            lru_seq = pool->slots[i].last_used_seq;
            lru_idx = i;
        }
    }
    return lru_idx;
}

/* =========================================================================
 * Acquire (main routing logic)
 * ========================================================================= */

/* Prepare a newly-assigned slot: try disk load, then fall back to invalidate
 * (cold prefill).  Returns the slot index. */
static int prepare_fresh_slot(ds4_pool *pool, int idx, const char *session_id,
                              const ds4_tokens *prompt) {
    pool->slots[idx].active = true;
    pool->slots[idx].last_used_seq = pool->seq;
    pool->slots[idx].disk_pos = 0;
    if (session_id && session_id[0]) {
        snprintf(pool->slots[idx].session_id,
                 sizeof(pool->slots[idx].session_id), "%s", session_id);
    }

    /* Try loading from disk L2 cache (returning sessions). */
    int disk_match = 0;
    bool disk_loaded = false;
    if (pool->kv_disk_dir && session_id && session_id[0]) {
        if (disk_load_session(pool->kv_disk_dir, session_id,
                              pool->slots[idx].session)) {
            disk_match = ds4_session_common_prefix(pool->slots[idx].session, prompt);
            pool->slots[idx].disk_pos = ds4_session_pos(pool->slots[idx].session);
            pool->slots[idx].pos_snapshot = pool->slots[idx].disk_pos;
            disk_loaded = true;
        }
    }

    if (disk_loaded && disk_match > 0) {
        fprintf(stderr, "ds4_pool: using disk L2 for '%s' (prefix=%d tokens)\n",
                session_id, disk_match);
        return idx;
    }

    /* Cold start — invalidate so sync() will full-prefill. */
    ds4_session_invalidate(pool->slots[idx].session);
    pool->slots[idx].pos_snapshot = 0;
    return idx;
}

int ds4_pool_acquire(ds4_pool *pool, const char *session_id,
                     const ds4_tokens *prompt) {
    pool->seq++;
    pool->last_acquire = (ds4_pool_stats){
        .total_slots = pool->n_slots,
        .active_slots = pool->active_count,
        .hit_type = 0,
        .matched_slot = -1,
        .evicted_slot = -1,
    };

    /* Strategy 1: Exact session_id match. */
    int idx = find_by_session_id(pool, session_id);
    if (idx >= 0) {
        pool->slots[idx].last_used_seq = pool->seq;
        pool->last_acquire.hit_type = 1;
        pool->last_acquire.matched_slot = idx;
        return idx;
    }

    /* Strategy 2: Best prefix match among active sessions.
     * Only used when session_id is NOT set (legacy mode without affinity).
     * When session_id IS set and Strategy 1 missed, skip prefix matching
     * entirely — the shared prefix makes all sessions look alike, causing
     * slot theft ping-pong between panes. */
    if (session_id && session_id[0]) {
        fprintf(stderr, "ds4_pool: session '%s' not in pool, skipping prefix match → disk/empty/evict\n", session_id);
        goto strategy3;
    }
    idx = find_best_prefix_match(pool, prompt);
    if (idx >= 0) {
        pool->slots[idx].last_used_seq = pool->seq;
        if (session_id && session_id[0] &&
            strcmp(pool->slots[idx].session_id, session_id) != 0) {
            if (pool->kv_disk_dir && pool->slots[idx].session_id[0] &&
                ds4_session_pos(pool->slots[idx].session) >= DISK_EVICT_MIN_TOKENS) {
                disk_save_session(pool->kv_disk_dir,
                                  pool->slots[idx].session_id,
                                  pool->slots[idx].session);
            }
            snprintf(pool->slots[idx].session_id,
                     sizeof(pool->slots[idx].session_id),
                     "%s", session_id);
        }
        pool->last_acquire.hit_type = 2;
        pool->last_acquire.matched_slot = idx;
        return idx;
    }

strategy3:
    /* Strategy 3: Allocate an empty slot. */
    idx = find_empty_slot(pool);
    if (idx >= 0) {
        pool->active_count++;
        pool->last_acquire.active_slots = pool->active_count;
        pool->last_acquire.matched_slot = idx;
        return prepare_fresh_slot(pool, idx, session_id, prompt);
    }

    /* Strategy 4: Evict LRU slot, persist to disk if worthwhile. */
    idx = find_lru_slot(pool);
    if (idx >= 0) {
        if (pool->kv_disk_dir && pool->slots[idx].session_id[0] &&
            ds4_session_pos(pool->slots[idx].session) >= DISK_EVICT_MIN_TOKENS) {
            disk_save_session(pool->kv_disk_dir, pool->slots[idx].session_id,
                              pool->slots[idx].session);
        }

        pool->slots[idx].session_id[0] = '\0';
        pool->slots[idx].active = false;
        pool->slots[idx].pos_snapshot = 0;
        pool->last_acquire.hit_type = 3;
        pool->last_acquire.matched_slot = idx;
        pool->last_acquire.evicted_slot = idx;
        return prepare_fresh_slot(pool, idx, session_id, prompt);
    }

    return -1; /* Should be unreachable with n_slots >= 1 */
}

ds4_session *ds4_pool_session(ds4_pool *pool, int slot_idx) {
    if (slot_idx < 0 || slot_idx >= pool->n_slots) return NULL;
    return pool->slots[slot_idx].session;
}

void ds4_pool_touch(ds4_pool *pool, int slot_idx) {
    if (slot_idx < 0 || slot_idx >= pool->n_slots) return;
    pool->seq++;
    pool->slots[slot_idx].last_used_seq = pool->seq;
    pool->slots[slot_idx].pos_snapshot = ds4_session_pos(pool->slots[slot_idx].session);
}

void ds4_pool_persist_slot(ds4_pool *pool, int slot_idx) {
    if (!pool || slot_idx < 0 || slot_idx >= pool->n_slots) return;
    if (!pool->kv_disk_dir) return;
    const char *sid = pool->slots[slot_idx].session_id;
    if (!sid[0]) return;
    ds4_session *sess = pool->slots[slot_idx].session;
    if (!ds4_session_checkpoint_valid(sess)) return;
    int pos = ds4_session_pos(sess);
    if (pos < DISK_EVICT_MIN_TOKENS) return;
    /* Don't overwrite a larger disk file with a smaller state. */
    if (pos < pool->slots[slot_idx].disk_pos) return;
    disk_save_session(pool->kv_disk_dir, sid, sess);
    pool->slots[slot_idx].disk_pos = pos;
}

/* Warmup is identical to acquire — same routing, same disk logic. */
int ds4_pool_warmup(ds4_pool *pool, const char *session_id,
                    const ds4_tokens *prompt) {
    return ds4_pool_acquire(pool, session_id, prompt);
}
