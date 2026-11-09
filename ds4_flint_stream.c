/* SSD streaming of Recursive Flint's routed experts (fork-owned).  See
 * ds4_flint_stream.h for the model; docs/RECURSIVE_FLINT.md step 4 for the
 * measurements.  No Metal here: tests/flint/test_flint_stream.c runs this
 * file against a temporary file and malloc'd slots. */

#include "ds4_flint_stream.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define FS_N_LAYER DS4_FLINT_STREAM_N_LAYER
#define FS_N_EXPERT DS4_FLINT_STREAM_N_EXPERT
#define FS_PACK DS4_FLINT_STREAM_PACK_BYTES
#define FS_KEYS (FS_N_LAYER * FS_N_EXPERT)
#define FS_NONE UINT32_MAX
#define FS_REGION_SLOTS 1024u                 /* one GPU buffer holds up to this many slots (1.69 GiB) */
#define FS_MAX_JOBS 512u
#define FS_STAGE_CHUNK (16ull << 20)          /* whole-layer reads split across the pool */

/* Chunk-size policy for prefill (tokens): below CACHE_MAX the layer's experts
 * go through the cache; from FULL_MIN the whole layer is staged and the next
 * one read ahead; between, only the needed experts are staged. */
#define FS_PREFILL_CACHE_MAX_DEFAULT 32u
#define FS_PREFILL_FULL_MIN_DEFAULT 128u

typedef struct {
    uint64_t offset;
    uint64_t len;
    uint8_t *dst;
    int ok;
} fs_job;

struct ds4_flint_stream {
    int fd;
    const uint64_t *pack_offsets;
    uint32_t n_slots;
    uint8_t **slot_ptr;
    uint64_t *slot_gpu;
    uint32_t *slot_key;          /* layer*256+expert, or FS_NONE when free */
    uint32_t *lru_prev, *lru_next;
    uint32_t lru_head, lru_tail; /* head: least recently used */
    uint32_t *free_list;
    uint32_t n_free;
    uint32_t *table;             /* key -> slot + 1, 0 when absent */
    uint64_t *gpu_table;
    uint8_t *pin;                /* [FS_KEYS] transient pins during a fetch */

    ds4_flint_stream_region stage[2];
    int stage_ready;
    ds4_flint_stream_alloc_fn alloc;
    void *alloc_ud;

    /* pread pool */
    pthread_t *workers;
    uint32_t n_workers;
    pthread_mutex_t mu;
    pthread_cond_t cv_job, cv_done;
    fs_job jobs[FS_MAX_JOBS];
    uint32_t n_jobs, next_job, done_jobs;
    int quit;
    pthread_mutex_t submit_mu;   /* one submitter at a time */

    /* background whole-layer read */
    pthread_t bg;
    int bg_active;
    uint32_t bg_layer;
    int bg_parity;
    int bg_ok;
    double bg_ms;

    uint32_t prefill_cache_max, prefill_full_min;
    ds4_flint_stream_stats stats;
};

static double fs_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void fs_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
}

/* ---- pread pool ---- */

static int fs_pread_range(int fd, uint8_t *dst, uint64_t offset, uint64_t len) {
    uint64_t pos = 0;
    while (pos < len) {
        const uint64_t rem = len - pos;
        const size_t want = rem > (1ull << 30) ? (size_t)(1ull << 30) : (size_t)rem;
        ssize_t n;
        do {
            n = pread(fd, dst + pos, want, (off_t)(offset + pos));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return 0;
        pos += (uint64_t)n;
    }
    return 1;
}

static void *fs_worker(void *arg) {
    ds4_flint_stream *st = arg;
    pthread_mutex_lock(&st->mu);
    for (;;) {
        while (!st->quit && st->next_job >= st->n_jobs) pthread_cond_wait(&st->cv_job, &st->mu);
        if (st->quit) break;
        fs_job *j = &st->jobs[st->next_job++];
        pthread_mutex_unlock(&st->mu);
        j->ok = fs_pread_range(st->fd, j->dst, j->offset, j->len);
        pthread_mutex_lock(&st->mu);
        if (++st->done_jobs == st->n_jobs) pthread_cond_broadcast(&st->cv_done);
    }
    pthread_mutex_unlock(&st->mu);
    return NULL;
}

/* Runs the jobs on the pool (the caller's thread joins in) and returns 1
 * when every read completed. */
static int fs_run_jobs(ds4_flint_stream *st, fs_job *jobs, uint32_t n) {
    if (n == 0) return 1;
    int ok = 1;
    pthread_mutex_lock(&st->submit_mu);
    for (uint32_t base = 0; base < n; base += FS_MAX_JOBS) {
        const uint32_t cnt = n - base < FS_MAX_JOBS ? n - base : FS_MAX_JOBS;
        pthread_mutex_lock(&st->mu);
        memcpy(st->jobs, jobs + base, cnt * sizeof(fs_job));
        st->n_jobs = cnt;
        st->next_job = 0;
        st->done_jobs = 0;
        pthread_cond_broadcast(&st->cv_job);
        /* help out */
        while (st->next_job < st->n_jobs) {
            fs_job *j = &st->jobs[st->next_job++];
            pthread_mutex_unlock(&st->mu);
            j->ok = fs_pread_range(st->fd, j->dst, j->offset, j->len);
            pthread_mutex_lock(&st->mu);
            if (++st->done_jobs == st->n_jobs) pthread_cond_broadcast(&st->cv_done);
        }
        while (st->done_jobs < st->n_jobs) pthread_cond_wait(&st->cv_done, &st->mu);
        for (uint32_t i = 0; i < cnt; i++) {
            jobs[base + i].ok = st->jobs[i].ok;
            ok = ok && st->jobs[i].ok;
        }
        st->n_jobs = st->next_job = st->done_jobs = 0;
        pthread_mutex_unlock(&st->mu);
    }
    pthread_mutex_unlock(&st->submit_mu);
    return ok;
}

/* ---- LRU ---- */

static void fs_lru_unlink(ds4_flint_stream *st, uint32_t s) {
    const uint32_t p = st->lru_prev[s], n = st->lru_next[s];
    if (p != FS_NONE) st->lru_next[p] = n; else st->lru_head = n;
    if (n != FS_NONE) st->lru_prev[n] = p; else st->lru_tail = p;
    st->lru_prev[s] = st->lru_next[s] = FS_NONE;
}

static void fs_lru_push_mru(ds4_flint_stream *st, uint32_t s) {
    st->lru_prev[s] = st->lru_tail;
    st->lru_next[s] = FS_NONE;
    if (st->lru_tail != FS_NONE) st->lru_next[st->lru_tail] = s; else st->lru_head = s;
    st->lru_tail = s;
}

/* A slot for a new entry: a free one, else the least recently used that is
 * not pinned.  Returns FS_NONE when everything is pinned. */
static uint32_t fs_take_slot(ds4_flint_stream *st) {
    if (st->n_free) return st->free_list[--st->n_free];
    for (uint32_t s = st->lru_head; s != FS_NONE; s = st->lru_next[s]) {
        const uint32_t key = st->slot_key[s];
        if (st->pin[key]) continue;
        fs_lru_unlink(st, s);
        st->table[key] = 0;
        st->gpu_table[key] = 0;
        st->slot_key[s] = FS_NONE;
        st->stats.evictions++;
        st->stats.used--;
        return s;
    }
    return FS_NONE;
}

static void fs_install(ds4_flint_stream *st, uint32_t key, uint32_t slot) {
    st->slot_key[slot] = key;
    st->table[key] = slot + 1;
    st->gpu_table[key] = st->slot_gpu[slot];
    fs_lru_push_mru(st, slot);
    st->stats.used++;
}

/* Reads the missing keys of a set into slots.  Pins the whole set first so
 * one member's eviction never victimises another. */
static int fs_ensure_cached(ds4_flint_stream *st, const uint32_t *keys, uint32_t n, uint64_t *bytes_out,
                            char *err, size_t errlen) {
    for (uint32_t i = 0; i < n; i++) st->pin[keys[i]] = 1;
    fs_job *jobs = calloc(n ? n : 1, sizeof(fs_job));
    uint32_t *slots = calloc(n ? n : 1, sizeof(uint32_t));
    uint32_t *miss_keys = calloc(n ? n : 1, sizeof(uint32_t));
    uint32_t n_miss = 0;
    int ok = 1;
    for (uint32_t i = 0; i < n && ok; i++) {
        const uint32_t key = keys[i];
        if (st->table[key]) {
            const uint32_t s = st->table[key] - 1;
            fs_lru_unlink(st, s);
            fs_lru_push_mru(st, s);
            continue;
        }
        int dup = 0;
        for (uint32_t j = 0; j < n_miss; j++) if (miss_keys[j] == key) { dup = 1; break; }
        if (dup) continue;
        const uint32_t s = fs_take_slot(st);
        if (s == FS_NONE) {
            fs_err(err, errlen, "expert cache too small for one layer's selection");
            ok = 0;
            break;
        }
        const uint64_t off = st->pack_offsets[key];
        if (!off) {
            fs_err(err, errlen, "expert of a dense layer requested");
            st->free_list[st->n_free++] = s;
            ok = 0;
            break;
        }
        jobs[n_miss] = (fs_job){ off, FS_PACK, st->slot_ptr[s], 0 };
        slots[n_miss] = s;
        miss_keys[n_miss] = key;
        n_miss++;
    }
    if (ok && n_miss) {
        ok = fs_run_jobs(st, jobs, n_miss);
        if (!ok) fs_err(err, errlen, "reading expert packs from the model file failed");
    }
    for (uint32_t i = 0; i < n_miss; i++) {
        if (ok && jobs[i].ok) fs_install(st, miss_keys[i], slots[i]);
        else st->free_list[st->n_free++] = slots[i];
    }
    if (bytes_out) *bytes_out = (uint64_t)n_miss * FS_PACK;
    for (uint32_t i = 0; i < n; i++) st->pin[keys[i]] = 0;
    free(jobs); free(slots); free(miss_keys);
    return ok;
}

/* ---- open / close ---- */

static uint32_t fs_env_u32(const char *name, uint32_t dflt) {
    const char *v = getenv(name);
    if (!v || !v[0]) return dflt;
    const long x = atol(v);
    return x > 0 ? (uint32_t)x : dflt;
}

ds4_flint_stream *ds4_flint_stream_open(const ds4_flint_stream_config *cfg, char *err, size_t errlen) {
    if (!cfg || !cfg->path || !cfg->pack_offsets || !cfg->gpu_table || !cfg->alloc) {
        fs_err(err, errlen, "incomplete streaming configuration");
        return NULL;
    }
    if (cfg->n_slots < DS4_FLINT_STREAM_MIN_SLOTS) {
        fs_err(err, errlen, "the expert cache needs at least 64 slots (108 MiB)");
        return NULL;
    }
    ds4_flint_stream *st = calloc(1, sizeof(*st));
    if (!st) return NULL;
    st->fd = open(cfg->path, O_RDONLY | O_CLOEXEC);
    if (st->fd < 0) {
        fs_err(err, errlen, "cannot open the model for expert reads");
        free(st);
        return NULL;
    }
#ifdef __APPLE__
    if (!cfg->cached_io) {
        (void)fcntl(st->fd, F_NOCACHE, 1);
        (void)fcntl(st->fd, F_RDAHEAD, 0);
    }
#endif
    st->pack_offsets = cfg->pack_offsets;
    st->gpu_table = cfg->gpu_table;
    st->alloc = cfg->alloc;
    st->alloc_ud = cfg->alloc_ud;
    st->n_slots = cfg->n_slots > FS_KEYS ? FS_KEYS : cfg->n_slots;
    st->slot_ptr = calloc(st->n_slots, sizeof(uint8_t *));
    st->slot_gpu = calloc(st->n_slots, sizeof(uint64_t));
    st->slot_key = malloc(st->n_slots * sizeof(uint32_t));
    st->lru_prev = malloc(st->n_slots * sizeof(uint32_t));
    st->lru_next = malloc(st->n_slots * sizeof(uint32_t));
    st->free_list = malloc(st->n_slots * sizeof(uint32_t));
    st->table = calloc(FS_KEYS, sizeof(uint32_t));
    st->pin = calloc(FS_KEYS, 1);
    if (!st->slot_ptr || !st->slot_gpu || !st->slot_key || !st->lru_prev || !st->lru_next || !st->free_list ||
        !st->table || !st->pin) {
        fs_err(err, errlen, "out of memory for the expert cache tables");
        ds4_flint_stream_close(st);
        return NULL;
    }
    for (uint32_t s0 = 0; s0 < st->n_slots; s0 += FS_REGION_SLOTS) {
        const uint32_t cnt = st->n_slots - s0 < FS_REGION_SLOTS ? st->n_slots - s0 : FS_REGION_SLOTS;
        ds4_flint_stream_region r = {0};
        char label[48];
        snprintf(label, sizeof(label), "ds4_flint_experts_%u", s0 / FS_REGION_SLOTS);
        if (!st->alloc(st->alloc_ud, (uint64_t)cnt * FS_PACK, label, &r) || !r.ptr || !r.gpu) {
            fs_err(err, errlen, "out of GPU memory for the expert cache");
            ds4_flint_stream_close(st);
            return NULL;
        }
        for (uint32_t i = 0; i < cnt; i++) {
            st->slot_ptr[s0 + i] = r.ptr + (uint64_t)i * FS_PACK;
            st->slot_gpu[s0 + i] = r.gpu + (uint64_t)i * FS_PACK;
        }
    }
    for (uint32_t s = 0; s < st->n_slots; s++) {
        st->slot_key[s] = FS_NONE;
        st->lru_prev[s] = st->lru_next[s] = FS_NONE;
        st->free_list[s] = st->n_slots - 1 - s;   /* slot 0 is taken first */
    }
    st->n_free = st->n_slots;
    st->lru_head = st->lru_tail = FS_NONE;
    memset(st->gpu_table, 0, FS_KEYS * sizeof(uint64_t));
    st->stats.slots = st->n_slots;

    pthread_mutex_init(&st->mu, NULL);
    pthread_mutex_init(&st->submit_mu, NULL);
    pthread_cond_init(&st->cv_job, NULL);
    pthread_cond_init(&st->cv_done, NULL);
    st->n_workers = cfg->threads ? cfg->threads : fs_env_u32("DS4_FLINT_STREAM_THREADS", 6);
    if (st->n_workers > 32) st->n_workers = 32;
    st->workers = calloc(st->n_workers, sizeof(pthread_t));
    for (uint32_t i = 0; i < st->n_workers; i++) {
        if (pthread_create(&st->workers[i], NULL, fs_worker, st) != 0) {
            st->n_workers = i;
            break;
        }
    }
    st->prefill_cache_max = fs_env_u32("DS4_FLINT_PREFILL_CACHE_MAX", FS_PREFILL_CACHE_MAX_DEFAULT);
    st->prefill_full_min = fs_env_u32("DS4_FLINT_PREFILL_FULL_MIN", FS_PREFILL_FULL_MIN_DEFAULT);
    if (st->prefill_full_min <= st->prefill_cache_max) st->prefill_full_min = st->prefill_cache_max + 1;
    return st;
}

void ds4_flint_stream_close(ds4_flint_stream *st) {
    if (!st) return;
    ds4_flint_stream_prefill_end(st);
    if (st->workers) {
        pthread_mutex_lock(&st->mu);
        st->quit = 1;
        pthread_cond_broadcast(&st->cv_job);
        pthread_mutex_unlock(&st->mu);
        for (uint32_t i = 0; i < st->n_workers; i++) pthread_join(st->workers[i], NULL);
        free(st->workers);
        pthread_mutex_destroy(&st->mu);
        pthread_mutex_destroy(&st->submit_mu);
        pthread_cond_destroy(&st->cv_job);
        pthread_cond_destroy(&st->cv_done);
    }
    if (st->fd >= 0) close(st->fd);
    free(st->slot_ptr); free(st->slot_gpu); free(st->slot_key); free(st->lru_prev); free(st->lru_next);
    free(st->free_list); free(st->table); free(st->pin);
    free(st);
}

uint32_t ds4_flint_stream_slots(const ds4_flint_stream *st) { return st ? st->n_slots : 0; }
uint64_t ds4_flint_stream_staging_bytes(void) { return 2ull * FS_N_EXPERT * FS_PACK; }

/* ---- decode ---- */

int ds4_flint_stream_fetch(ds4_flint_stream *st, uint32_t layer, const uint32_t *ids, uint32_t n,
                           char *err, size_t errlen) {
    if (!st || layer >= FS_N_LAYER || n > 8) { fs_err(err, errlen, "bad fetch request"); return 0; }
    uint32_t keys[8];
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] >= FS_N_EXPERT) { fs_err(err, errlen, "expert id out of range"); return 0; }
        keys[i] = layer * FS_N_EXPERT + ids[i];
    }
    const double t0 = fs_now_ms();
    uint64_t bytes = 0;
    const int ok = fs_ensure_cached(st, keys, n, &bytes, err, errlen);
    st->stats.decode_read_ms += fs_now_ms() - t0;
    st->stats.decode_bytes += bytes;
    st->stats.decode_misses += bytes / FS_PACK;
    st->stats.decode_relaunches++;
    return ok;
}

void ds4_flint_stream_touch(ds4_flint_stream *st, uint32_t layer, const uint32_t *ids, uint32_t n) {
    if (!st || layer >= FS_N_LAYER) return;
    st->stats.decode_layers++;
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] >= FS_N_EXPERT) continue;
        const uint32_t key = layer * FS_N_EXPERT + ids[i];
        if (!st->table[key]) continue;
        const uint32_t s = st->table[key] - 1;
        fs_lru_unlink(st, s);
        fs_lru_push_mru(st, s);
    }
    /* every selection of a completed layer was resident; the ones fetch
     * brought in were counted as misses there */
    st->stats.decode_hits += n;
}

/* ---- prefill ---- */

static int fs_stage_alloc(ds4_flint_stream *st, char *err, size_t errlen) {
    if (st->stage_ready) return 1;
    for (int p = 0; p < 2; p++) {
        char label[32];
        snprintf(label, sizeof(label), "ds4_flint_stage%d", p);
        if (!st->alloc(st->alloc_ud, (uint64_t)FS_N_EXPERT * FS_PACK, label, &st->stage[p]) || !st->stage[p].ptr) {
            fs_err(err, errlen, "out of GPU memory for the prefill staging buffers");
            return 0;
        }
    }
    st->stage_ready = 1;
    return 1;
}

/* Reads experts of `layer` with need[e] set into stage[parity]; need NULL
 * reads the whole layer as one contiguous range split across the pool. */
static int fs_stage_read(ds4_flint_stream *st, uint32_t layer, const uint8_t *need, int parity, uint64_t *bytes_out) {
    fs_job *jobs = calloc(FS_N_EXPERT * 2, sizeof(fs_job));
    uint32_t n = 0;
    uint64_t bytes = 0;
    uint8_t *base = st->stage[parity].ptr;
    const uint64_t layer_off = st->pack_offsets[layer * FS_N_EXPERT];
    if (!need) {
        const uint64_t total = (uint64_t)FS_N_EXPERT * FS_PACK;
        for (uint64_t off = 0; off < total; off += FS_STAGE_CHUNK) {
            const uint64_t len = total - off < FS_STAGE_CHUNK ? total - off : FS_STAGE_CHUNK;
            jobs[n++] = (fs_job){ layer_off + off, len, base + off, 0 };
        }
        bytes = total;
    } else {
        for (uint32_t e = 0; e < FS_N_EXPERT; e++) {
            if (!need[e]) continue;
            jobs[n++] = (fs_job){ st->pack_offsets[layer * FS_N_EXPERT + e], FS_PACK, base + (uint64_t)e * FS_PACK, 0 };
            bytes += FS_PACK;
        }
    }
    const int ok = fs_run_jobs(st, jobs, n);
    free(jobs);
    if (bytes_out) *bytes_out = bytes;
    return ok;
}

static void *fs_bg_main(void *arg) {
    ds4_flint_stream *st = arg;
    const double t0 = fs_now_ms();
    uint64_t bytes = 0;
    st->bg_ok = fs_stage_read(st, st->bg_layer, NULL, st->bg_parity, &bytes);
    st->bg_ms = fs_now_ms() - t0;
    return NULL;
}

static void fs_bg_join(ds4_flint_stream *st) {
    if (!st->bg_active) return;
    pthread_join(st->bg, NULL);
    st->bg_active = 0;
}

static void fs_bg_start(ds4_flint_stream *st, uint32_t layer, int parity) {
    fs_bg_join(st);
    st->bg_layer = layer;
    st->bg_parity = parity;
    st->bg_ok = 0;
    if (pthread_create(&st->bg, NULL, fs_bg_main, st) == 0) st->bg_active = 1;
}

int ds4_flint_stream_prefill_layer(ds4_flint_stream *st, uint32_t layer, const uint8_t *need, uint32_t n_tokens,
                                   uint64_t *table_out, char *err, size_t errlen) {
    if (!st || layer >= FS_N_LAYER || !need || !table_out) { fs_err(err, errlen, "bad prefill request"); return 0; }
    if (!st->pack_offsets[layer * FS_N_EXPERT]) { fs_err(err, errlen, "prefill staging asked for a dense layer"); return 0; }
    memset(table_out, 0, FS_N_EXPERT * sizeof(uint64_t));
    uint32_t keys[FS_N_EXPERT], n = 0;
    for (uint32_t e = 0; e < FS_N_EXPERT; e++) if (need[e]) keys[n++] = layer * FS_N_EXPERT + e;
    if (n == 0) return 1;
    /* short appends go through the cache; a selection that would take more
     * than half of it is staged instead so the cache keeps its working set */
    if (n_tokens < st->prefill_cache_max && n <= st->n_slots / 2) {
        fs_bg_join(st);
        const double t0 = fs_now_ms();
        uint64_t bytes = 0;
        if (!fs_ensure_cached(st, keys, n, &bytes, err, errlen)) return 0;
        st->stats.prefill_read_ms += fs_now_ms() - t0;
        st->stats.prefill_bytes += bytes;
        st->stats.prefill_layers_cache++;
        for (uint32_t i = 0; i < n; i++) table_out[keys[i] % FS_N_EXPERT] = st->gpu_table[keys[i]];
        return 1;
    }
    if (!fs_stage_alloc(st, err, errlen)) return 0;
    const int parity = (int)(layer & 1u);
    const int full = n_tokens >= st->prefill_full_min;
    int have = 0;
    if (st->bg_active) {
        const double t0 = fs_now_ms();
        const uint32_t bg_layer = st->bg_layer;
        const int bg_parity = st->bg_parity;
        fs_bg_join(st);
        st->stats.prefill_wait_ms += fs_now_ms() - t0;
        if (bg_layer == layer && bg_parity == parity && st->bg_ok) {
            have = 1;
            st->stats.prefill_bytes += (uint64_t)FS_N_EXPERT * FS_PACK;
        }
    }
    if (!have) {
        const double t0 = fs_now_ms();
        uint64_t bytes = 0;
        if (!fs_stage_read(st, layer, full ? NULL : need, parity, &bytes)) {
            fs_err(err, errlen, "reading an expert layer from the model file failed");
            return 0;
        }
        st->stats.prefill_read_ms += fs_now_ms() - t0;
        st->stats.prefill_bytes += bytes;
    }
    st->stats.prefill_layers_staged++;
    for (uint32_t e = 0; e < FS_N_EXPERT; e++) {
        if (need[e] || full) table_out[e] = st->stage[parity].gpu + (uint64_t)e * FS_PACK;
    }
    /* read the next sparse layer ahead into the other buffer, which the
     * GPU finished with before the caller waited on this layer's routing */
    if (full && layer + 1 < FS_N_LAYER && st->pack_offsets[(layer + 1) * FS_N_EXPERT]) {
        fs_bg_start(st, layer + 1, parity ^ 1);
    }
    return 1;
}

void ds4_flint_stream_prefill_end(ds4_flint_stream *st) {
    if (st) fs_bg_join(st);
}

/* ---- stats ---- */

void ds4_flint_stream_get_stats(const ds4_flint_stream *st, ds4_flint_stream_stats *out) {
    if (!out) return;
    if (!st) { memset(out, 0, sizeof(*out)); return; }
    *out = st->stats;
}

void ds4_flint_stream_reset_stats(ds4_flint_stream *st) {
    if (!st) return;
    const uint32_t slots = st->stats.slots, used = st->stats.used;
    memset(&st->stats, 0, sizeof(st->stats));
    st->stats.slots = slots;
    st->stats.used = used;
}
