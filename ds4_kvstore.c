#include "ds4_kvstore.h"
#include "lz4.h"

/* Shared disk KV checkpoint file support.
 *
 * The low-level file layout and payload helpers are intentionally shared.  The
 * ds4-server still owns the automatic byte-prefix cache policy built on top of
 * this file; ds4-agent uses only the same durable format for explicit sessions,
 * with its own policy in ds4_agent.c.  Protocol-specific extras, such as the
 * server's tool-id -> exact DSML trailer, are attached through trailer hooks and
 * still live with the protocol code that owns those mappings. */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define KV_CACHE_MAGIC0 'K'
#define KV_CACHE_MAGIC1 'V'
#define KV_CACHE_MAGIC2 'C'
/* v1 was uncompressed.  v2 adds an in-place codec byte at header offset 7
 * (formerly reserved) and a chunk_size u32 at offset 20 (formerly reserved).
 * Older builds reject v2 files at read_header and the existing "replaced
 * incompatible file" path retires them; the cache directory is disposable. */
#define KV_CACHE_VERSION 2u
#define KV_CACHE_DEFAULT_MIN_TOKENS 512
#define KV_CACHE_DEFAULT_COLD_MAX_TOKENS 30000
/* Tokenizers may merge text across the prompt boundary. Trimming a small tail
 * still improves the cheap token-prefix path, while text-prefix lookup handles
 * cases where canonical prompt tokenization spells the same bytes differently.
 * The 2048 alignment also matches the backend prefill chunk schedule, which
 * keeps compressor row finalization identical to a cold full prompt. */
#define KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS 32
#define KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS 2048
#define KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS 10000
/* Disk-hit counts are evidence that a checkpoint was useful, but only while
 * the workload still resembles the one that produced those hits. */
#define KV_CACHE_MIN_EFFECTIVE_HITS 0.01

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} kv_buf;

static void kv_die(const char *msg) {
    fprintf(stderr, "ds4-kvstore: %s\n", msg);
    exit(1);
}

static void *kv_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

static void *kv_xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

static char *kv_xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = kv_xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static void kv_buf_reserve(kv_buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) kv_die("buffer overflow");
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) cap *= 2;
    b->ptr = kv_xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void kv_buf_append(kv_buf *b, const void *p, size_t n) {
    kv_buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void kv_buf_putc(kv_buf *b, char c) {
    kv_buf_append(b, &c, 1);
}

static void kv_buf_puts(kv_buf *b, const char *s) {
    kv_buf_append(b, s, strlen(s));
}

static void kv_buf_printf(kv_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) kv_die("vsnprintf failed");
    kv_buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static char *kv_buf_take(kv_buf *b) {
    if (!b->ptr) return kv_xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static double kv_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static const char *kv_log_name(const ds4_kvstore *kc) {
    return kc && kc->log_name ? kc->log_name : "ds4";
}

static void kv_logf(ds4_kvstore *kc, ds4_kvstore_log_type type,
                    const char *fmt, ...) {
    if (!kc || !kc->log) return;
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(ap2);
        return;
    }
    char *msg = kv_xmalloc((size_t)n + 1);
    vsnprintf(msg, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    kc->log(kc->log_ud, type, msg);
    free(msg);
}

/* Forward declarations: the lz4 streaming block below uses these helpers,
 * whose canonical definitions live further down. */
static void kv_le_put64(uint8_t *p, uint64_t v);
static uint64_t kv_le_get64(const uint8_t *p);

/* Cap at 8 so a save cannot monopolise cores that should still be running
 * inference.  Env override honoured for parity with DS4_THREADS;
 * --kv-cache-compression-threads overrides both. */
static int kv_cache_default_compression_threads(void) {
    const char *env = getenv("DS4_KV_CACHE_COMPRESSION_THREADS");
    if (env && env[0]) {
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end && !*end && v >= 0 && v <= 64) return (int)v;
    }
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online < 1) online = 1;
    if (online > 8) online = 8;
    return (int)online;
}

/* ---------------------- KV Cache LZ4 streaming -----------------------------
 *
 * Wrap the engine's ds4_session_save_payload(fp) / load_payload(fp) FILE *
 * with a cookie FILE that compresses or decompresses chunks of the payload
 * region in parallel.  Engine signatures are unchanged.
 *
 * Payload framing on disk, when DS4_KVSTORE_CODEC_LZ4 is set in the KVC header:
 *
 *   u64 uncompressed_total
 *   u32 chunk_count
 *   for i in 0..chunk_count:
 *       u32 raw_size      (= chunk_size for all but possibly the last)
 *       u32 comp_size
 *       u8[comp_size]     LZ4_compress_default output
 *
 * DS4_KVSTORE_CODEC_NONE skips the framing and writes raw bytes.
 * See misc/COMPRESSED_KV_CACHE.md for design rationale.
 * --------------------------------------------------------------------------- */

/* fopencookie (glibc) and funopen (Darwin) take different callback shapes.
 * We hide both behind kv_lz4_fwrap_open() which always returns a FILE *
 * tied to a void cookie plus our write/read/close callbacks.  We never call
 * fseek on the wrapper, so no seek callback is required.  Closing the
 * wrapper does NOT close the underlying outer FILE: the caller owns that. */
#if defined(__APPLE__)
typedef int    kv_lz4_io_ssize_t;
typedef int    kv_lz4_io_size_t;
#else
typedef ssize_t kv_lz4_io_ssize_t;
typedef size_t  kv_lz4_io_size_t;
#endif

typedef kv_lz4_io_ssize_t (*kv_lz4_write_fn)(void *cookie, const char *buf, kv_lz4_io_size_t n);
typedef kv_lz4_io_ssize_t (*kv_lz4_read_fn)(void *cookie, char *buf, kv_lz4_io_size_t n);
typedef int (*kv_lz4_close_fn)(void *cookie);

static FILE *kv_lz4_fwrap_open(void *cookie, const char *mode,
                               kv_lz4_write_fn writefn,
                               kv_lz4_read_fn readfn,
                               kv_lz4_close_fn closefn)
{
    (void)mode;
#if defined(__APPLE__)
    return funopen(cookie, readfn, writefn, NULL, closefn);
#elif defined(__GLIBC__)
    cookie_io_functions_t io = (cookie_io_functions_t){
        .read = readfn,
        .write = writefn,
        .seek = NULL,
        .close = closefn,
    };
    return fopencookie(cookie, mode, io);
#else
    (void)cookie; (void)writefn; (void)readfn; (void)closefn;
    return NULL;
#endif
}

/* One compressor or decompressor job.  raw_size is set by the caller; the
 * worker fills comp_size on the write side or copies raw_size bytes on the
 * read side.  ok stays true unless lz4 returns an error. */
typedef struct {
    const uint8_t *src;
    uint8_t *dst;
    int src_size;
    int dst_capacity;
    int out_size;
    bool ok;
} kv_lz4_job;

static void *kv_lz4_compress_worker(void *arg) {
    kv_lz4_job *j = arg;
    j->out_size = LZ4_compress_default((const char *)j->src,
                                       (char *)j->dst,
                                       j->src_size,
                                       j->dst_capacity);
    j->ok = j->out_size > 0;
    return NULL;
}

static void *kv_lz4_decompress_worker(void *arg) {
    kv_lz4_job *j = arg;
    j->out_size = LZ4_decompress_safe((const char *)j->src,
                                      (char *)j->dst,
                                      j->src_size,
                                      j->dst_capacity);
    j->ok = j->out_size == j->dst_capacity;
    return NULL;
}

/* Run up to n_workers jobs in parallel and join them.  n_workers == 1 runs
 * inline on the calling thread; the pthread fork is just dead overhead for
 * a single job. */
static void kv_lz4_run_batch(kv_lz4_job *jobs, int n_jobs, int n_workers,
                             void *(*worker)(void *)) {
    if (n_jobs <= 0) return;
    if (n_jobs == 1 || n_workers <= 1) {
        for (int i = 0; i < n_jobs; i++) worker(&jobs[i]);
        return;
    }
    pthread_t tids[64];
    if (n_jobs > 64) n_jobs = 64;  /* compile-time sanity; we cap threads at 64 */
    for (int i = 0; i < n_jobs; i++) {
        if (pthread_create(&tids[i], NULL, worker, &jobs[i]) != 0) {
            /* Fall back to running this and the rest inline. */
            for (int j = i; j < n_jobs; j++) worker(&jobs[j]);
            for (int j = 0; j < i; j++) pthread_join(tids[j], NULL);
            return;
        }
    }
    for (int i = 0; i < n_jobs; i++) pthread_join(tids[i], NULL);
}

/* Writer cookie state.  Lives only for the duration of one
 * kv_lz4_writer_open / _close pair.  raw[] holds the per-batch raw chunk
 * scratch; comp[] holds the per-batch compressed output.  current_filled
 * tracks how many bytes are in raw[batch_n] so far. */
typedef struct {
    FILE *out;
    uint32_t chunk_size;
    int n_workers;
    int batch_cap;           /* number of slots allocated; == n_workers */
    uint8_t **raw;
    uint8_t **comp;
    uint32_t *raw_sizes;     /* raw bytes in slot[i]; chunk_size or partial */
    int comp_capacity;       /* per-slot LZ4_compressBound(chunk_size) */
    int batch_n;             /* slots fully filled in this batch */
    uint32_t current_filled; /* bytes in slot[batch_n], the partially-filled head */
    uint64_t uncompressed_total;
    uint64_t on_disk_total;  /* payload bytes written, excluding framing header */
    uint32_t chunk_count;    /* number of chunks written so far */
    bool framing_written;
    bool err;
} kv_lz4_writer;

/* Drain the current batch: compress all filled slots in parallel, then write
 * each (raw_size, comp_size, comp_bytes) record in slot order.  Each slot's
 * raw byte count was recorded in raw_sizes[i] when the slot was filled, so
 * close-time partial chunks travel through here the same way as full chunks
 * from the write callback. */
static void kv_lz4_writer_flush_batch(kv_lz4_writer *w) {
    if (w->err || w->batch_n == 0) return;
    kv_lz4_job jobs[64];
    for (int i = 0; i < w->batch_n; i++) {
        jobs[i] = (kv_lz4_job){
            .src = w->raw[i],
            .dst = w->comp[i],
            .src_size = (int)w->raw_sizes[i],
            .dst_capacity = w->comp_capacity,
            .out_size = 0,
            .ok = false,
        };
    }
    kv_lz4_run_batch(jobs, w->batch_n, w->n_workers, kv_lz4_compress_worker);
    for (int i = 0; i < w->batch_n; i++) {
        if (!jobs[i].ok) { w->err = true; return; }
        uint8_t hdr[8];
        ds4_kvstore_le_put32(hdr, (uint32_t)jobs[i].src_size);
        ds4_kvstore_le_put32(hdr + 4, (uint32_t)jobs[i].out_size);
        if (fwrite(hdr, 1, sizeof(hdr), w->out) != sizeof(hdr) ||
            fwrite(jobs[i].dst, 1, (size_t)jobs[i].out_size, w->out) != (size_t)jobs[i].out_size)
        {
            w->err = true;
            return;
        }
        w->uncompressed_total += (uint64_t)jobs[i].src_size;
        w->on_disk_total += (uint64_t)sizeof(hdr) + (uint64_t)jobs[i].out_size;
        w->chunk_count++;
    }
    w->batch_n = 0;
}

/* Cookie write callback.  Append n bytes into the current raw slot, advance
 * to the next slot when this one fills, run a batch when all slots are full.
 * We do not pre-write the framing header here; we patch the payload region
 * upfront with zero placeholders and rewrite them in kv_lz4_writer_close. */
static kv_lz4_io_ssize_t kv_lz4_writer_write(void *cookie, const char *buf, kv_lz4_io_size_t n_in) {
    kv_lz4_writer *w = cookie;
    const size_t n = (size_t)n_in;
    if (w->err) return -1;
    if (!w->framing_written) {
        /* uncompressed_total (u64) and chunk_count (u32) are patched in close. */
        uint8_t zero[12] = {0};
        if (fwrite(zero, 1, sizeof(zero), w->out) != sizeof(zero)) {
            w->err = true; return -1;
        }
        w->framing_written = true;
    }
    const char *p = buf;
    size_t left = n;
    while (left > 0) {
        if (w->batch_n >= w->batch_cap) {
            kv_lz4_writer_flush_batch(w);
            if (w->err) return -1;
        }
        /* batch_n is the index of the slot we are currently filling.  Filled
         * slots live at indices < batch_n, and their byte counts are in
         * raw_sizes[]. */
        uint8_t *slot = w->raw[w->batch_n];
        uint32_t room = w->chunk_size - w->current_filled;
        uint32_t take = left < room ? (uint32_t)left : room;
        memcpy(slot + w->current_filled, p, take);
        w->current_filled += take;
        p += take;
        left -= take;
        if (w->current_filled == w->chunk_size) {
            w->raw_sizes[w->batch_n] = w->chunk_size;
            w->batch_n++;
            w->current_filled = 0;
            if (w->batch_n == w->batch_cap) {
                kv_lz4_writer_flush_batch(w);
                if (w->err) return -1;
            }
        }
    }
    return (kv_lz4_io_ssize_t)n;
}

/* Cookie close callback.  Flush any partial trailing chunk through the same
 * batch path, then patch the payload-region framing header with the actual
 * uncompressed_total and chunk_count we observed.  Does NOT close w->out;
 * the caller owns the outer file. */
static int kv_lz4_writer_close(void *cookie) {
    kv_lz4_writer *w = cookie;
    /* Promote the partially-filled trailing chunk to a full slot so flush
     * counts it.  The slot's actual raw size is current_filled, not
     * chunk_size; record that in raw_sizes before promoting. */
    if (w->current_filled > 0) {
        w->raw_sizes[w->batch_n] = w->current_filled;
        w->batch_n++;
        w->current_filled = 0;
    }
    kv_lz4_writer_flush_batch(w);

    if (!w->err && w->framing_written) {
        /* The framing header begins 12 bytes before the chunk records, i.e. at
         * (cur - on_disk_total - 12). */
        const off_t after = ftello(w->out);
        const off_t framing_start = after - (off_t)w->on_disk_total - 12;
        if (after < 0 || framing_start < 0 ||
            fseeko(w->out, framing_start, SEEK_SET) != 0)
        {
            w->err = true;
        } else {
            uint8_t hdr[12];
            kv_le_put64(hdr, w->uncompressed_total);
            ds4_kvstore_le_put32(hdr + 8, w->chunk_count);
            if (fwrite(hdr, 1, sizeof(hdr), w->out) != sizeof(hdr) ||
                fseeko(w->out, 0, SEEK_END) != 0)
            {
                w->err = true;
            }
        }
    }
    int rc = w->err ? -1 : 0;
    for (int i = 0; i < w->batch_cap; i++) { free(w->raw[i]); free(w->comp[i]); }
    free(w->raw);
    free(w->comp);
    free(w->raw_sizes);
    free(w);
    return rc;
}

/* Public entry: wrap `out` so subsequent writes are chunked and lz4-encoded.
 * The returned FILE * must be fclose()d; fclose calls our close callback,
 * which patches the framing header in `out`.  `out` remains owned by the
 * caller; the caller measures the on-disk payload size with ftell before
 * opening the wrapper and again after fclose. */
FILE *kv_lz4_writer_open(FILE *out, uint32_t chunk_size, int n_workers) {
    if (n_workers < 1) n_workers = 1;
    if (n_workers > 64) n_workers = 64;
    int comp_bound = LZ4_compressBound((int)chunk_size);
    if (comp_bound <= 0) return NULL;

    kv_lz4_writer *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->out = out;
    w->chunk_size = chunk_size;
    w->n_workers = n_workers;
    w->batch_cap = n_workers;
    w->comp_capacity = comp_bound;
    w->raw = calloc((size_t)n_workers, sizeof(uint8_t *));
    w->comp = calloc((size_t)n_workers, sizeof(uint8_t *));
    w->raw_sizes = calloc((size_t)n_workers, sizeof(uint32_t));
    if (!w->raw || !w->comp || !w->raw_sizes) goto fail;
    for (int i = 0; i < n_workers; i++) {
        w->raw[i] = malloc(chunk_size);
        w->comp[i] = malloc((size_t)comp_bound);
        if (!w->raw[i] || !w->comp[i]) goto fail;
    }
    FILE *fp = kv_lz4_fwrap_open(w, "wb",
                                 kv_lz4_writer_write, NULL, kv_lz4_writer_close);
    if (!fp) goto fail;
    return fp;
fail:
    if (w) {
        if (w->raw) { for (int i = 0; i < n_workers; i++) free(w->raw[i]); free(w->raw); }
        if (w->comp) { for (int i = 0; i < n_workers; i++) free(w->comp[i]); free(w->comp); }
        free(w->raw_sizes);
        free(w);
    }
    return NULL;
}

/* Reader cookie state.  We read one chunk at a time, decompress it, and
 * hand bytes out through kv_lz4_reader_read until the slot is drained.
 * The first call lazily reads the 12-byte framing header so kv_lz4_reader_open
 * can be called before payload bytes are known. */
typedef struct {
    FILE *in;
    uint32_t chunk_size;
    int n_workers;
    int batch_cap;
    uint8_t **raw;          /* per-slot decompressed scratch */
    uint8_t **comp;         /* per-slot compressed input scratch */
    int comp_capacity;
    uint32_t *raw_sizes;    /* per-slot uncompressed size for the current batch */
    uint32_t *comp_sizes;   /* per-slot compressed size for the current batch */

    uint64_t payload_bytes; /* on-disk payload budget remaining (compressed) */
    uint64_t uncompressed_total;
    uint32_t chunk_count;
    uint32_t chunks_consumed;
    bool framing_read;

    int batch_n;            /* slots filled by the most recent read-batch */
    int batch_pos;          /* next slot to hand out from the current batch */
    uint32_t pos_in_slot;   /* offset within raw[batch_pos] */
    bool err;
} kv_lz4_reader;

/* Read the 12-byte framing header (uncompressed_total, chunk_count) at the
 * start of the payload region.  Called lazily on the first read.
 * Validates that chunk_count matches ceil(uncompressed_total / chunk_size) so
 * a tampered or truncated framing header is rejected before we allocate. */
static bool kv_lz4_reader_read_framing(kv_lz4_reader *r) {
    uint8_t hdr[12];
    if (r->payload_bytes < sizeof(hdr)) { r->err = true; return false; }
    if (fread(hdr, 1, sizeof(hdr), r->in) != sizeof(hdr)) { r->err = true; return false; }
    r->uncompressed_total = kv_le_get64(hdr);
    r->chunk_count = ds4_kvstore_le_get32(hdr + 8);
    r->payload_bytes -= sizeof(hdr);
    const uint64_t q = r->uncompressed_total / (uint64_t)r->chunk_size;
    const uint64_t rem = r->uncompressed_total % (uint64_t)r->chunk_size;
    const uint64_t expected = q + (rem ? 1u : 0u);
    if ((uint64_t)r->chunk_count != expected) { r->err = true; return false; }
    r->framing_read = true;
    return true;
}

/* Read the next batch of up to batch_cap chunks, decompress them in parallel,
 * leave them in raw[0..batch_n) for kv_lz4_reader_read to hand out in order. */
static void kv_lz4_reader_fill_batch(kv_lz4_reader *r) {
    r->batch_n = 0;
    r->batch_pos = 0;
    r->pos_in_slot = 0;
    int n = 0;
    while (n < r->batch_cap && r->chunks_consumed + (uint32_t)n < r->chunk_count) {
        uint8_t lens[8];
        if (r->payload_bytes < sizeof(lens) ||
            fread(lens, 1, sizeof(lens), r->in) != sizeof(lens))
        {
            r->err = true;
            return;
        }
        uint32_t raw = ds4_kvstore_le_get32(lens);
        uint32_t comp = ds4_kvstore_le_get32(lens + 4);
        if (raw == 0 || raw > r->chunk_size || comp == 0 ||
            comp > (uint32_t)r->comp_capacity ||
            r->payload_bytes < sizeof(lens) + (uint64_t)comp ||
            fread(r->comp[n], 1, comp, r->in) != comp)
        {
            r->err = true;
            return;
        }
        r->payload_bytes -= sizeof(lens) + (uint64_t)comp;
        r->raw_sizes[n] = raw;
        r->comp_sizes[n] = comp;
        n++;
    }
    /* After the final chunk we must have consumed exactly the declared
     * compressed payload region.  Any leftover bytes mean the file's
     * payload_bytes header was lying about its size, or chunk records do
     * not pack tightly as the writer claims. */
    if (r->chunks_consumed + (uint32_t)n == r->chunk_count && r->payload_bytes != 0) {
        r->err = true;
        return;
    }
    if (n == 0) return;
    kv_lz4_job jobs[64];
    for (int i = 0; i < n; i++) {
        jobs[i] = (kv_lz4_job){
            .src = r->comp[i],
            .dst = r->raw[i],
            .src_size = (int)r->comp_sizes[i],
            .dst_capacity = (int)r->raw_sizes[i],
            .out_size = 0,
            .ok = false,
        };
    }
    kv_lz4_run_batch(jobs, n, r->n_workers, kv_lz4_decompress_worker);
    for (int i = 0; i < n; i++) {
        if (!jobs[i].ok) { r->err = true; return; }
    }
    r->batch_n = n;
}

/* Cookie read callback.  Hand out bytes from the current decompressed batch;
 * pull a new batch when the current one is drained. */
static kv_lz4_io_ssize_t kv_lz4_reader_read(void *cookie, char *buf, kv_lz4_io_size_t n_in) {
    kv_lz4_reader *r = cookie;
    if (r->err) return -1;
    if (!r->framing_read && !kv_lz4_reader_read_framing(r)) return -1;
    const size_t n = (size_t)n_in;
    size_t produced = 0;
    while (produced < n) {
        if (r->chunks_consumed >= r->chunk_count && r->batch_pos >= r->batch_n) break;
        if (r->batch_pos >= r->batch_n) {
            kv_lz4_reader_fill_batch(r);
            if (r->err) return -1;
            if (r->batch_n == 0) break;
        }
        uint32_t slot_size = r->raw_sizes[r->batch_pos];
        uint32_t avail = slot_size - r->pos_in_slot;
        size_t take = n - produced < avail ? n - produced : avail;
        memcpy(buf + produced, r->raw[r->batch_pos] + r->pos_in_slot, take);
        produced += take;
        r->pos_in_slot += (uint32_t)take;
        if (r->pos_in_slot == slot_size) {
            r->pos_in_slot = 0;
            r->batch_pos++;
            r->chunks_consumed++;
        }
    }
    return (kv_lz4_io_ssize_t)produced;
}

static int kv_lz4_reader_close(void *cookie) {
    kv_lz4_reader *r = cookie;
    for (int i = 0; i < r->batch_cap; i++) { free(r->raw[i]); free(r->comp[i]); }
    free(r->raw);
    free(r->comp);
    free(r->raw_sizes);
    free(r->comp_sizes);
    free(r);
    return 0;
}

/* Public entry: wrap `in` so subsequent reads transparently decompress.
 * payload_bytes is the on-disk payload byte count from the KVC header
 * (covers framing + all chunks).  chunk_size is also from the header.
 * If uncompressed_total_out != NULL, the framing header is read eagerly
 * so the caller learns the uncompressed payload size before reading.
 * The returned FILE * must be fclose()d when the caller is done. */
FILE *kv_lz4_reader_open(FILE *in, uint64_t payload_bytes,
                                uint32_t chunk_size, int n_workers,
                                uint64_t *uncompressed_total_out) {
    if (n_workers < 1) n_workers = 1;
    if (n_workers > 64) n_workers = 64;
    if (chunk_size == 0 || chunk_size > DS4_KVSTORE_MAX_CHUNK_BYTES) return NULL;
    int comp_bound = LZ4_compressBound((int)chunk_size);
    if (comp_bound <= 0) return NULL;

    kv_lz4_reader *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->in = in;
    r->chunk_size = chunk_size;
    r->n_workers = n_workers;
    r->batch_cap = n_workers;
    r->comp_capacity = comp_bound;
    r->payload_bytes = payload_bytes;
    r->raw = calloc((size_t)n_workers, sizeof(uint8_t *));
    r->comp = calloc((size_t)n_workers, sizeof(uint8_t *));
    r->raw_sizes = calloc((size_t)n_workers, sizeof(uint32_t));
    r->comp_sizes = calloc((size_t)n_workers, sizeof(uint32_t));
    if (!r->raw || !r->comp || !r->raw_sizes || !r->comp_sizes) goto fail;
    for (int i = 0; i < n_workers; i++) {
        r->raw[i] = malloc(chunk_size);
        r->comp[i] = malloc((size_t)comp_bound);
        if (!r->raw[i] || !r->comp[i]) goto fail;
    }
    if (uncompressed_total_out) {
        if (!kv_lz4_reader_read_framing(r)) goto fail;
        *uncompressed_total_out = r->uncompressed_total;
    }
    FILE *fp = kv_lz4_fwrap_open(r, "rb",
                                 NULL, kv_lz4_reader_read, kv_lz4_reader_close);
    if (!fp) goto fail;
    return fp;
fail:
    if (r) {
        if (r->raw) { for (int i = 0; i < n_workers; i++) free(r->raw[i]); free(r->raw); }
        if (r->comp) { for (int i = 0; i < n_workers; i++) free(r->comp[i]); free(r->comp); }
        free(r->raw_sizes);
        free(r->comp_sizes);
        free(r);
    }
    return NULL;
}

ds4_kvstore_options ds4_kvstore_default_options(void) {
    return (ds4_kvstore_options){
        .min_tokens = KV_CACHE_DEFAULT_MIN_TOKENS,
        .cold_max_tokens = KV_CACHE_DEFAULT_COLD_MAX_TOKENS,
        .continued_interval_tokens = KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS,
        .boundary_trim_tokens = KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS,
        .boundary_align_tokens = KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS,
        .compression_threads = kv_cache_default_compression_threads(),
    };
}

uint8_t ds4_kvstore_reason_code(const char *reason) {
    if (!reason) return DS4_KVSTORE_REASON_UNKNOWN;
    if (!strcmp(reason, "cold")) return DS4_KVSTORE_REASON_COLD;
    if (!strcmp(reason, "continued")) return DS4_KVSTORE_REASON_CONTINUED;
    if (!strcmp(reason, "evict")) return DS4_KVSTORE_REASON_EVICT;
    if (!strcmp(reason, "shutdown")) return DS4_KVSTORE_REASON_SHUTDOWN;
    if (!strcmp(reason, "agent-system")) return DS4_KVSTORE_REASON_AGENT_SYSTEM;
    if (!strcmp(reason, "agent-session")) return DS4_KVSTORE_REASON_AGENT_SESSION;
    return DS4_KVSTORE_REASON_UNKNOWN;
}

const char *ds4_kvstore_key_kind(uint8_t ext_flags) {
    if (ext_flags & DS4_KVSTORE_EXT_RESPONSES_VISIBLE) return "responses-visible";
    if (ext_flags & DS4_KVSTORE_EXT_THINKING_VISIBLE) return "thinking-visible";
    return "token-text";
}

void ds4_kvstore_le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void kv_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

uint32_t ds4_kvstore_le_get32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t kv_le_get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

typedef struct {
    uint32_t h[5];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} sha1_ctx;

static uint32_t rol32(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void sha1_transform(sha1_ctx *c, const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 80; i++)
        w[i] = rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = c->h[0], b = c->h[1], d = c->h[3], e = c->h[4];
    uint32_t cc = c->h[2];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40) {
            f = b ^ cc ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60) {
            f = (b & cc) | (b & d) | (cc & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ cc ^ d;
            k = 0xca62c1d6u;
        }
        uint32_t tmp = rol32(a, 5) + f + e + k + w[i];
        e = d;
        d = cc;
        cc = rol32(b, 30);
        b = a;
        a = tmp;
    }
    c->h[0] += a;
    c->h[1] += b;
    c->h[2] += cc;
    c->h[3] += d;
    c->h[4] += e;
}

static void sha1_init(sha1_ctx *c) {
    c->h[0] = 0x67452301u;
    c->h[1] = 0xefcdab89u;
    c->h[2] = 0x98badcfeu;
    c->h[3] = 0x10325476u;
    c->h[4] = 0xc3d2e1f0u;
    c->bytes = 0;
    c->used = 0;
}

static void sha1_update(sha1_ctx *c, const void *ptr, size_t len) {
    const uint8_t *p = ptr;
    c->bytes += len;
    while (len != 0) {
        size_t n = 64 - c->used;
        if (n > len) n = len;
        memcpy(c->block + c->used, p, n);
        c->used += n;
        p += n;
        len -= n;
        if (c->used == 64) {
            sha1_transform(c, c->block);
            c->used = 0;
        }
    }
}

static void sha1_final(sha1_ctx *c, uint8_t out[20]) {
    uint64_t bits = c->bytes * 8;
    uint8_t one = 0x80;
    uint8_t zero = 0;
    sha1_update(c, &one, 1);
    while (c->used != 56) sha1_update(c, &zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[7 - i] = (uint8_t)(bits >> (8 * i));
    sha1_update(c, len, sizeof(len));
    for (int i = 0; i < 5; i++) {
        out[i * 4] = (uint8_t)(c->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)c->h[i];
    }
}

static void hex20(const uint8_t in[20], char out[41]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 20; i++) {
        out[i * 2] = hex[in[i] >> 4];
        out[i * 2 + 1] = hex[in[i] & 15];
    }
    out[40] = '\0';
}

void ds4_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    sha1_ctx c;
    sha1_init(&c);
    sha1_update(&c, ptr, len);
    uint8_t digest[20];
    sha1_final(&c, digest);
    hex20(digest, out);
}

bool ds4_kvstore_sha_hex_name(const char *name, char sha[41]) {
    if (strlen(name) != 43 || strcmp(name + 40, ".kv")) return false;
    for (int i = 0; i < 40; i++) {
        if (!isxdigit((unsigned char)name[i])) return false;
        sha[i] = (char)tolower((unsigned char)name[i]);
    }
    sha[40] = '\0';
    return true;
}

char *ds4_kvstore_path_join(const char *dir, const char *name) {
    kv_buf b = {0};
    kv_buf_puts(&b, dir);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') kv_buf_putc(&b, '/');
    kv_buf_puts(&b, name);
    return kv_buf_take(&b);
}

char *ds4_kvstore_path_for_sha(ds4_kvstore *kc, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return ds4_kvstore_path_join(kc->dir, name);
}

static bool kv_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = kv_xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

void ds4_kvstore_entry_free(ds4_kvstore_entry *e) {
    free(e->path);
    memset(e, 0, sizeof(*e));
}

void ds4_kvstore_clear(ds4_kvstore *kc) {
    for (int i = 0; i < kc->len; i++) ds4_kvstore_entry_free(&kc->entry[i]);
    free(kc->entry);
    kc->entry = NULL;
    kc->len = 0;
    kc->cap = 0;
}

static void kv_cache_push(ds4_kvstore *kc, ds4_kvstore_entry e) {
    if (kc->len == kc->cap) {
        kc->cap = kc->cap ? kc->cap * 2 : 16;
        kc->entry = kv_xrealloc(kc->entry, (size_t)kc->cap * sizeof(kc->entry[0]));
    }
    kc->entry[kc->len++] = e;
}

void ds4_kvstore_fill_header(uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                             uint8_t quant_bits, uint8_t reason,
                             uint8_t ext_flags, uint8_t codec,
                             uint32_t chunk_size, uint32_t tokens,
                             uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes) {
    memset(h, 0, DS4_KVSTORE_FIXED_HEADER);
    h[0] = KV_CACHE_MAGIC0;
    h[1] = KV_CACHE_MAGIC1;
    h[2] = KV_CACHE_MAGIC2;
    h[3] = KV_CACHE_VERSION;
    h[4] = quant_bits;
    h[5] = reason;
    h[6] = ext_flags;
    h[7] = codec;
    ds4_kvstore_le_put32(h + 8, tokens);
    ds4_kvstore_le_put32(h + 12, hits);
    ds4_kvstore_le_put32(h + 16, ctx_size);
    ds4_kvstore_le_put32(h + 20, chunk_size);
    kv_le_put64(h + 24, created_at);
    kv_le_put64(h + 32, last_used);
    kv_le_put64(h + 40, payload_bytes);
}

bool ds4_kvstore_read_header(FILE *fp, ds4_kvstore_entry *e,
                             uint32_t *text_bytes) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    if (fread(h, 1, sizeof(h), fp) != sizeof(h)) return false;
    if (h[0] != KV_CACHE_MAGIC0 || h[1] != KV_CACHE_MAGIC1 ||
        h[2] != KV_CACHE_MAGIC2) return false;
    /* Accept v1 (uncompressed; codec/chunk_size were reserved) and v2 (codec
     * byte at offset 7, chunk_size at offset 20).  For v1 we synthesise
     * codec=NONE and chunk_size=0 so the rest of the load path can branch
     * uniformly on entry->codec without caring which version produced it. */
    if (h[3] != 1u && h[3] != KV_CACHE_VERSION) return false;
    e->quant_bits = h[4];
    e->reason = h[5] <= DS4_KVSTORE_REASON_AGENT_SESSION ? h[5] :
                DS4_KVSTORE_REASON_UNKNOWN;
    e->ext_flags = h[6];
    if (h[3] == KV_CACHE_VERSION) {
        e->codec = h[7];
        e->chunk_size = ds4_kvstore_le_get32(h + 20);
    } else {
        e->codec = DS4_KVSTORE_CODEC_NONE;
        e->chunk_size = 0;
    }
    e->tokens = ds4_kvstore_le_get32(h + 8);
    e->hits = ds4_kvstore_le_get32(h + 12);
    e->ctx_size = ds4_kvstore_le_get32(h + 16);
    e->created_at = kv_le_get64(h + 24);
    e->last_used = kv_le_get64(h + 32);
    e->payload_bytes = kv_le_get64(h + 40);
    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) return false;
    *text_bytes = ds4_kvstore_le_get32(tb);
    e->text_bytes = *text_bytes;
    return e->tokens != 0 && (e->quant_bits == 2 || e->quant_bits == 4) &&
           (e->codec == DS4_KVSTORE_CODEC_NONE ||
            (e->codec == DS4_KVSTORE_CODEC_LZ4 &&
             e->chunk_size > 0 &&
             e->chunk_size <= DS4_KVSTORE_MAX_CHUNK_BYTES));
}

bool ds4_kvstore_read_entry_file(const char *path, const char sha[41],
                                 ds4_kvstore_entry *out) {
    struct stat st;
    if (stat(path, &st) != 0 ||
        st.st_size < (off_t)(DS4_KVSTORE_FIXED_HEADER + 4))
        return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    ds4_kvstore_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &e, &text_bytes);
    fclose(fp);
    if (!ok) return false;
    const uint64_t fixed = DS4_KVSTORE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < (uint64_t)text_bytes ||
        UINT64_MAX - fixed - (uint64_t)text_bytes < e.payload_bytes)
        return false;
    const uint64_t expected = fixed + (uint64_t)text_bytes + e.payload_bytes;
    if ((uint64_t)st.st_size < expected) return false;
    memcpy(e.sha, sha, 41);
    e.path = kv_xstrdup(path);
    e.file_size = (uint64_t)st.st_size;
    *out = e;
    return true;
}

static void kv_cache_refresh(ds4_kvstore *kc) {
    if (!kc->enabled) return;
    ds4_kvstore_clear(kc);
    DIR *d = opendir(kc->dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        char *path = ds4_kvstore_path_join(kc->dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (ds4_kvstore_read_entry_file(path, sha, &e)) kv_cache_push(kc, e);
        free(path);
    }
    closedir(d);
}

bool ds4_kvstore_touch_file(const char *path, uint32_t hits) {
    FILE *fp = fopen(path, "r+b");
    if (!fp) return false;
    ds4_kvstore_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &e, &text_bytes);
    if (ok) {
        uint8_t h[DS4_KVSTORE_FIXED_HEADER];
        uint64_t now = (uint64_t)time(NULL);
        ds4_kvstore_fill_header(h, e.quant_bits, e.reason, e.ext_flags,
                                e.codec, e.chunk_size,
                                e.tokens, hits, e.ctx_size,
                                e.created_at, now, e.payload_bytes);
        ok = fseek(fp, 0, SEEK_SET) == 0 &&
             fwrite(h, 1, sizeof(h), fp) == sizeof(h);
    }
    fclose(fp);
    return ok;
}

double ds4_kvstore_entry_eviction_score(const ds4_kvstore_entry *e,
                                        const ds4_tokens *live,
                                        const char *protected_sha,
                                        uint64_t now) {
    if (!e || e->file_size == 0) return 0.0;
    (void)live;
    if (protected_sha && !strcmp(e->sha, protected_sha)) {
        /* The store path calls eviction immediately after renaming the new
         * checkpoint into place. Without a protected score, a full cache can
         * choose that brand-new zero-hit file as the cheapest deletion. */
        return DBL_MAX;
    }
    double effective_hits = (double)e->hits;
    uint64_t used_at = e->last_used ? e->last_used : e->created_at;
    if (used_at == 0) {
        effective_hits = 0.0;
    } else if (now > used_at) {
        double elapsed = (double)(now - used_at);
        effective_hits *= exp2(-elapsed / (double)DS4_KVSTORE_HIT_HALF_LIFE_SECONDS);
        if (effective_hits < KV_CACHE_MIN_EFFECTIVE_HITS) effective_hits = 0.0;
    }
    return (effective_hits + 1.0) * (double)e->tokens / (double)e->file_size;
}

void ds4_kvstore_evict(ds4_kvstore *kc, const ds4_tokens *live,
                       const char *protected_sha) {
    if (!kc->enabled || kc->budget_bytes == 0) return;
    kv_cache_refresh(kc);
    const uint64_t now = (uint64_t)time(NULL);
    uint64_t total = 0;
    for (int i = 0; i < kc->len; i++) total += kc->entry[i].file_size;
    if (protected_sha) {
        uint64_t protected_size = 0;
        for (int i = 0; i < kc->len; i++) {
            if (!strcmp(kc->entry[i].sha, protected_sha)) {
                protected_size = kc->entry[i].file_size;
                break;
            }
        }
        if (protected_size > kc->budget_bytes) protected_sha = NULL;
    }
    while (total > kc->budget_bytes && kc->len > 0) {
        int victim = 0;
        double victim_score =
            ds4_kvstore_entry_eviction_score(&kc->entry[0], live, protected_sha, now);
        for (int i = 1; i < kc->len; i++) {
            double score =
                ds4_kvstore_entry_eviction_score(&kc->entry[i], live, protected_sha, now);
            if (score < victim_score ||
                (score == victim_score &&
                 kc->entry[i].last_used < kc->entry[victim].last_used))
            {
                victim = i;
                victim_score = score;
            }
        }
        ds4_kvstore_entry e = kc->entry[victim];
        if (unlink(e.path) == 0) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache evicted reason=disk-cache-full tokens=%u hits=%u size=%.2f MiB file=%s",
                    kv_log_name(kc),
                    e.tokens,
                    e.hits,
                    (double)e.file_size / (1024.0 * 1024.0),
                    e.path ? e.path : "?");
            if (total >= e.file_size) total -= e.file_size;
            else total = 0;
        } else {
            total = 0;
        }
        ds4_kvstore_entry_free(&e);
        memmove(kc->entry + victim, kc->entry + victim + 1,
                (size_t)(kc->len - victim - 1) * sizeof(kc->entry[0]));
        kc->len--;
    }
}

bool ds4_kvstore_open(ds4_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, ds4_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg),
                      void *log_ud) {
    memset(kc, 0, sizeof(*kc));
    if (!dir) return false;
    kc->log_name = log_name;
    kc->log = log;
    kc->log_ud = log_ud;
    if (!kv_mkdir_p(dir)) {
        kv_logf(kc, DS4_KVSTORE_LOG_DEFAULT,
                "%s: failed to create KV cache directory %s: %s",
                kv_log_name(kc), dir, strerror(errno));
        return false;
    }
    kc->enabled = true;
    kc->dir = kv_xstrdup(dir);
    if (budget_mb == 0) budget_mb = DS4_KVSTORE_DEFAULT_MB;
    kc->budget_bytes = budget_mb * 1024ull * 1024ull;
    kc->reject_different_quant = reject_different_quant;
    kc->opt = opt;
    ds4_kvstore_evict(kc, NULL, NULL);
    kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
            "%s: KV disk cache %s (budget=%llu MiB, cross-quant=%s, min=%d, cold_max=%d, continued=%d, trim=%d, align=%d, hit_half_life=%llus, codec=%s threads=%d chunk=%uK)",
            kv_log_name(kc),
            kc->dir,
            (unsigned long long)(kc->budget_bytes / (1024ull * 1024ull)),
            reject_different_quant ? "reject" : "accept",
            kc->opt.min_tokens,
            kc->opt.cold_max_tokens,
            kc->opt.continued_interval_tokens,
            kc->opt.boundary_trim_tokens,
            kc->opt.boundary_align_tokens,
            (unsigned long long)DS4_KVSTORE_HIT_HALF_LIFE_SECONDS,
            kc->opt.compression_threads > 0 ? "lz4" : "none",
            kc->opt.compression_threads,
            (unsigned int)(DS4_KVSTORE_DEFAULT_CHUNK_BYTES / 1024u));
    return true;
}

void ds4_kvstore_close(ds4_kvstore *kc) {
    ds4_kvstore_clear(kc);
    free(kc->dir);
    memset(kc, 0, sizeof(*kc));
}

char *ds4_kvstore_render_tokens_text(ds4_engine *engine,
                                     const ds4_tokens *tokens,
                                     size_t *out_len) {
    kv_buf b = {0};
    for (int i = 0; i < tokens->len; i++) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, tokens->v[i], &len);
        kv_buf_append(&b, piece, len);
        free(piece);
    }
    if (out_len) *out_len = b.len;
    return kv_buf_take(&b);
}

bool ds4_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len) {
    return prefix_len <= text_len &&
           (prefix_len == 0 || memcmp(text, prefix, prefix_len) == 0);
}

void ds4_kvstore_tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n) {
    dst->len = 0;
    if (!src) return;
    if (n > src->len) n = src->len;
    for (int i = 0; i < n; i++) ds4_tokens_push(dst, src->v[i]);
}

static void tokens_append(ds4_tokens *dst, const ds4_tokens *src) {
    if (!dst || !src) return;
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}

void ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out) {
    ds4_tokens_copy(out, exact_prefix);

    ds4_tokens suffix = {0};
    /* The suffix may start with DS4 chat markers such as <｜User｜> or
     * </think>, so use the rendered-chat tokenizer, not plain text BPE. */
    ds4_tokenize_rendered_chat(engine, suffix_text ? suffix_text : "", &suffix);
    tokens_append(out, &suffix);
    ds4_tokens_free(&suffix);
}

int ds4_kvstore_store_len(const ds4_kvstore *kc, int tokens) {
    const int trim = kc->opt.boundary_trim_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (tokens > kc->opt.min_tokens + trim) {
        int stable = tokens - trim;
        if (align > 0) stable -= stable % align;
        if (stable >= kc->opt.min_tokens) return stable;
    }
    return tokens;
}

int ds4_kvstore_chat_anchor_pos(const ds4_kvstore *kc,
                                const ds4_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id) {
    if (!prompt || user_token_id < 0 || assistant_token_id < 0) return -1;

    /* Cold checkpoints maximize reuse across independent agent sessions.  The
     * stable rendered chat prefix is everything before the user message that
     * asks this specific task.  Some clients put stable user-role scaffolding
     * first, so use the last user marker before the first assistant marker. */
    int last_user = -1;
    for (int i = 0; i < prompt->len; i++) {
        const int token = prompt->v[i];
        if (token == assistant_token_id) break;
        if (token == user_token_id) last_user = i;
    }
    return last_user >= kc->opt.min_tokens ? last_user : -1;
}

static int kv_cache_continued_step(const ds4_kvstore *kc) {
    if (!kc->enabled || kc->opt.continued_interval_tokens <= 0) return 0;
    int step = kc->opt.continued_interval_tokens;
    const int align = kc->opt.boundary_align_tokens;
    if (align > 0) {
        step = ((step + align - 1) / align) * align;
        if (step <= 0) step = align;
    }
    return step;
}

int ds4_kvstore_continued_store_target(const ds4_kvstore *kc, int live_tokens) {
    const int step = kv_cache_continued_step(kc);
    if (step <= 0) return 0;
    if (live_tokens < kc->opt.min_tokens) return 0;
    if (live_tokens % step != 0) return 0;
    if (live_tokens <= kc->continued_last_store_tokens) return 0;
    return live_tokens;
}

void ds4_kvstore_note_store(ds4_kvstore *kc, int tokens) {
    if (tokens > kc->continued_last_store_tokens) {
        kc->continued_last_store_tokens = tokens;
    }
}

int ds4_kvstore_suppress_continued_store(ds4_kvstore *kc, int tokens) {
    if (ds4_kvstore_continued_store_target(kc, tokens) != tokens) return -1;
    int old = kc->continued_last_store_tokens;
    ds4_kvstore_note_store(kc, tokens);
    return old;
}

void ds4_kvstore_restore_suppressed_continued(ds4_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens) {
    if (old_tokens >= 0 && kc->continued_last_store_tokens == suppressed_tokens) {
        kc->continued_last_store_tokens = old_tokens;
    }
}

static bool kv_cache_file_size_bytes(uint64_t text_bytes,
                                     uint64_t payload_bytes,
                                     uint64_t trailer_bytes,
                                     uint64_t *file_bytes) {
    const uint64_t fixed = DS4_KVSTORE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < text_bytes ||
        UINT64_MAX - fixed - text_bytes < payload_bytes ||
        UINT64_MAX - fixed - text_bytes - payload_bytes < trailer_bytes)
        return false;
    if (file_bytes) *file_bytes = fixed + text_bytes + payload_bytes + trailer_bytes;
    return true;
}

static bool kv_cache_budget_required(uint64_t file_bytes,
                                     uint64_t *required_bytes) {
    /* The serialized size is deterministic for one snapshot, including the
     * optional trailer.  Reserve 1% headroom so filesystem/accounting surprises
     * cannot produce a file that is immediately removed by the budget pass. */
    uint64_t slack = file_bytes / 100u;
    if (file_bytes % 100u) slack++;
    if (UINT64_MAX - file_bytes < slack) return false;
    if (required_bytes) *required_bytes = file_bytes + slack;
    return true;
}

bool ds4_kvstore_file_size_fits(const ds4_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out) {
    uint64_t file_bytes = 0;
    if (!kv_cache_file_size_bytes(text_bytes, payload_bytes, trailer_bytes,
                                  &file_bytes))
        return false;
    if (file_bytes_out) *file_bytes_out = file_bytes;
    if (!kc || kc->budget_bytes == 0) return true;
    uint64_t required = 0;
    if (!kv_cache_budget_required(file_bytes, &required)) return false;
    if (required_bytes_out) *required_bytes_out = required;
    return required <= kc->budget_bytes;
}

static bool kv_cache_file_text_matches(const char *path, const char sha[41],
                                       const char *text, size_t text_len) {
    if (text_len > UINT32_MAX) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;

    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
              text_bytes == (uint32_t)text_len;
    char *stored = NULL;
    if (ok) {
        stored = kv_xmalloc((size_t)text_bytes + 1);
        ok = fread(stored, 1, text_bytes, fp) == text_bytes;
    }
    fclose(fp);
    if (!ok) {
        free(stored);
        return false;
    }

    char stored_sha[41];
    ds4_kvstore_sha1_bytes_hex(stored, text_bytes, stored_sha);
    ok = !strcmp(stored_sha, sha) &&
         (text_len == 0 || memcmp(stored, text, text_len) == 0);
    free(stored);
    return ok;
}

static bool kv_cache_existing_compatible(ds4_kvstore *kc, const char *path,
                                         const char sha[41],
                                         const char *text, size_t text_len,
                                         int quant_bits, int ctx_size) {
    if (access(path, F_OK) != 0) return false;
    ds4_kvstore_entry e = {0};
    if (!ds4_kvstore_read_entry_file(path, sha, &e)) return false;
    bool compatible = (!kc->reject_different_quant ||
                       e.quant_bits == (uint8_t)quant_bits) &&
                      e.ctx_size <= (uint32_t)ctx_size &&
                      kv_cache_file_text_matches(path, sha, text, text_len);
    ds4_kvstore_entry_free(&e);
    if (!compatible) {
        if (unlink(path) == 0) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache replaced incompatible file %s",
                    kv_log_name(kc), path);
        }
        return false;
    }
    return true;
}

static bool kv_trailer_serialized_size(const ds4_kvstore_trailer_hooks *hooks,
                                       const char *text,
                                       uint64_t *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!hooks || !hooks->serialized_size) return true;
    return hooks->serialized_size(hooks->ud, text, bytes_out);
}

static bool kv_trailer_write(const ds4_kvstore_trailer_hooks *hooks,
                             FILE *fp, const char *text,
                             uint64_t *written_bytes) {
    if (written_bytes) *written_bytes = 0;
    if (!hooks || !hooks->write) return true;
    return hooks->write(hooks->ud, fp, text, written_bytes);
}

static void kv_cache_rewrite_trailer(ds4_kvstore *kc, const char *path,
                                     const char *text,
                                     const ds4_kvstore_trailer_hooks *hooks) {
    uint64_t trailer_est = 0;
    if (!hooks || !hooks->write || !hooks->serialized_size ||
        !kv_trailer_serialized_size(hooks, text, &trailer_est) ||
        trailer_est == 0)
    {
        return;
    }
    FILE *fp = fopen(path, "r+b");
    if (!fp) return;
    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    uint64_t end = DS4_KVSTORE_FIXED_HEADER + 4ull +
                   (uint64_t)text_bytes + hdr.payload_bytes;
    if (ok && end <= (uint64_t)INT64_MAX &&
        fseeko(fp, (off_t)end, SEEK_SET) == 0 &&
        ftruncate(fileno(fp), (off_t)end) == 0)
    {
        uint64_t ignored = 0;
        ok = kv_trailer_write(hooks, fp, text, &ignored) && fflush(fp) == 0;
        if (ok && ignored > 0) {
            uint8_t h[DS4_KVSTORE_FIXED_HEADER];
            uint64_t now = (uint64_t)time(NULL);
            ds4_kvstore_fill_header(h, hdr.quant_bits, hdr.reason,
                                    (uint8_t)(hdr.ext_flags | hooks->ext_flag),
                                    hdr.codec, hdr.chunk_size,
                                    hdr.tokens, hdr.hits, hdr.ctx_size,
                                    hdr.created_at, now, hdr.payload_bytes);
            ok = fseeko(fp, 0, SEEK_SET) == 0 &&
                 fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
                 fflush(fp) == 0;
        }
    }
    fclose(fp);
    (void)kc;
    (void)ok;
}

bool ds4_kvstore_store_live_prefix_text(ds4_kvstore *kc,
                                        ds4_engine *engine,
                                        ds4_session *session,
                                        const ds4_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const ds4_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len) {
    if (!kc->enabled) return false;
    if (!tokens || store_len < kc->opt.min_tokens) return false;
    const int original_len = tokens->len;

    ds4_tokens store_tokens = {0};
    ds4_kvstore_tokens_copy_prefix(&store_tokens, tokens, store_len);

    const int quant_bits = ds4_engine_routed_quant_bits(engine);
    if (quant_bits != 2 && quant_bits != 4) {
        ds4_tokens_free(&store_tokens);
        return false;
    }

    char save_err[160] = {0};
    const ds4_tokens *live_tokens = ds4_session_tokens(session);
    if (!live_tokens ||
        live_tokens->len != store_tokens.len ||
        !ds4_tokens_starts_with(live_tokens, &store_tokens))
    {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because live checkpoint is at %d",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                live_tokens ? live_tokens->len : -1);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    uint64_t payload_bytes = ds4_session_payload_bytes(session);
    if (payload_bytes == 0) {
        ds4_tokens_free(&store_tokens);
        return false;
    }

    size_t text_len = 0;
    char *text = NULL;
    const bool text_override = cache_text_override && cache_text_override[0];
    if (text_override) {
        text = kv_xstrdup(cache_text_override);
        text_len = strlen(text);
    } else {
        text = ds4_kvstore_render_tokens_text(engine, &store_tokens, &text_len);
    }
    if (text_len > UINT32_MAX) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d because rendered text is too large",
                kv_log_name(kc), store_tokens.len);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    uint64_t trailer_est_bytes = 0;
    if (!kv_trailer_serialized_size(hooks, text, &trailer_est_bytes)) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because tool map size overflowed",
                kv_log_name(kc), store_tokens.len, reason);
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }
    uint64_t est_file_bytes = 0, est_required_bytes = 0;
    if (!ds4_kvstore_file_size_fits(kc, (uint64_t)text_len, payload_bytes,
                                    trailer_est_bytes,
                                    &est_file_bytes, &est_required_bytes)) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache skipped tokens=%d reason=%s because estimated file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB",
                kv_log_name(kc),
                store_tokens.len,
                reason,
                (double)est_file_bytes / (1024.0 * 1024.0),
                (double)est_required_bytes / (1024.0 * 1024.0),
                (double)kc->budget_bytes / (1024.0 * 1024.0));
        free(text);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    char *path = ds4_kvstore_path_for_sha(kc, sha);

    if (kv_cache_existing_compatible(kc, path, sha, text, text_len,
                                     quant_bits, ds4_session_ctx(session))) {
        kv_cache_rewrite_trailer(kc, path, text, hooks);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return true;
    }

    kv_buf tmpb = {0};
    kv_buf_printf(&tmpb, "%s.tmp.%ld", path, (long)getpid());
    char *tmp = kv_buf_take(&tmpb);
    const double save_t0 = kv_now_sec();
    FILE *fp = fopen(tmp, "wb");
    if (!fp) {
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache failed to create %s: %s save=%.1f ms",
                kv_log_name(kc), tmp, strerror(errno),
                (kv_now_sec() - save_t0) * 1000.0);
        free(tmp);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return false;
    }

    const uint64_t now = (uint64_t)time(NULL);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    uint8_t ext_flags = trailer_est_bytes > 0 && hooks ? hooks->ext_flag : 0;
    if (text_override) ext_flags |= cache_text_ext;
    /* Codec, chunk_size, and final payload_bytes are filled in after the
     * payload region is written below.  Stamp NONE/0 here so a partially-
     * written file is at least readable as uncompressed v2 if the patch-back
     * step fails for some reason. */
    ds4_kvstore_fill_header(h, (uint8_t)quant_bits,
                            ds4_kvstore_reason_code(reason), ext_flags,
                            DS4_KVSTORE_CODEC_NONE, 0,
                            (uint32_t)store_tokens.len, 0,
                            (uint32_t)ds4_session_ctx(session),
                            now, now, payload_bytes);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)text_len);
    uint64_t trailer_bytes = 0;
    errno = 0;
    const int n_workers = kc->opt.compression_threads;
    const uint32_t chunk_bytes = DS4_KVSTORE_DEFAULT_CHUNK_BYTES;
    uint8_t codec = DS4_KVSTORE_CODEC_NONE;
    uint64_t on_disk_payload = 0;
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(text, 1, text_len, fp) == text_len;
    if (ok) {
        const off_t payload_start = ftello(fp);
        if (payload_start < 0) {
            ok = false;
        } else if (n_workers == 0) {
            ok = ds4_session_save_payload(session, fp, save_err, sizeof(save_err)) == 0;
        } else {
            FILE *cw = kv_lz4_writer_open(fp, chunk_bytes, n_workers);
            if (!cw) {
                ok = false;
            } else {
                ok = ds4_session_save_payload(session, cw, save_err, sizeof(save_err)) == 0;
                /* fclose on the cookie patches the framing header in fp and
                 * does NOT close fp.  Any cookie-side write error sets the
                 * close return value. */
                if (fclose(cw) != 0) ok = false;
                if (ok) codec = DS4_KVSTORE_CODEC_LZ4;
            }
        }
        const off_t payload_end = ftello(fp);
        if (ok && payload_end < 0) ok = false;
        if (ok) on_disk_payload = (uint64_t)(payload_end - payload_start);
    }
    if (ok) {
        ok = kv_trailer_write(hooks, fp, text, &trailer_bytes) &&
             fflush(fp) == 0;
    }
    /* Patch the KVC header in place with the actual codec, chunk_size, and
     * on-disk payload byte count.  This is the only field rewrite; the rest
     * of the header was already correct. */
    if (ok) {
        uint8_t patched[DS4_KVSTORE_FIXED_HEADER];
        ds4_kvstore_fill_header(patched, (uint8_t)quant_bits,
                                ds4_kvstore_reason_code(reason),
                                ext_flags, codec,
                                codec == DS4_KVSTORE_CODEC_LZ4 ? chunk_bytes : 0,
                                (uint32_t)store_tokens.len, 0,
                                (uint32_t)ds4_session_ctx(session),
                                now, now, on_disk_payload);
        if (fseeko(fp, 0, SEEK_SET) != 0 ||
            fwrite(patched, 1, sizeof(patched), fp) != sizeof(patched) ||
            fflush(fp) != 0)
        {
            ok = false;
        }
    }
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    uint64_t final_file_bytes = 0, final_required_bytes = 0;
    bool final_size_over_budget = false;
    /* Use the real on-disk payload size — generally smaller than the
     * uncompressed estimate, never larger. */
    if (ok && !ds4_kvstore_file_size_fits(kc, (uint64_t)text_len, on_disk_payload,
                                          trailer_bytes,
                                          &final_file_bytes,
                                          &final_required_bytes))
    {
        final_size_over_budget = true;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    const double save_ms = (kv_now_sec() - save_t0) * 1000.0;
    if (!ok) {
        if (final_size_over_budget) {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache skipped tokens=%d reason=%s because final file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB save=%.1f ms",
                    kv_log_name(kc),
                    store_tokens.len,
                    reason,
                    (double)final_file_bytes / (1024.0 * 1024.0),
                    (double)final_required_bytes / (1024.0 * 1024.0),
                    (double)kc->budget_bytes / (1024.0 * 1024.0),
                    save_ms);
        } else {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache store failed (%s): %s save=%.1f ms",
                    kv_log_name(kc),
                    reason,
                    saved_errno ? strerror(saved_errno) :
                    (save_err[0] ? save_err : "unknown error"),
                    save_ms);
        }
        if (err && err_len) {
            snprintf(err, err_len, "%s",
                     saved_errno ? strerror(saved_errno) :
                     (save_err[0] ? save_err : "unknown error"));
        }
        unlink(tmp);
    } else {
        const double size_mib = (double)on_disk_payload / (1024.0 * 1024.0);
        if (codec == DS4_KVSTORE_CODEC_LZ4) {
            const double ratio = on_disk_payload ?
                (double)payload_bytes / (double)on_disk_payload : 0.0;
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache stored tokens=%d trimmed=%d reason=%s key=%s codec=lz4 size=%.2f MiB comp=%.2fx save=%.1f ms",
                    kv_log_name(kc),
                    store_tokens.len,
                    original_len - store_tokens.len,
                    reason,
                    text_override ? (cache_text_key ? cache_text_key : "visible-transcript") : "token-text",
                    size_mib, ratio, save_ms);
        } else {
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache stored tokens=%d trimmed=%d reason=%s key=%s size=%.2f MiB save=%.1f ms",
                    kv_log_name(kc),
                    store_tokens.len,
                    original_len - store_tokens.len,
                    reason,
                    text_override ? (cache_text_key ? cache_text_key : "visible-transcript") : "token-text",
                    (double)(DS4_KVSTORE_FIXED_HEADER + 4ull + text_len + on_disk_payload + trailer_bytes) / (1024.0 * 1024.0),
                    save_ms);
        }
        ds4_kvstore_evict(kc, live_tokens, sha);
    }
    free(tmp);
    free(text);
    free(path);
    ds4_tokens_free(&store_tokens);
    return ok;
}

bool ds4_kvstore_store_live_prefix(ds4_kvstore *kc,
                                   ds4_engine *engine,
                                   ds4_session *session,
                                   const ds4_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const ds4_kvstore_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len) {
    return ds4_kvstore_store_live_prefix_text(kc, engine, session, tokens,
                                              store_len, reason, NULL, 0, NULL,
                                              hooks, err, err_len);
}

bool ds4_kvstore_maybe_store_continued(ds4_kvstore *kc,
                                       ds4_engine *engine,
                                       ds4_session *session,
                                       const ds4_kvstore_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len) {
    const ds4_tokens *tokens = ds4_session_tokens(session);
    if (!tokens) return false;
    const int target = ds4_kvstore_continued_store_target(kc, tokens->len);
    if (target == 0) return false;
    if (ds4_kvstore_store_live_prefix(kc, engine, session, tokens, target,
                                      "continued", hooks, err, err_len))
    {
        ds4_kvstore_note_store(kc, target);
        return true;
    }
    return false;
}

int ds4_kvstore_find_text_prefix(ds4_kvstore *kc, const char *prompt_text,
                                 int quant_bits, int ctx_size) {
    if (!prompt_text) return -1;
    const size_t prompt_bytes = strlen(prompt_text);
    kv_cache_refresh(kc);
    int best = -1;
    for (int i = 0; i < kc->len; i++) {
        ds4_kvstore_entry *e = &kc->entry[i];
        if (e->text_bytes > prompt_bytes || e->text_bytes > SIZE_MAX) continue;
        if ((int)e->tokens < kc->opt.min_tokens) continue;
        if ((uint32_t)ctx_size < e->ctx_size) continue;
        if (kc->reject_different_quant && e->quant_bits != (uint8_t)quant_bits) continue;
        if (best >= 0) {
            ds4_kvstore_entry *b = &kc->entry[best];
            if (e->text_bytes < b->text_bytes) continue;
            if (e->text_bytes == b->text_bytes && e->tokens <= b->tokens) continue;
        }
        char sha[41];
        ds4_kvstore_sha1_bytes_hex(prompt_text, (size_t)e->text_bytes, sha);
        if (!strcmp(sha, e->sha)) best = i;
    }
    return best;
}

int ds4_kvstore_try_load_text(ds4_kvstore *kc,
                              ds4_engine *engine,
                              ds4_session *session,
                              const char *prompt_text,
                              ds4_tokens *effective_prompt,
                              ds4_kvstore_load_result *result,
                              const ds4_kvstore_trailer_hooks *hooks,
                              bool responses_protocol) {
    if (result) memset(result, 0, sizeof(*result));
    if (effective_prompt) effective_prompt->len = 0;
    if (!kc->enabled || !prompt_text) return 0;
    const int quant_bits = ds4_engine_routed_quant_bits(engine);
    if (quant_bits != 2 && quant_bits != 4) return 0;
    const size_t prompt_bytes = strlen(prompt_text);
    int idx = ds4_kvstore_find_text_prefix(kc, prompt_text, quant_bits,
                                           ds4_session_ctx(session));
    if (idx < 0) return 0;

    ds4_kvstore_entry e = kc->entry[idx];
    char *path = kv_xstrdup(e.path);
    const double load_t0 = kv_now_sec();
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        free(path);
        return 0;
    }
    uint32_t text_bytes = 0;
    ds4_kvstore_entry hdr = {0};
    const char *fail_reason = "invalid header";
    bool header_ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    char *cached_text = NULL;
    if (header_ok) {
        if ((uint64_t)text_bytes > prompt_bytes) {
            header_ok = false;
            fail_reason = "cached text is longer than prompt";
        } else {
            cached_text = kv_xmalloc((size_t)text_bytes + 1);
            if (fread(cached_text, 1, text_bytes, fp) != text_bytes) {
                header_ok = false;
                fail_reason = "truncated cached text";
            } else {
                cached_text[text_bytes] = '\0';
                char text_sha[41];
                ds4_kvstore_sha1_bytes_hex(cached_text, text_bytes, text_sha);
                if (strcmp(text_sha, e.sha)) {
                    header_ok = false;
                    fail_reason = "cached text hash mismatch";
                } else if (!ds4_kvstore_byte_prefix_match(prompt_text, prompt_bytes,
                                                          cached_text, text_bytes)) {
                    header_ok = false;
                    fail_reason = "cached text prefix mismatch";
                }
            }
        }
    }
    char err[160] = {0};
    int loaded = 0;
    int load_rc = 1;
    if (header_ok) {
        if (hdr.codec == DS4_KVSTORE_CODEC_LZ4) {
            const off_t payload_start = ftello(fp);
            /* The engine reads UNCOMPRESSED bytes from cr and requires the
             * "remaining" budget to be exactly the uncompressed payload size
             * (it errors on both shortfall and trailing bytes).  Have the
             * reader peek the framing header eagerly so we know the
             * uncompressed total before handing the wrapper to the engine. */
            uint64_t uncompressed_total = 0;
            FILE *cr = kv_lz4_reader_open(fp, hdr.payload_bytes, hdr.chunk_size,
                                          kc->opt.compression_threads > 0 ? kc->opt.compression_threads : 1,
                                          &uncompressed_total);
            if (!cr) {
                load_rc = 1;
                snprintf(err, sizeof(err), "failed to open lz4 reader");
            } else {
                load_rc = ds4_session_load_payload(session, cr, uncompressed_total,
                                                   err, sizeof(err));
                fclose(cr);
                /* Skip the outer fp past the payload region so the trailer
                 * load lands at the right place. */
                if (payload_start >= 0) {
                    (void)fseeko(fp, payload_start + (off_t)hdr.payload_bytes, SEEK_SET);
                }
            }
        } else {
            load_rc = ds4_session_load_payload(session, fp, hdr.payload_bytes,
                                               err, sizeof(err));
        }
    }
    if (load_rc == 0) {
        const ds4_tokens *loaded_tokens = ds4_session_tokens(session);
        if (loaded_tokens && loaded_tokens->len == (int)hdr.tokens) {
            loaded = (int)hdr.tokens;
            if (effective_prompt) {
                /* The cache lookup was by bytes, but the graph state is still
                 * the exact token history stored in the payload.  Build the
                 * prompt from that exact history and tokenize only the text
                 * suffix after the byte prefix. */
                ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
                    engine, loaded_tokens, prompt_text + text_bytes,
                    effective_prompt);
            }
            if (hooks && hooks->load && (hdr.ext_flags & hooks->ext_flag)) {
                hooks->load(hooks->ud, fp, hooks->load_wanted);
            }
        } else {
            ds4_session_invalidate(session);
            unlink(path);
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache discarded corrupt text-prefix payload%s%s %s",
                    kv_log_name(kc),
                    responses_protocol ? " " : "",
                    responses_protocol ? "RESPPROTO" : "",
                    path);
        }
    } else {
        if (header_ok) ds4_session_invalidate(session);
        kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                "%s: kv cache load failed%s%s %s: %s load=%.1f ms",
                kv_log_name(kc),
                responses_protocol ? " " : "",
                responses_protocol ? "RESPPROTO" : "",
                path,
                header_ok ? err : fail_reason,
                (kv_now_sec() - load_t0) * 1000.0);
    }
    fclose(fp);

    if (loaded > 0) {
        const double load_ms = (kv_now_sec() - load_t0) * 1000.0;
        kc->continued_last_store_tokens = loaded;
        const char *key_kind = ds4_kvstore_key_kind(hdr.ext_flags);
        bool consumed = false;
        if (kc->opt.cold_max_tokens > 0 && loaded > kc->opt.cold_max_tokens) {
            unlink(path);
            consumed = true;
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms consumed file=%s",
                    kv_log_name(kc),
                    responses_protocol ? " " : "",
                    responses_protocol ? "RESPPROTO" : "",
                    loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
        } else {
            ds4_kvstore_touch_file(path, hdr.hits + 1);
            kv_logf(kc, DS4_KVSTORE_LOG_KVCACHE,
                    "%s: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms file=%s",
                    kv_log_name(kc),
                    responses_protocol ? " " : "",
                    responses_protocol ? "RESPPROTO" : "",
                    loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
        }
        if (result) {
            result->tokens = loaded;
            result->text_bytes = text_bytes;
            result->quant_bits = hdr.quant_bits;
            result->ext_flags = hdr.ext_flags;
            result->load_ms = load_ms;
            result->consumed = consumed;
            result->path = kv_xstrdup(path);
        }
    }
    free(cached_text);
    free(path);
    return loaded;
}

void ds4_kvstore_load_result_free(ds4_kvstore_load_result *result) {
    if (!result) return;
    free(result->path);
    memset(result, 0, sizeof(*result));
}
