#include "ds4_kvstore.h"

#include "sha1.hpp"

/* Shared disk KV checkpoint file support (C++ port).
 *
 * The low-level file layout and payload helpers are intentionally shared.  The
 * ds4-server still owns the automatic byte-prefix cache policy built on top of
 * this file; ds4-agent uses only the same durable format for explicit sessions,
 * with its own policy in ds4_agent.c.  Protocol-specific extras, such as the
 * server's tool-id -> exact DSML trailer, are attached through trailer hooks and
 * still live with the protocol code that owns those mappings.
 *
 * Structure: pulsar::KvStore is a typed view over the C ds4_kvstore struct
 * (the state stays C so the still-C server/agent own it inline); the public
 * extern "C" functions at the bottom are one-line facades. */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
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
#define KV_CACHE_VERSION 1u
/* Header byte 20 carries the graph-payload ABI.  It is separate from the outer
 * file version because the KVC envelope can remain stable while the serialized
 * ds4_session internals become unsafe to restore across runtime changes. */
#define KV_CACHE_PAYLOAD_ABI 2u
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
/* A continued checkpoint that is a strict prefix of the incoming store is a
 * routine waypoint on the same path. Keep recent hits meaningful, but make
 * never-hit or stale waypoints cheap victims while pre-evicting for the new
 * store. */
#define KV_CACHE_CONTINUED_PREFIX_MIN_FACTOR 0.05
#define KV_CACHE_CONTINUED_PREFIX_HIT_FACTOR 0.45
/* Cold/evict/shutdown checkpoints are intentional anchors, not just automatic
 * waypoints in a single growing conversation. Give them a soft prior so they
 * survive comparable continued entries, while still allowing pressure and poor
 * density to evict them. */
#define KV_CACHE_ANCHOR_REASON_SCORE_FACTOR 2.0

namespace pulsar {
namespace {

[[noreturn]] void kv_die(const char *msg) {
    fprintf(stderr, "ds4-kvstore: %s\n", msg);
    exit(1);
}

void *kv_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

void *kv_xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) kv_die("out of memory");
    return p;
}

char *kv_xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = static_cast<char *>(kv_xmalloc(n + 1));
    memcpy(p, s, n + 1);
    return p;
}

/* Growable NUL-terminated byte builder. Deliberately malloc-backed, not
 * std::string: take() hands the buffer to C callers that free() it. Will
 * merge with the server's `buf` when that TU ports. */
class KvBuf {
public:
    KvBuf() = default;
    KvBuf(const KvBuf &) = delete;
    KvBuf &operator=(const KvBuf &) = delete;
    ~KvBuf() { free(ptr_); }

    size_t len() const { return len_; }
    const char *data() const { return ptr_; }

    void append(const void *p, size_t n) {
        reserve(n);
        memcpy(ptr_ + len_, p, n);
        len_ += n;
        ptr_[len_] = '\0';
    }

    void putc(char c) { append(&c, 1); }
    void puts(const char *s) { append(s, strlen(s)); }

    __attribute__((format(printf, 2, 3)))
    void printf(const char *fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        va_list ap2;
        va_copy(ap2, ap);
        int n = vsnprintf(NULL, 0, fmt, ap);
        va_end(ap);
        if (n < 0) kv_die("vsnprintf failed");
        reserve((size_t)n);
        vsnprintf(ptr_ + len_, cap_ - len_, fmt, ap2);
        va_end(ap2);
        len_ += (size_t)n;
    }

    /* Transfer ownership to the caller (who free()s it). */
    char *take() {
        if (!ptr_) return kv_xstrdup("");
        char *p = ptr_;
        ptr_ = NULL;
        len_ = 0;
        cap_ = 0;
        return p;
    }

private:
    void reserve(size_t add) {
        if (add > SIZE_MAX - len_ - 1) kv_die("buffer overflow");
        size_t need = len_ + add + 1;
        if (need <= cap_) return;
        size_t cap = cap_ ? cap_ * 2 : 256;
        while (cap < need) cap *= 2;
        ptr_ = static_cast<char *>(kv_xrealloc(ptr_, cap));
        cap_ = cap;
    }

    char *ptr_ = nullptr;
    size_t len_ = 0;
    size_t cap_ = 0;
};

double kv_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

void kv_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

uint64_t kv_le_get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

bool kv_mkdir_p(const char *path) {
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

/* "<name>.tmp.<pid>" left by a store that never reached rename().  Reclaim it
 * once the writing process is gone; a live pid means a save is in flight. */
bool kv_cache_tmp_is_orphan(const char *name) {
    const char *tag = strstr(name, ".tmp.");
    if (!tag || !tag[5]) return false;
    char *end = NULL;
    long pid = strtol(tag + 5, &end, 10);
    if (end == tag + 5 || *end != '\0' || pid <= 0) return true;
    return kill((pid_t)pid, 0) != 0 && errno == ESRCH;
}

bool kv_cache_reason_is_anchor(uint8_t reason) {
    return reason == DS4_KVSTORE_REASON_COLD ||
           reason == DS4_KVSTORE_REASON_EVICT ||
           reason == DS4_KVSTORE_REASON_SHUTDOWN;
}

bool kv_cache_file_size_bytes(uint64_t text_bytes,
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

bool kv_cache_budget_required(uint64_t file_bytes,
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

bool kv_cache_incoming_supersedes_continued(
        const ds4_kvstore_entry *e,
        const ds4_kvstore_eviction_context *incoming) {
    if (!e || !incoming || !incoming->text) return false;
    if (e->reason != DS4_KVSTORE_REASON_CONTINUED) return false;
    if (e->text_bytes == 0 || e->text_bytes > SIZE_MAX) return false;
    if ((size_t)e->text_bytes >= incoming->text_len) return false;
    if (e->model_id != incoming->model_id) return false;
    if (incoming->reject_different_quant &&
        e->quant_bits != incoming->quant_bits)
        return false;
    /* A smaller-context checkpoint can be loaded by a larger context, but not
     * the reverse.  The incoming checkpoint dominates this one only if it is at
     * least as widely reusable. */
    if (incoming->ctx_size > e->ctx_size) return false;

    char prefix_sha[41];
    Sha1::bytes_hex(incoming->text, (size_t)e->text_bytes, prefix_sha);
    return !strcmp(prefix_sha, e->sha);
}

bool kv_cache_file_text_matches(const char *path, const char sha[41],
                                const char *text, size_t text_len) {
    if (text_len > UINT32_MAX) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;

    ds4_kvstore_entry hdr = {};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
              text_bytes == (uint32_t)text_len;
    char *stored = NULL;
    if (ok) {
        stored = static_cast<char *>(kv_xmalloc((size_t)text_bytes + 1));
        ok = fread(stored, 1, text_bytes, fp) == text_bytes;
    }
    fclose(fp);
    if (!ok) {
        free(stored);
        return false;
    }

    char stored_sha[41];
    Sha1::bytes_hex(stored, text_bytes, stored_sha);
    ok = !strcmp(stored_sha, sha) &&
         (text_len == 0 || memcmp(stored, text, text_len) == 0);
    free(stored);
    return ok;
}

bool kv_trailer_serialized_size(const ds4_kvstore_trailer_hooks *hooks,
                                const char *text,
                                uint64_t *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!hooks || !hooks->serialized_size) return true;
    return hooks->serialized_size(hooks->ud, text, bytes_out);
}

bool kv_trailer_write(const ds4_kvstore_trailer_hooks *hooks,
                      FILE *fp, const char *text,
                      uint64_t *written_bytes) {
    if (written_bytes) *written_bytes = 0;
    if (!hooks || !hooks->write) return true;
    return hooks->write(hooks->ud, fp, text, written_bytes);
}

void tokens_append(ds4_tokens *dst, const ds4_tokens *src) {
    if (!dst || !src) return;
    for (int i = 0; i < src->len; i++) ds4_tokens_push(dst, src->v[i]);
}

/* Typed view over the C ds4_kvstore struct: policy and stateful flows live
 * here; stateless format helpers stay free functions above. */
class KvStore {
public:
    explicit KvStore(ds4_kvstore &kc) : kc_(kc) {}

    const char *log_name() const {
        return kc_.log_name ? kc_.log_name : "ds4";
    }

    __attribute__((format(printf, 3, 4)))
    void logf(ds4_kvstore_log_type type, const char *fmt, ...) {
        if (!kc_.log) return;
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
        char *msg = static_cast<char *>(kv_xmalloc((size_t)n + 1));
        vsnprintf(msg, (size_t)n + 1, fmt, ap2);
        va_end(ap2);
        kc_.log(kc_.log_ud, type, msg);
        free(msg);
    }

    char *path_for_sha(const char sha[41]) {
        char name[44];
        memcpy(name, sha, 40);
        memcpy(name + 40, ".kv", 4);
        return ds4_kvstore_path_join(kc_.dir, name);
    }

    void clear() {
        for (int i = 0; i < kc_.len; i++) ds4_kvstore_entry_free(&kc_.entry[i]);
        free(kc_.entry);
        kc_.entry = NULL;
        kc_.len = 0;
        kc_.cap = 0;
    }

    bool open(const char *dir, uint64_t budget_mb,
              bool reject_different_quant, ds4_kvstore_options opt,
              const char *log_name,
              void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg),
              void *log_ud) {
        memset(&kc_, 0, sizeof(kc_));
        if (!dir) return false;
        kc_.log_name = log_name;
        kc_.log = log;
        kc_.log_ud = log_ud;
        if (!kv_mkdir_p(dir)) {
            logf(DS4_KVSTORE_LOG_DEFAULT,
                 "%s: failed to create KV cache directory %s: %s",
                 this->log_name(), dir, strerror(errno));
            return false;
        }
        /* Probe writability up front: a pre-existing read-only directory passes
         * kv_mkdir_p (EEXIST) but would make every later store fail one at a
         * time.  Fail the open instead so the caller runs with the cache cleanly
         * disabled after one clear log line.  (This intentionally also refuses a
         * read-only pre-populated dir that restores alone could serve from —
         * this store is a read-write cache, not a frozen-snapshot reader.)
         * AT_EACCESS: check the EFFECTIVE credentials, the ones fopen/rename
         * will actually use. */
        if (faccessat(AT_FDCWD, dir, W_OK | X_OK, AT_EACCESS) != 0) {
            logf(DS4_KVSTORE_LOG_DEFAULT,
                 "%s: KV cache directory %s is not writable: %s",
                 this->log_name(), dir, strerror(errno));
            return false;
        }
        kc_.enabled = true;
        kc_.dir = kv_xstrdup(dir);
        if (budget_mb == 0) budget_mb = DS4_KVSTORE_DEFAULT_MB;
        kc_.budget_bytes = budget_mb * 1024ull * 1024ull;
        kc_.reject_different_quant = reject_different_quant;
        kc_.opt = opt;
        evict(NULL, 0, NULL);
        logf(DS4_KVSTORE_LOG_KVCACHE,
             "%s: KV disk cache %s (budget=%llu MiB, cross-quant=%s, min=%d, cold_max=%d, continued=%d, trim=%d, align=%d, hit_half_life=%llus)",
             this->log_name(),
             kc_.dir,
             (unsigned long long)(kc_.budget_bytes / (1024ull * 1024ull)),
             reject_different_quant ? "reject" : "accept",
             kc_.opt.min_tokens,
             kc_.opt.cold_max_tokens,
             kc_.opt.continued_interval_tokens,
             kc_.opt.boundary_trim_tokens,
             kc_.opt.boundary_align_tokens,
             (unsigned long long)DS4_KVSTORE_HIT_HALF_LIFE_SECONDS);
        return true;
    }

    void close() {
        clear();
        free(kc_.dir);
        memset(&kc_, 0, sizeof(kc_));
    }

    void refresh() {
        if (!kc_.enabled) return;
        clear();
        DIR *d = opendir(kc_.dir);
        if (!d) return;
        struct dirent *de;
        while ((de = readdir(d)) != NULL) {
            char sha[41];
            if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) {
                if (kv_cache_tmp_is_orphan(de->d_name)) {
                    char *path = ds4_kvstore_path_join(kc_.dir, de->d_name);
                    if (unlink(path) == 0) {
                        logf(DS4_KVSTORE_LOG_KVCACHE,
                             "%s: kv cache removed orphaned temp file %s",
                             log_name(), de->d_name);
                    }
                    free(path);
                }
                continue;
            }
            char *path = ds4_kvstore_path_join(kc_.dir, de->d_name);
            ds4_kvstore_entry e = {};
            if (ds4_kvstore_read_entry_file(path, sha, &e)) push(e);
            free(path);
        }
        closedir(d);
    }

    void evict(const ds4_tokens *live, uint64_t extra_bytes,
               const ds4_kvstore_eviction_context *incoming) {
        if (!kc_.enabled || kc_.budget_bytes == 0) return;
        if (extra_bytes > kc_.budget_bytes) return;
        refresh();
        const uint64_t now = (uint64_t)time(NULL);
        uint64_t total = 0;
        for (int i = 0; i < kc_.len; i++) total += kc_.entry[i].file_size;
        const uint64_t target = kc_.budget_bytes - extra_bytes;
        while (total > target && kc_.len > 0) {
            int victim = 0;
            double victim_score =
                ds4_kvstore_entry_eviction_score(&kc_.entry[0], live, now,
                                                 incoming);
            for (int i = 1; i < kc_.len; i++) {
                double score =
                    ds4_kvstore_entry_eviction_score(&kc_.entry[i], live, now,
                                                     incoming);
                if (score < victim_score ||
                    (score == victim_score &&
                     kc_.entry[i].last_used < kc_.entry[victim].last_used))
                {
                    victim = i;
                    victim_score = score;
                }
            }
            ds4_kvstore_entry e = kc_.entry[victim];
            if (unlink(e.path) == 0) {
                logf(DS4_KVSTORE_LOG_KVCACHE,
                     "%s: kv cache evicted reason=disk-cache-full tokens=%u hits=%u size=%.2f MiB file=%s",
                     log_name(),
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
            memmove(kc_.entry + victim, kc_.entry + victim + 1,
                    (size_t)(kc_.len - victim - 1) * sizeof(kc_.entry[0]));
            kc_.len--;
        }
    }

    int store_len(int tokens) const {
        const int trim = kc_.opt.boundary_trim_tokens;
        const int align = kc_.opt.boundary_align_tokens;
        if (tokens > kc_.opt.min_tokens + trim) {
            int stable = tokens - trim;
            if (align > 0) stable -= stable % align;
            if (stable >= kc_.opt.min_tokens) return stable;
        }
        return tokens;
    }

    int chat_anchor_pos(const ds4_tokens *prompt,
                        int user_token_id,
                        int assistant_token_id) const {
        if (!prompt || user_token_id < 0 || assistant_token_id < 0) return -1;

        /* Cold checkpoints maximize reuse across independent agent sessions.
         * The stable rendered chat prefix is everything before the user message
         * that asks this specific task.  Some clients put stable user-role
         * scaffolding first, so use the last user marker before the first
         * assistant marker. */
        int last_user = -1;
        for (int i = 0; i < prompt->len; i++) {
            const int token = prompt->v[i];
            if (token == assistant_token_id) break;
            if (token == user_token_id) last_user = i;
        }
        return last_user >= kc_.opt.min_tokens ? last_user : -1;
    }

    int continued_step() const {
        if (!kc_.enabled || kc_.opt.continued_interval_tokens <= 0) return 0;
        int step = kc_.opt.continued_interval_tokens;
        const int align = kc_.opt.boundary_align_tokens;
        if (align > 0) {
            step = ((step + align - 1) / align) * align;
            if (step <= 0) step = align;
        }
        return step;
    }

    int continued_store_target(int live_tokens) const {
        const int step = continued_step();
        if (step <= 0) return 0;
        if (live_tokens < kc_.opt.min_tokens) return 0;
        if (live_tokens % step != 0) return 0;
        if (live_tokens <= kc_.continued_last_store_tokens) return 0;
        return live_tokens;
    }

    void note_store(int tokens) {
        if (tokens > kc_.continued_last_store_tokens) {
            kc_.continued_last_store_tokens = tokens;
        }
    }

    int suppress_continued_store(int tokens) {
        if (continued_store_target(tokens) != tokens) return -1;
        int old = kc_.continued_last_store_tokens;
        note_store(tokens);
        return old;
    }

    void restore_suppressed_continued(int old_tokens, int suppressed_tokens) {
        if (old_tokens >= 0 &&
            kc_.continued_last_store_tokens == suppressed_tokens) {
            kc_.continued_last_store_tokens = old_tokens;
        }
    }

    bool file_size_fits(uint64_t text_bytes,
                        uint64_t payload_bytes,
                        uint64_t trailer_bytes,
                        uint64_t *file_bytes_out,
                        uint64_t *required_bytes_out) const {
        uint64_t file_bytes = 0;
        if (!kv_cache_file_size_bytes(text_bytes, payload_bytes, trailer_bytes,
                                      &file_bytes))
            return false;
        if (file_bytes_out) *file_bytes_out = file_bytes;
        if (kc_.budget_bytes == 0) return true;
        uint64_t required = 0;
        if (!kv_cache_budget_required(file_bytes, &required)) return false;
        if (required_bytes_out) *required_bytes_out = required;
        return required <= kc_.budget_bytes;
    }

    int find_text_prefix(const char *prompt_text,
                         int model_id, int quant_bits, int ctx_size) {
        if (!prompt_text) return -1;
        const size_t prompt_bytes = strlen(prompt_text);
        refresh();
        int best = -1;
        for (int i = 0; i < kc_.len; i++) {
            ds4_kvstore_entry *e = &kc_.entry[i];
            if (e->text_bytes > prompt_bytes || e->text_bytes > SIZE_MAX) continue;
            if ((int)e->tokens < kc_.opt.min_tokens) continue;
            if (e->model_id != (uint8_t)model_id) continue;
            if ((uint32_t)ctx_size < e->ctx_size) continue;
            if (kc_.reject_different_quant && e->quant_bits != (uint8_t)quant_bits) continue;
            if (best >= 0) {
                ds4_kvstore_entry *b = &kc_.entry[best];
                if (e->text_bytes < b->text_bytes) continue;
                if (e->text_bytes == b->text_bytes && e->tokens <= b->tokens) continue;
            }
            char sha[41];
            Sha1::bytes_hex(prompt_text, (size_t)e->text_bytes, sha);
            if (!strcmp(sha, e->sha)) best = i;
        }
        return best;
    }

    bool existing_compatible(const char *path, const char sha[41],
                             const char *text, size_t text_len,
                             int model_id, int quant_bits, int ctx_size) {
        if (access(path, F_OK) != 0) return false;
        ds4_kvstore_entry e = {};
        if (!ds4_kvstore_read_entry_file(path, sha, &e)) return false;
        bool compatible = e.model_id == (uint8_t)model_id &&
                          (!kc_.reject_different_quant ||
                           e.quant_bits == (uint8_t)quant_bits) &&
                          e.ctx_size <= (uint32_t)ctx_size &&
                          kv_cache_file_text_matches(path, sha, text, text_len);
        ds4_kvstore_entry_free(&e);
        if (!compatible) {
            if (unlink(path) == 0) {
                logf(DS4_KVSTORE_LOG_KVCACHE,
                     "%s: kv cache replaced incompatible file %s",
                     log_name(), path);
            }
            return false;
        }
        return true;
    }

    void rewrite_trailer(const char *path, const char *text,
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
        ds4_kvstore_entry hdr = {};
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
                ds4_kvstore_fill_header(h, hdr.model_id, hdr.quant_bits, hdr.reason,
                                        (uint8_t)(hdr.ext_flags | hooks->ext_flag),
                                        hdr.tokens, hdr.hits, hdr.ctx_size,
                                        hdr.created_at, now, hdr.payload_bytes);
                ok = fseeko(fp, 0, SEEK_SET) == 0 &&
                     fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
                     fflush(fp) == 0;
            }
        }
        fclose(fp);
        (void)ok;
    }

    bool store_live_prefix_text(ds4_engine *engine,
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
        if (!kc_.enabled) return false;
        if (!tokens || store_len < kc_.opt.min_tokens) return false;
        const int original_len = tokens->len;

        ds4_tokens store_tokens = {};
        ds4_kvstore_tokens_copy_prefix(&store_tokens, tokens, store_len);

        const int quant_bits = ds4_engine_routed_quant_bits(engine);
        if (quant_bits != 2 && quant_bits != 4) {
            ds4_tokens_free(&store_tokens);
            return false;
        }
        const int model_id = ds4_engine_model_id(engine);

        char save_err[160] = {};
        const ds4_tokens *live_tokens = ds4_session_tokens(session);
        if (!live_tokens ||
            live_tokens->len != store_tokens.len ||
            !ds4_tokens_starts_with(live_tokens, &store_tokens))
        {
            logf(DS4_KVSTORE_LOG_KVCACHE,
                 "%s: kv cache skipped tokens=%d reason=%s because live checkpoint is at %d",
                 log_name(),
                 store_tokens.len,
                 reason,
                 live_tokens ? live_tokens->len : -1);
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
            logf(DS4_KVSTORE_LOG_KVCACHE,
                 "%s: kv cache skipped tokens=%d because rendered text is too large",
                 log_name(), store_tokens.len);
            free(text);
            ds4_tokens_free(&store_tokens);
            return false;
        }

        uint64_t trailer_est_bytes = 0;
        if (!kv_trailer_serialized_size(hooks, text, &trailer_est_bytes)) {
            logf(DS4_KVSTORE_LOG_KVCACHE,
                 "%s: kv cache skipped tokens=%d reason=%s because tool map size overflowed",
                 log_name(), store_tokens.len, reason);
            free(text);
            ds4_tokens_free(&store_tokens);
            return false;
        }
        char sha[41];
        Sha1::bytes_hex(text, text_len, sha);
        char *path = path_for_sha(sha);
        const uint8_t reason_code = ds4_kvstore_reason_code(reason);

        if (existing_compatible(path, sha, text, text_len,
                                model_id,
                                quant_bits, ds4_session_ctx(session))) {
            rewrite_trailer(path, text, hooks);
            free(text);
            free(path);
            ds4_tokens_free(&store_tokens);
            return true;
        }

        ds4_session_payload_file staged = {};
        if (ds4_session_stage_payload(session, &staged,
                                      save_err, sizeof(save_err)) != 0) {
            logf(DS4_KVSTORE_LOG_KVCACHE,
                 "%s: kv cache skipped tokens=%d reason=%s because KV payload staging failed: %s",
                 log_name(),
                 store_tokens.len,
                 reason,
                 save_err[0] ? save_err : "unknown error");
            if (err && err_len) snprintf(err, err_len, "%s",
                                         save_err[0] ? save_err : "unknown error");
            free(text);
            free(path);
            ds4_tokens_free(&store_tokens);
            return false;
        }
        uint64_t payload_bytes = staged.bytes;

        uint64_t est_file_bytes = 0, est_required_bytes = 0;
        if (!file_size_fits((uint64_t)text_len, payload_bytes,
                            trailer_est_bytes,
                            &est_file_bytes, &est_required_bytes)) {
            logf(DS4_KVSTORE_LOG_KVCACHE,
                 "%s: kv cache skipped tokens=%d reason=%s because estimated file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB",
                 log_name(),
                 store_tokens.len,
                 reason,
                 (double)est_file_bytes / (1024.0 * 1024.0),
                 (double)est_required_bytes / (1024.0 * 1024.0),
                 (double)kc_.budget_bytes / (1024.0 * 1024.0));
            ds4_session_payload_file_free(&staged);
            free(text);
            free(path);
            ds4_tokens_free(&store_tokens);
            return false;
        }

        ds4_kvstore_eviction_context incoming = {};
        incoming.text = text;
        incoming.text_len = text_len;
        incoming.model_id = (uint8_t)model_id;
        incoming.quant_bits = (uint8_t)quant_bits;
        incoming.ctx_size = (uint32_t)ds4_session_ctx(session);
        incoming.reject_different_quant = kc_.reject_different_quant;
        evict(live_tokens, est_file_bytes, &incoming);

        KvBuf tmpb;
        tmpb.printf("%s.tmp.%ld", path, (long)getpid());
        char *tmp = tmpb.take();
        const double save_t0 = kv_now_sec();
        FILE *fp = fopen(tmp, "wb");
        if (!fp) {
            logf(DS4_KVSTORE_LOG_KVCACHE,
                 "%s: kv cache failed to create %s: %s save=%.1f ms",
                 log_name(), tmp, strerror(errno),
                 (kv_now_sec() - save_t0) * 1000.0);
            ds4_session_payload_file_free(&staged);
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
        ds4_kvstore_fill_header(h, (uint8_t)model_id, (uint8_t)quant_bits,
                                reason_code, ext_flags,
                                (uint32_t)store_tokens.len, 0,
                                (uint32_t)ds4_session_ctx(session),
                                now, now, payload_bytes);
        uint8_t tb[4];
        ds4_kvstore_le_put32(tb, (uint32_t)text_len);
        uint64_t trailer_bytes = 0;
        errno = 0;
        bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
                  fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
                  fwrite(text, 1, text_len, fp) == text_len &&
                  ds4_session_write_staged_payload(&staged, fp,
                                                   save_err, sizeof(save_err)) == 0 &&
                  kv_trailer_write(hooks, fp, text, &trailer_bytes) &&
                  fflush(fp) == 0 &&
                  /* fsync before the atomic rename: without it a crash can leave
                   * a zero/partial .kv visible under the final name. */
                  fsync(fileno(fp)) == 0;
        int saved_errno = errno;
        if (fclose(fp) != 0) {
            if (!saved_errno) saved_errno = errno;
            ok = false;
        }
        uint64_t final_file_bytes = 0, final_required_bytes = 0;
        bool final_size_over_budget = false;
        if (ok && !file_size_fits((uint64_t)text_len, payload_bytes,
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
        if (ok) {
            /* Persist the rename itself: fsync the containing directory so the
             * new entry survives a crash. Best-effort (some filesystems reject
             * directory fsync); the file contents are already durable above. */
            char *slash = strrchr(path, '/');
            if (slash && slash != path) {
                *slash = '\0';
                int dfd = ::open(path, O_RDONLY | O_DIRECTORY);
                *slash = '/';
                if (dfd >= 0) {
                    (void)fsync(dfd);
                    ::close(dfd);
                }
            }
        }
        const double save_ms = (kv_now_sec() - save_t0) * 1000.0;
        if (!ok) {
            if (final_size_over_budget) {
                logf(DS4_KVSTORE_LOG_KVCACHE,
                     "%s: kv cache skipped tokens=%d reason=%s because final file size %.2f MiB (%.2f MiB with safety) exceeds budget %.2f MiB save=%.1f ms",
                     log_name(),
                     store_tokens.len,
                     reason,
                     (double)final_file_bytes / (1024.0 * 1024.0),
                     (double)final_required_bytes / (1024.0 * 1024.0),
                     (double)kc_.budget_bytes / (1024.0 * 1024.0),
                     save_ms);
            } else {
                logf(DS4_KVSTORE_LOG_KVCACHE,
                     "%s: kv cache store failed (%s): %s save=%.1f ms",
                     log_name(),
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
            logf(DS4_KVSTORE_LOG_KVCACHE,
                 "%s: kv cache stored tokens=%d trimmed=%d reason=%s key=%s size=%.2f MiB save=%.1f ms",
                 log_name(),
                 store_tokens.len,
                 original_len - store_tokens.len,
                 reason,
                 text_override ? (cache_text_key ? cache_text_key : "visible-transcript") : "token-text",
                 (double)(DS4_KVSTORE_FIXED_HEADER + 4ull + text_len + payload_bytes + trailer_bytes) / (1024.0 * 1024.0),
                 save_ms);
        }
        ds4_session_payload_file_free(&staged);
        free(tmp);
        free(text);
        free(path);
        ds4_tokens_free(&store_tokens);
        return ok;
    }

    int try_load_text(ds4_engine *engine,
                      ds4_session *session,
                      const char *prompt_text,
                      ds4_tokens *effective_prompt,
                      ds4_kvstore_load_result *result,
                      const ds4_kvstore_trailer_hooks *hooks,
                      bool responses_protocol) {
        if (result) memset(result, 0, sizeof(*result));
        if (effective_prompt) effective_prompt->len = 0;
        if (!kc_.enabled || !prompt_text) return 0;
        const int quant_bits = ds4_engine_routed_quant_bits(engine);
        if (quant_bits != 2 && quant_bits != 4) return 0;
        const int model_id = ds4_engine_model_id(engine);
        const size_t prompt_bytes = strlen(prompt_text);
        int idx = find_text_prefix(prompt_text, model_id, quant_bits,
                                   ds4_session_ctx(session));
        if (idx < 0) return 0;

        ds4_kvstore_entry e = kc_.entry[idx];
        char *path = kv_xstrdup(e.path);
        const double load_t0 = kv_now_sec();
        FILE *fp = fopen(path, "rb");
        if (!fp) {
            free(path);
            return 0;
        }
        uint32_t text_bytes = 0;
        ds4_kvstore_entry hdr = {};
        const char *fail_reason = "invalid header";
        bool header_ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
        char *cached_text = NULL;
        if (header_ok) {
            if (hdr.model_id != (uint8_t)model_id) {
                header_ok = false;
                fail_reason = "cached checkpoint was written for a different model";
            } else if ((uint64_t)text_bytes > prompt_bytes) {
                header_ok = false;
                fail_reason = "cached text is longer than prompt";
            } else {
                cached_text = static_cast<char *>(kv_xmalloc((size_t)text_bytes + 1));
                if (fread(cached_text, 1, text_bytes, fp) != text_bytes) {
                    header_ok = false;
                    fail_reason = "truncated cached text";
                } else {
                    cached_text[text_bytes] = '\0';
                    char text_sha[41];
                    Sha1::bytes_hex(cached_text, text_bytes, text_sha);
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
        char err[160] = {};
        int loaded = 0;
        if (header_ok &&
            ds4_session_load_payload(session, fp, hdr.payload_bytes, err, sizeof(err)) == 0)
        {
            const ds4_tokens *loaded_tokens = ds4_session_tokens(session);
            if (loaded_tokens && loaded_tokens->len == (int)hdr.tokens) {
                loaded = (int)hdr.tokens;
                if (effective_prompt) {
                    /* The cache lookup was by bytes, but the graph state is
                     * still the exact token history stored in the payload.
                     * Build the prompt from that exact history and tokenize
                     * only the text suffix after the byte prefix. */
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
                logf(DS4_KVSTORE_LOG_KVCACHE,
                     "%s: kv cache discarded corrupt text-prefix payload%s%s %s",
                     log_name(),
                     responses_protocol ? " " : "",
                     responses_protocol ? "RESPPROTO" : "",
                     path);
            }
        } else {
            if (header_ok) ds4_session_invalidate(session);
            logf(DS4_KVSTORE_LOG_KVCACHE,
                 "%s: kv cache load failed%s%s %s: %s load=%.1f ms",
                 log_name(),
                 responses_protocol ? " " : "",
                 responses_protocol ? "RESPPROTO" : "",
                 path,
                 header_ok ? err : fail_reason,
                 (kv_now_sec() - load_t0) * 1000.0);
        }
        fclose(fp);

        if (loaded > 0) {
            const double load_ms = (kv_now_sec() - load_t0) * 1000.0;
            kc_.continued_last_store_tokens = loaded;
            const char *key_kind = ds4_kvstore_key_kind(hdr.ext_flags);
            bool consumed = false;
            if (kc_.opt.cold_max_tokens > 0 && loaded > kc_.opt.cold_max_tokens) {
                unlink(path);
                consumed = true;
                logf(DS4_KVSTORE_LOG_KVCACHE,
                     "%s: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms consumed file=%s",
                     log_name(),
                     responses_protocol ? " " : "",
                     responses_protocol ? "RESPPROTO" : "",
                     loaded, text_bytes, hdr.quant_bits, key_kind, load_ms, path);
            } else {
                ds4_kvstore_touch_file(path, hdr.hits + 1);
                logf(DS4_KVSTORE_LOG_KVCACHE,
                     "%s: kv cache hit text%s%s tokens=%d text=%u quant=%u key=%s load=%.1f ms file=%s",
                     log_name(),
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

private:
    void push(ds4_kvstore_entry e) {
        if (kc_.len == kc_.cap) {
            kc_.cap = kc_.cap ? kc_.cap * 2 : 16;
            kc_.entry = static_cast<ds4_kvstore_entry *>(
                kv_xrealloc(kc_.entry, (size_t)kc_.cap * sizeof(kc_.entry[0])));
        }
        kc_.entry[kc_.len++] = e;
    }

    ds4_kvstore &kc_;
};

} // namespace
} // namespace pulsar

using pulsar::KvStore;

/* ---- stateless format/helper API ---- */

ds4_kvstore_options ds4_kvstore_default_options(void) {
    ds4_kvstore_options o = {};
    o.min_tokens = KV_CACHE_DEFAULT_MIN_TOKENS;
    o.cold_max_tokens = KV_CACHE_DEFAULT_COLD_MAX_TOKENS;
    o.continued_interval_tokens = KV_CACHE_DEFAULT_CONTINUED_INTERVAL_TOKENS;
    o.boundary_trim_tokens = KV_CACHE_DEFAULT_BOUNDARY_TRIM_TOKENS;
    o.boundary_align_tokens = KV_CACHE_DEFAULT_BOUNDARY_ALIGN_TOKENS;
    return o;
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

uint32_t ds4_kvstore_le_get32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

void ds4_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    pulsar::Sha1::bytes_hex(ptr, len, out);
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
    pulsar::KvBuf b;
    b.puts(dir);
    if (b.len() == 0 || b.data()[b.len() - 1] != '/') b.putc('/');
    b.puts(name);
    return b.take();
}

char *ds4_kvstore_path_for_sha(ds4_kvstore *kc, const char sha[41]) {
    return KvStore(*kc).path_for_sha(sha);
}

void ds4_kvstore_entry_free(ds4_kvstore_entry *e) {
    free(e->path);
    memset(e, 0, sizeof(*e));
}

void ds4_kvstore_fill_header(uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
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
    h[7] = model_id;
    ds4_kvstore_le_put32(h + 8, tokens);
    ds4_kvstore_le_put32(h + 12, hits);
    ds4_kvstore_le_put32(h + 16, ctx_size);
    h[20] = KV_CACHE_PAYLOAD_ABI;
    pulsar::kv_le_put64(h + 24, created_at);
    pulsar::kv_le_put64(h + 32, last_used);
    pulsar::kv_le_put64(h + 40, payload_bytes);
}

bool ds4_kvstore_read_header(FILE *fp, ds4_kvstore_entry *e,
                             uint32_t *text_bytes) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    if (fread(h, 1, sizeof(h), fp) != sizeof(h)) return false;
    if (h[0] != KV_CACHE_MAGIC0 || h[1] != KV_CACHE_MAGIC1 ||
        h[2] != KV_CACHE_MAGIC2 || h[3] != KV_CACHE_VERSION) return false;
    if (h[20] != KV_CACHE_PAYLOAD_ABI) return false;
    e->quant_bits = h[4];
    e->reason = h[5] <= DS4_KVSTORE_REASON_AGENT_SESSION ? h[5] :
                (uint8_t)DS4_KVSTORE_REASON_UNKNOWN;
    e->ext_flags = h[6];
    e->model_id = h[7];
    e->tokens = ds4_kvstore_le_get32(h + 8);
    e->hits = ds4_kvstore_le_get32(h + 12);
    e->ctx_size = ds4_kvstore_le_get32(h + 16);
    e->created_at = pulsar::kv_le_get64(h + 24);
    e->last_used = pulsar::kv_le_get64(h + 32);
    e->payload_bytes = pulsar::kv_le_get64(h + 40);
    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) return false;
    *text_bytes = ds4_kvstore_le_get32(tb);
    e->text_bytes = *text_bytes;
    return e->tokens != 0 && (e->quant_bits == 2 || e->quant_bits == 4);
}

bool ds4_kvstore_read_entry_file(const char *path, const char sha[41],
                                 ds4_kvstore_entry *out) {
    struct stat st;
    if (stat(path, &st) != 0 ||
        st.st_size < (off_t)(DS4_KVSTORE_FIXED_HEADER + 4))
        return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    ds4_kvstore_entry e = {};
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
    e.path = pulsar::kv_xstrdup(path);
    e.file_size = (uint64_t)st.st_size;
    *out = e;
    return true;
}

bool ds4_kvstore_touch_file(const char *path, uint32_t hits) {
    FILE *fp = fopen(path, "r+b");
    if (!fp) return false;
    ds4_kvstore_entry e = {};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &e, &text_bytes);
    if (ok) {
        uint8_t h[DS4_KVSTORE_FIXED_HEADER];
        uint64_t now = (uint64_t)time(NULL);
        ds4_kvstore_fill_header(h, e.model_id, e.quant_bits, e.reason, e.ext_flags,
                                e.tokens, hits, e.ctx_size,
                                e.created_at, now, e.payload_bytes);
        ok = fseek(fp, 0, SEEK_SET) == 0 &&
             fwrite(h, 1, sizeof(h), fp) == sizeof(h);
    }
    fclose(fp);
    return ok;
}

double ds4_kvstore_entry_eviction_score(
        const ds4_kvstore_entry *e,
        const ds4_tokens *live,
        uint64_t now,
        const ds4_kvstore_eviction_context *incoming) {
    if (!e || e->file_size == 0) return 0.0;
    (void)live;
    double effective_hits = (double)e->hits;
    uint64_t used_at = e->last_used ? e->last_used : e->created_at;
    if (used_at == 0) {
        effective_hits = 0.0;
    } else if (now > used_at) {
        double elapsed = (double)(now - used_at);
        effective_hits *= exp2(-elapsed / (double)DS4_KVSTORE_HIT_HALF_LIFE_SECONDS);
        if (effective_hits < KV_CACHE_MIN_EFFECTIVE_HITS) effective_hits = 0.0;
    }
    double score = (effective_hits + 1.0) *
                   (double)e->tokens / (double)e->file_size;
    if (pulsar::kv_cache_reason_is_anchor(e->reason))
        score *= KV_CACHE_ANCHOR_REASON_SCORE_FACTOR;
    if (pulsar::kv_cache_incoming_supersedes_continued(e, incoming)) {
        double h = effective_hits > 0.0 ?
            effective_hits / (effective_hits + 1.0) : 0.0;
        score *= KV_CACHE_CONTINUED_PREFIX_MIN_FACTOR +
                 KV_CACHE_CONTINUED_PREFIX_HIT_FACTOR * h;
    }
    return score;
}

char *ds4_kvstore_render_tokens_text(ds4_engine *engine,
                                     const ds4_tokens *tokens,
                                     size_t *out_len) {
    pulsar::KvBuf b;
    for (int i = 0; i < tokens->len; i++) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, tokens->v[i], &len);
        b.append(piece, len);
        free(piece);
    }
    if (out_len) *out_len = b.len();
    return b.take();
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

void ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out) {
    ds4_tokens_copy(out, exact_prefix);

    ds4_tokens suffix = {};
    /* The suffix may start with DS4 chat markers such as <｜User｜> or
     * </think>, so use the rendered-chat tokenizer, not plain text BPE. */
    ds4_tokenize_rendered_chat(engine, suffix_text ? suffix_text : "", &suffix);
    pulsar::tokens_append(out, &suffix);
    ds4_tokens_free(&suffix);
}

void ds4_kvstore_load_result_free(ds4_kvstore_load_result *result) {
    if (!result) return;
    free(result->path);
    memset(result, 0, sizeof(*result));
}

/* ---- stateful facade over pulsar::KvStore ---- */

bool ds4_kvstore_open(ds4_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, ds4_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg),
                      void *log_ud) {
    return KvStore(*kc).open(dir, budget_mb, reject_different_quant, opt,
                             log_name, log, log_ud);
}

void ds4_kvstore_close(ds4_kvstore *kc) {
    KvStore(*kc).close();
}

void ds4_kvstore_clear(ds4_kvstore *kc) {
    KvStore(*kc).clear();
}

int ds4_kvstore_store_len(const ds4_kvstore *kc, int tokens) {
    return KvStore(*const_cast<ds4_kvstore *>(kc)).store_len(tokens);
}

int ds4_kvstore_chat_anchor_pos(const ds4_kvstore *kc,
                                const ds4_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id) {
    return KvStore(*const_cast<ds4_kvstore *>(kc))
        .chat_anchor_pos(prompt, user_token_id, assistant_token_id);
}

int ds4_kvstore_continued_store_target(const ds4_kvstore *kc, int live_tokens) {
    return KvStore(*const_cast<ds4_kvstore *>(kc)).continued_store_target(live_tokens);
}

void ds4_kvstore_note_store(ds4_kvstore *kc, int tokens) {
    KvStore(*kc).note_store(tokens);
}

int ds4_kvstore_suppress_continued_store(ds4_kvstore *kc, int tokens) {
    return KvStore(*kc).suppress_continued_store(tokens);
}

void ds4_kvstore_restore_suppressed_continued(ds4_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens) {
    KvStore(*kc).restore_suppressed_continued(old_tokens, suppressed_tokens);
}

bool ds4_kvstore_file_size_fits(const ds4_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out) {
    if (!kc) {
        /* No-store call: only the overflow check applies. */
        return pulsar::kv_cache_file_size_bytes(text_bytes, payload_bytes,
                                                trailer_bytes, file_bytes_out);
    }
    return KvStore(*const_cast<ds4_kvstore *>(kc))
        .file_size_fits(text_bytes, payload_bytes, trailer_bytes,
                        file_bytes_out, required_bytes_out);
}

void ds4_kvstore_evict(ds4_kvstore *kc, const ds4_tokens *live,
                       uint64_t extra_bytes,
                       const ds4_kvstore_eviction_context *incoming) {
    KvStore(*kc).evict(live, extra_bytes, incoming);
}

int ds4_kvstore_find_text_prefix(ds4_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size) {
    return KvStore(*kc).find_text_prefix(prompt_text, model_id, quant_bits, ctx_size);
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
    return KvStore(*kc).store_live_prefix_text(engine, session, tokens, store_len,
                                               reason, cache_text_override,
                                               cache_text_ext, cache_text_key,
                                               hooks, err, err_len);
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

int ds4_kvstore_try_load_text(ds4_kvstore *kc,
                              ds4_engine *engine,
                              ds4_session *session,
                              const char *prompt_text,
                              ds4_tokens *effective_prompt,
                              ds4_kvstore_load_result *result,
                              const ds4_kvstore_trailer_hooks *hooks,
                              bool responses_protocol) {
    return KvStore(*kc).try_load_text(engine, session, prompt_text,
                                      effective_prompt, result, hooks,
                                      responses_protocol);
}
