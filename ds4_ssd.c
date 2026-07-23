#include "ds4_ssd.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

static const uint64_t DS4_GIB = 1024ull * 1024ull * 1024ull;

ds4_ssd_io_plan ds4_ssd_io_plan_compute(uint64_t host_bytes,
                                        uint64_t recommended_bytes,
                                        uint64_t hot_bytes) {
    ds4_ssd_io_plan plan;
    memset(&plan, 0, sizeof(plan));
    plan.hot_bytes = hot_bytes;
    plan.use_direct_io = true;

    /*
     * Zero means unavailable for the required host/hot inputs. UINT64_MAX is
     * the public overflow sentinel, so callers can feed the result of checked
     * working-set aggregation directly into this policy.
     */
    if (host_bytes == 0 || hot_bytes == 0 ||
        host_bytes == UINT64_MAX || recommended_bytes == UINT64_MAX ||
        hot_bytes == UINT64_MAX) {
        return plan;
    }

    const uint64_t proportional_reserve = host_bytes / 8ull;
    plan.reserve_bytes =
        proportional_reserve > 8ull * DS4_GIB ?
            proportional_reserve : 8ull * DS4_GIB;
    if (host_bytes <= plan.reserve_bytes) {
        return plan;
    }

    plan.budget_bytes = host_bytes - plan.reserve_bytes;
    if (recommended_bytes != 0) {
        /*
         * floor(recommended_bytes * 9 / 10), split to avoid overflowing the
         * multiplication for every valid uint64_t input.
         */
        const uint64_t recommended_budget =
            (recommended_bytes / 10ull) * 9ull +
            ((recommended_bytes % 10ull) * 9ull) / 10ull;
        if (recommended_budget == 0) {
            plan.budget_bytes = 0;
            return plan;
        }
        if (recommended_budget < plan.budget_bytes) {
            plan.budget_bytes = recommended_budget;
        }
    }

    plan.inputs_valid = true;
    plan.use_direct_io = hot_bytes > plan.budget_bytes;
    return plan;
}

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    if (!s || !s[0] || !bytes) return false;

    size_t len = strlen(s);
    if (len > 2 &&
        (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B')) {
        len -= 2;
    }
    if (len == 0) return false;
    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }

    char numbuf[32];
    if (len >= sizeof(numbuf)) return false;
    memcpy(numbuf, s, len);
    numbuf[len] = '\0';

    errno = 0;
    unsigned long long v = strtoull(numbuf, NULL, 10);
    if (errno != 0 || v == 0 || v > UINT64_MAX / DS4_GIB) return false;

    *bytes = (uint64_t)v * DS4_GIB;
    return true;
}

bool ds4_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes) {
    if (experts) *experts = 0;
    if (bytes) *bytes = 0;
    if (!s || !s[0] || !experts || !bytes) return false;

    const size_t len = strlen(s);
    if (len > 2 &&
        (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B')) {
        return ds4_parse_gib_arg(s, bytes);
    }

    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }

    errno = 0;
    unsigned long v = strtoul(s, NULL, 10);
    if (errno != 0 || v == 0 || v > UINT32_MAX) return false;

    *experts = (uint32_t)v;
    return true;
}

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes) {
    if (bytes == 0 || per_expert_bytes == 0) return 0;
    const uint64_t experts = bytes / per_expert_bytes;
    if (experts == 0 || experts > UINT32_MAX) return 0;
    return (uint32_t)experts;
}

static uint64_t ds4_ssd_auto_cache_percent(void) {
    const char *env = getenv("DS4_SSD_AUTO_CACHE_PCT");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && errno == 0 &&
            v >= 50 && v <= 95) {
            return (uint64_t)v;
        }
        fprintf(stderr,
                "ds4: invalid DS4_SSD_AUTO_CACHE_PCT=%s (want 50..95); "
                "using default\n",
                env);
    }
    /*
     * Decode on the ROCm streaming path is SSD-bandwidth bound: every routed
     * expert miss is a random NVMe read, so a larger resident expert cache is
     * the biggest decode-throughput lever.  BUT the expert cache lives in the
     * same physical RAM as the OS on a unified-memory APU, and transient
     * spikes (pinned read/upload staging, the prefill headroom, cache growth)
     * ride on top of the steady-state plan.  Pushing the split too high runs
     * the machine out of RAM and trips the Linux OOM killer.  80% was measured
     * safe here; opt into more only deliberately via DS4_SSD_AUTO_CACHE_PCT.
     */
    return 80;
}

bool ds4_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             ds4_ssd_cache_plan *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (recommended_bytes == 0 || per_expert_bytes == 0) return false;

    const uint64_t pct = ds4_ssd_auto_cache_percent();
    out->model_target_bytes =
        recommended_bytes > UINT64_MAX / pct ?
            UINT64_MAX : (recommended_bytes * pct) / 100ull;
    if (out->model_target_bytes > non_routed_bytes) {
        out->cache_bytes = out->model_target_bytes - non_routed_bytes;
    }

    uint64_t cache_experts = out->cache_bytes / per_expert_bytes;
    if (cache_experts == 0) cache_experts = 1;
    if (max_model_experts != 0 && cache_experts > max_model_experts) {
        cache_experts = max_model_experts;
    }
    if (cache_experts > UINT32_MAX) cache_experts = UINT32_MAX;

    out->cache_experts = (uint32_t)cache_experts;
    out->effective_cache_bytes = cache_experts * per_expert_bytes;
    return out->cache_experts != 0;
}

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock,
                                 uint64_t             bytes) {
    if (!lock) return false;
    lock->ptr = NULL;
    lock->bytes = 0;
    if (bytes == 0) return true;
    if (bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr,
                "ds4: --simulate-used-memory is too large for this process\n");
        return false;
    }

    void *ptr = mmap(NULL,
                     (size_t)bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1,
                     0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr,
                "ds4: --simulate-used-memory mmap %.2f GiB failed: %s\n",
                (double)bytes / (double)DS4_GIB,
                strerror(errno));
        return false;
    }

    const long page_long = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_long > 0 ? (uint64_t)page_long : 4096ull;
    const uint64_t chunk_bytes = 256ull * 1024ull * 1024ull;
    volatile unsigned char *p = (volatile unsigned char *)ptr;

    /*
     * Touch and lock in bounded chunks.  A single very large mlock() is harder
     * to diagnose when it fails and can create long uninterruptible VM work on
     * macOS; chunking mirrors the standalone diagnostic utility.
     */
    uint64_t locked = 0;
    for (uint64_t off = 0; off < bytes; off += chunk_bytes) {
        uint64_t len = bytes - off;
        if (len > chunk_bytes) len = chunk_bytes;

        for (uint64_t pos = off; pos < off + len; pos += page) {
            p[pos] = (unsigned char)(pos / page);
        }
        if (len != 0) p[off + len - 1u] = 1;

        if (mlock((void *)(p + off), (size_t)len) != 0) {
            fprintf(stderr,
                    "ds4: --simulate-used-memory mlock failed after %.2f/%.2f GiB: %s\n",
                    (double)locked / (double)DS4_GIB,
                    (double)bytes / (double)DS4_GIB,
                    strerror(errno));
            if (locked != 0) munlock(ptr, (size_t)locked);
            munmap(ptr, (size_t)bytes);
            return false;
        }
        locked += len;
    }

    lock->ptr = ptr;
    lock->bytes = bytes;
    fprintf(stderr,
            "ds4: simulated used memory: locked %.2f GiB before model load\n",
            (double)bytes / (double)DS4_GIB);
    return true;
}

void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock) {
    if (!lock || !lock->ptr || lock->bytes == 0) return;
    munlock(lock->ptr, (size_t)lock->bytes);
    munmap(lock->ptr, (size_t)lock->bytes);
    lock->ptr = NULL;
    lock->bytes = 0;
}

/* =========================================================================
 * SSD streaming expert-bundle sidecar ("DSEB" version 1).
 * =========================================================================
 *
 * See ds4_ssd.h for the format. The layout constants below must not change:
 * they keep the file byte-compatible with the DwarfStar Swift port, so one
 * bundle on disk serves both implementations.
 */

#define DS4_EXPERT_BUNDLE_MAGIC   0x42455344u   /* "DSEB" little-endian */
#define DS4_EXPERT_BUNDLE_VERSION 1u
#define DS4_EXPERT_BUNDLE_ALIGN   4096ull
#define DS4_EXPERT_BUNDLE_SUFFIX  ".expbundle"

/*
 * DSEB v1 deliberately leaves the rest of its first 4 KiB page zero-filled.
 * Original Swift readers consume exactly 56 + 8*layer_count bytes, then round
 * the record base up to 4 KiB, so an optional extension in that padding is
 * invisible to them and does not move a single record.  DSEI records a strong
 * full-tensor fingerprint plus the source file identity.  Matching identity
 * avoids a full payload scan; hashes are reread only after a move/copy or
 * metadata change.  A legacy bundle without DSEI is compared byte-for-byte
 * with the GGUF once before this extension is installed best-effort.
 */
#define DS4_EXPERT_BUNDLE_ID_MAGIC   0x49455344u /* "DSEI" little-endian */
#define DS4_EXPERT_BUNDLE_ID_VERSION 1u
#define DS4_EXPERT_BUNDLE_ID_FIXED   72ull
#define DS4_EXPERT_BUNDLE_ID_HASHES_PER_LAYER 3ull
#define DS4_EXPERT_BUNDLE_HASH_CHUNK (1024u * 1024u)
#define DS4_FNV1A64_OFFSET 0xcbf29ce484222325ull
#define DS4_FNV1A64_PRIME  0x100000001b3ull

static uint64_t expert_bundle_align_up(uint64_t v) {
    return (v + DS4_EXPERT_BUNDLE_ALIGN - 1) /
           DS4_EXPERT_BUNDLE_ALIGN * DS4_EXPERT_BUNDLE_ALIGN;
}

static uint64_t expert_bundle_header_bytes(uint32_t layer_count) {
    return 56ull + (uint64_t)layer_count * 8ull;
}

static uint64_t expert_bundle_align8(uint64_t v) {
    return (v + 7ull) & ~7ull;
}

static uint64_t expert_bundle_identity_bytes(uint32_t layer_count) {
    const uint64_t hashes =
        (uint64_t)layer_count * DS4_EXPERT_BUNDLE_ID_HASHES_PER_LAYER;
    if (hashes > (UINT64_MAX - DS4_EXPERT_BUNDLE_ID_FIXED - 8ull) / 8ull) {
        return 0;
    }
    return DS4_EXPERT_BUNDLE_ID_FIXED + hashes * 8ull + 8ull;
}

static bool expert_bundle_identity_fits(uint32_t layer_count) {
    const uint64_t hb = expert_bundle_header_bytes(layer_count);
    const uint64_t data_base = expert_bundle_align_up(hb);
    const uint64_t offset = expert_bundle_align8(hb);
    const uint64_t bytes = expert_bundle_identity_bytes(layer_count);
    return bytes != 0 && offset <= data_base && bytes <= data_base - offset;
}

static void expert_bundle_put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static void expert_bundle_put_u64(uint8_t *p, uint64_t v) {
    expert_bundle_put_u32(p, (uint32_t)v);
    expert_bundle_put_u32(p + 4, (uint32_t)(v >> 32));
}

static uint32_t expert_bundle_get_u32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t expert_bundle_get_u64(const uint8_t *p) {
    return (uint64_t)expert_bundle_get_u32(p) |
           ((uint64_t)expert_bundle_get_u32(p + 4) << 32);
}

static uint64_t expert_bundle_fnv1a_update(uint64_t h,
                                           const void *data,
                                           size_t      bytes) {
    const uint8_t *p = data;
    for (size_t i = 0; i < bytes; i++) {
        h = (h ^ p[i]) * DS4_FNV1A64_PRIME;
    }
    return h;
}

static bool expert_bundle_pread_full(int      fd,
                                     void    *dst,
                                     uint64_t bytes,
                                     uint64_t offset) {
    uint8_t *p = dst;
    uint64_t pos = 0;
    while (pos < bytes) {
        ssize_t n;
        do {
            n = pread(fd, p + pos, (size_t)(bytes - pos), (off_t)(offset + pos));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return false;
        pos += (uint64_t)n;
    }
    return true;
}

static bool expert_bundle_write_full(int fd, const void *src, uint64_t bytes) {
    const uint8_t *p = src;
    uint64_t pos = 0;
    while (pos < bytes) {
        ssize_t n;
        do {
            n = write(fd, p + pos, (size_t)(bytes - pos));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return false;
        pos += (uint64_t)n;
    }
    return true;
}

static bool expert_bundle_pwrite_full(int fd, const void *src, uint64_t bytes,
                                      uint64_t offset) {
    const uint8_t *p = src;
    uint64_t pos = 0;
    while (pos < bytes) {
        ssize_t n;
        do {
            n = pwrite(fd, p + pos, (size_t)(bytes - pos),
                       (off_t)(offset + pos));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) return false;
        pos += (uint64_t)n;
    }
    return true;
}

typedef struct {
    uint64_t device;
    uint64_t inode;
    uint64_t size;
    uint64_t mtime_sec;
    uint32_t mtime_nsec;
    uint64_t ctime_sec;
    uint32_t ctime_nsec;
} expert_bundle_model_identity;

static bool expert_bundle_model_identity_get(
        int                           fd,
        expert_bundle_model_identity *identity) {
    if (fd < 0 || !identity) return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) return false;
    memset(identity, 0, sizeof(*identity));
    identity->device = (uint64_t)st.st_dev;
    identity->inode = (uint64_t)st.st_ino;
    identity->size = (uint64_t)st.st_size;
#if defined(__APPLE__)
    identity->mtime_sec = (uint64_t)(int64_t)st.st_mtimespec.tv_sec;
    identity->mtime_nsec = (uint32_t)st.st_mtimespec.tv_nsec;
    identity->ctime_sec = (uint64_t)(int64_t)st.st_ctimespec.tv_sec;
    identity->ctime_nsec = (uint32_t)st.st_ctimespec.tv_nsec;
#elif defined(__linux__)
    identity->mtime_sec = (uint64_t)(int64_t)st.st_mtim.tv_sec;
    identity->mtime_nsec = (uint32_t)st.st_mtim.tv_nsec;
    identity->ctime_sec = (uint64_t)(int64_t)st.st_ctim.tv_sec;
    identity->ctime_nsec = (uint32_t)st.st_ctim.tv_nsec;
#else
    identity->mtime_sec = (uint64_t)(int64_t)st.st_mtime;
    identity->ctime_sec = (uint64_t)(int64_t)st.st_ctime;
#endif
    return identity->mtime_nsec < 1000000000u &&
           identity->ctime_nsec < 1000000000u;
}

static bool expert_bundle_model_identity_equal(
        const expert_bundle_model_identity *a,
        const expert_bundle_model_identity *b) {
    return a && b &&
           a->device == b->device &&
           a->inode == b->inode &&
           a->size == b->size &&
           a->mtime_sec == b->mtime_sec &&
           a->mtime_nsec == b->mtime_nsec &&
           a->ctime_sec == b->ctime_sec &&
           a->ctime_nsec == b->ctime_nsec;
}

static bool expert_bundle_stat_same_file_version(const struct stat *a,
                                                 const struct stat *b) {
    if (!a || !b || !S_ISREG(a->st_mode) || !S_ISREG(b->st_mode) ||
        a->st_dev != b->st_dev || a->st_ino != b->st_ino ||
        a->st_size != b->st_size) {
        return false;
    }
#if defined(__APPLE__)
    return a->st_mtimespec.tv_sec == b->st_mtimespec.tv_sec &&
           a->st_mtimespec.tv_nsec == b->st_mtimespec.tv_nsec &&
           a->st_ctimespec.tv_sec == b->st_ctimespec.tv_sec &&
           a->st_ctimespec.tv_nsec == b->st_ctimespec.tv_nsec;
#elif defined(__linux__)
    return a->st_mtim.tv_sec == b->st_mtim.tv_sec &&
           a->st_mtim.tv_nsec == b->st_mtim.tv_nsec &&
           a->st_ctim.tv_sec == b->st_ctim.tv_sec &&
           a->st_ctim.tv_nsec == b->st_ctim.tv_nsec;
#else
    return a->st_mtime == b->st_mtime && a->st_ctime == b->st_ctime;
#endif
}

typedef struct {
    uint64_t canonical;
    uint64_t swift_v1;
} expert_bundle_layer_fingerprints;

/*
 * FNV-1a over the first 4 KiB of a layer's gate-experts tensor in the model
 * file: the bundle must match the MODEL BYTES, not just its size and shape.
 *
 * Early Swift writers accidentally used 0x10000000001b3 (2^48 + 0x1b3)
 * instead of the FNV-1a prime 0x100000001b3 (2^40 + 0x1b3).  Keep both
 * fingerprints so those already-large v1 bundles remain readable.  New
 * bundles always write the canonical fingerprint.
 */
static bool expert_bundle_layer_hashes(
        int                               model_fd,
        uint64_t                          gate_offset,
        uint64_t                          gate_tensor_bytes,
        expert_bundle_layer_fingerprints *hashes_out) {
    uint8_t buf[4096];
    uint64_t n = gate_tensor_bytes < sizeof(buf) ? gate_tensor_bytes : sizeof(buf);
    if (n == 0 ||
        !expert_bundle_pread_full(model_fd, buf, n, gate_offset)) {
        return false;
    }
    uint64_t canonical = 0xcbf29ce484222325ull;
    uint64_t swift_v1 = canonical;
    for (uint64_t i = 0; i < n; i++) {
        canonical = (canonical ^ buf[i]) * 0x100000001b3ull;
        swift_v1 = (swift_v1 ^ buf[i]) * 0x10000000001b3ull;
    }
    hashes_out->canonical = canonical;
    hashes_out->swift_v1 = swift_v1;
    return true;
}

typedef enum {
    EXPERT_BUNDLE_ID_ABSENT = 0,
    EXPERT_BUNDLE_ID_VALID = 1,
    EXPERT_BUNDLE_ID_INVALID = 2,
} expert_bundle_identity_status;

static bool expert_bundle_identity_encode(
        uint8_t                            *dst,
        uint64_t                            capacity,
        const expert_bundle_model_identity *identity,
        uint32_t                            layer_count,
        const uint64_t                     *strong_hashes,
        bool                                publish_magic) {
    const uint64_t bytes = expert_bundle_identity_bytes(layer_count);
    const uint64_t n_hashes =
        (uint64_t)layer_count * DS4_EXPERT_BUNDLE_ID_HASHES_PER_LAYER;
    if (!dst || !identity || !strong_hashes || bytes == 0 || bytes > capacity ||
        bytes > UINT32_MAX || identity->mtime_nsec >= 1000000000u ||
        identity->ctime_nsec >= 1000000000u) {
        return false;
    }
    memset(dst, 0, (size_t)bytes);
    if (publish_magic) {
        expert_bundle_put_u32(dst, DS4_EXPERT_BUNDLE_ID_MAGIC);
    }
    expert_bundle_put_u32(dst + 4, DS4_EXPERT_BUNDLE_ID_VERSION);
    expert_bundle_put_u32(dst + 8, (uint32_t)bytes);
    expert_bundle_put_u32(dst + 12, layer_count);
    expert_bundle_put_u64(dst + 16, identity->device);
    expert_bundle_put_u64(dst + 24, identity->inode);
    expert_bundle_put_u64(dst + 32, identity->size);
    expert_bundle_put_u64(dst + 40, identity->mtime_sec);
    expert_bundle_put_u32(dst + 48, identity->mtime_nsec);
    expert_bundle_put_u64(dst + 56, identity->ctime_sec);
    expert_bundle_put_u32(dst + 64, identity->ctime_nsec);
    for (uint64_t i = 0; i < n_hashes; i++) {
        expert_bundle_put_u64(dst + DS4_EXPERT_BUNDLE_ID_FIXED + i * 8ull,
                              strong_hashes[i]);
    }
    const uint64_t checksum_offset = bytes - 8ull;
    const uint64_t checksum = expert_bundle_fnv1a_update(
        DS4_FNV1A64_OFFSET,
        dst + 4,
        (size_t)(checksum_offset - 4ull));
    expert_bundle_put_u64(dst + checksum_offset, checksum);
    return true;
}

static expert_bundle_identity_status expert_bundle_identity_decode(
        const uint8_t                       *header,
        uint64_t                             header_capacity,
        uint64_t                             legacy_header_bytes,
        uint32_t                             layer_count,
        uint64_t                             model_size,
        expert_bundle_model_identity        *identity,
        uint64_t                            *strong_hashes) {
    const uint64_t offset = expert_bundle_align8(legacy_header_bytes);
    const uint64_t expected_bytes = expert_bundle_identity_bytes(layer_count);
    if (!header || expected_bytes == 0 || offset > header_capacity ||
        expected_bytes > header_capacity - offset) {
        return EXPERT_BUNDLE_ID_INVALID;
    }
    const uint8_t *src = header + offset;
    if (expert_bundle_get_u32(src) != DS4_EXPERT_BUNDLE_ID_MAGIC) {
        return EXPERT_BUNDLE_ID_ABSENT;
    }
    if (!identity || !strong_hashes ||
        expert_bundle_get_u32(src + 4) != DS4_EXPERT_BUNDLE_ID_VERSION ||
        expert_bundle_get_u32(src + 8) != expected_bytes ||
        expert_bundle_get_u32(src + 12) != layer_count ||
        expert_bundle_get_u64(src + 32) != model_size) {
        return EXPERT_BUNDLE_ID_INVALID;
    }
    const uint64_t checksum_offset = expected_bytes - 8ull;
    const uint64_t stored_checksum =
        expert_bundle_get_u64(src + checksum_offset);
    const uint64_t checksum = expert_bundle_fnv1a_update(
        DS4_FNV1A64_OFFSET,
        src + 4,
        (size_t)(checksum_offset - 4ull));
    if (stored_checksum != checksum) return EXPERT_BUNDLE_ID_INVALID;

    memset(identity, 0, sizeof(*identity));
    identity->device = expert_bundle_get_u64(src + 16);
    identity->inode = expert_bundle_get_u64(src + 24);
    identity->size = expert_bundle_get_u64(src + 32);
    identity->mtime_sec = expert_bundle_get_u64(src + 40);
    identity->mtime_nsec = expert_bundle_get_u32(src + 48);
    identity->ctime_sec = expert_bundle_get_u64(src + 56);
    identity->ctime_nsec = expert_bundle_get_u32(src + 64);
    if (identity->mtime_nsec >= 1000000000u ||
        identity->ctime_nsec >= 1000000000u) {
        return EXPERT_BUNDLE_ID_INVALID;
    }
    const uint64_t n_hashes =
        (uint64_t)layer_count * DS4_EXPERT_BUNDLE_ID_HASHES_PER_LAYER;
    for (uint64_t i = 0; i < n_hashes; i++) {
        strong_hashes[i] = expert_bundle_get_u64(
            src + DS4_EXPERT_BUNDLE_ID_FIXED + i * 8ull);
    }
    return EXPERT_BUNDLE_ID_VALID;
}

static bool expert_bundle_identity_write_fd(
        int                                 fd,
        uint64_t                            legacy_header_bytes,
        uint64_t                            data_base,
        const expert_bundle_model_identity *identity,
        uint32_t                            layer_count,
        const uint64_t                     *strong_hashes,
        bool                                crash_safe_publish) {
    const uint64_t offset = expert_bundle_align8(legacy_header_bytes);
    const uint64_t bytes = expert_bundle_identity_bytes(layer_count);
    if (fd < 0 || bytes == 0 || offset > data_base || bytes > data_base - offset ||
        bytes > (uint64_t)SIZE_MAX) {
        return false;
    }
    uint8_t *encoded = malloc((size_t)bytes);
    if (!encoded ||
        !expert_bundle_identity_encode(encoded, bytes, identity, layer_count,
                                       strong_hashes,
                                       !crash_safe_publish)) {
        free(encoded);
        return false;
    }
    bool ok = expert_bundle_pwrite_full(fd, encoded, bytes, offset);
    if (ok && crash_safe_publish) {
        ok = fsync(fd) == 0;
        uint8_t magic[4];
        expert_bundle_put_u32(magic, DS4_EXPERT_BUNDLE_ID_MAGIC);
        if (ok) ok = expert_bundle_pwrite_full(fd, magic, sizeof(magic), offset);
    }
    if (ok && crash_safe_publish) ok = fsync(fd) == 0;
    free(encoded);
    return ok;
}

static bool expert_bundle_identity_install_best_effort(
        const char                          *path,
        const struct stat                   *validated_stat,
        uint64_t                             legacy_header_bytes,
        uint64_t                             data_base,
        const expert_bundle_model_identity  *identity,
        uint32_t                             layer_count,
        const uint64_t                      *strong_hashes) {
    if (!path || !validated_stat || !S_ISREG(validated_stat->st_mode)) {
        return false;
    }
    const int fd = open(path, O_RDWR);
    if (fd < 0) return false;
    struct stat reopened_stat;
    bool ok = fstat(fd, &reopened_stat) == 0 &&
              expert_bundle_stat_same_file_version(validated_stat,
                                                   &reopened_stat);
    if (ok) {
        ok = expert_bundle_identity_write_fd(fd,
                                             legacy_header_bytes,
                                             data_base,
                                             identity,
                                             layer_count,
                                             strong_hashes,
                                             true);
    }
    if (close(fd) != 0) ok = false;
    return ok;
}

static bool expert_bundle_hash_range(int       fd,
                                     uint64_t  offset,
                                     uint64_t  bytes,
                                     uint8_t  *scratch,
                                     size_t    scratch_bytes,
                                     uint64_t *hash_out) {
    if (fd < 0 || !scratch || scratch_bytes == 0 || !hash_out) return false;
    uint64_t hash = DS4_FNV1A64_OFFSET;
    uint64_t pos = 0;
    while (pos < bytes) {
        size_t n = scratch_bytes;
        if ((uint64_t)n > bytes - pos) n = (size_t)(bytes - pos);
        if (!expert_bundle_pread_full(fd, scratch, n, offset + pos)) {
            return false;
        }
        hash = expert_bundle_fnv1a_update(hash, scratch, n);
        pos += n;
    }
    *hash_out = hash;
    return true;
}

static bool expert_bundle_hash_model_full(
        int                            model_fd,
        uint32_t                       layer_count,
        uint32_t                       n_expert,
        uint64_t                       gate_bytes,
        uint64_t                       up_bytes,
        uint64_t                       down_bytes,
        const ds4_expert_bundle_layer *layers,
        uint64_t                      *strong_hashes) {
    if (model_fd < 0 || !layers || !strong_hashes || n_expert == 0 ||
        gate_bytes > UINT64_MAX / n_expert ||
        up_bytes > UINT64_MAX / n_expert ||
        down_bytes > UINT64_MAX / n_expert) {
        return false;
    }
    uint8_t *scratch = malloc(DS4_EXPERT_BUNDLE_HASH_CHUNK);
    if (!scratch) return false;
    const uint64_t tensor_bytes[3] = {
        gate_bytes * n_expert,
        up_bytes * n_expert,
        down_bytes * n_expert,
    };
    bool ok = true;
    for (uint32_t il = 0; ok && il < layer_count; il++) {
        const uint64_t offsets[3] = {
            layers[il].gate_offset,
            layers[il].up_offset,
            layers[il].down_offset,
        };
        for (uint32_t kind = 0; ok && kind < 3; kind++) {
            ok = expert_bundle_hash_range(
                model_fd,
                offsets[kind],
                tensor_bytes[kind],
                scratch,
                DS4_EXPERT_BUNDLE_HASH_CHUNK,
                &strong_hashes[(uint64_t)il * 3ull + kind]);
        }
    }
    free(scratch);
    return ok;
}

static bool expert_bundle_verify_legacy_payload(
        int                            bundle_fd,
        int                            model_fd,
        uint64_t                       data_base,
        uint64_t                       record,
        uint32_t                       layer_count,
        uint32_t                       n_expert,
        uint64_t                       gate_bytes,
        uint64_t                       up_bytes,
        uint64_t                       down_bytes,
        const ds4_expert_bundle_layer *layers,
        uint64_t                      *strong_hashes) {
    if (bundle_fd < 0 || model_fd < 0 || !layers || !strong_hashes) return false;
    uint8_t *model_buf = malloc(DS4_EXPERT_BUNDLE_HASH_CHUNK);
    uint8_t *bundle_buf = malloc(DS4_EXPERT_BUNDLE_HASH_CHUNK);
    if (!model_buf || !bundle_buf) {
        free(model_buf);
        free(bundle_buf);
        return false;
    }
    const uint64_t slab_bytes[3] = { gate_bytes, up_bytes, down_bytes };
    const uint64_t record_kind_offset[3] = {
        0,
        gate_bytes,
        gate_bytes + up_bytes,
    };
    bool ok = true;
    for (uint32_t il = 0; ok && il < layer_count; il++) {
        const uint64_t model_kind_offset[3] = {
            layers[il].gate_offset,
            layers[il].up_offset,
            layers[il].down_offset,
        };
        uint64_t layer_hashes[3] = {
            DS4_FNV1A64_OFFSET,
            DS4_FNV1A64_OFFSET,
            DS4_FNV1A64_OFFSET,
        };
        for (uint32_t expert = 0; ok && expert < n_expert; expert++) {
            const uint64_t bundle_record = data_base +
                ((uint64_t)il * n_expert + expert) * record;
            for (uint32_t kind = 0; ok && kind < 3; kind++) {
                uint64_t pos = 0;
                while (ok && pos < slab_bytes[kind]) {
                    size_t n = DS4_EXPERT_BUNDLE_HASH_CHUNK;
                    if ((uint64_t)n > slab_bytes[kind] - pos) {
                        n = (size_t)(slab_bytes[kind] - pos);
                    }
                    const uint64_t model_offset = model_kind_offset[kind] +
                        (uint64_t)expert * slab_bytes[kind] + pos;
                    const uint64_t bundle_offset = bundle_record +
                        record_kind_offset[kind] + pos;
                    ok = expert_bundle_pread_full(model_fd, model_buf, n,
                                                  model_offset) &&
                         expert_bundle_pread_full(bundle_fd, bundle_buf, n,
                                                  bundle_offset) &&
                         memcmp(model_buf, bundle_buf, n) == 0;
                    if (ok) {
                        layer_hashes[kind] = expert_bundle_fnv1a_update(
                            layer_hashes[kind], model_buf, n);
                    }
                    pos += n;
                }
            }
        }
        if (ok) {
            memcpy(&strong_hashes[(uint64_t)il * 3ull],
                   layer_hashes,
                   sizeof(layer_hashes));
        }
    }
    free(model_buf);
    free(bundle_buf);
    return ok;
}

static void expert_bundle_disable(ds4_expert_bundle *b) {
    memset(b, 0, sizeof(*b));
    b->fd = -1;
}

uint64_t ds4_expert_bundle_record_offset(const ds4_expert_bundle *b,
                                         uint32_t                 layer,
                                         uint32_t                 expert) {
    return b->data_base +
           ((uint64_t)(layer - b->layer_lo) * b->n_expert + expert) * b->record;
}

/* Validate and open an existing bundle file. false = absent or mismatched. */
static bool expert_bundle_open_existing(ds4_expert_bundle             *b,
                                        const char                    *path,
                                        int                            model_fd,
                                        uint64_t                       model_size,
                                        uint32_t                       layer_lo,
                                        uint32_t                       layer_count,
                                        uint32_t                       n_expert,
                                        uint64_t                       gate_bytes,
                                        uint64_t                       up_bytes,
                                        uint64_t                       down_bytes,
                                        const ds4_expert_bundle_layer *layers,
                                        const expert_bundle_layer_fingerprints
                                                                       *hashes,
                                        bool                            use_direct_io) {
    const uint64_t hb = expert_bundle_header_bytes(layer_count);
    const uint64_t data_base = expert_bundle_align_up(hb);
    const uint64_t record =
        expert_bundle_align_up(gate_bytes + up_bytes + down_bytes);
    const uint64_t n_records = (uint64_t)layer_count * n_expert;
    const bool identity_supported =
        expert_bundle_identity_fits(layer_count);
    if (!b || !path || model_fd < 0 || !layers || !hashes ||
        record == 0 || n_records > UINT64_MAX / record ||
        data_base > UINT64_MAX - n_records * record) {
        return false;
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
#ifdef __APPLE__
    if (use_direct_io) (void)fcntl(fd, F_NOCACHE, 1);
#else
    (void)use_direct_io;
#endif
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0 ||
        (uint64_t)st.st_size < data_base + n_records * record) {
        close(fd);
        fprintf(stderr, "ds4: expert bundle truncated: %s\n", path);
        return false;
    }
    uint8_t *head = malloc((size_t)data_base);
    if (!head || !expert_bundle_pread_full(fd, head, data_base, 0)) {
        free(head);
        close(fd);
        return false;
    }
    if (expert_bundle_get_u32(head) != DS4_EXPERT_BUNDLE_MAGIC ||
        expert_bundle_get_u32(head + 4) != DS4_EXPERT_BUNDLE_VERSION ||
        expert_bundle_get_u64(head + 8) != model_size ||
        expert_bundle_get_u32(head + 16) != layer_lo ||
        expert_bundle_get_u32(head + 20) != layer_count ||
        expert_bundle_get_u32(head + 24) != n_expert ||
        expert_bundle_get_u64(head + 32) != gate_bytes ||
        expert_bundle_get_u64(head + 40) != up_bytes ||
        expert_bundle_get_u64(head + 48) != down_bytes) {
        free(head);
        close(fd);
        fprintf(stderr,
                "ds4: expert bundle incompatible (header/model changed): %s\n",
                path);
        return false;
    }
    bool canonical_match = true;
    bool swift_v1_match = true;
    uint32_t invalid_layer = UINT32_MAX;
    for (uint32_t i = 0; i < layer_count; i++) {
        const uint64_t stored =
            expert_bundle_get_u64(head + 56 + (uint64_t)i * 8);
        const bool canonical = stored == hashes[i].canonical;
        const bool swift_v1 = stored == hashes[i].swift_v1;
        canonical_match = canonical_match && canonical;
        swift_v1_match = swift_v1_match && swift_v1;
        if (!canonical && !swift_v1 && invalid_layer == UINT32_MAX) {
            invalid_layer = layer_lo + i;
        }
    }
    if (!canonical_match && !swift_v1_match) {
        free(head);
        close(fd);
        if (invalid_layer != UINT32_MAX) {
            fprintf(stderr,
                    "ds4: expert bundle incompatible (layer %u changed): %s\n",
                    invalid_layer,
                    path);
        } else {
            fprintf(stderr,
                    "ds4: expert bundle incompatible "
                    "(mixed fingerprint schemes): %s\n",
                    path);
        }
        return false;
    }

    /* DSEI is optional. Very large layer tables consume the complete first
     * page, so preserve the exact v1 validation/read behavior instead of
     * rejecting, rebuilding, or imposing a full scan on every open. */
    if (!identity_supported) {
        free(head);
        if (swift_v1_match && !canonical_match) {
            fprintf(stderr,
                    "ds4: expert bundle accepted with legacy Swift v1 "
                    "fingerprints: %s\n",
                    path);
        }
        b->fd = fd;
        b->layer_lo = layer_lo;
        b->layer_count = layer_count;
        b->n_expert = n_expert;
        b->gate_bytes = gate_bytes;
        b->up_bytes = up_bytes;
        b->down_bytes = down_bytes;
        b->data_base = data_base;
        b->record = record;
        snprintf(b->path, sizeof(b->path), "%s", path);
        return true;
    }

    const uint64_t n_strong =
        (uint64_t)layer_count * DS4_EXPERT_BUNDLE_ID_HASHES_PER_LAYER;
    uint64_t *stored_strong = malloc((size_t)n_strong * sizeof(*stored_strong));
    uint64_t *current_strong = malloc((size_t)n_strong * sizeof(*current_strong));
    expert_bundle_model_identity stored_identity;
    expert_bundle_model_identity identity_before;
    expert_bundle_model_identity identity_after;
    if (!stored_strong || !current_strong ||
        !expert_bundle_model_identity_get(model_fd, &identity_before) ||
        identity_before.size != model_size) {
        free(stored_strong);
        free(current_strong);
        free(head);
        close(fd);
        return false;
    }
    const expert_bundle_identity_status id_status =
        expert_bundle_identity_decode(head,
                                      data_base,
                                      hb,
                                      layer_count,
                                      model_size,
                                      &stored_identity,
                                      stored_strong);
    free(head);

    bool content_ok = false;
    bool refresh_identity = false;
    if (id_status == EXPERT_BUNDLE_ID_VALID &&
        expert_bundle_model_identity_equal(&stored_identity,
                                           &identity_before)) {
        /* Normal startup: no model or bundle payload read. */
        content_ok = true;
    } else if (id_status == EXPERT_BUNDLE_ID_VALID) {
        /* The model moved, was cloned, or changed. Hash every routed tensor
         * once; identical copies reuse the sidecar and refresh only DSEI. */
        content_ok = expert_bundle_hash_model_full(model_fd,
                                                   layer_count,
                                                   n_expert,
                                                   gate_bytes,
                                                   up_bytes,
                                                   down_bytes,
                                                   layers,
                                                   current_strong) &&
                     expert_bundle_model_identity_get(model_fd,
                                                      &identity_after) &&
                     expert_bundle_model_identity_equal(&identity_before,
                                                        &identity_after) &&
                     memcmp(stored_strong,
                            current_strong,
                            (size_t)n_strong * sizeof(*stored_strong)) == 0;
        refresh_identity = content_ok;
        if (!content_ok) {
            fprintf(stderr,
                    "ds4: expert bundle incompatible "
                    "(full expert tensors changed): %s\n",
                    path);
        }
    } else {
        /* Old C/Swift v1 files carry only a 4 KiB gate prefix hash. Before
         * trusting one by default, compare every payload byte to the GGUF.
         * Successful files are upgraded in-place without moving records. */
        content_ok = expert_bundle_verify_legacy_payload(fd,
                                                         model_fd,
                                                         data_base,
                                                         record,
                                                         layer_count,
                                                         n_expert,
                                                         gate_bytes,
                                                         up_bytes,
                                                         down_bytes,
                                                         layers,
                                                         current_strong) &&
                     expert_bundle_model_identity_get(model_fd,
                                                      &identity_after) &&
                     expert_bundle_model_identity_equal(&identity_before,
                                                        &identity_after);
        refresh_identity = content_ok;
        if (!content_ok) {
            fprintf(stderr,
                    "ds4: expert bundle incompatible "
                    "(legacy payload differs from model): %s\n",
                    path);
        }
    }
    if (!content_ok) {
        free(stored_strong);
        free(current_strong);
        close(fd);
        return false;
    }
    if (refresh_identity) {
        const bool installed = expert_bundle_identity_install_best_effort(
            path,
            &st,
            hb,
            data_base,
            &identity_after,
            layer_count,
            current_strong);
        if (!installed && id_status != EXPERT_BUNDLE_ID_VALID) {
            fprintf(stderr,
                    "ds4: expert bundle legacy payload verified, but the "
                    "DSEI upgrade could not be persisted; full verification "
                    "will repeat next load: %s\n",
                    path);
        }
    }
    free(stored_strong);
    free(current_strong);

    if (swift_v1_match && !canonical_match) {
        fprintf(stderr,
                "ds4: expert bundle accepted with legacy Swift v1 "
                "fingerprints: %s\n",
                path);
    }

    b->fd = fd;
    b->layer_lo = layer_lo;
    b->layer_count = layer_count;
    b->n_expert = n_expert;
    b->gate_bytes = gate_bytes;
    b->up_bytes = up_bytes;
    b->down_bytes = down_bytes;
    b->data_base = data_base;
    b->record = record;
    snprintf(b->path, sizeof(b->path), "%s", path);
    return true;
}

/*
 * One-time build: stream every expert's three slabs from the GGUF into
 * contiguous records (.tmp + rename, so torn files are impossible). Refuses
 * when free disk space is short. Source reads are sequential WITHIN each
 * tensor (expert e then e+1), so the build runs near sequential speed.
 */
static bool expert_bundle_build(const char                    *path,
                                int                            model_fd,
                                uint64_t                       model_size,
                                uint32_t                       layer_lo,
                                uint32_t                       layer_count,
                                uint32_t                       n_expert,
                                uint64_t                       gate_bytes,
                                uint64_t                       up_bytes,
                                uint64_t                       down_bytes,
                                const ds4_expert_bundle_layer *layers,
                                const expert_bundle_layer_fingerprints
                                                              *hashes) {
    const uint64_t hb = expert_bundle_header_bytes(layer_count);
    const uint64_t data_base = expert_bundle_align_up(hb);
    const bool identity_supported =
        expert_bundle_identity_fits(layer_count);
    const uint64_t record =
        expert_bundle_align_up(gate_bytes + up_bytes + down_bytes);
    const uint64_t n_records = (uint64_t)layer_count * n_expert;
    if (record == 0 || n_records > UINT64_MAX / record ||
        data_base > UINT64_MAX - n_records * record) {
        return false;
    }
    const uint64_t total_bytes = data_base + n_records * record;

    char dir[sizeof(((ds4_expert_bundle *)0)->path)];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = '\0';
    } else {
        snprintf(dir, sizeof(dir), ".");
    }
    /* Keep an incompatible sidecar recoverable until its replacement has
     * been completely written and synced. The free-space check therefore
     * requires room for the complete temporary file in addition to it. */
    struct statvfs vfs;
    if (statvfs(dir, &vfs) == 0) {
        const uint64_t free_bytes = (uint64_t)vfs.f_bavail * vfs.f_frsize;
        if (free_bytes <= total_bytes + DS4_GIB) {
            fprintf(stderr,
                    "ds4: expert bundle needs ~%.1f GiB free but only %.1f GiB "
                    "are available in %s; skipping (the Trash counts!)\n",
                    (double)total_bytes / (double)DS4_GIB,
                    (double)free_bytes / (double)DS4_GIB,
                    dir);
            return false;
        }
    } else {
        /* Some network/FUSE mounts cannot answer; unknown is not "full". */
        fprintf(stderr,
                "ds4: expert bundle free-space check unavailable for %s (%s); "
                "proceeding\n",
                dir, strerror(errno));
    }
    fprintf(stderr,
            "ds4: expert bundle build (one-time): %u layers x %u experts, "
            "~%.1f GiB -> %s\n",
            layer_count,
            n_expert,
            (double)total_bytes / (double)DS4_GIB,
            path);

    /* Per-process temp name: two concurrent builders (e.g. distributed
     * workers sharing a model directory) must never interleave writes into
     * one file; the loser of the final rename just wasted its build. */
    char tmp[sizeof(((ds4_expert_bundle *)0)->path)];
    if ((size_t)snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path,
                         (int)getpid()) >= sizeof(tmp)) {
        return false;
    }
    const int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        fprintf(stderr, "ds4: expert bundle write failed (%s): %s\n",
                strerror(errno), tmp);
        return false;
    }

    uint8_t *head = NULL;
    uint8_t *rec = NULL;
    uint64_t *strong_hashes = NULL;
    expert_bundle_model_identity identity_before;
    expert_bundle_model_identity identity_after;
    const uint64_t n_strong =
        (uint64_t)layer_count * DS4_EXPERT_BUNDLE_ID_HASHES_PER_LAYER;
    if (identity_supported) {
        if (!expert_bundle_model_identity_get(model_fd, &identity_before) ||
            identity_before.size != model_size) {
            goto cleanup;
        }
        strong_hashes = malloc((size_t)n_strong * sizeof(*strong_hashes));
        if (!strong_hashes) goto cleanup;
        for (uint64_t i = 0; i < n_strong; i++) {
            strong_hashes[i] = DS4_FNV1A64_OFFSET;
        }
    }
    head = calloc(1, (size_t)data_base);
    if (!head) goto cleanup;
    expert_bundle_put_u32(head, DS4_EXPERT_BUNDLE_MAGIC);
    expert_bundle_put_u32(head + 4, DS4_EXPERT_BUNDLE_VERSION);
    expert_bundle_put_u64(head + 8, model_size);
    expert_bundle_put_u32(head + 16, layer_lo);
    expert_bundle_put_u32(head + 20, layer_count);
    expert_bundle_put_u32(head + 24, n_expert);
    expert_bundle_put_u32(head + 28, 0);
    expert_bundle_put_u64(head + 32, gate_bytes);
    expert_bundle_put_u64(head + 40, up_bytes);
    expert_bundle_put_u64(head + 48, down_bytes);
    for (uint32_t i = 0; i < layer_count; i++) {
        expert_bundle_put_u64(head + 56 + (uint64_t)i * 8,
                              hashes[i].canonical);
    }
    if (!expert_bundle_write_full(fd, head, data_base)) goto cleanup;

    if (posix_memalign((void **)&rec,
                       (size_t)DS4_EXPERT_BUNDLE_ALIGN,
                       (size_t)record) != 0) {
        rec = NULL;
        goto cleanup;
    }
    memset(rec, 0, (size_t)record);   /* record padding stays zero */
    for (uint32_t il = 0; il < layer_count; il++) {
        const ds4_expert_bundle_layer *l = &layers[il];
        for (uint32_t e = 0; e < n_expert; e++) {
            if (!expert_bundle_pread_full(model_fd, rec,
                                          gate_bytes,
                                          l->gate_offset + (uint64_t)e * gate_bytes) ||
                !expert_bundle_pread_full(model_fd, rec + gate_bytes,
                                          up_bytes,
                                          l->up_offset + (uint64_t)e * up_bytes) ||
                !expert_bundle_pread_full(model_fd, rec + gate_bytes + up_bytes,
                                          down_bytes,
                                          l->down_offset + (uint64_t)e * down_bytes)) {
                fprintf(stderr,
                        "ds4: expert bundle source read failed "
                        "(layer %u expert %u)\n",
                        layer_lo + il, e);
                goto cleanup;
            }
            if (strong_hashes) {
                strong_hashes[(uint64_t)il * 3ull] =
                    expert_bundle_fnv1a_update(
                        strong_hashes[(uint64_t)il * 3ull],
                        rec,
                        (size_t)gate_bytes);
                strong_hashes[(uint64_t)il * 3ull + 1ull] =
                    expert_bundle_fnv1a_update(
                        strong_hashes[(uint64_t)il * 3ull + 1ull],
                        rec + gate_bytes,
                        (size_t)up_bytes);
                strong_hashes[(uint64_t)il * 3ull + 2ull] =
                    expert_bundle_fnv1a_update(
                        strong_hashes[(uint64_t)il * 3ull + 2ull],
                        rec + gate_bytes + up_bytes,
                        (size_t)down_bytes);
            }
            if (!expert_bundle_write_full(fd, rec, record)) {
                fprintf(stderr, "ds4: expert bundle write failed: %s\n",
                        strerror(errno));
                goto cleanup;
            }
        }
        if ((il + 1) % 8 == 0 || il + 1 == layer_count) {
            fprintf(stderr, "ds4: expert bundle build: layer %u/%u\n",
                    il + 1, layer_count);
        }
    }
    if (identity_supported) {
        if (!expert_bundle_model_identity_get(model_fd, &identity_after) ||
            !expert_bundle_model_identity_equal(&identity_before,
                                                &identity_after)) {
            fprintf(stderr,
                    "ds4: expert bundle source model changed during build; "
                    "discarding temporary file\n");
            goto cleanup;
        }
        if (!expert_bundle_identity_write_fd(fd,
                                             hb,
                                             data_base,
                                             &identity_after,
                                             layer_count,
                                             strong_hashes,
                                             false)) {
            fprintf(stderr,
                    "ds4: expert bundle identity extension write failed: %s\n",
                    strerror(errno));
            goto cleanup;
        }
    }
    const bool synced = fsync(fd) == 0;
    if (!synced || close(fd) != 0) {
        fprintf(stderr, "ds4: expert bundle flush failed: %s\n",
                strerror(errno));
        if (!synced) close(fd);
        free(head);
        free(rec);
        free(strong_hashes);
        unlink(tmp);
        return false;
    }
    free(head);
    free(rec);
    free(strong_hashes);
    if (rename(tmp, path) != 0) {
        fprintf(stderr, "ds4: expert bundle rename failed (%s): %s\n",
                strerror(errno), path);
        unlink(tmp);
        return false;
    }
    fprintf(stderr, "ds4: expert bundle written (~%.1f GiB): %s\n",
            (double)total_bytes / (double)DS4_GIB, path);
    return true;

cleanup:
    free(head);
    free(rec);
    free(strong_hashes);
    close(fd);
    unlink(tmp);
    return false;
}

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
                                     bool                            use_direct_io) {
    if (!b) return false;
    expert_bundle_disable(b);
    if (!model_path || model_fd < 0 || !layers ||
        layer_count == 0 || n_expert == 0 ||
        gate_bytes == 0 || up_bytes == 0 || down_bytes == 0 ||
        gate_bytes > UINT64_MAX - up_bytes ||
        gate_bytes + up_bytes > UINT64_MAX - down_bytes) {
        return false;
    }

    expert_bundle_layer_fingerprints *hashes =
        malloc((size_t)layer_count * sizeof(*hashes));
    if (!hashes) return false;
    for (uint32_t i = 0; i < layer_count; i++) {
        if (n_expert > UINT64_MAX / gate_bytes ||
            !expert_bundle_layer_hashes(model_fd,
                                        layers[i].gate_offset,
                                        (uint64_t)n_expert * gate_bytes,
                                        &hashes[i])) {
            fprintf(stderr,
                    "ds4: expert bundle model fingerprint read failed "
                    "(layer %u); skipping\n",
                    layer_lo + i);
            free(hashes);
            return false;
        }
    }

    /*
     * Location: READING tries the sibling first (a bundle built next to the
     * GGUF is always reused), then $DS4_BUNDLE_DIR. BUILDING goes to
     * $DS4_BUNDLE_DIR whenever it is set, so a read-only model directory
     * never blocks the sidecar.
     */
    char sibling[sizeof(b->path)];
    char in_dir[sizeof(b->path)];
    const char *candidates[2];
    uint32_t n_candidates = 0;
    const char *build_path = NULL;
    if ((size_t)snprintf(sibling, sizeof(sibling), "%s%s",
                         model_path, DS4_EXPERT_BUNDLE_SUFFIX) <
        sizeof(sibling)) {
        candidates[n_candidates++] = sibling;
        build_path = sibling;
    }
    const char *env_dir = getenv("DS4_BUNDLE_DIR");
    if (env_dir && env_dir[0]) {
        (void)mkdir(env_dir, 0755);
        const char *base = strrchr(model_path, '/');
        base = base ? base + 1 : model_path;
        if ((size_t)snprintf(in_dir, sizeof(in_dir), "%s/%s%s",
                             env_dir, base, DS4_EXPERT_BUNDLE_SUFFIX) <
            sizeof(in_dir)) {
            candidates[n_candidates++] = in_dir;
            build_path = in_dir;
        }
    }
    if (n_candidates == 0 || !build_path) {
        free(hashes);
        return false;
    }

    for (uint32_t i = 0; i < n_candidates; i++) {
        if (expert_bundle_open_existing(b, candidates[i], model_fd, model_size,
                                        layer_lo, layer_count, n_expert,
                                        gate_bytes, up_bytes, down_bytes,
                                        layers, hashes, use_direct_io)) {
            fprintf(stderr, "ds4: expert bundle loaded: %s\n", b->path);
            free(hashes);
            return true;
        }
    }

    if (!expert_bundle_build(build_path, model_fd, model_size,
                             layer_lo, layer_count, n_expert,
                             gate_bytes, up_bytes, down_bytes,
                             layers, hashes) ||
        !expert_bundle_open_existing(b, build_path, model_fd, model_size,
                                     layer_lo, layer_count, n_expert,
                                     gate_bytes, up_bytes, down_bytes,
                                     layers, hashes, use_direct_io)) {
        free(hashes);
        expert_bundle_disable(b);
        return false;
    }
    free(hashes);
    return true;
}

void ds4_expert_bundle_close(ds4_expert_bundle *b) {
    if (!b) return;
    if (b->fd >= 0) close(b->fd);
    expert_bundle_disable(b);
}
