/*
 * Model-free test of ds4_flint_stream.c (docs/RECURSIVE_FLINT.md step 4):
 * the expert cache over a sparse temporary file shaped like the Flint GGUF's
 * pack region.  Each pack carries a signature in its first and last bytes;
 * everything in between reads as zeros, so the file costs 100 MB on disk.
 *
 *   make test-flint-stream         (fork.mk)
 *
 * Covers: cold misses, hits without reads, LRU eviction with the layer's
 * selection pinned, touch, the three prefill policies (cache, staged
 * selection, whole layer with read-ahead), failed reads with the slot
 * returned, and dense-layer requests.  pread is counted through
 * tests/flint/flint_pread_probe.h.
 */
#undef pread
#include "../../ds4_flint_stream.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define N_LAYER DS4_FLINT_STREAM_N_LAYER
#define N_EXPERT DS4_FLINT_STREAM_N_EXPERT
#define PACK DS4_FLINT_STREAM_PACK_BYTES
#define HEADER 65536ull

static uint64_t g_pread_calls, g_pread_bytes;
static uint64_t g_fail_offset = UINT64_MAX;

ssize_t ds4_test_pread(int fd, void *dst, size_t bytes, off_t offset) {
    g_pread_calls++;
    if ((uint64_t)offset <= g_fail_offset && g_fail_offset < (uint64_t)offset + bytes) {
        errno = EIO;
        return -1;
    }
    const ssize_t n = pread(fd, dst, bytes, offset);
    if (n > 0) g_pread_bytes += (uint64_t)n;
    return n;
}

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static uint64_t pack_offsets[N_LAYER * N_EXPERT];

static uint64_t pack_offset(uint32_t layer, uint32_t e) {
    return HEADER + ((uint64_t)(layer - 2) * N_EXPERT + e) * PACK;
}

static void signature(uint32_t layer, uint32_t e, uint8_t out[16]) {
    const uint32_t w[4] = { 0xf11a7u, layer, e, layer * 1000003u + e };
    memcpy(out, w, 16);
}

static int slot_matches(const uint8_t *p, uint32_t layer, uint32_t e) {
    uint8_t sig[16];
    signature(layer, e, sig);
    return memcmp(p, sig, 16) == 0 && memcmp(p + PACK - 16, sig, 16) == 0;
}

/* region allocator: host memory, the GPU address is the pointer */
typedef struct { void *ptrs[64]; int n; } regions;
static int alloc_region(void *ud, uint64_t bytes, const char *label, ds4_flint_stream_region *out) {
    (void)label;
    regions *r = ud;
    void *p = NULL;
    if (posix_memalign(&p, 16384, (size_t)bytes) != 0) return 0;
    memset(p, 0xee, (size_t)bytes);
    r->ptrs[r->n++] = p;
    out->ptr = p;
    out->gpu = (uint64_t)(uintptr_t)p;
    return 1;
}

int main(void) {
    char path[] = "/tmp/flint-stream-test-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); return 1; }
    const uint64_t file_size = pack_offset(N_LAYER - 1, N_EXPERT - 1) + PACK;
    if (ftruncate(fd, (off_t)file_size) != 0) { perror("ftruncate"); return 1; }
    for (uint32_t layer = 2; layer < N_LAYER; layer++) {
        for (uint32_t e = 0; e < N_EXPERT; e++) {
            uint8_t sig[16];
            signature(layer, e, sig);
            const uint64_t off = pack_offset(layer, e);
            pack_offsets[layer * N_EXPERT + e] = off;
            if (pwrite(fd, sig, 16, (off_t)off) != 16 || pwrite(fd, sig, 16, (off_t)(off + PACK - 16)) != 16) {
                perror("pwrite");
                return 1;
            }
        }
    }
    close(fd);

    regions regs = {0};
    uint64_t *gpu_table = calloc(N_LAYER * N_EXPERT, sizeof(uint64_t));
    ds4_flint_stream_config cfg = {0};
    cfg.path = path;
    cfg.pack_offsets = pack_offsets;
    cfg.n_slots = 64;
    cfg.gpu_table = gpu_table;
    cfg.alloc = alloc_region;
    cfg.alloc_ud = &regs;
    cfg.threads = 4;
    cfg.cached_io = 1;
    char err[256] = {0};
    ds4_flint_stream *st = ds4_flint_stream_open(&cfg, err, sizeof(err));
    if (!st) { fprintf(stderr, "open failed: %s\n", err); return 1; }
    CHECK(ds4_flint_stream_slots(st) == 64, "slots");

    /* 1. cold misses */
    const uint32_t ids[6] = { 3, 17, 42, 100, 200, 255 };
    g_pread_calls = g_pread_bytes = 0;
    CHECK(ds4_flint_stream_fetch(st, 2, ids, 6, err, sizeof(err)), "fetch: %s", err);
    CHECK(g_pread_bytes == 6 * PACK, "cold fetch read %llu bytes", (unsigned long long)g_pread_bytes);
    for (int i = 0; i < 6; i++) {
        const uint64_t a = gpu_table[2 * N_EXPERT + ids[i]];
        CHECK(a != 0, "table entry for expert %u", ids[i]);
        CHECK(a && slot_matches((const uint8_t *)(uintptr_t)a, 2, ids[i]), "slot bytes for expert %u", ids[i]);
    }
    /* 2. hits read nothing and keep the addresses */
    uint64_t before[6];
    for (int i = 0; i < 6; i++) before[i] = gpu_table[2 * N_EXPERT + ids[i]];
    g_pread_calls = g_pread_bytes = 0;
    CHECK(ds4_flint_stream_fetch(st, 2, ids, 6, err, sizeof(err)), "refetch: %s", err);
    CHECK(g_pread_calls == 0, "hits issued %llu preads", (unsigned long long)g_pread_calls);
    for (int i = 0; i < 6; i++) CHECK(gpu_table[2 * N_EXPERT + ids[i]] == before[i], "hit moved expert %u", ids[i]);

    /* 3. fill the cache with layer 5's experts 0..57 (6 + 58 = 64 slots) */
    for (uint32_t e = 0; e < 58; e += 6) {
        uint32_t six[6];
        uint32_t n = 0;
        for (uint32_t k = 0; k < 6 && e + k < 58; k++) six[n++] = e + k;
        CHECK(ds4_flint_stream_fetch(st, 5, six, n, err, sizeof(err)), "fill: %s", err);
    }
    ds4_flint_stream_stats stats;
    ds4_flint_stream_get_stats(st, &stats);
    CHECK(stats.used == 64, "used %u after filling", stats.used);
    CHECK(stats.evictions == 0, "evictions %llu before the cache is full", (unsigned long long)stats.evictions);
    /* layer 2's six are the least recently used: a new selection evicts them first */
    const uint32_t fresh[6] = { 60, 61, 62, 63, 64, 65 };
    CHECK(ds4_flint_stream_fetch(st, 5, fresh, 6, err, sizeof(err)), "evicting fetch: %s", err);
    ds4_flint_stream_get_stats(st, &stats);
    CHECK(stats.evictions == 6, "evictions %llu", (unsigned long long)stats.evictions);
    for (int i = 0; i < 6; i++) CHECK(gpu_table[2 * N_EXPERT + ids[i]] == 0, "layer 2 expert %u survived eviction", ids[i]);
    for (int i = 0; i < 6; i++) CHECK(gpu_table[5 * N_EXPERT + fresh[i]] != 0, "fresh expert %u absent", fresh[i]);
    /* pinned: hits 0..4 of the same request must not be the victims; 5 is the LRU head */
    const uint32_t pinned[6] = { 0, 1, 2, 3, 4, 210 };
    CHECK(ds4_flint_stream_fetch(st, 5, pinned, 6, err, sizeof(err)), "pinned fetch: %s", err);
    for (int i = 0; i < 5; i++) CHECK(gpu_table[5 * N_EXPERT + i] != 0, "pinned expert %d evicted", i);
    CHECK(gpu_table[5 * N_EXPERT + 5] == 0, "expert 5 should have been the LRU victim");
    CHECK(gpu_table[5 * N_EXPERT + 210] != 0, "expert 210 absent");
    /* touch makes 6..11 most recent; the next victim is then 12 */
    const uint32_t touched[6] = { 6, 7, 8, 9, 10, 11 };
    ds4_flint_stream_touch(st, 5, touched, 6);
    const uint32_t one[1] = { 211 };
    CHECK(ds4_flint_stream_fetch(st, 5, one, 1, err, sizeof(err)), "touch fetch: %s", err);
    CHECK(gpu_table[5 * N_EXPERT + 12] == 0 && gpu_table[5 * N_EXPERT + 6] != 0, "touch did not protect 6..11");

    /* 4. prefill, cache policy: 10 experts for 8 tokens */
    uint8_t need[N_EXPERT];
    uint64_t table[N_EXPERT];
    memset(need, 0, sizeof(need));
    for (uint32_t e = 100; e < 110; e++) need[e] = 1;
    g_pread_calls = g_pread_bytes = 0;
    CHECK(ds4_flint_stream_prefill_layer(st, 7, need, 8, table, err, sizeof(err)), "prefill cache: %s", err);
    CHECK(g_pread_bytes == 10 * PACK, "prefill cache read %llu", (unsigned long long)g_pread_bytes);
    for (uint32_t e = 0; e < N_EXPERT; e++) {
        if (need[e]) {
            CHECK(table[e] != 0 && table[e] == gpu_table[7 * N_EXPERT + e], "cache-path table entry %u", e);
            CHECK(table[e] && slot_matches((const uint8_t *)(uintptr_t)table[e], 7, e), "cache-path bytes %u", e);
        } else {
            CHECK(table[e] == 0, "unneeded expert %u has an address", e);
        }
    }
    ds4_flint_stream_get_stats(st, &stats);
    CHECK(stats.prefill_layers_cache == 1 && stats.prefill_layers_staged == 0, "prefill policy counters");

    /* 5. prefill, staged selection: 40 experts for 8 tokens (more than half the cache) */
    memset(need, 0, sizeof(need));
    for (uint32_t e = 0; e < 40; e++) need[e * 6 % N_EXPERT] = 1;
    uint32_t needed = 0;
    for (uint32_t e = 0; e < N_EXPERT; e++) needed += need[e];
    g_pread_calls = g_pread_bytes = 0;
    CHECK(ds4_flint_stream_prefill_layer(st, 9, need, 8, table, err, sizeof(err)), "prefill staged: %s", err);
    CHECK(g_pread_bytes == (uint64_t)needed * PACK, "staged selection read %llu for %u experts", (unsigned long long)g_pread_bytes, needed);
    for (uint32_t e = 0; e < N_EXPERT; e++) {
        if (need[e]) {
            CHECK(table[e] != 0 && gpu_table[9 * N_EXPERT + e] == 0, "staged entry %u must not enter the cache", e);
            CHECK(table[e] && slot_matches((const uint8_t *)(uintptr_t)table[e], 9, e), "staged bytes %u", e);
        }
    }
    ds4_flint_stream_get_stats(st, &stats);
    CHECK(stats.used <= 64, "cache overfilled");

    /* 6. prefill, whole layer with read-ahead: 2048 tokens */
    memset(need, 1, sizeof(need));
    g_pread_calls = g_pread_bytes = 0;
    CHECK(ds4_flint_stream_prefill_layer(st, 20, need, 2048, table, err, sizeof(err)), "prefill full: %s", err);
    for (uint32_t e = 0; e < N_EXPERT; e += 37) CHECK(table[e] && slot_matches((const uint8_t *)(uintptr_t)table[e], 20, e), "full-layer bytes %u", e);
    /* the next layer is read in the background; the call for it waits instead of reading */
    CHECK(ds4_flint_stream_prefill_layer(st, 21, need, 2048, table, err, sizeof(err)), "prefill next: %s", err);
    CHECK(g_pread_bytes == 2ull * N_EXPERT * PACK + 0, "two whole layers read %llu (expected %llu); read-ahead missing?",
          (unsigned long long)g_pread_bytes, (unsigned long long)(2ull * N_EXPERT * PACK));
    for (uint32_t e = 1; e < N_EXPERT; e += 41) CHECK(table[e] && slot_matches((const uint8_t *)(uintptr_t)table[e], 21, e), "read-ahead bytes %u", e);
    /* the two layers use different staging buffers */
    uint64_t t20[N_EXPERT];
    CHECK(ds4_flint_stream_prefill_layer(st, 22, need, 2048, t20, err, sizeof(err)), "prefill 22: %s", err);
    CHECK(t20[0] != table[0], "consecutive layers share a staging buffer");
    ds4_flint_stream_prefill_end(st);
    ds4_flint_stream_get_stats(st, &stats);
    CHECK(stats.prefill_layers_staged == 4, "staged layers %llu", (unsigned long long)stats.prefill_layers_staged);

    /* 7. a failed read: the slot returns to the pool and the table stays clean */
    ds4_flint_stream_get_stats(st, &stats);
    const uint32_t used_before = stats.used;
    const uint32_t bad[2] = { 77, 78 };
    g_fail_offset = pack_offset(30, 78) + 4096;
    CHECK(!ds4_flint_stream_fetch(st, 30, bad, 2, err, sizeof(err)), "failed read not reported");
    CHECK(gpu_table[30 * N_EXPERT + 78] == 0, "failed expert installed");
    g_fail_offset = UINT64_MAX;
    CHECK(ds4_flint_stream_fetch(st, 30, bad, 2, err, sizeof(err)), "retry: %s", err);
    CHECK(gpu_table[30 * N_EXPERT + 77] && gpu_table[30 * N_EXPERT + 78], "retry did not install");
    CHECK(slot_matches((const uint8_t *)(uintptr_t)gpu_table[30 * N_EXPERT + 78], 30, 78), "retry bytes");
    ds4_flint_stream_get_stats(st, &stats);
    CHECK(stats.used <= 64 && stats.used >= used_before, "used %u after failure and retry", stats.used);

    /* 8. dense layers have no packs */
    CHECK(!ds4_flint_stream_fetch(st, 0, ids, 6, err, sizeof(err)), "dense layer fetch accepted");
    CHECK(!ds4_flint_stream_prefill_layer(st, 1, need, 2048, table, err, sizeof(err)), "dense layer prefill accepted");

    /* 9. a cache that cannot hold a selection fails cleanly instead of looping */
    ds4_flint_stream_close(st);
    cfg.n_slots = 63;
    st = ds4_flint_stream_open(&cfg, err, sizeof(err));
    CHECK(st == NULL, "63 slots accepted");

    for (int i = 0; i < regs.n; i++) free(regs.ptrs[i]);
    free(gpu_table);
    unlink(path);
    if (failures) {
        fprintf(stderr, "test_flint_stream: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_flint_stream: all checks passed\n");
    return 0;
}
