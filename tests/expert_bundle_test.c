/*
 * Unit test for the SSD streaming expert-bundle sidecar (ds4_ssd.c).
 * Build:  cc -O2 -Wall -Wextra -std=c99 -I. -o tests/expert_bundle_test tests/expert_bundle_test.c ds4_ssd.c -lm -pthread
 * Run:    ./tests/expert_bundle_test
 *
 * Builds a tiny synthetic "model" file with two layers of routed expert
 * tensors, then exercises the whole sidecar lifecycle: one-time build,
 * header layout ("DSEB" v1, byte-compatible with the DwarfStar Swift port),
 * record content and zero padding, reuse without rebuild, fingerprint
 * invalidation when the model bytes change, and $DS4_BUNDLE_DIR placement.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ds4_ssd.h"

#define LAYER_LO 3u
#define LAYER_COUNT 2u
#define N_EXPERT 4u
#define GATE_BYTES 6000ull
#define UP_BYTES 6000ull
#define DOWN_BYTES 7000ull
#define ALIGN 4096ull
#define RECORD 20480ull   /* align4096(6000+6000+7000) */
#define FNV1A64_OFFSET 0xcbf29ce484222325ull
#define FNV1A64_PRIME 0x100000001b3ull
#define SWIFT_V1_PRIME 0x10000000001b3ull
#define DSEI_MAGIC 0x49455344u
#define GIB (1024ull * 1024ull * 1024ull)

static int g_failures;

#define CHECK(cond, msg)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            fprintf(stderr, "FAIL %s:%d: %s\n",                   \
                    __FILE__, __LINE__, msg);                     \
            g_failures++;                                         \
        }                                                         \
    } while (0)

static uint8_t slab_byte(uint32_t layer, uint32_t kind, uint32_t expert,
                         uint64_t i) {
    return (uint8_t)(layer * 31u + kind * 17u + expert * 7u + (uint32_t)i);
}

typedef struct {
    ds4_expert_bundle_layer layers[LAYER_COUNT];
    uint64_t model_size;
} model_layout;

/* Slab kinds, in record order. */
enum { KIND_GATE = 0, KIND_UP = 1, KIND_DOWN = 2 };

static uint64_t kind_bytes(uint32_t kind) {
    return kind == KIND_DOWN ? DOWN_BYTES :
           kind == KIND_UP ? UP_BYTES : GATE_BYTES;
}

static uint64_t tensor_offset(const model_layout *m, uint32_t layer,
                              uint32_t kind) {
    const ds4_expert_bundle_layer *l = &m->layers[layer];
    return kind == KIND_DOWN ? l->down_offset :
           kind == KIND_UP ? l->up_offset : l->gate_offset;
}

static void write_model(const char *path, model_layout *m) {
    /* Tensors are laid out with deliberate gaps and unaligned offsets so a
     * remap bug cannot hide behind a convenient layout. */
    uint64_t pos = 1000;
    for (uint32_t layer = 0; layer < LAYER_COUNT; layer++) {
        m->layers[layer].gate_offset = pos;
        pos += N_EXPERT * GATE_BYTES + 512;
        m->layers[layer].up_offset = pos;
        pos += N_EXPERT * UP_BYTES + 768;
        m->layers[layer].down_offset = pos;
        pos += N_EXPERT * DOWN_BYTES + 1024;
    }
    m->model_size = pos;

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "cannot create %s: %s\n", path, strerror(errno));
        exit(1);
    }
    uint8_t *buf = calloc(1, (size_t)m->model_size);
    for (uint32_t layer = 0; layer < LAYER_COUNT; layer++) {
        for (uint32_t kind = 0; kind < 3; kind++) {
            const uint64_t bytes = kind_bytes(kind);
            uint8_t *t = buf + tensor_offset(m, layer, kind);
            for (uint32_t e = 0; e < N_EXPERT; e++) {
                for (uint64_t i = 0; i < bytes; i++) {
                    t[(uint64_t)e * bytes + i] = slab_byte(layer, kind, e, i);
                }
            }
        }
    }
    if (fwrite(buf, 1, (size_t)m->model_size, fp) != m->model_size ||
        fclose(fp) != 0) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(1);
    }
    free(buf);
}

static int pread_full(int fd, void *dst, uint64_t bytes, uint64_t offset) {
    uint8_t *p = dst;
    uint64_t pos = 0;
    while (pos < bytes) {
        ssize_t n = pread(fd, p + pos, (size_t)(bytes - pos),
                          (off_t)(offset + pos));
        if (n <= 0) return 0;
        pos += (uint64_t)n;
    }
    return 1;
}

static uint64_t identity_offset(void) {
    const uint64_t hb = 56ull + 8ull * LAYER_COUNT;
    return (hb + 7ull) & ~7ull;
}

static int clear_identity_extension(const char *bundle_path) {
    const uint64_t off = identity_offset();
    uint8_t zeros[ALIGN];
    memset(zeros, 0, sizeof(zeros));
    const int fd = open(bundle_path, O_WRONLY);
    if (fd < 0) return 0;
    const ssize_t n = pwrite(fd, zeros, (size_t)(ALIGN - off), (off_t)off);
    return close(fd) == 0 && n == (ssize_t)(ALIGN - off);
}

static int has_identity_extension(const char *bundle_path) {
    uint8_t magic[4];
    const int fd = open(bundle_path, O_RDONLY);
    if (fd < 0) return 0;
    const int ok = pread_full(fd, magic, sizeof(magic), identity_offset());
    close(fd);
    return ok && magic[0] == (uint8_t)DSEI_MAGIC &&
           magic[1] == (uint8_t)(DSEI_MAGIC >> 8) &&
           magic[2] == (uint8_t)(DSEI_MAGIC >> 16) &&
           magic[3] == (uint8_t)(DSEI_MAGIC >> 24);
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t get_u64(const uint8_t *p) {
    return (uint64_t)get_u32(p) | ((uint64_t)get_u32(p + 4) << 32);
}

static void put_u64(uint8_t *p, uint64_t v) {
    for (uint32_t i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (i * 8));
}

static uint64_t fnv1a64(const uint8_t *p, uint64_t n, uint64_t prime) {
    uint64_t h = FNV1A64_OFFSET;
    for (uint64_t i = 0; i < n; i++) h = (h ^ p[i]) * prime;
    return h;
}

enum bundle_hash_scheme {
    HASH_CANONICAL,
    HASH_SWIFT_V1,
    HASH_MIXED,
};

/* Rewrite only the tiny fingerprint table, leaving the records and padding
 * untouched. This models bundles produced by the canonical C writer and by
 * the original Swift v1 writer without building two large fixtures. */
static int rewrite_bundle_hashes(const char             *bundle_path,
                                 int                     model_fd,
                                 const model_layout     *m,
                                 enum bundle_hash_scheme scheme) {
    uint8_t source[4096];
    uint8_t encoded[8];
    const int fd = open(bundle_path, O_WRONLY);
    if (fd < 0) return 0;
    for (uint32_t layer = 0; layer < LAYER_COUNT; layer++) {
        if (!pread_full(model_fd, source, sizeof(source),
                        tensor_offset(m, layer, KIND_GATE))) {
            close(fd);
            return 0;
        }
        const bool legacy = scheme == HASH_SWIFT_V1 ||
                            (scheme == HASH_MIXED && layer != 0);
        put_u64(encoded, fnv1a64(source, sizeof(source),
                                 legacy ? SWIFT_V1_PRIME : FNV1A64_PRIME));
        if (pwrite(fd, encoded, sizeof(encoded),
                   (off_t)(56 + (uint64_t)layer * 8)) !=
            (ssize_t)sizeof(encoded)) {
            close(fd);
            return 0;
        }
    }
    return close(fd) == 0;
}

static int open_bundle_with_io(ds4_expert_bundle *b, const char *model_path,
                               int model_fd, const model_layout *m,
                               bool use_direct_io) {
    return ds4_expert_bundle_open_or_build(b, model_path, model_fd,
                                           m->model_size,
                                           LAYER_LO, LAYER_COUNT, N_EXPERT,
                                           GATE_BYTES, UP_BYTES, DOWN_BYTES,
                                           m->layers, use_direct_io);
}

static int open_bundle(ds4_expert_bundle *b, const char *model_path,
                       int model_fd, const model_layout *m) {
    return open_bundle_with_io(b, model_path, model_fd, m, true);
}

static void check_records(const ds4_expert_bundle *b, const uint8_t *expect_or_null,
                          uint32_t tam_layer, uint32_t tam_expert, uint64_t tam_i) {
    uint8_t *rec = malloc((size_t)RECORD);
    for (uint32_t layer = 0; layer < LAYER_COUNT; layer++) {
        for (uint32_t e = 0; e < N_EXPERT; e++) {
            const uint64_t off =
                ds4_expert_bundle_record_offset(b, LAYER_LO + layer, e);
            CHECK(pread_full(b->fd, rec, RECORD, off), "record read");
            uint64_t base = 0;
            for (uint32_t kind = 0; kind < 3; kind++) {
                const uint64_t bytes = kind_bytes(kind);
                for (uint64_t i = 0; i < bytes; i++) {
                    uint8_t want = slab_byte(layer, kind, e, i);
                    if (expect_or_null && kind == KIND_GATE &&
                        layer == tam_layer && e == tam_expert && i == tam_i) {
                        want = *expect_or_null;
                    }
                    if (rec[base + i] != want) {
                        CHECK(0, "record slab bytes mismatch");
                        goto next_record;
                    }
                }
                base += bytes;
            }
            for (uint64_t i = base; i < RECORD; i++) {
                if (rec[i] != 0) {
                    CHECK(0, "record padding is not zero");
                    break;
                }
            }
next_record:;
        }
    }
    free(rec);
}

static void check_record_byte(const ds4_expert_bundle *b,
                              uint32_t layer, uint32_t kind,
                              uint32_t expert, uint64_t index,
                              uint8_t want) {
    const uint64_t kind_off = kind == KIND_DOWN ? GATE_BYTES + UP_BYTES :
                              kind == KIND_UP ? GATE_BYTES : 0;
    const uint64_t off = ds4_expert_bundle_record_offset(
                             b, LAYER_LO + layer, expert) +
                         kind_off + index;
    uint8_t got = 0;
    CHECK(pread_full(b->fd, &got, 1, off) && got == want,
          "rebuilt record contains same-size model mutation");
}

static void check_io_policy(void) {
    static const struct {
        uint64_t host_gib;
        uint64_t reserve_gib;
        uint64_t budget_gib;
    } memory_tiers[] = {
        {  32,  8,  24 },
        {  48,  8,  40 },
        {  64,  8,  56 },
        {  96, 12,  84 },
        { 128, 16, 112 },
        { 256, 32, 224 },
        { 512, 64, 448 },
    };

    ds4_ssd_io_plan p =
        ds4_ssd_io_plan_compute(16ull * GIB, 0, 8ull * GIB);
    CHECK(p.inputs_valid, "16 GiB page-cache plan is valid");
    CHECK(p.reserve_bytes == 8ull * GIB, "16 GiB reserves 8 GiB");
    CHECK(p.budget_bytes == 8ull * GIB, "16 GiB host budget is 8 GiB");
    CHECK(p.hot_bytes == 8ull * GIB, "16 GiB plan records hot bytes");
    CHECK(!p.use_direct_io, "hot set equal to budget uses page cache");

    p = ds4_ssd_io_plan_compute(16ull * GIB, 0, 8ull * GIB + 1ull);
    CHECK(p.inputs_valid && p.use_direct_io,
          "hot set one byte above 16 GiB budget uses direct I/O");

    /*
     * Keep every currently shipping/high-memory Apple Silicon tier covered.
     * These checks intentionally omit the Metal recommendation so they
     * isolate the physical-RAM reserve; the recommended-limit cap is tested
     * separately below.
     */
    for (size_t i = 0;
         i < sizeof(memory_tiers) / sizeof(memory_tiers[0]);
         i++) {
        const uint64_t host = memory_tiers[i].host_gib * GIB;
        const uint64_t reserve = memory_tiers[i].reserve_gib * GIB;
        const uint64_t budget = memory_tiers[i].budget_gib * GIB;

        p = ds4_ssd_io_plan_compute(host, 0, budget);
        CHECK(p.inputs_valid,
              "memory-tier page-cache plan is valid");
        CHECK(p.reserve_bytes == reserve,
              "memory-tier host reserve scales correctly");
        CHECK(p.budget_bytes == budget,
              "memory-tier host budget scales correctly");
        CHECK(!p.use_direct_io,
              "memory-tier hot set equal to budget uses page cache");

        p = ds4_ssd_io_plan_compute(host, 0, budget + 1ull);
        CHECK(p.inputs_valid && p.use_direct_io,
              "memory-tier hot set above budget uses direct I/O");
    }

    /*
     * Representative recurring sets for the 80.76 GiB DeepSeek report.
     * The 344-expert case is about 10.56 GiB and remains cacheable on 32/48
     * GiB hosts. The 32 GiB expert-budget case is about 41.06 GiB: it does not
     * fit the safe 32/48 GiB host budgets, while 64 GiB and larger hosts
     * retain mmap/page-cache reuse. The real boundary may be lower when Metal
     * reports a tighter recommended working-set limit.
     */
    p = ds4_ssd_io_plan_compute(32ull * GIB, 0, 11ull * GIB);
    CHECK(p.inputs_valid && !p.use_direct_io,
          "32 GiB compact DeepSeek hot set uses page cache");
    p = ds4_ssd_io_plan_compute(48ull * GIB, 0, 11ull * GIB);
    CHECK(p.inputs_valid && !p.use_direct_io,
          "48 GiB compact DeepSeek hot set uses page cache");
    p = ds4_ssd_io_plan_compute(32ull * GIB, 0, 42ull * GIB);
    CHECK(p.inputs_valid && p.use_direct_io,
          "32 GiB DeepSeek-sized hot set uses direct I/O");
    p = ds4_ssd_io_plan_compute(48ull * GIB, 0, 42ull * GIB);
    CHECK(p.inputs_valid && p.use_direct_io,
          "48 GiB DeepSeek-sized hot set uses direct I/O");
    static const uint64_t page_cache_tiers[] = {
        64, 96, 128, 256, 512,
    };
    for (size_t i = 0;
         i < sizeof(page_cache_tiers) / sizeof(page_cache_tiers[0]);
         i++) {
        p = ds4_ssd_io_plan_compute(page_cache_tiers[i] * GIB,
                                    0,
                                    42ull * GIB);
        CHECK(p.inputs_valid && !p.use_direct_io,
              "64+ GiB DeepSeek-sized hot set uses page cache");
    }

    p = ds4_ssd_io_plan_compute(64ull * GIB, 60ull * GIB, 54ull * GIB);
    CHECK(p.inputs_valid, "64 GiB recommended-limit plan is valid");
    CHECK(p.reserve_bytes == 8ull * GIB, "64 GiB reserves one eighth");
    CHECK(p.budget_bytes == 54ull * GIB,
          "64 GiB budget is capped at 90 percent recommended");
    CHECK(!p.use_direct_io,
          "64 GiB hot set equal to recommended budget uses page cache");

    p = ds4_ssd_io_plan_compute(64ull * GIB, 60ull * GIB,
                                54ull * GIB + 1ull);
    CHECK(p.inputs_valid && p.use_direct_io,
          "hot set above 64 GiB recommended budget uses direct I/O");

    p = ds4_ssd_io_plan_compute(64ull * GIB, 0, 56ull * GIB);
    CHECK(p.inputs_valid && p.budget_bytes == 56ull * GIB &&
              !p.use_direct_io,
          "missing recommended limit falls back to host-minus-reserve");

    p = ds4_ssd_io_plan_compute(0, 0, 1ull * GIB);
    CHECK(!p.inputs_valid && p.use_direct_io,
          "unknown host RAM conservatively uses direct I/O");

    p = ds4_ssd_io_plan_compute(16ull * GIB, 0, 0);
    CHECK(!p.inputs_valid && p.use_direct_io,
          "unknown hot working set conservatively uses direct I/O");

    p = ds4_ssd_io_plan_compute(64ull * GIB, UINT64_MAX, 1ull * GIB);
    CHECK(!p.inputs_valid && p.use_direct_io,
          "overflow sentinel conservatively uses direct I/O");

    p = ds4_ssd_io_plan_compute(UINT64_MAX, 0, 1ull * GIB);
    CHECK(!p.inputs_valid && p.use_direct_io,
          "overflowed host RAM conservatively uses direct I/O");
}

int main(void) {
    check_io_policy();

    static const uint8_t hello[] = "hello";
    CHECK(fnv1a64(hello, sizeof(hello) - 1, FNV1A64_PRIME) ==
              0xa430d84680aabd0bull,
          "canonical FNV-1a known vector");
    CHECK(fnv1a64(hello, sizeof(hello) - 1, SWIFT_V1_PRIME) ==
              0xb476bc4680aabd0bull,
          "legacy Swift v1 known vector");

    char dir[] = "/tmp/ds4_expert_bundle_test.XXXXXX";
    if (!mkdtemp(dir)) {
        fprintf(stderr, "mkdtemp failed: %s\n", strerror(errno));
        return 1;
    }
    char model_path[128], bundle_path[192], bundle_dir[192];
    snprintf(model_path, sizeof(model_path), "%s/model.gguf", dir);
    snprintf(bundle_path, sizeof(bundle_path), "%s.expbundle", model_path);
    snprintf(bundle_dir, sizeof(bundle_dir), "%s/bundles", dir);

    model_layout m;
    write_model(model_path, &m);
    const int model_fd = open(model_path, O_RDWR);
    CHECK(model_fd >= 0, "model open");

    /* Build next to the model, then verify the DSEB v1 header layout. */
    ds4_expert_bundle b;
    CHECK(open_bundle(&b, model_path, model_fd, &m), "first build");
    CHECK(strcmp(b.path, bundle_path) == 0, "sibling path");
    CHECK(b.fd >= 0, "bundle fd");
    CHECK(b.data_base == ALIGN && b.record == RECORD, "geometry");
    uint8_t head[56 + 8 * LAYER_COUNT];
    CHECK(pread_full(b.fd, head, sizeof(head), 0), "header read");
    CHECK(get_u32(head) == 0x42455344u, "magic DSEB");
    CHECK(get_u32(head + 4) == 1u, "version");
    CHECK(get_u64(head + 8) == m.model_size, "header model size");
    CHECK(get_u32(head + 16) == LAYER_LO, "header layer lo");
    CHECK(get_u32(head + 20) == LAYER_COUNT, "header layer count");
    CHECK(get_u32(head + 24) == N_EXPERT, "header expert count");
    CHECK(get_u64(head + 32) == GATE_BYTES, "header gate bytes");
    CHECK(get_u64(head + 40) == UP_BYTES, "header up bytes");
    CHECK(get_u64(head + 48) == DOWN_BYTES, "header down bytes");
    uint8_t source[4096];
    for (uint32_t layer = 0; layer < LAYER_COUNT; layer++) {
        CHECK(pread_full(model_fd, source, sizeof(source),
                         tensor_offset(&m, layer, KIND_GATE)),
              "read source fingerprint bytes");
        CHECK(get_u64(head + 56 + (uint64_t)layer * 8) ==
                  fnv1a64(source, sizeof(source), FNV1A64_PRIME),
              "writer emits canonical FNV-1a fingerprints");
    }
    CHECK(ds4_expert_bundle_record_offset(&b, LAYER_LO + 1, 2) ==
              ALIGN + (1 * N_EXPERT + 2) * RECORD,
          "record offset math");
    check_records(&b, NULL, 0, 0, 0);
    ds4_expert_bundle_close(&b);
    CHECK(b.fd == -1, "close resets fd");
    CHECK(has_identity_extension(bundle_path), "new bundle carries DSEI");
    CHECK(open_bundle_with_io(&b, model_path, model_fd, &m, false),
          "page-cache bundle reopen");
    check_records(&b, NULL, 0, 0, 0);
    ds4_expert_bundle_close(&b);

    /* Mark the padding of the first record: a REUSED bundle keeps the mark,
     * a rebuilt one zeroes it. */
    const uint64_t mark_off =
        ALIGN + GATE_BYTES + UP_BYTES + DOWN_BYTES;   /* record 0 padding */
    int wfd = open(bundle_path, O_WRONLY);
    uint8_t mark = 0xAB;
    CHECK(wfd >= 0 && pwrite(wfd, &mark, 1, (off_t)mark_off) == 1, "mark");
    close(wfd);
    CHECK(clear_identity_extension(bundle_path), "strip DSEI for legacy upgrade");
    CHECK(open_bundle(&b, model_path, model_fd, &m), "upgrade legacy C v1");
    uint8_t got = 0;
    CHECK(pread_full(b.fd, &got, 1, mark_off) && got == 0xAB,
          "legacy C v1 reused without rebuild");
    ds4_expert_bundle_close(&b);
    CHECK(has_identity_extension(bundle_path), "legacy C v1 upgraded with DSEI");

    /* A bundle written by the original Swift v1 code must be reused as-is:
     * the padding marker proves open_or_build did not replace the file. */
    CHECK(rewrite_bundle_hashes(bundle_path, model_fd, &m, HASH_SWIFT_V1),
          "write legacy Swift v1 fingerprints");
    CHECK(clear_identity_extension(bundle_path), "strip DSEI from Swift v1 fixture");
    CHECK(open_bundle(&b, model_path, model_fd, &m),
          "open legacy Swift v1 bundle");
    CHECK(pread_full(b.fd, &got, 1, mark_off) && got == 0xAB,
          "legacy Swift v1 bundle reused without rebuild");
    ds4_expert_bundle_close(&b);
    CHECK(has_identity_extension(bundle_path), "Swift v1 upgraded with DSEI");

    /* Compatibility is selected for the complete table, not independently
     * per layer. A hybrid header is corrupt and must be rebuilt. */
    CHECK(rewrite_bundle_hashes(bundle_path, model_fd, &m, HASH_MIXED),
          "write mixed fingerprint schemes");
    CHECK(open_bundle(&b, model_path, model_fd, &m),
          "rebuild mixed fingerprint bundle");
    CHECK(pread_full(b.fd, &got, 1, mark_off) && got == 0,
          "mixed fingerprint schemes trigger a rebuild");
    ds4_expert_bundle_close(&b);

    /* Restore the rebuild marker for the independent source-tamper test. */
    wfd = open(bundle_path, O_WRONLY);
    CHECK(wfd >= 0 && pwrite(wfd, &mark, 1, (off_t)mark_off) == 1,
          "mark before source tamper");
    if (wfd >= 0) close(wfd);

    /* Tamper with the model inside a fingerprinted region (first 4 KiB of
     * layer 1's gate tensor): the bundle must be rebuilt with the new byte. */
    const uint64_t tam_model_off = tensor_offset(&m, 1, KIND_GATE) + 123;
    uint8_t tampered = (uint8_t)(slab_byte(1, KIND_GATE, 0, 123) ^ 0xFF);
    CHECK(pwrite(model_fd, &tampered, 1, (off_t)tam_model_off) == 1, "tamper");
    CHECK(open_bundle(&b, model_path, model_fd, &m), "rebuild after tamper");
    CHECK(pread_full(b.fd, &got, 1, mark_off) && got == 0,
          "tampered model triggers a rebuild");
    check_records(&b, &tampered, 1, 0, 123);
    ds4_expert_bundle_close(&b);

    /* DSEI must catch same-size edits that the Swift v1 prefix hash cannot:
     * a late gate byte plus independent up/down bytes. */
    static const struct {
        uint32_t layer, kind, expert;
        uint64_t index;
        uint8_t mask;
    } late_tampers[] = {
        { 0, KIND_GATE, 0, 5000, 0x11 },
        { 0, KIND_UP,   2, 5000, 0x22 },
        { 1, KIND_DOWN, 3, 6000, 0x44 },
    };
    for (size_t i = 0; i < sizeof(late_tampers) / sizeof(late_tampers[0]); i++) {
        const uint32_t layer = late_tampers[i].layer;
        const uint32_t kind = late_tampers[i].kind;
        const uint32_t expert = late_tampers[i].expert;
        const uint64_t index = late_tampers[i].index;
        const uint64_t model_off = tensor_offset(&m, layer, kind) +
                                   (uint64_t)expert * kind_bytes(kind) + index;
        uint8_t changed = 0;
        CHECK(pread_full(model_fd, &changed, 1, model_off),
              "read byte before late same-size tamper");
        changed ^= late_tampers[i].mask;
        CHECK(pwrite(model_fd, &changed, 1, (off_t)model_off) == 1,
              "write late same-size tamper");
        wfd = open(bundle_path, O_WRONLY);
        CHECK(wfd >= 0 && pwrite(wfd, &mark, 1, (off_t)mark_off) == 1,
              "mark before late same-size tamper validation");
        if (wfd >= 0) close(wfd);
        CHECK(open_bundle(&b, model_path, model_fd, &m),
              "rebuild after late same-size tamper");
        CHECK(pread_full(b.fd, &got, 1, mark_off) && got == 0,
              "late same-size tamper triggers rebuild");
        check_record_byte(&b, layer, kind, expert, index, changed);
        ds4_expert_bundle_close(&b);
    }

    /* $DS4_BUNDLE_DIR: with no sibling present, the build must land there. */
    CHECK(unlink(bundle_path) == 0, "remove sibling bundle");
    setenv("DS4_BUNDLE_DIR", bundle_dir, 1);
    CHECK(open_bundle(&b, model_path, model_fd, &m), "build in DS4_BUNDLE_DIR");
    CHECK(strncmp(b.path, bundle_dir, strlen(bundle_dir)) == 0,
          "bundle path is under DS4_BUNDLE_DIR");
    check_record_byte(&b, 1, KIND_GATE, 0, 123, tampered);
    for (size_t i = 0; i < sizeof(late_tampers) / sizeof(late_tampers[0]); i++) {
        const uint64_t model_off = tensor_offset(&m,
                                                 late_tampers[i].layer,
                                                 late_tampers[i].kind) +
            (uint64_t)late_tampers[i].expert * kind_bytes(late_tampers[i].kind) +
            late_tampers[i].index;
        uint8_t expected = 0;
        CHECK(pread_full(model_fd, &expected, 1, model_off),
              "read final same-size mutation");
        check_record_byte(&b,
                          late_tampers[i].layer,
                          late_tampers[i].kind,
                          late_tampers[i].expert,
                          late_tampers[i].index,
                          expected);
    }
    ds4_expert_bundle_close(&b);
    unsetenv("DS4_BUNDLE_DIR");

    /* At 124 layers DSEI no longer fits in the original 4 KiB header page.
     * The sidecar must remain a usable, record-compatible plain v1 file. */
    enum { LARGE_LAYER_COUNT = 124 };
    char large_model_path[192];
    snprintf(large_model_path, sizeof(large_model_path),
             "%s/large-layer-table.gguf", dir);
    const int large_fd = open(large_model_path,
                              O_RDWR | O_CREAT | O_TRUNC,
                              0600);
    uint8_t large_model[LARGE_LAYER_COUNT * 3];
    ds4_expert_bundle_layer large_layers[LARGE_LAYER_COUNT];
    for (uint32_t il = 0; il < LARGE_LAYER_COUNT; il++) {
        large_layers[il].gate_offset = (uint64_t)il * 3;
        large_layers[il].up_offset = (uint64_t)il * 3 + 1;
        large_layers[il].down_offset = (uint64_t)il * 3 + 2;
        large_model[il * 3] = (uint8_t)il;
        large_model[il * 3 + 1] = (uint8_t)(il + 1);
        large_model[il * 3 + 2] = (uint8_t)(il + 2);
    }
    CHECK(large_fd >= 0 &&
          write(large_fd, large_model, sizeof(large_model)) ==
              (ssize_t)sizeof(large_model),
          "write 124-layer synthetic model");
    ds4_expert_bundle large_bundle;
    CHECK(ds4_expert_bundle_open_or_build(&large_bundle,
                                          large_model_path,
                                          large_fd,
                                          sizeof(large_model),
                                          0,
                                          LARGE_LAYER_COUNT,
                                          1,
                                          1,
                                          1,
                                          1,
                                          large_layers,
                                          true),
          "124-layer plain v1 bundle builds without DSEI");
    uint8_t large_magic[4] = {0};
    const uint64_t large_id_off =
        (56ull + 8ull * LARGE_LAYER_COUNT + 7ull) & ~7ull;
    CHECK(pread_full(large_bundle.fd, large_magic, sizeof(large_magic),
                     large_id_off) &&
              !(large_magic[0] == 'D' && large_magic[1] == 'S' &&
                large_magic[2] == 'E' && large_magic[3] == 'I'),
          "124-layer bundle leaves DSEI optional");
    ds4_expert_bundle_close(&large_bundle);
    CHECK(ds4_expert_bundle_open_or_build(&large_bundle,
                                          large_model_path,
                                          large_fd,
                                          sizeof(large_model),
                                          0,
                                          LARGE_LAYER_COUNT,
                                          1,
                                          1,
                                          1,
                                          1,
                                          large_layers,
                                          true),
          "124-layer plain v1 bundle reopens");
    ds4_expert_bundle_close(&large_bundle);
    if (large_fd >= 0) close(large_fd);

    close(model_fd);
    if (g_failures == 0) {
        printf("expert_bundle_test: OK\n");
        return 0;
    }
    fprintf(stderr, "expert_bundle_test: %d failure(s)\n", g_failures);
    return 1;
}
