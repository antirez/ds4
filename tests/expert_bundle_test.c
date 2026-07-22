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

static int open_bundle(ds4_expert_bundle *b, const char *model_path,
                       int model_fd, const model_layout *m) {
    return ds4_expert_bundle_open_or_build(b, model_path, model_fd,
                                           m->model_size,
                                           LAYER_LO, LAYER_COUNT, N_EXPERT,
                                           GATE_BYTES, UP_BYTES, DOWN_BYTES,
                                           m->layers);
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

int main(void) {
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

    /* Mark the padding of the first record: a REUSED bundle keeps the mark,
     * a rebuilt one zeroes it. */
    const uint64_t mark_off =
        ALIGN + GATE_BYTES + UP_BYTES + DOWN_BYTES;   /* record 0 padding */
    int wfd = open(bundle_path, O_WRONLY);
    uint8_t mark = 0xAB;
    CHECK(wfd >= 0 && pwrite(wfd, &mark, 1, (off_t)mark_off) == 1, "mark");
    close(wfd);
    CHECK(open_bundle(&b, model_path, model_fd, &m), "reopen");
    uint8_t got = 0;
    CHECK(pread_full(b.fd, &got, 1, mark_off) && got == 0xAB,
          "existing bundle reused without rebuild");
    ds4_expert_bundle_close(&b);

    /* A bundle written by the original Swift v1 code must be reused as-is:
     * the padding marker proves open_or_build did not replace the file. */
    CHECK(rewrite_bundle_hashes(bundle_path, model_fd, &m, HASH_SWIFT_V1),
          "write legacy Swift v1 fingerprints");
    CHECK(open_bundle(&b, model_path, model_fd, &m),
          "open legacy Swift v1 bundle");
    CHECK(pread_full(b.fd, &got, 1, mark_off) && got == 0xAB,
          "legacy Swift v1 bundle reused without rebuild");
    ds4_expert_bundle_close(&b);

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

    /* $DS4_BUNDLE_DIR: with no sibling present, the build must land there. */
    CHECK(unlink(bundle_path) == 0, "remove sibling bundle");
    setenv("DS4_BUNDLE_DIR", bundle_dir, 1);
    CHECK(open_bundle(&b, model_path, model_fd, &m), "build in DS4_BUNDLE_DIR");
    CHECK(strncmp(b.path, bundle_dir, strlen(bundle_dir)) == 0,
          "bundle path is under DS4_BUNDLE_DIR");
    check_records(&b, &tampered, 1, 0, 123);
    ds4_expert_bundle_close(&b);
    unsetenv("DS4_BUNDLE_DIR");

    close(model_fd);
    if (g_failures == 0) {
        printf("expert_bundle_test: OK\n");
        return 0;
    }
    fprintf(stderr, "expert_bundle_test: %d failure(s)\n", g_failures);
    return 1;
}
