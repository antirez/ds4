/* Disk KV codec: byte-4 shuffle inverse, lz4 stream round-trip, and
 * rejection of malformed payload regions.  No model or GPU required. */
#include "../ds4_kvstore.c"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t rng;
static void rng_seed(uint64_t s) { rng = s ? s : 1; }
static uint8_t rng_byte(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint8_t)(rng >> 33);
}

/* Independent definition of the transpose: position-stream b holds every
 * 4th byte starting at b, tail bytes pass through.  The SIMD paths must
 * agree with this for every length. */
static void ref_shuffle(const uint8_t *in, uint8_t *out, size_t n) {
    size_t m = n >> 2;
    for (size_t b = 0; b < 4; b++)
        for (size_t i = 0; i < m; i++) out[b * m + i] = in[i * 4 + b];
    for (size_t i = m << 2; i < n; i++) out[i] = in[i];
}

static void fill(uint8_t *p, size_t n, int kind) {
    for (size_t i = 0; i < n; i++) {
        switch (kind) {
        case 0: p[i] = 0; break;                       /* fully compressible */
        case 1: p[i] = rng_byte(); break;              /* incompressible */
        case 2: p[i] = (uint8_t)(i & 3 ? 0 : i >> 2); break; /* KV-like: 1 of 4 varies */
        default: p[i] = (uint8_t)("the quick brown fox "[i % 20]); break;
        }
    }
}

static void check_shuffle(size_t n) {
    uint8_t *in = malloc(n + 1), *sh = malloc(n + 1);
    uint8_t *ref = malloc(n + 1), *back = malloc(n + 1);
    assert(in && sh && ref && back);
    fill(in, n, 1);
    /* Guard byte catches a one-past-the-end write in either direction. */
    sh[n] = 0xA5; back[n] = 0x5A;

    kv_shuffle_byte4(in, sh, n);
    ref_shuffle(in, ref, n);
    assert(memcmp(sh, ref, n) == 0);
    assert(sh[n] == 0xA5);

    kv_unshuffle_byte4(sh, back, n);
    assert(memcmp(back, in, n) == 0);
    assert(back[n] == 0x5A);

    free(in); free(sh); free(ref); free(back);
}

static void test_shuffle(void) {
    /* Every small length, then the 64-byte SIMD block boundaries and the
     * non-multiple-of-4 tails around them. */
    for (size_t n = 0; n <= 300; n++) check_shuffle(n);
    const size_t edges[] = {511, 512, 513, 1023, 1024, 1025, 4093, 4096, 4099,
                            65535, 65536, 65537, 1u << 20};
    for (size_t i = 0; i < sizeof(edges) / sizeof(*edges); i++)
        check_shuffle(edges[i]);
    printf("  shuffle/unshuffle inverse and reference agreement: ok\n");
}

/* Write n bytes through the codec; return the on-disk payload size. */
static uint64_t codec_write(FILE *fp, const uint8_t *data, size_t n,
                            uint32_t chunk, int workers) {
    const long start = ftell(fp);
    assert(start >= 0);
    FILE *cw = kv_lz4_writer_open(fp, chunk, workers);
    assert(cw);
    if (n) assert(fwrite(data, 1, n, cw) == n);
    assert(fclose(cw) == 0);
    const long end = ftell(fp);
    assert(end >= start);
    return (uint64_t)(end - start);
}

static void check_roundtrip(size_t n, uint32_t chunk, int workers, int kind) {
    uint8_t *data = malloc(n + 1), *out = malloc(n + 1);
    assert(data && out);
    fill(data, n, kind);
    out[n] = 0xC3;

    FILE *fp = tmpfile();
    assert(fp);
    const uint64_t payload = codec_write(fp, data, n, chunk, workers);

    assert(fseek(fp, 0, SEEK_SET) == 0);
    uint64_t total = 0;
    FILE *cr = kv_lz4_reader_open(fp, payload, chunk, workers, &total);
    assert(cr);
    assert(total == (uint64_t)n);
    if (n) assert(fread(out, 1, n, cr) == n);
    assert(fclose(cr) == 0);
    assert(memcmp(out, data, n) == 0);
    assert(out[n] == 0xC3);

    fclose(fp);
    free(data); free(out);
}

static void test_roundtrip(void) {
    /* Zero bytes is excluded here: the writer emits no framing for an empty
     * payload, so the region it produces cannot be reopened. */
    const uint32_t chunk = 1u << 16;
    const size_t sizes[] = {1, 3, 4, 5, 1024, chunk - 1, chunk, chunk + 1,
                            3 * chunk, 3 * chunk + 7};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++)
        for (int w = 1; w <= 4; w++)
            for (int kind = 0; kind < 4; kind++)
                check_roundtrip(sizes[i], chunk, w, kind);
    /* Chunk sizes other than the default, including the documented bound. */
    check_roundtrip(300000, 1u << 12, 2, 2);
    check_roundtrip(300000, DS4_KVSTORE_DEFAULT_CHUNK_BYTES, 2, 2);
    printf("  lz4 round-trip across sizes, chunks, workers and data kinds: ok\n");
}

/* Incompressible input must still round-trip: it exercises the stored-block
 * path where the encoded chunk is no smaller than the raw chunk. */
static void test_incompressible_is_not_larger_than_bound(void) {
    const uint32_t chunk = 1u << 16;
    const size_t n = 4 * chunk;
    uint8_t *data = malloc(n);
    assert(data);
    fill(data, n, 1);
    FILE *fp = tmpfile();
    assert(fp);
    const uint64_t payload = codec_write(fp, data, n, chunk, 2);
    /* Framing plus per-chunk headers stay bounded; a blow-up here means a
     * stored chunk is being re-expanded. */
    assert(payload < (uint64_t)n + n / 16 + 1024);
    fclose(fp);
    free(data);
    printf("  incompressible payload stays within the stored-block bound: ok\n");
}

/* An empty region carries no framing, so opening one must fail rather than
 * read past the region into whatever follows it in the file. */
static void test_reader_rejects_empty_region(void) {
    FILE *fp = tmpfile();
    assert(fp);
    const uint8_t junk[64] = {0};
    assert(fwrite(junk, 1, sizeof(junk), fp) == sizeof(junk));
    assert(fseek(fp, 0, SEEK_SET) == 0);
    uint64_t total = 0;
    assert(kv_lz4_reader_open(fp, 0, 1u << 16, 1, &total) == NULL);
    fclose(fp);
    printf("  reader rejects an empty payload region: ok\n");
}

/* Backed by a stream whose framing is valid, so the rejection has to come from
 * the chunk-size bound itself rather than from a failed framing read. */
static void test_reader_rejects_bad_chunk_size(void) {
    const uint32_t chunk = 1u << 12;
    const size_t n = 3 * chunk;
    uint8_t *data = malloc(n);
    assert(data);
    fill(data, n, 3);

    FILE *fp = tmpfile();
    assert(fp);
    const uint64_t payload = codec_write(fp, data, n, chunk, 1);

    uint64_t total = 0;
    assert(fseek(fp, 0, SEEK_SET) == 0);
    assert(kv_lz4_reader_open(fp, payload, 0, 1, &total) == NULL);
    assert(fseek(fp, 0, SEEK_SET) == 0);
    assert(kv_lz4_reader_open(fp, payload, DS4_KVSTORE_MAX_CHUNK_BYTES + 1,
                              1, &total) == NULL);
    /* Sanity: the same file does open at its real chunk size, so the two
     * rejections above are attributable to the bound, not a broken fixture. */
    assert(fseek(fp, 0, SEEK_SET) == 0);
    FILE *ok = kv_lz4_reader_open(fp, payload, chunk, 1, &total);
    assert(ok && total == (uint64_t)n);
    fclose(ok);

    fclose(fp);
    free(data);
    printf("  reader rejects zero and over-bound chunk sizes: ok\n");
}

/* A truncated or corrupted payload region must fail the read rather than
 * return short data as success, hang, or touch memory outside the buffers. */
static void test_truncated_and_corrupt(void) {
    const uint32_t chunk = 1u << 14;
    const size_t n = 5 * chunk + 11;
    uint8_t *data = malloc(n), *out = malloc(n);
    assert(data && out);
    fill(data, n, 3);

    for (int mode = 0; mode < 3; mode++) {
        FILE *fp = tmpfile();
        assert(fp);
        uint64_t payload = codec_write(fp, data, n, chunk, 2);
        assert(payload > 64);

        if (mode == 0) payload /= 2;              /* region cut short */
        else if (mode == 1) payload = 4;          /* framing header cut short */
        else {                                    /* body bytes corrupted */
            assert(fseek(fp, (long)(payload / 2), SEEK_SET) == 0);
            for (int i = 0; i < 64; i++) {
                uint8_t b = 0xFF;
                assert(fwrite(&b, 1, 1, fp) == 1);
            }
        }

        assert(fseek(fp, 0, SEEK_SET) == 0);
        uint64_t total = 0;
        FILE *cr = kv_lz4_reader_open(fp, payload, chunk, 2, &total);
        bool failed = (cr == NULL);
        if (cr) {
            /* Either the read reports short, or the data differs; silently
             * returning n bytes that do not match the input is a failure. */
            size_t got = fread(out, 1, n, cr);
            if (got != n || memcmp(out, data, n) != 0) failed = true;
            fclose(cr);
        }
        assert(failed);
        fclose(fp);
    }
    free(data); free(out);
    printf("  truncated framing, truncated region and corrupt body all rejected: ok\n");
}

/* Mutated and wholly random regions must fail or return wrong bytes, never
 * crash, read out of bounds or spin.  Deterministic seed so a failure repeats. */
static void test_fuzz_regions(void) {
    const uint32_t chunk = 1u << 12;
    const size_t n = 9 * chunk + 5;
    uint8_t *data = malloc(n), *out = malloc(n);
    assert(data && out);
    fill(data, n, 2);

    /* One valid stream, reused as the mutation base. */
    FILE *base = tmpfile();
    assert(base);
    const uint64_t base_payload = codec_write(base, data, n, chunk, 2);
    uint8_t *blob = malloc((size_t)base_payload);
    assert(blob);
    assert(fseek(base, 0, SEEK_SET) == 0);
    assert(fread(blob, 1, (size_t)base_payload, base) == base_payload);
    fclose(base);

    for (int iter = 0; iter < 3000; iter++) {
        FILE *fp = tmpfile();
        assert(fp);
        uint64_t payload;
        if (iter % 3 == 0) {                       /* wholly random region */
            payload = 16 + (rng_byte() | ((uint64_t)rng_byte() << 8)) % 4096;
            for (uint64_t i = 0; i < payload; i++) {
                uint8_t b = rng_byte();
                assert(fwrite(&b, 1, 1, fp) == 1);
            }
        } else {                                   /* mutated valid stream */
            uint8_t *m = malloc((size_t)base_payload);
            assert(m);
            memcpy(m, blob, (size_t)base_payload);
            int flips = 1 + rng_byte() % 8;
            for (int f = 0; f < flips; f++) {
                uint64_t off = ((uint64_t)rng_byte() << 8 | rng_byte()) % base_payload;
                m[off] ^= (uint8_t)(1u << (rng_byte() & 7));
            }
            payload = base_payload;
            if (iter % 9 == 4) payload = base_payload / (2 + rng_byte() % 4);
            assert(fwrite(m, 1, (size_t)payload, fp) == payload);
            free(m);
        }
        assert(fseek(fp, 0, SEEK_SET) == 0);
        uint64_t total = 0;
        FILE *cr = kv_lz4_reader_open(fp, payload, chunk, 2, &total);
        if (cr) {
            /* Cap the claimed size: a mutated header must not make the test
             * itself allocate without bound. */
            if (total <= 64u * 1024u * 1024u) {
                size_t want = total < n ? (size_t)total : n;
                (void)fread(out, 1, want, cr);
            }
            fclose(cr);
        }
        fclose(fp);
    }
    free(blob); free(data); free(out);
    printf("  3000 mutated and random regions handled without crash or hang: ok\n");
}

int main(void) {
    rng_seed(0x9E3779B97F4A7C15ull);
    test_shuffle();
    test_roundtrip();
    test_incompressible_is_not_larger_than_bound();
    test_reader_rejects_empty_region();
    test_reader_rejects_bad_chunk_size();
    test_truncated_and_corrupt();
    test_fuzz_regions();
    printf("Disk KV lz4 codec: PASS\n");
    return 0;
}
