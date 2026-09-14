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

static bool core_read_all(FILE *fp, uint64_t payload, uint32_t chunk, int workers,
                          uint8_t *out, uint64_t expect_total);

static void check_shuffle(size_t n) {
    uint8_t *in = malloc(n + 1), *sh = malloc(n + 1);
    uint8_t *ref = malloc(n + 1), *back = malloc(n + 1);
    assert(in && sh && ref && back);
    fill(in, n, 1);
    in[n] = 0; /* Also initialize the allocation when testing n == 0. */
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

/* Write n bytes through the writer core; return the on-disk payload size. */
static uint64_t codec_write(FILE *fp, const uint8_t *data, size_t n,
                            uint32_t chunk, int workers) {
    const long start = ftell(fp);
    assert(start >= 0);
    kv_lz4_writer *w = kv_lz4_writer_new(fp, chunk, workers);
    assert(w);
    if (n) assert(kv_lz4_writer_write(w, (const char *)data, (kv_lz4_io_size_t)n) == (kv_lz4_io_ssize_t)n);
    assert(kv_lz4_writer_close(w) == 0);
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
    assert(core_read_all(fp, payload, chunk, workers, out, (uint64_t)n));
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

/* Incompressible input still compresses through LZ4, which can only add its
 * own framing overhead.  Bound the growth so a future encoder change cannot
 * silently double the region. */
static void test_incompressible_is_not_larger_than_bound(void) {
    const uint32_t chunk = 1u << 16;
    const size_t n = 4 * chunk;
    uint8_t *data = malloc(n);
    assert(data);
    fill(data, n, 1);
    FILE *fp = tmpfile();
    assert(fp);
    const uint64_t payload = codec_write(fp, data, n, chunk, 2);
    /* LZ4's worst case is n + n/255 + 16 per chunk, plus the chunk record
     * header and 12 bytes of framing. */
    const uint64_t chunks = (n + chunk - 1) / chunk;
    assert(payload <= (uint64_t)n + n / 255 + chunks * (16 + KV_LZ4_RECORD_BYTES) + 12);
    fclose(fp);
    free(data);
    printf("  incompressible payload stays within the stored-block bound: ok\n");
}

/* An empty region carries no framing, so opening one must fail as malformed
 * rather than read past the region into whatever follows it in the file. */
static void test_reader_rejects_empty_region(void) {
    FILE *fp = tmpfile();
    assert(fp);
    const uint8_t junk[64] = {0};
    assert(fwrite(junk, 1, sizeof(junk), fp) == sizeof(junk));
    assert(fseek(fp, 0, SEEK_SET) == 0);
    bool corrupt = false;
    assert(kv_lz4_reader_new(fp, 0, 1u << 16, 1, &corrupt) == NULL && corrupt);
    rewind(fp);
    /* A complete zero-total/zero-chunk frame used to allocate eight pools of
     * maximum-sized scratch despite representing no payload at all. */
    corrupt = false;
    assert(kv_lz4_reader_new(fp, 12, DS4_KVSTORE_MAX_CHUNK_BYTES, 8, &corrupt) == NULL && corrupt);
    fclose(fp);
    printf("  reader rejects an empty payload region as malformed: ok\n");
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

    bool corrupt = false;
    assert(fseek(fp, 0, SEEK_SET) == 0);
    assert(kv_lz4_reader_new(fp, payload, 0, 1, &corrupt) == NULL && corrupt);
    assert(fseek(fp, 0, SEEK_SET) == 0);
    corrupt = false;
    assert(kv_lz4_reader_new(fp, payload, DS4_KVSTORE_MAX_CHUNK_BYTES + 1, 1, &corrupt) == NULL && corrupt);
    /* Sanity: the same file does open at its real chunk size, so the two
     * rejections above are attributable to the bound, not a broken fixture. */
    assert(fseek(fp, 0, SEEK_SET) == 0);
    kv_lz4_reader *ok = kv_lz4_reader_new(fp, payload, chunk, 1, &corrupt);
    assert(ok && !corrupt && ok->uncompressed_total == (uint64_t)n);
    (void)kv_lz4_reader_close(ok);

    fclose(fp);
    free(data);
    printf("  reader rejects zero and over-bound chunk sizes: ok\n");
}

/* Emit one chunk record in the writer's format: shuffle, LZ4-HC level 1, then
 * [le32 raw][le32 comp][le32 xxh32(raw)][comp bytes].  bad_sum perturbs only
 * the checksum, so a test can vary one field at a time. */
static void emit_chunk_sum(FILE *fp, const uint8_t *raw, uint32_t raw_len, bool bad_sum) {
    uint8_t *shuf = malloc(raw_len);
    const int bound = LZ4_compressBound((int)raw_len);
    uint8_t *comp = malloc((size_t)bound);
    assert(shuf && comp);
    kv_shuffle_byte4(raw, shuf, raw_len);
    const int n = LZ4_compress_HC((const char *)shuf, (char *)comp,
                                  (int)raw_len, bound, 1);
    assert(n > 0);
    uint8_t rec[KV_LZ4_RECORD_BYTES];
    ds4_kvstore_le_put32(rec, raw_len);
    ds4_kvstore_le_put32(rec + 4, (uint32_t)n);
    ds4_kvstore_le_put32(rec + 8, kv_xxh32(raw, raw_len) ^ (bad_sum ? 1u : 0u));
    assert(fwrite(rec, 1, sizeof(rec), fp) == sizeof(rec));
    assert(fwrite(comp, 1, (size_t)n, fp) == (size_t)n);
    free(shuf); free(comp);
}

/* Build a stream whose framing declares `total` over `count` chunks, with the
 * given raw sizes.  bad_sum_chunk >= 0 perturbs that chunk's checksum.
 * Returns the region size. */
static uint64_t emit_stream(FILE *fp, uint64_t total, uint32_t count,
                            const uint32_t *raws, const uint8_t *src,
                            int bad_sum_chunk) {
    const long start = ftell(fp);
    uint8_t framing[12];
    kv_le_put64(framing, total);
    ds4_kvstore_le_put32(framing + 8, count);
    assert(fwrite(framing, 1, sizeof(framing), fp) == sizeof(framing));
    size_t off = 0;
    for (uint32_t i = 0; i < count; i++) {
        emit_chunk_sum(fp, src + off, raws[i], (int)i == bad_sum_chunk);
        off += raws[i];
    }
    const long end = ftell(fp);
    return (uint64_t)(end - start);
}

/* Decode a region through the reader core, which needs no cookie streams.
 * Returns true only if every declared byte was produced; out may be NULL. */
static bool core_read_all(FILE *fp, uint64_t payload, uint32_t chunk, int workers,
                          uint8_t *out, uint64_t expect_total) {
    bool corrupt = false;
    kv_lz4_reader *r = kv_lz4_reader_new(fp, payload, chunk, workers, &corrupt);
    if (!r) return false;
    bool ok = r->uncompressed_total == expect_total;
    uint8_t *buf = out ? out : (ok ? malloc((size_t)expect_total) : NULL);
    if (ok) assert(buf);
    uint64_t got = 0;
    while (ok && got < expect_total) {
        const kv_lz4_io_ssize_t n = kv_lz4_reader_read(
                r, (char *)buf + got, (kv_lz4_io_size_t)(expect_total - got));
        if (n <= 0) ok = false; else got += (uint64_t)n;
    }
    if (!out) free(buf);
    (void)kv_lz4_reader_close(r);
    return ok;
}

/* Read a whole region back; true if the reader accepted every declared byte. */
static bool stream_accepted(FILE *fp, uint64_t payload, uint32_t chunk,
                            uint64_t expect_total) {
    assert(fseek(fp, 0, SEEK_SET) == 0);
    return core_read_all(fp, payload, chunk, 2, NULL, expect_total);
}

/* Scratch is sized from the header, so a region too small to hold the chunks
 * it claims must be rejected before anything is allocated. */
static void test_reader_rejects_impossible_framing(void) {
    FILE *fp = tmpfile();
    assert(fp);
    uint8_t framing[12] = {0};
    kv_le_put64(framing, 16);             /* uncompressed_total = 16 */
    ds4_kvstore_le_put32(framing + 8, 1); /* chunk_count = 1 */
    assert(fwrite(framing, 1, sizeof(framing), fp) == sizeof(framing));
    assert(fseek(fp, 0, SEEK_SET) == 0);
    /* Framing consumes all 12 bytes, leaving nothing for the chunk record. */
    bool corrupt = false;
    assert(kv_lz4_reader_new(fp, sizeof(framing), DS4_KVSTORE_MAX_CHUNK_BYTES,
                             8, &corrupt) == NULL && corrupt);
    /* Nonempty framing must not bypass the same resource guard merely by
     * including enough bytes for the advertised record headers. */
    rewind(fp);
    kv_le_put64(framing, (uint64_t)DS4_KVSTORE_MAX_CHUNK_BYTES * 8u);
    ds4_kvstore_le_put32(framing + 8, 8);
    assert(fwrite(framing, 1, sizeof(framing), fp) == sizeof(framing));
    const uint8_t records[8 * KV_LZ4_RECORD_BYTES] = {0};
    assert(fwrite(records, 1, sizeof(records), fp) == sizeof(records));
    rewind(fp);
    corrupt = false;
    assert(kv_lz4_reader_new(fp, sizeof(framing) + sizeof(records),
                             DS4_KVSTORE_MAX_CHUNK_BYTES, 8, &corrupt) == NULL && corrupt);
    fclose(fp);
    printf("  reader rejects a region too small for the chunks it claims: ok\n");
}

/* Only the final chunk may be short.  Everything else about this stream is
 * self-consistent -- chunk_count matches ceil(total/chunk) and the raw sizes
 * sum to total -- so only the invariant can reject it. */
static void test_reader_enforces_full_non_final_chunks(void) {
    const uint32_t chunk = 4096;
    const uint64_t total = 5000;                 /* ceil(5000/4096) == 2 */
    const uint32_t raws[2] = {2048, 2952};       /* chunk 0 short, sum == total */
    uint8_t *src = malloc((size_t)total);
    assert(src);
    fill(src, (size_t)total, 3);

    FILE *fp = tmpfile();
    assert(fp);
    const uint64_t payload = emit_stream(fp, total, 2, raws, src, -1);
    assert(!stream_accepted(fp, payload, chunk, total));

    /* Control: the same builder with a full leading chunk is accepted, so the
     * rejection above is the invariant and not a broken fixture. */
    FILE *ok = tmpfile();
    assert(ok);
    const uint32_t good[2] = {4096, 904};
    const uint64_t good_payload = emit_stream(ok, total, 2, good, src, -1);
    assert(stream_accepted(ok, good_payload, chunk, total));

    fclose(ok); fclose(fp); free(src);
    printf("  reader rejects a short non-final chunk: ok\n");
}

/* The raw sizes must sum to the declared total. */
static void test_reader_enforces_raw_total(void) {
    const uint32_t chunk = 4096;
    const uint64_t total = 5000;
    const uint32_t raws[2] = {4096, 1000};       /* sums to 5096, not 5000 */
    uint8_t *src = malloc((size_t)raws[0] + raws[1]);
    assert(src);
    fill(src, (size_t)raws[0] + raws[1], 3);

    FILE *fp = tmpfile();
    assert(fp);
    const uint64_t payload = emit_stream(fp, total, 2, raws, src, -1);
    /* Unchecked, the reader would return a full 5000 bytes and drop the rest. */
    assert(!stream_accepted(fp, payload, chunk, total));

    fclose(fp); free(src);
    printf("  reader rejects raw sizes that do not sum to the declared total: ok\n");
}

/* Closing the writer must restore the position to the end of the region it
 * wrote, not the end of the file. */
static void test_writer_restores_region_end(void) {
    const uint32_t chunk = 1u << 12;
    uint8_t junk[8192];
    fill(junk, sizeof(junk), 3);
    FILE *fp = tmpfile();
    assert(fp);
    assert(fwrite(junk, 1, sizeof(junk), fp) == sizeof(junk));
    assert(fseek(fp, 0, SEEK_SET) == 0);

    const uint8_t small[64] = {0};
    kv_lz4_writer *w = kv_lz4_writer_new(fp, chunk, 1);
    assert(w);
    assert(kv_lz4_writer_write(w, (const char *)small, sizeof(small)) == (kv_lz4_io_ssize_t)sizeof(small));
    assert(kv_lz4_writer_close(w) == 0);

    const long pos = ftell(fp);
    assert(pos > 0 && (size_t)pos < sizeof(junk));
    fclose(fp);
    printf("  writer close leaves the stream at the region end: ok\n");
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
        /* With per-chunk checksums no mode may produce the full payload. */
        assert(!core_read_all(fp, payload, chunk, 2, out, n));
        fclose(fp);
    }
    free(data); free(out);
    printf("  truncated framing, truncated region and corrupt body all rejected: ok\n");
}

/* The payload region is compressed through the writer core on every platform,
 * starting at a nonzero offset as in production, and decodes byte for byte.
 * A chunk size that cannot round-trip through log2 falls back to the default. */
static void test_write_payload_region(void) {
    const uint32_t chunks[] = {1u << 12, 5000, 0, 128u << 20};
    for (size_t ci = 0; ci < sizeof(chunks) / sizeof(*chunks); ci++) {
        const uint32_t chunk_bytes = chunks[ci];
        const uint32_t used = (chunk_bytes == 0 || chunk_bytes > DS4_KVSTORE_MAX_CHUNK_BYTES ||
                               (chunk_bytes & (chunk_bytes - 1u)))
                              ? DS4_KVSTORE_DEFAULT_CHUNK_BYTES : chunk_bytes;
        const size_t n = 9 * (1u << 12) + 37;   /* several 4 KiB chunks */
        uint8_t *data = malloc(n), *back = malloc(n);
        assert(data && back);
        fill(data, n, 2);

        char staged_path[] = "/tmp/ds4_kv_lz4_staged_XXXXXX";
        int sfd = mkstemp(staged_path);
        assert(sfd >= 0);
        FILE *sf = fdopen(sfd, "wb");
        assert(sf && fwrite(data, 1, n, sf) == n);
        assert(fclose(sf) == 0);
        const ds4_session_payload_file staged = {.path = staged_path, .bytes = n};

        FILE *fp = tmpfile();
        assert(fp);
        const uint8_t lead[97] = {0};
        assert(fwrite(lead, 1, sizeof(lead), fp) == sizeof(lead));
        uint8_t codec = 0xFF, chunk_log2 = 0xFF;
        uint64_t on_disk = 0;
        char err[256] = {0};
        const bool ok = ds4_kvstore_write_payload_region(
                fp, &staged, 4, chunk_bytes,
                &codec, &chunk_log2, &on_disk, err, sizeof(err));
        assert(ok && err[0] == '\0');
        assert(codec == DS4_KVSTORE_CODEC_LZ4);
        assert((1u << chunk_log2) == used);
        assert(on_disk > 0);

        assert(fseek(fp, (long)sizeof(lead), SEEK_SET) == 0);
        assert(core_read_all(fp, on_disk, used, 4, back, n));
        assert(memcmp(back, data, n) == 0);

        fclose(fp);
        unlink(staged_path);
        free(data); free(back);
    }
    printf("  payload region compresses without cookie streams and round-trips: ok\n");
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

    int silent_wrong = 0;
    for (int iter = 0; iter < 3000; iter++) {
        FILE *fp = tmpfile();
        assert(fp);
        uint64_t payload;
        if (iter % 3 == 0) {                       /* wholly random region */
            const uint8_t plo = rng_byte(), phi = rng_byte();
            payload = 16 + ((uint64_t)plo | ((uint64_t)phi << 8)) % 4096;
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
                const uint8_t hi = rng_byte(), lo = rng_byte();
                uint64_t off = ((uint64_t)hi << 8 | lo) % base_payload;
                m[off] ^= (uint8_t)(1u << (rng_byte() & 7));
            }
            payload = base_payload;
            if (iter % 9 == 4) payload = base_payload / (2 + rng_byte() % 4);
            assert(fwrite(m, 1, (size_t)payload, fp) == payload);
            free(m);
        }
        assert(fseek(fp, 0, SEEK_SET) == 0);
        bool corrupt = false;
        kv_lz4_reader *r = kv_lz4_reader_new(fp, payload, chunk, 2, &corrupt);
        if (r) {
            const uint64_t total = r->uncompressed_total;
            /* Cap the claimed size: a mutated header must not make the test
             * itself allocate without bound. */
            if (total <= 64u * 1024u * 1024u) {
                size_t want = total < n ? (size_t)total : n;
                size_t got = 0;
                while (got < want) {
                    const kv_lz4_io_ssize_t k = kv_lz4_reader_read(
                            r, (char *)out + got, (kv_lz4_io_size_t)(want - got));
                    if (k <= 0) break;
                    got += (size_t)k;
                }
                /* Each chunk carries a checksum of its raw bytes, so a
                 * mutation must never decode to full-length wrong bytes. */
                if (got == want && want == n && memcmp(out, data, n) != 0)
                    silent_wrong++;
            }
            (void)kv_lz4_reader_close(r);
        }
        fclose(fp);
    }
    free(blob); free(data); free(out);
    assert(silent_wrong == 0);
    printf("  3000 mutated and random regions: no crash, hang or silent wrong bytes: ok\n");
}

/* Without cookie streams the only missing piece is the stdio wrapper used to
 * hand a decoder to the engine: compression stays on by default. */
static void test_no_cookie_streams(void) {
    const char *env = getenv("DS4_KV_CACHE_COMPRESSION_THREADS");
    if (!env || !env[0]) assert(kv_cache_default_compression_threads() >= 1);
    assert(kv_lz4_fwrap_open(&env, "rb", NULL, NULL, NULL) == NULL);
    printf("  no cookie streams: stdio wrapper absent, compression on by default: ok\n");
}

/* Reference values from xxhsum 0.8.3 -H32, covering the short, 4-byte and
 * 1-byte tail paths and the 16-byte stripes. */
static void test_xxh32(void) {
    assert(kv_xxh32((const uint8_t *)"", 0) == 0x02cc5d05u);
    assert(kv_xxh32((const uint8_t *)"abc", 3) == 0x32d153ffu);
    const size_t n = 100003;
    uint8_t *p = malloc(n);
    assert(p);
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)((i * 131 + 7) & 255);
    assert(kv_xxh32(p, n) == 0xf0ef88e1u);
    static const struct { size_t len; uint32_t want; } v[] = {
        {7, 0x02226fe7u}, {16, 0x96ac9fa6u}, {17, 0x3cdfb1a2u}, {23, 0x12a0921bu},
        {31, 0xdf853fbfu}, {32, 0xda8072dcu}, {33, 0x3c8fd07du},
    };
    for (size_t k = 0; k < sizeof(v) / sizeof(*v); k++) {
        for (size_t i = 0; i < v[k].len; i++) p[i] = (uint8_t)((i * 37 + 11) & 255);
        assert(kv_xxh32(p, v[k].len) == v[k].want);
    }
    free(p);
    printf("  XXH32 matches xxhsum reference values: ok\n");
}

/* Raw files keep version 1 so upstream binaries can read and refresh them;
 * only compressed files are version 2, and refreshing must not upgrade. */
static void test_header_versions(void) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, 1, 2, DS4_KVSTORE_REASON_COLD, 0,
                            DS4_KVSTORE_CODEC_NONE, 0, 128, 0, 65536, 1, 1, 64);
    assert(h[3] == KV_CACHE_VERSION_COMPAT);
    ds4_kvstore_fill_header(h, 1, 2, DS4_KVSTORE_REASON_COLD, 0,
                            DS4_KVSTORE_CODEC_LZ4, 24, 128, 0, 65536, 1, 1, 64);
    assert(h[3] == KV_CACHE_VERSION);

    char path[] = "/tmp/ds4_kv_lz4_hdr_XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    assert(fp);
    static const char text[] = "cached text";
    uint8_t tb[4];
    ds4_kvstore_fill_header(h, 1, 2, DS4_KVSTORE_REASON_COLD, 0,
                            DS4_KVSTORE_CODEC_NONE, 0, 128, 0, 65536, 1, 1, 0);
    ds4_kvstore_le_put32(tb, (uint32_t)strlen(text));
    assert(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    assert(fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb));
    assert(fwrite(text, 1, strlen(text), fp) == strlen(text));
    assert(fclose(fp) == 0);

    assert(ds4_kvstore_touch_file(path, 7));
    fp = fopen(path, "rb");
    assert(fp);
    ds4_kvstore_entry e = {0};
    uint32_t text_bytes = 0;
    assert(ds4_kvstore_read_header(fp, &e, &text_bytes) && e.hits == 7);
    assert(fseek(fp, 3, SEEK_SET) == 0 && fgetc(fp) == (int)KV_CACHE_VERSION_COMPAT);
    assert(fclose(fp) == 0);

    /* A version-1 header never carried a codec; reject that combination. */
    fp = fopen(path, "r+b");
    assert(fp && fseek(fp, 21, SEEK_SET) == 0);
    assert(fputc(DS4_KVSTORE_CODEC_LZ4, fp) != EOF && fputc(24, fp) != EOF);
    assert(fseek(fp, 0, SEEK_SET) == 0);
    assert(!ds4_kvstore_read_header(fp, &e, &text_bytes));

    /* An earlier build stamped raw files as version 2.  They still load, and a
     * refresh rewrites them as version 1. */
    assert(fseek(fp, 3, SEEK_SET) == 0 && fputc(2, fp) != EOF);
    assert(fseek(fp, 21, SEEK_SET) == 0 && fputc(0, fp) != EOF && fputc(0, fp) != EOF);
    assert(fseek(fp, 0, SEEK_SET) == 0);
    assert(ds4_kvstore_read_header(fp, &e, &text_bytes));
    assert(fclose(fp) == 0);
    assert(ds4_kvstore_touch_file(path, 8));
    fp = fopen(path, "rb");
    assert(fp && fseek(fp, 3, SEEK_SET) == 0 && fgetc(fp) == (int)KV_CACHE_VERSION_COMPAT);
    assert(fclose(fp) == 0);
    unlink(path);
    printf("  raw headers end version 1 through refresh; v1 with a codec rejected: ok\n");
}

/* A stream that is well formed and decodes to the right length must still be
 * rejected when one chunk's checksum disagrees with its bytes. */
static void test_reader_verifies_chunk_checksums(void) {
    const uint32_t chunk = 4096;
    const uint64_t total = 5000;
    const uint32_t raws[2] = {4096, 904};
    uint8_t *src = malloc((size_t)total);
    assert(src);
    fill(src, (size_t)total, 3);
    for (int bad = -1; bad < 2; bad++) {
        FILE *fp = tmpfile();
        assert(fp);
        const uint64_t payload = emit_stream(fp, total, 2, raws, src, bad);
        assert(stream_accepted(fp, payload, chunk, total) == (bad < 0));
        fclose(fp);
    }
    free(src);
    printf("  reader rejects a mismatched checksum on either chunk: ok\n");
}

int main(void) {
    rng_seed(0x9E3779B97F4A7C15ull);
    test_shuffle();
    test_xxh32();
    test_header_versions();
    test_roundtrip();
    test_incompressible_is_not_larger_than_bound();
    test_reader_rejects_empty_region();
    test_reader_rejects_bad_chunk_size();
    test_reader_rejects_impossible_framing();
    test_reader_enforces_full_non_final_chunks();
    test_reader_enforces_raw_total();
    test_reader_verifies_chunk_checksums();
    test_writer_restores_region_end();
    test_truncated_and_corrupt();
    test_fuzz_regions();
    test_write_payload_region();
    if (!KV_LZ4_HAVE_FWRAP) test_no_cookie_streams();
    printf("Disk KV lz4 codec%s: PASS\n",
           KV_LZ4_HAVE_FWRAP ? "" : ", no cookie streams");
    return 0;
}
