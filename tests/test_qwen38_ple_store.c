/* Qwen 3.8 PLE row store: explicit F_NOCACHE reads plus the set-associative
 * row cache that replaced faulting the 51 GiB sidecar through its mapping.
 * The table is synthetic here -- what matters is that every id yields its own
 * row byte for byte whatever the cache does underneath, that eviction never
 * hands back a stale or half-written row, and that the accounting adds up. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ds4.h"

#define ROW_BYTES   170
#define BASE_OFFSET 37   /* deliberately unaligned: exercises the offset math */
#define TABLE_ROWS  65536

static char g_path[512];

static void fail(const char *what) {
    fprintf(stderr, "test_qwen38_ple_store: %s\n", what);
    unlink(g_path);
    exit(1);
}

/* Row i is a pattern only row i can produce, and it is valid Q8_0: five
 * blocks of an f16 scale (1, 2, 4, 8, 16 -- exact, so the dequantized value
 * is predictable below without an f16 decoder) plus 32 quantized bytes. */
static void expected_row(uint64_t i, uint8_t *out) {
    for (uint32_t b = 0; b < ROW_BYTES / 34; b++) {
        uint8_t *blk = out + b * 34;
        const uint16_t scale = (uint16_t)(0x3C00u + (b << 10));
        memcpy(blk, &scale, sizeof(scale));
        for (uint32_t j = 0; j < 32; j++) {
            blk[2 + j] = (uint8_t)((i * 31u + b * 17u + j * 7u + (i >> 8)) & 0xFFu);
        }
    }
}

/* The float that row `id` must produce at position `idx` of its 160 values. */
static float expected_value(uint64_t id, uint32_t idx) {
    const uint32_t b = idx / 32, j = idx % 32;
    const uint8_t q = (uint8_t)((id * 31u + b * 17u + j * 7u + (id >> 8)) & 0xFFu);
    return (float)(int8_t)q * (float)(1u << b);
}

static void write_table(void) {
    const char *tmp = getenv("TMPDIR");
    snprintf(g_path, sizeof(g_path), "%sds4_ple_store_test.bin",
             tmp && tmp[0] ? tmp : "/tmp/");
    FILE *f = fopen(g_path, "wb");
    if (!f) fail("cannot create the temporary table");

    uint8_t pad[BASE_OFFSET];
    memset(pad, 0xA5, sizeof(pad));
    if (fwrite(pad, 1, sizeof(pad), f) != sizeof(pad)) fail("cannot write padding");

    uint8_t row[ROW_BYTES];
    for (uint64_t i = 0; i < TABLE_ROWS; i++) {
        expected_row(i, row);
        if (fwrite(row, 1, ROW_BYTES, f) != ROW_BYTES) fail("cannot write a row");
    }
    if (fclose(f) != 0) fail("cannot flush the temporary table");
}

/* xorshift so the id sequence is reproducible across runs and platforms. */
static uint64_t next_rand(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* `exact_reads` asserts that nothing was evicted: one read per distinct id. */
static void run_case(const char *name, const char *cache_mb,
                     uint64_t distinct, uint32_t n, int exact_reads) {
    setenv("DS4_QWEN38_PLE_CACHE_MB", cache_mb, 1);

    uint64_t *ids = malloc((size_t)n * sizeof(uint64_t));
    uint8_t *got = malloc((size_t)n * ROW_BYTES);
    uint8_t want[ROW_BYTES];
    if (!ids || !got) fail("out of memory");

    uint64_t state = 0x2545F4914F6CDD1Dull;
    for (uint32_t i = 0; i < n; i++) ids[i] = next_rand(&state) % distinct;

    uint64_t stats[3] = {0};
    if (ds4_test_qwen38_ple_store(g_path, BASE_OFFSET, TABLE_ROWS, ids, n,
                                  got, stats) != 0) {
        fail("the store refused a valid id");
    }

    for (uint32_t i = 0; i < n; i++) {
        expected_row(ids[i], want);
        if (memcmp(got + (size_t)i * ROW_BYTES, want, ROW_BYTES) != 0) {
            fprintf(stderr, "%s: row %llu came back wrong at lookup %u\n",
                    name, (unsigned long long)ids[i], i);
            fail("row mismatch");
        }
    }

    /* A random walk does not necessarily draw every id, so the cold-row floor
     * is the number of ids actually drawn, not the size of the range. */
    uint8_t *seen = calloc((size_t)distinct, 1);
    if (!seen) fail("out of memory");
    uint64_t drawn = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!seen[ids[i]]) { seen[ids[i]] = 1; drawn++; }
    }
    free(seen);

    /* Every lookup either hits or costs exactly one read. */
    if (stats[0] != n) fail("lookup count is off");
    if (stats[1] + stats[2] != stats[0]) fail("hits + reads != lookups");
    if (stats[2] < drawn) fail("fewer reads than cold rows");
    if (exact_reads && stats[2] != drawn) fail("a cached row was evicted early");

    printf("  %-22s %u lookups, %llu hits, %llu reads (%llu cold)\n", name, n,
           (unsigned long long)stats[1], (unsigned long long)stats[2],
           (unsigned long long)drawn);
    free(ids);
    free(got);
}


/* The gather is what the engine calls: hashing walks a rolling context on the
 * calling thread while the rows that missed are read and dequantized by the
 * pool. Two things have to hold -- every float is the one its row id dictates,
 * and the parallel result is bit for bit the serial one. */
#define GATHER_TOKENS 600
#define HEADS         16
#define ROW_LEN       160
#define HEAD_VOCAB    4096   /* 16 * 4096 = 65536 = TABLE_ROWS */

static void run_gather_case(const char *name, uint32_t alphabet) {
    static const uint64_t multipliers[3] = {
        23703573157769ull, 20109073645365ull, 8052911324071ull,
    };
    uint64_t offsets[HEADS], vocab[HEADS];
    for (uint32_t h = 0; h < HEADS; h++) {
        offsets[h] = (uint64_t)h * HEAD_VOCAB;
        vocab[h] = HEAD_VOCAB;
    }
    ds4_qwen38_ple_test_config(multipliers, offsets, vocab, 1);

    int *tokens = malloc(GATHER_TOKENS * sizeof(int));
    float *par = malloc((size_t)GATHER_TOKENS * HEADS * ROW_LEN * sizeof(float));
    float *ser = malloc((size_t)GATHER_TOKENS * HEADS * ROW_LEN * sizeof(float));
    if (!tokens || !par || !ser) fail("out of memory");

    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0; i < GATHER_TOKENS; i++) {
        tokens[i] = (int)(next_rand(&state) % alphabet);
    }

    uint64_t par_stats[3] = {0}, ser_stats[3] = {0};
    setenv("DS4_QWEN38_PLE_THREADS", "8", 1);
    if (ds4_test_qwen38_ple_gather(g_path, BASE_OFFSET, TABLE_ROWS, tokens,
                                   GATHER_TOKENS, par, par_stats) != 0) {
        fail("the parallel gather failed");
    }
    setenv("DS4_QWEN38_PLE_THREADS", "1", 1);
    if (ds4_test_qwen38_ple_gather(g_path, BASE_OFFSET, TABLE_ROWS, tokens,
                                   GATHER_TOKENS, ser, ser_stats) != 0) {
        fail("the serial gather failed");
    }

    if (memcmp(par, ser,
               (size_t)GATHER_TOKENS * HEADS * ROW_LEN * sizeof(float)) != 0) {
        fail("the parallel gather does not match the serial one");
    }
    for (uint32_t i = 0; i < 3; i++) {
        if (par_stats[i] != ser_stats[i]) fail("gather accounting diverged");
    }

    /* Independently rebuild the ids and check every float. */
    ds4_qwen38_ple_ctx ctx;
    ds4_qwen38_ple_ctx_reset(&ctx);
    for (uint32_t t = 0; t < GATHER_TOKENS; t++) {
        uint64_t ids[HEADS];
        ds4_qwen38_ple_hash(&ctx, tokens[t], ids);
        ds4_qwen38_ple_ctx_push(&ctx, tokens[t]);
        for (uint32_t h = 0; h < HEADS; h++) {
            const float *row = par + ((size_t)t * HEADS + h) * ROW_LEN;
            for (uint32_t v = 0; v < ROW_LEN; v++) {
                if (row[v] != expected_value(ids[h], v)) {
                    fprintf(stderr, "token %u head %u value %u: got %g, "
                            "expected %g (row %llu)\n", t, h, v, (double)row[v],
                            (double)expected_value(ids[h], v),
                            (unsigned long long)ids[h]);
                    fail("gather value mismatch");
                }
            }
        }
    }

    if (par_stats[0] != (uint64_t)GATHER_TOKENS * HEADS) {
        fail("gather lookup count is off");
    }
    if (par_stats[1] + par_stats[2] != par_stats[0]) {
        fail("gather hits + reads != lookups");
    }

    printf("  %-22s %llu lookups, %llu hits, %llu reads, parallel == serial\n",
           name, (unsigned long long)par_stats[0],
           (unsigned long long)par_stats[1], (unsigned long long)par_stats[2]);
    free(tokens);
    free(par);
    free(ser);
    unsetenv("DS4_QWEN38_PLE_THREADS");
}


/* A read that fails inside a batch has to fail the whole gather rather than
 * leave the stage holding whatever the slab contained. The hash is configured
 * so that only the top of the last head falls past the end of the table, which
 * keeps the failure rare and the expected noise short. */
static void run_gather_failure_case(void) {
    static const uint64_t multipliers[3] = {
        23703573157769ull, 20109073645365ull, 8052911324071ull,
    };
    uint64_t offsets[HEADS], vocab[HEADS];
    for (uint32_t h = 0; h < HEADS; h++) {
        offsets[h] = (uint64_t)h * 4100u;
        vocab[h] = 4100;   /* 16 * 4100 = 65600 > TABLE_ROWS */
    }
    ds4_qwen38_ple_test_config(multipliers, offsets, vocab, 1);

    int tokens[GATHER_TOKENS];
    float *out = malloc((size_t)GATHER_TOKENS * HEADS * ROW_LEN * sizeof(float));
    if (!out) fail("out of memory");
    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0; i < GATHER_TOKENS; i++) {
        tokens[i] = (int)(next_rand(&state) % 8u);
    }

    uint64_t stats[3] = {0};
    fprintf(stderr, "test_qwen38_ple_store: short reads are expected below\n");
    if (ds4_test_qwen38_ple_gather(g_path, BASE_OFFSET, 16 * 4100, tokens,
                                   GATHER_TOKENS, out, stats) == 0) {
        fail("a gather containing an unreadable row reported success");
    }
    printf("  %-22s failed the batch\n", "gather, bad row");
    free(out);
}

int main(void) {
    write_table();

    /* Roomy cache: 4096 distinct rows land in distinct sets, so every row is
     * read exactly once and everything after is a hit. */
    run_case("roomy cache", "8", 4096, 20000, 1);

    /* No cache: every lookup goes to the device, contents unchanged. */
    setenv("DS4_QWEN38_PLE_CACHE_MB", "0", 1);
    {
        uint64_t ids[64];
        uint8_t got[64 * ROW_BYTES], want[ROW_BYTES];
        uint64_t stats[3] = {0};
        for (uint32_t i = 0; i < 64; i++) ids[i] = (i * 7u) % TABLE_ROWS;
        if (ds4_test_qwen38_ple_store(g_path, BASE_OFFSET, TABLE_ROWS, ids, 64,
                                      got, stats) != 0) {
            fail("the store refused a valid id with the cache off");
        }
        for (uint32_t i = 0; i < 64; i++) {
            expected_row(ids[i], want);
            if (memcmp(got + (size_t)i * ROW_BYTES, want, ROW_BYTES) != 0) {
                fail("row mismatch with the cache off");
            }
        }
        if (stats[1] != 0 || stats[2] != 64) fail("cache-off accounting is off");
        printf("  %-22s 64 lookups, 0 hits, 64 reads\n", "cache off");
    }

    /* Smallest cache against the whole table: sustained eviction, and rows
     * must still come back byte for byte. */
    run_case("eviction pressure", "1", TABLE_ROWS, 40000, 0);

    /* An id past the end is refused rather than read out of bounds. */
    {
        const uint64_t bad = TABLE_ROWS;
        uint64_t stats[3] = {0};
        fprintf(stderr, "test_qwen38_ple_store: one refusal is expected below\n");
        if (ds4_test_qwen38_ple_store(g_path, BASE_OFFSET, TABLE_ROWS, &bad, 1,
                                      NULL, stats) == 0) {
            fail("an out-of-range row id was served");
        }
        printf("  %-22s refused\n", "out-of-range id");
    }

    /* Wide alphabet: repeats are rare, nearly every row is its own read. */
    run_gather_case("gather", 5000);
    /* Narrow alphabet: the same bigrams and trigrams recur constantly, so a
     * batch is full of items wanting a row another item already asked for.
     * That is the deduplication path, and its results must not differ. */
    run_gather_case("gather, duplicates", 8);
    run_gather_failure_case();

    /* The last row of the table: its block-aligned span runs past the end of
     * the file, so the widened read comes back short and must still be
     * accepted, because the row itself landed. */
    {
        const uint64_t ids[1] = {TABLE_ROWS - 1};
        uint8_t got[ROW_BYTES], want[ROW_BYTES];
        uint64_t stats[3] = {0};
        if (ds4_test_qwen38_ple_store(g_path, BASE_OFFSET, TABLE_ROWS, ids, 1,
                                      got, stats) != 0) {
            fail("the last row of the table was refused");
        }
        expected_row(TABLE_ROWS - 1, want);
        if (memcmp(got, want, ROW_BYTES) != 0) fail("last row mismatch");
        printf("  %-22s served\n", "last row, short read");
    }

    /* A row the table claims but the file does not hold must fail the read
     * rather than serve whatever the buffer happened to contain. */
    {
        const uint64_t ids[3] = {7, TABLE_ROWS + 10, 9};
        uint8_t got[3 * ROW_BYTES];
        uint64_t stats[3] = {0};
        fprintf(stderr, "test_qwen38_ple_store: one short read is expected below\n");
        if (ds4_test_qwen38_ple_store(g_path, BASE_OFFSET, TABLE_ROWS + 64, ids,
                                      3, got, stats) == 0) {
            fail("a row past the end of the file was served");
        }
        printf("  %-22s refused\n", "row past end of file");
    }

    unlink(g_path);
    printf("test_qwen38_ple_store: ok\n");
    return 0;
}
