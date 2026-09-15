#define _POSIX_C_SOURCE 200809L
#include "ds4_engram.h"

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { TOKENS = 128, WIDTH = DS4_ENGRAM_COLS * DS4_ENGRAM_DIM };

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static uint64_t number(const char *text) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || !text[0] || text[0] == '-' || !end || *end) {
        fprintf(stderr, "invalid number: %s\n", text);
        exit(2);
    }
    return value;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s MODEL OFFSET0 ROWS0 OFFSET1 ROWS1\n", argv[0]);
        return 2;
    }
    const uint64_t offset[] = {number(argv[2]), number(argv[4])};
    const uint64_t rows[] = {number(argv[3]), number(argv[5])};
    if (!rows[0] || !rows[1] || rows[0] > UINT32_MAX || rows[1] > UINT32_MAX) return 2;
    ds4_engram_table tables[2];
    for (int i = 0; i < 2; i++) {
        if (!ds4_engram_table_open(&tables[i], argv[1], offset[i], (uint32_t)rows[i])) {
            perror("Engram table");
            return 1;
        }
    }
    uint32_t ids[2][TOKENS][DS4_ENGRAM_COLS];
    uint64_t random = 41;
    for (int table = 0; table < 2; table++) {
        for (int token = 0; token < TOKENS; token++) {
            for (int col = 0; col < DS4_ENGRAM_COLS; col++) {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                ids[table][token][col] = random % tables[table].rows;
            }
        }
    }
    float *reference = malloc((size_t)2 * TOKENS * WIDTH * sizeof(*reference));
    float out[WIDTH];
    assert(reference);
    unsetenv("DS4_ENGRAM_PARALLEL_DECODE");
    for (int table = 0; table < 2; table++) {
        for (int token = 0; token < TOKENS; token++) {
            assert(ds4_engram_read(&tables[table], ids[table][token], DS4_ENGRAM_COLS,
                                  reference + ((size_t)table * TOKENS + token) * WIDTH));
        }
    }
    puts("pass,variant,two_tables_ms_per_token");
    for (int pass = 0; pass < 8; pass++) {
        for (int order = 0; order < 2; order++) {
            const int parallel = (pass + order) % 2;
            if (parallel) setenv("DS4_ENGRAM_PARALLEL_DECODE", "1", 1);
            else unsetenv("DS4_ENGRAM_PARALLEL_DECODE");
            double elapsed = 0;
            for (int token = 0; token < TOKENS; token++) {
                for (int table = 0; table < 2; table++) {
                    const double begin = now_sec();
                    assert(ds4_engram_read(&tables[table], ids[table][token], DS4_ENGRAM_COLS, out));
                    elapsed += now_sec() - begin;
                    assert(!memcmp(out, reference + ((size_t)table * TOKENS + token) * WIDTH,
                                   sizeof(out)));
                }
            }
            printf("%d,%s,%.6f\n", pass, parallel ? "parallel" : "serial", elapsed * 1000 / TOKENS);
            fflush(stdout);
        }
    }
    free(reference);
    for (int table = 0; table < 2; table++) ds4_engram_table_close(&tables[table]);
    return 0;
}
