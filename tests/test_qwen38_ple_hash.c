/* Qwen 3.8 PLE hash vs vectors transcribed from the official checkpoint's
 * own n-gram hashing, using the real checkpoint metadata and including
 * EOS-segment resets. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ds4.h"

static const uint64_t MULTIPLIERS[3] = {
    23703573157769ull, 20109073645365ull, 8052911324071ull,
};
static const uint64_t OFFSETS[16] = {
    0ull, 20000003ull, 40000026ull, 60000059ull, 80000106ull, 100000165ull,
    120000228ull, 140000297ull, 160000374ull, 180000455ull, 200000548ull,
    220000655ull, 240000802ull, 260000955ull, 280001114ull, 300001275ull,
};
static const uint64_t VOCAB[16] = {
    20000003ull, 20000023ull, 20000033ull, 20000047ull, 20000059ull,
    20000063ull, 20000069ull, 20000077ull, 20000081ull, 20000093ull,
    20000107ull, 20000147ull, 20000153ull, 20000159ull, 20000161ull,
    20000171ull,
};

static void check(const char *what, uint64_t actual, uint64_t expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: got %llu, expected %llu\n", what,
                (unsigned long long)actual, (unsigned long long)expected);
        exit(1);
    }
}

static void run_case(const char *name, const int *history, int n,
                     uint64_t id0, uint64_t id7, uint64_t id8, uint64_t id15) {
    ds4_qwen38_ple_ctx ctx;
    ds4_qwen38_ple_ctx_reset(&ctx);
    uint64_t rows[16] = {0};
    for (int i = 0; i < n; i++) {
        ds4_qwen38_ple_hash(&ctx, history[i], rows);
        ds4_qwen38_ple_ctx_push(&ctx, history[i]);
    }
    check(name, rows[0], id0);
    check(name, rows[7], id7);
    check(name, rows[8], id8);
    check(name, rows[15], id15);
}

int main(void) {
    ds4_qwen38_ple_test_config(MULTIPLIERS, OFFSETS, VOCAB, 248044u);

    const int c1[] = {9707};
    run_case("single token", c1, 1,
             16410909ull, 152932897ull, 169641436ull, 300984276ull);
    const int c2[] = {9707, 11};
    run_case("two tokens", c2, 2,
             18158303ull, 159566246ull, 175132467ull, 307699687ull);
    const int c3[] = {9707, 11, 220};
    run_case("three tokens", c3, 3,
             15029315ull, 143479041ull, 162897307ull, 308764256ull);
    const int c4[] = {9707, 248044, 220};
    run_case("eos reset", c4, 3,
             11925343ull, 156015647ull, 167243700ull, 303591383ull);
    const int c5[] = {9707, 11, 248044};
    run_case("current is eos", c5, 3,
             4679147ull, 141408284ull, 179487506ull, 304757900ull);

    puts("Qwen3.8 PLE hash tests: PASS");
    return 0;
}
