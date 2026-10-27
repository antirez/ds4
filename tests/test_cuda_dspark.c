/* Synthetic, file-backed Q8_0 DSpark argmax regression tests. */
#include "ds4_gpu.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void fill_q8(unsigned char *p, uint64_t blocks, uint32_t seed) {
    for (uint64_t b = 0; b < blocks; b++, p += 34) {
        const uint16_t h = (uint16_t)((10 + b % 3) << 10);
        memcpy(p, &h, 2); /* Exact scales: 1/32, 1/16, 1/8. */
        for (unsigned k = 0; k < 32; k++) {
            seed = seed * 1664525u + 1013904223u;
            p[2 + k] = (unsigned char)((int)(seed >> 28) - 8);
        }
    }
}

static double scale(const unsigned char *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    assert(h == 0x2800 || h == 0x2c00 || h == 0x3000);
    return 1.0 / (1u << (15 - (h >> 10)));
}

static double dot_q8(const unsigned char *a, const unsigned char *b, unsigned rank) {
    double sum = 0;
    for (unsigned j = 0; j < rank / 32; j++, a += 34, b += 34) {
        const double d = scale(a) * scale(b);
        const int8_t *qa = (const int8_t *)(a + 2), *qb = (const int8_t *)(b + 2);
        for (unsigned k = 0; k < 32; k++) sum += d * qa[k] * qb[k];
    }
    return sum;
}

static void run_case(uint32_t rank, uint32_t vocab) {
    const uint64_t row = rank / 32 * 34, matrix = (uint64_t)vocab * row;
    const uint64_t w1 = 64, w2 = w1 + matrix, size = w2 + matrix, offset = 0;
    const uint64_t logbytes = (uint64_t)vocab * sizeof(float);
    unsigned char *model = calloc(1, (size_t)size);
    float *logits = malloc((size_t)logbytes);
    assert(model && logits);
    fill_q8(model + w1, matrix / 34, 123);
    fill_q8(model + w2, matrix / 34, 456);
    char path[] = "/tmp/ds4-dspark-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    assert(unlink(path) == 0);
    for (uint64_t n = 0; n < size;) {
        const ssize_t wrote = write(fd, model + n, (size_t)(size - n));
        if (wrote < 0 && errno == EINTR) continue;
        assert(wrote > 0);
        n += (uint64_t)wrote;
    }
    assert(ds4_gpu_init());
    assert(ds4_gpu_set_model_map_spans(model, size, &offset, &size, 1, 0));
    assert(ds4_gpu_set_model_fd_for_map(fd, model));
    const uint64_t guard[3] = {UINT64_C(0x13579bdf2468ace0), UINT64_MAX,
                               UINT64_C(0xfedcba9876543210)};
    ds4_gpu_tensor *slab = ds4_gpu_tensor_alloc(sizeof(guard));
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(logbytes);
    assert(slab && input);
    ds4_gpu_tensor *out = ds4_gpu_tensor_view(slab, 8, 8);
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_view(slab, 8, 7);
    ds4_gpu_tensor *short_input = ds4_gpu_tensor_view(input, 0, logbytes - 1);
    assert(out && short_out && short_input);
#define STEP(o, l, n, a, b, p, r) \
    ds4_gpu_dspark_markov_argmax_tensor(o, l, model, n, a, b, p, vocab, r)
    const uint32_t prevs[] = {0, 1, vocab - 1};
    for (unsigned mode = 0; mode < 9; mode++) {
        const uint32_t prev = mode < 3 ? prevs[mode] : vocab / 2;
        uint32_t best = 0;
        double best_score = -INFINITY;
        for (uint32_t i = 0; i < vocab; i++) {
            const double dot = dot_q8(model + w1 + prev * row, model + w2 + i * row, rank);
            logits[i] = ((int)(i * 13u % 33u) - 16) / 4.0f;
            /* Cancel the exact dot to construct exact ties and tail winners. */
            if (mode >= 3) {
                double target = -512;
                if (mode == 3) target = i == vocab - 1 ? 16 : -16;
                if (mode == 4) target = i == vocab - 1 ? -512 : -1024;
                if (mode == 5) target = i == 1 || i == 2 ? -16 : -32;
                if (mode == 6) target = i == 255 || i == 256 ? 16 : -16;
                if (mode == 7) target = i == 2 || i == vocab - 1 ? -512 : -1024;
                logits[i] = (float)(target - dot);
                assert((double)logits[i] + dot == target);
                if (mode == 4 || mode == 7 || mode == 8) assert(logits[i] < 0);
            }
            const double score = (double)logits[i] + dot;
            assert((double)(float)score == score);
            if (score > best_score) { best_score = score; best = i; }
        }
        if (mode == 3 || mode == 4) assert(best == vocab - 1);
        if (mode == 5) assert(best == 1);
        if (mode == 6) assert(best == 255);
        if (mode == 7) assert(best == 2);
        if (mode == 8) assert(best == 0);
        const float score = (float)best_score;
        uint32_t bits;
        memcpy(&bits, &score, sizeof(bits));
        const uint32_t key = bits & 0x80000000u ? ~bits : bits | 0x80000000u;
        const uint64_t expected = ((uint64_t)key << 32) | (uint32_t)~best;
        assert(ds4_gpu_tensor_write(input, 0, logits, logbytes));
        for (unsigned repeat = 0; repeat < (mode >= 5 ? 4u : 1u); repeat++) {
            uint64_t actual[3];
            assert(ds4_gpu_tensor_write(slab, 0, guard, sizeof(guard)));
            assert(STEP(out, input, size, w1, w2, prev, rank));
            assert(ds4_gpu_tensor_read(slab, 0, actual, sizeof(actual)));
            assert(actual[0] == guard[0] && actual[2] == guard[2]);
            if (actual[1] != expected)
                fprintf(stderr, "rank=%u vocab=%u mode=%u: got=%u expected=%u\n",
                        rank, vocab, mode, ~(uint32_t)actual[1], best);
            assert(actual[1] == expected);
        }
    }
    assert(ds4_gpu_tensor_write(slab, 0, guard, sizeof(guard)));
    const uint32_t bad_ranks[] = {0, 1, 31, 33, 255, 257, 288};
    for (unsigned i = 0; i < sizeof(bad_ranks) / sizeof(*bad_ranks); i++)
        assert(!STEP(out, input, size, w1, w2, 0, bad_ranks[i]));
    assert(!STEP(NULL, input, size, w1, w2, 0, rank));
    assert(!STEP(short_out, input, size, w1, w2, 0, rank));
    assert(!STEP(out, NULL, size, w1, w2, 0, rank));
    assert(!STEP(out, short_input, size, w1, w2, 0, rank));
    assert(!STEP(out, input, size, size + 1, w2, 0, rank));
    assert(!STEP(out, input, size, UINT64_MAX, w2, 0, rank));
    assert(!STEP(out, input, size, size - row + 1, w2, 0, rank));
    assert(!STEP(out, input, size, size - row, w2, 1, rank));
    assert(!STEP(out, input, size, w1, size + 1, 0, rank));
    assert(!STEP(out, input, size, w1, UINT64_MAX, 0, rank));
    assert(!STEP(out, input, size, w1, w2 + 1, 0, rank));
    assert(!STEP(out, input, size - 1, w1, w2, 0, rank));
    /* These W1 addresses still fit in the file (inside W2). */
    assert(!STEP(out, input, size, w1, w2, vocab, rank));
    assert(!STEP(out, input, size, w1, w2, vocab + 1, rank));
    uint64_t unchanged[3];
    assert(ds4_gpu_tensor_read(slab, 0, unchanged, sizeof(unchanged)));
    assert(memcmp(guard, unchanged, sizeof(guard)) == 0);
#undef STEP
    ds4_gpu_tensor_free(short_input); ds4_gpu_tensor_free(short_out);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(input); ds4_gpu_tensor_free(slab);
    ds4_gpu_cleanup();
    assert(close(fd) == 0);
    free(logits); free(model);
    printf("CUDA DSpark argmax rank=%u vocab=%u: PASS\n", rank, vocab);
}

int main(void) {
    for (uint32_t rank = 32; rank <= 256; rank += 32) {
        run_case(rank, 257);
        run_case(rank, 32771);
    }
    run_case(256, 129283);
    return 0;
}
