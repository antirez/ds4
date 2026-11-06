/* Lockstep TP drafter: DSpark Markov argmax over a vocabulary range.
 *
 * Each TP rank scores rows [row0, row0 + rows) and the ranks keep the larger
 * of their two keys.  For every case the combined key must equal the
 * full-vocabulary kernel's key, and each range key must follow the kernel's
 * rule restricted to its rows: start at (-inf, row0), replace only when
 * strictly better (larger, or equal with a lower id), NaN never wins.
 *
 * Splits: the TP halves (V/2), single-row edges (1, V-1) and 256 (inside a
 * CTA's span), over synthetic file-backed Q8_0 weights with exact scores.
 * Cases: the public test's exact ties and tail winners, ties straddling the
 * split, all -inf, all NaN, NaN at the would-be winner, +inf in both halves. */
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

static uint64_t key_of(float v, uint32_t id) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    const uint32_t k = bits & 0x80000000u ? ~bits : bits | 0x80000000u;
    return ((uint64_t)k << 32) | (uint32_t)~id;
}

/* The kernel's rule over [lo, hi). */
static uint64_t rule_key(const float *score, uint32_t lo, uint32_t hi) {
    float best = -INFINITY;
    uint32_t id = lo;
    for (uint32_t i = lo; i < hi; i++) {
        if (score[i] > best) { best = score[i]; id = i; }
    }
    return key_of(best, id);
}

static unsigned g_checks;

static void run_case(uint32_t rank, uint32_t vocab) {
    const uint64_t row = rank / 32 * 34, matrix = (uint64_t)vocab * row;
    const uint64_t w1 = 64, w2 = w1 + matrix, size = w2 + matrix, offset = 0;
    const uint64_t logbytes = (uint64_t)vocab * sizeof(float);
    unsigned char *model = calloc(1, (size_t)size);
    float *logits = malloc((size_t)logbytes), *score = malloc((size_t)logbytes);
    double *dots = malloc((size_t)vocab * sizeof(double));
    assert(model && logits && score && dots);
    fill_q8(model + w1, matrix / 34, 123);
    fill_q8(model + w2, matrix / 34, 456);
    char path[] = "/tmp/ds4-dspark-range-XXXXXX";
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
    assert(out);

    const uint32_t half = vocab / 2;
    const uint32_t splits[] = {half, 1, vocab - 1, 256 < vocab ? 256 : half};
    const uint32_t prev = vocab / 2;
    for (uint32_t i = 0; i < vocab; i++)
        dots[i] = dot_q8(model + w1 + prev * row, model + w2 + i * row, rank);
    for (unsigned mode = 0; mode < 9; mode++) {
        for (uint32_t i = 0; i < vocab; i++) {
            const double dot = dots[i];
            double target = -16;
            if (mode == 1) target = i == half - 1 || i == half ? 16 : -16;   /* tie across V/2 */
            if (mode == 2) target = i == 255 || i == 256 ? 16 : -16;         /* tie across 256 */
            if (mode == 3) target = i == vocab - 1 ? 16 : -16;               /* tail winner */
            if (mode == 4) target = i == 0 || i == vocab - 1 ? 8 : -8;       /* edge tie */
            logits[i] = (float)(target - dot);
            if (mode == 0 || mode >= 7) logits[i] = ((int)(i * 13u % 33u) - 16) / 4.0f;
            if (mode == 5) logits[i] = -INFINITY;
            if (mode == 6) logits[i] = NAN;
            if (mode == 8) logits[i] = i == 3 || i == vocab - 2 ? INFINITY : logits[i];
            const double s = (double)logits[i] + dot;
            score[i] = isfinite(logits[i]) ? (float)s : logits[i];
            if (isfinite(logits[i])) assert((double)score[i] == s);
        }
        if (mode == 7) {
            /* NaN at the would-be winner: the runner-up must win. */
            const uint64_t k = rule_key(score, 0, vocab);
            const uint32_t best = ~(uint32_t)k;
            logits[best] = NAN;
            score[best] = NAN;
        }
        const uint64_t want = rule_key(score, 0, vocab);
        if (mode == 1) assert(~(uint32_t)want == half - 1);
        if (mode == 5 || mode == 6) assert(want == key_of(-INFINITY, 0));
        if (mode == 8) assert(~(uint32_t)want == 3);
        assert(ds4_gpu_tensor_write(input, 0, logits, logbytes));
        uint64_t actual[3];
        assert(ds4_gpu_tensor_write(slab, 0, guard, sizeof(guard)));
        assert(ds4_gpu_dspark_markov_argmax_tensor(out, input, model, size, w1, w2,
                                                   prev, vocab, rank));
        assert(ds4_gpu_tensor_read(slab, 0, actual, sizeof(actual)));
        assert(actual[0] == guard[0] && actual[2] == guard[2] && actual[1] == want);
        g_checks++;
        for (unsigned s = 0; s < sizeof(splits) / sizeof(*splits); s++) {
            const uint32_t cut = splits[s];
            uint64_t keys[2];
            for (unsigned r = 0; r < 2; r++) {
                const uint32_t row0 = r ? cut : 0, rows = r ? vocab - cut : cut;
                ds4_gpu_tensor *part = ds4_gpu_tensor_view(input, (uint64_t)row0 * 4u,
                                                           (uint64_t)rows * 4u);
                assert(part);
                assert(ds4_gpu_tensor_write(slab, 0, guard, sizeof(guard)));
                assert(ds4_gpu_dspark_markov_argmax_range_tensor(out, part, model, size,
                                                                 w1, w2, prev, vocab,
                                                                 row0, rows, rank));
                assert(ds4_gpu_tensor_read(slab, 0, actual, sizeof(actual)));
                assert(actual[0] == guard[0] && actual[2] == guard[2]);
                if (actual[1] != rule_key(score, row0, row0 + rows))
                    fprintf(stderr, "rank=%u vocab=%u mode=%u cut=%u r=%u: got %u want %u\n",
                            rank, vocab, mode, cut, r, ~(uint32_t)actual[1],
                            ~(uint32_t)rule_key(score, row0, row0 + rows));
                assert(actual[1] == rule_key(score, row0, row0 + rows));
                keys[r] = actual[1];
                ds4_gpu_tensor_free(part);
                g_checks++;
            }
            assert((keys[0] > keys[1] ? keys[0] : keys[1]) == want);
            g_checks++;
        }
    }
    /* Malformed ranges write nothing. */
    assert(ds4_gpu_tensor_write(slab, 0, guard, sizeof(guard)));
    ds4_gpu_tensor *short_part = ds4_gpu_tensor_view(input, 0, (uint64_t)half * 4u - 1u);
    assert(short_part);
#define RANGE(l, r0, n) ds4_gpu_dspark_markov_argmax_range_tensor(out, l, model, size, w1, w2, \
                                                                  prev, vocab, r0, n, rank)
    assert(!RANGE(input, 0, 0));
    assert(!RANGE(input, vocab, 1));
    assert(!RANGE(input, vocab + 1, 1));
    assert(!RANGE(input, half, vocab - half + 1));
    assert(!RANGE(input, 1, UINT32_MAX));
    assert(!RANGE(short_part, 0, half));
#undef RANGE
    uint64_t unchanged[3];
    assert(ds4_gpu_tensor_read(slab, 0, unchanged, sizeof(unchanged)));
    assert(memcmp(guard, unchanged, sizeof(guard)) == 0);
    ds4_gpu_tensor_free(short_part);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(input); ds4_gpu_tensor_free(slab);
    ds4_gpu_cleanup();
    assert(close(fd) == 0);
    free(dots); free(score); free(logits); free(model);
    printf("CUDA DSpark range argmax rank=%u vocab=%u: PASS\n", rank, vocab);
}

int main(void) {
    for (uint32_t rank = 32; rank <= 256; rank += 32) {
        run_case(rank, 257);
        run_case(rank, 32771);
    }
    run_case(256, 129280);
    run_case(256, 129283);
    printf("CUDA DSpark range argmax: %u checks PASS\n", g_checks);
    return 0;
}
