/* Bounded Metal conversion oracle and A/B benchmark; no model required.
 * make test-deepseek41-bf16-rhs
 * ./tests/test_deepseek41_bf16_rhs --bench */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

enum { OLD_LOW, NEW_LOW, OLD_RHS, NEW_RHS, NT, PAD = 16 };
typedef struct {
    ds4_gpu_tensor *storage[NT], *t[NT];
    uint64_t bytes[NT], count;
    uint32_t width, rows;
} fixture;

static uint32_t bf16_bits(uint32_t bits) {
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return bits & 0xffff0000u;
}

static uint32_t round_shift(uint32_t bits, unsigned shift) {
    const uint32_t half = 1u << (shift - 1u), tail = bits & ((1u << shift) - 1u);
    const uint32_t value = bits >> shift;
    return value + (tail > half || (tail == half && (value & 1u)));
}

/* Independent IEEE conversion for finite values and infinities. NaN payload
 * encoding is device-specific and is compared against the old GPU path. */
static uint16_t half_bits(uint32_t bits) {
    const uint16_t sign = (bits >> 16u) & 0x8000u;
    const uint32_t fraction = bits & 0x7fffffu, exponent = (bits >> 23u) & 255u;
    if (exponent == 255u) return sign | 0x7c00u | (fraction ? 0x200u : 0u);
    const int exp = (int)exponent - 112;
    if (exp >= 31) return sign | 0x7c00u;
    if (exp < -10) return sign;
    if (exp <= 0) return sign | round_shift(fraction | 0x800000u, 14u - exp);
    return sign | (((uint32_t)exp << 10u) + round_shift(fraction, 13u));
}

static uint32_t input_bits(uint64_t i) {
    static const uint32_t edges[] = {
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
        0x3f807fff, 0x3f808000, 0x3f808001, 0xbf808001,
        0x3f817fff, 0x3f818000, 0x3f818001, 0xbf818000,
        0x32ffffff, 0x33000000, 0x33000001, 0x33800000,
        0x387fffff, 0x38800000, 0x477f7fff, 0x477f8000,
        0x477fffff, 0x47800000, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
        0x7f812345, 0xff812345, 0x7f800001, 0xff800001,
    };
    if (i % 3u != 2u) return edges[i % (sizeof(edges) / sizeof(*edges))];
    /* Reproducible sign, exponent and significand coverage. */
    uint32_t x = (uint32_t)i * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

static int initialize(fixture *f, uint32_t width, uint32_t rows) {
    f->width = width; f->rows = rows; f->count = (uint64_t)width * rows;
    for (unsigned i = 0; i < NT; i++) {
        f->bytes[i] = f->count * (i < OLD_RHS ? 4u : 2u);
        f->storage[i] = ds4_gpu_tensor_alloc(f->bytes[i] + 2u * PAD);
        CHECK(f->storage[i]);
        memset(ds4_gpu_tensor_contents(f->storage[i]), 0xa5, f->bytes[i] + 2u * PAD);
        f->t[i] = ds4_gpu_tensor_view(f->storage[i], PAD, f->bytes[i]);
        CHECK(f->t[i]);
    }
    uint32_t *old = ds4_gpu_tensor_contents(f->t[OLD_LOW]);
    uint32_t *new = ds4_gpu_tensor_contents(f->t[NEW_LOW]);
    for (uint64_t i = 0; i < f->count; i++) old[i] = new[i] = input_bits(i);
    return 1;
}

static void destroy(fixture *f) {
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    for (unsigned i = 0; i < NT; i++) {
        ds4_gpu_tensor_free(f->t[i]);
        ds4_gpu_tensor_free(f->storage[i]);
    }
}

static int sequence(fixture *f, int fused) {
    if (fused) return ds4_gpu_dsv41_bf16_f16_rhs(f->t[NEW_LOW], f->t[NEW_RHS],
                                               f->width, f->rows);
    return ds4_gpu_dsv41_quantize(f->t[OLD_LOW], f->width, f->rows, DS4_V41_BF16) &&
        ds4_gpu_tensor_copy_f32_to_f16(f->t[OLD_RHS], 0, f->t[OLD_LOW], 0, f->count);
}

static int compare(fixture *f) {
    const uint32_t *low = ds4_gpu_tensor_contents(f->t[NEW_LOW]);
    const uint16_t *rhs = ds4_gpu_tensor_contents(f->t[NEW_RHS]);
    CHECK(!memcmp(low, ds4_gpu_tensor_contents(f->t[OLD_LOW]), f->bytes[NEW_LOW]));
    CHECK(!memcmp(rhs, ds4_gpu_tensor_contents(f->t[OLD_RHS]), f->bytes[NEW_RHS]));
    for (uint64_t i = 0; i < f->count; i++) {
        const uint32_t expected = bf16_bits(input_bits(i));
        CHECK(low[i] == expected);
        if ((expected & 0x7fffffffu) > 0x7f800000u)
            CHECK((rhs[i] & 0x7c00u) == 0x7c00u && (rhs[i] & 0x3ffu));
        else CHECK(rhs[i] == half_bits(expected));
    }
    for (unsigned t = 0; t < NT; t++) {
        const unsigned char *p = ds4_gpu_tensor_contents(f->storage[t]);
        for (unsigned i = 0; i < PAD; i++)
            CHECK(p[i] == 0xa5 && p[PAD + f->bytes[t] + i] == 0xa5);
    }
    return 1;
}

static int parity(uint32_t width, uint32_t rows) {
    fixture f = {0};
    int ok = initialize(&f, width, rows) && ds4_gpu_begin_commands() &&
        sequence(&f, 0) && sequence(&f, 1) && ds4_gpu_end_commands() && compare(&f);
    destroy(&f);
    return ok;
}

static int rejected(void) {
    fixture f = {0};
    CHECK(initialize(&f, 67, 1));
    unsigned char before[NT][300];
    for (unsigned i = 0; i < NT; i++)
        memcpy(before[i], ds4_gpu_tensor_contents(f.storage[i]), f.bytes[i] + 2u * PAD);
    CHECK(!ds4_gpu_dsv41_bf16_f16_rhs(NULL, f.t[NEW_RHS], 67, 1));
    CHECK(!ds4_gpu_dsv41_bf16_f16_rhs(f.t[NEW_LOW], NULL, 67, 1));
    CHECK(!ds4_gpu_dsv41_bf16_f16_rhs(f.t[NEW_LOW], f.t[NEW_RHS], 0, 1));
    CHECK(!ds4_gpu_dsv41_bf16_f16_rhs(f.t[NEW_LOW], f.t[NEW_RHS], 67, 0));
    CHECK(!ds4_gpu_dsv41_bf16_f16_rhs(f.t[NEW_LOW], f.t[NEW_RHS], UINT32_MAX, 2));
    CHECK(!ds4_gpu_dsv41_bf16_f16_rhs(f.t[NEW_LOW], f.t[NEW_RHS], 68, 1));
    CHECK(!ds4_gpu_dsv41_bf16_f16_rhs(f.t[NEW_LOW], f.t[NEW_LOW], 67, 1));
    const uint64_t offsets[] = {PAD, PAD + 2u, PAD + 4u};
    for (unsigned i = 0; i < 3; i++) {
        ds4_gpu_tensor *bad = ds4_gpu_tensor_view(f.storage[NEW_RHS], offsets[i],
                                                f.bytes[NEW_RHS] - (i == 0));
        CHECK(bad && !ds4_gpu_dsv41_bf16_f16_rhs(f.t[NEW_LOW], bad, 67, 1));
        ds4_gpu_tensor_free(bad);
    }
    ds4_gpu_tensor *bad = ds4_gpu_tensor_view(f.storage[NEW_LOW], PAD + 4u, f.bytes[NEW_LOW]);
    CHECK(bad && !ds4_gpu_dsv41_bf16_f16_rhs(bad, f.t[NEW_RHS], 67, 1));
    ds4_gpu_tensor_free(bad);
    for (unsigned i = 0; i < NT; i++)
        CHECK(!memcmp(before[i], ds4_gpu_tensor_contents(f.storage[i]), f.bytes[i] + 2u * PAD));
    destroy(&f);
    return 1;
}

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int order_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int benchmark(uint32_t rows) {
    enum { SAMPLES = 11, ITER = 32 };
    fixture f = {0};
    CHECK(initialize(&f, 8192u, rows));
    CHECK(ds4_gpu_begin_commands() && sequence(&f, 0) && sequence(&f, 1) &&
          ds4_gpu_end_commands() && compare(&f));
    /* Both timed arms reuse the same physical buffers. Conversion is
     * idempotent after the parity run, so no CPU reset or cache-color bias
     * enters the timing. Keep the independent new buffers as the oracle. */
    fixture timed = f;
    timed.t[NEW_LOW] = f.t[OLD_LOW];
    timed.t[NEW_RHS] = f.t[OLD_RHS];
    double sample[2][SAMPLES];
    for (unsigned pass = 0; pass <= SAMPLES; pass++) {
        double elapsed[2] = {0, 0};
        for (unsigned turn = 0; turn < 4; turn++) {
            const unsigned fused = (turn == 1 || turn == 2) ^ (pass & 1u);
            const double start = seconds();
            CHECK(ds4_gpu_begin_commands());
            for (unsigned i = 0; i < ITER; i++) CHECK(sequence(&timed, fused));
            CHECK(ds4_gpu_end_commands());
            elapsed[fused] += seconds() - start;
        }
        if (pass) for (unsigned fused = 0; fused < 2; fused++)
            sample[fused][pass - 1] = elapsed[fused] * 1e6 / (2 * ITER);
    }
    for (unsigned i = 0; i < 2; i++) qsort(sample[i], SAMPLES, sizeof(double), order_double);
    CHECK(compare(&f));
    printf("BF16+half RHS 8192x%u resident median us: old=%.3f fused=%.3f speedup=%.3fx"
           " ranges=[%.3f,%.3f]/[%.3f,%.3f]\n",
        rows, sample[0][SAMPLES / 2], sample[1][SAMPLES / 2],
        sample[0][SAMPLES / 2] / sample[1][SAMPLES / 2],
        sample[0][0], sample[0][SAMPLES - 1], sample[1][0], sample[1][SAMPLES - 1]);
    destroy(&f);
    return 1;
}

int main(int argc, char **argv) {
    const int bench = argc == 2 && !strcmp(argv[1], "--bench");
    if (argc > 1 && !bench) return 2;
    /* Prove these inputs distinguish the required two rounding steps. */
    if (half_bits(bf16_bits(0x3f808001u)) == half_bits(0x3f808001u) ||
        half_bits(bf16_bits(0x3f807fffu)) == half_bits(0x3f807fffu)) return 1;
    if (!ds4_gpu_init()) return 1;
    const uint32_t tails[] = {1, 2, 3, 4, 5, 7, 67, 1023, 1024, 1025, 4099};
    for (unsigned i = 0; i < sizeof(tails) / sizeof(*tails); i++)
        if (!parity(tails[i], 1)) return 1;
    if (!parity(8192, 1) || !parity(8192, 257) || !rejected()) return 1;
    puts("V4.1 BF16 low/F16 RHS: bitwise legacy parity, CPU rounding and guards PASS");
    if (bench) {
        /* Fixed shuffled order avoids coupling shape to increasing warmup. */
        const uint32_t rows[] = {512, 128, 2048, 256, 1024};
        for (unsigned i = 0; i < sizeof(rows) / sizeof(*rows); i++)
            if (!benchmark(rows[i])) return 1;
    }
    ds4_gpu_cleanup();
    return 0;
}
