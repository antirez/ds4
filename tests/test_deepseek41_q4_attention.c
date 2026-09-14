/* Bounded native Metal oracle; no model download or CPU model inference.
 * make test-deepseek41-q4-attention
 * ./tests/test_deepseek41_q4_attention --bench
 * Timings describe these synthetic projection kernels, not model throughput. */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { Q8 = 8, Q4 = 12, MAX_ROWS = 513, HEADS = 32768, LOW = 8192,
       OUTPUT = 5120, GROUP = 4096, RANK = 1024, PAD = 4 };
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
typedef struct { uint32_t in, out; const char *name; } shape;
typedef struct {
    uint8_t *model;
    uint64_t size, oa[2], ob[2], dense;
    ds4_gpu_tensor *xs, *ls, *ys, *x, *low, *y, *reference, *reference_low;
} fixture;

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

static uint32_t random_state = 7919;
static uint32_t random_bits(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static float half_value(uint16_t h) {
    const int exponent = (h >> 10) & 31;
    const double mantissa = exponent ? 1024 + (h & 1023) : h & 1023;
    return (float)copysign(ldexp(mantissa, exponent ? exponent - 25 : -24),
                          h & 0x8000 ? -1.0 : 1.0);
}

static float bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

static uint64_t row_bytes(uint32_t type, uint32_t width) {
    return type == Q4 ? (uint64_t)width / 256 * sizeof(q4_block) :
                        (uint64_t)width / 32 * sizeof(q8_block);
}

/* Decode the GGUF layout independently of production helpers. Both halves
 * of the packed six-bit scales/minima and every nibble position are used. */
static double weight_value(const uint8_t *row, uint32_t type, uint32_t column) {
    if (type == Q8) {
        const q8_block *b = (const q8_block *)row + column / 32;
        return (double)half_value(b->d) * b->qs[column % 32];
    }
    const q4_block *b = (const q4_block *)row + column / 256;
    const uint32_t k = column % 256, sub = k / 32;
    const unsigned scale = sub < 4 ? b->scales[sub] & 63 :
        (b->scales[sub + 4] & 15) | ((b->scales[sub - 4] >> 6) << 4);
    const unsigned minimum = sub < 4 ? b->scales[sub + 4] & 63 :
        (b->scales[sub + 4] >> 4) | ((b->scales[sub] >> 6) << 4);
    const unsigned packed = b->qs[(k / 64) * 32 + k % 32];
    const unsigned q = (k & 32) ? packed >> 4 : packed & 15;
    return (double)half_value(b->d) * scale * q -
           (double)half_value(b->dmin) * minimum;
}

static double dot(const uint8_t *row, uint32_t type, const float *x,
                  uint32_t offset, uint32_t width, double *absolute) {
    double sum = 0, bound = 0;
    for (uint32_t k = 0; k < width; k++) {
        const double term = weight_value(row, type, offset + k) * x[k];
        sum += term;
        bound += fabs(term);
    }
    if (absolute) *absolute = bound;
    return sum;
}

static void fill_weights(uint8_t *dst, uint32_t type, uint64_t elements) {
    if (type == Q4) {
        q4_block *b = (q4_block *)dst;
        for (uint64_t i = 0; i < elements / 256; i++) {
            /* Dyadic, exactly representable FP16 weights avoid conflating
             * dequantization rounding with the BF16 activation boundary. */
            b[i].d = 0x0400;
            b[i].dmin = (i & 1) ? 0x0400 : 0x0800;
            for (unsigned j = 0; j < 12; j++) b[i].scales[j] = random_bits();
            for (unsigned j = 0; j < 128; j++) b[i].qs[j] = random_bits();
        }
    } else {
        q8_block *b = (q8_block *)dst;
        for (uint64_t i = 0; i < elements / 32; i++) {
            b[i].d = 0x0800;
            for (unsigned j = 0; j < 32; j++) b[i].qs[j] = (int)(random_bits() % 127) - 63;
        }
    }
}

static void fill_input(ds4_gpu_tensor *x, uint64_t count) {
    float *data = ds4_gpu_tensor_contents(x);
    for (uint64_t i = 0; i < count; i++)
        data[i] = ((int)(random_bits() % 15) - 7) / 4096.0f;
}

static int guards(ds4_gpu_tensor *storage, uint64_t count, int initialize) {
    uint32_t *p = ds4_gpu_tensor_contents(storage);
    CHECK(p);
    for (uint64_t i = 0; i < 2u * PAD; i++) {
        const uint64_t index = i < PAD ? i : count + i;
        if (initialize) p[index] = 0x7fc12345;
        else CHECK(p[index] == 0x7fc12345);
    }
    return 1;
}

static int initialize(fixture *f) {
    CHECK(sizeof(q4_block) == 144 && sizeof(q8_block) == 34);
    uint64_t cursor = 128;
    for (unsigned i = 0; i < 2; i++) {
        const uint32_t type = i ? Q8 : Q4;
        f->oa[i] = cursor;
        cursor += row_bytes(type, GROUP) * LOW + 128;
        f->ob[i] = cursor;
        cursor += row_bytes(type, LOW) * OUTPUT + 128;
    }
    f->dense = cursor;
    cursor += row_bytes(Q4, 1280) * HEADS + 128;
    const uint64_t page = getpagesize();
    f->size = (cursor + page - 1) / page * page;
    CHECK(!posix_memalign((void **)&f->model, page, f->size));
    memset(f->model, 0xA5, f->size);
    for (unsigned i = 0; i < 2; i++) {
        fill_weights(f->model + f->oa[i], i ? Q8 : Q4, (uint64_t)GROUP * LOW);
        fill_weights(f->model + f->ob[i], i ? Q8 : Q4, (uint64_t)LOW * OUTPUT);
    }
    fill_weights(f->model + f->dense, Q4, (uint64_t)1280 * HEADS);
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map(f->model, f->size));
    f->xs = ds4_gpu_tensor_alloc(((uint64_t)MAX_ROWS * HEADS + 2 * PAD) * 4);
    f->ls = ds4_gpu_tensor_alloc(((uint64_t)MAX_ROWS * LOW + 2 * PAD) * 4);
    f->ys = ds4_gpu_tensor_alloc(((uint64_t)MAX_ROWS * HEADS + 2 * PAD) * 4);
    CHECK(f->xs && f->ls && f->ys);
    f->x = ds4_gpu_tensor_view(f->xs, PAD * 4, (uint64_t)MAX_ROWS * HEADS * 4);
    f->low = ds4_gpu_tensor_view(f->ls, PAD * 4, (uint64_t)MAX_ROWS * LOW * 4);
    f->y = ds4_gpu_tensor_view(f->ys, PAD * 4, (uint64_t)MAX_ROWS * HEADS * 4);
    f->reference = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * HEADS * 4);
    f->reference_low = ds4_gpu_tensor_alloc((uint64_t)MAX_ROWS * LOW * 4);
    CHECK(f->x && f->low && f->y && f->reference && f->reference_low);
    CHECK(guards(f->xs, (uint64_t)MAX_ROWS * HEADS, 1));
    CHECK(guards(f->ys, (uint64_t)MAX_ROWS * HEADS, 1));
    CHECK(guards(f->ls, (uint64_t)MAX_ROWS * LOW, 1));
    return 1;
}

static void cleanup(fixture *f) {
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(f->reference_low);
    ds4_gpu_tensor_free(f->reference);
    ds4_gpu_tensor_free(f->y); ds4_gpu_tensor_free(f->low); ds4_gpu_tensor_free(f->x);
    ds4_gpu_tensor_free(f->ys); ds4_gpu_tensor_free(f->ls); ds4_gpu_tensor_free(f->xs);
    ds4_gpu_cleanup();
    free(f->model);
}

static int projection_cases(fixture *f) {
    const shape shapes[] = {{5120, 1280, "Q_A"}, {1280, HEADS, "Q_B"},
        {5120, 512, "KV"}, {GROUP, RANK, "O_A group"}, {LOW, OUTPUT, "O_B"}};
    const uint32_t counts[] = {1, 2, 8, 31, 32, 33, 128};
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(*shapes); s++) {
        const shape d = shapes[s];
        const uint64_t offset = s == 4 ? f->ob[0] : f->dense;
        fill_input(f->x, (uint64_t)MAX_ROWS * d.in);
        const float *x = ds4_gpu_tensor_contents(f->x);
        for (unsigned n = 0; n < sizeof(counts) / sizeof(*counts); n++) {
            const uint32_t rows = counts[n];
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_matmul_quant_tensor(f->y, f->model, f->size, offset,
                Q4, d.in, d.out, f->x, rows));
            CHECK(ds4_gpu_end_commands());
            const float *actual = ds4_gpu_tensor_contents(f->y);
            const uint32_t rr[] = {0, rows / 2, rows - 1};
            const uint32_t cc[] = {0, 1, 31, 32, d.out / 2, d.out - 2, d.out - 1};
            for (unsigned i = 0; i < 3; i++) for (unsigned j = 0; j < 7; j++) {
                double absolute;
                const double expected = dot(f->model + offset + cc[j] * row_bytes(Q4, d.in),
                    Q4, x + (uint64_t)rr[i] * d.in, 0, d.in, &absolute);
                const double got = actual[(uint64_t)rr[i] * d.out + cc[j]];
                /* FP32 reduction error scales with the sum of absolute terms;
                 * input and dequantized weights here are exact FP16 values. */
                CHECK(isfinite(got) && fabs(got - expected) <= 2e-5 * absolute + 1e-7);
            }
            if (rows <= 8) {
                CHECK(ds4_gpu_begin_commands());
                for (uint32_t r = 0; r < rows; r++) {
                    ds4_gpu_tensor *xr = ds4_gpu_tensor_view(f->x, (uint64_t)r * d.in * 4, d.in * 4);
                    ds4_gpu_tensor *yr = ds4_gpu_tensor_view(f->reference, (uint64_t)r * d.out * 4, d.out * 4);
                    const int ok = xr && yr && ds4_gpu_matmul_quant_tensor(yr,
                        f->model, f->size, offset, Q4, d.in, d.out, xr, 1);
                    ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(xr);
                    CHECK(ok);
                }
                CHECK(ds4_gpu_end_commands());
                CHECK(!memcmp(actual, ds4_gpu_tensor_contents(f->reference), (uint64_t)rows * d.out * 4));
            }
        }
        fprintf(stderr, "Q4 projection %s (%u -> %u): CPU samples and scalar batches PASS\n", d.name, d.in, d.out);
    }
    return 1;
}

static int output_call(fixture *f, ds4_gpu_tensor *out, ds4_gpu_tensor *low,
                       const ds4_gpu_tensor *heads, unsigned a, unsigned b,
                       uint32_t rows, uint32_t world, uint32_t rank) {
    return ds4_gpu_dsv41_attention_output_typed_batch(out, low, f->model, f->size,
        f->oa[a], f->ob[b], a ? Q8 : Q4, b ? Q8 : Q4, heads, rows, world, rank);
}

static int output_case(fixture *f, unsigned a, unsigned b, uint32_t rows,
                       uint32_t world, uint32_t rank, int bench) {
    const uint32_t head_width = HEADS / world, low_width = LOW / world;
    const uint32_t first_group = rank * (8 / world);
    const uint32_t atype = a ? Q8 : Q4, btype = b ? Q8 : Q4;
    fill_input(f->x, (uint64_t)rows * head_width);
    CHECK(ds4_gpu_begin_commands());
    CHECK(output_call(f, f->y, f->low, f->x, a, b, rows, world, rank));
    CHECK(ds4_gpu_end_commands());
    const float *x = ds4_gpu_tensor_contents(f->x), *low = ds4_gpu_tensor_contents(f->low);
    const float *actual = ds4_gpu_tensor_contents(f->y);
    for (uint64_t i = 0; i < (uint64_t)rows * low_width; i++)
        CHECK(isfinite(low[i]) && low[i] == bf16(low[i]));
    const uint32_t rr[] = {0, rows / 2, rows - 1};
    const uint32_t oc[] = {0, 1, 31, 32, 1023, 1024, OUTPUT / 2, OUTPUT - 1};
    for (unsigned i = 0; i < 3; i++) {
        const uint32_t r = rr[i];
        for (uint32_t group = 0; group < 8 / world; group++) {
            const uint32_t cc[] = {0, 15, 16, RANK / 2, RANK - 1};
            for (unsigned j = 0; j < 5; j++) {
                double absolute;
                const uint64_t weight_row = (first_group + group) * RANK + cc[j];
                const double exact = dot(f->model + f->oa[a] + weight_row * row_bytes(atype, GROUP),
                    atype, x + (uint64_t)r * head_width + group * GROUP, 0, GROUP, &absolute);
                const float expected = bf16((float)exact);
                const float got = low[(uint64_t)r * low_width + group * RANK + cc[j]];
                /* At BF16 ties, different FP32 reduction orders can select
                 * adjacent BF16 numbers. Bound that to one BF16 step. */
                const double bf16_step = ldexp(1.0, ilogb(fmax(fabs(expected), 0x1p-126)) - 7);
                CHECK(fabs((double)got - expected) <= bf16_step + 2e-5 * absolute + 1e-7);
            }
        }
        for (unsigned j = 0; j < sizeof(oc) / sizeof(*oc); j++) {
            double absolute;
            /* Stage two is independently decoded against the actual BF16
             * boundary, whose values and grouped stage were checked above. */
            const double expected = dot(f->model + f->ob[b] + oc[j] * row_bytes(btype, LOW),
                btype, low + (uint64_t)r * low_width, rank * low_width, low_width, &absolute);
            const double got = actual[(uint64_t)r * OUTPUT + oc[j]];
            CHECK(isfinite(got) && fabs(got - expected) <= 2e-5 * absolute + 1e-7);
        }
    }
    if (a && b) {
        /* Q8/Q8 remains precisely the old release implementation. */
        CHECK(ds4_gpu_begin_commands());
        CHECK(world == 1 ? ds4_gpu_dsv41_attention_output_batch(f->reference, f->reference_low,
            f->model, f->size, f->oa[a], f->ob[b], f->x, rows) :
            ds4_gpu_dsv41_attention_output_tp_batch(f->reference, f->reference_low,
                f->model, f->size, f->oa[a], f->ob[b], f->x, rows, rank));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(low, ds4_gpu_tensor_contents(f->reference_low), (uint64_t)rows * low_width * 4));
        CHECK(!memcmp(actual, ds4_gpu_tensor_contents(f->reference), (uint64_t)rows * OUTPUT * 4));
    } else if (rows <= 8) {
        CHECK(ds4_gpu_begin_commands());
        for (uint32_t r = 0; r < rows; r++) {
            ds4_gpu_tensor *xr = ds4_gpu_tensor_view(f->x, (uint64_t)r * head_width * 4, head_width * 4);
            ds4_gpu_tensor *lr = ds4_gpu_tensor_view(f->reference_low, (uint64_t)r * low_width * 4, low_width * 4);
            ds4_gpu_tensor *yr = ds4_gpu_tensor_view(f->reference, (uint64_t)r * OUTPUT * 4, OUTPUT * 4);
            const int ok = xr && lr && yr && output_call(f, yr, lr, xr, a, b, 1, world, rank);
            ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(lr); ds4_gpu_tensor_free(xr);
            CHECK(ok);
        }
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(low, ds4_gpu_tensor_contents(f->reference_low), (uint64_t)rows * low_width * 4));
        CHECK(!memcmp(actual, ds4_gpu_tensor_contents(f->reference), (uint64_t)rows * OUTPUT * 4));
    }
    if (bench && world == 1 && (rows == 1 || rows == 128)) {
        const double start = seconds();
        for (unsigned i = 0; i < 5; i++) {
            CHECK(ds4_gpu_begin_commands());
            CHECK(output_call(f, f->y, f->low, f->x, a, b, rows, world, rank));
            CHECK(ds4_gpu_end_commands());
        }
        fprintf(stderr, "synthetic OA/OB types=%u/%u rows=%u mean=%.3f ms (5 warm runs)\n",
            atype, btype, rows, (seconds() - start) * 200.0);
    }
    fprintf(stderr, "OA/OB types=%u/%u rows=%u TP=%u/%u: PASS\n", atype, btype, rows, rank, world);
    return 1;
}

static int invalid_cases(fixture *f) {
    ds4_gpu_tensor *short_low = ds4_gpu_tensor_view(f->low, 0, LOW * 4 - 4);
    ds4_gpu_tensor *unaligned = ds4_gpu_tensor_view(f->low, 4, LOW * 4);
    CHECK(short_low && unaligned);
    CHECK(ds4_gpu_begin_commands());
    CHECK(!output_call(f, f->y, short_low, f->x, 0, 0, 1, 1, 0));
    CHECK(!output_call(f, f->y, unaligned, f->x, 0, 0, 1, 1, 0));
    CHECK(!output_call(f, f->y, f->x, f->x, 0, 0, 1, 1, 0));
    CHECK(!output_call(f, f->y, f->y, f->x, 0, 0, 1, 1, 0));
    CHECK(!output_call(f, f->y, f->low, f->x, 0, 0, 0, 1, 0));
    CHECK(!output_call(f, f->y, f->low, f->x, 0, 0, UINT32_MAX, 1, 0));
    CHECK(!output_call(f, f->y, f->low, f->x, 0, 0, 1, 3, 0));
    CHECK(!output_call(f, f->y, f->low, f->x, 0, 0, 1, 2, 2));
    CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->low, f->model, f->size,
        f->size - 32, f->ob[0], Q4, Q4, f->x, 1, 1, 0));
    CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->low, f->model, f->size,
        f->oa[0], f->size - 32, Q4, Q4, f->x, 1, 1, 0));
    CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->low, f->model, f->size,
        f->oa[0], f->ob[0], 2, Q4, f->x, 1, 1, 0));
    CHECK(ds4_gpu_end_commands());
    ds4_gpu_tensor_free(short_low);
    ds4_gpu_tensor_free(unaligned);
    return 1;
}

static int compare_times(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int direct_cases(fixture *f, int bench) {
    const char *previous = getenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT");
    char *saved = previous ? strdup(previous) : NULL;
    CHECK(!previous || saved);
    for (uint32_t rows = 512; rows <= 513; rows++) {
        CHECK(unsetenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT") == 0);
        CHECK(output_case(f, 0, 0, rows, 1, 0, 0));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_tensor_copy(f->reference_low, 0, f->low, 0, (uint64_t)rows * LOW * 4));
        CHECK(ds4_gpu_tensor_copy(f->reference, 0, f->y, 0, (uint64_t)rows * OUTPUT * 4));
        CHECK(ds4_gpu_end_commands());
        CHECK(setenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT", "1", 1) == 0);
        CHECK(ds4_gpu_begin_commands());
        CHECK(output_call(f, f->y, f->low, f->x, 0, 0, rows, 1, 0));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(f->low), ds4_gpu_tensor_contents(f->reference_low),
                       (uint64_t)rows * LOW * 4));
        CHECK(!memcmp(ds4_gpu_tensor_contents(f->y), ds4_gpu_tensor_contents(f->reference),
                       (uint64_t)rows * OUTPUT * 4));
        fprintf(stderr, "Q4 direct/native grouped rows=%u bitwise PASS\n", rows);
        if (bench && rows == 512) {
            double samples[2][7];
            for (unsigned run = 0; run < 9; run++) for (unsigned turn = 0; turn < 2; turn++) {
                const unsigned mode = (run + turn) & 1u;
                CHECK(mode ? unsetenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT") == 0 :
                    setenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT", "1", 1) == 0);
                const double start = seconds();
                CHECK(ds4_gpu_begin_commands());
                for (unsigned i = 0; i < 4; i++)
                    CHECK(output_call(f, f->y, f->low, f->x, 0, 0, rows, 1, 0));
                CHECK(ds4_gpu_end_commands());
                if (run >= 2) samples[mode][run - 2] = (seconds() - start) * 250.0;
            }
            for (unsigned mode = 0; mode < 2; mode++) qsort(samples[mode], 7, sizeof(double), compare_times);
            fprintf(stderr, "synthetic Q4 OA/OB rows=%u median ms native=%.3f direct=%.3f\n",
                    rows, samples[0][3], samples[1][3]);
        }
    }
    if (saved) { CHECK(setenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT", saved, 1) == 0); free(saved); }
    else CHECK(unsetenv("DS4_METAL_DISABLE_ATTN_OUT_LOW_DIRECT") == 0);
    return 1;
}

static int qb_rhs_cases(fixture *f, int bench) {
    enum { ROWS = 2049, WIDTH = 1280 };
    const uint64_t count = (uint64_t)ROWS * HEADS;
    ds4_gpu_tensor *storage = ds4_gpu_tensor_alloc((count + 2 * PAD) * sizeof(float));
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(count * sizeof(float));
    CHECK(storage && reference);
    ds4_gpu_tensor *out = ds4_gpu_tensor_view(storage, PAD * sizeof(float), count * sizeof(float));
    ds4_gpu_tensor *short_rhs = ds4_gpu_tensor_view(f->low, 0, 128u * WIDTH * 2u - 2u);
    ds4_gpu_tensor *unaligned = ds4_gpu_tensor_view(f->low, 4, 128u * WIDTH * 2u);
    CHECK(out && short_rhs && unaligned && guards(storage, count, 1));
    float *input = ds4_gpu_tensor_contents(f->x);
    for (uint64_t i = 0; i < (uint64_t)ROWS * WIDTH; i++)
        /* Deliberately include values between FP16 numbers: the dedicated
         * packing pass must round like native on-tile float-to-half casts. */
        input[i] = ((int)(random_bits() % 65535) - 32767) / 65536.0f;
    const uint32_t rows[] = {127, 128, 129, 512, 513, 2048, 2049};
    for (unsigned r = 0; r < sizeof(rows) / sizeof(*rows); r++) {
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_q4_qb_rows(out, f->low, f->model, f->size, f->dense, f->x, rows[r]));
        CHECK(ds4_gpu_matmul_quant_tensor(reference, f->model, f->size, f->dense,
                                          Q4, WIDTH, HEADS, f->x, rows[r]));
        CHECK(ds4_gpu_end_commands());
        CHECK(!memcmp(ds4_gpu_tensor_contents(out), ds4_gpu_tensor_contents(reference),
                       (uint64_t)rows[r] * HEADS * sizeof(float)));
        if (bench && (rows[r] == 128 || rows[r] == 512)) {
            double samples[2][7];
            for (unsigned run = 0; run < 9; run++) for (unsigned turn = 0; turn < 2; turn++) {
                const unsigned mode = (run + turn) & 1u;
                const double start = seconds();
                CHECK(ds4_gpu_begin_commands());
                for (unsigned i = 0; i < 4; i++) {
                    CHECK(mode ? ds4_gpu_dsv41_q4_qb_rows(out, f->low, f->model, f->size,
                        f->dense, f->x, rows[r]) : ds4_gpu_matmul_quant_tensor(reference,
                        f->model, f->size, f->dense, Q4, WIDTH, HEADS, f->x, rows[r]));
                }
                CHECK(ds4_gpu_end_commands());
                if (run >= 2) samples[mode][run - 2] = (seconds() - start) * 250.0;
            }
            for (unsigned mode = 0; mode < 2; mode++) qsort(samples[mode], 7, sizeof(double), compare_times);
            fprintf(stderr, "synthetic Q_B rows=%u median ms native=%.3f packed=%.3f\n",
                    rows[r], samples[0][3], samples[1][3]);
        }
        fprintf(stderr, "Q4 Q_B FP16 RHS rows=%u native bitwise PASS\n", rows[r]);
    }
    CHECK(ds4_gpu_begin_commands());
    CHECK(!ds4_gpu_dsv41_q4_qb_rows(out, short_rhs, f->model, f->size, f->dense, f->x, 128));
    CHECK(!ds4_gpu_dsv41_q4_qb_rows(out, unaligned, f->model, f->size, f->dense, f->x, 128));
    CHECK(!ds4_gpu_dsv41_q4_qb_rows(out, f->x, f->model, f->size, f->dense, f->x, 128));
    CHECK(!ds4_gpu_dsv41_q4_qb_rows(out, out, f->model, f->size, f->dense, f->x, 128));
    CHECK(!ds4_gpu_dsv41_q4_qb_rows(out, f->low, f->model, f->size, f->size - 32, f->x, 128));
    CHECK(!ds4_gpu_dsv41_q4_qb_rows(out, f->low, f->model, f->size, f->dense, f->x, UINT32_MAX));
    CHECK(ds4_gpu_end_commands());
    CHECK(guards(storage, count, 0));
    ds4_gpu_tensor_free(unaligned); ds4_gpu_tensor_free(short_rhs);
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(reference); ds4_gpu_tensor_free(storage);
    return 1;
}

int main(int argc, char **argv) {
    const int bench = argc == 2 && !strcmp(argv[1], "--bench");
    if (argc != 1 && !bench) {
        fprintf(stderr, "usage: %s [--bench]\n", argv[0]);
        return 2;
    }
    fixture f = {0};
    int ok = initialize(&f);
    if (ok) ok = projection_cases(&f);
    const uint32_t rows[] = {1, 2, 8, 31, 32, 33, 128};
    for (unsigned a = 0; ok && a < 2; a++) for (unsigned b = 0; ok && b < 2; b++)
        for (unsigned r = 0; ok && r < sizeof(rows) / sizeof(*rows); r++)
            ok = output_case(&f, a, b, rows[r], 1, 0, bench);
    const uint32_t tp_rows[] = {1, 2, 8, 32, 33};
    for (unsigned a = 0; ok && a < 2; a++) for (unsigned b = 0; ok && b < 2; b++)
        for (unsigned rank = 0; ok && rank < 2; rank++)
            for (unsigned r = 0; ok && r < sizeof(tp_rows) / sizeof(*tp_rows); r++)
                ok = output_case(&f, a, b, tp_rows[r], 2, rank, 0);
    if (ok) ok = invalid_cases(&f);
    if (ok) ok = direct_cases(&f, bench);
    if (ok) ok = qb_rhs_cases(&f, bench);
    if (ok) ok = guards(f.xs, (uint64_t)MAX_ROWS * HEADS, 0) &&
        guards(f.ys, (uint64_t)MAX_ROWS * HEADS, 0) && guards(f.ls, (uint64_t)MAX_ROWS * LOW, 0);
    cleanup(&f);
    if (ok) puts("V4.1 Q4 attention: CPU dequantization, BF16 boundary, scalar batches, Q8 regression, TP slices and guards PASS");
    return ok ? 0 : 1;
}
