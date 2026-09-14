/* Bounded CUDA V4.1 attention-output oracle; no model or distributed runtime.
 * make test-cuda-v41-q4-output
 * ./tests/test_cuda_v41_q4_output --bench
 * DS4_CUDA_MMQ=0 ./tests/test_cuda_v41_q4_output
 * Timings compare synthetic A -> BF16 -> B projections, not model throughput. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { Q8 = 8, Q4 = 12, HEADS = 32768, LOW = 8192, OUTPUT = 5120,
       GROUP = 4096, RANK = 1024, PAD = 16 };
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
typedef struct {
    uint8_t *model;
    uint64_t size, a[2], b[2];
    uint32_t capacity;
    float *input, *actual_low, *actual, *reference_low, *reference;
    ds4_gpu_tensor *xs, *ls, *ys, *x, *low, *y, *ref_low, *ref;
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

static uint64_t row_bytes(uint32_t type, uint32_t width) {
    return type == Q4 ? (uint64_t)width / 256 * sizeof(q4_block) :
                        (uint64_t)width / 32 * sizeof(q8_block);
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
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&x, &bits, sizeof(x));
    return x;
}

/* Only finite, nonnegative Q8_1 scales are converted here. Round to the
 * binary16 lattice independently of the backend's half conversion code. */
static float half_scale(float x) {
    if (!x) return 0;
    int exponent;
    (void)frexpf(x, &exponent);
    const float step = ldexpf(1, exponent - 11 < -24 ? -24 : exponent - 11);
    return nearbyintf(x / step) * step;
}

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

/* CUDA scalar Q4 MMVQ consumes Q8_1 (32 values, half scale, round-away);
 * its non-MMQ fallback and TP2 B consume Q8_K (256, float scale, nearest-even).
 * Q8 weights use Q8_0 activations (32, float scale, nearest-even).
 * A raw-F32 dot is therefore not the numerical oracle for any of these. */
enum { ACT_Q81, ACT_Q8K, ACT_Q80 };
static void quantized_input(float *out, const float *x, uint32_t width, int mode) {
    const uint32_t block = mode == ACT_Q8K ? 256 : 32;
    for (uint32_t first = 0; first < width; first += block) {
        float maxv = 0;
        for (uint32_t j = 0; j < block; j++)
            if (fabsf(x[first + j]) > fabsf(maxv)) maxv = x[first + j];
        const float d = fabsf(maxv) / 127.0f;
        const float inverse = mode == ACT_Q8K ? (maxv ? -127.0f / maxv : 0) :
                                               (d ? 1.0f / d : 0);
        const float stored = mode == ACT_Q81 ? half_scale(d) :
                             mode == ACT_Q8K ? (inverse ? 1.0f / inverse : 0) : d;
        for (uint32_t j = 0; j < block; j++) {
            float q = mode == ACT_Q81 ? (d ? roundf(x[first + j] / d) : 0) :
                                       nearbyintf(x[first + j] * inverse);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out[first + j] = stored * q;
        }
    }
}

static double dot(const uint8_t *row, uint32_t type, const float *x,
                  uint32_t offset, uint32_t width, double *absolute) {
    double sum = 0, bound = 0;
    for (uint32_t k = 0; k < width; k++) {
        const double term = weight_value(row, type, offset + k) * x[k];
        sum += term;
        bound += fabs(term);
    }
    *absolute = bound;
    return sum;
}

static int check_dot(const uint8_t *weight, uint32_t type, float qx[2][LOW],
                     uint32_t offset, uint32_t width, float got, int round_bf16,
                     int only_q8k) {
    CHECK(isfinite(got));
    const unsigned choices = type == Q4 && !only_q8k ? 2 : 1;
    for (unsigned c = 0; c < choices; c++) {
        double absolute;
        const double exact = dot(weight, type, qx[c], offset, width, &absolute);
        const double expected = round_bf16 ? bf16((float)exact) : exact;
        const double step = round_bf16 ?
            ldexp(1.0, ilogb(fmax(fabs(expected), 0x1p-126)) - 7) : 0;
        /* Different FP32 reductions can straddle a BF16 tie. The dot error
         * is bounded relative to absolute terms, including cancellation. */
        if (fabs((double)got - expected) <= step + 2e-5 * absolute + 1e-6) return 1;
    }
    fprintf(stderr, "CPU dot mismatch: type=%u width=%u BF16=%d got=%.9g\n",
            type, width, round_bf16, got);
    return 0;
}

static void fill_weights(uint8_t *dst, uint32_t type, uint64_t elements) {
    if (type == Q4) {
        q4_block *b = (q4_block *)dst;
        for (uint64_t i = 0; i < elements / 256; i++) {
            b[i].d = (i & 1) ? 0x0400 : 0x0800;
            b[i].dmin = (i & 2) ? 0x0400 : 0x0800;
            for (unsigned j = 0; j < 12; j++) b[i].scales[j] = random_bits();
            for (unsigned j = 0; j < 128; j++) b[i].qs[j] = random_bits();
        }
    } else {
        q8_block *b = (q8_block *)dst;
        for (uint64_t i = 0; i < elements / 32; i++) {
            b[i].d = (i & 1) ? 0x1000 : 0x1400;
            for (unsigned j = 0; j < 32; j++) b[i].qs[j] = (int)(random_bits() % 255) - 127;
        }
    }
}

static int guard(ds4_gpu_tensor *storage, uint64_t count, int initialize) {
    uint32_t data[PAD];
    for (unsigned i = 0; i < PAD; i++) data[i] = 0x7fc12345;
    for (unsigned side = 0; side < 2; side++) {
        const uint64_t offset = side ? (PAD + count) * sizeof(float) : 0;
        if (initialize) CHECK(ds4_gpu_tensor_write(storage, offset, data, sizeof(data)));
        else {
            CHECK(ds4_gpu_tensor_read(storage, offset, data, sizeof(data)));
            for (unsigned i = 0; i < PAD; i++) CHECK(data[i] == 0x7fc12345);
        }
    }
    return 1;
}

static int initialize(fixture *f, uint32_t capacity) {
    CHECK(sizeof(q4_block) == 144 && sizeof(q8_block) == 34);
    f->capacity = capacity;
    uint64_t cursor = 128;
    for (unsigned i = 0; i < 2; i++) {
        const uint32_t type = i ? Q8 : Q4;
        f->a[i] = cursor;
        cursor += row_bytes(type, GROUP) * LOW + 128;
        f->b[i] = cursor;
        cursor += row_bytes(type, LOW) * OUTPUT + 128;
    }
    f->size = cursor;
    f->model = malloc(f->size);
    f->input = malloc((uint64_t)capacity * HEADS * sizeof(float));
    f->actual_low = malloc((uint64_t)capacity * LOW * sizeof(float));
    f->reference_low = malloc((uint64_t)capacity * LOW * sizeof(float));
    f->actual = malloc((uint64_t)capacity * OUTPUT * sizeof(float));
    f->reference = malloc((uint64_t)capacity * OUTPUT * sizeof(float));
    CHECK(f->model && f->input && f->actual_low && f->reference_low && f->actual && f->reference);
    memset(f->model, 0xa5, f->size);
    for (unsigned i = 0; i < 2; i++) {
        fill_weights(f->model + f->a[i], i ? Q8 : Q4, (uint64_t)GROUP * LOW);
        fill_weights(f->model + f->b[i], i ? Q8 : Q4, (uint64_t)LOW * OUTPUT);
    }
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(f->model, f->size));
    f->xs = ds4_gpu_tensor_alloc(((uint64_t)capacity * HEADS + 2 * PAD) * sizeof(float));
    f->ls = ds4_gpu_tensor_alloc(((uint64_t)capacity * LOW + 2 * PAD) * sizeof(float));
    f->ys = ds4_gpu_tensor_alloc(((uint64_t)capacity * OUTPUT + 2 * PAD) * sizeof(float));
    CHECK(f->xs && f->ls && f->ys);
    f->x = ds4_gpu_tensor_view(f->xs, PAD * sizeof(float), (uint64_t)capacity * HEADS * sizeof(float));
    f->low = ds4_gpu_tensor_view(f->ls, PAD * sizeof(float), (uint64_t)capacity * LOW * sizeof(float));
    f->y = ds4_gpu_tensor_view(f->ys, PAD * sizeof(float), (uint64_t)capacity * OUTPUT * sizeof(float));
    f->ref_low = ds4_gpu_tensor_alloc((uint64_t)capacity * LOW * sizeof(float));
    f->ref = ds4_gpu_tensor_alloc((uint64_t)capacity * OUTPUT * sizeof(float));
    CHECK(f->x && f->low && f->y && f->ref_low && f->ref);
    CHECK(guard(f->xs, (uint64_t)capacity * HEADS, 1));
    CHECK(guard(f->ls, (uint64_t)capacity * LOW, 1));
    CHECK(guard(f->ys, (uint64_t)capacity * OUTPUT, 1));
    return 1;
}

static void cleanup(fixture *f) {
    ds4_gpu_tensor_free(f->ref); ds4_gpu_tensor_free(f->ref_low);
    ds4_gpu_tensor_free(f->y); ds4_gpu_tensor_free(f->low); ds4_gpu_tensor_free(f->x);
    ds4_gpu_tensor_free(f->ys); ds4_gpu_tensor_free(f->ls); ds4_gpu_tensor_free(f->xs);
    ds4_gpu_cleanup();
    free(f->model); free(f->input); free(f->actual_low); free(f->reference_low);
    free(f->actual); free(f->reference);
}

static int output_call(fixture *f, unsigned a, unsigned b, uint32_t rows,
                       uint32_t world, uint32_t rank) {
    return ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->low, f->model, f->size,
        f->a[a], f->b[b], a ? Q8 : Q4, b ? Q8 : Q4, f->x, rows, world, rank);
}

/* Reproduce the pre-batch CUDA graph's scalar projections, including the
 * grouped-Q4 admission/fallback, the mandatory BF16 boundary and TP K slice. */
static int scalar_reference(fixture *f, unsigned a, unsigned b, uint32_t rows,
                            uint32_t world, uint32_t rank) {
    if (a && b) return world == 1 ?
        ds4_gpu_dsv41_attention_output_batch(f->ref, f->ref_low, f->model, f->size,
            f->a[a], f->b[b], f->x, rows) :
        ds4_gpu_dsv41_attention_output_tp_batch(f->ref, f->ref_low, f->model, f->size,
            f->a[a], f->b[b], f->x, rows, rank);
    const uint32_t groups = 8 / world, group0 = rank * groups;
    for (uint32_t r = 0; r < rows; r++) {
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(f->x,
            (uint64_t)r * HEADS / world * sizeof(float), HEADS / world * sizeof(float));
        ds4_gpu_tensor *low = ds4_gpu_tensor_view(f->ref_low,
            (uint64_t)r * LOW / world * sizeof(float), LOW / world * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(f->ref,
            (uint64_t)r * OUTPUT * sizeof(float), OUTPUT * sizeof(float));
        CHECK(x && low && out);
        int ok;
        if (a) ok = ds4_gpu_attention_output_low_q8_tensor(low, f->model, f->size,
            f->a[a] + (uint64_t)group0 * RANK * row_bytes(Q8, GROUP), GROUP, RANK, groups, x);
        else {
            const int admission = ds4_gpu_attention_output_low_q4_K_slice_tensor(low,
                f->model, f->size, f->a[a], GROUP, RANK, group0, groups, x, 0);
            ok = admission > 0;
            if (!admission) {
                ok = 1;
                for (uint32_t g = 0; ok && g < groups; g++) {
                    ds4_gpu_tensor *gx = ds4_gpu_tensor_view(x, (uint64_t)g * GROUP * sizeof(float), GROUP * sizeof(float));
                    ds4_gpu_tensor *gl = ds4_gpu_tensor_view(low, (uint64_t)g * RANK * sizeof(float), RANK * sizeof(float));
                    ok = gx && gl && ds4_gpu_matmul_quant_tensor(gl, f->model, f->size,
                        f->a[a] + (uint64_t)(group0 + g) * RANK * row_bytes(Q4, GROUP),
                        Q4, GROUP, RANK, gx, 1);
                    ds4_gpu_tensor_free(gl); ds4_gpu_tensor_free(gx);
                }
            }
        }
        ok = ok && ds4_gpu_dsv41_quantize(low, LOW / world, 1, DS4_V41_BF16);
        if (ok) ok = world == 1 ? ds4_gpu_matmul_quant_tensor(out, f->model,
            f->size, f->b[b], b ? Q8 : Q4, LOW, OUTPUT, low, 1) :
            ds4_gpu_matmul_quant_kslice_tensor(out, f->model, f->size, f->b[b],
                b ? Q8 : Q4, LOW, rank * LOW / world, LOW / world, OUTPUT, low, 0);
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(x);
        CHECK(ok);
    }
    return 1;
}

static int cpu_samples(fixture *f, unsigned a, unsigned b, uint32_t rows,
                       uint32_t world, uint32_t rank) {
    const uint32_t atype = a ? Q8 : Q4, btype = b ? Q8 : Q4;
    const uint32_t groups = 8 / world, low_width = LOW / world;
    const uint32_t rr[] = {0, rows / 2, rows - 1};
    const uint32_t cc[] = {0, 15, 16, 31, 32, RANK / 2, RANK - 1};
    const uint32_t oc[] = {0, 1, 31, 32, 1023, 1024, OUTPUT / 2, OUTPUT - 1};
    float qx[2][LOW];
    for (uint64_t i = 0; i < (uint64_t)rows * low_width; i++)
        CHECK(isfinite(f->actual_low[i]) && f->actual_low[i] == bf16(f->actual_low[i]));
    for (unsigned i = 0; i < sizeof(rr) / sizeof(*rr); i++) {
        const uint32_t r = rr[i];
        for (uint32_t g = 0; g < groups; g++) {
            const float *x = f->input + (uint64_t)r * HEADS / world + g * GROUP;
            quantized_input(qx[0], x, GROUP, a ? ACT_Q80 : ACT_Q81);
            if (!a) quantized_input(qx[1], x, GROUP, ACT_Q8K);
            for (unsigned j = 0; j < sizeof(cc) / sizeof(*cc); j++) {
                const uint64_t wr = (rank * groups + g) * RANK + cc[j];
                CHECK(check_dot(f->model + f->a[a] + wr * row_bytes(atype, GROUP),
                    atype, qx, 0, GROUP, f->actual_low[(uint64_t)r * low_width + g * RANK + cc[j]], 1, 0));
            }
        }
        const float *low = f->actual_low + (uint64_t)r * low_width;
        quantized_input(qx[0], low, low_width, b ? ACT_Q80 : world == 2 ? ACT_Q8K : ACT_Q81);
        if (!b && world == 1) quantized_input(qx[1], low, low_width, ACT_Q8K);
        for (unsigned j = 0; j < sizeof(oc) / sizeof(*oc); j++)
            CHECK(check_dot(f->model + f->b[b] + oc[j] * row_bytes(btype, LOW),
                btype, qx, rank * low_width, low_width,
                f->actual[(uint64_t)r * OUTPUT + oc[j]], 0, !b && world == 2));
    }
    return 1;
}

static int same_values(const float *actual, const float *expected, uint64_t count,
                        const char *label) {
    if (!memcmp(actual, expected, count * sizeof(float))) return 1;
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(actual + i, expected + i, sizeof(float))) {
            fprintf(stderr, "%s differs at %llu: actual=%.9g reference=%.9g\n",
                label, (unsigned long long)i, actual[i], expected[i]);
            return 0;
        }
    }
    return 0;
}

static int output_case(fixture *f, unsigned a, unsigned b, uint32_t rows,
                       uint32_t world, uint32_t rank) {
    const uint64_t inputs = (uint64_t)rows * HEADS / world;
    const uint64_t lows = (uint64_t)rows * LOW / world, outputs = (uint64_t)rows * OUTPUT;
    for (uint64_t i = 0; i < inputs; i++)
        f->input[i] = ((int)(random_bits() % 65535) - 32767) / 65536.0f;
    CHECK(ds4_gpu_tensor_write(f->x, 0, f->input, inputs * sizeof(float)));
    CHECK(ds4_gpu_tensor_fill_f32(f->low, NAN, (uint64_t)f->capacity * LOW));
    CHECK(ds4_gpu_tensor_fill_f32(f->y, NAN, (uint64_t)f->capacity * OUTPUT));
    CHECK(scalar_reference(f, a, b, rows, world, rank));
    CHECK(ds4_gpu_synchronize());
    CHECK(output_call(f, a, b, rows, world, rank));
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(f->low, 0, f->actual_low, (uint64_t)f->capacity * LOW * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(f->y, 0, f->actual, (uint64_t)f->capacity * OUTPUT * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(f->ref_low, 0, f->reference_low, lows * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(f->ref, 0, f->reference, outputs * sizeof(float)));
    CHECK(same_values(f->actual_low, f->reference_low, lows, "BF16 low"));
    CHECK(same_values(f->actual, f->reference, outputs, "output"));
    for (uint64_t i = lows; i < (uint64_t)f->capacity * LOW; i++) CHECK(isnan(f->actual_low[i]));
    for (uint64_t i = outputs; i < (uint64_t)f->capacity * OUTPUT; i++) CHECK(isnan(f->actual[i]));
    CHECK(cpu_samples(f, a, b, rows, world, rank));
    CHECK(guard(f->xs, (uint64_t)f->capacity * HEADS, 0));
    CHECK(guard(f->ls, (uint64_t)f->capacity * LOW, 0));
    CHECK(guard(f->ys, (uint64_t)f->capacity * OUTPUT, 0));
    fprintf(stderr, "CUDA V4.1 OA/OB types=%u/%u rows=%u local-rank=%u/%u: bitwise + CPU samples PASS\n",
            a ? Q8 : Q4, b ? Q8 : Q4, rows, rank, world);
    return 1;
}

static int invalid_cases(fixture *f) {
    ds4_gpu_tensor *short_x = ds4_gpu_tensor_view(f->x, 0, HEADS * sizeof(float) - 4);
    ds4_gpu_tensor *short_low = ds4_gpu_tensor_view(f->low, 0, LOW * sizeof(float) - 4);
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_view(f->y, 0, OUTPUT * sizeof(float) - 4);
    ds4_gpu_tensor *unaligned_low = ds4_gpu_tensor_view(f->low, 1, LOW * sizeof(float));
    CHECK(short_x && short_low && short_out && unaligned_low);
    CHECK(ds4_gpu_tensor_fill_f32(f->low, NAN, (uint64_t)f->capacity * LOW));
    CHECK(ds4_gpu_tensor_fill_f32(f->y, NAN, (uint64_t)f->capacity * OUTPUT));
    CHECK(!output_call(f, 0, 0, 0, 1, 0));
    CHECK(!output_call(f, 0, 0, UINT32_MAX, 1, 0));
    CHECK(!output_call(f, 0, 0, 1, 0, 0));
    CHECK(!output_call(f, 0, 0, 1, 3, 0));
    CHECK(!output_call(f, 0, 0, 1, 1, 1));
    CHECK(!output_call(f, 0, 0, 1, 2, 2));
    CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, unaligned_low, f->model, f->size,
        f->a[0], f->b[0], Q4, Q4, f->x, 1, 1, 0));
    CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->x, f->model, f->size,
        f->a[0], f->b[0], Q4, Q4, f->x, 1, 1, 0));
    CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->y, f->model, f->size,
        f->a[0], f->b[0], Q4, Q4, f->x, 1, 1, 0));
    CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->x, f->low, f->model, f->size,
        f->a[0], f->b[0], Q4, Q4, f->x, 1, 1, 0));
    for (unsigned a = 0; a < 2; a++) for (unsigned b = 0; b < 2; b++) {
        const uint32_t atype = a ? Q8 : Q4, btype = b ? Q8 : Q4;
        CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->low, f->model, f->size,
            f->a[a], f->b[b], 2, btype, f->x, 1, 1, 0));
        CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->low, f->model, f->size,
            f->a[a], f->b[b], atype, 2, f->x, 1, 1, 0));
        CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->low, f->model, f->size,
            f->size - 32, f->b[b], atype, btype, f->x, 1, 1, 0));
        CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->low, f->model, f->size,
            f->a[a], f->size - 32, atype, btype, f->x, 1, 1, 0));
        CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, f->low, f->model, f->size,
            f->a[a], f->b[b], atype, btype, short_x, 1, 1, 0));
        CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(f->y, short_low, f->model, f->size,
            f->a[a], f->b[b], atype, btype, f->x, 1, 1, 0));
        CHECK(!ds4_gpu_dsv41_attention_output_typed_batch(short_out, f->low, f->model, f->size,
            f->a[a], f->b[b], atype, btype, f->x, 1, 1, 0));
    }
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_tensor_read(f->low, 0, f->actual_low, (uint64_t)f->capacity * LOW * sizeof(float)));
    CHECK(ds4_gpu_tensor_read(f->y, 0, f->actual, (uint64_t)f->capacity * OUTPUT * sizeof(float)));
    for (uint64_t i = 0; i < (uint64_t)f->capacity * LOW; i++) CHECK(isnan(f->actual_low[i]));
    for (uint64_t i = 0; i < (uint64_t)f->capacity * OUTPUT; i++) CHECK(isnan(f->actual[i]));
    ds4_gpu_tensor_free(unaligned_low);
    ds4_gpu_tensor_free(short_out); ds4_gpu_tensor_free(short_low); ds4_gpu_tensor_free(short_x);
    fprintf(stderr, "CUDA V4.1 invalid types/shapes/ranges: rejected with untouched outputs PASS\n");
    return 1;
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int compare_times(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int benchmark(fixture *f, unsigned a, unsigned b, uint32_t rows) {
    CHECK(output_case(f, a, b, rows, 1, 0));
    double samples[2][7];
    for (unsigned run = 0; run < 9; run++) for (unsigned turn = 0; turn < 2; turn++) {
        const unsigned mode = (run + turn) & 1;
        CHECK(ds4_gpu_synchronize());
        const double start = seconds();
        CHECK(mode ? output_call(f, a, b, rows, 1, 0) : scalar_reference(f, a, b, rows, 1, 0));
        CHECK(ds4_gpu_synchronize());
        if (run >= 2) samples[mode][run - 2] = (seconds() - start) * 1000;
    }
    for (unsigned mode = 0; mode < 2; mode++) qsort(samples[mode], 7, sizeof(double), compare_times);
    printf("synthetic CUDA OA/OB types=%u/%u rows=%u median_ms old=%.3f batch=%.3f speedup=%.2fx\n",
        a ? Q8 : Q4, b ? Q8 : Q4, rows, samples[0][3], samples[1][3], samples[0][3] / samples[1][3]);
    return 1;
}

int main(int argc, char **argv) {
    const int bench = argc == 2 && !strcmp(argv[1], "--bench");
    if (argc != 1 && !bench) {
        fprintf(stderr, "usage: %s [--bench]\n", argv[0]);
        return 2;
    }
    fixture f = {0};
    int ok = initialize(&f, bench ? 512 : 128);
    const uint32_t rows[] = {1, 2, 8, 9, 32, 63, 64, 65, 128};
    for (unsigned a = 0; ok && a < 2; a++) for (unsigned b = 0; ok && b < 2; b++)
        for (unsigned r = 0; ok && r < sizeof(rows) / sizeof(*rows); r++)
            ok = output_case(&f, a, b, rows[r], 1, 0);
    /* Local synthetic slices only: no peer process, transport or all-reduce. */
    const uint32_t tp_rows[] = {1, 9, 65};
    for (unsigned a = 0; ok && a < 2; a++) for (unsigned b = 0; ok && b < 2; b++)
        for (unsigned rank = 0; ok && rank < 2; rank++)
            for (unsigned r = 0; ok && r < sizeof(tp_rows) / sizeof(*tp_rows); r++)
                ok = output_case(&f, a, b, tp_rows[r], 2, rank);
    if (ok) ok = invalid_cases(&f);
    const uint32_t bench_rows[] = {1, 8, 9, 32, 128, 512};
    if (bench) for (unsigned a = 0; ok && a < 2; a++) for (unsigned b = 0; ok && b < 2; b++)
        for (unsigned r = 0; ok && r < sizeof(bench_rows) / sizeof(*bench_rows); r++)
            ok = benchmark(&f, a, b, bench_rows[r]);
    cleanup(&f);
    if (ok) puts("CUDA V4.1 typed output: scalar parity, CPU Q4/Q8 samples, BF16 boundary, local TP slices and guards PASS");
    return ok ? 0 : 1;
}
