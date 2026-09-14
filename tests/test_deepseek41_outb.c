#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

enum { Q8 = 8, Q4 = 12, HEADS = 32768, LOW = 8192, OUT = 5120, MAX_ROWS = 2048, PAD = 16 };
#define EXPANDED ((uint64_t)LOW * OUT * 2u)
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
typedef struct { ds4_gpu_tensor *storage, *view; uint64_t bytes; } buffer;
typedef struct {
    void *model, *input_copy;
    uint64_t size, a[2], b[2];
    buffer input, scratch;
    unsigned a_type, b_type;
} fixture;

static uint32_t seed = 7919;
static uint32_t random_bits(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return seed;
}

static int allocate(buffer *b, uint64_t bytes) {
    b->bytes = bytes;
    b->storage = ds4_gpu_tensor_alloc(bytes + 2u * PAD);
    CHECK(b->storage);
    b->view = ds4_gpu_tensor_view(b->storage, PAD, bytes);
    CHECK(b->view && ds4_gpu_tensor_contents(b->storage));
    memset(ds4_gpu_tensor_contents(b->storage), 0xa5, bytes + 2u * PAD);
    return 1;
}

static void release(buffer *b) {
    ds4_gpu_tensor_free(b->view); ds4_gpu_tensor_free(b->storage);
    memset(b, 0, sizeof(*b));
}

static int guards(const buffer *b) {
    const uint8_t *p = ds4_gpu_tensor_contents(b->storage);
    for (unsigned i = 0; i < PAD; i++)
        CHECK(p[i] == 0xa5 && p[PAD + b->bytes + i] == 0xa5);
    return 1;
}

static int canary(ds4_gpu_tensor *t, uint64_t first, uint64_t last) {
    const uint8_t *p = ds4_gpu_tensor_contents(t);
    for (uint64_t i = first; i < last; i++) CHECK(p[i] == 0xa5);
    return 1;
}

static void fill_q4(void *dst, uint64_t count) {
    q4_block *w = dst;
    for (uint64_t i = 0; i < count; i++) {
        /* Non-power-of-two scales exercise the half dequantization boundary. */
        w[i].d = (uint16_t)(0x0800u + random_bits() % 0x1400u);
        w[i].dmin = (uint16_t)(0x0800u + random_bits() % 0x1400u);
        for (unsigned j = 0; j < 12; j++) w[i].scales[j] = random_bits();
        for (unsigned j = 0; j < 128; j++) w[i].qs[j] = random_bits();
    }
}

static int initialize(fixture *f) {
    CHECK(sizeof(q4_block) == 144 && sizeof(q8_block) == 34);
    f->a[0] = 128;
    f->a[1] = f->a[0] + (uint64_t)4096 * LOW / 256u * 144u + 128;
    f->b[0] = f->a[1] + (uint64_t)4096 * LOW / 32u * 34u + 128;
    f->b[1] = f->b[0] + (uint64_t)LOW * OUT / 256u * 144u + 128;
    f->size = f->b[1] + (uint64_t)LOW * OUT / 32u * 34u + 128;
    f->model = mmap(NULL, f->size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(f->model != MAP_FAILED);
    fill_q4((char *)f->model + f->a[0], (uint64_t)4096 * LOW / 256u);
    fill_q4((char *)f->model + f->b[0], (uint64_t)LOW * OUT / 256u);
    for (unsigned matrix = 0; matrix < 2; matrix++) {
        q8_block *q8 = (q8_block *)((char *)f->model + (matrix ? f->b[1] : f->a[1]));
        const uint64_t blocks = (uint64_t)LOW * (matrix ? OUT : 4096u) / 32u;
        for (uint64_t i = 0; i < blocks; i++) {
            q8[i].d = (uint16_t)(0x0800u + random_bits() % 0x1000u);
            for (unsigned j = 0; j < 32; j++) q8[i].qs[j] = (int8_t)random_bits();
        }
    }
    CHECK(mprotect(f->model, f->size, PROT_READ) == 0);
    CHECK(allocate(&f->input, (uint64_t)(MAX_ROWS + 1u) * HEADS * 4u));
    /* Also holds the registered fake model in the no-copy alias rejection. */
    CHECK(allocate(&f->scratch, f->size + 16u * 1024u * 1024u));
    float *x = ds4_gpu_tensor_contents(f->input.view);
    const uint32_t ties[] = {0x3f800fffu, 0x3f801000u, 0x3f801001u,
        0x3f803000u, 0xbf801000u, 0xbf803000u, 0x00000000u, 0x80000000u};
    for (uint64_t i = 0; i < f->input.bytes / 4u; i++) {
        if (i % 17u == 0) {
            const uint32_t bits = ties[(i / 17u) % 8u];
            memcpy(x + i, &bits, 4);
        } else x[i] = ((int)(random_bits() % 65537u) - 32768) / 32768.0f;
    }
    f->input_copy = malloc(f->input.bytes);
    CHECK(f->input_copy);
    memcpy(f->input_copy, x, f->input.bytes);
    CHECK(ds4_gpu_set_model_map(f->model, f->size));
    return 1;
}

static int sequence(fixture *f, ds4_gpu_tensor *out, ds4_gpu_tensor *low,
                    ds4_gpu_tensor *scratch, const ds4_gpu_tensor *heads,
                    uint32_t rows, uint32_t tp, int candidate) {
    const unsigned type = f->a_type ? Q8 : Q4, btype = f->b_type ? Q8 : Q4;
    if (candidate) return ds4_gpu_dsv41_attention_output_typed_workspace_batch(
        out, low, scratch, f->model, f->size, f->a[f->a_type], f->b[f->b_type],
        type, btype, heads, rows, tp, 0);
    return ds4_gpu_dsv41_attention_output_typed_batch(out, low, f->model, f->size,
        f->a[f->a_type], f->b[f->b_type], type, btype, heads, rows, tp, 0);
}

static int equal(ds4_gpu_tensor *a, ds4_gpu_tensor *b, uint64_t bytes) {
    const uint32_t *x = ds4_gpu_tensor_contents(a), *y = ds4_gpu_tensor_contents(b);
    if (memcmp(x, y, bytes)) {
        for (uint64_t i = 0; i < bytes / 4u; i++) if (x[i] != y[i]) {
            fprintf(stderr, "output-B mismatch at %llu: reference=%08x candidate=%08x\n",
                    (unsigned long long)i, x[i], y[i]); break;
        }
        return 0;
    }
    return 1;
}

static int parity(fixture *f, uint32_t rows, int full, uint32_t tp, int admitted) {
    const uint64_t ob = (uint64_t)rows * OUT * 4u, lb = (uint64_t)rows * LOW * 4u / tp;
    const uint64_t rhs = (uint64_t)rows * LOW * 2u;
    const uint64_t sb = rhs + (full ? EXPANDED : 0u) - (full == 2 ? 16u : 0u);
    buffer out[2] = {{0}}, low[2] = {{0}};
    ds4_gpu_tensor *scratch = ds4_gpu_tensor_view(f->scratch.view, 0, sb);
    CHECK(scratch);
    memset(ds4_gpu_tensor_contents(f->scratch.view), 0xa5, f->scratch.bytes);
    for (unsigned mode = 0; mode < 2; mode++) {
        CHECK(allocate(&out[mode], ob) && allocate(&low[mode], lb));
        CHECK(sequence(f, out[mode].view, low[mode].view, scratch, f->input.view, rows, tp, mode));
    }
    CHECK(equal(low[0].view, low[1].view, lb) && equal(out[0].view, out[1].view, ob));
    if (admitted) {
        const uint8_t *p = ds4_gpu_tensor_contents(scratch);
        unsigned changed = 0;
        for (unsigned i = 0; i < 64; i++) changed += p[rhs + 16u * 1024u * 1024u + i] != 0xa5;
        CHECK(changed != 0); /* Detect a silently unexercised transient branch. */
    } else CHECK(canary(f->scratch.view, 0, f->scratch.bytes));
    CHECK(canary(f->scratch.view, sb, f->scratch.bytes));
    CHECK(!memcmp(f->input_copy, ds4_gpu_tensor_contents(f->input.view), f->input.bytes));
    CHECK(guards(&f->scratch) && guards(&f->input));
    for (unsigned i = 0; i < 2; i++) {
        CHECK(guards(&out[i]) && guards(&low[i])); release(&out[i]); release(&low[i]);
    }
    ds4_gpu_tensor_free(scratch);
    fprintf(stderr, "V4.1 out-B exact F32/low-BF16 rows=%u A=%s B=%s workspace=%d TP=%u transient=%d PASS\n",
            rows, f->a_type ? "Q8" : "Q4", f->b_type ? "Q8" : "Q4", full, tp, admitted);
    return 1;
}

static int queued(fixture *f) {
    enum { ROWS = 256 };
    buffer out[2][2] = {{{0}}}, low[2][2] = {{{0}}};
    const uint64_t hb = (uint64_t)ROWS * HEADS * 4u, lb = (uint64_t)ROWS * LOW * 4u;
    const uint64_t ob = (uint64_t)ROWS * OUT * 4u, sb = EXPANDED + lb / 2u;
    for (unsigned mode = 0; mode < 2; mode++) {
        CHECK(ds4_gpu_begin_commands());
        for (unsigned i = 0; i < 2; i++) {
            CHECK(allocate(&out[mode][i], ob) && allocate(&low[mode][i], lb));
            ds4_gpu_tensor *h = ds4_gpu_tensor_view(f->input.view, i * HEADS * 4u, hb);
            ds4_gpu_tensor *s = ds4_gpu_tensor_view(f->scratch.view, 0, sb);
            CHECK(h && s && sequence(f, out[mode][i].view, low[mode][i].view, s, h, ROWS, 1, mode));
            ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(h);
        }
        CHECK(ds4_gpu_end_commands());
    }
    for (unsigned i = 0; i < 2; i++) {
        CHECK(equal(out[0][i].view, out[1][i].view, ob));
        CHECK(equal(low[0][i].view, low[1][i].view, lb));
        for (unsigned m = 0; m < 2; m++) {
            CHECK(guards(&out[m][i]) && guards(&low[m][i]));
            release(&out[m][i]); release(&low[m][i]);
        }
    }
    CHECK(guards(&f->scratch) && guards(&f->input));
    CHECK(!memcmp(f->input_copy, ds4_gpu_tensor_contents(f->input.view), f->input.bytes));
    fprintf(stderr, "V4.1 out-B queued workspace reuse / released views PASS\n");
    return 1;
}

static int rejected(fixture *f) {
    enum { ROWS = 256 };
    const uint64_t lb = (uint64_t)ROWS * LOW * 4u, ob = (uint64_t)ROWS * OUT * 4u;
    const uint64_t hb = (uint64_t)ROWS * HEADS * 4u, sb = EXPANDED + lb / 2u;
    buffer out = {0}, low = {0};
    CHECK(allocate(&out, ob) && allocate(&low, lb));
    memset(ds4_gpu_tensor_contents(f->scratch.view), 0xa5, f->scratch.bytes);
    ds4_gpu_tensor *scratch = ds4_gpu_tensor_view(f->scratch.view, 0, sb);
    CHECK(scratch);
#define CALL(o, l, s, h) sequence(f, o, l, s, h, ROWS, 1, 1)
    CHECK(!CALL(NULL, low.view, scratch, f->input.view));
    CHECK(!CALL(out.view, NULL, scratch, f->input.view));
    CHECK(!CALL(out.view, low.view, scratch, NULL));
    ds4_gpu_tensor *bad = ds4_gpu_tensor_view(f->scratch.storage, 4, sb);
    CHECK(bad && !CALL(out.view, low.view, bad, f->input.view)); ds4_gpu_tensor_free(bad);
    /* These aliases overlap expanded weights, outside the original RHS. */
    const uint64_t internal = 16u * 1024u * 1024u;
    bad = ds4_gpu_tensor_view(f->scratch.view, internal, hb);
    CHECK(bad && !CALL(out.view, low.view, scratch, bad)); ds4_gpu_tensor_free(bad);
    bad = ds4_gpu_tensor_view(f->scratch.view, internal, lb);
    CHECK(bad && !CALL(out.view, bad, scratch, f->input.view)); ds4_gpu_tensor_free(bad);
    bad = ds4_gpu_tensor_view(f->scratch.view, internal, ob);
    CHECK(bad && !CALL(bad, low.view, scratch, f->input.view)); ds4_gpu_tensor_free(bad);
    bad = ds4_gpu_tensor_view(f->input.view, PAD, lb);
    CHECK(bad && !CALL(out.view, bad, scratch, f->input.view)); ds4_gpu_tensor_free(bad);
    const void *model = f->model;
    /* Model registration requires a page base, unlike the offset16 workspace
     * view. Round down inside its expanded-weight region and prove capacity. */
    const long page_size = sysconf(_SC_PAGESIZE);
    CHECK(page_size > 0);
    const uintptr_t scratch_start = (uintptr_t)ds4_gpu_tensor_contents(f->scratch.view);
    const uintptr_t alias = scratch_start + internal -
        (scratch_start + internal) % (uintptr_t)page_size;
    CHECK(alias >= scratch_start && alias - scratch_start <= f->scratch.bytes &&
          f->size <= f->scratch.bytes - (alias - scratch_start));
    f->model = (void *)alias;
    CHECK(ds4_gpu_set_model_map(f->model, f->size));
    CHECK(!CALL(out.view, low.view, scratch, f->input.view));
    f->model = (void *)model;
    CHECK(ds4_gpu_set_model_map(f->model, f->size));
    CHECK(canary(out.view, 0, ob) && canary(low.view, 0, lb));
    CHECK(canary(f->scratch.view, 0, f->scratch.bytes));
    CHECK(!memcmp(f->input_copy, ds4_gpu_tensor_contents(f->input.view), f->input.bytes));
    CHECK(guards(&out) && guards(&low) && guards(&f->scratch) && guards(&f->input));
#undef CALL
    ds4_gpu_tensor_free(scratch); release(&out); release(&low);
    fprintf(stderr, "V4.1 out-B full workspace alias and no-write rejection PASS\n");
    return 1;
}

static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int compare_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static int benchmark(fixture *f, uint32_t rows) {
    enum { SAMPLES = 9 };
    buffer out = {0}, low = {0};
    CHECK(allocate(&out, (uint64_t)rows * OUT * 4u));
    CHECK(allocate(&low, (uint64_t)rows * LOW * 4u));
    double samples[2][SAMPLES];
    for (unsigned sample = 0; sample < SAMPLES + 2; sample++) {
        double elapsed[2] = {0, 0};
        for (unsigned turn = 0; turn < 4; turn++) {
            const unsigned mode = (turn == 1 || turn == 2) ^ (sample & 1u);
            const double start = seconds();
            CHECK(sequence(f, out.view, low.view, f->scratch.view, f->input.view, rows, 1, mode));
            elapsed[mode] += seconds() - start;
        }
        if (sample >= 2) for (unsigned mode = 0; mode < 2; mode++)
            samples[mode][sample - 2] = elapsed[mode] * 500.0;
    }
    for (unsigned mode = 0; mode < 2; mode++) qsort(samples[mode], SAMPLES, sizeof(double), compare_double);
    fprintf(stderr, "V4.1 output-A+BF16+output-B rows=%u wall median ms native=%.3f transient=%.3f speedup=%.4f "
            "(alternating ABBA; %u samples; RHS conversion and dequant included)\n", rows,
            samples[0][SAMPLES / 2], samples[1][SAMPLES / 2],
            samples[0][SAMPLES / 2] / samples[1][SAMPLES / 2], SAMPLES);
    release(&out); release(&low); return 1;
}

int main(int argc, char **argv) {
    const int bench = argc >= 2 && !strcmp(argv[1], "--bench");
    const uint32_t bench_rows = argc == 3 ? (uint32_t)strtoul(argv[2], NULL, 10) : 1241;
    if (argc > 3 || (argc > 1 && !bench) || bench_rows < 256 || bench_rows > MAX_ROWS) {
        fprintf(stderr, "usage: %s [--bench [256..2048]]\n", argv[0]); return 1;
    }
    fixture f = {0};
    ds4_gpu_set_quality(false);
    if (unsetenv("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") || !ds4_gpu_init()) return 1;
    if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
        fprintf(stderr, "V4.1 out-B transient: SKIP (requires pre-M5 Apple GPU)\n");
        ds4_gpu_cleanup(); return 0;
    }
    if (!initialize(&f)) return 1;
    if (bench) {
        if (!parity(&f, bench_rows, 1, 1, 1) || !benchmark(&f, bench_rows)) return 1;
    } else {
        const uint32_t rows[] = {255, 256, 257, 437, 512, 513, 1024, 1241, 2048, 2049};
        for (unsigned i = 0; i < sizeof(rows) / sizeof(*rows); i++)
            if (!parity(&f, rows[i], 1, 1, rows[i] >= 256 && rows[i] <= MAX_ROWS)) return 1;
        if (!parity(&f, 437, 0, 1, 0) || !parity(&f, 512, 2, 1, 0) ||
            !parity(&f, 256, 1, 2, 0) || !queued(&f) || !rejected(&f)) return 1;
        ds4_gpu_set_quality(true);
        if (!parity(&f, 437, 1, 1, 0)) return 1;
        ds4_gpu_set_quality(false); ds4_gpu_set_ssd_streaming(true);
        if (!parity(&f, 437, 1, 1, 1)) return 1;
        ds4_gpu_set_ssd_streaming(false); f.a_type = 1;
        if (!parity(&f, 437, 1, 1, 1)) return 1;
        f.b_type = 1;
        if (!parity(&f, 437, 1, 1, 0)) return 1;
        f.a_type = 0;
        if (!parity(&f, 437, 1, 1, 0)) return 1;
    }
    release(&f.scratch); release(&f.input); free(f.input_copy);
    ds4_gpu_cleanup();
    return munmap(f.model, f.size) != 0;
}
