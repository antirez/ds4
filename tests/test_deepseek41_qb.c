#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

enum { IN = 1280, OUT = 32768, MAX_ROWS = 2048, PAD = 16 };
#define EXPANDED ((uint64_t)IN * OUT * 2u)
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
typedef struct { ds4_gpu_tensor *storage, *view; uint64_t bytes; } buffer;
typedef struct {
    void *model;
    uint64_t size, offset;
    buffer input, scratch;
    void *input_copy;
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
    ds4_gpu_tensor_free(b->view);
    ds4_gpu_tensor_free(b->storage);
    memset(b, 0, sizeof(*b));
}

static int guards(const buffer *b) {
    const uint8_t *p = ds4_gpu_tensor_contents(b->storage);
    for (unsigned i = 0; i < PAD; i++)
        CHECK(p[i] == 0xa5 && p[PAD + b->bytes + i] == 0xa5);
    return 1;
}

static int initialize(fixture *f) {
    CHECK(sizeof(q4_block) == 144);
    f->offset = 128;
    f->size = f->offset + (uint64_t)IN * OUT / 256u * sizeof(q4_block) + 128u;
    f->model = mmap(NULL, f->size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(f->model != MAP_FAILED);
    q4_block *w = (q4_block *)((uint8_t *)f->model + f->offset);
    for (uint64_t i = 0; i < (uint64_t)IN * OUT / 256u; i++) {
        /* Scales with fractional mantissas exercise dequantization rounding. */
        w[i].d = (uint16_t)(0x0800u + random_bits() % 0x1800u);
        w[i].dmin = (uint16_t)(0x0800u + random_bits() % 0x1800u);
        for (unsigned j = 0; j < 12; j++) w[i].scales[j] = random_bits();
        for (unsigned j = 0; j < 128; j++) w[i].qs[j] = random_bits();
    }
    CHECK(mprotect(f->model, f->size, PROT_READ) == 0);
    CHECK(allocate(&f->input, (uint64_t)(MAX_ROWS + 1u) * IN * 4u));
    CHECK(allocate(&f->scratch, EXPANDED + (uint64_t)MAX_ROWS * IN * 2u));
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

static int reference(fixture *f, ds4_gpu_tensor *out,
                     const ds4_gpu_tensor *x, uint32_t rows) {
    return ds4_gpu_test_q4_attn_q_b_mm_variant_tensor(out, NULL,
        f->model, f->size, f->offset, IN, OUT, x, rows,
        DS4_GPU_TEST_Q4_QB_MM_Q4_F32, false);
}

static int equal(ds4_gpu_tensor *a, ds4_gpu_tensor *b, uint64_t bytes) {
    const uint32_t *x = ds4_gpu_tensor_contents(a), *y = ds4_gpu_tensor_contents(b);
    CHECK(x && y);
    if (memcmp(x, y, bytes)) {
        for (uint64_t i = 0; i < bytes / 4u; i++) if (x[i] != y[i]) {
            fprintf(stderr, "Q-B F32 mismatch at %llu: expected=%08x got=%08x\n",
                    (unsigned long long)i, x[i], y[i]);
            break;
        }
        return 0;
    }
    return 1;
}

static int parity(fixture *f, uint32_t rows, int large) {
    const uint64_t output_bytes = (uint64_t)rows * OUT * 4u;
    const uint64_t rhs_bytes = (uint64_t)rows * IN * 2u;
    const uint64_t scratch_bytes = rhs_bytes + (large ? EXPANDED : 0u) - (large == 2 ? 16u : 0u);
    buffer out = {0}, ref = {0};
    CHECK(allocate(&out, output_bytes) && allocate(&ref, output_bytes));
    ds4_gpu_tensor *scratch = ds4_gpu_tensor_view(f->scratch.view, 0, scratch_bytes);
    CHECK(scratch);
    uint8_t *p = ds4_gpu_tensor_contents(f->scratch.view);
    memset(p, 0xa5, f->scratch.bytes);
    CHECK(reference(f, ref.view, f->input.view, rows));
    CHECK(ds4_gpu_dsv41_q4_qb_rows(out.view, scratch,
        f->model, f->size, f->offset, f->input.view, rows));
    CHECK(equal(ref.view, out.view, output_bytes));
    if (large == 1 && rows >= 256u) {
        /* Parity alone could silently exercise the old RHS-only fallback. */
        unsigned changed = 0;
        for (unsigned i = 0; i < 64; i++)
            changed += p[rhs_bytes + 16u * 1024u * 1024u + i] != 0xa5;
        CHECK(changed != 0);
    }
    CHECK(!memcmp(f->input_copy, ds4_gpu_tensor_contents(f->input.view), f->input.bytes));
    /* Guard the declared workspace view, not only its larger backing buffer. */
    for (uint64_t i = scratch_bytes; i < f->scratch.bytes; i++) CHECK(p[i] == 0xa5);
    CHECK(guards(&f->scratch) && guards(&f->input) && guards(&out) && guards(&ref));
    fprintf(stderr, "V4.1 Q-B F32 exact rows=%u workspace=%s PASS\n", rows,
            large == 2 ? "RHS+80MiB-16B" : large ? "RHS+80MiB" : "RHS-only");
    ds4_gpu_tensor_free(scratch);
    release(&ref); release(&out);
    return 1;
}

static int queued(fixture *f) {
    enum { ROWS = 256 };
    buffer out[2] = {{0}}, ref[2] = {{0}};
    const uint64_t xb = (uint64_t)ROWS * IN * 4u, yb = (uint64_t)ROWS * OUT * 4u;
    for (unsigned i = 0; i < 2; i++) {
        CHECK(allocate(&out[i], yb) && allocate(&ref[i], yb));
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(f->input.view, i * IN * 4u, xb);
        CHECK(x && reference(f, ref[i].view, x, ROWS));
        ds4_gpu_tensor_free(x);
    }
    CHECK(ds4_gpu_begin_commands());
    for (unsigned i = 0; i < 2; i++) {
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(f->input.view, i * IN * 4u, xb);
        ds4_gpu_tensor *s = ds4_gpu_tensor_view(f->scratch.view, 0, EXPANDED + xb / 2u);
        ds4_gpu_tensor *y = ds4_gpu_tensor_view(out[i].view, 0, yb);
        CHECK(x && s && y && ds4_gpu_dsv41_q4_qb_rows(y, s,
            f->model, f->size, f->offset, x, ROWS));
        /* Encoded commands must retain their resources after these views die;
         * the next projection overwrites the same scratch with another RHS. */
        ds4_gpu_tensor_free(y); ds4_gpu_tensor_free(s); ds4_gpu_tensor_free(x);
    }
    CHECK(ds4_gpu_end_commands());
    for (unsigned i = 0; i < 2; i++) {
        CHECK(equal(ref[i].view, out[i].view, yb) && guards(&out[i]));
        release(&ref[i]); release(&out[i]);
    }
    CHECK(!memcmp(f->input_copy, ds4_gpu_tensor_contents(f->input.view), f->input.bytes));
    CHECK(guards(&f->scratch) && guards(&f->input));
    fprintf(stderr, "V4.1 Q-B queued scratch reuse and view lifetime PASS\n");
    return 1;
}

static int rejected(fixture *f) {
    enum { ROWS = 256 };
    const uint64_t xb = (uint64_t)ROWS * IN * 4u, yb = (uint64_t)ROWS * OUT * 4u;
    const uint64_t sb = EXPANDED + xb / 2u, internal = 16u * 1024u * 1024u;
    buffer out = {0};
    CHECK(allocate(&out, yb));
    memset(ds4_gpu_tensor_contents(f->scratch.view), 0xa5, f->scratch.bytes);
    ds4_gpu_tensor *scratch = ds4_gpu_tensor_view(f->scratch.view, 0, sb);
    CHECK(scratch);
#define TRY(y, s, model, size, off, x, rows) \
    ds4_gpu_dsv41_q4_qb_rows(y, s, model, size, off, x, rows)
#define CALL(y, s, x) TRY(y, s, f->model, f->size, f->offset, x, ROWS)
    CHECK(!CALL(NULL, scratch, f->input.view));
    CHECK(!CALL(out.view, NULL, f->input.view));
    CHECK(!CALL(out.view, scratch, NULL));
    CHECK(!TRY(out.view, scratch, f->model, f->size, f->offset + 1, f->input.view, ROWS));
    CHECK(!TRY(out.view, scratch, f->model, f->size - 129, f->offset, f->input.view, ROWS));
    CHECK(!TRY(out.view, scratch, f->model, f->size, f->offset, f->input.view, 0));
    ds4_gpu_tensor *bad = ds4_gpu_tensor_view(f->input.storage, 4, xb);
    CHECK(bad && !CALL(out.view, scratch, bad));
    ds4_gpu_tensor_free(bad);
    bad = ds4_gpu_tensor_view(f->scratch.view, 0, xb / 2u - 16u);
    CHECK(bad && !CALL(out.view, bad, f->input.view));
    ds4_gpu_tensor_free(bad);
    /* Neither view overlaps the old RHS prefix: both overlap expanded weights. */
    bad = ds4_gpu_tensor_view(f->scratch.view, internal, xb);
    CHECK(bad && !CALL(out.view, scratch, bad));
    ds4_gpu_tensor_free(bad);
    bad = ds4_gpu_tensor_view(f->scratch.view, internal, yb);
    CHECK(bad && !CALL(bad, scratch, f->input.view));
    ds4_gpu_tensor_free(bad);
    bad = ds4_gpu_tensor_view(out.view, PAD, xb);
    CHECK(bad && !CALL(out.view, scratch, bad));
    ds4_gpu_tensor_free(bad);
    /* Register each map so rejection cannot merely come from an unknown
     * model. Distinct Metal wrappers still alias these CPU-addressed ranges. */
    CHECK(ds4_gpu_set_model_map(ds4_gpu_tensor_contents(out.storage), out.bytes + 2u * PAD));
    CHECK(!TRY(out.view, scratch, ds4_gpu_tensor_contents(out.storage),
        out.bytes + 2u * PAD, PAD, f->input.view, ROWS));
    CHECK(ds4_gpu_set_model_map(ds4_gpu_tensor_contents(f->scratch.storage),
        f->scratch.bytes + 2u * PAD));
    CHECK(!TRY(out.view, scratch, ds4_gpu_tensor_contents(f->scratch.storage),
        f->scratch.bytes + 2u * PAD, PAD + internal, f->input.view, ROWS));
    CHECK(ds4_gpu_set_model_map(f->model, f->size));
    const uint8_t *a = ds4_gpu_tensor_contents(out.storage);
    const uint8_t *b = ds4_gpu_tensor_contents(f->scratch.storage);
    for (uint64_t i = 0; i < out.bytes + 2u * PAD; i++) CHECK(a[i] == 0xa5);
    for (uint64_t i = 0; i < f->scratch.bytes + 2u * PAD; i++) CHECK(b[i] == 0xa5);
    CHECK(!memcmp(f->input_copy, ds4_gpu_tensor_contents(f->input.view), f->input.bytes));
    CHECK(guards(&f->input));
#undef CALL
#undef TRY
    ds4_gpu_tensor_free(scratch);
    release(&out);
    fprintf(stderr, "V4.1 Q-B full scratch/model/input/output alias guards PASS\n");
    return 1;
}

int main(void) {
    fixture f = {0};
    ds4_gpu_set_quality(false);
    if (unsetenv("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") || !ds4_gpu_init()) return 1;
    if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
        fprintf(stderr, "V4.1 Q-B transient: SKIP (requires pre-M5 Apple GPU)\n");
        ds4_gpu_cleanup();
        return 0;
    }
    if (!initialize(&f)) return 1;
    const uint32_t rows[] = {127, 128, 255, 256, 257, 437, 512, 513, 1024, 2048};
    for (unsigned i = 0; i < sizeof(rows) / sizeof(*rows); i++) {
        if (!parity(&f, rows[i], 1)) return 1;
        if ((rows[i] == 256 || rows[i] == 437 || rows[i] == 512 || rows[i] == 2048) &&
            !parity(&f, rows[i], 0)) return 1;
    }
    if (!parity(&f, 256, 2) || !queued(&f) || !rejected(&f)) return 1;
    ds4_gpu_set_ssd_streaming(true);
    if (!parity(&f, 437, 1)) return 1;
    ds4_gpu_set_ssd_streaming(false);
    fprintf(stderr, "V4.1 Q-B SSD streaming rows=437 PASS\n");
    release(&f.scratch); release(&f.input);
    free(f.input_copy);
    ds4_gpu_cleanup();
    if (munmap(f.model, f.size)) return 1;
    return 0;
}
