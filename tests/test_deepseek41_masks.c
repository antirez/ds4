#include "ds4_gpu.h"
#include "ds4_image.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)

enum { HEADS = 64, DIM = 512, PAD = 64, VOCAB = 100, MODEL_BYTES = 4096 };
enum { Q, RAW, COMP, HALF_COMP, MASK, REF, CACHED, NT };
enum { DYNAMIC = 1, NONCAUSAL = 2, VISUAL = 4 };
static const char *disable_cache = "DS4_METAL_DISABLE_ZERO_PREFIX_PREFILL_MASK_CACHE";
static void *model;

typedef struct {
    uint32_t rows, ratio, pos, past, window, raw_start, flags;
} shape;
typedef struct {
    shape s;
    uint32_t n_raw, raw_cap, n_comp, comp_f16;
    uint64_t bytes[NT], input_hash[REF];
    ds4_gpu_tensor *storage[NT], *t[NT];
    int32_t *tokens;
} fixture;

static uint64_t hash(const void *p, uint64_t bytes) {
    const uint32_t *v = p;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint64_t i = 0; i < bytes / 4; i++) h = (h ^ v[i]) * 0x100000001b3ULL;
    return h;
}

static int initialize(fixture *f, shape s, uint32_t pattern) {
    f->s = s;
    f->n_raw = s.rows + s.past;
    f->raw_cap = f->n_raw + 17 + (pattern & 1u) * 13;
    f->n_comp = s.ratio ? (s.pos + s.rows) / s.ratio : 0;
    f->bytes[Q] = f->bytes[REF] = f->bytes[CACHED] = (uint64_t)s.rows * HEADS * DIM * 4;
    f->bytes[RAW] = (uint64_t)f->raw_cap * DIM * 4;
    f->bytes[COMP] = (uint64_t)(f->n_comp ? f->n_comp : 1) * DIM * 4;
    f->bytes[HALF_COMP] = f->bytes[COMP] / 2;
    f->bytes[MASK] = (uint64_t)s.rows * (f->n_comp ? f->n_comp : 1) * 4;
    for (unsigned i = 0; i < NT; i++) {
        f->storage[i] = ds4_gpu_tensor_alloc(f->bytes[i] + 2 * PAD * 4);
        CHECK(f->storage[i]);
        f->t[i] = ds4_gpu_tensor_view(f->storage[i], PAD * 4, f->bytes[i]);
        CHECK(f->t[i] && ds4_gpu_tensor_contents(f->t[i]));
        uint32_t *v = ds4_gpu_tensor_contents(f->storage[i]);
        for (uint64_t j = 0; j < f->bytes[i] / 4 + 2 * PAD; j++) v[j] = 0x7fc12345u;
    }
    for (unsigned k = Q; k <= COMP; k++) {
        float *v = ds4_gpu_tensor_contents(f->t[k]);
        for (uint64_t j = 0; j < f->bytes[k] / 4; j++) {
            const uint32_t z = (uint32_t)(j * 747796405u + (k + pattern * 11u) * 2891336453u);
            v[j] = ((int32_t)((z ^ (z >> 13)) % 129u) - 64) / 64.0f;
        }
    }
    const float *comp = ds4_gpu_tensor_contents(f->t[COMP]);
    uint16_t *half = ds4_gpu_tensor_contents(f->t[HALF_COMP]);
    for (uint64_t i = 0; i < f->bytes[COMP] / 4; i++) {
        /* Inputs are dyadic and exactly representable in both formats. */
        const _Float16 h = (_Float16)comp[i];
        memcpy(half + i, &h, sizeof(h));
    }
    float *mask = ds4_gpu_tensor_contents(f->t[MASK]);
    for (uint32_t q = 0; q < s.rows; q++) {
        for (uint32_t c = 0; c < (f->n_comp ? f->n_comp : 1); c++) {
            const int visible = s.ratio && c < (s.pos + q + 1) / s.ratio;
            mask[(uint64_t)q * (f->n_comp ? f->n_comp : 1) + c] =
                visible && (q + c + pattern) % 3 ? 0.0f : -INFINITY;
        }
    }
    f->tokens = calloc(s.rows, sizeof(*f->tokens));
    CHECK(f->tokens);
    if (s.flags & VISUAL) {
        CHECK(s.rows >= 10);
        f->tokens[1] = VOCAB + DS4_DEEPSEEK4_IMAGE_PAD;
        f->tokens[2] = VOCAB + DS4_DEEPSEEK4_IMAGE_START;
        for (unsigned i = 3; i < 8; i++) f->tokens[i] = VOCAB + DS4_DEEPSEEK4_IMAGE;
        f->tokens[8] = VOCAB + DS4_DEEPSEEK4_IMAGE_END;
    }
    for (unsigned i = 0; i < REF; i++)
        f->input_hash[i] = hash(ds4_gpu_tensor_contents(f->t[i]), f->bytes[i]);
    return 1;
}

static int encode(fixture *f, unsigned output) {
    const shape s = f->s;
    if (s.flags & NONCAUSAL)
        return ds4_gpu_attention_noncausal_raw_batch_heads_tensor(f->t[output],
            model, MODEL_BYTES, 128, f->t[Q], f->t[RAW], s.rows,
            f->n_raw, f->raw_cap, s.raw_start, HEADS, DIM);
    if (s.flags & VISUAL)
        return ds4_gpu_attention_visual_mixed_batch_heads_tensor(f->t[output],
            model, MODEL_BYTES, 128, f->t[Q], f->t[RAW],
            f->t[f->comp_f16 ? HALF_COMP : COMP], f->comp_f16,
            s.flags & DYNAMIC ? f->t[MASK] : NULL, (s.flags & DYNAMIC) != 0,
            f->tokens, VOCAB, s.rows, s.pos, f->n_raw, f->raw_cap, s.raw_start,
            f->n_comp, s.window, s.ratio, HEADS, DIM);
    if (s.ratio)
        return ds4_gpu_attention_decode_mixed_batch_heads_tensor(f->t[output],
            model, MODEL_BYTES, 128, f->t[Q], f->t[RAW],
            f->t[f->comp_f16 ? HALF_COMP : COMP], f->comp_f16,
            s.flags & DYNAMIC ? f->t[MASK] : NULL, (s.flags & DYNAMIC) != 0,
            s.rows, s.pos, f->n_raw, f->raw_cap, s.raw_start,
            f->n_comp, s.window, s.ratio, HEADS, DIM);
    return ds4_gpu_attention_decode_raw_batch_heads_tensor(f->t[output],
        model, MODEL_BYTES, 128, f->t[Q], f->t[RAW], s.rows, s.pos,
        f->n_raw, f->raw_cap, s.raw_start, s.window, HEADS, DIM);
}

static int compare(fixture *f) {
    const uint32_t *a = ds4_gpu_tensor_contents(f->t[REF]);
    const uint32_t *b = ds4_gpu_tensor_contents(f->t[CACHED]);
    const float *out = ds4_gpu_tensor_contents(f->t[CACHED]);
    for (uint64_t i = 0; i < f->bytes[REF] / 4; i++) {
        if (a[i] != b[i] || !isfinite(out[i])) {
            fprintf(stderr, "mask rows=%u ratio=%u pos=%u flags=%u f16=%u "
                "word=%llu expected=%08x got=%08x\n", f->s.rows, f->s.ratio,
                f->s.pos, f->s.flags, f->comp_f16, (unsigned long long)i, a[i], b[i]);
            return 0;
        }
    }
    for (unsigned i = 0; i < NT; i++) {
        const uint32_t *p = ds4_gpu_tensor_contents(f->storage[i]);
        for (unsigned j = 0; j < PAD; j++)
            CHECK(p[j] == 0x7fc12345u && p[PAD + f->bytes[i] / 4 + j] == 0x7fc12345u);
        if (i < REF) CHECK(hash(ds4_gpu_tensor_contents(f->t[i]), f->bytes[i]) == f->input_hash[i]);
    }
    return 1;
}

static void destroy(fixture *f) {
    for (unsigned i = 0; i < NT; i++) {
        ds4_gpu_tensor_free(f->t[i]);
        ds4_gpu_tensor_free(f->storage[i]);
    }
    free(f->tokens);
}

static int parity(shape s) {
    ds4_gpu_release_zero_prefix_prefill_mask_cache();
    if (s.flags || s.pos || s.past || s.window != 128 || s.raw_start || s.ratio == 4) {
        fixture warm = {0};
        const shape base = {s.rows, s.ratio == 4 ? 0 : s.ratio, 0, 0, 128, 0, 0};
        CHECK(initialize(&warm, base, 17));
        CHECK(encode(&warm, CACHED));
        destroy(&warm);
    }
    /* Two independent Q/KV inputs use the same warm mask. Switch comp dtype
     * too: neither input values nor their storage format belong in its key. */
    for (unsigned pattern = 0; pattern < 2; pattern++) {
        fixture f = {0};
        CHECK(initialize(&f, s, pattern));
        f.comp_f16 = pattern;
        CHECK(setenv(disable_cache, "1", 1) == 0);
        CHECK(encode(&f, REF));
        CHECK(unsetenv(disable_cache) == 0);
        CHECK(ds4_gpu_begin_commands());
        CHECK(encode(&f, CACHED) && encode(&f, CACHED));
        /* Release while a CB is open must leave its users intact. */
        ds4_gpu_release_zero_prefix_prefill_mask_cache();
        CHECK(ds4_gpu_end_commands() && compare(&f));
        destroy(&f);
    }
    return 1;
}

static int queued_replacements(void) {
    const shape shapes[] = {
        {7,1,0,0,128,0,0}, {7,1,0,0,128,0,0}, {8,1,0,0,128,0,0},
        {65,2,0,0,128,0,0}, {7,1,0,0,128,0,0}, {65,2,0,0,128,0,0},
        {63,0,0,0,128,0,0}, {64,0,0,0,128,0,0}, {63,0,0,0,128,0,0},
    };
    enum { N = sizeof(shapes) / sizeof(*shapes) };
    fixture f[N];
    memset(f, 0, sizeof(f));
    ds4_gpu_release_zero_prefix_prefill_mask_cache();
    CHECK(setenv(disable_cache, "1", 1) == 0);
    for (unsigned i = 0; i < N; i++) {
        CHECK(initialize(&f[i], shapes[i], i + 7));
        f[i].comp_f16 = i & 1u;
        CHECK(encode(&f[i], REF));
    }
    CHECK(unsetenv(disable_cache) == 0);
    CHECK(ds4_gpu_begin_commands());
    for (unsigned i = 0; i < N; i++) {
        CHECK(encode(&f[i], CACHED));
        if (i == 1 || i == 4) CHECK(ds4_gpu_flush_commands());
        ds4_gpu_release_zero_prefix_prefill_mask_cache();
    }
    CHECK(ds4_gpu_end_commands());
    for (unsigned i = 0; i < N; i++) {
        CHECK(compare(&f[i]));
        destroy(&f[i]);
    }
    ds4_gpu_release_zero_prefix_prefill_mask_cache();
    return 1;
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int benchmark(void) {
    enum { SAMPLES = 7 };
    fixture f[3] = {0};
    const uint32_t ratios[] = {0,2,1}, layers[] = {2,18,20};
    double samples[2][SAMPLES];
    for (unsigned i = 0; i < 3; i++)
        CHECK(initialize(&f[i], (shape){437,ratios[i],0,0,128,0,0}, i));
    for (unsigned sample = 0; sample < SAMPLES + 1; sample++) {
        double elapsed[2] = {0,0};
        for (unsigned turn = 0; turn < 4; turn++) {
            const unsigned cached = (turn == 1 || turn == 2) ^ (sample & 1u);
            if (cached) CHECK(unsetenv(disable_cache) == 0);
            else CHECK(setenv(disable_cache, "1", 1) == 0);
            ds4_gpu_release_zero_prefix_prefill_mask_cache();
            const double start = seconds();
            for (unsigned i = 0; i < 3; i++) for (unsigned il = 0; il < layers[i]; il++) {
                CHECK(ds4_gpu_begin_commands());
                CHECK(encode(&f[i], cached ? CACHED : REF));
                CHECK(ds4_gpu_end_commands());
            }
            elapsed[cached] += seconds() - start;
        }
        if (sample) for (unsigned mode = 0; mode < 2; mode++)
            samples[mode][sample - 1] = elapsed[mode] * 500.0;
    }
    for (unsigned mode = 0; mode < 2; mode++)
        qsort(samples[mode], SAMPLES, sizeof(double), cmp_double);
    fprintf(stderr, "V4.1 masks 437 rows, 2/18/20 layers median wall ms: "
        "uncached=%.3f cached=%.3f (alternating ABBA, %u samples)\n",
        samples[0][SAMPLES/2], samples[1][SAMPLES/2], SAMPLES);
    for (unsigned i = 0; i < 3; i++) { CHECK(compare(&f[i])); destroy(&f[i]); }
    CHECK(unsetenv(disable_cache) == 0);
    ds4_gpu_release_zero_prefix_prefill_mask_cache();
    return 1;
}

static int check_all(int bench) {
    model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(model != MAP_FAILED);
    float *sinks = (float *)((char *)model + 128);
    for (unsigned i = 0; i < HEADS; i++) sinks[i] = ((int)i - 32) / 16.0f;
    CHECK(ds4_gpu_set_model_map(model, MODEL_BYTES));
    const uint32_t rows[] = {2,7,8,63,64,65,127,128,129,437};
    const uint32_t ratios[] = {0,2,1};
    for (unsigned i = 0; i < sizeof(rows)/sizeof(*rows); i++)
        for (unsigned j = 0; j < 3; j++)
            CHECK(parity((shape){rows[i],ratios[j],0,0,128,0,0}));
    /* Key changes and rejected admission must not reuse a preceding mask. */
    const shape edges[] = {
        {1,0,0,0,128,0,0}, {511,1,0,0,128,0,0}, {512,1,0,0,128,0,0},
        {1023,2,0,0,128,0,0}, {1024,2,0,0,128,0,0},
        {65,1,0,0,128,0,DYNAMIC}, {65,2,0,0,128,0,DYNAMIC},
        {65,0,0,0,128,0,NONCAUSAL},
        {65,0,0,0,128,0,VISUAL}, {65,1,0,0,128,0,VISUAL},
        {65,0,0,0,64,0,0}, {65,1,0,0,64,0,0},
        {65,0,0,0,128,79,0}, {65,1,0,0,128,79,0},
        {65,0,127,127,128,185,0}, {65,2,127,127,128,185,0},
        {64,2,1,0,128,0,0},
        {65,4,0,0,128,0,0},
    };
    for (unsigned i = 0; i < sizeof(edges)/sizeof(*edges); i++) CHECK(parity(edges[i]));
    CHECK(queued_replacements());
    fprintf(stderr, "V4.1 mask cache: bitwise cold/warm, changed Q/KV/dtype, "
        "tails, append/ring/dynamic/visual fallbacks, queued replacement, canaries PASS\n");
    if (bench) CHECK(benchmark());
    ds4_gpu_release_zero_prefix_prefill_mask_cache();
    ds4_gpu_cleanup();
    CHECK(munmap(model, MODEL_BYTES) == 0);
    return 1;
}

int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--bench"))) {
        fprintf(stderr, "usage: %s [--bench]\n", argv[0]);
        return 1;
    }
    if (unsetenv(disable_cache) || unsetenv("DS4_METAL_FLASH_ATTN_STAGE_PROFILE")) return 1;
    return ds4_gpu_init() && check_all(argc == 2) ? 0 : 1;
}
