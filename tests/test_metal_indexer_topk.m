// SPDX-License-Identifier: MIT
// Test the production indexer API, including its encoder/scratch lifetime.
// The legacy bitonic leaf is not stable on ties: tied IDs must match the
// legacy GPU path, while the CPU oracle independently checks top-k validity.
#define DS4_METAL_INDEXER_TOPK_TESTING 1
#include "../ds4_metal.m"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
int ds4_deepseek4_attention_bounds(const int *a, uint32_t b, uint32_t c,
        uint32_t d, uint32_t e, uint32_t f, uint32_t *g) {
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
    return 0;
}

enum { TK_GUARD = 256, TK_SLACK = 36, TK_SAMPLES = 30, TK_WARMUP_BLOCKS = 4 };
static const uint32_t tk_poison = 0xdead715eu;
static unsigned tk_cases;
static unsigned tk_compact_cases;
typedef struct {
    ds4_gpu_tensor *base, *view;
    NSUInteger payload, offset, total;
} tk_tensor;
typedef struct { float score; uint32_t id; } tk_rank;
typedef struct { uint32_t n, tokens, k; } tk_shape;
typedef struct { double wall, gpu; } tk_time;

static void tk_require(bool ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "Metal indexer top-k FAIL: %s\n", message);
        exit(1);
    }
}
static uint32_t tk_random(uint32_t *state) {
    return *state = *state * 1664525u + 1013904223u;
}
static uint64_t tk_hash(const void *data, size_t bytes) {
    const uint8_t *p = data;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; ++i)
        h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static void tk_reset(tk_tensor *t) {
    uint32_t *p = ds4_gpu_tensor_contents(t->base);
    for (NSUInteger i = 0; i < t->total / 4; ++i) p[i] = tk_poison;
}
static tk_tensor tk_alloc(NSUInteger payload, unsigned offset_words) {
    tk_require(payload <= NSUIntegerMax - 2u*TK_GUARD - TK_SLACK - 32u,
               "guarded allocation overflow");
    tk_tensor t = {.payload=payload, .offset=TK_GUARD + 4u*offset_words};
    t.total = t.offset + payload + TK_SLACK + TK_GUARD;
    t.base = ds4_gpu_tensor_alloc(t.total);
    t.view = t.base ? ds4_gpu_tensor_view(t.base, t.offset, payload + TK_SLACK) : NULL;
    tk_require(t.base && t.view, "guarded tensor allocation");
    tk_reset(&t);
    return t;
}
static void tk_guards(const tk_tensor *t) {
    const uint32_t *p = ds4_gpu_tensor_contents(t->base);
    for (NSUInteger i = 0; i < t->offset/4u; ++i)
        tk_require(p[i] == tk_poison, "tensor prefix modified");
    for (NSUInteger i = (t->offset + t->payload)/4u; i < t->total/4u; ++i)
        tk_require(p[i] == tk_poison, "tensor suffix or inactive payload modified");
}
static void tk_free(tk_tensor *t) {
    ds4_gpu_tensor_free(t->view);
    ds4_gpu_tensor_free(t->base);
    *t = (tk_tensor){0};
}
static int tk_rank_compare(const void *av, const void *bv) {
    const tk_rank *a = av, *b = bv;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    return a->id < b->id ? -1 : a->id > b->id;
}
static void tk_fill(tk_tensor *scores, tk_shape s, unsigned pattern) {
    float *p = ds4_gpu_tensor_contents(scores->view);
    uint32_t seed = 137u + s.n + 73u*s.tokens + pattern;
    for (uint32_t row = 0; row < s.tokens; ++row) {
        float *r = p + (size_t)row*s.n;
        for (uint32_t i = 0; i < s.n; ++i) {
            switch (pattern) {
                case 0: r[i] = (float)i - (float)s.n/2.f; break;
                case 1: r[i] = (float)((int)(tk_random(&seed) % 11u) - 5); break;
                case 2: {
                    // Include rows with fewer than K visible entries, so valid
                    // causal -infinity IDs must still fill the selected row.
                    const uint32_t visible = (uint32_t)(((uint64_t)row*997u + s.k/2u) % (s.n + 1ull));
                    r[i] = i < visible ? (float)((i*17u) % 113u) : -INFINITY;
                    break;
                }
                case 3: r[i] = i & 1u ? -0.f : 0.f; break;
                case 4: r[i] = -INFINITY; break;
                default: r[i] = i % 17u == 0 ? INFINITY :
                               i % 7u == 0 ? -INFINITY : -7.f; break;
            }
        }
        if (pattern == 0) {
            for (uint32_t i = s.n; i > 1u; --i) {
                const uint32_t j = tk_random(&seed) % i;
                const float tmp = r[i-1]; r[i-1] = r[j]; r[j] = tmp;
            }
        }
    }
}
static void tk_oracle(const tk_tensor *scores, const tk_tensor *selected,
                      tk_shape s, bool unique) {
    tk_rank *rank = malloc((size_t)s.n*sizeof(*rank));
    uint8_t *seen = malloc(s.n);
    tk_require(rank && seen, "CPU oracle allocation");
    const float *x = ds4_gpu_tensor_contents(scores->view);
    const uint32_t *y = ds4_gpu_tensor_contents(selected->view);
    for (uint32_t row = 0; row < s.tokens; ++row) {
        const float *r = x + (size_t)row*s.n;
        const uint32_t *out = y + (size_t)row*s.k;
        for (uint32_t i = 0; i < s.n; ++i) rank[i] = (tk_rank){r[i], i};
        qsort(rank, s.n, sizeof(*rank), tk_rank_compare);
        memset(seen, 0, s.n);
        for (uint32_t k = 0; k < s.k; ++k) {
            if (out[k] >= s.n || seen[out[k]] ||
                r[out[k]] != rank[k].score || (unique && out[k] != rank[k].id)) {
                fprintf(stderr, "n=%u tokens=%u K=%u row=%u rank=%u actual=%u expected_score=%g\n",
                        s.n, s.tokens, s.k, row, k, out[k], rank[k].score);
                tk_require(false, "CPU top-k order, threshold, uniqueness or unique-score permutation");
            }
            seen[out[k]] = 1;
        }
    }
    free(seen); free(rank);
}
static bool tk_expect_compact(tk_shape s) {
    NSUInteger max_threads = g_argsort_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup;
    if (!max_threads) max_threads = 256;
    uint32_t threads = 1;
    while (threads < s.n && 2ull*threads <= max_threads && threads < 1024)
        threads *= 2;
    ds4_indexer_topk_geometry geometry;
    tk_require(ds4_indexer_topk_initial(s.n, s.tokens, s.k, threads, &geometry),
               "test shape has representable production geometry");
    return ds4_indexer_topk_compaction_useful(geometry, s.k);
}
static void tk_dispatch(tk_tensor *out, const tk_tensor *scores, tk_shape s,
                        bool legacy, bool batch) {
    const uint64_t before = g_indexer_topk_compact_launches;
    const bool expected_compact = !legacy && tk_expect_compact(s);
    ds4_gpu_test_set_flags(legacy ? DS4_GPU_TEST_INDEXER_TOPK_LEGACY : 0u);
    if (batch) tk_require(ds4_gpu_begin_commands(), "begin top-k batch");
    tk_require(ds4_gpu_indexer_topk_tensor(out->view, scores->view,
                s.n, s.tokens, s.k), "production top-k dispatch");
    if (batch) tk_require(ds4_gpu_end_commands(), "finish top-k batch");
    tk_require(g_indexer_topk_compact_launches == before + expected_compact,
               "production branch differs from compact geometry or legacy flag");
    tk_compact_cases += expected_compact;
    ds4_gpu_test_set_flags(0u);
}
static void tk_check_pair(const tk_tensor *scores, const tk_tensor *a,
                          const tk_tensor *b, tk_shape s, unsigned pattern,
                          uint64_t input_hash) {
    tk_guards(a); tk_guards(b); tk_guards(scores);
    tk_require(input_hash == tk_hash(ds4_gpu_tensor_contents(scores->base), scores->total),
               "input scores or guards changed");
    if (memcmp(ds4_gpu_tensor_contents(a->view), ds4_gpu_tensor_contents(b->view), a->payload)) {
        fprintf(stderr, "n=%u tokens=%u K=%u pattern=%u\n", s.n, s.tokens, s.k, pattern);
        tk_require(false, "candidate must preserve exact legacy IDs, including tied scores");
    }
    tk_oracle(scores, a, s, pattern == 0);
    tk_oracle(scores, b, s, pattern == 0);
}
static void tk_case(tk_shape s, unsigned pattern) {
    @autoreleasepool {
        tk_tensor scores = tk_alloc((NSUInteger)s.n*s.tokens*4u, 1u + tk_cases%3u);
        tk_tensor legacy = tk_alloc((NSUInteger)s.k*s.tokens*4u, 4u);
        tk_tensor actual = tk_alloc(legacy.payload, 7u);
        tk_fill(&scores, s, pattern);
        const uint64_t input_hash = tk_hash(ds4_gpu_tensor_contents(scores.base), scores.total);
        tk_dispatch(&legacy, &scores, s, true, tk_cases%2u == 0);
        tk_dispatch(&actual, &scores, s, false, tk_cases%2u != 0);
        tk_check_pair(&scores, &legacy, &actual, s, pattern, input_hash);
        ++tk_cases;
        tk_free(&actual); tk_free(&legacy); tk_free(&scores);
    }
}
static void tk_invalid(void) {
    tk_tensor scores = tk_alloc(128u, 1), out = tk_alloc(128u, 3);
    ds4_gpu_tensor *overlap = ds4_gpu_tensor_view(scores.view, 4, 16);
    ds4_gpu_tensor *unaligned_scores = ds4_gpu_tensor_view(scores.view, 1, 16);
    ds4_gpu_tensor *unaligned_out = ds4_gpu_tensor_view(out.view, 1, 16);
    ds4_gpu_tensor *short_scores = ds4_gpu_tensor_view(scores.view, 0, 4);
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_view(out.view, 0, 4);
    tk_require(overlap && unaligned_scores && unaligned_out && short_scores && short_out,
               "invalid-input tensor views");
    const uint64_t sh = tk_hash(ds4_gpu_tensor_contents(scores.base), scores.total);
    const uint64_t oh = tk_hash(ds4_gpu_tensor_contents(out.base), out.total);
    for (unsigned legacy = 0; legacy < 2; ++legacy) {
        const uint64_t before = g_indexer_topk_compact_launches;
        ds4_gpu_test_set_flags(legacy ? DS4_GPU_TEST_INDEXER_TOPK_LEGACY : 0u);
        tk_require(!ds4_gpu_indexer_topk_tensor(NULL, scores.view, 1, 1, 1), "null output admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, NULL, 1, 1, 1), "null scores admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, scores.view, 0, 1, 1), "zero columns admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, scores.view, 1, 0, 1), "zero rows admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, scores.view, 1, 1, 0), "zero K admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, scores.view, 1, 1, 2), "K greater than columns admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, scores.view, 1024, 1, 1), "undersized scores admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, scores.view, 32, 2, 32), "undersized tensors admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, scores.view, UINT32_MAX, 1, 1), "unsigned column overflow admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, scores.view, 1, UINT32_MAX, 1), "unsigned row overflow admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(scores.view, scores.view, 8, 1, 4), "identical score/output tensor admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(overlap, scores.view, 8, 1, 4), "overlapping tensor views admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, unaligned_scores, 2, 1, 2), "unaligned score offset admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(unaligned_out, scores.view, 2, 1, 2), "unaligned selected offset admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(short_out, scores.view, 8, 1, 4), "undersized selected tensor admitted");
        tk_require(!ds4_gpu_indexer_topk_tensor(out.view, short_scores, 8, 1, 4), "short score view admitted");
        tk_require(g_indexer_topk_compact_launches == before && !g_batch_cb,
                   "rejected request must not dispatch or open a batch");
        tk_cases += 16;
    }
    ds4_gpu_test_set_flags(0u);
    tk_require(sh == tk_hash(ds4_gpu_tensor_contents(scores.base), scores.total) &&
               oh == tk_hash(ds4_gpu_tensor_contents(out.base), out.total),
               "rejected request modified data");
    ds4_gpu_tensor_free(short_out); ds4_gpu_tensor_free(short_scores);
    ds4_gpu_tensor_free(unaligned_out); ds4_gpu_tensor_free(unaligned_scores);
    ds4_gpu_tensor_free(overlap);
    tk_free(&out); tk_free(&scores);
}
static double tk_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec*1e3 + (double)t.tv_nsec/1e6;
}
static tk_time tk_timed(tk_tensor *out, const tk_tensor *scores, tk_shape s,
                        bool legacy, unsigned repeats) {
    @autoreleasepool {
        const uint64_t before = g_indexer_topk_compact_launches;
        const bool expected_compact = !legacy && tk_expect_compact(s);
        ds4_gpu_test_set_flags(legacy ? DS4_GPU_TEST_INDEXER_TOPK_LEGACY : 0u);
        const double start = tk_now();
        tk_require(ds4_gpu_begin_commands(), "begin timed batch");
        id<MTLCommandBuffer> cb = g_batch_cb;
        for (unsigned i = 0; i < repeats; ++i)
            tk_require(ds4_gpu_indexer_topk_tensor(out->view, scores->view,
                       s.n, s.tokens, s.k), "timed full-API dispatch");
        tk_require(ds4_gpu_end_commands(), "finish timed batch");
        const double wall = (tk_now() - start)/repeats;
        const double gpu = 1e3*(cb.GPUEndTime - cb.GPUStartTime)/repeats;
        tk_require(isfinite(wall) && wall > 0 && isfinite(gpu) && gpu > 0,
                   "valid completed-command-buffer timing");
        tk_require(g_indexer_topk_compact_launches == before + repeats*expected_compact,
                   "timing must execute the intended branch on every API call");
        ds4_gpu_test_set_flags(0u);
        return (tk_time){wall, gpu};
    }
}
static int tk_double_compare(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double tk_statistics(const tk_time *times, bool gpu, const char *label) {
    double values[TK_SAMPLES], sum = 0;
    printf("  %s raw_ms=", label);
    for (unsigned i = 0; i < TK_SAMPLES; ++i) {
        values[i] = gpu ? times[i].gpu : times[i].wall;
        sum += values[i];
        printf("%s%.6f", i ? "," : "", values[i]);
    }
    qsort(values, TK_SAMPLES, sizeof(*values), tk_double_compare);
    const double median = (values[TK_SAMPLES/2-1] + values[TK_SAMPLES/2])/2;
    printf(" mean=%.6f median=%.6f range=%.6f..%.6f\n",
           sum/TK_SAMPLES, median, values[0], values[TK_SAMPLES-1]);
    return median;
}
static void tk_bench(tk_shape s) {
    @autoreleasepool {
        tk_tensor scores = tk_alloc((NSUInteger)s.n*s.tokens*4u, 1);
        tk_tensor legacy = tk_alloc((NSUInteger)s.k*s.tokens*4u, 4);
        tk_tensor actual = tk_alloc(legacy.payload, 7);
        tk_fill(&scores, s, 1);
        const uint64_t hash = tk_hash(ds4_gpu_tensor_contents(scores.base), scores.total);
        const unsigned repeats = s.tokens == 1 ? 64u : 8u;
        tk_time a[TK_SAMPLES], b[TK_SAMPLES];
        // Warm both pipelines and shared scratch with the measured ABBA order.
        for (unsigned block = 0; block < TK_WARMUP_BLOCKS; ++block) {
            (void)tk_timed(&legacy, &scores, s, true, repeats);
            (void)tk_timed(&actual, &scores, s, false, repeats);
            (void)tk_timed(&actual, &scores, s, false, repeats);
            (void)tk_timed(&legacy, &scores, s, true, repeats);
        }
        tk_check_pair(&scores, &legacy, &actual, s, 1, hash);
        for (unsigned block = 0; block < TK_SAMPLES/2; ++block) {
            a[2*block] = tk_timed(&legacy, &scores, s, true, repeats);
            b[2*block] = tk_timed(&actual, &scores, s, false, repeats);
            b[2*block+1] = tk_timed(&actual, &scores, s, false, repeats);
            a[2*block+1] = tk_timed(&legacy, &scores, s, true, repeats);
        }
        tk_check_pair(&scores, &legacy, &actual, s, 1, hash);
        printf("BENCH ncomp=%u ntok=%u K=%u repeats_per_CB=%u samples_per_arm=%u warmup_ABBA_blocks=%u warmup_CBs_per_arm=%u\n",
               s.n, s.tokens, s.k, repeats, TK_SAMPLES, TK_WARMUP_BLOCKS, 2u*TK_WARMUP_BLOCKS);
        const double aw = tk_statistics(a, false, "legacy wall");
        const double bw = tk_statistics(b, false, "compact wall");
        const double ag = tk_statistics(a, true, "legacy GPU");
        const double bg = tk_statistics(b, true, "compact GPU");
        printf("  median throughput gain wall=%+.3f%% GPU=%+.3f%%\n",
               100*(aw/bw-1), 100*(ag/bg-1));
        fflush(stdout);
        tk_free(&actual); tk_free(&legacy); tk_free(&scores);
    }
}
int main(int argc, char **argv) {
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--bench"))) {
        fprintf(stderr, "Usage: %s [--bench]\n", argv[0]); return 2;
    }
    @autoreleasepool {
        tk_require(ds4_gpu_init(), "Metal initialization");
        printf("Device: %s; argsort max threads=%lu\n", g_device.name.UTF8String,
               (unsigned long)g_argsort_f32_i32_desc_pipeline.maxTotalThreadsPerThreadgroup);
        const tk_shape shapes[] = {
            {1,1,1}, {2,3,1}, {3,3,3}, {6,3,6}, {7,3,6},
            {31,3,7}, {32,3,32}, {33,3,31}, {255,3,6}, {256,3,1},
            {257,3,257}, {511,3,31}, {512,3,512}, {513,3,512},
            {1023,3,6}, {1024,3,512}, {1025,3,512}, {2047,3,7},
            {2048,3,1025}, {2049,3,1025}, {4095,3,512}, {4096,3,512},
            {4097,3,512}, {16385,1,512}, {65536,1,512},
            {4097,128,512}, {2049,3,2049}, {4097,3,1},
        };
        tk_invalid();
        for (unsigned i = 0; i < sizeof(shapes)/sizeof(shapes[0]); ++i)
            for (unsigned pattern = 0; pattern < 6; ++pattern) tk_case(shapes[i], pattern);
        tk_require(tk_compact_cases > 0, "corpus must execute compact merge");
        printf("PASS: %u native indexer top-k cases (%u compact); exact legacy IDs, independent CPU top-k, dispatch counters, guards and immutable scores.\n", tk_cases, tk_compact_cases);
        fflush(stdout);
        if (argc == 2) {
            puts("Warm fixed-input full top-k API ABBA timing, including command-buffer encoding/submission/completion and every sort/merge pass; excludes score production, allocations and model execution. GPU span is also reported. This is not end-to-end model throughput.");
            const uint32_t columns[] = {4096, 16384, 65536};
            for (unsigned i = 0; i < 3; ++i)
                for (unsigned t = 0; t < 2; ++t)
                    tk_bench((tk_shape){columns[i], t ? 128u : 1u, 512});
        }
        ds4_gpu_cleanup();
    }
    return 0;
}
