/* Exercise the real graph candidate bypass without a model or projections. */
#include "../ds4.c"

#define CHECK(v) do { if (!(v)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #v); exit(1); \
} } while (0)
enum { KEYS = 16449, BLOCKS = (KEYS + 7) / 8, TOP = 512, PAD = 4, FIELDS = 5 };
static const uint32_t widths[FIELDS] = {KEYS, BLOCKS, TOP, BLOCKS, 2048};
typedef struct {
    ds41_gpu_graph graph;
    ds4_gpu_tensor *storage[FIELDS], *view[FIELDS];
} fixture;

static void fixture_init(fixture *f) {
    memset(f, 0, sizeof(*f));
    for (unsigned i = 0; i < FIELDS; i++) {
        f->storage[i] = ds4_gpu_tensor_alloc((widths[i] + 2u * PAD) * sizeof(float));
        CHECK(f->storage[i] && ds4_gpu_tensor_contents(f->storage[i]));
        memset(ds4_gpu_tensor_contents(f->storage[i]), 0x35,
               (widths[i] + 2u * PAD) * sizeof(float));
        f->view[i] = ds4_gpu_tensor_view(f->storage[i], PAD * sizeof(float), widths[i] * sizeof(float));
        CHECK(f->view[i]);
    }
    f->graph.index_scores = f->view[0]; f->graph.block_mask = f->view[1];
    f->graph.selected_comp = f->view[2]; f->graph.block_scores = f->view[3];
    f->graph.block_selected = f->view[4];
    ds41_graph_reset(&f->graph);
}

static void fixture_free(fixture *f) {
    for (unsigned i = 0; i < FIELDS; i++) {
        ds4_gpu_tensor_free(f->view[i]);
        ds4_gpu_tensor_free(f->storage[i]);
    }
}

/* The sequence before the bypass: keep its real sort, tie ordering and mask
 * scatter as the oracle, including the forced most recent candidate block. */
static bool legacy_pick(ds41_gpu_graph *g, uint32_t il) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t n = ratio ? (g->pos + 1u) / ratio : 0u;
    if (!n || !ds41_index_source(il)) return true;
    const uint32_t blocks = (n + 7u) / 8u;
    if (il == 20u) {
        const uint32_t top = blocks < 2048u ? blocks : 2048u;
        if (!ds4_gpu_dsv41_candidate_blocks(g->block_scores, g->index_scores, n, 1, g->pos, ratio) ||
            !ds4_gpu_indexer_topk_tensor(g->block_selected, g->block_scores, blocks, 1, top) ||
            !ds4_gpu_dsv4_topk_mask_tensor(g->block_mask, g->block_selected, blocks, 1, top)) return false;
    } else if (il > 20u &&
        !ds4_gpu_dsv41_candidate_filter(g->index_scores, g->block_mask, n, 1, g->pos, ratio)) return false;
    /* The first cold layer-20 call grows scratch here while its block sort
     * is still queued; the old allocation must survive this key sort. */
    return ds4_gpu_indexer_topk_tensor(g->selected_comp, g->index_scores, n, 1, n < TOP ? n : TOP);
}

static void stage(fixture *fast, fixture *reference, ds4_gpu_tensor *scores,
                  ds4_gpu_tensor *poison, uint32_t il, uint32_t pos,
                  uint32_t pattern, bool restore) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t n = ratio ? (pos + 1u) / ratio : 0u;
    CHECK(n <= KEYS);
    float *s = ds4_gpu_tensor_contents(scores);
    for (uint32_t i = 0; i < KEYS; i++) {
        /* Future entries are deliberately attractive; their physical tail
         * must not enter selection or get overwritten by the candidate path. */
        s[i] = i >= n ? 123456.0f : pattern == 0u ?
            (float)((i * 7919u + il * 101u) % 65521u) - 32768.0f :
            pattern == 1u ? (i % 3u ? 0.0f : -0.0f) : -INFINITY;
    }
    if (n && pattern == 1u) s[n / 2u] = INFINITY;
    uint32_t previous_mask[BLOCKS];
    memcpy(previous_mask, ds4_gpu_tensor_contents(restore ? poison : fast->view[1]), sizeof(previous_mask));
    uint32_t scratch_before[BLOCKS + 2048];
    memcpy(scratch_before, ds4_gpu_tensor_contents(fast->view[3]), BLOCKS * 4u);
    memcpy(scratch_before + BLOCKS, ds4_gpu_tensor_contents(fast->view[4]), 2048u * 4u);
    fast->graph.pos = reference->graph.pos = pos;
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds4_gpu_tensor_copy(fast->view[0], 0, scores, 0, KEYS * 4u));
    CHECK(ds4_gpu_tensor_copy(reference->view[0], 0, scores, 0, KEYS * 4u));
    if (restore) {
        /* A CPU fill here would race with this queued carry restoration. */
        CHECK(ds4_gpu_tensor_copy(fast->view[1], 0, poison, 0, BLOCKS * 4u));
        CHECK(ds4_gpu_tensor_copy(reference->view[1], 0, poison, 0, BLOCKS * 4u));
    }
    CHECK(legacy_pick(&reference->graph, il));
    /* The reference leaves a compute encoder open before the fill dispatch. */
    CHECK(ds41_attention_pick(&fast->graph, il));
    CHECK(ds4_gpu_end_commands());
    for (unsigned field = 0; field < 3; field++)
        CHECK(!memcmp(ds4_gpu_tensor_contents(fast->storage[field]),
                      ds4_gpu_tensor_contents(reference->storage[field]),
                      (widths[field] + 2u * PAD) * 4u));
    for (unsigned field = 0; field < FIELDS; field++) {
        const uint32_t *p = ds4_gpu_tensor_contents(fast->storage[field]);
        for (unsigned i = 0; i < PAD; i++)
            CHECK(p[i] == 0x35353535u && p[PAD + widths[field] + i] == 0x35353535u);
    }
    const uint32_t blocks = (n + 7u) / 8u;
    const uint32_t *mask = ds4_gpu_tensor_contents(fast->view[1]);
    CHECK(!memcmp(mask + blocks, previous_mask + blocks, (BLOCKS - blocks) * 4u));
    if (n && il >= 20u && n <= 16384u) {
        for (uint32_t i = 0; i < blocks; i++) CHECK(mask[i] == 0u);
        CHECK(!memcmp(scratch_before, ds4_gpu_tensor_contents(fast->view[3]), BLOCKS * 4u));
        CHECK(!memcmp(scratch_before + BLOCKS, ds4_gpu_tensor_contents(fast->view[4]), 2048u * 4u));
    }
    if (n && ds41_index_source(il)) {
        const int32_t *ids = ds4_gpu_tensor_contents(fast->view[2]);
        for (uint32_t i = 0; i < (n < TOP ? n : TOP); i++)
            CHECK(ids[i] >= 0 && (uint32_t)ids[i] < n);
    }
}

static void fill_contract(fixture *f) {
    uint32_t before[BLOCKS + 2 * PAD];
    memcpy(before, ds4_gpu_tensor_contents(f->storage[1]), sizeof(before));
    ds4_gpu_tensor *short_view = ds4_gpu_tensor_view(f->storage[1], PAD * 4u, 2048u * 4u - 1u);
    ds4_gpu_tensor *unaligned = ds4_gpu_tensor_view(f->storage[1], 2u, BLOCKS * 4u);
    CHECK(short_view && unaligned);
    CHECK(!ds4_gpu_dsv41_candidate_mask_all(NULL, 1));
    CHECK(!ds4_gpu_dsv41_candidate_mask_all(f->view[1], 0));
    CHECK(!ds4_gpu_dsv41_candidate_mask_all(f->view[1], 16385));
    CHECK(!ds4_gpu_dsv41_candidate_mask_all(f->view[1], UINT32_MAX));
    CHECK(!ds4_gpu_dsv41_candidate_mask_all(short_view, 16384));
    CHECK(!ds4_gpu_dsv41_candidate_mask_all(unaligned, 1));
    CHECK(!memcmp(before, ds4_gpu_tensor_contents(f->storage[1]), sizeof(before)));
    ds4_gpu_tensor_free(unaligned); ds4_gpu_tensor_free(short_view);
    for (unsigned queued = 0; queued < 2; queued++) {
        memcpy(ds4_gpu_tensor_contents(f->storage[1]), before, sizeof(before));
        ds4_gpu_tensor *view = ds4_gpu_tensor_view(f->storage[1], PAD * 4u, BLOCKS * 4u);
        CHECK(view);
        if (queued) CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_dsv41_candidate_mask_all(view, 16384));
        ds4_gpu_tensor_free(view);
        /* The fill is the only work: it must prevent the empty-CB shortcut. */
        if (queued) CHECK(ds4_gpu_end_commands());
        const uint32_t *p = ds4_gpu_tensor_contents(f->storage[1]);
        for (unsigned i = 0; i < BLOCKS + 2u * PAD; i++)
            CHECK(p[i] == (i >= PAD && i < PAD + 2048u ? 0u : before[i]));
    }
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_FLASH41;
    for (uint32_t i = 0; i < DS4_N_LAYER; i++)
        g_ds4_compress_ratios[i] = ds4_expected_layer_compress_ratio(i);
    CHECK(ds4_gpu_init());
    fixture fast, reference;
    fixture_init(&fast); fixture_init(&reference);
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(KEYS * 4u);
    ds4_gpu_tensor *poison = ds4_gpu_tensor_alloc(BLOCKS * 4u);
    CHECK(scores && poison && ds4_gpu_tensor_contents(scores) && ds4_gpu_tensor_contents(poison));
    float *mask = ds4_gpu_tensor_contents(poison);
    for (uint32_t i = 0; i < BLOCKS; i++) mask[i] = i & 1u ? -INFINITY : 1.0f;
    /* Start cold above the bypass: block top-k then key top-k grows the
     * shared scratch inside one CB. Never warm it away; this is also a
     * regression for old-buffer liveness under unretained command buffers. */
    const uint32_t counts[] = {16385, 1, 7, 8, 9, 127, 511, 512, 513,
                               16376, 16377, 16383, 16384, 16385, KEYS, 1};
    const uint32_t layers[] = {20, 21, 24, 25, 28, 29, 32, 33, 36, 39};
    for (uint32_t pattern = 0; pattern < 3; pattern++) {
        ds4_gpu_set_quality(pattern == 1u);
        ds4_gpu_set_ssd_streaming(pattern == 2u);
        for (unsigned ci = 0; ci < sizeof(counts) / sizeof(*counts); ci++) {
            const uint32_t n = counts[ci];
            if (n == 1u) {
                /* Reset does not clear the old physical mask. */
                ds41_graph_reset(&fast.graph); ds41_graph_reset(&reference.graph);
            }
            for (unsigned li = 0; li < sizeof(layers) / sizeof(*layers); li++)
                stage(&fast, &reference, scores, poison, layers[li], n - 1u,
                      pattern, !li && n != 1u);
        }
        stage(&fast, &reference, scores, poison, 0, 0, pattern, false);
        stage(&fast, &reference, scores, poison, 2, 0, pattern, false);
        /* Ratio-two Full/Reuse keep their existing path on both pair phases. */
        for (uint32_t phase = 0; phase < 2; phase++) {
            stage(&fast, &reference, scores, poison, 2, 1023u + phase, pattern, false);
            stage(&fast, &reference, scores, poison, 3, 1023u + phase, pattern, false);
        }
        fprintf(stderr, "V4.1 candidate bypass pattern=%u: exact masks/scores/IDs, reset, causal tails and guards PASS\n", pattern);
    }
    fill_contract(&fast);
    ds4_gpu_tensor_free(poison); ds4_gpu_tensor_free(scores);
    fixture_free(&reference); fixture_free(&fast);
    ds4_gpu_cleanup();
    puts("V4.1 candidate bypass: graph parity and ordered GPU mask publication PASS");
    return 0;
}
