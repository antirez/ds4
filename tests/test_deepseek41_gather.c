/* Real graph gather helper versus unconditional gathering, without a model. */
#include "../ds4.c"

#define CHECK(v) do { if (!(v)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #v); exit(1); \
} } while (0)
enum { ROWS = 1030, DIM = 512, TOP = 512, PAD = 4, BUFFERS = 7 };
static ds41_gpu_graph graph;
static ds4_gpu_tensor *storage[BUFFERS], *view[BUFFERS];
static uint64_t counts[BUFFERS];
static uint32_t expected_ids[TOP];

static void set_ids(uint32_t n, uint32_t salt) {
    if (!n) return;
    for (uint32_t i = 0; i < (n < TOP ? n : TOP); i++)
        expected_ids[i] = (n - 1u - i + salt) % n;
    CHECK(ds4_gpu_tensor_write(graph.selected_comp, 0, expected_ids, (n < TOP ? n : TOP) * 4u));
}

static void check_output(void) {
    CHECK(!memcmp(ds4_gpu_tensor_contents(storage[5]), ds4_gpu_tensor_contents(storage[6]),
                  (TOP * DIM + 2u * PAD) * 4u));
    for (unsigned j = 0; j < BUFFERS; j++) {
        const uint32_t *p = ds4_gpu_tensor_contents(storage[j]);
        for (unsigned i = 0; i < PAD; i++)
            CHECK(p[i] == 0x35353535u && p[PAD + counts[j] + i] == 0x35353535u);
    }
}

static void run_layer(uint32_t il, uint32_t pos, bool projected, bool change_ids,
                       uint32_t salt) {
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const uint32_t n = ratio ? (pos + 1u) / ratio : 0u;
    const uint32_t owner = il < 8u ? 0u : il < 14u ? 1u : il < 20u ? 2u : 3u;
    graph.pos = pos;
    CHECK(n <= ROWS);
    if (change_ids) set_ids(n, salt);
    CHECK(ds4_gpu_begin_commands());
    CHECK(ds41_attention_gather(&graph, il, n, projected));
    if (n) CHECK(ds4_gpu_dsv41_gather_kv(view[6], graph.compressed[owner],
                                       graph.selected_comp, n, n < TOP ? n : TOP));
    CHECK(ds4_gpu_end_commands());
    check_output();
    if (n) CHECK(!memcmp(expected_ids, ds4_gpu_tensor_contents(graph.selected_comp),
                         (n < TOP ? n : TOP) * 4u));
}

static uint64_t cache_hash(void) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (unsigned owner = 0; owner < 4; owner++) {
        const uint32_t *p = ds4_gpu_tensor_contents(view[owner]);
        for (uint64_t i = 0; i < counts[owner]; i++)
            hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static void queued_sources(void) {
    ds4_gpu_tensor *ids[2], *snapshot[2];
    uint32_t selected[2][TOP];
    for (unsigned generation = 0; generation < 2; generation++) {
        ids[generation] = ds4_gpu_tensor_alloc(TOP * 4u);
        snapshot[generation] = ds4_gpu_tensor_alloc(TOP * DIM * 4u);
        CHECK(ids[generation] && snapshot[generation]);
        for (unsigned i = 0; i < TOP; i++) selected[generation][i] = (i + generation * 137u) % ROWS;
        CHECK(ds4_gpu_tensor_write(ids[generation], 0, selected[generation], TOP * 4u));
    }
    CHECK(ds4_gpu_begin_commands());
    for (unsigned generation = 0; generation < 2; generation++) {
        CHECK(ds4_gpu_tensor_copy(graph.selected_comp, 0, ids[generation], 0, TOP * 4u));
        CHECK(ds41_attention_gather(&graph, 20u + generation * 4u, ROWS, false));
        CHECK(ds41_attention_gather(&graph, 21u + generation * 4u, ROWS, false));
        CHECK(ds4_gpu_tensor_copy(snapshot[generation], 0, graph.selected_kv, 0, TOP * DIM * 4u));
    }
    CHECK(ds4_gpu_end_commands());
    const uint32_t *cache = ds4_gpu_tensor_contents(graph.compressed[3]);
    for (unsigned generation = 0; generation < 2; generation++) {
        const uint32_t *actual = ds4_gpu_tensor_contents(snapshot[generation]);
        for (unsigned i = 0; i < TOP; i++)
            CHECK(!memcmp(actual + i * DIM, cache + selected[generation][i] * DIM, DIM * 4u));
        ds4_gpu_tensor_free(snapshot[generation]); ds4_gpu_tensor_free(ids[generation]);
    }
}

int main(void) {
    g_ds4_shape = DS4_SHAPE_FLASH41;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++)
        g_ds4_compress_ratios[il] = ds4_expected_layer_compress_ratio(il);
    CHECK(ds4_gpu_init());
    for (unsigned j = 0; j < BUFFERS; j++) {
        counts[j] = j < 4 ? ROWS * DIM : j == 4 ? TOP : TOP * DIM;
        storage[j] = ds4_gpu_tensor_alloc((counts[j] + 2u * PAD) * 4u);
        CHECK(storage[j] && ds4_gpu_tensor_contents(storage[j]));
        memset(ds4_gpu_tensor_contents(storage[j]), 0x35, (counts[j] + 2u * PAD) * 4u);
        view[j] = ds4_gpu_tensor_view(storage[j], PAD * 4u, counts[j] * 4u);
        CHECK(view[j]);
        if (j < 4) {
            graph.compressed[j] = view[j];
            uint32_t *p = ds4_gpu_tensor_contents(view[j]);
            for (uint64_t i = 0; i < counts[j]; i++) p[i] = 0x3f000000u + (uint32_t)i + j * 0x00100000u;
        }
    }
    graph.selected_comp = view[4]; graph.selected_kv = view[5];
    const uint64_t before = cache_hash();
    const uint32_t positions[] = {0, 1, 2, 511, 512, 1023, 1024, 1025, 0};
    ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_DECODE);
    for (unsigned mode = 0; mode < 3; mode++) {
        ds4_gpu_set_quality(mode == 1u);
        ds4_gpu_set_ssd_streaming(mode == 2u);
        for (unsigned p = 0; p < sizeof(positions) / sizeof(*positions); p++) {
            if (!positions[p]) ds41_graph_reset(&graph);
            for (uint32_t il = 0; il < DS4_N_LAYER; il++)
                run_layer(il, positions[p], false, ds41_index_source(il), il + p);
        }
    }
    const ds4_gpu_execution_phase phases[] = {DS4_GPU_PHASE_AUTO, DS4_GPU_PHASE_PREFILL,
        DS4_GPU_PHASE_VERIFY, DS4_GPU_PHASE_BATCH_DECODE, DS4_GPU_PHASE_MIXED, DS4_GPU_PHASE_DECODE};
    for (unsigned pi = 0; pi < sizeof(phases) / sizeof(*phases); pi++) {
        ds4_gpu_exchange_execution_phase(phases[pi]);
        for (unsigned projected = 0; projected < 2; projected++) {
            /* Without resources only the admitted skip can succeed. This
             * proves dispatch admission independently of equal copied values. */
            ds41_gpu_graph empty = {0};
            const bool skip = !projected && phases[pi] == DS4_GPU_PHASE_DECODE;
            CHECK(ds41_attention_gather(&empty, 21, TOP, projected) == skip);
            CHECK(!ds41_attention_gather(&empty, 20, TOP, projected));
            CHECK(ds41_attention_gather(&empty, 21, 0, projected));
            if (skip) continue;
            for (uint32_t il = 20; il < 28; il++)
                run_layer(il, 1025, projected, true, 3u * il + pi + projected);
        }
    }
    ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_DECODE);
    queued_sources();
    CHECK(cache_hash() == before);
    for (unsigned j = 0; j < BUFFERS; j++) {
        ds4_gpu_tensor_free(view[j]); ds4_gpu_tensor_free(storage[j]);
    }
    ds4_gpu_cleanup();
    puts("V4.1 decode gather reuse: exact source/reuse/reset outputs, phase fallbacks, queued updates and input guards PASS");
    return 0;
}
