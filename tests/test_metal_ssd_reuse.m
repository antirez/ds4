/* CPU-only regression for the Metal SSD cache's eviction plan. No Metal
 * device, model mapping, command buffer, or disk payload is needed. The
 * original scalar selector is the oracle, including its numerical tie order.
 * Build/run: make test-metal-ssd-reuse */
#include "../ds4_metal.m"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

@interface DS4ReuseTestBuffer : NSObject
@property(nonatomic, assign) uint32_t identifier;
@end
@implementation DS4ReuseTestBuffer
@end

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "SSD reuse check failed at line %d: %s\n", __LINE__, #c); \
    exit(1); \
} } while (0)

enum { TEST_GATE_BYTES = 422400, TEST_DOWN_BYTES = 645120 };

static id<MTLBuffer> test_buffer(uint32_t identifier) {
    DS4ReuseTestBuffer *buffer = [DS4ReuseTestBuffer new];
    buffer.identifier = identifier;
    return (id<MTLBuffer>)buffer;
}

static void reset_cache(void) {
    /* Release mock objects directly: production cleanup also manages real
     * mlocked/slab buffers, which this metadata-only fixture never creates. */
    for (uint32_t l = 0; l < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; l++) {
        for (uint32_t e = 0; e < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; e++) {
            g_stream_expert_cache[l][e] = (ds4_gpu_stream_expert_cache_entry){0};
        }
    }
    memset(g_stream_expert_cache_route_hotness, 0, sizeof(g_stream_expert_cache_route_hotness));
    memset(g_stream_expert_cache_layer_count, 0, sizeof(g_stream_expert_cache_layer_count));
    memset(g_stream_expert_cache_layer_evictions, 0, sizeof(g_stream_expert_cache_layer_evictions));
    g_ssd_streaming_mode = 1;
    g_stream_expert_cache_budget_override = DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES;
    g_stream_expert_cache_mlock_budget_cap = 0;
    g_stream_expert_cache_entry_count = 0;
    g_stream_expert_cache_bytes = 0;
    g_stream_expert_cache_evictions = 0;
    g_stream_expert_cache_buffer_reuses = 0;
    g_stream_expert_cache_hits = 37;
    g_stream_expert_cache_misses = 41;
    g_stream_expert_cache_done_seq = 7;
    g_stream_expert_service_thread_set = 0;
}

static void add_entry(uint32_t id, uint32_t hotness, uint64_t last_used,
                      bool wrong_size) {
    const uint32_t l = id / DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
    const uint32_t i = id % DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
    ds4_gpu_stream_expert_cache_entry *e = &g_stream_expert_cache[l][i];
    CHECK(!e->valid);
    e->gate_buffer = test_buffer(id * 3u);
    e->up_buffer = id % 2 ? test_buffer(id * 3u + 1u) : e->gate_buffer;
    e->down_buffer = id % 2 ? test_buffer(id * 3u + 2u) : e->gate_buffer;
    e->gate_inner = (NSUInteger)id * 64u;
    e->up_inner = e->gate_inner + TEST_GATE_BYTES;
    e->down_inner = e->up_inner + TEST_GATE_BYTES;
    e->gate_expert_bytes = TEST_GATE_BYTES + wrong_size;
    e->down_expert_bytes = TEST_DOWN_BYTES;
    e->logical_bytes = e->gate_expert_bytes * 2u + e->down_expert_bytes;
    e->last_used = last_used;
    e->use_count = 23;
    e->inflight_seq = 7; /* A completed command is reusable. */
    e->valid = 1;
    g_stream_expert_cache_route_hotness[l][i] = hotness;
    g_stream_expert_cache_entry_count++;
    g_stream_expert_cache_layer_count[l]++;
    g_stream_expert_cache_bytes += e->logical_bytes;
}

static void seed_cache(uint32_t entries) {
    reset_cache();
    for (uint32_t i = 0; i < entries; i++) {
        /* Many equal keys, interleaved sizes, and both buffer layouts. */
        add_entry(i, (i * 13u) % 29u, (i * 7u) % 17u, i % 19u == 0);
    }
    g_stream_expert_cache_budget_override = entries;
}

static uint32_t check_reuse(const ds4_gpu_stream_expert_reusable_buffers *reuse) {
    CHECK(reuse->gate_buffer && reuse->up_buffer && reuse->down_buffer);
    const uint32_t id = ((DS4ReuseTestBuffer *)reuse->gate_buffer).identifier / 3u;
    CHECK(((DS4ReuseTestBuffer *)reuse->up_buffer).identifier == id * 3u + (id % 2 ? 1u : 0u));
    CHECK(((DS4ReuseTestBuffer *)reuse->down_buffer).identifier == id * 3u + (id % 2 ? 2u : 0u));
    CHECK(reuse->gate_inner == (NSUInteger)id * 64u);
    CHECK(reuse->up_inner == reuse->gate_inner + TEST_GATE_BYTES);
    CHECK(reuse->down_inner == reuse->up_inner + TEST_GATE_BYTES);
    return id;
}

typedef struct {
    uint32_t count, layer_count[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint64_t bytes, evictions, reuses, hits, misses;
    uint64_t layer_evictions[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
    uint8_t valid[DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES];
} cache_state;

static cache_state state(void) {
    cache_state s = {0};
    s.count = g_stream_expert_cache_entry_count;
    s.bytes = g_stream_expert_cache_bytes;
    s.evictions = g_stream_expert_cache_evictions;
    s.reuses = g_stream_expert_cache_buffer_reuses;
    s.hits = g_stream_expert_cache_hits;
    s.misses = g_stream_expert_cache_misses;
    memcpy(s.layer_count, g_stream_expert_cache_layer_count, sizeof(s.layer_count));
    memcpy(s.layer_evictions, g_stream_expert_cache_layer_evictions, sizeof(s.layer_evictions));
    for (uint32_t l = 0; l < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER; l++) {
        for (uint32_t e = 0; e < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; e++) {
            s.valid[l * DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT + e] = g_stream_expert_cache[l][e].valid;
        }
    }
    return s;
}

static void check_state(const cache_state *expected) {
    const cache_state actual = state();
    CHECK(actual.count == expected->count && actual.bytes == expected->bytes);
    CHECK(actual.evictions == expected->evictions && actual.reuses == expected->reuses);
    CHECK(actual.hits == expected->hits && actual.misses == expected->misses);
    CHECK(!memcmp(actual.layer_count, expected->layer_count, sizeof(actual.layer_count)));
    CHECK(!memcmp(actual.layer_evictions, expected->layer_evictions, sizeof(actual.layer_evictions)));
    CHECK(!memcmp(actual.valid, expected->valid, sizeof(actual.valid)));
}

static void compare_sequence(uint32_t entries, uint32_t needed, uint32_t consume) {
    int32_t protected_ids[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT];
    for (uint32_t i = 0; i < DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT; i++) protected_ids[i] = (int32_t)i;
    uint32_t expected[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT], n_expected = 0;
    seed_cache(entries);
    for (uint32_t i = 0; i < consume; i++) {
        ds4_gpu_stream_expert_reusable_buffers reuse = {0};
        if (!ds4_gpu_stream_expert_cache_take_reusable(1, 2, protected_ids, 512,
                TEST_GATE_BYTES, TEST_DOWN_BYTES, &reuse)) break;
        expected[n_expected++] = check_reuse(&reuse);
    }
    const cache_state after_scalar = state();
    seed_cache(entries);
    const cache_state before_plan = state();
    ds4_gpu_stream_expert_reuse_plan plan;
    const int planned = ds4_gpu_stream_expert_cache_plan_reuse(&plan, needed,
            2, protected_ids, 512, TEST_GATE_BYTES, TEST_DOWN_BYTES);
    check_state(&before_plan); /* Planning cannot evict ahead of an error. */
    CHECK(planned == (plan.count != 0));
    CHECK(plan.count <= needed && plan.count >= n_expected);
    for (uint32_t i = 0; i < n_expected; i++) {
        ds4_gpu_stream_expert_reusable_buffers reuse = {0};
        CHECK(ds4_gpu_stream_expert_cache_take_planned_reuse(&plan, 2,
                protected_ids, 512, TEST_GATE_BYTES, TEST_DOWN_BYTES, &reuse));
        CHECK(check_reuse(&reuse) == expected[i]);
    }
    check_state(&after_scalar);
    if (n_expected < consume) {
        ds4_gpu_stream_expert_reusable_buffers reuse = {0};
        CHECK(!ds4_gpu_stream_expert_cache_take_planned_reuse(&plan, 2,
                protected_ids, 512, TEST_GATE_BYTES, TEST_DOWN_BYTES, &reuse));
        check_state(&after_scalar);
    }
    printf("  entries=%u planned=%u consumed=%u: exact scalar order and state\n",
           entries, plan.count, n_expected);
}

static void numerical_ties(void) {
    reset_cache();
    add_entry(40959, 0, 9, false);
    add_entry(512, 0, 9, false);
    add_entry(511, 0, 9, false);
    add_entry(0, 0, 9, false);
    add_entry(513, UINT32_MAX, 0, false);
    add_entry(1024, 1, UINT64_MAX, false);
    add_entry(1025, UINT32_MAX, UINT64_MAX, false);
    const uint32_t expected[] = {0, 511, 512, 40959, 1024, 513};
    ds4_gpu_stream_expert_reuse_plan plan;
    CHECK(ds4_gpu_stream_expert_cache_plan_reuse(&plan, 512,
            UINT32_MAX, NULL, 0, TEST_GATE_BYTES, TEST_DOWN_BYTES));
    CHECK(plan.count == sizeof(expected) / sizeof(expected[0]));
    for (uint32_t i = 0; i < plan.count; i++) {
        ds4_gpu_stream_expert_reusable_buffers reuse = {0};
        CHECK(ds4_gpu_stream_expert_cache_take_planned_reuse(&plan,
                UINT32_MAX, NULL, 0, TEST_GATE_BYTES, TEST_DOWN_BYTES, &reuse));
        CHECK(check_reuse(&reuse) == expected[i]);
    }
    CHECK(g_stream_expert_cache[2][1].valid); /* Original sentinel bounds. */
}

static void guarded_fallbacks(void) {
    ds4_gpu_stream_expert_reuse_plan plan;
    seed_cache(32);
    g_stream_expert_cache[0][1].inflight_seq = 8;
    const cache_state inflight = state();
    CHECK(!ds4_gpu_stream_expert_cache_plan_reuse(&plan, 16,
            UINT32_MAX, NULL, 0, TEST_GATE_BYTES, TEST_DOWN_BYTES));
    check_state(&inflight);
    ds4_gpu_stream_expert_reusable_buffers reuse = {0};
    CHECK(ds4_gpu_stream_expert_cache_take_reusable(1, UINT32_MAX, NULL, 0,
            TEST_GATE_BYTES, TEST_DOWN_BYTES, &reuse));
    CHECK(check_reuse(&reuse) != 1 && g_stream_expert_cache[0][1].valid);

    /* A stale plan never consumes a changed, newly protected, or inflight
     * candidate. The unchanged scalar selector remains the recovery path. */
    for (uint32_t mutation = 0; mutation < 4; mutation++) {
        seed_cache(32);
        CHECK(ds4_gpu_stream_expert_cache_plan_reuse(&plan, 16,
                UINT32_MAX, NULL, 0, TEST_GATE_BYTES, TEST_DOWN_BYTES));
        const ds4_gpu_stream_expert_reuse_candidate c = plan.candidates[0];
        ds4_gpu_stream_expert_cache_entry *e = &g_stream_expert_cache[c.layer][c.expert];
        uint32_t protect_layer = UINT32_MAX;
        int32_t protect_id = (int32_t)c.expert;
        if (mutation == 0) e->inflight_seq = 8;
        if (mutation == 1) e->last_used++;
        if (mutation == 2) g_stream_expert_cache_route_hotness[c.layer][c.expert]++;
        if (mutation == 3) protect_layer = c.layer;
        const cache_state changed = state();
        CHECK(!ds4_gpu_stream_expert_cache_take_planned_reuse(&plan,
                protect_layer, &protect_id, 1, TEST_GATE_BYTES, TEST_DOWN_BYTES, &reuse));
        CHECK(!reuse.gate_buffer && !reuse.up_buffer && !reuse.down_buffer);
        CHECK(!reuse.gate_inner && !reuse.up_inner && !reuse.down_inner);
        check_state(&changed);
        CHECK(plan.next == plan.count);
        CHECK(ds4_gpu_stream_expert_cache_take_reusable(1, protect_layer,
                &protect_id, 1, TEST_GATE_BYTES, TEST_DOWN_BYTES, &reuse));
        if (mutation == 0 || mutation == 3) CHECK(check_reuse(&reuse) != c.expert);
    }

    seed_cache(32);
    ds4_gpu_stream_expert_cache_note_service_thread();
    const cache_state service = state();
    CHECK(!ds4_gpu_stream_expert_cache_plan_reuse(&plan, 16,
            UINT32_MAX, NULL, 0, TEST_GATE_BYTES, TEST_DOWN_BYTES));
    check_state(&service);
    g_stream_expert_service_thread_set = 0;
    CHECK(!ds4_gpu_stream_expert_cache_plan_reuse(&plan, 0,
            UINT32_MAX, NULL, 0, TEST_GATE_BYTES, TEST_DOWN_BYTES));
    CHECK(!ds4_gpu_stream_expert_cache_plan_reuse(&plan, 513,
            UINT32_MAX, NULL, 0, TEST_GATE_BYTES, TEST_DOWN_BYTES));

    /* A live target that misses metadata matching must keep its own buffers,
     * rather than evict the plan's first unrelated cache entry. */
    seed_cache(32);
    __strong id<MTLBuffer> gate = nil, up = nil, down = nil;
    NSUInteger gi = 0, ui = 0, di = 0;
    CHECK(ds4_gpu_stream_expert_cache_prepare_load_buffers(0, 1,
            UINT32_MAX, NULL, 0, TEST_GATE_BYTES, TEST_DOWN_BYTES, 1,
            &gate, &up, &down, &gi, &ui, &di));
    CHECK(((DS4ReuseTestBuffer *)gate).identifier == 3);
    CHECK(g_stream_expert_cache_evictions == 0 && g_stream_expert_cache_entry_count == 31);
    CHECK(g_stream_expert_cache_buffer_reuses == 3);
}

int main(void) {
    @autoreleasepool {
        const uint32_t sizes[] = {1, 16, 17, 511, 512};
        for (uint32_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
            compare_sequence(2048, sizes[i], sizes[i]);
        }
        compare_sequence(DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES, 512, 512);
        compare_sequence(17, 512, 512); /* Fewer reusable entries than misses. */
        compare_sequence(2048, 512, 7); /* Later preparation/read failure. */
        numerical_ties();
        guarded_fallbacks();
        reset_cache();
        puts("Metal SSD reuse planner: PASS (CPU-only)");
    }
    return 0;
}
