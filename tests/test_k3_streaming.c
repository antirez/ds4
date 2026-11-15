/* CPU-only default-selection tests: no environment writes, filesystem or GPU. */
#include "../k3_streaming.h"
#include "../k3_memory.h"
#include <assert.h>
#include <stdio.h>


#define MIB (UINT64_C(1024) * 1024u)
#define GIB (MIB * 1024u)

static void test_adaptive_cache_policy(void) {
    const uint64_t base = 10u * GIB;
    /* Small/unknown machines preserve the validated target even with a
     * generous supplied budget; the planner still caps actual admission. */
    const uint64_t conservative[] = {0, 16u*GIB, 32u*GIB};
    for (size_t i = 0; i < sizeof(conservative)/sizeof(conservative[0]); i++) {
        assert(k3_ssd_expert_cache_bytes(0, 0, true, conservative[i],
                                         UINT64_MAX, 0) == base);
        assert(k3_ssd_expert_cache_bytes(0, 0, true, conservative[i],
                                         0, UINT64_MAX) == base);
    }
    assert(k3_ssd_expert_cache_bytes(0, 0, true, 64u*GIB,
                                     47u*GIB, 3u*GIB) == 27u*GIB);
    assert(k3_ssd_expert_cache_bytes(0, 0, true, 128u*GIB,
                                     95u*GIB, 3u*GIB) == 51u*GIB);
    assert(k3_ssd_expert_cache_bytes(0, 0, true, 256u*GIB,
                                     191u*GIB, 3u*GIB) == 99u*GIB);
    /* No subtraction underflow or zero sentinel when staging exhausts the
     * budget, and odd additional bytes are rounded conservatively. */
    assert(k3_ssd_expert_cache_bytes(0, 0, true, 128u*GIB, 0, 0) == base);
    assert(k3_ssd_expert_cache_bytes(0, 0, true, 128u*GIB,
                                     GIB, UINT64_MAX) == base);
    assert(k3_ssd_expert_cache_bytes(0, 0, true, 128u*GIB,
                                     3u*GIB, 3u*GIB) == base);
    assert(k3_ssd_expert_cache_bytes(0, 0, true, 128u*GIB,
                                     base+3u*GIB+1u, 3u*GIB) == base);
    assert(k3_ssd_expert_cache_bytes(0, 0, true, 128u*GIB,
                                     base+3u*GIB+3u, 3u*GIB) == base+1u);
    assert(k3_ssd_expert_cache_bytes(0, 0, true, UINT64_MAX,
                                     UINT64_MAX, 0) == base+(UINT64_MAX-base)/2u);
    assert(k3_ssd_expert_cache_bytes(0, 0, true, UINT64_MAX,
                                     UINT64_MAX, UINT64_MAX) == base);
}

static void assert_adaptive_plan(k3_memory_plan plan, uint64_t budget,
                                 uint64_t transient, uint64_t target,
                                 uint64_t slot, uint32_t required,
                                 uint32_t maximum) {
    assert(budget >= transient);
    const uint64_t usable = budget - transient;
    assert(plan.cache_slots >= required && plan.cache_slots <= maximum);
    assert(plan.cache_bytes == (uint64_t)plan.cache_slots * slot);
    assert(plan.cache_bytes <= usable && plan.cache_bytes <= target);
    assert(plan.static_bytes == usable - plan.cache_bytes);
    const uint64_t cap = usable < target ? usable : target;
    if (plan.cache_slots < maximum) assert(cap - plan.cache_bytes < slot);
}

static void test_adaptive_memory_admission(void) {
    const uint64_t slot = 9u*MIB, allocated = GIB, transient = 3u*GIB;
    const uint32_t required = 896, desired = 1472, maximum = 92u*896u;
    /* Simulated capacities only: these CPU tests never allocate their budgets
     * or override the machine's actual physical-memory detection. */
    const struct {
        uint64_t physical, recommended, reclaimable, reserve, target;
        int fits;
    } machines[] = {
        {16u*GIB, 12u*GIB, 12u*GIB, 12u*GIB, 10u*GIB, 0},
        {32u*GIB, 24u*GIB, 24u*GIB, 12u*GIB, 10u*GIB, 1},
        {64u*GIB, 48u*GIB, 56u*GIB, 12u*GIB, 27u*GIB, 1},
        {128u*GIB, 96u*GIB, 120u*GIB, 16u*GIB, 51u*GIB, 1},
        {256u*GIB, 192u*GIB, 248u*GIB, 16u*GIB, 99u*GIB, 1},
    };
    uint64_t previous_cache = 0;
    for (size_t i = 0; i < sizeof(machines)/sizeof(machines[0]); i++) {
        const uint64_t budget = k3_memory_available(machines[i].physical,
            machines[i].recommended, allocated, machines[i].reclaimable,
            machines[i].reserve, 4u*GIB);
        const uint64_t target = k3_ssd_expert_cache_bytes(0, 0, true,
            machines[i].physical, budget, transient);
        assert(target == machines[i].target);
        k3_memory_plan plan;
        const int fits = k3_plan_gguf_memory(budget, transient, slot,
            required, desired, maximum, 0, target, 0, &plan);
        assert(fits == machines[i].fits);
        if (!fits) {
            assert(plan.cache_slots == 0 && plan.cache_bytes == 0);
            continue;
        }
        assert_adaptive_plan(plan, budget, transient, target, slot, required, maximum);
        assert(plan.cache_bytes >= previous_cache);
        previous_cache = plan.cache_bytes;
        assert(plan.cache_bytes + transient + allocated <=
               machines[i].physical - machines[i].reserve);
        assert(plan.cache_bytes + transient + allocated <= machines[i].recommended);
        assert(plan.cache_bytes + transient <= machines[i].reclaimable - 4u*GIB);
        if (machines[i].physical > 32u*GIB) {
            const uint64_t extra = budget - transient - 10u*GIB;
            assert(plan.static_bytes >= extra - extra/2u);
        }
    }

    /* More background use must not increase either the target or admitted
     * cache. A 128 GiB machine with little live headroom can admit less than
     * the 10 GiB baseline or reject startup if the batch cannot fit. */
    const uint64_t reclaimable[] = {120u*GIB, 80u*GIB, 32u*GIB, 16u*GIB, 12u*GIB, 0};
    uint64_t previous_target = UINT64_MAX;
    previous_cache = UINT64_MAX;
    for (size_t i = 0; i < sizeof(reclaimable)/sizeof(reclaimable[0]); i++) {
        const uint64_t budget = k3_memory_available(128u*GIB, 96u*GIB,
            allocated, reclaimable[i], 16u*GIB, 4u*GIB);
        const uint64_t target = k3_ssd_expert_cache_bytes(0, 0, true,
            128u*GIB, budget, transient);
        assert(target <= previous_target);
        previous_target = target;
        k3_memory_plan plan;
        const int fits = k3_plan_gguf_memory(budget, transient, slot,
            required, desired, maximum, 0, target, 0, &plan);
        assert(fits == (i < 4u));
        assert(plan.cache_bytes <= previous_cache);
        previous_cache = plan.cache_bytes;
        if (fits) assert_adaptive_plan(plan, budget, transient, target, slot, required, maximum);
        if (i == 3u) assert(plan.cache_bytes == 9u*GIB);
    }
    /* The same pressured snapshot stays safe on the 32 GiB machine. */
    const uint64_t small_budget = k3_memory_available(32u*GIB, 24u*GIB,
        allocated, 16u*GIB, 12u*GIB, 4u*GIB);
    const uint64_t small_target = k3_ssd_expert_cache_bytes(0, 0, true,
        32u*GIB, small_budget, transient);
    k3_memory_plan plan;
    assert(small_target == 10u*GIB);
    assert(k3_plan_gguf_memory(small_budget, transient, slot, required,
        desired, maximum, 0, small_target, 0, &plan));
    assert(plan.cache_bytes == 9u*GIB);

    /* Exact admission boundaries still enforce the simultaneous working set. */
    const uint64_t minimum = transient + (uint64_t)required*slot;
    const uint64_t target = k3_ssd_expert_cache_bytes(0, 0, true,
        128u*GIB, minimum, transient);
    assert(k3_plan_gguf_memory(minimum, transient, slot, required,
        desired, maximum, 0, target, 0, &plan));
    assert(plan.cache_slots == required);
    assert(!k3_plan_gguf_memory(minimum-1u, transient, slot, required,
        desired, maximum, 0, target, 0, &plan));
    /* Missing Metal recommendation must retain physical and live limits. */
    const uint64_t fallback = k3_memory_available(128u*GIB, 0, allocated,
        32u*GIB, 16u*GIB, 4u*GIB);
    assert(fallback == 28u*GIB);
    assert(k3_ssd_expert_cache_bytes(0, 0, true, 128u*GIB,
        fallback, transient) == (35u*GIB)/2u);
    /* Unknown physical RAM cannot authorize allocation through this chain. */
    const uint64_t unknown = k3_memory_available(0, 0, 0, UINT64_MAX, 0, 0);
    assert(unknown == 0);
    assert(!k3_plan_gguf_memory(unknown, transient, slot, required,
        desired, maximum, 0, 10u*GIB, 0, &plan));
    /* The real expert count still caps an extreme but valid target. */
    const uint64_t extreme = k3_ssd_expert_cache_bytes(0, 0, true,
        UINT64_MAX, UINT64_MAX, 0);
    assert(k3_plan_gguf_memory(UINT64_MAX, 0, slot, required,
        desired, maximum, 0, extreme, 0, &plan));
    assert(plan.cache_slots == maximum);
}

static void test_adaptive_explicit_overrides(void) {
    const uint64_t budget = 95u*GIB, transient = 3u*GIB, slot = 9u*MIB;
    const uint32_t required = 128, desired = 1472, maximum = 92u*896u;
    const struct { uint64_t bytes; uint32_t slots; } requests[] = {
        {2u*GIB, 0}, {0, 256}, {2u*GIB, 192}, {UINT64_MAX, UINT32_MAX},
    };
    for (size_t i = 0; i < sizeof(requests)/sizeof(requests[0]); i++) {
        for (unsigned enabled = 0; enabled <= 1u; enabled++) {
            const uint64_t target = k3_ssd_expert_cache_bytes(requests[i].bytes,
                requests[i].slots, enabled != 0, 128u*GIB, budget, transient);
            assert(target == requests[i].bytes);
            k3_memory_plan actual, legacy;
            assert(k3_plan_gguf_memory(budget, transient, slot, required,
                desired, maximum, 0, target, requests[i].slots, &actual));
            assert(k3_plan_gguf_memory(budget, transient, slot, required,
                desired, maximum, 0, requests[i].bytes, requests[i].slots, &legacy));
            assert(actual.cache_slots == legacy.cache_slots);
            assert(actual.cache_bytes == legacy.cache_bytes);
            assert(actual.static_bytes == legacy.static_bytes);
        }
    }
    assert(k3_ssd_expert_cache_bytes(0, 0, false, 128u*GIB,
                                     budget, transient) == 0);
    /* An explicit cache below the batch minimum remains a clear failure. */
    k3_memory_plan plan;
    const uint64_t tiny = k3_ssd_expert_cache_bytes(slot, 0, true,
        128u*GIB, budget, transient);
    assert(tiny == slot);
    assert(!k3_plan_gguf_memory(budget, transient, slot, required,
        desired, maximum, 0, tiny, 0, &plan));
}

int main(void) {
    test_adaptive_cache_policy();
    test_adaptive_memory_admission();
    test_adaptive_explicit_overrides();
    const uint64_t mib = UINT64_C(1024) * 1024u;
    const uint64_t gib = mib * 1024u;
    assert(!k3_ssd_defaults_enabled(false, true, NULL));
    assert(!k3_ssd_defaults_enabled(true, false, NULL));
    assert(!k3_ssd_defaults_enabled(false, false, "1"));
    assert(k3_ssd_defaults_enabled(true, true, NULL));
    assert(!k3_ssd_defaults_enabled(true, true, ""));
    assert(!k3_ssd_defaults_enabled(true, true, "0"));
    assert(k3_ssd_defaults_enabled(true, true, "1"));
    assert(k3_ssd_defaults_enabled(true, true, "auto"));
    /* Match existing K3 boolean spelling: only exact "0" is false. */
    assert(k3_ssd_defaults_enabled(true, true, "00"));

    assert(k3_ssd_prefill_chunk(0, false) == 512u);
    assert(k3_ssd_prefill_chunk(0, true) == 64u);
    assert(k3_ssd_prefill_chunk(8, true) == 8u);
    assert(k3_ssd_prefill_chunk(512, true) == 512u);
    assert(k3_ssd_prefill_chunk(UINT32_MAX, true) == UINT32_MAX);
    /* Selecting chunk64 must not shrink an unspecified API context to64. */
    assert(k3_ssd_initial_context(0, 0) == 512u);
    assert(k3_ssd_initial_context(-1, 0) == 512u);
    assert(k3_ssd_initial_context(0, 8) == 8u);
    assert(k3_ssd_initial_context(1024, 0) == 1024u);
    assert(k3_ssd_initial_context(16, 64) == 16u);
    assert(k3_ssd_expert_cache_bytes(0, 0, true, 32u*gib, 20u*gib, 3u*gib) == 10u * gib);
    assert(k3_ssd_expert_cache_bytes(0, 0, false, 128u*gib, 95u*gib, 3u*gib) == 0);
    assert(k3_ssd_expert_cache_bytes(2u * gib, 0, true, 128u*gib, 95u*gib, 3u*gib) == 2u * gib);
    assert(k3_ssd_expert_cache_bytes(0, 231, true, 128u*gib, 95u*gib, 3u*gib) == 0);
    assert(k3_ssd_expert_cache_bytes(2u * gib, 231, true, 128u*gib, 95u*gib, 3u*gib) == 2u * gib);
    assert(k3_ssd_expert_cache_bytes(UINT64_MAX, UINT32_MAX, true, UINT64_MAX, UINT64_MAX, 0) == UINT64_MAX);

    assert(k3_ssd_slab_target_bytes(true, NULL) == 256u * mib);
    assert(k3_ssd_slab_target_bytes(false, NULL) == 4096u * mib);
    const char *fallback[] = {"", "0", "invalid", "256x", "256 "};
    for (size_t i = 0; i < sizeof(fallback) / sizeof(fallback[0]); i++) {
        assert(k3_ssd_slab_target_bytes(true, fallback[i]) == 4096u * mib);
        assert(k3_ssd_slab_target_bytes(false, fallback[i]) == 4096u * mib);
    }
    const char *explicit_values[] = {"256", "1024", "4096", " 512", "+512"};
    const uint64_t expected[] = {256u*mib, 1024u*mib, 4096u*mib, 512u*mib, 512u*mib};
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
        assert(k3_ssd_slab_target_bytes(true, explicit_values[i]) == expected[i]);
        assert(k3_ssd_slab_target_bytes(false, explicit_values[i]) == expected[i]);
    }
    assert(k3_ssd_slab_target_bytes(true, "18446744073709551615") == UINT64_MAX);
    assert(k3_ssd_slab_target_bytes(false, "18446744073709551616") == UINT64_MAX);

    assert(k3_ssd_expert_readahead_enabled(true, false, NULL));
    assert(!k3_ssd_expert_readahead_enabled(false, false, NULL));
    assert(!k3_ssd_expert_readahead_enabled(true, true, NULL));
    const char *present[] = {"", "0", "1"};
    for (size_t i = 0; i < sizeof(present) / sizeof(present[0]); i++) {
        assert(!k3_ssd_expert_readahead_enabled(true, false, present[i]));
        assert(!k3_ssd_expert_readahead_enabled(true, true, present[i]));
    }
    /* Turning the profile off restores the ordinary streaming choices. */
    assert(k3_ssd_prefill_chunk(0, false) == 512u);
    assert(k3_ssd_expert_cache_bytes(0, 0, false, 128u*gib, 95u*gib, 3u*gib) == 0);
    assert(k3_ssd_slab_target_bytes(false, NULL) == 4096u * mib);
    assert(k3_ssd_expert_readahead_enabled(true, false, NULL));
    puts("K3 SSD streaming policy: adaptive admission, defaults, explicit overrides and presence semantics passed (CPU only)");
    return 0;
}
