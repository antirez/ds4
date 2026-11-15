#include "k3_memory.h"

#include <stdio.h>
#include <stdlib.h>

#define MIB (UINT64_C(1024) * 1024)
#define GIB (UINT64_C(1024) * 1024 * 1024)
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static void check_plan(k3_memory_plan plan, uint64_t budget,
                       uint64_t transient, uint64_t slot_bytes,
                       uint32_t required, uint32_t maximum) {
    CHECK(transient <= budget);
    CHECK(plan.cache_bytes <= budget - transient);
    CHECK(plan.static_bytes <= budget - transient - plan.cache_bytes);
    CHECK(plan.cache_bytes == (uint64_t)plan.cache_slots * slot_bytes);
    CHECK(plan.cache_slots >= required);
    CHECK(plan.cache_slots <= maximum);
    /* All remaining capacity belongs to optional static weights. */
    CHECK(plan.static_bytes == budget - transient - plan.cache_bytes);
}

static void test_step_required(void) {
    const uint64_t transient = (UINT64_C(302) * GIB + 99) / 100;
    const uint64_t cache = 2 * GIB, reserve = 4 * GIB;
    const uint64_t reclaimable = UINT64_C(758) * GIB / 100;
    uint64_t required = 0;

    /* The reported 7.58 GiB is enough once the 2 GiB cache is locked. The
     * previous fixed 128-expert charge incorrectly still required 8.13 GiB. */
    CHECK(k3_memory_step_required(transient, cache, cache, reserve, &required));
    CHECK(required == reserve + transient);
    CHECK(reclaimable >= required);
    CHECK(reclaimable < required + UINT64_C(111) * GIB / 100);

    /* A new or entirely unlocked slab may still commit the whole 2 GiB. */
    CHECK(k3_memory_step_required(transient, cache, 0, reserve, &required));
    CHECK(required == reserve + transient + cache);
    CHECK(reclaimable < required);
    CHECK(k3_memory_step_required(transient, cache, GIB, reserve, &required));
    CHECK(required == reserve + transient + GIB);
    CHECK(reclaimable < required);
    CHECK(k3_memory_step_required(transient, cache, cache - 1, reserve, &required));
    CHECK(required == reserve + transient + 1);

    /* This also covers free-but-still-locked slab slots: no cache-entry
     * count participates. Oversized locked counters must not underflow. */
    CHECK(k3_memory_step_required(transient, cache, cache + 1, reserve, &required));
    CHECK(required == reserve + transient);
    CHECK(k3_memory_step_required(transient, 0, 0, reserve, &required));
    const uint64_t boundary = (UINT64_C(702) * GIB + 99) / 100;
    CHECK(required == boundary);
    CHECK(boundary - 1 < required);
    CHECK(boundary >= required);
    CHECK(k3_memory_step_required(0, 0, 0, 0, &required));
    CHECK(required == 0);

    CHECK(k3_memory_step_required(UINT64_MAX, 0, 0, 0, &required));
    CHECK(required == UINT64_MAX);
    CHECK(!k3_memory_step_required(UINT64_MAX, 0, 0, 1, &required));
    CHECK(required == 0);
    CHECK(!k3_memory_step_required(1, UINT64_MAX, 0, 0, &required));
    CHECK(required == 0);
    CHECK(k3_memory_step_required(UINT64_MAX, UINT64_MAX, UINT64_MAX,
                                  0, &required));
    CHECK(required == UINT64_MAX);
    CHECK(!k3_memory_step_required(1, 2, 0, 4, NULL));
}

static void test_available(void) {
    CHECK(k3_memory_remaining(10, 4) == 6);
    CHECK(k3_memory_remaining(10, 10) == 0);
    CHECK(k3_memory_remaining(10, 11) == 0);

    /* Each independent cap must win when it is the smallest. */
    CHECK(k3_memory_available(32 * GIB, 24 * GIB, 2 * GIB,
                              32 * GIB, 12 * GIB, 4 * GIB) == 18 * GIB);
    CHECK(k3_memory_available(32 * GIB, 10 * GIB, 2 * GIB,
                              32 * GIB, 12 * GIB, 4 * GIB) == 8 * GIB);
    CHECK(k3_memory_available(32 * GIB, 24 * GIB, 2 * GIB,
                              20 * GIB, 12 * GIB, 4 * GIB) == 16 * GIB);
    /* Live availability already reflects allocations: do not subtract twice. */
    CHECK(k3_memory_available(32 * GIB, 0, 2 * GIB,
                              20 * GIB, 12 * GIB, 4 * GIB) == 16 * GIB);
    CHECK(k3_memory_available(8 * GIB, 0, 0,
                              32 * GIB, 12 * GIB, 4 * GIB) == 0);
    CHECK(k3_memory_available(32 * GIB, 0, 24 * GIB,
                              32 * GIB, 12 * GIB, 4 * GIB) == 0);
    CHECK(k3_memory_available(32 * GIB, GIB, 2 * GIB,
                              32 * GIB, 12 * GIB, 4 * GIB) == 0);
    CHECK(k3_memory_available(32 * GIB, 0, 0,
                              4 * GIB, 12 * GIB, 4 * GIB) == 0);
    CHECK(k3_memory_available(32 * GIB, 0, 0,
                              4 * GIB - 1, 12 * GIB, 4 * GIB) == 0);
    CHECK(k3_memory_available(32 * GIB, 0, 0,
                              4 * GIB + 1, 12 * GIB, 4 * GIB) == 1);
    CHECK(k3_memory_available(UINT64_MAX, 0, 1, UINT64_MAX, 1, 1)
          == UINT64_MAX - 2);
}

static void test_live_reserve_regression(void) {
    /* Actual vm_stat snapshot from the 32 GiB M1 Max's rejected startup. */
    const uint64_t reclaimable =
        (UINT64_C(29178) - 671 + 971933 + 50069) * 16384;
    const uint64_t host_reserve = 12 * GIB, live_floor = 4 * GIB;
    const uint64_t initial_budget = k3_memory_available(
        32 * GIB, 24 * GIB, 0, reclaimable, host_reserve, live_floor);
    CHECK(initial_budget == reclaimable - live_floor);
    CHECK(initial_budget == 12 * GIB + UINT64_C(31670272));
    CHECK(initial_budget <= 32 * GIB - host_reserve);

    /* Retain the previous 5.9524 GiB transient estimate, even though it is
     * conservative: it plus 1 GiB scratch and a 2 GiB cache must fit. */
    const uint64_t transient = (UINT64_C(59524) * GIB + 9999) / 10000;
    const uint64_t scratch = GIB, slot = 9 * MIB;
    const uint64_t old_budget = k3_memory_available(
        32 * GIB, 24 * GIB, 0, reclaimable, host_reserve, host_reserve);
    CHECK(old_budget < transient);
    const uint64_t after_scratch = k3_memory_available(
        32 * GIB, 24 * GIB, scratch, reclaimable - scratch,
        host_reserve, live_floor);
    CHECK(after_scratch == initial_budget - scratch);
    CHECK(after_scratch >= transient + 2 * GIB);
    k3_memory_plan plan;
    CHECK(k3_plan_gguf_memory(after_scratch, transient, slot, 128, 1472,
                              92 * 896, 16 * GIB, 2 * GIB, 0, &plan));
    check_plan(plan, after_scratch, transient, slot, 128, 92 * 896);
    CHECK(plan.cache_slots == (2 * GIB) / slot);
    CHECK(plan.cache_bytes <= 2 * GIB);
    CHECK(reclaimable - scratch - transient - plan.cache_bytes >= live_floor);
    CHECK(32 * GIB - scratch - transient - plan.cache_bytes >= host_reserve);

    /* Other activity between snapshots must still reduce the live budget. */
    const uint64_t pressured = k3_memory_available(
        32 * GIB, 24 * GIB, scratch, reclaimable - scratch - 2 * GIB,
        host_reserve, live_floor);
    CHECK(pressured == after_scratch - 2 * GIB);
}

static void test_alignment(void) {
    const uint64_t page = 16384;
    uint64_t aligned = 42;
    CHECK(k3_memory_align(0, page, &aligned) && aligned == 0);
    CHECK(k3_memory_align(1, page, &aligned) && aligned == page);
    CHECK(k3_memory_align(page, page, &aligned) && aligned == page);
    CHECK(k3_memory_align(page + 1, page, &aligned) && aligned == 2 * page);
    CHECK(k3_memory_align(UINT64_MAX - (page - 1), page, &aligned));
    CHECK(aligned == UINT64_MAX - (page - 1));
    CHECK(!k3_memory_align(UINT64_MAX - (page - 2), page, &aligned));
    CHECK(!k3_memory_align(UINT64_MAX, page, &aligned));
    CHECK(!k3_memory_align(1, 0, &aligned));
    CHECK(!k3_memory_align(1, 3, &aligned));
    CHECK(!k3_memory_align(1, page, NULL));
}

static void test_transient_peak(void) {
    uint64_t bytes = 0;
    const uint64_t slack = 256 * MIB;
    /* The two largest layers cannot be live together when nonadjacent. */
    const uint64_t separated[] = {6 * GIB, GIB, 5 * GIB};
    CHECK(k3_memory_transient_peak(separated, 3, GIB, 2 * GIB, slack, &bytes));
    CHECK(bytes == 7 * GIB + slack);
    const uint64_t single[] = {2 * GIB};
    CHECK(k3_memory_transient_peak(single, 1, 0, 0, slack, &bytes));
    CHECK(bytes == 2 * GIB + slack);
    CHECK(k3_memory_transient_peak(single, 1, 4 * GIB, 4 * GIB, slack, &bytes));
    CHECK(bytes == 6 * GIB + slack);
    /* Output processing may have a larger mapping than the head alone. */
    CHECK(k3_memory_transient_peak(single, 1, GIB, 5 * GIB, slack, &bytes));
    CHECK(bytes == 5 * GIB + slack);

    /* Representative K3 GGUF static spans, rounded to actual 16 KiB pages. */
    uint64_t actual_layers[3];
    CHECK(k3_memory_align(UINT64_C(1757451) * GIB / 1000000,
                          16384, &actual_layers[0]));
    CHECK(k3_memory_align(UINT64_C(977523) * GIB / 1000000,
                          16384, &actual_layers[1]));
    CHECK(k3_memory_align(UINT64_C(583193) * GIB / 1000000,
                          16384, &actual_layers[2]));
    const uint64_t head = 2240 * MIB; /* 2.1875 GiB BF16 output head. */
    CHECK(k3_memory_transient_peak(actual_layers, 3, head, head + MIB,
                                   slack, &bytes));
    CHECK(bytes == actual_layers[2] + head + slack);
    CHECK(bytes > 3 * GIB && bytes < 303 * GIB / 100);

    const uint64_t extreme[] = {UINT64_MAX, 1};
    CHECK(!k3_memory_transient_peak(extreme, 2, 0, 0, 0, &bytes));
    CHECK(bytes == 0);
    CHECK(!k3_memory_transient_peak(extreme, 1, 1, 0, 0, &bytes));
    CHECK(!k3_memory_transient_peak(extreme, 1, 0, 0, 1, &bytes));
    CHECK(k3_memory_transient_peak(extreme, 1, 0, 0, 0, &bytes));
    CHECK(bytes == UINT64_MAX);
    CHECK(!k3_memory_transient_peak(single, 1, 0, UINT64_MAX, 1, &bytes));
    CHECK(!k3_memory_transient_peak(NULL, 1, 0, 0, 0, &bytes));
    CHECK(!k3_memory_transient_peak(single, 0, 0, 0, 0, &bytes));
    CHECK(!k3_memory_transient_peak(single, 1, 0, 0, 0, NULL));
}

static void test_32gb_cache_cap(void) {
    /* Synthetic 9 MiB aligned experts model the K3 size class. */
    const uint64_t slot = 9 * MIB;
    const uint64_t budget = k3_memory_available(
        32 * GIB, 24 * GIB, 2 * GIB, 16 * GIB, 12 * GIB, 4 * GIB);
    const uint64_t transient = 3 * GIB;
    const uint32_t maximum = 92 * 896;
    k3_memory_plan plan;
    CHECK(budget == 12 * GIB);
    CHECK(k3_plan_gguf_memory(budget, transient, slot, 128, 1472, maximum,
                              16 * GIB, 0, 0, &plan));
    check_plan(plan, budget, transient, slot, 128, maximum);
    CHECK(plan.cache_slots == 1024);
    CHECK(plan.cache_slots < 1472);
    CHECK(plan.cache_bytes == 9 * GIB);
    CHECK(plan.static_bytes == 0);

    /* 128 simultaneous slots fit a small prefill; a large prefill needs 896. */
    const uint64_t small_budget = transient + 128 * slot;
    CHECK(k3_plan_gguf_memory(small_budget, transient, slot, 128, 1472,
                              maximum, 16 * GIB, 0, 0, &plan));
    CHECK(plan.cache_slots == 128);
    check_plan(plan, small_budget, transient, slot, 128, maximum);
    CHECK(!k3_plan_gguf_memory(small_budget - 1, transient, slot, 128,
                               1472, maximum, 16 * GIB, 0, 0, &plan));
    CHECK(!k3_plan_gguf_memory(small_budget, transient, slot, 896, 1472,
                               maximum, 16 * GIB, 0, 0, &plan));
    const uint64_t large_prefill_budget = transient + 896 * slot;
    CHECK(k3_plan_gguf_memory(large_prefill_budget, transient, slot, 896,
                              1472, maximum, 16 * GIB, 0, 0, &plan));
    CHECK(plan.cache_slots == 896);
    check_plan(plan, large_prefill_budget, transient, slot, 896, maximum);
    CHECK(!k3_plan_gguf_memory(large_prefill_budget - 1, transient, slot,
                               896, 1472, maximum, 16 * GIB, 0, 0, &plan));
}

static void test_user_cache_limits(void) {
    const uint64_t slot = 9 * MIB, budget = 12 * GIB, transient = 3 * GIB;
    const uint32_t maximum = 92 * 896;
    k3_memory_plan plan;
    CHECK(k3_plan_gguf_memory(budget, transient, slot, 128, 1472, maximum,
                              16 * GIB, 2 * GIB, 0, &plan));
    CHECK(plan.cache_slots == (2 * GIB) / slot);
    CHECK(plan.cache_bytes <= 2 * GIB);
    check_plan(plan, budget, transient, slot, 128, maximum);
    CHECK(k3_plan_gguf_memory(budget, transient, slot, 128, 1472, maximum,
                              16 * GIB, 0, 256, &plan));
    CHECK(plan.cache_slots == 256);
    check_plan(plan, budget, transient, slot, 128, maximum);
    CHECK(k3_plan_gguf_memory(budget, transient, slot, 128, 1472, maximum,
                              16 * GIB, 2 * GIB, 192, &plan));
    CHECK(plan.cache_slots == 192);
    check_plan(plan, budget, transient, slot, 128, maximum);
    CHECK(!k3_plan_gguf_memory(budget, transient, slot, 128, 1472, maximum,
                               16 * GIB, 128 * slot - 1, 0, &plan));
    CHECK(!k3_plan_gguf_memory(budget, transient, slot, 128, 1472, maximum,
                               16 * GIB, 0, 127, &plan));
    /* An oversized request must still respect available memory. */
    CHECK(k3_plan_gguf_memory(budget, transient, slot, 128, 1472, maximum,
                              16 * GIB, UINT64_MAX, UINT32_MAX, &plan));
    CHECK(plan.cache_slots == 1024);
    check_plan(plan, budget, transient, slot, 128, maximum);
}

static void test_explicit_cache_targets(void) {
    /* Actual K3 expert slots: the automatic 1472-slot target is 12.735 GiB.
     * A 14 GiB request must reach 1618 slots when the memory budget permits. */
    const uint64_t slot = UINT64_C(9289728);
    const uint64_t budget = 20 * GIB, transient = 3 * GIB;
    const uint64_t static_weights = 83 * GIB;
    const uint32_t required = 512, desired = 1472, maximum = 92 * 896;
    k3_memory_plan plan;
    CHECK(k3_plan_gguf_memory(budget, transient, slot, required, desired,
                              maximum, static_weights, 0, 0, &plan));
    CHECK(plan.cache_slots == desired);
    check_plan(plan, budget, transient, slot, required, maximum);

    CHECK(k3_plan_gguf_memory(budget, transient, slot, required, desired,
                              maximum, static_weights, 14 * GIB, 0, &plan));
    CHECK(plan.cache_slots == 1618);
    CHECK(plan.cache_bytes <= 14 * GIB);
    CHECK(14 * GIB - plan.cache_bytes < slot);
    check_plan(plan, budget, transient, slot, required, maximum);
    CHECK(k3_plan_gguf_memory(budget, transient, slot, required, desired,
                              maximum, static_weights, 15 * GIB, 0, &plan));
    CHECK(plan.cache_slots == 1733);
    check_plan(plan, budget, transient, slot, required, maximum);
    CHECK(k3_plan_gguf_memory(budget, transient, slot, required, desired,
                              maximum, static_weights, 0, 1800, &plan));
    CHECK(plan.cache_slots == 1800);
    check_plan(plan, budget, transient, slot, required, maximum);

    /* When both forms are supplied, the tighter target wins. */
    CHECK(k3_plan_gguf_memory(budget, transient, slot, required, desired,
                              maximum, static_weights, 14 * GIB, 1800, &plan));
    CHECK(plan.cache_slots == 1618);
    check_plan(plan, budget, transient, slot, required, maximum);
    CHECK(k3_plan_gguf_memory(budget, transient, slot, required, desired,
                              maximum, static_weights, 16 * GIB, 1500, &plan));
    CHECK(plan.cache_slots == 1500);
    check_plan(plan, budget, transient, slot, required, maximum);

    /* Targets remain soft: memory, physical expert count, and the minimum
     * simultaneous working set retain their original authority. */
    CHECK(k3_plan_gguf_memory(budget, transient, slot, required, desired,
                              maximum, static_weights, UINT64_MAX, 0, &plan));
    CHECK(plan.cache_slots == (budget - transient) / slot);
    CHECK(plan.static_bytes < slot);
    check_plan(plan, budget, transient, slot, required, maximum);
    CHECK(k3_plan_gguf_memory(budget, transient, slot, required, desired,
                              1500, static_weights, UINT64_MAX, UINT32_MAX, &plan));
    CHECK(plan.cache_slots == 1500);
    check_plan(plan, budget, transient, slot, required, 1500);
    const uint64_t minimum_budget = transient + required * slot;
    CHECK(k3_plan_gguf_memory(minimum_budget, transient, slot, required, desired,
                              maximum, static_weights, 14 * GIB, 1800, &plan));
    CHECK(plan.cache_slots == required && plan.static_bytes == 0);
    check_plan(plan, minimum_budget, transient, slot, required, maximum);
    CHECK(!k3_plan_gguf_memory(minimum_budget - 1, transient, slot, required,
                               desired, maximum, static_weights, 14 * GIB, 0, &plan));
    CHECK(!k3_plan_gguf_memory(budget, transient, slot, required, desired,
                               maximum, static_weights, required * slot - 1, 0, &plan));
    CHECK(!k3_plan_gguf_memory(budget, transient, slot, required, desired,
                               maximum, static_weights, 0, required - 1, &plan));
    CHECK(plan.cache_slots == 0 && plan.cache_bytes == 0 && plan.static_bytes == 0);

    CHECK(k3_plan_gguf_memory(budget, transient, 0, 0, 0, 0,
                              static_weights, UINT64_MAX, UINT32_MAX, &plan));
    CHECK(plan.cache_slots == 0 && plan.cache_bytes == 0);
    CHECK(plan.static_bytes == budget - transient);
}

static void test_large_machine_and_no_experts(void) {
    const uint64_t slot = 9 * MIB, budget = 400 * GIB, transient = 4 * GIB;
    const uint32_t maximum = 92 * 896;
    k3_memory_plan plan;
    CHECK(k3_plan_gguf_memory(budget, transient, slot, 896, 1472, maximum,
                              20 * GIB, 0, 0, &plan));
    CHECK(plan.cache_slots == (budget - transient - 20 * GIB) / slot);
    CHECK(plan.cache_slots > 1472);
    CHECK(plan.static_bytes >= 20 * GIB);
    check_plan(plan, budget, transient, slot, 896, maximum);
    CHECK(k3_plan_gguf_memory(1024 * GIB, transient, slot, 896, 1472,
                              maximum, 20 * GIB, 0, 0, &plan));
    CHECK(plan.cache_slots == maximum);
    check_plan(plan, 1024 * GIB, transient, slot, 896, maximum);

    /* A layer-0-only shard has no routed experts and needs no cache slots. */
    CHECK(k3_plan_gguf_memory(8 * GIB, 3 * GIB, 0, 0, 0, 0,
                              20 * GIB, 0, 0, &plan));
    CHECK(plan.cache_slots == 0 && plan.cache_bytes == 0);
    CHECK(plan.static_bytes == 5 * GIB);
    check_plan(plan, 8 * GIB, 3 * GIB, 0, 0, 0);
    CHECK(k3_plan_gguf_memory(3 * GIB, 3 * GIB, 0, 0, 0, 0,
                              0, 0, 0, &plan));
    CHECK(plan.static_bytes == 0);
    CHECK(!k3_plan_gguf_memory(3 * GIB - 1, 3 * GIB, 0, 0, 0, 0,
                               0, 0, 0, &plan));
}

static void test_invalid_and_extreme_inputs(void) {
    k3_memory_plan plan = {123, 456, 789};
    CHECK(!k3_plan_gguf_memory(10, 11, 1, 1, 1, 1, 0, 0, 0, &plan));
    CHECK(plan.cache_slots == 0 && plan.cache_bytes == 0 && plan.static_bytes == 0);
    CHECK(!k3_plan_gguf_memory(10, 0, 1, 2, 2, 1, 0, 0, 0, &plan));
    CHECK(!k3_plan_gguf_memory(10, 0, 0, 1, 1, 1, 1, 0, 0, &plan));
    CHECK(!k3_plan_gguf_memory(10, 0, 1, 1, 1, 0, 0, 0, 0, &plan));
    CHECK(!k3_plan_gguf_memory(10, 0, 1, 1, 1, 1, 0, 0, 0, NULL));
    CHECK(k3_plan_gguf_memory(UINT64_MAX, 0, UINT64_MAX, 1,
                              UINT32_MAX, UINT32_MAX, 0, 0, 0, &plan));
    CHECK(plan.cache_slots == 1 && plan.cache_bytes == UINT64_MAX);
    check_plan(plan, UINT64_MAX, 0, UINT64_MAX, 1, UINT32_MAX);
    CHECK(k3_plan_gguf_memory(UINT64_MAX, 0, 1, 1, UINT32_MAX,
                              UINT32_MAX, 0, 0, 0, &plan));
    CHECK(plan.cache_slots == UINT32_MAX);
    check_plan(plan, UINT64_MAX, 0, 1, 1, UINT32_MAX);
}

int main(void) {
    test_step_required();
    test_available();
    test_live_reserve_regression();
    test_alignment();
    test_transient_peak();
    test_32gb_cache_cap();
    test_user_cache_limits();
    test_explicit_cache_targets();
    test_large_machine_and_no_experts();
    test_invalid_and_extreme_inputs();
    puts("K3 memory planner tests passed");
    return 0;
}
