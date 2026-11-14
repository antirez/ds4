#ifndef K3_MEMORY_H
#define K3_MEMORY_H

#include <stdint.h>

/* GPU tensors, the lazy expert cache, resident weights and transient model
 * views must share one budget. A cache target must never act as a minimum. */
typedef struct {
    uint32_t cache_slots;
    uint64_t cache_bytes;
    uint64_t static_bytes;
} k3_memory_plan;

static inline uint64_t k3_memory_remaining(uint64_t budget, uint64_t used) {
    return budget > used ? budget - used : 0;
}

static inline int k3_memory_step_required(uint64_t transient,
                                         uint64_t cache_budget,
                                         uint64_t locked_cache,
                                         uint64_t live_reserve,
                                         uint64_t *required) {
    if (!required) return 0;
    *required = 0;
    /* Locked cache pages are already absent from live reclaimable memory.
     * Unused slab capacity is not necessarily resident, so reserve all cache
     * bytes that have not yet been locked, including the first complete slab. */
    const uint64_t growth = k3_memory_remaining(cache_budget, locked_cache);
    if (transient > UINT64_MAX - live_reserve) return 0;
    const uint64_t fixed = transient + live_reserve;
    if (growth > UINT64_MAX - fixed) return 0;
    *required = fixed + growth;
    return 1;
}

static inline uint64_t k3_memory_available(uint64_t physical,
                                          uint64_t recommended,
                                          uint64_t allocated,
                                          uint64_t reclaimable,
                                          uint64_t host_reserve,
                                          uint64_t live_floor) {
    /* The host reserve caps the process's share of physical RAM. Live
     * reclaimable memory already excludes other applications' allocations;
     * retain separate emergency headroom instead of reserving the host twice. */
    uint64_t available = k3_memory_remaining(
        k3_memory_remaining(physical, host_reserve), allocated);
    const uint64_t live = k3_memory_remaining(reclaimable, live_floor);
    if (available > live) available = live;
    if (recommended != 0) {
        const uint64_t gpu = k3_memory_remaining(recommended, allocated);
        if (available > gpu) available = gpu;
    }
    return available;
}

static inline int k3_memory_align(uint64_t bytes, uint64_t page,
                                 uint64_t *aligned) {
    if (!aligned || page == 0 || (page & (page - 1u)) != 0 ||
        bytes > UINT64_MAX - (page - 1u)) return 0;
    *aligned = (bytes + page - 1u) & ~(page - 1u);
    return 1;
}

/* Layer prefetch overlaps adjacent layers; output-head prefetch overlaps the
 * final layer. Output views are released before the next evaluation begins.
 * Every supplied size must already include its mapping's page alignment. */
static inline int k3_memory_transient_peak(const uint64_t *layers,
                                          uint32_t count,
                                          uint64_t head,
                                          uint64_t output,
                                          uint64_t slack,
                                          uint64_t *bytes) {
    if (!bytes) return 0;
    *bytes = 0;
    if (!layers || count == 0) return 0;
    uint64_t peak = output;
    for (uint32_t i = 0; i < count; i++) {
        if (peak < layers[i]) peak = layers[i];
        if (i != 0) {
            if (layers[i - 1] > UINT64_MAX - layers[i]) return 0;
            const uint64_t pair = layers[i - 1] + layers[i];
            if (peak < pair) peak = pair;
        }
    }
    if (layers[count - 1] > UINT64_MAX - head) return 0;
    const uint64_t last_and_head = layers[count - 1] + head;
    if (peak < last_and_head) peak = last_and_head;
    if (peak > UINT64_MAX - slack) return 0;
    *bytes = peak + slack;
    return 1;
}

static inline int k3_plan_gguf_memory(uint64_t budget,
                                     uint64_t transient_bytes,
                                     uint64_t slot_bytes,
                                     uint32_t required_slots,
                                     uint32_t desired_slots,
                                     uint32_t maximum_slots,
                                     uint64_t static_bytes,
                                     uint64_t requested_bytes,
                                     uint32_t requested_slots,
                                     k3_memory_plan *plan) {
    if (!plan) return 0;
    *plan = (k3_memory_plan){0};
    if (budget < transient_bytes || required_slots > maximum_slots) return 0;
    budget -= transient_bytes;
    if (maximum_slots == 0) {
        plan->static_bytes = budget;
        return 1;
    }
    if (slot_bytes == 0) return 0;
    uint64_t limit = budget / slot_bytes;
    if (limit > maximum_slots) limit = maximum_slots;
    if (requested_bytes != 0 && limit > requested_bytes / slot_bytes)
        limit = requested_bytes / slot_bytes;
    if (requested_slots != 0 && limit > requested_slots)
        limit = requested_slots;
    if (limit < required_slots) return 0;

    /* Larger machines can use spare space after keeping the static weights.
     * Small machines retain only the cache that fits, with enough slots for
     * all experts selected by one prefill batch. */
    uint64_t slots = k3_memory_remaining(budget, static_bytes) / slot_bytes;
    if (slots < desired_slots) slots = desired_slots;
    if (slots < required_slots) slots = required_slots;
    if (slots > limit) slots = limit;
    plan->cache_slots = (uint32_t)slots;
    plan->cache_bytes = slots * slot_bytes;
    plan->static_bytes = budget - plan->cache_bytes;
    return 1;
}

#endif
