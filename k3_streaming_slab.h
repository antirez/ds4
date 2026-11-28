#ifndef K3_STREAMING_SLAB_H
#define K3_STREAMING_SLAB_H

#include <stdbool.h>
#include <stdint.h>

/* The caller supplies already page-aligned slot bytes and the admitted cache
 * budget. Grow only the automatic K3 slab default: explicit slab settings and
 * other models keep their existing selection/allocation-failure behavior.
 * Recompute from the remaining capacity so a previously shortened allocation
 * does not silently strand the rest of a large cache behind the slab limit. */
static inline uint32_t k3_streaming_slab_slots(uint64_t target_bytes,
                                               uint64_t slot_bytes,
                                               uint32_t cache_slots,
                                               uint32_t allocated_slots,
                                               uint32_t used_slabs,
                                               uint32_t maximum_slabs,
                                               uint64_t maximum_buffer_bytes,
                                               bool automatic) {
    if (!slot_bytes || used_slabs >= maximum_slabs ||
        (cache_slots && allocated_slots >= cache_slots)) return 0;

    uint64_t slots = target_bytes / slot_bytes;
    if (!slots) slots = 1;
    if (slots > UINT32_MAX) slots = UINT32_MAX;

    const uint32_t remaining = cache_slots ? cache_slots - allocated_slots : 0;
    if (automatic) {
        if (remaining) {
            const uint32_t available_slabs = maximum_slabs - used_slabs;
            const uint32_t needed = remaining / available_slabs +
                (remaining % available_slabs != 0);
            if (slots < needed) slots = needed;
        }
        /* A single slot larger than the device limit cannot use slabs. Keep
         * the caller's existing fallback instead of attempting an oversized
         * Metal allocation. Division also prevents byte-count overflow. */
        const uint64_t device_slots = maximum_buffer_bytes / slot_bytes;
        if (slots > device_slots) slots = device_slots;
    }
    if (remaining && slots > remaining) slots = remaining;
    return (uint32_t)slots;
}

#endif
