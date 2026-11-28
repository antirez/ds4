/* CPU-only slab planning tests: no Metal objects or model files. */
#include "../k3_streaming_slab.h"
#include <assert.h>
#include <stdio.h>

#define MIB (UINT64_C(1024) * 1024)
#define GIB (UINT64_C(1024) * 1024 * 1024)

static uint32_t plan(uint64_t target, uint64_t slot, uint32_t budget,
                     uint32_t allocated, uint32_t used, uint64_t device,
                     bool automatic) {
    return k3_streaming_slab_slots(target, slot, budget, allocated,
                                   used, 256, device, automatic);
}

/* Simulate slab accounting including the final partial slab. Every successful
 * allocation must stay within both the admitted cache and the device limit. */
static void fills_cache(uint32_t budget, uint64_t slot, uint64_t device,
                        bool shorten_first) {
    uint32_t allocated = 0, used = 0;
    while (allocated < budget) {
        uint32_t slots = plan(256 * MIB, slot, budget, allocated,
                              used, device, true);
        assert(slots > 0);
        assert(slots <= budget - allocated);
        assert((uint64_t)slots <= device / slot);
        if (shorten_first && used == 0) slots /= 2;
        assert(slots > 0);
        allocated += slots;
        used++;
        assert(used <= 256);
    }
    assert(allocated == budget);
}

int main(void) {
    const uint64_t slot = UINT64_C(9289728); /* Real aligned K3 expert slot. */
    const uint32_t ordinary = (uint32_t)(256 * MIB / slot);
    const uint32_t ten_gib = (uint32_t)(10 * GIB / slot);
    assert(ordinary == 28);
    /* Preserve the proven 32 GiB profile and ordinary 128 GiB cache sizes. */
    assert(plan(256 * MIB, slot, ten_gib, 0, 0, 4 * GIB, true) == ordinary);
    assert(plan(256 * MIB, slot, (uint32_t)(50 * GIB / slot),
                0, 0, 4 * GIB, true) == ordinary);
    assert(plan(256 * MIB, slot, ordinary * 256,
                0, 0, 4 * GIB, true) == ordinary);
    /* One extra slot crosses the actual rounded slab capacity, before64GiB. */
    assert(plan(256 * MIB, slot, ordinary * 256 + 1,
                0, 0, 4 * GIB, true) == ordinary + 1);
    assert(plan(256 * MIB, slot, ten_gib, ten_gib - 3,
                42, 4 * GIB, true) == 3);
    fills_cache(ten_gib, slot, 4 * GIB, false);
    fills_cache(ordinary * 256 + 1, slot, 4 * GIB, false);
    fills_cache((uint32_t)(100 * GIB / slot), slot, 4 * GIB, false);
    fills_cache(92u * 896u, slot, 4 * GIB, true);

    /* A smaller successful slab must leave room to replan the remainder. */
    assert(plan(256 * MIB, slot, 10000, 25, 1, 4 * GIB, true) == 40);
    /* Legacy and explicit settings retain original sizing, including a
     * target smaller than one slot and targets larger than device limits. */
    assert(plan(256 * MIB, slot, 10000, 0, 0, 4 * GIB, false) == ordinary);
    assert(plan(4096 * MIB, slot, 10000, 0, 0, 256 * MIB, false)
           == 4096 * MIB / slot);
    assert(plan(0, slot, ten_gib, 0, 0, 0, false) == 1);
    assert(plan(UINT64_MAX, 1, 0, 0, 0, 0, false) == UINT32_MAX);

    /* Device limits win over the desired slab count; an undersized device
     * can exhaust the slab array and fall back, never overallocate it. */
    assert(plan(256 * MIB, slot, 10000, 0, 0, 16 * slot, true) == 16);
    assert(plan(256 * MIB, slot, ten_gib, 0, 0, slot - 1, true) == 0);
    assert(plan(256 * MIB, slot, ten_gib, 0, 0, 0, true) == 0);
    assert(plan(256 * MIB, slot, ten_gib, ten_gib, 42, 4 * GIB, true) == 0);
    assert(plan(256 * MIB, slot, ten_gib, ten_gib + 1, 42, 4 * GIB, true) == 0);
    assert(plan(256 * MIB, slot, ten_gib, 0, 256, 4 * GIB, true) == 0);
    assert(plan(256 * MIB, 0, ten_gib, 0, 0, 4 * GIB, true) == 0);
    assert(k3_streaming_slab_slots(1, 1, 1, 0, 0, 0, 1, true) == 0);

    /* Ceil division and byte limits remain valid at integer boundaries. */
    assert(k3_streaming_slab_slots(1, 1, UINT32_MAX, 0, 0, 1,
                                   UINT64_MAX, true) == UINT32_MAX);
    assert(k3_streaming_slab_slots(0, UINT64_MAX, 2, 0, 0, 1,
                                   UINT64_MAX, true) == 1);
    assert(plan(UINT64_MAX, UINT64_MAX, 2, 0, 0,
                UINT64_MAX, true) == 1);
    assert(plan(UINT64_MAX, 1, UINT32_MAX, UINT32_MAX - 1, 255,
                UINT64_MAX, true) == 1);
    puts("K3 slab policy: adaptive capacity, legacy overrides and device limits passed (CPU only)");
    return 0;
}
