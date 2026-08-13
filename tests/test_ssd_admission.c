#include "ds4_ssd.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void test_exact_boundary_passes(void) {
    const ds4_ssd_admission_request request = {
        .capacity_bytes = 120,
        .target_mapped_bytes = 10,
        .support_bytes = 20,
        .expert_cache_bytes = 30,
        .prefill_reserve_bytes = 15,
        .kv_bytes = 25,
        .scratch_bytes = 5,
        .safety_headroom_bytes = 15,
    };
    ds4_ssd_admission_result plan;
    assert(ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 120);
    assert(plan.budget_bytes == 120);
}

static void test_insufficient_headroom_fails_closed(void) {
    const ds4_ssd_admission_request request = {
        .capacity_bytes = 119,
        .target_mapped_bytes = 10,
        .support_bytes = 20,
        .expert_cache_bytes = 30,
        .prefill_reserve_bytes = 15,
        .kv_bytes = 25,
        .scratch_bytes = 5,
        .safety_headroom_bytes = 15,
    };
    ds4_ssd_admission_result plan;
    assert(!ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 120);
    assert(plan.budget_bytes == 119);
}

static void test_overflow_fails_closed(void) {
    const ds4_ssd_admission_request request = {
        .capacity_bytes = UINT64_MAX,
        .target_mapped_bytes = UINT64_MAX,
        .support_bytes = 1,
    };
    ds4_ssd_admission_result plan;
    memset(&plan, 0xff, sizeof(plan));
    assert(!ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 0);
    assert(plan.budget_bytes == 0);
}

int main(void) {
    test_exact_boundary_passes();
    test_insufficient_headroom_fails_closed();
    test_overflow_fails_closed();
    puts("test_ssd_admission: PASS");
    return 0;
}
