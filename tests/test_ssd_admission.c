#include "ds4_ssd.h"

#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static ds4_ssd_admission_request base_request(uint64_t capacity_bytes) {
    const ds4_ssd_admission_request request = {
        .capacity_bytes = capacity_bytes,
        .target_mapped_bytes = 10,
        .support_bytes = 20,
        .expert_cache_bytes = 30,
        .prefill_reserve_bytes = 15,
        .session_kv_bytes = 25,
        .session_context_scratch_bytes = 5,
        .session_graph_bytes = 17,
        .session_speculative_bytes = 7,
        .session_host_bytes = 3,
        .session_prefill_workspace_bytes = 11,
        .shared_prefill_workspace_bytes = 13,
        .safety_headroom_bytes = 15,
    };
    return request;
}

static void test_exact_boundary_passes(void) {
    const ds4_ssd_admission_request request = base_request(171);
    ds4_ssd_admission_result plan;
    assert(ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 171);
    assert(plan.budget_bytes == 171);
    assert(plan.total_session_kv_bytes == 25);
    assert(plan.total_session_context_scratch_bytes == 5);
    assert(plan.total_session_graph_bytes == 17);
    assert(plan.total_session_speculative_bytes == 7);
    assert(plan.total_session_host_bytes == 3);
    assert(plan.total_session_prefill_workspace_bytes == 11);
    assert(plan.shared_prefill_workspace_bytes == 13);
}

static void test_insufficient_headroom_fails_closed(void) {
    const ds4_ssd_admission_request request = base_request(170);
    ds4_ssd_admission_result plan;
    assert(!ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 171);
    assert(plan.budget_bytes == 170);
}

static void test_independent_sessions_scale_every_session_category(void) {
    ds4_ssd_admission_request request = base_request(307);
    request.session_count = 3;

    ds4_ssd_admission_result plan;
    assert(ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 307);
    assert(plan.total_session_kv_bytes == 75);
    assert(plan.total_session_context_scratch_bytes == 15);
    assert(plan.total_session_graph_bytes == 51);
    assert(plan.total_session_speculative_bytes == 21);
    assert(plan.total_session_host_bytes == 9);
    assert(plan.total_session_prefill_workspace_bytes == 33);

    request.capacity_bytes = 306;
    assert(!ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 307);
    assert(plan.budget_bytes == 306);
}

static void test_shared_workspace_is_a_distinct_once_only_category(void) {
    ds4_ssd_admission_request request = base_request(272);
    request.session_count = 3;
    request.session_prefill_workspace_bytes = 0;
    request.shared_prefill_workspace_bytes = 11;

    ds4_ssd_admission_result plan;
    assert(ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 272);
    assert(plan.total_session_kv_bytes == 75);
    assert(plan.total_session_context_scratch_bytes == 15);
    assert(plan.total_session_graph_bytes == 51);
    assert(plan.total_session_prefill_workspace_bytes == 0);
    assert(plan.shared_prefill_workspace_bytes == 11);

    request.session_prefill_workspace_bytes = 11;
    request.shared_prefill_workspace_bytes = 0;
    assert(!ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 294);
    assert(plan.total_session_context_scratch_bytes == 15);
    assert(plan.total_session_graph_bytes == 51);
    assert(plan.total_session_prefill_workspace_bytes == 33);
    assert(plan.shared_prefill_workspace_bytes == 0);
}

static void test_each_session_category_multiplication_overflow_fails(void) {
    static const size_t offsets[] = {
        offsetof(ds4_ssd_admission_request, session_kv_bytes),
        offsetof(ds4_ssd_admission_request, session_context_scratch_bytes),
        offsetof(ds4_ssd_admission_request, session_graph_bytes),
        offsetof(ds4_ssd_admission_request, session_speculative_bytes),
        offsetof(ds4_ssd_admission_request, session_host_bytes),
        offsetof(ds4_ssd_admission_request, session_prefill_workspace_bytes),
    };
    for (size_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); i++) {
        ds4_ssd_admission_request request = {0};
        request.capacity_bytes = UINT64_MAX;
        request.session_count = 2;
        uint64_t *field = (uint64_t *)((uint8_t *)&request + offsets[i]);
        *field = UINT64_MAX / 2 + 1;

        ds4_ssd_admission_result plan;
        memset(&plan, 0xff, sizeof(plan));
        assert(!ds4_ssd_admission_plan(&request, &plan));
        assert(plan.required_bytes == 0);
        assert(plan.budget_bytes == 0);
    }
}

static void test_ledger_addition_overflow_fails_closed(void) {
    const ds4_ssd_admission_request request = {
        .capacity_bytes = UINT64_MAX,
        .support_bytes = 1,
        .shared_prefill_workspace_bytes = UINT64_MAX,
    };
    ds4_ssd_admission_result plan;
    memset(&plan, 0xff, sizeof(plan));
    assert(!ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 0);
    assert(plan.budget_bytes == 0);
}

static void test_prelocked_memory_is_admitted_at_exact_boundary(void) {
    ds4_ssd_admission_request request = base_request(190);
    request.prelocked_bytes = 20;

    ds4_ssd_admission_result plan;
    assert(!ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 191);
    assert(plan.budget_bytes == 190);

    request.capacity_bytes = 191;
    assert(ds4_ssd_admission_plan(&request, &plan));
    assert(plan.required_bytes == 191);
    assert(plan.budget_bytes == 191);
}

int main(void) {
    test_exact_boundary_passes();
    test_insufficient_headroom_fails_closed();
    test_independent_sessions_scale_every_session_category();
    test_shared_workspace_is_a_distinct_once_only_category();
    test_each_session_category_multiplication_overflow_fails();
    test_ledger_addition_overflow_fails_closed();
    test_prelocked_memory_is_admitted_at_exact_boundary();
    puts("test_ssd_admission: PASS");
    return 0;
}
