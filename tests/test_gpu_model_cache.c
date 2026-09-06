/* Unit test for the per-device selective model cache
 * (mgpu-selective-model-cache).
 *
 * Exercises:
 *   - ds4_gpu_device_cache_tensors with disjoint ranges on device 0
 *   - ds4_gpu_lookup_cache at range bases and at interior offsets
 *     (proves the subrange pointer offset arithmetic is right)
 *   - device-id resolution
 *   - selected-expert batched-I/O policy, planner, scatter and byte oracle
 *   - on multi-GPU boxes: caching on device 1 and active-device
 *     preference in lookup */

#include "ds4_gpu.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                \
    do {                                                                \
        if (!(cond)) {                                                  \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);   \
            return 1;                                                   \
        }                                                               \
    } while (0)

int main(void) {
    int enabled = -1;
    int required = -1;
    int oracle = -1;
    int fused = -1;
    int owned_forced = -1;
    int multi_gpu_forced = -1;
    int capture = -1;
    CHECK(!ds4_cuda_test_q8_hc_expand_env_value(NULL) &&
          !ds4_cuda_test_q8_hc_expand_env_value("") &&
          !ds4_cuda_test_q8_hc_expand_env_value("0") &&
          !ds4_cuda_test_q8_hc_expand_env_value("false") &&
          !ds4_cuda_test_q8_hc_expand_env_value("NO") &&
          !ds4_cuda_test_q8_hc_expand_env_value("off") &&
          ds4_cuda_test_q8_hc_expand_env_value("1") &&
          ds4_cuda_test_q8_hc_expand_env_value("true"),
          "Q8 HC value-aware environment parser");
    CHECK(ds4_cuda_test_q8_hc_expand_policy(
              0, 0, 1, 0, 0, &fused, &owned_forced,
              &multi_gpu_forced, &capture) &&
          fused == 1 && owned_forced == 0 && multi_gpu_forced == 0 &&
          capture == 0,
          "Q8 HC defaults to fused");
    CHECK(ds4_cuda_test_q8_hc_expand_policy(
              0, 1, 1, 0, 0, &fused, &owned_forced,
              &multi_gpu_forced, &capture) &&
          fused == 0 && owned_forced == 0 && multi_gpu_forced == 0,
          "Q8 HC split opt-in on single GPU");
    CHECK(ds4_cuda_test_q8_hc_expand_policy(
              1, 1, 1, 0, 0, &fused, &owned_forced,
              &multi_gpu_forced, &capture) &&
          fused == 1 && owned_forced == 0 && multi_gpu_forced == 0,
          "Q8 HC force-fused dominates conflicting split request");
    CHECK(ds4_cuda_test_q8_hc_expand_policy(
              0, 1, 2, 0, 0, &fused, &owned_forced,
              &multi_gpu_forced, &capture) &&
          fused == 1 && owned_forced == 0 && multi_gpu_forced == 1,
          "Q8 HC multi-GPU remains fused");
    CHECK(ds4_cuda_test_q8_hc_expand_policy(
              0, 1, 1, 1, 0, &fused, &owned_forced,
              &multi_gpu_forced, &capture) &&
          fused == 1 && owned_forced == 1 && multi_gpu_forced == 0,
          "Q8 HC owned dispatch remains fused");
    CHECK(ds4_cuda_test_q8_hc_expand_policy(
              0, 1, 1, 0, 1, &fused, &owned_forced,
              &multi_gpu_forced, &capture) &&
          fused == 0 && capture == 1,
          "Q8 HC split graph-capture policy matches eager policy");
    CHECK(!ds4_cuda_test_stream_selected_batch_env_value(NULL) &&
          !ds4_cuda_test_stream_selected_batch_env_value("") &&
          !ds4_cuda_test_stream_selected_batch_env_value("0") &&
          !ds4_cuda_test_stream_selected_batch_env_value("false") &&
          !ds4_cuda_test_stream_selected_batch_env_value("FALSE") &&
          !ds4_cuda_test_stream_selected_batch_env_value("no") &&
          !ds4_cuda_test_stream_selected_batch_env_value("off") &&
          ds4_cuda_test_stream_selected_batch_env_value("1") &&
          ds4_cuda_test_stream_selected_batch_env_value("true"),
          "selected-expert batched-I/O value-aware environment parser");
    CHECK(ds4_cuda_test_stream_selected_batch_policy(
              1, 0, 0, 0, &enabled, &required, &oracle) &&
          enabled == 1 && required == 0 && oracle == 0,
          "selected-expert batched-I/O enable policy");
    CHECK(ds4_cuda_test_stream_selected_batch_policy(
              0, 0, 1, 0, &enabled, &required, &oracle) &&
          enabled == 1 && required == 1 && oracle == 0,
          "selected-expert batched-I/O require policy");
    CHECK(ds4_cuda_test_stream_selected_batch_policy(
              1, 1, 1, 1, &enabled, &required, &oracle) &&
          enabled == 0 && required == 0 && oracle == 0,
          "selected-expert batched-I/O disable-dominant policy");
    CHECK(!ds4_cuda_test_stream_selected_event_env_value(NULL) &&
          !ds4_cuda_test_stream_selected_event_env_value("") &&
          !ds4_cuda_test_stream_selected_event_env_value("0") &&
          !ds4_cuda_test_stream_selected_event_env_value("false") &&
          !ds4_cuda_test_stream_selected_event_env_value("NO") &&
          !ds4_cuda_test_stream_selected_event_env_value("off") &&
          ds4_cuda_test_stream_selected_event_env_value("1") &&
          ds4_cuda_test_stream_selected_event_env_value("true"),
          "selected-expert event-pipeline value-aware environment parser");
    CHECK(ds4_cuda_test_stream_selected_event_pipeline_policy(
              1, 0, 0, 0, &enabled, &required, &oracle) &&
          enabled == 1 && required == 0 && oracle == 0,
          "selected-expert event-pipeline enable policy");
    CHECK(ds4_cuda_test_stream_selected_event_pipeline_policy(
              0, 0, 1, 0, &enabled, &required, &oracle) &&
          enabled == 1 && required == 1 && oracle == 0,
          "selected-expert event-pipeline require policy");
    CHECK(ds4_cuda_test_stream_selected_event_pipeline_policy(
              0, 0, 0, 1, &enabled, &required, &oracle) &&
          enabled == 1 && required == 0 && oracle == 1,
          "selected-expert event-pipeline oracle policy");
    CHECK(ds4_cuda_test_stream_selected_event_pipeline_policy(
              1, 1, 1, 1, &enabled, &required, &oracle) &&
          enabled == 0 && required == 0 && oracle == 0,
          "selected-expert event-pipeline disable-dominant policy");
    int stats = -1;
    CHECK(!ds4_cuda_test_stream_expert_persistent_env_value(NULL) &&
          !ds4_cuda_test_stream_expert_persistent_env_value("") &&
          !ds4_cuda_test_stream_expert_persistent_env_value("0") &&
          !ds4_cuda_test_stream_expert_persistent_env_value("false") &&
          !ds4_cuda_test_stream_expert_persistent_env_value("NO") &&
          !ds4_cuda_test_stream_expert_persistent_env_value("off") &&
          ds4_cuda_test_stream_expert_persistent_env_value("1") &&
          ds4_cuda_test_stream_expert_persistent_env_value("true"),
          "persistent expert planner value-aware environment parser");
    CHECK(ds4_cuda_test_stream_expert_persistent_policy(
              1, 0, 0, 0, 0,
              &enabled, &required, &stats, &oracle) &&
          enabled == 1 && required == 0 && stats == 0 && oracle == 0,
          "persistent expert planner enable policy");
    CHECK(ds4_cuda_test_stream_expert_persistent_policy(
              0, 0, 1, 1, 1,
              &enabled, &required, &stats, &oracle) &&
          enabled == 1 && required == 1 && stats == 1 && oracle == 1,
          "persistent expert planner require/stats/oracle policy");
    CHECK(ds4_cuda_test_stream_expert_persistent_policy(
              0, 0, 0, 1, 0,
              &enabled, &required, &stats, &oracle) &&
          enabled == 0 && required == 0 && stats == 1 && oracle == 0,
          "persistent expert planner stats-only policy");
    CHECK(ds4_cuda_test_stream_expert_persistent_policy(
              1, 1, 1, 1, 1,
              &enabled, &required, &stats, &oracle) &&
          enabled == 0 && required == 0 && stats == 0 && oracle == 0,
          "persistent expert planner disable-dominant policy");
    ds4_cuda_stream_expert_persistent_report persistent_before;
    memset(&persistent_before, 0, sizeof(persistent_before));
    ds4_cuda_stream_expert_persistent_get_report(&persistent_before);
    CHECK(ds4_cuda_test_stream_expert_persistent_planner(),
          "persistent expert LRU/free-list transaction planner oracle");
    ds4_cuda_stream_expert_persistent_report persistent_after;
    memset(&persistent_after, 0, sizeof(persistent_after));
    ds4_cuda_stream_expert_persistent_get_report(&persistent_after);
    CHECK(persistent_after.oracle_runs == persistent_before.oracle_runs + 1u &&
          persistent_after.oracle_failures ==
              persistent_before.oracle_failures &&
          persistent_after.plan_attempts > persistent_before.plan_attempts &&
          persistent_after.plans_built > persistent_before.plans_built &&
          persistent_after.commits > persistent_before.commits &&
          persistent_after.rollbacks > persistent_before.rollbacks &&
          persistent_after.hits > persistent_before.hits &&
          persistent_after.misses > persistent_before.misses &&
          persistent_after.duplicates > persistent_before.duplicates &&
          persistent_after.free_assignments >
              persistent_before.free_assignments &&
          persistent_after.evictions > persistent_before.evictions &&
          persistent_after.rejects > persistent_before.rejects &&
          persistent_after.budget_rejects >
              persistent_before.budget_rejects &&
          persistent_after.class_rejects >
              persistent_before.class_rejects &&
          persistent_after.protected_rejects >
              persistent_before.protected_rejects &&
          persistent_after.key_misses > persistent_before.key_misses &&
          persistent_after.overflow_rejects >
              persistent_before.overflow_rejects,
          "persistent expert planner invariant coverage counters");
    CHECK(ds4_cuda_test_iq2_ssd_grouped_policy(
              1, 0, 0, 0, &enabled, &required, &stats) &&
          enabled == 1 && required == 0 && stats == 0,
          "IQ2 SSD grouped-MMQ enable policy");
    CHECK(ds4_cuda_test_iq2_ssd_grouped_policy(
              0, 0, 1, 1, &enabled, &required, &stats) &&
          enabled == 1 && required == 1 && stats == 1,
          "IQ2 SSD grouped-MMQ require and stats policy");
    CHECK(ds4_cuda_test_iq2_ssd_grouped_policy(
              1, 1, 1, 1, &enabled, &required, &stats) &&
          enabled == 0 && required == 0 && stats == 0,
          "IQ2 SSD grouped-MMQ disable-dominant policy");
    CHECK(ds4_cuda_test_iq2_ssd_grouped_eligibility(
              1, 1, 1, 1, 0, 0, 0, 1,
              32u, 6u, 1, 1, 1),
          "IQ2 SSD grouped-MMQ canonical eligibility");
    CHECK(!ds4_cuda_test_iq2_ssd_grouped_candidate(
               1, 1, 1, 0, 1u, 6u, 1, 1, 1) &&
          !ds4_cuda_test_iq2_ssd_grouped_candidate(
               1, 1, 1, 0, 31u, 6u, 1, 1, 1) &&
          ds4_cuda_test_iq2_ssd_grouped_candidate(
               1, 1, 1, 0, 32u, 6u, 1, 1, 1),
          "IQ2 SSD grouped-MMQ REQUIRE candidate excludes decode/tail");
    CHECK(!ds4_cuda_test_iq2_ssd_grouped_candidate(
               1, 1, 1, 0, 32u, 5u, 1, 1, 1) &&
          !ds4_cuda_test_iq2_ssd_grouped_candidate(
               1, 1, 1, 0, 32u, 6u, 0, 1, 1) &&
          !ds4_cuda_test_iq2_ssd_grouped_candidate(
               1, 1, 1, 0, 32u, 6u, 1, 0, 1) &&
          !ds4_cuda_test_iq2_ssd_grouped_candidate(
               1, 1, 1, 0, 32u, 6u, 1, 1, 0),
          "IQ2 SSD grouped-MMQ REQUIRE excludes non-candidate layouts");
    CHECK(!ds4_cuda_test_iq2_ssd_grouped_eligibility(
               1, 1, 1, 1, 0, 0, 0, 1,
               31u, 6u, 1, 1, 1) &&
          !ds4_cuda_test_iq2_ssd_grouped_eligibility(
               1, 1, 1, 1, 0, 0, 0, 1,
               32u, 5u, 1, 1, 1) &&
          !ds4_cuda_test_iq2_ssd_grouped_eligibility(
               1, 1, 1, 1, 0, 0, 0, 1,
               32u, 6u, 0, 1, 1) &&
          !ds4_cuda_test_iq2_ssd_grouped_eligibility(
               1, 1, 1, 1, 1, 0, 0, 1,
               32u, 6u, 1, 1, 1) &&
          !ds4_cuda_test_iq2_ssd_grouped_eligibility(
               1, 1, 1, 1, 0, 0, 1, 1,
               32u, 6u, 1, 1, 1) &&
          !ds4_cuda_test_iq2_ssd_grouped_eligibility(
               1, 1, 1, 1, 0, 0, 0, 1,
               32u, 6u, 1, 1, 0),
          "IQ2 SSD grouped-MMQ exclusion matrix");
    const uint64_t iq2_row = 16u * 66u;
    const uint64_t iq2_expert = 2048u * iq2_row;
    const uint64_t q2_row = 8u * 84u;
    const uint64_t q2_expert = 4096u * q2_row;
    CHECK(ds4_cuda_test_iq2_ssd_grouped_raw_layout(
              16u, 10u, iq2_expert, iq2_row,
              q2_expert, q2_row, 4096u, 2048u, 4096u),
          "IQ2 SSD grouped-MMQ canonical raw layout");
    CHECK(!ds4_cuda_test_iq2_ssd_grouped_raw_layout(
               12u, 10u, iq2_expert, iq2_row,
               q2_expert, q2_row, 4096u, 2048u, 4096u) &&
          !ds4_cuda_test_iq2_ssd_grouped_raw_layout(
               16u, 10u, iq2_expert, iq2_row + 2u,
               q2_expert, q2_row, 4096u, 2048u, 4096u) &&
          !ds4_cuda_test_iq2_ssd_grouped_raw_layout(
               16u, 10u, iq2_expert, iq2_row,
               q2_expert + 84u, q2_row, 4096u, 2048u, 4096u),
          "IQ2 SSD grouped-MMQ raw-layout rejection matrix");
    CHECK(ds4_cuda_test_stream_selected_batch_plan(),
          "selected-expert batched-I/O planner");

    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    fprintf(stderr, "test_gpu_model_cache: %d CUDA devices visible\n",
            dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "no CUDA devices\n");
        return 0;
    }

    CHECK(ds4_gpu_init(), "ds4_gpu_init");
    ds4_cuda_q8_hc_expand_report q8_hc_before;
    memset(&q8_hc_before, 0, sizeof(q8_hc_before));
    ds4_cuda_q8_hc_expand_get_report(&q8_hc_before);
    CHECK(ds4_cuda_test_q8_hc_expand_oracle(),
          "Q8 HC fused/split graph-capture parity oracle");
    ds4_cuda_q8_hc_expand_report q8_hc_after;
    memset(&q8_hc_after, 0, sizeof(q8_hc_after));
    ds4_cuda_q8_hc_expand_get_report(&q8_hc_after);
    CHECK(q8_hc_after.oracle_runs == q8_hc_before.oracle_runs + 1u &&
          q8_hc_after.oracle_failures == q8_hc_before.oracle_failures,
          "Q8 HC oracle coverage counters");
    ds4_gpu_set_streaming_expert_cache_budget(3u);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(40u);
    CHECK(ds4_gpu_stream_expert_cache_configured_count() == 0u &&
          ds4_gpu_stream_expert_cache_budget_for_expert_size(16u, 8u) == 0u &&
          ds4_gpu_stream_expert_cache_current_count() == 0u,
          "persistent expert arena remains runtime-inert before loader wiring");
    ds4_cuda_stream_expert_persistent_report arena_before;
    memset(&arena_before, 0, sizeof(arena_before));
    ds4_cuda_stream_expert_persistent_get_report(&arena_before);
    CHECK(ds4_cuda_test_stream_expert_persistent_arena(),
          "persistent expert device arena allocation/reuse/reinit oracle");
    ds4_cuda_stream_expert_persistent_report arena_after;
    memset(&arena_after, 0, sizeof(arena_after));
    ds4_cuda_stream_expert_persistent_get_report(&arena_after);
    CHECK(arena_after.arena_allocations >=
              arena_before.arena_allocations + 2u &&
          arena_after.arena_reuses > arena_before.arena_reuses &&
          arena_after.arena_releases >= arena_before.arena_releases + 2u &&
          arena_after.arena_failures == arena_before.arena_failures &&
          arena_after.arena_oracle_runs ==
              arena_before.arena_oracle_runs + 1u &&
          arena_after.arena_oracle_failures ==
              arena_before.arena_oracle_failures,
          "persistent expert device arena coverage counters");
    ds4_cuda_stream_expert_persistent_report runtime_before;
    memset(&runtime_before, 0, sizeof(runtime_before));
    ds4_cuda_stream_expert_persistent_get_report(&runtime_before);
    CHECK(ds4_cuda_test_stream_expert_persistent_runtime(),
          "persistent expert cold/hit/mixed/eviction/fault runtime oracle");
    ds4_cuda_stream_expert_persistent_report runtime_after;
    memset(&runtime_after, 0, sizeof(runtime_after));
    ds4_cuda_stream_expert_persistent_get_report(&runtime_after);
    CHECK(runtime_after.runtime_oracle_runs ==
              runtime_before.runtime_oracle_runs + 1u &&
          runtime_after.runtime_oracle_failures ==
              runtime_before.runtime_oracle_failures &&
          runtime_after.epochs_attempted >=
              runtime_before.epochs_attempted + 7u &&
          runtime_after.epochs_published ==
              runtime_before.epochs_published + 5u &&
          runtime_after.all_hit_epochs ==
              runtime_before.all_hit_epochs + 2u &&
          runtime_after.miss_epochs == runtime_before.miss_epochs + 3u &&
          runtime_after.miss_experts >= runtime_before.miss_experts + 5u &&
          runtime_after.weight_bytes_uploaded >=
              runtime_before.weight_bytes_uploaded + 200u &&
          runtime_after.remap_bytes_uploaded >=
              runtime_before.remap_bytes_uploaded + 44u &&
          runtime_after.upload_failures ==
              runtime_before.upload_failures + 1u &&
          runtime_after.slot_invalidations >=
              runtime_before.slot_invalidations + 2u &&
          runtime_after.poisons >= runtime_before.poisons + 1u &&
          runtime_after.persistent_dispatches >=
              runtime_before.persistent_dispatches + 5u,
          "persistent expert runtime coverage counters");
    ds4_cuda_iq2_ssd_grouped_report lease_before;
    memset(&lease_before, 0, sizeof(lease_before));
    ds4_cuda_iq2_ssd_grouped_get_report(&lease_before);
    CHECK(ds4_cuda_test_iq2_ssd_grouped_lease(),
          "IQ2 SSD compact-binding consume/reuse lease oracle");
    ds4_cuda_iq2_ssd_grouped_report lease_after;
    memset(&lease_after, 0, sizeof(lease_after));
    ds4_cuda_iq2_ssd_grouped_get_report(&lease_after);
    CHECK(lease_after.lease_records > lease_before.lease_records &&
          lease_after.lease_waits > lease_before.lease_waits &&
          lease_after.lease_drains > lease_before.lease_drains,
          "IQ2 SSD compact-binding lease coverage counters");
    CHECK(ds4_cuda_test_stream_selected_event_pipeline(),
          "selected-expert compute/readback/upload event ordering oracle");
    ds4_cuda_stream_selected_event_pipeline_report event_report;
    memset(&event_report, 0, sizeof(event_report));
    ds4_cuda_stream_selected_event_pipeline_get_report(&event_report);
    CHECK(event_report.candidates >= 1 &&
          event_report.signals >= 1 &&
          event_report.readbacks >= 1 &&
          event_report.uploads >= 1 &&
          event_report.compute_waits >= 1 &&
          event_report.oracle_runs >= 1 &&
          event_report.oracle_failures == 0,
          "selected-expert event-pipeline coverage counters");
    CHECK(ds4_cuda_test_stream_selected_batch_copy(),
          "selected-expert batched-I/O scatter + byte oracle");
    ds4_cuda_stream_selected_batch_io_report batch_report;
    memset(&batch_report, 0, sizeof(batch_report));
    ds4_cuda_stream_selected_batch_io_get_report(&batch_report);
    CHECK(batch_report.oracle_runs >= 2 &&
          batch_report.oracle_failures == 0 &&
          batch_report.tasks >= 9 &&
          batch_report.segments >= batch_report.tasks &&
          batch_report.reads >= 7 && batch_report.bytes >= 18u * 1024u,
          "selected-expert batched-I/O coverage counters");

    /* Build a synthetic 1-MiB "model" in host memory. */
    const size_t total = 1024 * 1024;
    void *host = NULL;
    CHECK(cudaMallocHost(&host, total) == cudaSuccess, "cudaMallocHost");
    unsigned char *bytes = (unsigned char *)host;
    for (size_t i = 0; i < total; i++) bytes[i] = (unsigned char)(i & 0xff);
    CHECK(ds4_gpu_set_model_map(host, total), "set_model_map");

    /* Three disjoint ranges on device 0. */
    ds4_tensor_range ranges[3];
    ranges[0].source_offset = 0;          ranges[0].bytes = 256 * 1024; ranges[0].target_device = 0;
    ranges[1].source_offset = 384 * 1024; ranges[1].bytes = 128 * 1024; ranges[1].target_device = 0;
    ranges[2].source_offset = 768 * 1024; ranges[2].bytes = 256 * 1024; ranges[2].target_device = 0;

    CHECK(ds4_gpu_device_cache_tensors(0, ranges, 3) == 0,
          "device_cache_tensors dev 0 (3 ranges)");

    /* Base lookups + interior offset arithmetic. */
    int dev = -1; void *base0 = NULL, *interior0 = NULL;
    CHECK(ds4_gpu_lookup_cache(0, 1024, &dev, &base0) == 1, "lookup range 0 base");
    CHECK(dev == 0, "range 0 device");
    CHECK(base0 != NULL, "range 0 ptr");
    /* An interior offset must return base0 + delta. */
    CHECK(ds4_gpu_lookup_cache(100, 1024, &dev, &interior0) == 1, "lookup range 0 interior");
    CHECK(dev == 0, "range 0 interior device");
    CHECK(interior0 == (char *)base0 + 100, "interior offset arithmetic");

    void *base1 = NULL;
    CHECK(ds4_gpu_lookup_cache(384 * 1024, 1024, &dev, &base1) == 1, "lookup range 1 base");
    CHECK(dev == 0, "range 1 device");
    /* Interior of range 1 at +200 bytes should be base1 + 200. */
    void *interior1 = NULL;
    CHECK(ds4_gpu_lookup_cache(384 * 1024 + 200, 1024, &dev, &interior1) == 1, "lookup range 1 interior");
    CHECK(interior1 == (char *)base1 + 200, "range 1 interior offset");

    void *base2 = NULL;
    CHECK(ds4_gpu_lookup_cache(900 * 1024, 1024, &dev, &base2) == 1, "lookup range 2");
    CHECK(dev == 0 && base2 != NULL, "range 2 device+ptr");

    /* Convenience wrapper. */
    CHECK(ds4_gpu_lookup_cache_device(0, 1024) == 0, "lookup_device range 0");

    /* Lookup must be overflow-safe: a query with bytes=UINT64_MAX must
     * not wrap around into a false hit. */
    int dev_overflow = -1; void *ptr_overflow = NULL;
    int hit = ds4_gpu_lookup_cache(100, UINT64_MAX, &dev_overflow, &ptr_overflow);
    /* Either miss (preferred), or hit but the path must NOT have wrapped.
     * Accept miss only — a wrap-induced hit would be a bug. */
    CHECK(hit == 0, "lookup with bytes=UINT64_MAX does not wrap into a false hit");

    /* Bounds-check: ranges that overflow the model must be rejected
     * before any allocation. */
    ds4_tensor_range bad_overflow = { 0, total + 1, 0 };
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_overflow, 1) != 0,
          "overflow range rejected");
    ds4_tensor_range bad_offset = { total + 1, 16, 0 };
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_offset, 1) != 0,
          "out-of-range offset rejected");
    ds4_tensor_range bad_wrap = { total - 4, UINT64_MAX, 0 };
    CHECK(ds4_gpu_device_cache_tensors(0, &bad_wrap, 1) != 0,
          "wrap-around range rejected");

    /* Gap not covered by selective ranges. The legacy chunked path may
     * happen to cover it (it caches the whole model span); accept either
     * outcome, but if it returns 1 the device must be 0. */
    int dev_gap = -1; void *ptr_gap = NULL;
    int gap_hit = ds4_gpu_lookup_cache(300 * 1024, 1024, &dev_gap, &ptr_gap);
    if (gap_hit) {
        CHECK(dev_gap == 0, "gap fallback device");
    }

    if (dev_count >= 2) {
        /* Cache a different range on device 1. */
        ds4_tensor_range r2;
        r2.source_offset = 256 * 1024;
        r2.bytes         = 128 * 1024;
        r2.target_device = 1;
        CHECK(ds4_gpu_device_cache_tensors(1, &r2, 1) == 0, "cache dev 1");

        /* With cudaGetDevice() == 1, the lookup should resolve to dev 1
         * for this range (the only selective entry covering it). */
        (void)cudaSetDevice(1);
        int dd = -1; void *pp = NULL;
        CHECK(ds4_gpu_lookup_cache(256 * 1024 + 10, 1024, &dd, &pp) == 1,
              "lookup dev 1");
        CHECK(dd == 1, "lookup resolves to dev 1");
        CHECK(pp != NULL, "lookup ptr non-null");
        (void)cudaSetDevice(0);
    }

    ds4_gpu_cleanup();
    (void)cudaFreeHost(host);
    fprintf(stderr, "test_gpu_model_cache PASS (devs=%d)\n", dev_count);
    return 0;
}
