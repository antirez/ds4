#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __APPLE__

enum {
    N_TOTAL_EXPERT = 6,
    N_SELECTED = 6,
    CACHE_BUDGET = 3,
};

static const uint64_t GATE_EXPERT_BYTES = UINT64_C(2162688);
static const uint64_t DOWN_EXPERT_BYTES = UINT64_C(2752512);
static const uint64_t TOTAL_EXPERT_BYTES = UINT64_C(7077888);

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static int check_policy(void) {
    int ok = 1;
#define CHECK_POLICY(expected, ssd, gate, down, enable, disable) do {       \
    const int got = ds4_gpu_test_stream_expert_live_index_policy(          \
            (ssd), (gate), (down), (enable), (disable));                   \
    if (got != (expected)) {                                                \
        fprintf(stderr,                                                     \
                "policy FAIL ssd=%d gate=%llu down=%llu enable=%d "        \
                "disable=%d got=%d expected=%d\n",                         \
                (ssd), (unsigned long long)(gate),                          \
                (unsigned long long)(down), (enable), (disable),            \
                got, (expected));                                           \
        ok = 0;                                                             \
    }                                                                       \
} while (0)

    CHECK_POLICY(1, 1, GATE_EXPERT_BYTES, DOWN_EXPERT_BYTES, -1, -1);
    CHECK_POLICY(1, 1, GATE_EXPERT_BYTES, DOWN_EXPERT_BYTES, 1, 0);
    CHECK_POLICY(0, 0, GATE_EXPERT_BYTES, DOWN_EXPERT_BYTES, 1, 0);
    CHECK_POLICY(0, 1, GATE_EXPERT_BYTES - 1u,
                 DOWN_EXPERT_BYTES + 2u, 1, 0);
    CHECK_POLICY(0, 1, GATE_EXPERT_BYTES, DOWN_EXPERT_BYTES, 0, 0);
    CHECK_POLICY(0, 1, GATE_EXPERT_BYTES, DOWN_EXPERT_BYTES, 1, 1);
#undef CHECK_POLICY
    return ok;
}

static int seed_route(const ds4_gpu_stream_expert_table *table,
                      const int32_t route[N_SELECTED]) {
    return ds4_gpu_stream_expert_cache_seed_selected(table,
                                                      route,
                                                      N_SELECTED);
}

static int run_churn(const ds4_gpu_stream_expert_table *table) {
    static const int32_t route_a[N_SELECTED] = {0, 1, 2, 0, 1, 2};
    static const int32_t route_b[N_SELECTED] = {3, 4, 5, 3, 4, 5};
    return seed_route(table, route_a) &&
           seed_route(table, route_b) &&
           seed_route(table, route_a);
}

static void print_report(
        const char *name,
        const ds4_gpu_stream_expert_live_index_report *r) {
    fprintf(stderr,
            "%s scans=%llu entries=%llu fallbacks=%llu inserts=%llu "
            "removes=%llu reuse_calls=%llu reuse_entries=%llu "
            "live=%u cache=%u eligible=%u active=%u broken=%u hash=%016llx\n",
            name,
            (unsigned long long)r->scans,
            (unsigned long long)r->entries,
            (unsigned long long)r->fallbacks,
            (unsigned long long)r->inserts,
            (unsigned long long)r->removes,
            (unsigned long long)r->reuse_scan_calls,
            (unsigned long long)r->reuse_scan_entries,
            r->live_count,
            r->cache_entries,
            r->eligible,
            r->active,
            r->broken,
            (unsigned long long)r->resident_hash);
}

int main(void) {
    int ok = check_policy();
    int fd = -1;
    void *model = MAP_FAILED;
    char path[] = "/tmp/ds4-metal-iq2-live-index.XXXXXX";

    if (2u * GATE_EXPERT_BYTES + DOWN_EXPERT_BYTES !=
        TOTAL_EXPERT_BYTES) {
        fprintf(stderr, "production size constants FAIL\n");
        return 1;
    }

    const uint64_t gate_tensor_bytes =
        (uint64_t)N_TOTAL_EXPERT * GATE_EXPERT_BYTES;
    const uint64_t down_tensor_bytes =
        (uint64_t)N_TOTAL_EXPERT * DOWN_EXPERT_BYTES;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_tensor_bytes;
    const uint64_t down_offset = up_offset + gate_tensor_bytes;
    const uint64_t model_size = down_offset + down_tensor_bytes;

    fd = mkstemp(path);
    if (fd < 0 || ftruncate(fd, (off_t)model_size) != 0) {
        perror("IQ2 live-index fixture");
        ok = 0;
        goto cleanup;
    }
    model = mmap(NULL, (size_t)model_size,
                 PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (model == MAP_FAILED) {
        perror("IQ2 live-index mmap");
        ok = 0;
        goto cleanup;
    }

    if (!ds4_gpu_init() || !ds4_gpu_set_model_map(model, model_size)) {
        fprintf(stderr, "Metal initialization/model map FAIL\n");
        ok = 0;
        goto cleanup;
    }
    ds4_gpu_set_ssd_streaming(true);
    ds4_gpu_set_model_fd(fd);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(TOTAL_EXPERT_BYTES);
    setenv("DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY", "1", 1);

    const ds4_gpu_stream_expert_table table = {
        .model_map = model,
        .model_size = model_size,
        .layer = 0,
        .n_total_expert = N_TOTAL_EXPERT,
        .gate_offset = gate_offset,
        .up_offset = up_offset,
        .down_offset = down_offset,
        .gate_expert_bytes = GATE_EXPERT_BYTES,
        .down_expert_bytes = DOWN_EXPERT_BYTES,
    };

    setenv("DS4_METAL_DISABLE_STREAMING_EXPERT_LIVE_INDEX", "1", 1);
    unsetenv("DS4_METAL_ENABLE_STREAMING_EXPERT_LIVE_INDEX");
    ds4_gpu_set_streaming_expert_cache_budget(CACHE_BUDGET);
    ok = ok && run_churn(&table);
    ds4_gpu_stream_expert_live_index_report control = {0};
    ds4_gpu_test_stream_expert_live_index_report(&control);
    print_report("control", &control);
    if (control.scans != 0 || control.entries != 0 ||
        control.cache_entries != CACHE_BUDGET || control.broken != 0 ||
        control.reuse_scan_calls == 0 ||
        control.reuse_scan_entries < 30720u) {
        fprintf(stderr, "control coverage FAIL\n");
        ok = 0;
    }

    unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_LIVE_INDEX");
    unsetenv("DS4_METAL_ENABLE_STREAMING_EXPERT_LIVE_INDEX");
    ds4_gpu_set_streaming_expert_cache_budget(CACHE_BUDGET);
    ok = ok && run_churn(&table);
    ds4_gpu_stream_expert_live_index_report candidate = {0};
    ds4_gpu_test_stream_expert_live_index_report(&candidate);
    print_report("candidate", &candidate);
    if (candidate.scans == 0 || candidate.entries == 0 ||
        candidate.fallbacks != 0 || candidate.inserts < CACHE_BUDGET ||
        candidate.removes == 0 ||
        candidate.live_count != candidate.cache_entries ||
        candidate.cache_entries != CACHE_BUDGET ||
        candidate.eligible == 0 || candidate.active == 0 ||
        candidate.broken != 0 ||
        candidate.reuse_scan_calls == 0 ||
        candidate.reuse_scan_entries >= control.reuse_scan_entries ||
        candidate.entries > candidate.scans * CACHE_BUDGET ||
        candidate.resident_hash != control.resident_hash) {
        fprintf(stderr, "candidate coverage/state FAIL\n");
        ok = 0;
    }

    ds4_gpu_set_streaming_expert_cache_budget(CACHE_BUDGET);
    static const int32_t route_a[N_SELECTED] = {0, 1, 2, 0, 1, 2};
    static const int32_t route_b[N_SELECTED] = {3, 4, 5, 3, 4, 5};
    ok = ok && seed_route(&table, route_a);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_STREAMING_LIVE_INDEX_FAILURE);
    ok = ok && seed_route(&table, route_b);
    ds4_gpu_test_set_flags(0);
    ds4_gpu_stream_expert_live_index_report fault = {0};
    ds4_gpu_test_stream_expert_live_index_report(&fault);
    print_report("fault-fallback", &fault);
    if (fault.fallbacks == 0 || fault.broken == 0 ||
        fault.active != 0 || fault.cache_entries != CACHE_BUDGET) {
        fprintf(stderr, "fault fallback FAIL\n");
        ok = 0;
    }

    ds4_gpu_set_streaming_expert_cache_budget(CACHE_BUDGET);
    ds4_gpu_stream_expert_live_index_report reset = {0};
    ds4_gpu_test_stream_expert_live_index_report(&reset);
    print_report("post-reset", &reset);
    if (reset.broken != 0 || reset.live_count != 0 ||
        reset.cache_entries != 0 || reset.fallbacks != 0) {
        fprintf(stderr, "reset lifecycle FAIL\n");
        ok = 0;
    }

cleanup:
    ds4_gpu_test_set_flags(0);
    unsetenv("DS4_METAL_ENABLE_STREAMING_EXPERT_LIVE_INDEX");
    unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_LIVE_INDEX");
    unsetenv("DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY");
    ds4_gpu_set_streaming_expert_cache_budget(0);
    ds4_gpu_set_model_fd(-1);
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_cleanup();
    if (model != MAP_FAILED) munmap(model, (size_t)model_size);
    if (fd >= 0) close(fd);
    unlink(path);

    fprintf(stderr, "IQ2 Metal streaming expert live-index %s\n",
            ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#else

int main(void) {
    fprintf(stderr, "test_metal_iq2_live_index: SKIP (requires Apple Metal)\n");
    return 0;
}

#endif
