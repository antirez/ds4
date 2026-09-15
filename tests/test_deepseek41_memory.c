/* Exercise actual ROCm admission with controlled host-memory observations. */
#include "../ds4_linux_memory.h"
static uint64_t available;
static bool test_available(uint64_t *bytes) { *bytes = available; return true; }
#define ds4_linux_nonmovable_memory test_available
#include "../ds4.c"
#include <assert.h>

static uint64_t recommended;
uint64_t ds4_gpu_recommended_working_set_size(void) { return recommended; }
int ds4_gpu_stream_expert_cache_get_memory(ds4_gpu_stream_expert_memory *out) {
    memset(out, 0, sizeof(*out));
    return 1;
}

int main(void) {
    const uint64_t gib = UINT64_C(1) << 30;
    ds4_engine e = {0};
    e.ds41_host_memory_baseline = 180 * gib;
    e.model.size = 152 * gib;
    e.vision_model.size = gib;
    e.vision_ready = e.vision_map_ready = e.ds41_model_loaded = true;
    e.startup_model_span_bytes = e.model.size;
    recommended = 188 * gib;
    /* The observed 192 GB resident failure: loaded weights, 1.5 GiB future
     * graph and 25.9 GiB available. It must tolerate small host fluctuations. */
    for (unsigned mib = 24 * 1024; mib <= 26 * 1024; mib += 64) {
        available = (uint64_t)mib << 20;
        assert(ds41_memory_admit(&e, gib + gib / 2, false));
    }
    /* Existing graph bytes are charged once on restoration/session growth. */
    e.ds41_session_bytes = 2 * gib;
    available = 14 * gib;
    assert(ds41_memory_admit(&e, 2 * gib, false));
    assert(!ds41_memory_admit(&e, 4 * gib, false));
    e.ds41_session_bytes = 0;
    available = 13 * gib;
    assert(!ds41_memory_admit(&e, gib, false));
    /* Sidecar upload is still a real outstanding allocation. */
    available = 15 * gib;
    assert(ds41_memory_admit(&e, gib, false));
    e.vision_map_ready = false;
    assert(!ds41_memory_admit(&e, gib, false));
    e.vision_map_ready = true;
    /* The OS minimum and accelerator cap cannot be bypassed. */
    e.ds41_host_memory_baseline = 8 * gib;
    assert(!ds41_memory_admit(&e, gib, false));
    e.ds41_host_memory_baseline = 180 * gib;
    recommended = 150 * gib;
    available = 100 * gib;
    assert(!ds41_memory_admit(&e, gib, false));
    recommended = 188 * gib;
    assert(!ds41_memory_admit(&e, UINT64_MAX, false));
    /* A 128 GB rank loads only its owned expert half and replicated dense
     * tensors. Both total admission and remaining-allocation checks must
     * charge that same footprint, while keeping the existing reserves. */
    e.ds41_host_memory_baseline = available = 120 * gib;
    recommended = 124 * gib;
    e.startup_model_span_bytes = 0;
    g_tp_shard_model_bytes = 81 * gib;
    assert(ds41_memory_admit(&e, 3 * gib, false));
    available = 93 * gib;
    assert(!ds41_memory_admit(&e, 3 * gib, false));
    available = 120 * gib;
    g_tp_shard_model_bytes = 0;
    assert(!ds41_memory_admit(&e, 3 * gib, false));
    g_tp_shard_model_bytes = e.startup_model_span_bytes = 81 * gib;
    available = 15 * gib;
    assert(ds41_memory_admit(&e, 3 * gib, false));
    available = 13 * gib;
    assert(!ds41_memory_admit(&e, 3 * gib, false));
    g_tp_shard_model_bytes = 0;
    e.ssd_streaming = true;
    assert(ds41_rocm_host_reserve_bytes(128 * gib) == 8 * gib);
    assert(ds41_rocm_stream_reserve_bytes(128 * gib) == 10 * gib);
    assert(ds41_rocm_stream_reserve_bytes(180 * gib) == 13 * gib + gib / 4);
    assert(ds41_rocm_stream_reserve_bytes(0) == 10 * gib);
    puts("V4.1 ROCm resident admission, reuse, pressure, sidecar and cap checks PASS");
    return 0;
}
