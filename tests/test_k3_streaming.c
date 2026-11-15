/* CPU-only default-selection tests: no environment writes, filesystem or GPU. */
#include "../k3_streaming.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
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
    assert(k3_ssd_expert_cache_bytes(0, 0, true) == 10u * gib);
    assert(k3_ssd_expert_cache_bytes(0, 0, false) == 0);
    assert(k3_ssd_expert_cache_bytes(2u * gib, 0, true) == 2u * gib);
    assert(k3_ssd_expert_cache_bytes(0, 231, true) == 0);
    assert(k3_ssd_expert_cache_bytes(2u * gib, 231, true) == 2u * gib);
    assert(k3_ssd_expert_cache_bytes(UINT64_MAX, UINT32_MAX, true) == UINT64_MAX);

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
    assert(k3_ssd_expert_cache_bytes(0, 0, false) == 0);
    assert(k3_ssd_slab_target_bytes(false, NULL) == 4096u * mib);
    assert(k3_ssd_expert_readahead_enabled(true, false, NULL));
    puts("K3 SSD streaming policy: defaults, explicit overrides and presence semantics passed (CPU only)");
    return 0;
}
