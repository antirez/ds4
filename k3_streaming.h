#ifndef K3_STREAMING_H
#define K3_STREAMING_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Pure policy: the caller supplies the environment value. No process-global
 * environment changes, filesystem access, model loading or GPU calls. */
static inline bool k3_ssd_defaults_enabled(bool requested, bool gguf,
                                           const char *value) {
    return requested && gguf &&
        (!value || (value[0] && strcmp(value, "0") != 0));
}

static inline uint32_t k3_ssd_prefill_chunk(uint32_t requested, bool enabled) {
    return requested ? requested : (enabled ? 64u : 512u);
}

/* Preserve the existing API context default independently of the faster
 * prefill chunk. In particular, opt context=0/chunk=0 still starts at512. */
static inline uint32_t k3_ssd_initial_context(int requested,
                                             uint32_t prefill_requested) {
    return requested > 0 ? (uint32_t)requested
                         : k3_ssd_prefill_chunk(prefill_requested, false);
}

/* An explicit expert count has precedence over the byte-budget default. */
static inline uint64_t k3_ssd_expert_cache_bytes(uint64_t bytes,
                                               uint32_t slots,
                                               bool enabled) {
    return enabled && bytes == 0 && slots == 0
        ? UINT64_C(10) * 1024u * 1024u * 1024u : bytes;
}

/* Preserve the existing Metal slab parser for every explicit value, including
 * its fallback on empty/zero/invalid strings and saturation on overflow. Only
 * an unset value receives the optimized 256MiB default. */
static inline uint64_t k3_ssd_slab_target_bytes(bool enabled,
                                              const char *value) {
    const uint64_t mib = UINT64_C(1024) * 1024u;
    uint64_t target = (enabled && !value ? 256u : 4096u) * mib;
    if (value && value[0]) {
        char *end = NULL;
        const unsigned long long parsed = strtoull(value, &end, 10);
        if (end != value && *end == '\0' && parsed != 0) {
            target = parsed > UINT64_MAX / mib
                ? UINT64_MAX : (uint64_t)parsed * mib;
        }
    }
    return target;
}

/* The existing disable switch has presence semantics: even "0" or an empty
 * value disables advice. The optimized profile also disables it by default. */
static inline bool k3_ssd_expert_readahead_enabled(bool streaming,
                                                 bool enabled,
                                                 const char *disable_value) {
    return streaming && !enabled && disable_value == NULL;
}

#endif
