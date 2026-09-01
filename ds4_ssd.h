#ifndef DS4_SSD_H
#define DS4_SSD_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    void *ptr;
    uint64_t bytes;
} ds4_ssd_memory_lock;

typedef struct {
    uint64_t model_target_bytes;
    uint64_t cache_bytes;
    uint64_t effective_cache_bytes;
    uint32_t cache_experts;
} ds4_ssd_cache_plan;

/* A bounded-memory admission ledger for SSD streaming. It describes one
 * process's planned accelerator allocations; it cannot observe or guarantee
 * the safety of unrelated processes. Every field is a separately reportable
 * reservation so callers cannot accidentally omit persistent DSpark support
 * from their cache arithmetic. */
typedef struct {
    uint64_t capacity_bytes;
    uint64_t prelocked_bytes;
    uint64_t target_mapped_bytes;
    uint64_t support_bytes;
    uint64_t expert_cache_bytes;
    uint64_t prefill_reserve_bytes;
    /* Every session owns each of these allocations independently.
     * session_count defaults to one when zero. */
    uint64_t session_kv_bytes;
    uint64_t session_context_scratch_bytes;
    /* Persistent, non-KV GPU graph state: live compressor frontiers plus
     * fixed decode/output scratch. This deliberately excludes speculative
     * tensors and prefill workspace, which have their own categories. */
    uint64_t session_graph_bytes;
    uint64_t session_speculative_bytes;
    uint64_t session_host_bytes;
    uint64_t session_prefill_workspace_bytes;
    /* The serialized batched server may instead transfer one prefill
     * workspace to engine ownership and alias it from every session. */
    uint64_t shared_prefill_workspace_bytes;
    uint64_t safety_headroom_bytes;
    uint32_t session_count;
} ds4_ssd_admission_request;

typedef struct {
    uint64_t required_bytes;
    uint64_t budget_bytes;
    uint64_t total_session_kv_bytes;
    uint64_t total_session_context_scratch_bytes;
    uint64_t total_session_graph_bytes;
    uint64_t total_session_speculative_bytes;
    uint64_t total_session_host_bytes;
    uint64_t total_session_prefill_workspace_bytes;
    uint64_t shared_prefill_workspace_bytes;
} ds4_ssd_admission_result;

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes);
bool ds4_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes);

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes);
bool ds4_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             ds4_ssd_cache_plan *out);

/* Returns false for overflow, zero capacity, or insufficient capacity after
 * the requested safety headroom. The result is intentionally pure so it can
 * be exercised without loading a model or initializing an accelerator. */
bool ds4_ssd_admission_plan(const ds4_ssd_admission_request *request,
                            ds4_ssd_admission_result        *out);

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock,
                                 uint64_t             bytes);
void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock);

#endif
