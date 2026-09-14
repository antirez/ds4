/* SPDX-License-Identifier: MIT */
#ifndef DS4_INDEXER_PREPARED_H
#define DS4_INDEXER_PREPARED_H

#include "ds4_indexer_plan.h"

/* A caller owns the scratch for the entire queued operation. This plan does
 * not cache converted rows across calls, layers, sessions or graph replays. */
typedef struct ds4_indexer_prepared_caps {
    bool device_supported, capturing, native_mxf4;
    bool register_scores, register_scores_supported;
    uint32_t max_threads, max_shared_bytes, max_grid_x, max_grid_y;
    uint64_t max_buffer_bytes;
} ds4_indexer_prepared_caps;

typedef struct ds4_indexer_prepared_plan {
    ds4_indexer_plan base;
    uint64_t q_elements, k_elements, q_half_bytes, k_half_bytes;
    uint64_t k_offset, scratch_bytes;
    uint32_t prepare_blocks, score_grid_x, score_grid_y, shared_bytes;
    bool register_scores;
} ds4_indexer_prepared_plan;

static inline bool ds4_indexer_prepared_build(
        const ds4_indexer_plan_request *request,
        const ds4_indexer_prepared_caps *caps,
        ds4_indexer_prepared_plan *out) {
    if (!request || !caps || !out || !caps->device_supported || caps->capturing ||
        caps->native_mxf4 || request->quality || request->n_tokens < 2 ||
        request->n_head != 64 || request->head_dim != 128 ||
        !ds4_gpu_execution_phase_allows_prefill(request->phase) ||
        (request->backend != DS4_INDEXER_BACKEND_CUDA &&
         request->backend != DS4_INDEXER_BACKEND_HIP) ||
        (caps->register_scores &&
         (request->backend != DS4_INDEXER_BACKEND_HIP || !caps->register_scores_supported)) ||
        caps->max_threads < 256 || !caps->max_grid_x || !caps->max_grid_y) return false;
#ifdef __cplusplus
    ds4_indexer_prepared_plan plan = {};
    ds4_indexer_plan_caps common = {};
#else
    ds4_indexer_prepared_plan plan = {0};
    ds4_indexer_plan_caps common = {0};
#endif
    common.max_buffer_bytes = caps->max_buffer_bytes;
    if (!ds4_indexer_plan_build(request, &common, &plan.base)) return false;
    plan.q_elements = plan.base.q_bytes / sizeof(float);
    plan.k_elements = plan.base.index_bytes / sizeof(float);
    plan.q_half_bytes = plan.base.q_bytes / 2;
    plan.k_half_bytes = plan.base.index_bytes / 2;
    if (plan.q_half_bytes > UINT64_MAX - 255) return false;
    plan.k_offset = (plan.q_half_bytes + 255) & ~(uint64_t)255;
    if (plan.k_half_bytes > UINT64_MAX - plan.k_offset) return false;
    plan.scratch_bytes = plan.k_offset + plan.k_half_bytes;
    if (plan.scratch_bytes > caps->max_buffer_bytes ||
        plan.q_elements > UINT64_MAX - plan.k_elements) return false;
    const uint64_t pairs = (plan.q_elements + plan.k_elements) / 2;
    uint64_t blocks = (pairs + 255) / 256;
    /* The producer has a grid-stride loop; cap its grid without dropping rows. */
    if (blocks > 65535) blocks = 65535;
    if (blocks > caps->max_grid_x) blocks = caps->max_grid_x;
    plan.prepare_blocks = (uint32_t)blocks;
    plan.score_grid_x = (request->n_comp + 127u) / 128u;
    const uint32_t tile_t = request->backend == DS4_INDEXER_BACKEND_CUDA ? 32u : 16u;
    plan.score_grid_y = (request->n_tokens + tile_t - 1u) / tile_t;
    plan.register_scores = caps->register_scores;
    plan.shared_bytes = request->backend == DS4_INDEXER_BACKEND_CUDA ? 43520u :
        caps->register_scores ? 37888u : 45056u;
    if (plan.score_grid_x > caps->max_grid_x || plan.score_grid_y > caps->max_grid_y ||
        plan.shared_bytes > caps->max_shared_bytes) return false;
    *out = plan;
    return true;
}

typedef struct ds4_indexer_prepared_buffers {
    uintptr_t q, keys, weights, scores, scratch;
    uint64_t q_bytes, key_bytes, weight_bytes, score_bytes, scratch_bytes;
} ds4_indexer_prepared_buffers;

static inline bool ds4_indexer_prepared_range(uintptr_t p, uint64_t bytes) {
    return p && bytes && bytes - 1 <= UINTPTR_MAX - p;
}
static inline bool ds4_indexer_prepared_overlap(
        uintptr_t a, uint64_t an, uintptr_t b, uint64_t bn) {
    return a <= b ? b - a < an : a - b < bn;
}

/* Prefix checks precede the producer. The read-only inputs may alias each
 * other, but neither writer may alias any reader or the other writer. */
static inline bool ds4_indexer_prepared_buffers_valid(
        const ds4_indexer_prepared_plan *p,
        const ds4_indexer_prepared_buffers *b) {
    if (!p || !b || b->q_bytes < p->base.q_bytes || b->key_bytes < p->base.index_bytes ||
        b->weight_bytes < p->base.weight_bytes || b->score_bytes < p->base.score_bytes ||
        b->scratch_bytes < p->scratch_bytes ||
        ((b->q | b->keys | b->weights | b->scores) & 3u) || (b->scratch & 255u) ||
        !ds4_indexer_prepared_range(b->q, p->base.q_bytes) ||
        !ds4_indexer_prepared_range(b->keys, p->base.index_bytes) ||
        !ds4_indexer_prepared_range(b->weights, p->base.weight_bytes) ||
        !ds4_indexer_prepared_range(b->scores, p->base.score_bytes) ||
        !ds4_indexer_prepared_range(b->scratch, p->scratch_bytes)) return false;
    return !ds4_indexer_prepared_overlap(b->scores,p->base.score_bytes,b->q,p->base.q_bytes) &&
        !ds4_indexer_prepared_overlap(b->scores,p->base.score_bytes,b->keys,p->base.index_bytes) &&
        !ds4_indexer_prepared_overlap(b->scores,p->base.score_bytes,b->weights,p->base.weight_bytes) &&
        !ds4_indexer_prepared_overlap(b->scores,p->base.score_bytes,b->scratch,p->scratch_bytes) &&
        !ds4_indexer_prepared_overlap(b->scratch,p->scratch_bytes,b->q,p->base.q_bytes) &&
        !ds4_indexer_prepared_overlap(b->scratch,p->scratch_bytes,b->keys,p->base.index_bytes) &&
        !ds4_indexer_prepared_overlap(b->scratch,p->scratch_bytes,b->weights,p->base.weight_bytes);
}
#endif
