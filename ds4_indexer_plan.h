/* SPDX-License-Identifier: MIT */
/* Pure, checked host planning for the existing indexer score kernels. */
#ifndef DS4_INDEXER_PLAN_H
#define DS4_INDEXER_PLAN_H

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include "ds4_gpu_phase.h"

typedef enum ds4_indexer_backend {
    DS4_INDEXER_BACKEND_METAL = 0,
    DS4_INDEXER_BACKEND_CUDA,
    DS4_INDEXER_BACKEND_HIP,
} ds4_indexer_backend;

typedef enum ds4_indexer_kernel {
    DS4_INDEXER_KERNEL_BACKEND_NATIVE = 0,
    DS4_INDEXER_KERNEL_METAL_TILED_HALF,
    DS4_INDEXER_KERNEL_METAL_TILED_F32,
    DS4_INDEXER_KERNEL_METAL_NAX,
    DS4_INDEXER_KERNEL_METAL_HEAD2,
    DS4_INDEXER_KERNEL_METAL_HEAD4,
    DS4_INDEXER_KERNEL_METAL_HEAD1,
} ds4_indexer_kernel;

typedef enum ds4_indexer_precision {
    DS4_INDEXER_PRECISION_BACKEND_NATIVE = 0,
    DS4_INDEXER_PRECISION_F16_F32_ACC,
    DS4_INDEXER_PRECISION_F32,
} ds4_indexer_precision;

typedef struct ds4_indexer_plan_request {
    ds4_indexer_backend backend;
    ds4_gpu_execution_phase phase;
    uint32_t n_comp, n_tokens, pos0, n_head, head_dim, ratio;
    bool quality;
} ds4_indexer_plan_request;

typedef struct ds4_indexer_plan_caps {
    /* Metal device/PSO limits, not performance thresholds. Native CUDA/HIP
     * dispatch keeps responsibility for its own launch geometry. */
    uint32_t max_threads, max_shared_bytes, max_grid_x, max_grid_y;
    uint64_t max_buffer_bytes;
    bool nax_available;
    /* This must come from a measured capability or a test. Merely having
     * enough shared memory never enables a candidate. Group 1 changes only
     * staging to vector loads; groups 2/4 also reuse K across heads. */
    bool allow_grouped;
    uint32_t preferred_head_group; /* 0 selects 2; otherwise exactly 1, 2 or 4. */
} ds4_indexer_plan_caps;

typedef struct ds4_indexer_plan {
    ds4_indexer_backend backend;
    ds4_gpu_execution_phase phase;
    ds4_indexer_kernel kernel;
    ds4_indexer_precision precision;
    uint32_t tile_m, tile_n, head_group, threads, shared_bytes, grid_x, grid_y;
    /* Source tensors and output remain F32 even when operands are staged as
     * F16. Counts describe the required contiguous prefixes, not allocation. */
    uint64_t q_bytes, weight_bytes, index_bytes, score_bytes;
} ds4_indexer_plan;

static inline bool ds4_indexer_plan_product(
        uint64_t a, uint64_t b, uint64_t *out) {
    if (b && a > UINT64_MAX / b) return false;
    *out = a * b;
    return true;
}

/* A successful plan records execution intent; cache users must also include
 * the request's shape/position and device capabilities in their own key.
 * Failure never modifies *out. This does not validate actual tensor handles,
 * offsets, aliasing, or loaded pipelines: those remain the wrapper's job. */
static inline bool ds4_indexer_plan_build(
        const ds4_indexer_plan_request *request,
        const ds4_indexer_plan_caps *caps,
        ds4_indexer_plan *out) {
    if (!request || !caps || !out || !caps->max_buffer_bytes ||
        (caps->allow_grouped && caps->preferred_head_group != 0 &&
         caps->preferred_head_group != 1 &&
         caps->preferred_head_group != 2 && caps->preferred_head_group != 4) ||
        (unsigned)request->backend > DS4_INDEXER_BACKEND_HIP ||
        (unsigned)request->phase > DS4_GPU_PHASE_MIXED ||
        !request->n_comp || !request->n_tokens || !request->n_head ||
        !request->head_dim || !request->ratio ||
        request->n_comp > INT32_MAX || request->n_tokens > INT32_MAX ||
        request->n_head > INT32_MAX || request->head_dim > INT32_MAX ||
        (uint64_t)request->pos0 + request->n_tokens > INT32_MAX) return false;

#ifdef __cplusplus
    ds4_indexer_plan plan = {};
#else
    ds4_indexer_plan plan = {0};
#endif
    plan.backend = request->backend;
    plan.phase = request->phase;
    uint64_t q_elements;
    if (!ds4_indexer_plan_product(request->n_tokens, request->n_head, &q_elements) ||
        !ds4_indexer_plan_product(q_elements, request->head_dim, &q_elements) ||
        !ds4_indexer_plan_product(q_elements, sizeof(float), &plan.q_bytes) ||
        !ds4_indexer_plan_product((uint64_t)request->n_tokens * request->n_head,
                                 sizeof(float), &plan.weight_bytes) ||
        !ds4_indexer_plan_product((uint64_t)request->n_comp * request->head_dim,
                                 sizeof(float), &plan.index_bytes) ||
        !ds4_indexer_plan_product((uint64_t)request->n_comp * request->n_tokens,
                                 sizeof(float), &plan.score_bytes) ||
        plan.q_bytes > caps->max_buffer_bytes ||
        plan.weight_bytes > caps->max_buffer_bytes ||
        plan.index_bytes > caps->max_buffer_bytes ||
        plan.score_bytes > caps->max_buffer_bytes) return false;

    if (request->backend != DS4_INDEXER_BACKEND_METAL) {
        /* Geometry and operand precision depend on native backend policy.
         * Zero geometry explicitly means that no launch has been planned. */
        plan.kernel = DS4_INDEXER_KERNEL_BACKEND_NATIVE;
        plan.precision = DS4_INDEXER_PRECISION_BACKEND_NATIVE;
        *out = plan;
        return true;
    }
    if (request->head_dim != 128 || caps->max_threads < 128) return false;

    plan.tile_m = 8;
    plan.tile_n = 32;
    plan.head_group = 1;
    plan.threads = 128;
    if (caps->nax_available && request->n_tokens >= 16) {
        /* Preserve the current NAX priority, including quality mode and
         * explicitly tagged phases. Grouping never supersedes this path. */
        plan.kernel = DS4_INDEXER_KERNEL_METAL_NAX;
        plan.precision = DS4_INDEXER_PRECISION_F16_F32_ACC;
        plan.tile_m = 16;
        plan.head_group = 2;
        plan.shared_bytes = 16384;
    } else if (request->quality) {
        plan.kernel = DS4_INDEXER_KERNEL_METAL_TILED_F32;
        plan.precision = DS4_INDEXER_PRECISION_F32;
        plan.shared_bytes = 21504;
    } else {
        plan.kernel = DS4_INDEXER_KERNEL_METAL_TILED_HALF;
        plan.precision = DS4_INDEXER_PRECISION_F16_F32_ACC;
        plan.shared_bytes = 11264;
        if (caps->allow_grouped && request->n_head == 64 &&
            ds4_gpu_execution_phase_allows_prefill(request->phase)) {
            const uint32_t group = caps->preferred_head_group ?
                caps->preferred_head_group : 2;
            const uint32_t shared = 8192 + 3072 * group;
            if (shared <= caps->max_shared_bytes) {
                plan.kernel = group == 1 ? DS4_INDEXER_KERNEL_METAL_HEAD1 :
                              group == 2 ? DS4_INDEXER_KERNEL_METAL_HEAD2 :
                                           DS4_INDEXER_KERNEL_METAL_HEAD4;
                plan.head_group = group;
                plan.shared_bytes = shared;
            }
        }
    }
    /* Padded last tiles are computed in 64 bits even though current bounds
     * also keep every kernel's uint32 tile origin/addition representable. */
    plan.grid_x = (uint32_t)(((uint64_t)request->n_comp + plan.tile_n - 1) / plan.tile_n);
    plan.grid_y = (uint32_t)(((uint64_t)request->n_tokens + plan.tile_m - 1) / plan.tile_m);
    if (plan.shared_bytes > caps->max_shared_bytes ||
        plan.grid_x > caps->max_grid_x || plan.grid_y > caps->max_grid_y) return false;
    *out = plan;
    return true;
}

#endif
