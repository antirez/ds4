// SPDX-License-Identifier: MIT
// Geometry of the exact, compact Metal top-k merge tree.
#ifndef DS4_INDEXER_TOPK_H
#define DS4_INDEXER_TOPK_H
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

typedef struct ds4_indexer_topk_geometry {
    uint32_t leaves;
    uint32_t leaf_length;
    uint32_t width;
} ds4_indexer_topk_geometry;

typedef struct ds4_indexer_topk_stage {
    uint32_t groups;
    uint32_t run_length;
    uint32_t width;
} ds4_indexer_topk_stage;

// Mirrors the 32-byte Metal constant buffer, with explicit tail padding.
typedef struct ds4_indexer_topk_merge_args {
    uint64_t score_stride;
    uint32_t n_rows;
    uint32_t input_width;
    uint32_t run_length;
    uint32_t output_width;
    uint32_t output_run_length;
    uint32_t pad;
} ds4_indexer_topk_merge_args;

static inline bool ds4_indexer_topk_initial(
        uint32_t columns, uint32_t rows, uint32_t k, uint32_t threads,
        ds4_indexer_topk_geometry *out) {
    if (!out || columns == 0u || rows == 0u || k == 0u || k > columns ||
        columns > INT32_MAX || rows > INT32_MAX ||
        threads == 0u || threads > 1024u || (threads & (threads - 1u))) return false;
    const uint32_t leaves = (uint32_t)(((uint64_t)columns + threads - 1u) / threads);
    const uint32_t leaf_length = k < threads ? k : threads;
    const uint32_t tail = columns - (leaves - 1u) * threads;
    const uint64_t width = (uint64_t)(leaves - 1u) * leaf_length +
                          (tail < leaf_length ? tail : leaf_length);
    // The unchanged leaf/legacy kernels use signed index products and 2*len.
    // Decline unrepresentable geometry before either path submits GPU work.
    if (width > INT32_MAX / 2u || width * rows > INT32_MAX) return false;
    out->leaves = leaves;
    out->leaf_length = leaf_length;
    out->width = (uint32_t)width;
    return true;
}

static inline bool ds4_indexer_topk_next(
        uint32_t width, uint32_t run_length, uint32_t k,
        ds4_indexer_topk_stage *out) {
    if (!out || width == 0u || run_length == 0u || k == 0u ||
        width > INT32_MAX || run_length > width || run_length > k) return false;
    const uint64_t pair_length = 2ull * run_length;
    const uint32_t groups = (uint32_t)(((uint64_t)width + pair_length - 1u) / pair_length);
    const uint32_t next_length = pair_length < k ? (uint32_t)pair_length : k;
    const uint32_t tail = (uint32_t)((uint64_t)width - (groups - 1ull) * pair_length);
    const uint64_t next_width = (uint64_t)(groups - 1u) * next_length +
                               (tail < next_length ? tail : next_length);
    if (next_width > width || next_width > INT32_MAX) return false;
    out->groups = groups;
    out->run_length = next_length;
    out->width = (uint32_t)next_width;
    return true;
}

static inline bool ds4_indexer_topk_compaction_useful(
        ds4_indexer_topk_geometry geometry, uint32_t k) {
    uint32_t width = geometry.width, run_length = geometry.leaf_length;
    while (run_length < width) {
        ds4_indexer_topk_stage next;
        if (!ds4_indexer_topk_next(width, run_length, k, &next)) return false;
        if (next.groups > 1u && next.width < width) return true;
        if (next.groups <= 1u) return false;
        width = next.width;
        run_length = next.run_length;
    }
    return false;
}
#endif
