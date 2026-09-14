// SPDX-License-Identifier: MIT
// ABI and bounded geometry for exact Metal score-range / top-k streaming.
#ifndef DS4_INDEXER_STREAM_H
#define DS4_INDEXER_STREAM_H
#include "ds4_indexer_topk.h"

typedef struct ds4_indexer_stream_pair {
    float score;
    int32_t index;
} ds4_indexer_stream_pair;

typedef struct ds4_indexer_stream_leaf_args {
    uint32_t columns;
    uint32_t n_rows;
    uint32_t output_width;
    uint32_t leaf_length;
    uint32_t index_base;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
} ds4_indexer_stream_leaf_args;

typedef struct ds4_indexer_stream_geometry {
    uint32_t chunk_columns;
    uint32_t chunks;
    uint32_t candidate_width;
    uint32_t local_width;
    uint32_t local_second_width;
    uint32_t leaf_length;
} ds4_indexer_stream_geometry;

static inline bool ds4_indexer_stream_initial(
        uint32_t columns, uint32_t rows, uint32_t k, uint32_t threads,
        uint32_t chunk_columns, ds4_indexer_stream_geometry *out) {
    ds4_indexer_topk_geometry global, local;
    if (!out || !ds4_indexer_topk_initial(columns, rows, k, threads, &global) ||
        chunk_columns < 32u || chunk_columns > INT32_MAX / 2u ||
        (chunk_columns & (chunk_columns - 1u)) != 0u ||
        chunk_columns < threads || chunk_columns < k) return false;
    // Power-of-two chunks cover complete 32-column scoring tiles and a whole
    // power-of-two number of the baseline's leaves. Only the last may be short.
    const uint32_t local_columns = columns < chunk_columns ? columns : chunk_columns;
    if (!ds4_indexer_topk_initial(local_columns, rows, k, threads, &local)) return false;
    const uint32_t chunks = (uint32_t)(((uint64_t)columns + chunk_columns - 1u) / chunk_columns);
    const uint32_t tail = (uint32_t)((uint64_t)columns - (chunks - 1ull) * chunk_columns);
    const uint64_t candidate_width = (chunks - 1ull) * k + (tail < k ? tail : k);
    if (candidate_width > global.width || candidate_width * rows > INT32_MAX) return false;
    uint32_t second_width = 0;
    if (local.leaf_length < local.width) {
        ds4_indexer_topk_stage first;
        if (!ds4_indexer_topk_next(local.width, local.leaf_length, k, &first)) return false;
        second_width = first.width;
    }
    out->chunk_columns = chunk_columns;
    out->chunks = chunks;
    out->candidate_width = (uint32_t)candidate_width;
    out->local_width = local.width;
    out->local_second_width = second_width;
    out->leaf_length = global.leaf_length;
    return true;
}
#endif
