#ifndef DS4_EXACTNESS_PROBE_H
#define DS4_EXACTNESS_PROBE_H

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/*
 * Lightweight, model-independent helpers for DSpark numerical-exactness work.
 *
 * These helpers intentionally take an explicit float count instead of trying
 * to infer tensor shape/size.  That keeps them independent of internal graph
 * metadata and makes them usable at arbitrary verifier checkpoints.
 */

typedef struct ds4_probe_diff {
    bool bit_exact;
    uint64_t first_mismatch;
    uint64_t mismatch_count;

    float first_actual;
    float first_expected;

    float max_abs_diff;
    float max_rel_diff;
    uint32_t max_ulp_diff;
} ds4_probe_diff;

/* Enabled when DS4_DSPARK_EXACTNESS_PROBE is set to a non-empty, non-"0" value. */
bool ds4_exactness_probe_enabled(void);

/*
 * Compare two CPU float arrays.
 *
 * Returns true when the comparison itself completed successfully.
 * result->bit_exact says whether all float bit patterns were identical.
 */
bool ds4_probe_compare_f32_arrays(
        const float *actual,
        const float *expected,
        uint64_t count,
        ds4_probe_diff *result);

/*
 * Read two GPU tensors in bounded chunks and compare them as float32.
 *
 * actual_offset_bytes / expected_offset_bytes must be 4-byte aligned.
 * count is the number of float32 elements to compare.
 *
 * The function avoids allocating a buffer proportional to the whole tensor.
 */
bool ds4_probe_compare_f32_tensors(
        const ds4_gpu_tensor *actual,
        uint64_t actual_offset_bytes,
        const ds4_gpu_tensor *expected,
        uint64_t expected_offset_bytes,
        uint64_t count,
        ds4_probe_diff *result);

/* Print one compact machine-readable-ish diagnostic line. */
void ds4_probe_print_diff(
        FILE *fp,
        const char *label,
        uint32_t layer,
        uint32_t row,
        const ds4_probe_diff *diff);

#endif
