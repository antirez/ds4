#ifndef DS4_FLOAT_COMPARE_H
#define DS4_FLOAT_COMPARE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Pure CPU float32 comparison. Bit identity is authoritative and is based on
 * the IEEE-754 object representation copied with memcpy, never float ==.
 *
 * A zero-length comparison is valid even when either input is NULL. For a
 * nonzero length, NULL input makes the call invalid: the function returns
 * false, valid_input and bit_exact are false, and first_mismatch_index is
 * SIZE_MAX.
 *
 * Diagnostics are accumulated only for bit-mismatching elements:
 *   - absolute difference is undefined for a pair containing NaN;
 *   - relative difference is undefined for a pair containing NaN or infinity;
 *   - ULP distance is defined only when both values are finite.
 * If any mismatching pair makes a diagnostic undefined, its *_defined flag is
 * false and its value is NAN (absolute/relative) or UINT32_MAX (ULP). Empty
 * and bit-identical comparisons have defined zero diagnostics.
 */
typedef struct {
    bool valid_input;
    bool bit_exact;
    size_t length;
    size_t mismatch_count;
    size_t first_mismatch_index;
    float first_actual;
    float first_expected;
    uint32_t first_actual_bits;
    uint32_t first_expected_bits;
    bool max_abs_diff_defined;
    bool max_rel_diff_defined;
    bool max_ulp_distance_defined;
    double max_abs_diff;
    double max_rel_diff;
    uint32_t max_ulp_distance;
} ds4_float_compare_result;

bool ds4_float_compare_exact(const float *actual, const float *expected,
                             size_t length,
                             ds4_float_compare_result *result);

#endif
