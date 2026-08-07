#include "ds4_float_compare.h"

#include <float.h>
#include <math.h>
#include <string.h>

#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128
#error "ds4_float_compare requires IEEE-754 binary32 float"
#endif

typedef char ds4_float_must_be_32_bits[
    sizeof(float) == sizeof(uint32_t) ? 1 : -1];

static uint32_t float_bits(const float *value) {
    uint32_t bits;
    memcpy(&bits, value, sizeof(bits));
    return bits;
}

static bool bits_are_nan(uint32_t bits) {
    return (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) &&
           (bits & UINT32_C(0x007fffff)) != 0;
}

static bool bits_are_infinity(uint32_t bits) {
    return (bits & UINT32_C(0x7fffffff)) == UINT32_C(0x7f800000);
}

static uint32_t ordered_float_key(uint32_t bits) {
    return (bits & UINT32_C(0x80000000)) ? ~bits
                                         : bits | UINT32_C(0x80000000);
}

static uint32_t finite_ulp_distance(uint32_t a_bits, uint32_t b_bits) {
    const uint32_t a_key = ordered_float_key(a_bits);
    const uint32_t b_key = ordered_float_key(b_bits);
    return a_key >= b_key ? a_key - b_key : b_key - a_key;
}

static void initialize_result(ds4_float_compare_result *result,
                              size_t length) {
    memset(result, 0, sizeof(*result));
    result->length = length;
    result->first_mismatch_index = SIZE_MAX;
    result->max_abs_diff_defined = true;
    result->max_rel_diff_defined = true;
    result->max_ulp_distance_defined = true;
}

bool ds4_float_compare_exact(const float *actual, const float *expected,
                             size_t length,
                             ds4_float_compare_result *result) {
    size_t i;

    if (result == NULL) {
        return false;
    }

    initialize_result(result, length);
    if (length != 0 && (actual == NULL || expected == NULL)) {
        return false;
    }

    result->valid_input = true;
    result->bit_exact = true;

    for (i = 0; i < length; ++i) {
        const uint32_t actual_bits = float_bits(&actual[i]);
        const uint32_t expected_bits = float_bits(&expected[i]);
        double abs_diff;
        double denominator;
        double rel_diff;
        uint32_t ulp_distance;

        if (actual_bits == expected_bits) {
            continue;
        }

        result->bit_exact = false;
        ++result->mismatch_count;
        if (result->first_mismatch_index == SIZE_MAX) {
            result->first_mismatch_index = i;
            result->first_actual = actual[i];
            result->first_expected = expected[i];
            result->first_actual_bits = actual_bits;
            result->first_expected_bits = expected_bits;
        }

        if (bits_are_nan(actual_bits) || bits_are_nan(expected_bits)) {
            result->max_abs_diff_defined = false;
            result->max_rel_diff_defined = false;
            result->max_ulp_distance_defined = false;
            continue;
        }

        if (bits_are_infinity(actual_bits) ||
            bits_are_infinity(expected_bits)) {
            result->max_abs_diff = INFINITY;
            result->max_rel_diff_defined = false;
            result->max_ulp_distance_defined = false;
            continue;
        }

        abs_diff = fabs((double)actual[i] - (double)expected[i]);
        if (result->max_abs_diff_defined && abs_diff > result->max_abs_diff) {
            result->max_abs_diff = abs_diff;
        }

        denominator = fmax(fabs((double)actual[i]),
                           fabs((double)expected[i]));
        rel_diff = denominator == 0.0 ? 0.0 : abs_diff / denominator;
        if (result->max_rel_diff_defined && rel_diff > result->max_rel_diff) {
            result->max_rel_diff = rel_diff;
        }

        ulp_distance = finite_ulp_distance(actual_bits, expected_bits);
        if (result->max_ulp_distance_defined &&
            ulp_distance > result->max_ulp_distance) {
            result->max_ulp_distance = ulp_distance;
        }
    }

    if (!result->max_abs_diff_defined) {
        result->max_abs_diff = NAN;
    }
    if (!result->max_rel_diff_defined) {
        result->max_rel_diff = NAN;
    }
    if (!result->max_ulp_distance_defined) {
        result->max_ulp_distance = UINT32_MAX;
    }
    return true;
}
