#include "ds4_float_compare.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

static int failures;

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,          \
                    #condition);                                               \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

static float float_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void test_identical_finite_arrays(void) {
    const float values[] = {1.0f, -2.5f, 0.25f, 1024.0f};
    ds4_float_compare_result result;

    CHECK(ds4_float_compare_exact(values, values, ARRAY_LEN(values), &result));
    CHECK(result.valid_input);
    CHECK(result.bit_exact);
    CHECK(result.mismatch_count == 0);
    CHECK(result.first_mismatch_index == SIZE_MAX);
    CHECK(result.max_abs_diff_defined && result.max_abs_diff == 0.0);
    CHECK(result.max_rel_diff_defined && result.max_rel_diff == 0.0);
    CHECK(result.max_ulp_distance_defined && result.max_ulp_distance == 0);
}

static void test_first_and_multiple_mismatches(void) {
    const float actual[] = {1.0f, 2.0f, 3.0f, 5.0f};
    const float expected[] = {1.0f, 4.0f, 3.0f, 8.0f};
    ds4_float_compare_result result;

    CHECK(ds4_float_compare_exact(actual, expected, ARRAY_LEN(actual), &result));
    CHECK(!result.bit_exact);
    CHECK(result.first_mismatch_index == 1);
    CHECK(result.mismatch_count == 2);
    CHECK(result.first_actual == 2.0f);
    CHECK(result.first_expected == 4.0f);
    CHECK(result.first_actual_bits == UINT32_C(0x40000000));
    CHECK(result.first_expected_bits == UINT32_C(0x40800000));
    CHECK(result.max_abs_diff_defined && result.max_abs_diff == 3.0);
}

static void test_zero_length_and_invalid_inputs(void) {
    const float value = 1.0f;
    ds4_float_compare_result result;

    CHECK(ds4_float_compare_exact(NULL, NULL, 0, &result));
    CHECK(result.valid_input && result.bit_exact && result.length == 0);
    CHECK(result.first_mismatch_index == SIZE_MAX);

    CHECK(!ds4_float_compare_exact(NULL, &value, 1, &result));
    CHECK(!result.valid_input && !result.bit_exact);
    CHECK(result.first_mismatch_index == SIZE_MAX);
    CHECK(!ds4_float_compare_exact(&value, NULL, 1, &result));
    CHECK(!ds4_float_compare_exact(&value, &value, 1, NULL));
}

static void test_signed_zero(void) {
    const float actual = 0.0f;
    const float expected = -0.0f;
    ds4_float_compare_result result;

    CHECK(ds4_float_compare_exact(&actual, &expected, 1, &result));
    CHECK(!result.bit_exact && result.mismatch_count == 1);
    CHECK(result.first_actual_bits == UINT32_C(0x00000000));
    CHECK(result.first_expected_bits == UINT32_C(0x80000000));
    CHECK(result.max_abs_diff == 0.0);
    CHECK(result.max_rel_diff == 0.0);
    CHECK(result.max_ulp_distance_defined && result.max_ulp_distance == 1);
}

static void test_nan_payloads(void) {
    const float nan_a = float_from_bits(UINT32_C(0x7fc00001));
    const float nan_a_copy = float_from_bits(UINT32_C(0x7fc00001));
    const float nan_b = float_from_bits(UINT32_C(0x7fc00002));
    ds4_float_compare_result result;

    CHECK(ds4_float_compare_exact(&nan_a, &nan_a_copy, 1, &result));
    CHECK(result.bit_exact && result.mismatch_count == 0);

    CHECK(ds4_float_compare_exact(&nan_a, &nan_b, 1, &result));
    CHECK(!result.bit_exact && result.mismatch_count == 1);
    CHECK(result.first_actual_bits == UINT32_C(0x7fc00001));
    CHECK(result.first_expected_bits == UINT32_C(0x7fc00002));
    CHECK(!result.max_abs_diff_defined && isnan(result.max_abs_diff));
    CHECK(!result.max_rel_diff_defined && isnan(result.max_rel_diff));
    CHECK(!result.max_ulp_distance_defined &&
          result.max_ulp_distance == UINT32_MAX);
}

static void test_infinities(void) {
    const float pos_inf = INFINITY;
    const float neg_inf = -INFINITY;
    ds4_float_compare_result result;

    CHECK(ds4_float_compare_exact(&pos_inf, &pos_inf, 1, &result));
    CHECK(result.bit_exact);

    CHECK(ds4_float_compare_exact(&pos_inf, &neg_inf, 1, &result));
    CHECK(!result.bit_exact);
    CHECK(result.max_abs_diff_defined && isinf(result.max_abs_diff));
    CHECK(!result.max_rel_diff_defined && isnan(result.max_rel_diff));
    CHECK(!result.max_ulp_distance_defined &&
          result.max_ulp_distance == UINT32_MAX);
}

static void test_adjacent_and_negative_finite_values(void) {
    const float one = 1.0f;
    const float one_next = nextafterf(one, INFINITY);
    const float neg_one = -1.0f;
    const float neg_one_next = nextafterf(neg_one, -INFINITY);
    ds4_float_compare_result result;

    CHECK(ds4_float_compare_exact(&one, &one_next, 1, &result));
    CHECK(!result.bit_exact);
    CHECK(result.max_ulp_distance_defined && result.max_ulp_distance == 1);

    CHECK(ds4_float_compare_exact(&neg_one, &neg_one_next, 1, &result));
    CHECK(!result.bit_exact);
    CHECK(result.max_ulp_distance_defined && result.max_ulp_distance == 1);
}

static void test_mixed_sign_values(void) {
    const float actual[] = {-1.0f, -2.0f};
    const float expected[] = {1.0f, -2.0f};
    ds4_float_compare_result result;

    CHECK(ds4_float_compare_exact(actual, expected, ARRAY_LEN(actual), &result));
    CHECK(!result.bit_exact && result.mismatch_count == 1);
    CHECK(result.max_abs_diff_defined && result.max_abs_diff == 2.0);
    CHECK(result.max_rel_diff_defined && result.max_rel_diff == 2.0);
    CHECK(result.max_ulp_distance_defined && result.max_ulp_distance > 1);
}

static void test_large_deterministic_sparse_mismatches(void) {
    enum { LENGTH = 16384 };
    float *actual = malloc(sizeof(*actual) * LENGTH);
    float *expected = malloc(sizeof(*expected) * LENGTH);
    ds4_float_compare_result result;
    size_t i;

    CHECK(actual != NULL && expected != NULL);
    if (actual == NULL || expected == NULL) {
        free(actual);
        free(expected);
        return;
    }

    for (i = 0; i < LENGTH; ++i) {
        actual[i] = (float)((int)(i % 257) - 128) * 0.125f;
        expected[i] = actual[i];
    }
    expected[17] = nextafterf(actual[17], INFINITY);
    expected[4096] = actual[4096] + 4.0f;
    expected[16383] = -actual[16383];

    CHECK(ds4_float_compare_exact(actual, expected, LENGTH, &result));
    CHECK(!result.bit_exact);
    CHECK(result.first_mismatch_index == 17);
    CHECK(result.mismatch_count == 3);
    CHECK(result.max_ulp_distance_defined);

    free(actual);
    free(expected);
}

int main(void) {
    test_identical_finite_arrays();
    test_first_and_multiple_mismatches();
    test_zero_length_and_invalid_inputs();
    test_signed_zero();
    test_nan_payloads();
    test_infinities();
    test_adjacent_and_negative_finite_values();
    test_mixed_sign_values();
    test_large_deterministic_sparse_mismatches();

    if (failures != 0) {
        fprintf(stderr, "test_float_compare: %d failure(s)\n", failures);
        return 1;
    }
    puts("test_float_compare: PASS");
    return 0;
}
