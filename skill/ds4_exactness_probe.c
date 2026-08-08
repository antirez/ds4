#include "ds4_exactness_probe.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DS4_PROBE_CHUNK_FLOATS 4096u

static uint32_t ds4_probe_float_bits(float v) {
    uint32_t u = 0;
    memcpy(&u, &v, sizeof(u));
    return u;
}

/*
 * Map IEEE-754 float bits to a monotonically ordered uint32 domain so that
 * integer distance corresponds to ULP distance across negative/positive
 * finite values as well as signed zero.
 */
static uint32_t ds4_probe_ordered_float_bits(float v) {
    uint32_t u = ds4_probe_float_bits(v);
    if (u & 0x80000000u) {
        return ~u;
    }
    return u | 0x80000000u;
}

static uint32_t ds4_probe_ulp_distance(float a, float b) {
    /* NaNs do not have a meaningful ULP distance for this diagnostic. */
    if (isnan(a) || isnan(b)) return UINT32_MAX;

    uint32_t oa = ds4_probe_ordered_float_bits(a);
    uint32_t ob = ds4_probe_ordered_float_bits(b);
    return oa >= ob ? oa - ob : ob - oa;
}

static void ds4_probe_diff_init(ds4_probe_diff *r) {
    memset(r, 0, sizeof(*r));
    r->bit_exact = true;
    r->first_mismatch = UINT64_MAX;
}

bool ds4_exactness_probe_enabled(void) {
    const char *env = getenv("DS4_DSPARK_EXACTNESS_PROBE");
    return env && env[0] && strcmp(env, "0") != 0;
}

static void ds4_probe_accumulate(
        const float *actual,
        const float *expected,
        uint64_t count,
        uint64_t global_offset,
        ds4_probe_diff *r) {

    for (uint64_t i = 0; i < count; i++) {
        uint32_t abits = ds4_probe_float_bits(actual[i]);
        uint32_t ebits = ds4_probe_float_bits(expected[i]);

        if (abits == ebits) continue;

        if (r->bit_exact) {
            r->bit_exact = false;
            r->first_mismatch = global_offset + i;
            r->first_actual = actual[i];
            r->first_expected = expected[i];
        }
        r->mismatch_count++;

        float abs_diff = fabsf(actual[i] - expected[i]);
        if (isfinite(abs_diff) && abs_diff > r->max_abs_diff) {
            r->max_abs_diff = abs_diff;
        } else if (!isfinite(abs_diff)) {
            r->max_abs_diff = INFINITY;
        }

        float denom = fmaxf(fmaxf(fabsf(actual[i]), fabsf(expected[i])), FLT_MIN);
        float rel_diff = abs_diff / denom;
        if (isfinite(rel_diff) && rel_diff > r->max_rel_diff) {
            r->max_rel_diff = rel_diff;
        } else if (!isfinite(rel_diff)) {
            r->max_rel_diff = INFINITY;
        }

        uint32_t ulp = ds4_probe_ulp_distance(actual[i], expected[i]);
        if (ulp > r->max_ulp_diff) r->max_ulp_diff = ulp;
    }
}

bool ds4_probe_compare_f32_arrays(
        const float *actual,
        const float *expected,
        uint64_t count,
        ds4_probe_diff *result) {

    if (!actual || !expected || !result) return false;

    ds4_probe_diff_init(result);
    ds4_probe_accumulate(actual, expected, count, 0, result);
    return true;
}

bool ds4_probe_compare_f32_tensors(
        const ds4_gpu_tensor *actual,
        uint64_t actual_offset_bytes,
        const ds4_gpu_tensor *expected,
        uint64_t expected_offset_bytes,
        uint64_t count,
        ds4_probe_diff *result) {

    if (!actual || !expected || !result) return false;
    if ((actual_offset_bytes | expected_offset_bytes) & 3u) return false;

    ds4_probe_diff_init(result);
    if (count == 0) return true;

    float *a = malloc(sizeof(float) * DS4_PROBE_CHUNK_FLOATS);
    float *e = malloc(sizeof(float) * DS4_PROBE_CHUNK_FLOATS);
    if (!a || !e) {
        free(a);
        free(e);
        return false;
    }

    uint64_t done = 0;
    while (done < count) {
        uint64_t left = count - done;
        uint64_t n = left < DS4_PROBE_CHUNK_FLOATS
                   ? left
                   : DS4_PROBE_CHUNK_FLOATS;
        uint64_t bytes = n * sizeof(float);

        if (!ds4_gpu_tensor_read(
                    actual,
                    actual_offset_bytes + done * sizeof(float),
                    a,
                    bytes) ||
            !ds4_gpu_tensor_read(
                    expected,
                    expected_offset_bytes + done * sizeof(float),
                    e,
                    bytes)) {
            free(a);
            free(e);
            return false;
        }

        ds4_probe_accumulate(a, e, n, done, result);
        done += n;
    }

    free(a);
    free(e);
    return true;
}

void ds4_probe_print_diff(
        FILE *fp,
        const char *label,
        uint32_t layer,
        uint32_t row,
        const ds4_probe_diff *d) {

    if (!fp || !d) return;
    if (!label) label = "unnamed";

    if (d->bit_exact) {
        fprintf(fp,
                "ds4: exactness label=%s layer=%u row=%u exact=1\n",
                label, layer, row);
        return;
    }

    fprintf(fp,
            "ds4: exactness label=%s layer=%u row=%u exact=0 "
            "first=%llu actual=%.9g expected=%.9g mismatches=%llu "
            "max_abs=%.9g max_rel=%.9g max_ulp=%u\n",
            label,
            layer,
            row,
            (unsigned long long)d->first_mismatch,
            d->first_actual,
            d->first_expected,
            (unsigned long long)d->mismatch_count,
            d->max_abs_diff,
            d->max_rel_diff,
            d->max_ulp_diff);
}
