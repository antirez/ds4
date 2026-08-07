#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>

static int checks;
static int failures;

#define CHECK(expr) do { \
    ++checks; \
    if (!(expr)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

static int sample_is_reset(const ds4_gpu_timing_sample *sample) {
    return sample && sample->slot == 0 && sample->generation == 0 &&
           sample->mark_count == 0 && sample->reserved == 0;
}

static ds4_gpu_timing_sample poisoned_sample(void) {
    ds4_gpu_timing_sample sample = {1, 2, 3, 4};
    return sample;
}

int main(void) {
    ds4_gpu_timing_sample sample = poisoned_sample();
    CHECK(ds4_gpu_timing_begin(&sample) == 0);
    CHECK(sample_is_reset(&sample));

    CHECK(ds4_gpu_timing_mark(&sample) == 0);
    CHECK(sample_is_reset(&sample));

    sample = poisoned_sample();
    float segments[2] = {17.0f, 23.0f};
    uint32_t segment_count = 99;
    CHECK(ds4_gpu_timing_collect(&sample, segments, 2, &segment_count) == 0);
    CHECK(sample_is_reset(&sample));
    CHECK(segment_count == 0);
    CHECK(segments[0] == 17.0f && segments[1] == 23.0f);

    sample = poisoned_sample();
    CHECK(ds4_gpu_timing_collect(&sample, NULL, 0, NULL) == 0);
    CHECK(sample_is_reset(&sample));

    sample = poisoned_sample();
    ds4_gpu_timing_discard(&sample);
    CHECK(sample_is_reset(&sample));

    segment_count = 99;
    CHECK(ds4_gpu_timing_begin(NULL) == 0);
    CHECK(ds4_gpu_timing_mark(NULL) == 0);
    CHECK(ds4_gpu_timing_collect(NULL, segments, 2, &segment_count) == 0);
    CHECK(segment_count == 0);
    ds4_gpu_timing_discard(NULL);

    if (failures) {
        fprintf(stderr, "GPU timing stubs: %d/%d checks failed\n", failures,
                checks);
        return 1;
    }
    printf("GPU timing stubs: %d checks passed\n", checks);
    return 0;
}
