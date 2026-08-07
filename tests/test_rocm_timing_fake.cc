#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Host-only fake for the ROCm event timing backend.  Deliberately do not
 * provide hipEventSynchronize, hipStreamSynchronize, or hipDeviceSynchronize:
 * this test must fail to link if collection ever becomes blocking.
 */
struct fake_hip_event {
    unsigned int id;
    int recorded;
    int ready;
    float timestamp_ms;
};

typedef int hipError_t;
typedef fake_hip_event *hipEvent_t;
typedef void *hipStream_t;

static const hipError_t hipSuccess = 0;
static const hipError_t hipErrorNotReady = 600;
static const hipError_t hipErrorUnknown = 999;
static const unsigned int hipEventDefault = 0;

static unsigned int fake_create_calls;
static unsigned int fake_destroy_calls;
static unsigned int fake_record_calls;
static unsigned int fake_query_calls;
static unsigned int fake_elapsed_calls;
static unsigned int fake_next_event_id;
static unsigned int fake_fail_create_call;
static unsigned int fake_fail_record_call;
static unsigned int fake_fail_elapsed_call;
static hipError_t fake_query_result;
static int fake_recorded_events_ready;

static hipError_t hipEventCreateWithFlags(hipEvent_t *event,
                                           unsigned int flags) {
    ++fake_create_calls;
    if (!event || flags != hipEventDefault ||
        fake_create_calls == fake_fail_create_call) {
        if (event) *event = NULL;
        return hipErrorUnknown;
    }
    fake_hip_event *created =
        static_cast<fake_hip_event *>(calloc(1, sizeof(*created)));
    if (!created) return hipErrorUnknown;
    created->id = ++fake_next_event_id;
    *event = created;
    return hipSuccess;
}

static hipError_t hipEventDestroy(hipEvent_t event) {
    if (!event) return hipErrorUnknown;
    ++fake_destroy_calls;
    free(event);
    return hipSuccess;
}

static hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    ++fake_record_calls;
    if (!event || stream != NULL ||
        fake_record_calls == fake_fail_record_call) {
        return hipErrorUnknown;
    }
    event->recorded = 1;
    event->ready = fake_recorded_events_ready;
    event->timestamp_ms = static_cast<float>(fake_record_calls) * 1.25f;
    return hipSuccess;
}

static hipError_t hipEventQuery(hipEvent_t event) {
    ++fake_query_calls;
    if (!event || !event->recorded) return hipErrorUnknown;
    if (fake_query_result != hipSuccess) return fake_query_result;
    return event->ready ? hipSuccess : hipErrorNotReady;
}

static hipError_t hipEventElapsedTime(float *elapsed,
                                      hipEvent_t start,
                                      hipEvent_t end) {
    ++fake_elapsed_calls;
    if (!elapsed || !start || !end || !start->recorded || !end->recorded ||
        fake_elapsed_calls == fake_fail_elapsed_call) {
        return hipErrorUnknown;
    }
    *elapsed = end->timestamp_ms - start->timestamp_ms;
    return hipSuccess;
}

#define DS4_ROCM_BUILD 1
#include "ds4_gpu.h"
#include "rocm/ds4_rocm_timing.cuh"

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
    ds4_gpu_timing_sample sample;
    sample.slot = 91;
    sample.generation = 92;
    sample.mark_count = 93;
    sample.reserved = 94;
    return sample;
}

static void reset_fixture(void) {
    ds4_rocm_timing_cleanup();
    g_ds4_rocm_timing_next_generation = 0;
    fake_create_calls = 0;
    fake_destroy_calls = 0;
    fake_record_calls = 0;
    fake_query_calls = 0;
    fake_elapsed_calls = 0;
    fake_next_event_id = 0;
    fake_fail_create_call = 0;
    fake_fail_record_call = 0;
    fake_fail_elapsed_call = 0;
    fake_query_result = hipSuccess;
    fake_recorded_events_ready = 1;
}

static void test_happy_path(void) {
    reset_fixture();
    ds4_gpu_timing_sample sample = poisoned_sample();
    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(sample.slot == 1);
    CHECK(sample.generation != 0);
    CHECK(sample.mark_count == 1);
    CHECK(sample.reserved == 0);
    CHECK(fake_create_calls == 4u * DS4_GPU_TIMING_MAX_MARKS);
    CHECK(fake_record_calls == 1);

    CHECK(ds4_gpu_timing_mark(&sample) == 1);
    CHECK(ds4_gpu_timing_mark(&sample) == 1);
    CHECK(sample.mark_count == 3);

    float segments[2] = {-1.0f, -1.0f};
    uint32_t segment_count = 99;
    CHECK(ds4_gpu_timing_collect(&sample, segments, 2, &segment_count) == 1);
    CHECK(segment_count == 2);
    CHECK(segments[0] == 1.25f);
    CHECK(segments[1] == 1.25f);
    CHECK(fake_query_calls == 1);
    CHECK(fake_elapsed_calls == 2);
    CHECK(sample_is_reset(&sample));

    ds4_rocm_timing_cleanup();
    CHECK(fake_destroy_calls == 4u * DS4_GPU_TIMING_MAX_MARKS);
}

static void test_pool_exhaustion_and_reuse(void) {
    reset_fixture();
    ds4_gpu_timing_sample samples[5];
    uint32_t generations[4];
    for (unsigned int i = 0; i < 5; ++i) samples[i] = poisoned_sample();

    for (unsigned int i = 0; i < 4; ++i) {
        CHECK(ds4_gpu_timing_begin(&samples[i]) == 1);
        CHECK(samples[i].slot == i + 1);
        generations[i] = samples[i].generation;
    }
    CHECK(ds4_gpu_timing_begin(&samples[4]) == 0);
    CHECK(sample_is_reset(&samples[4]));
    CHECK(fake_create_calls == 4u * DS4_GPU_TIMING_MAX_MARKS);

    ds4_gpu_timing_discard(&samples[1]);
    CHECK(sample_is_reset(&samples[1]));
    CHECK(ds4_gpu_timing_begin(&samples[4]) == 1);
    CHECK(samples[4].slot == 2);
    CHECK(samples[4].generation != generations[1]);

    for (unsigned int i = 0; i < 4; ++i) ds4_gpu_timing_discard(&samples[i]);
    ds4_gpu_timing_discard(&samples[4]);
}

static void test_mark_limit_releases_slot(void) {
    reset_fixture();
    ds4_gpu_timing_sample sample = poisoned_sample();
    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    const uint32_t first_generation = sample.generation;
    for (uint32_t i = 1; i < DS4_GPU_TIMING_MAX_MARKS; ++i) {
        CHECK(ds4_gpu_timing_mark(&sample) == 1);
        CHECK(sample.mark_count == i + 1);
    }
    CHECK(fake_record_calls == DS4_GPU_TIMING_MAX_MARKS);
    CHECK(ds4_gpu_timing_mark(&sample) == 0);
    CHECK(sample_is_reset(&sample));
    CHECK(fake_record_calls == DS4_GPU_TIMING_MAX_MARKS);

    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(sample.slot == 1);
    CHECK(sample.generation != first_generation);
    ds4_gpu_timing_discard(&sample);
}

static void test_allocation_and_record_failures(void) {
    reset_fixture();
    fake_fail_create_call = 10;
    ds4_gpu_timing_sample sample = poisoned_sample();
    CHECK(ds4_gpu_timing_begin(&sample) == 0);
    CHECK(sample_is_reset(&sample));
    CHECK(fake_create_calls == 10);
    CHECK(fake_destroy_calls == 9);

    /* Allocation failure latches unavailable until backend cleanup. */
    fake_fail_create_call = 0;
    CHECK(ds4_gpu_timing_begin(&sample) == 0);
    CHECK(fake_create_calls == 10);
    ds4_rocm_timing_cleanup();
    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    ds4_gpu_timing_discard(&sample);

    reset_fixture();
    fake_fail_record_call = 1;
    sample = poisoned_sample();
    CHECK(ds4_gpu_timing_begin(&sample) == 0);
    CHECK(sample_is_reset(&sample));
    fake_fail_record_call = 0;
    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(sample.slot == 1);
    ds4_gpu_timing_discard(&sample);

    reset_fixture();
    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    fake_fail_record_call = 2;
    CHECK(ds4_gpu_timing_mark(&sample) == 0);
    CHECK(sample_is_reset(&sample));
    fake_fail_record_call = 0;
    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(sample.slot == 1);
    ds4_gpu_timing_discard(&sample);
}

static void test_collect_validation_and_failures(void) {
    reset_fixture();
    ds4_gpu_timing_sample sample = poisoned_sample();
    float segments[2] = {-1.0f, -1.0f};
    uint32_t segment_count = 99;

    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(ds4_gpu_timing_collect(&sample, segments, 2, &segment_count) == 0);
    CHECK(segment_count == 0);
    CHECK(sample_is_reset(&sample));
    CHECK(fake_query_calls == 0);

    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(ds4_gpu_timing_mark(&sample) == 1);
    segment_count = 99;
    CHECK(ds4_gpu_timing_collect(&sample, segments, 0, &segment_count) == 0);
    CHECK(segment_count == 0);
    CHECK(sample_is_reset(&sample));
    CHECK(fake_query_calls == 0);

    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(ds4_gpu_timing_mark(&sample) == 1);
    CHECK(ds4_gpu_timing_collect(&sample, NULL, 2, &segment_count) == 0);
    CHECK(sample_is_reset(&sample));

    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(ds4_gpu_timing_mark(&sample) == 1);
    CHECK(ds4_gpu_timing_collect(&sample, segments, 2, NULL) == 0);
    CHECK(sample_is_reset(&sample));

    segment_count = 99;
    CHECK(ds4_gpu_timing_collect(NULL, segments, 2, &segment_count) == 0);
    CHECK(segment_count == 0);

    /* A not-ready final marker is queried once and consumed without waiting. */
    fake_recorded_events_ready = 0;
    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(ds4_gpu_timing_mark(&sample) == 1);
    const unsigned int queries_before = fake_query_calls;
    const unsigned int elapsed_before = fake_elapsed_calls;
    segment_count = 99;
    CHECK(ds4_gpu_timing_collect(&sample, segments, 2, &segment_count) == 0);
    CHECK(segment_count == 0);
    CHECK(sample_is_reset(&sample));
    CHECK(fake_query_calls == queries_before + 1);
    CHECK(fake_elapsed_calls == elapsed_before);

    fake_recorded_events_ready = 1;
    fake_query_result = hipErrorUnknown;
    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(ds4_gpu_timing_mark(&sample) == 1);
    segment_count = 99;
    CHECK(ds4_gpu_timing_collect(&sample, segments, 2, &segment_count) == 0);
    CHECK(segment_count == 0);
    CHECK(sample_is_reset(&sample));

    fake_query_result = hipSuccess;
    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(ds4_gpu_timing_mark(&sample) == 1);
    CHECK(ds4_gpu_timing_mark(&sample) == 1);
    fake_fail_elapsed_call = fake_elapsed_calls + 2;
    segment_count = 99;
    CHECK(ds4_gpu_timing_collect(&sample, segments, 2, &segment_count) == 0);
    CHECK(segment_count == 0);
    CHECK(sample_is_reset(&sample));
}

static void test_stale_generation_and_cleanup(void) {
    reset_fixture();
    ds4_gpu_timing_sample first = poisoned_sample();
    CHECK(ds4_gpu_timing_begin(&first) == 1);
    ds4_gpu_timing_sample stale = first;
    ds4_gpu_timing_discard(&first);

    ds4_gpu_timing_sample live = poisoned_sample();
    CHECK(ds4_gpu_timing_begin(&live) == 1);
    CHECK(live.slot == stale.slot);
    CHECK(live.generation != stale.generation);
    ds4_gpu_timing_discard(&stale);
    CHECK(sample_is_reset(&stale));
    CHECK(ds4_gpu_timing_mark(&live) == 1);

    ds4_gpu_timing_sample bad_count = live;
    ++bad_count.mark_count;
    ds4_gpu_timing_discard(&bad_count);
    CHECK(sample_is_reset(&bad_count));
    CHECK(ds4_gpu_timing_mark(&live) == 1);
    ds4_gpu_timing_discard(&live);

    ds4_gpu_timing_sample before_cleanup = poisoned_sample();
    CHECK(ds4_gpu_timing_begin(&before_cleanup) == 1);
    const uint32_t old_generation = before_cleanup.generation;
    ds4_rocm_timing_cleanup();

    ds4_gpu_timing_sample after_cleanup = poisoned_sample();
    CHECK(ds4_gpu_timing_begin(&after_cleanup) == 1);
    CHECK(after_cleanup.slot == before_cleanup.slot);
    CHECK(after_cleanup.generation != old_generation);
    CHECK(ds4_gpu_timing_mark(&before_cleanup) == 0);
    CHECK(sample_is_reset(&before_cleanup));
    CHECK(ds4_gpu_timing_mark(&after_cleanup) == 1);
    ds4_gpu_timing_discard(&after_cleanup);
}

static void test_nulls_and_generation_wrap(void) {
    reset_fixture();
    CHECK(ds4_gpu_timing_begin(NULL) == 0);
    CHECK(ds4_gpu_timing_mark(NULL) == 0);
    ds4_gpu_timing_discard(NULL);

    g_ds4_rocm_timing_next_generation = UINT32_MAX;
    ds4_gpu_timing_sample sample = poisoned_sample();
    CHECK(ds4_gpu_timing_begin(&sample) == 1);
    CHECK(sample.generation == 1);
    ds4_gpu_timing_discard(&sample);
}

struct begin_thread_result {
    ds4_gpu_timing_sample sample;
    int began;
};

static void *begin_thread(void *opaque) {
    begin_thread_result *result = static_cast<begin_thread_result *>(opaque);
    result->sample = poisoned_sample();
    result->began = ds4_gpu_timing_begin(&result->sample);
    return NULL;
}

static void test_concurrent_pool_limit(void) {
    reset_fixture();
    enum { THREADS = 12 };
    pthread_t threads[THREADS];
    begin_thread_result results[THREADS];
    memset(results, 0, sizeof(results));

    for (unsigned int i = 0; i < THREADS; ++i) {
        CHECK(pthread_create(&threads[i], NULL, begin_thread, &results[i]) == 0);
    }
    for (unsigned int i = 0; i < THREADS; ++i) {
        CHECK(pthread_join(threads[i], NULL) == 0);
    }

    unsigned int began = 0;
    unsigned int slot_mask = 0;
    for (unsigned int i = 0; i < THREADS; ++i) {
        if (results[i].began) {
            ++began;
            CHECK(results[i].sample.slot >= 1 && results[i].sample.slot <= 4);
            slot_mask |= 1u << (results[i].sample.slot - 1u);
            ds4_gpu_timing_discard(&results[i].sample);
        } else {
            CHECK(sample_is_reset(&results[i].sample));
        }
    }
    CHECK(began == 4);
    CHECK(slot_mask == 0x0fu);
}

int main(void) {
    test_happy_path();
    test_pool_exhaustion_and_reuse();
    test_mark_limit_releases_slot();
    test_allocation_and_record_failures();
    test_collect_validation_and_failures();
    test_stale_generation_and_cleanup();
    test_nulls_and_generation_wrap();
    test_concurrent_pool_limit();
    ds4_rocm_timing_cleanup();

    if (failures) {
        fprintf(stderr, "ROCm timing fake: %d/%d checks failed\n", failures,
                checks);
        return 1;
    }
    printf("ROCm timing fake: %d checks passed\n", checks);
    return 0;
}
