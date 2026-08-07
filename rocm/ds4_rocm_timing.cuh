#ifndef DS4_ROCM_TIMING_CUH
#define DS4_ROCM_TIMING_CUH

/*
 * A small, process-wide pool keeps diagnostic timing out of the allocation
 * hot path.  Events use the default timing-enabled mode and are recorded only
 * on stream 0.  The public API intentionally exposes no wait operation.
 */
enum { DS4_ROCM_TIMING_POOL_SAMPLES = 4u };

struct ds4_rocm_timing_slot {
    hipEvent_t events[DS4_GPU_TIMING_MAX_MARKS];
    uint32_t generation;
    uint32_t mark_count;
    int in_use;
};

static pthread_mutex_t g_ds4_rocm_timing_mutex = PTHREAD_MUTEX_INITIALIZER;
static ds4_rocm_timing_slot
    g_ds4_rocm_timing_slots[DS4_ROCM_TIMING_POOL_SAMPLES];
/* 0 = lazy/uninitialized, 1 = ready, -1 = unavailable after allocation error. */
static int g_ds4_rocm_timing_pool_state;
static uint32_t g_ds4_rocm_timing_next_generation;

static void ds4_rocm_timing_sample_reset(ds4_gpu_timing_sample *sample) {
    if (!sample) return;
    sample->slot = 0;
    sample->generation = 0;
    sample->mark_count = 0;
    sample->reserved = 0;
}

static void ds4_rocm_timing_destroy_events_locked(void) {
    for (uint32_t i = 0; i < DS4_ROCM_TIMING_POOL_SAMPLES; ++i) {
        ds4_rocm_timing_slot *slot = &g_ds4_rocm_timing_slots[i];
        for (uint32_t j = 0; j < DS4_GPU_TIMING_MAX_MARKS; ++j) {
            if (slot->events[j]) {
                (void)hipEventDestroy(slot->events[j]);
                slot->events[j] = NULL;
            }
        }
        slot->generation = 0;
        slot->mark_count = 0;
        slot->in_use = 0;
    }
}

static int ds4_rocm_timing_pool_init_locked(void) {
    if (g_ds4_rocm_timing_pool_state != 0) {
        return g_ds4_rocm_timing_pool_state > 0;
    }

    for (uint32_t i = 0; i < DS4_ROCM_TIMING_POOL_SAMPLES; ++i) {
        for (uint32_t j = 0; j < DS4_GPU_TIMING_MAX_MARKS; ++j) {
            hipError_t err = hipEventCreateWithFlags(
                &g_ds4_rocm_timing_slots[i].events[j], hipEventDefault);
            if (err != hipSuccess) {
                ds4_rocm_timing_destroy_events_locked();
                g_ds4_rocm_timing_pool_state = -1;
                return 0;
            }
        }
    }
    g_ds4_rocm_timing_pool_state = 1;
    return 1;
}

static ds4_rocm_timing_slot *ds4_rocm_timing_slot_locked(
        const ds4_gpu_timing_sample *sample) {
    if (!sample || sample->slot == 0 ||
        sample->slot > DS4_ROCM_TIMING_POOL_SAMPLES) {
        return NULL;
    }
    ds4_rocm_timing_slot *slot =
        &g_ds4_rocm_timing_slots[sample->slot - 1u];
    if (!slot->in_use || slot->generation != sample->generation ||
        slot->mark_count != sample->mark_count) {
        return NULL;
    }
    return slot;
}

static void ds4_rocm_timing_release_locked(ds4_rocm_timing_slot *slot) {
    if (!slot) return;
    slot->mark_count = 0;
    slot->in_use = 0;
}

extern "C" int ds4_gpu_timing_begin(ds4_gpu_timing_sample *sample) {
    if (!sample) return 0;
    ds4_rocm_timing_sample_reset(sample);

    (void)pthread_mutex_lock(&g_ds4_rocm_timing_mutex);
    if (!ds4_rocm_timing_pool_init_locked()) {
        (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
        return 0;
    }

    ds4_rocm_timing_slot *slot = NULL;
    uint32_t slot_index = 0;
    for (uint32_t i = 0; i < DS4_ROCM_TIMING_POOL_SAMPLES; ++i) {
        if (!g_ds4_rocm_timing_slots[i].in_use) {
            slot = &g_ds4_rocm_timing_slots[i];
            slot_index = i;
            break;
        }
    }
    if (!slot) {
        (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
        return 0;
    }

    ++g_ds4_rocm_timing_next_generation;
    if (g_ds4_rocm_timing_next_generation == 0) {
        ++g_ds4_rocm_timing_next_generation;
    }
    slot->generation = g_ds4_rocm_timing_next_generation;
    slot->mark_count = 1;
    slot->in_use = 1;

    hipError_t err = hipEventRecord(slot->events[0], (hipStream_t)0);
    if (err != hipSuccess) {
        ds4_rocm_timing_release_locked(slot);
        (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
        return 0;
    }

    sample->slot = slot_index + 1u;
    sample->generation = slot->generation;
    sample->mark_count = 1;
    sample->reserved = 0;
    (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
    return 1;
}

extern "C" int ds4_gpu_timing_mark(ds4_gpu_timing_sample *sample) {
    if (!sample) return 0;

    (void)pthread_mutex_lock(&g_ds4_rocm_timing_mutex);
    ds4_rocm_timing_slot *slot = ds4_rocm_timing_slot_locked(sample);
    if (!slot || slot->mark_count >= DS4_GPU_TIMING_MAX_MARKS) {
        ds4_rocm_timing_release_locked(slot);
        (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
        ds4_rocm_timing_sample_reset(sample);
        return 0;
    }

    const uint32_t mark = slot->mark_count;
    hipError_t err = hipEventRecord(slot->events[mark], (hipStream_t)0);
    if (err != hipSuccess) {
        ds4_rocm_timing_release_locked(slot);
        (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
        ds4_rocm_timing_sample_reset(sample);
        return 0;
    }

    slot->mark_count = mark + 1u;
    sample->mark_count = slot->mark_count;
    (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
    return 1;
}

extern "C" int ds4_gpu_timing_collect(ds4_gpu_timing_sample *sample,
                                        float *segment_ms,
                                        uint32_t segment_capacity,
                                        uint32_t *segment_count) {
    if (segment_count) *segment_count = 0;
    if (!sample || !segment_count) {
        ds4_gpu_timing_discard(sample);
        return 0;
    }

    (void)pthread_mutex_lock(&g_ds4_rocm_timing_mutex);
    ds4_rocm_timing_slot *slot = ds4_rocm_timing_slot_locked(sample);
    const uint32_t count = slot && slot->mark_count > 0
        ? slot->mark_count - 1u : 0u;
    if (!slot || count == 0 || !segment_ms || segment_capacity < count) {
        ds4_rocm_timing_release_locked(slot);
        (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
        ds4_rocm_timing_sample_reset(sample);
        return 0;
    }

    /* Query is nonblocking.  Same-stream ordering means a ready final marker
     * also guarantees that every preceding marker is complete. */
    hipError_t err = hipEventQuery(slot->events[slot->mark_count - 1u]);
    if (err == hipSuccess) {
        for (uint32_t i = 0; i < count; ++i) {
            err = hipEventElapsedTime(&segment_ms[i],
                                      slot->events[i],
                                      slot->events[i + 1u]);
            if (err != hipSuccess) break;
        }
    }

    ds4_rocm_timing_release_locked(slot);
    (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
    ds4_rocm_timing_sample_reset(sample);
    if (err != hipSuccess) return 0;
    *segment_count = count;
    return 1;
}

extern "C" void ds4_gpu_timing_discard(ds4_gpu_timing_sample *sample) {
    if (!sample) return;
    (void)pthread_mutex_lock(&g_ds4_rocm_timing_mutex);
    ds4_rocm_timing_slot *slot = ds4_rocm_timing_slot_locked(sample);
    ds4_rocm_timing_release_locked(slot);
    (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
    ds4_rocm_timing_sample_reset(sample);
}

/* ds4_gpu_cleanup already synchronizes the device before calling this. */
static void ds4_rocm_timing_cleanup(void) {
    (void)pthread_mutex_lock(&g_ds4_rocm_timing_mutex);
    ds4_rocm_timing_destroy_events_locked();
    g_ds4_rocm_timing_pool_state = 0;
    (void)pthread_mutex_unlock(&g_ds4_rocm_timing_mutex);
}

#endif /* DS4_ROCM_TIMING_CUH */
