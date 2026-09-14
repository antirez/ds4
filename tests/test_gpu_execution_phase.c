// SPDX-License-Identifier: MIT
// Link against the real CPU or GPU backend: no private replacement TLS state.
#include "../ds4_gpu_phase.h"
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static void phase_require(bool ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "GPU execution phase FAIL: %s\n", message);
        exit(1);
    }
}
static const ds4_gpu_execution_phase phase_values[] = {
    DS4_GPU_PHASE_AUTO, DS4_GPU_PHASE_PREFILL, DS4_GPU_PHASE_DECODE,
    DS4_GPU_PHASE_VERIFY, DS4_GPU_PHASE_BATCH_DECODE, DS4_GPU_PHASE_MIXED,
};

static int phase_nested(unsigned depth) {
    const ds4_gpu_execution_phase expected = phase_values[depth];
    DS4_GPU_PHASE_SCOPE(scope, expected);
    phase_require(ds4_gpu_get_execution_phase() == expected, "nested scope entry");
    if (depth == 0) return 79; // Real cleanup guard must run on an early return.
    phase_require(phase_nested(depth-1) == 79, "nested return value");
    phase_require(ds4_gpu_get_execution_phase() == expected, "nested scope restoration");
    return 79;
}

static int phase_scope_exits(bool early) {
    DS4_GPU_PHASE_SCOPE(outer, DS4_GPU_PHASE_PREFILL);
    {
        DS4_GPU_PHASE_SCOPE(inner, DS4_GPU_PHASE_VERIFY);
        phase_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_VERIFY, "inner phase");
        if (early) return 17;
        goto after_inner;
    }
after_inner:
    phase_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_PREFILL, "goto cleanup");
    for (unsigned i = 0; i < 2; ++i) {
        DS4_GPU_PHASE_SCOPE(iteration, DS4_GPU_PHASE_DECODE);
        phase_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_DECODE, "loop scope");
        if (i == 0) continue;
        break;
    }
    phase_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_PREFILL, "loop cleanup");
    return 23;
}

// pthread_barrier_t is unavailable on macOS. This reusable barrier ensures
// every worker keeps a distinct phase live at the same time as the caller.
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    unsigned arrived, generation;
} phase_barrier;
static phase_barrier phase_sync = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0};
enum { PHASE_WORKERS = 3, PHASE_ITERATIONS = 32 };
static void phase_rendezvous(void) {
    phase_require(pthread_mutex_lock(&phase_sync.mutex) == 0, "barrier lock");
    const unsigned generation = phase_sync.generation;
    if (++phase_sync.arrived == PHASE_WORKERS+1) {
        phase_sync.arrived = 0;
        ++phase_sync.generation;
        phase_require(pthread_cond_broadcast(&phase_sync.cond) == 0, "barrier broadcast");
    } else {
        while (generation == phase_sync.generation)
            phase_require(pthread_cond_wait(&phase_sync.cond, &phase_sync.mutex) == 0, "barrier wait");
    }
    phase_require(pthread_mutex_unlock(&phase_sync.mutex) == 0, "barrier unlock");
}
typedef struct { ds4_gpu_execution_phase phase; } phase_worker_args;
static void *phase_worker(void *opaque) {
    const phase_worker_args *args = opaque;
    phase_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_AUTO,
                  "new thread must start AUTO, not inherit its creator phase");
    {
        DS4_GPU_PHASE_SCOPE(worker, args->phase);
        for (unsigned iteration = 0; iteration < PHASE_ITERATIONS; ++iteration) {
            phase_rendezvous();
            phase_require(ds4_gpu_get_execution_phase() == args->phase, "thread-local phase isolation");
            phase_require(phase_nested(5) == 79, "worker nested scope");
            phase_require(ds4_gpu_get_execution_phase() == args->phase, "worker restoration");
            phase_rendezvous();
        }
    }
    phase_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_AUTO, "worker exit restoration");
    return NULL;
}

int ds4_test_gpu_execution_phase(void) {
    phase_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_AUTO, "initial AUTO phase");
    for (unsigned i = 0; i < sizeof(phase_values)/sizeof(phase_values[0]); ++i) {
        const ds4_gpu_execution_phase phase = phase_values[i];
        phase_require(ds4_gpu_exchange_execution_phase(phase) == DS4_GPU_PHASE_AUTO, "exchange returns previous phase");
        phase_require(ds4_gpu_get_execution_phase() == phase, "getter returns exchanged phase");
        phase_require(ds4_gpu_execution_phase_allows_prefill(phase) ==
                      (phase == DS4_GPU_PHASE_AUTO || phase == DS4_GPU_PHASE_PREFILL), "prefill permission matrix");
        for (unsigned j = 0; j < sizeof(phase_values)/sizeof(phase_values[0]); ++j)
            phase_require(ds4_gpu_execution_phase_or(phase_values[j]) ==
                          (phase == DS4_GPU_PHASE_AUTO ? phase_values[j] : phase), "explicit parent must override inferred fallback");
        phase_require(phase_nested(5) == 79, "nested scopes return");
        phase_require(ds4_gpu_get_execution_phase() == phase, "nested scopes restore caller phase");
        phase_require(phase_scope_exits(true) == 17, "early return value");
        phase_require(ds4_gpu_get_execution_phase() == phase, "early return restores both guards");
        phase_require(phase_scope_exits(false) == 23, "normal return value");
        phase_require(ds4_gpu_get_execution_phase() == phase, "goto/loop/function guards restore phase");
        phase_require(ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_AUTO) == phase, "exchange restores AUTO");
    }
    pthread_t threads[PHASE_WORKERS];
    phase_worker_args args[PHASE_WORKERS] = {
        {DS4_GPU_PHASE_PREFILL}, {DS4_GPU_PHASE_DECODE}, {DS4_GPU_PHASE_BATCH_DECODE},
    };
    {
        DS4_GPU_PHASE_SCOPE(caller, DS4_GPU_PHASE_MIXED);
        for (unsigned i = 0; i < PHASE_WORKERS; ++i)
            phase_require(pthread_create(&threads[i], NULL, phase_worker, &args[i]) == 0, "worker creation");
        for (unsigned iteration = 0; iteration < PHASE_ITERATIONS; ++iteration) {
            phase_rendezvous();
            phase_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_MIXED, "worker must not modify caller phase");
            phase_rendezvous();
        }
        for (unsigned i = 0; i < PHASE_WORKERS; ++i)
            phase_require(pthread_join(threads[i], NULL) == 0, "worker join");
    }
    phase_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_AUTO, "final AUTO restoration");
    puts("PASS: execution phase exchange, all six phases, real scope guards on return/goto/continue/break, nested overrides and three simultaneous TLS workers.");
    return 0;
}

#ifndef DS4_GPU_EXECUTION_PHASE_TEST_NO_MAIN
int main(void) { return ds4_test_gpu_execution_phase(); }
#endif
