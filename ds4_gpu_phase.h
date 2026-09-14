/* SPDX-License-Identifier: MIT */
#ifndef DS4_GPU_PHASE_H
#define DS4_GPU_PHASE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Execution intent is independent of row count: a multi-row dispatch may be
 * speculative verification, independent decode sessions, or a mixed batch.
 * AUTO preserves the existing policy for direct backend/API callers. The
 * backend stores this hint per host thread, never as process-global state. */
typedef enum ds4_gpu_execution_phase {
    DS4_GPU_PHASE_AUTO = 0,
    DS4_GPU_PHASE_PREFILL,
    DS4_GPU_PHASE_DECODE,
    DS4_GPU_PHASE_VERIFY,
    DS4_GPU_PHASE_BATCH_DECODE,
    DS4_GPU_PHASE_MIXED,
} ds4_gpu_execution_phase;

ds4_gpu_execution_phase ds4_gpu_get_execution_phase(void);
ds4_gpu_execution_phase ds4_gpu_exchange_execution_phase(ds4_gpu_execution_phase phase);

/* A phase hint gates preparation/reuse optimizations, not the legality of
 * GEMM itself: other phases may still select a GEMM according to their shape. */
static inline bool ds4_gpu_execution_phase_allows_prefill(ds4_gpu_execution_phase phase) {
    return phase == DS4_GPU_PHASE_AUTO || phase == DS4_GPU_PHASE_PREFILL;
}

/* Generic token/batch executors must not erase their caller's intent. */
static inline ds4_gpu_execution_phase ds4_gpu_execution_phase_or(
        ds4_gpu_execution_phase fallback) {
    const ds4_gpu_execution_phase current = ds4_gpu_get_execution_phase();
    return current == DS4_GPU_PHASE_AUTO ? fallback : current;
}

typedef struct ds4_gpu_execution_phase_scope {
    ds4_gpu_execution_phase previous;
} ds4_gpu_execution_phase_scope;

static inline ds4_gpu_execution_phase_scope ds4_gpu_execution_phase_scope_begin(
        ds4_gpu_execution_phase phase) {
    ds4_gpu_execution_phase_scope scope = { ds4_gpu_exchange_execution_phase(phase) };
    return scope;
}

static inline void ds4_gpu_execution_phase_scope_end(ds4_gpu_execution_phase_scope *scope) {
    (void)ds4_gpu_exchange_execution_phase(scope->previous);
}

/* Graph C is built with Clang/GCC. Cleanup restores nested scopes on normal
 * fallthrough, early return and goto out of scope. Async jobs must capture the
 * phase explicitly and open their own scope on the worker thread. */
#if defined(__GNUC__) || defined(__clang__)
#define DS4_GPU_PHASE_SCOPE(name, phase) \
    ds4_gpu_execution_phase_scope name \
        __attribute__((cleanup(ds4_gpu_execution_phase_scope_end), unused)) = \
            ds4_gpu_execution_phase_scope_begin(phase)
#endif

#ifdef __cplusplus
}
#endif

#endif
