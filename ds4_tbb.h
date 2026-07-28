#ifndef DS4_TBB_H
#define DS4_TBB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Matches ds4_parallel_fn in ds4.c */
typedef void (*ds4_tbb_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);

/* Initialize TBB thread pool with n_threads (0 = default). */
void ds4_tbb_init(uint32_t n_threads);

/* Parallel for loop using TBB work-stealing scheduler. */
void ds4_tbb_parallel_for(uint64_t n_rows, ds4_tbb_parallel_fn fn, void *ctx);

/* Shutdown TBB resources (called at exit). */
void ds4_tbb_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_TBB_H */
