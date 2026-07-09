#include "ds4_tbb.h"
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#include <cstdlib>
#include <cstdio>

static tbb::global_control *g_tbb_control = nullptr;

extern "C" void ds4_tbb_init(uint32_t n_threads) {
    if (g_tbb_control) return;
    if (n_threads == 0) {
        const char *env = std::getenv("DS4_THREADS");
        if (env && env[0]) {
            long v = std::strtol(env, nullptr, 10);
            if (v > 0) n_threads = (uint32_t)v;
        }
    }
    if (n_threads > 0) {
        g_tbb_control = new tbb::global_control(
            tbb::global_control::max_allowed_parallelism, n_threads);
    }
}

extern "C" void ds4_tbb_parallel_for(uint64_t n_rows, ds4_tbb_parallel_fn fn, void *ctx) {
    tbb::parallel_for(tbb::blocked_range<uint64_t>(0, n_rows),
        [fn, ctx](const tbb::blocked_range<uint64_t> &r) {
            fn(ctx, r.begin(), r.end());
        });
}

extern "C" void ds4_tbb_shutdown(void) {
    delete g_tbb_control;
    g_tbb_control = nullptr;
}
