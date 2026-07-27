#include "ds4_engine_internal.h"

/* Persistent CPU worker pool. Split out of util.c in the C++ port.
 *
 * Decode reuses these threads instead of creating pthreads in the token
 * loop. The row partitioning below must stay exactly as the C
 * implementation had it: CPU kernels rely on the same disjoint row split,
 * and the bit-exact gate compares outputs against the pre-port binary. */

namespace pulsar {

namespace {

/* Nested parallel_for calls run serially on the calling thread. */
thread_local int g_parallel_depth = 0;

} // namespace

class ThreadPool {
public:
    /* Lazy init: created on first parallel_for. A shutdown() returns the
     * pool to the never-initialized state, so a later use re-creates it —
     * the same lifecycle the C implementation had. */
    void ensure_init() {
        if (initialized_) return;

        pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);

        uint32_t n_threads = 12;
        const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (online_cpus > 0) {
            n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
        }

        const char *env = getenv("DS4_THREADS");
        if (env && env[0]) {
            long v = strtol(env, NULL, 10);
            if (v > 0) n_threads = (uint32_t)v;
        }
        if (g_requested_threads > 0) n_threads = g_requested_threads;
        if (n_threads > DS4_MAX_THREADS) n_threads = DS4_MAX_THREADS;
        if (n_threads == 0) n_threads = 1;

        pthread_mutex_init(&mutex_, NULL);
        pthread_cond_init(&work_cond_, NULL);
        pthread_cond_init(&done_cond_, NULL);
        n_threads_ = n_threads;
        n_workers_ = n_threads > 0 ? n_threads - 1 : 0;
        generation_ = 0;
        done_ = 0;
        shutdown_ = false;
        initialized_ = true;

        for (uint32_t i = 1; i < n_threads; i++) {
            if (pthread_create(&threads_[i], NULL, worker_trampoline,
                               (void *)(uintptr_t)i) != 0) {
                ds4_die("failed to create worker thread");
            }
        }
    }

    void shutdown() {
        if (!initialized_) return;

        pthread_mutex_lock(&mutex_);
        shutdown_ = true;
        generation_++;
        pthread_cond_broadcast(&work_cond_);
        pthread_mutex_unlock(&mutex_);

        for (uint32_t i = 1; i < n_threads_; i++) {
            pthread_join(threads_[i], NULL);
        }

        pthread_cond_destroy(&done_cond_);
        pthread_cond_destroy(&work_cond_);
        pthread_mutex_destroy(&mutex_);
        memset(this, 0, sizeof(*this));
    }

    /* Run a row-parallel CPU kernel, falling back to serial execution for
     * small jobs or nested calls where spawning more work would only add
     * latency. */
    void parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx,
                               uint64_t min_parallel_rows) {
        ensure_init();

        if (g_parallel_depth > 0 || n_threads_ <= 1 || n_rows < min_parallel_rows) {
            fn(ctx, 0, n_rows);
            return;
        }

        pthread_mutex_lock(&mutex_);
        fn_ = fn;
        ctx_ = ctx;
        n_rows_ = n_rows;
        done_ = 0;
        generation_++;
        pthread_cond_broadcast(&work_cond_);

        const uint64_t rows_per_thread = (n_rows + n_threads_ - 1) / n_threads_;
        uint64_t main_row1 = rows_per_thread;
        if (main_row1 > n_rows) main_row1 = n_rows;
        pthread_mutex_unlock(&mutex_);

        if (main_row1 > 0) {
            g_parallel_depth++;
            fn(ctx, 0, main_row1);
            g_parallel_depth--;
        }

        pthread_mutex_lock(&mutex_);
        while (done_ < n_workers_) {
            pthread_cond_wait(&done_cond_, &mutex_);
        }
        pthread_mutex_unlock(&mutex_);
    }

private:
    static void *worker_trampoline(void *arg);

    void worker_main(uint32_t tid) {
        uint32_t seen_generation = 0;

        for (;;) {
            pthread_mutex_lock(&mutex_);
            while (seen_generation == generation_ && !shutdown_) {
                pthread_cond_wait(&work_cond_, &mutex_);
            }
            if (shutdown_) {
                pthread_mutex_unlock(&mutex_);
                return;
            }

            seen_generation = generation_;
            ds4_parallel_fn fn = fn_;
            void *ctx = ctx_;
            const uint64_t n_rows = n_rows_;
            const uint32_t n_threads = n_threads_;
            pthread_mutex_unlock(&mutex_);

            const uint64_t rows_per_thread = (n_rows + n_threads - 1) / n_threads;
            const uint64_t row0 = (uint64_t)tid * rows_per_thread;
            uint64_t row1 = row0 + rows_per_thread;
            if (row1 > n_rows) row1 = n_rows;
            if (row0 < row1) {
                g_parallel_depth++;
                fn(ctx, row0, row1);
                g_parallel_depth--;
            }

            pthread_mutex_lock(&mutex_);
            done_++;
            if (done_ == n_workers_) {
                pthread_cond_signal(&done_cond_);
            }
            pthread_mutex_unlock(&mutex_);
        }
    }

    pthread_t threads_[DS4_MAX_THREADS] = {};
    pthread_mutex_t mutex_ = {};
    pthread_cond_t work_cond_ = {};
    pthread_cond_t done_cond_ = {};
    uint32_t n_threads_ = 0;
    uint32_t n_workers_ = 0;
    uint32_t generation_ = 0;
    uint32_t done_ = 0;
    bool initialized_ = false;
    bool shutdown_ = false;
    ds4_parallel_fn fn_ = nullptr;
    void *ctx_ = nullptr;
    uint64_t n_rows_ = 0;
};

static ThreadPool g_pool;

void *ThreadPool::worker_trampoline(void *arg) {
    g_pool.worker_main((uint32_t)(uintptr_t)arg);
    return NULL;
}

} // namespace pulsar



uint32_t g_requested_threads;



void ds4_threads_shutdown(void) {
    pulsar::g_pool.shutdown();
}



void ds4_parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx, uint64_t min_parallel_rows) {
    pulsar::g_pool.parallel_for_min_rows(n_rows, fn, ctx, min_parallel_rows);
}



void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx) {
    ds4_parallel_for_min_rows(n_rows, fn, ctx, 512);
}
