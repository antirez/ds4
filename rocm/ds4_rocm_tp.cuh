/* One compute stream, one CPU transport service. Each queue slot has exact
 * ready/done/consumed generations. The service never calls HIP; the GPU
 * releases a slot only after its guarded payload consumer has finished. */
static constexpr unsigned ROCM_TP_QUEUE = 64;
struct alignas(64) rocm_tp_flags {
    uint64_t ready, done, consumed;
};
struct rocm_tp_shared {
    uint32_t abort;
    rocm_tp_flags slots[ROCM_TP_QUEUE];
};
struct rocm_tp_job {
    uint64_t seq, bytes;
    uint32_t layer, arg, kind;
    const void *out;
    void *in;
};
struct rocm_tp_state {
    bool active = false, started = false;
    uint64_t seq = 0, posted = 0, pending = 0, timeout_ticks = 0;
    uint32_t pending_count = 0;
    bool pending_deferred = false;
    void *pending_big_in = nullptr;
    uint64_t pending_big_bytes = 0;
    ds4_gpu_tensor *slab = nullptr, *flags = nullptr;
    ds4_gpu_tensor *big_out = nullptr, *big_in = nullptr;
    rocm_tp_shared *host = nullptr, *device = nullptr;
    rocm_tp_job jobs[ROCM_TP_QUEUE] = {};
    pthread_t thread{};
    ds4_gpu_tp_exchange_fn exchange = nullptr;
    ds4_gpu_tp_batch_exchange_fn batch = nullptr;
    ds4_gpu_tp_big_exchange_fn big = nullptr;
    void *ud = nullptr;
};
static rocm_tp_state g_rocm_tp;
static int g_rocm_tp_failed;
static pthread_mutex_t g_rocm_tp_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_rocm_tp_cond = PTHREAD_COND_INITIALIZER;

static int rocm_tp_fail(void) {
    __atomic_store_n(&g_rocm_tp_failed, 1, __ATOMIC_RELEASE);
    if (g_rocm_tp.host) __atomic_store_n(&g_rocm_tp.host->abort, 1u, __ATOMIC_RELEASE);
    pthread_mutex_lock(&g_rocm_tp_mutex);
    pthread_cond_broadcast(&g_rocm_tp_cond);
    pthread_mutex_unlock(&g_rocm_tp_mutex);
    return 0;
}
extern "C" int ds4_gpu_tp_failed(void) {
    return __atomic_load_n(&g_rocm_tp_failed, __ATOMIC_ACQUIRE) ||
        (g_rocm_tp.host && __atomic_load_n(&g_rocm_tp.host->abort, __ATOMIC_ACQUIRE));
}
static __device__ bool rocm_tp_aborted(rocm_tp_shared *s) {
    return __hip_atomic_load(&s->abort, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 0;
}
static __global__ void rocm_tp_arrive(rocm_tp_shared *s, unsigned slot, uint64_t seq) {
    if (!rocm_tp_aborted(s))
        __hip_atomic_store(&s->slots[slot].ready, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
static __global__ void rocm_tp_wait(rocm_tp_shared *s, unsigned slot, uint64_t seq, uint64_t ticks) {
    const uint64_t start = wall_clock64();
    while (!rocm_tp_aborted(s)) {
        uint64_t done = __hip_atomic_load(&s->slots[slot].done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if (done == seq) return;
        if (done > seq || (uint64_t)wall_clock64() - start >= ticks) {
            __hip_atomic_store(&s->abort, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            return;
        }
        __builtin_amdgcn_s_sleep(1);
    }
}
static __global__ void rocm_tp_copy(rocm_tp_shared *s, float *out, const float *in, uint64_t n) {
    if (rocm_tp_aborted(s)) return;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        out[i] = in[i];
}
static __global__ void rocm_tp_add(rocm_tp_shared *s, unsigned slot, uint64_t seq,
                                  float *out, const float *a, const float *b, uint32_t n) {
    if (rocm_tp_aborted(s) ||
        __hip_atomic_load(&s->slots[slot].done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != seq) return;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        out[i] = a[i] + b[i];
}
static __global__ void rocm_tp_release(rocm_tp_shared *s, unsigned slot, uint64_t seq) {
    __hip_atomic_store(&s->slots[slot].consumed, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
static void rocm_tp_pause(unsigned *spins) {
    if (++*spins < 100) { sched_yield(); return; }
    const struct timespec delay = {0, 10000};
    nanosleep(&delay, nullptr);
}
static void *rocm_tp_service(void *) {
    for (uint64_t seq = 1;; ++seq) {
        pthread_mutex_lock(&g_rocm_tp_mutex);
        while (__atomic_load_n(&g_rocm_tp.posted, __ATOMIC_ACQUIRE) < seq && !ds4_gpu_tp_failed())
            pthread_cond_wait(&g_rocm_tp_cond, &g_rocm_tp_mutex);
        pthread_mutex_unlock(&g_rocm_tp_mutex);
        if (ds4_gpu_tp_failed()) return nullptr;
        const unsigned slot = (unsigned)((seq - 1) % ROCM_TP_QUEUE);
        unsigned spins = 0;
        for (;;) {
            if (ds4_gpu_tp_failed()) return nullptr;
            uint64_t ready = __atomic_load_n(&g_rocm_tp.host->slots[slot].ready, __ATOMIC_ACQUIRE);
            if (ready == seq) break;
            if (ready > seq) { rocm_tp_fail(); return nullptr; }
            rocm_tp_pause(&spins);
        }
        const rocm_tp_job job = g_rocm_tp.jobs[slot];
        int ok = job.seq == seq;
        if (ok && job.kind == 0) ok = g_rocm_tp.exchange(g_rocm_tp.ud, job.layer, job.arg, seq);
        else if (ok && job.kind == 1) ok = g_rocm_tp.batch(g_rocm_tp.ud, job.layer, job.arg, seq);
        else if (ok && job.kind == 2) ok = g_rocm_tp.big(g_rocm_tp.ud, job.layer, seq, job.out, job.in, job.bytes);
        else ok = 0;
        if (!ok) { rocm_tp_fail(); return nullptr; }
        if (ds4_gpu_tp_failed()) return nullptr;
        __atomic_store_n(&g_rocm_tp.host->slots[slot].done, seq, __ATOMIC_RELEASE);
    }
}
extern "C" void ds4_gpu_tp_shutdown(void) {
    if (!g_rocm_tp.active && !g_rocm_tp.flags) return;
    /* Wake GPU waits and the idle service before draining either side. The
     * transport callback has its own bounded I/O deadline. */
    rocm_tp_fail();
    if (g_rocm_tp.started) pthread_join(g_rocm_tp.thread, nullptr);
    (void)cudaDeviceSynchronize();
    ds4_gpu_tensor_free(g_rocm_tp.big_in);
    ds4_gpu_tensor_free(g_rocm_tp.big_out);
    ds4_gpu_tensor_free(g_rocm_tp.flags);
    g_rocm_tp = rocm_tp_state{};
    __atomic_store_n(&g_rocm_tp_failed, 0, __ATOMIC_RELEASE);
}
extern "C" int ds4_gpu_tp_init(uint32_t rank, ds4_gpu_tensor *slab,
        uint64_t gpu_flags_off, uint64_t out_off, uint64_t vec_bytes,
        ds4_gpu_tp_exchange_fn fn, void *ud) {
    if (g_rocm_tp.active || !g_deepseek41_model || rank > 1u ||
        !slab || !slab->host_ptr || !fn || vec_bytes != 20480u ||
        gpu_flags_off > slab->bytes || 80u * sizeof(uint32_t) > slab->bytes - gpu_flags_off ||
        out_off > slab->bytes || 80u * vec_bytes > slab->bytes - out_off) return 0;
    int device, khz;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device) != hipSuccess || khz <= 0) return 0;
    g_rocm_tp.flags = ds4_gpu_tensor_alloc_coherent(sizeof(rocm_tp_shared));
    if (!g_rocm_tp.flags) return 0;
    g_rocm_tp.host = (rocm_tp_shared *)g_rocm_tp.flags->host_ptr;
    g_rocm_tp.device = (rocm_tp_shared *)g_rocm_tp.flags->ptr;
    memset(g_rocm_tp.host, 0, sizeof(rocm_tp_shared));
    g_rocm_tp.timeout_ticks = (uint64_t)khz * 1000u * 5u;
    g_rocm_tp.active = true;
    g_rocm_tp.slab = slab;
    g_rocm_tp.exchange = fn;
    g_rocm_tp.ud = ud;
    __atomic_store_n(&g_rocm_tp_failed, 0, __ATOMIC_RELEASE);
    if (pthread_create(&g_rocm_tp.thread, nullptr, rocm_tp_service, nullptr)) {
        ds4_gpu_tp_shutdown(); return 0;
    }
    g_rocm_tp.started = true;
    return 1;
}
extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) { g_rocm_tp.batch = fn; }
extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) { g_rocm_tp.big = fn; }
extern "C" void ds4_gpu_tp_set_session_batch_mode(int enabled) { (void)enabled; }
extern "C" int ds4_gpu_tp_decode_split_flush_safe(void) { return 0; }

static int rocm_tp_enqueue(rocm_tp_job job, uint32_t count, bool defer_wait = false) {
    if (!g_rocm_tp.active || g_rocm_tp.pending || ds4_gpu_tp_failed() || g_rocm_tp.seq == UINT64_MAX)
        return rocm_tp_fail();
    const uint64_t seq = g_rocm_tp.seq + 1;
    const unsigned slot = (unsigned)((seq - 1) % ROCM_TP_QUEUE);
    unsigned spins = 0;
    while (seq > ROCM_TP_QUEUE && __atomic_load_n(&g_rocm_tp.host->slots[slot].consumed, __ATOMIC_ACQUIRE) != seq - ROCM_TP_QUEUE) {
        if (ds4_gpu_tp_failed()) return 0;
        rocm_tp_pause(&spins);
    }
    job.seq = seq;
    g_rocm_tp.jobs[slot] = job;
    rocm_tp_arrive<<<1, 1>>>(g_rocm_tp.device, slot, seq);
    if (!defer_wait) rocm_tp_wait<<<1, 1>>>(g_rocm_tp.device, slot, seq, g_rocm_tp.timeout_ticks);
    if (!cuda_ok(cudaGetLastError(), "TP gate enqueue")) return rocm_tp_fail();
    g_rocm_tp.seq = g_rocm_tp.pending = seq;
    g_rocm_tp.pending_count = count;
    g_rocm_tp.pending_deferred = defer_wait;
    pthread_mutex_lock(&g_rocm_tp_mutex);
    __atomic_store_n(&g_rocm_tp.posted, seq, __ATOMIC_RELEASE);
    pthread_cond_signal(&g_rocm_tp_cond);
    pthread_mutex_unlock(&g_rocm_tp_mutex);
    return 1;
}
extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    if (layer >= 40u || gate >= 2u) return rocm_tp_fail();
    return rocm_tp_enqueue({0, 0, layer, gate, 0, nullptr, nullptr}, 5120u);
}
extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    if (!g_rocm_tp.batch || layer >= 40u || !rows || rows > 8u) return rocm_tp_fail();
    return rocm_tp_enqueue({0, 0, layer, rows, 1, nullptr, nullptr}, rows * 5120u);
}
extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
        const ds4_gpu_tensor *out_t, ds4_gpu_tensor *in_t, uint64_t bytes) {
    if (!g_rocm_tp.active || g_rocm_tp.pending || !g_rocm_tp.big || ds4_gpu_tp_failed() || layer >= 40u ||
        !rows || rows > UINT32_MAX / 5120u || bytes != (uint64_t)rows * 20480u || !out_t || !in_t ||
        bytes > out_t->bytes || bytes > in_t->bytes) return rocm_tp_fail();
    if (!g_rocm_tp.big_out || g_rocm_tp.big_out->bytes < bytes) {
        if (!ds4_gpu_synchronize() || ds4_gpu_tp_failed()) return rocm_tp_fail();
        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_coherent(bytes);
        ds4_gpu_tensor *rx = ds4_gpu_tensor_alloc_coherent(bytes);
        if (!tx || !rx) { ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(rx); return rocm_tp_fail(); }
        ds4_gpu_tensor_free(g_rocm_tp.big_out); ds4_gpu_tensor_free(g_rocm_tp.big_in);
        g_rocm_tp.big_out = tx; g_rocm_tp.big_in = rx;
    }
    rocm_tp_copy<<<256, 256>>>(g_rocm_tp.device, (float *)g_rocm_tp.big_out->ptr, (const float *)out_t->ptr, bytes / 4);
    if (!rocm_tp_enqueue({0, bytes, layer, rows, 2, g_rocm_tp.big_out->host_ptr, g_rocm_tp.big_in->host_ptr}, rows * 5120u)) return 0;
    rocm_tp_copy<<<256, 256>>>(g_rocm_tp.device, (float *)in_t->ptr, (const float *)g_rocm_tp.big_in->ptr, bytes / 4);
    return cuda_ok(cudaGetLastError(), "TP receive enqueue") || rocm_tp_fail();
}
extern "C" int ds4_gpu_tp_big_gate_begin(uint32_t layer, uint32_t rows,
        const ds4_gpu_tensor *out_t, ds4_gpu_tensor *in_t, uint64_t bytes) {
    if (!g_rocm_tp.active || g_rocm_tp.pending || !g_rocm_tp.big || ds4_gpu_tp_failed() || layer >= 40u ||
        !rows || rows > UINT32_MAX / 5120u || bytes != (uint64_t)rows * 20480u || !out_t || !in_t ||
        bytes > out_t->bytes || bytes > in_t->bytes) return rocm_tp_fail();
    if (!g_rocm_tp.big_out || g_rocm_tp.big_out->bytes < bytes) {
        if (!ds4_gpu_synchronize() || ds4_gpu_tp_failed()) return rocm_tp_fail();
        ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc_coherent(bytes);
        ds4_gpu_tensor *rx = ds4_gpu_tensor_alloc_coherent(bytes);
        if (!tx || !rx) { ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(rx); return rocm_tp_fail(); }
        ds4_gpu_tensor_free(g_rocm_tp.big_out); ds4_gpu_tensor_free(g_rocm_tp.big_in);
        g_rocm_tp.big_out = tx; g_rocm_tp.big_in = rx;
    }
    rocm_tp_copy<<<256, 256>>>(g_rocm_tp.device, (float *)g_rocm_tp.big_out->ptr, (const float *)out_t->ptr, bytes / 4);
    if (!rocm_tp_enqueue({0, bytes, layer, rows, 2, g_rocm_tp.big_out->host_ptr, g_rocm_tp.big_in->host_ptr}, rows * 5120u, true)) return 0;
    g_rocm_tp.pending_big_in = in_t->ptr;
    g_rocm_tp.pending_big_bytes = bytes;
    return 1;
}
extern "C" int ds4_gpu_tp_big_gate_join(uint32_t layer, uint32_t rows,
        ds4_gpu_tensor *in_t, uint64_t bytes) {
    const uint64_t seq = g_rocm_tp.pending;
    if (!g_rocm_tp.active || !seq || !g_rocm_tp.pending_deferred ||
        !in_t || !in_t->ptr || in_t->ptr != g_rocm_tp.pending_big_in ||
        bytes != g_rocm_tp.pending_big_bytes || bytes > in_t->bytes ||
        rows > UINT32_MAX / 5120u || rows * 5120u != g_rocm_tp.pending_count ||
        ds4_gpu_tp_failed()) return rocm_tp_fail();
    const unsigned slot = (unsigned)((seq - 1) % ROCM_TP_QUEUE);
    const rocm_tp_job &job = g_rocm_tp.jobs[slot];
    if (job.seq != seq || job.kind != 2 || job.layer != layer || job.arg != rows ||
        job.bytes != bytes) return rocm_tp_fail();
    rocm_tp_wait<<<1, 1>>>(g_rocm_tp.device, slot, seq, g_rocm_tp.timeout_ticks);
    rocm_tp_copy<<<256, 256>>>(g_rocm_tp.device, (float *)in_t->ptr,
                            (const float *)g_rocm_tp.big_in->ptr, bytes / 4);
    if (!cuda_ok(cudaGetLastError(), "TP deferred receive")) return rocm_tp_fail();
    g_rocm_tp.pending_deferred = false;
    g_rocm_tp.pending_big_in = nullptr;
    g_rocm_tp.pending_big_bytes = 0;
    return 1;
}
extern "C" int ds4_gpu_tp_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a,
                                     const ds4_gpu_tensor *b, uint32_t n) {
    const uint64_t seq = g_rocm_tp.pending;
    const uint64_t bytes = (uint64_t)n * 4;
    if (!g_rocm_tp.active || !seq || g_rocm_tp.pending_deferred || !out || !a || !b || n != g_rocm_tp.pending_count ||
        bytes > out->bytes || bytes > a->bytes || bytes > b->bytes || ds4_gpu_tp_failed()) return rocm_tp_fail();
    const unsigned slot = (unsigned)((seq - 1) % ROCM_TP_QUEUE);
    /* Avoid coherent guard loads from idle workgroups on scalar payloads.
     * The grid-stride loop preserves full coverage for larger batches. */
    const uint32_t blocks = n / 256u + (n % 256u != 0u);
    rocm_tp_add<<<blocks < 256u ? blocks : 256u, 256>>>(g_rocm_tp.device, slot, seq, (float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    rocm_tp_release<<<1, 1>>>(g_rocm_tp.device, slot, seq);
    g_rocm_tp.pending = 0;
    return cuda_ok(cudaGetLastError(), "TP guarded reduction") || rocm_tp_fail();
}

extern "C" int ds4_gpu_tp_big_gate_overlap_supported(void) {
    return ds4_rocm_is_gfx1151();
}
extern "C" void ds4_gpu_tp_big_gate_abort(void) {
    rocm_tp_fail();
    (void)ds4_gpu_synchronize();
}
