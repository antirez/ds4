/* Two-machine tensor parallelism for the ROCm backend.  Mirrors the Metal
 * gate machinery (ds4_metal.m) on top of the eager default-stream execution
 * model of this backend:
 *
 *  - The TP transport slab lives in host-mapped pinned memory
 *    (ds4_gpu_tensor_alloc_shared), so GPU kernels and the CPU transport
 *    thread access the same bytes directly (coherent on Strix Halo APUs
 *    with HSA_XNACK=1).
 *
 *  - ds4_gpu_tp_gate_encode() appends two tiny kernels to the in-order
 *    default stream around each exchange: a flag-set kernel publishes the
 *    GPU arrival (ordered after the partial-producing kernels and fenced
 *    with __threadfence_system), and a spin kernel parks the stream until
 *    the CPU service thread publishes the release word in the slab's
 *    in-flags region after the TCP/RDMA exchange landed.  This reproduces
 *    Metal's shared-event arrival/release pair without command buffers.
 *
 *  - Row gates use per-slot arrival/release words.  Batch gates (speculative
 *    verify) and big gates (prefill row swaps) share the FFN slot word and
 *    are separated by tag bits; arrivals and releases are matched per tag
 *    with a monotonically increasing 30-bit sequence, so multiple in-flight
 *    big-gate kicks resolve in order.  Big gates bounce payloads through a
 *    pinned staging pair because the exchanged graph tensors are ordinary
 *    device allocations: a device kernel stages out_t into mapped memory,
 *    and the release spin kernel itself copies the staged peer data into
 *    in_t before the consumer kernels run.
 *
 *  - The service thread is the only CPU that talks to the transport (via
 *    the callbacks registered by ds4.c), preserving ds4_tp.c's single
 *    writer per socket / QP. */

#include <pthread.h>

/* ds4.c's tensor type enum is not visible from the backend translation
 * unit; match the numeric convention already used by the MoE launcher
 * (Q4_K = 12u there). */
#define DS4_ROCM_TP_TENSOR_Q8_0 8u

#define DS4_ROCM_TP_QUEUE 512u
#define DS4_ROCM_TP_TIMEOUT_SEC 300.0

/* Arrival (u32, gpu_flags[slot]) and release (u64, in_flags[slot]) tagging.
 * Row gates use slot = layer*2 + gate, batch/big gates slot = layer*2 + 1. */
#define ROCM_TP_TAG_ROW 0x00000000u
#define ROCM_TP_TAG_BATCH 0x80000000u
#define ROCM_TP_TAG_BIG 0x40000000u
#define ROCM_TP_TAG_MASK 0xC0000000u
#define ROCM_TP_SEQ_MASK 0x3FFFFFFFu

static ds4_gpu_tp_exchange_fn g_tp_exchange_fn;
static ds4_gpu_tp_batch_exchange_fn g_tp_batch_exchange_fn;
static ds4_gpu_tp_big_exchange_fn g_tp_big_exchange_fn;
static void *g_tp_exchange_ud;

static pthread_t g_tp_thread;
static int g_tp_thread_running;
static pthread_mutex_t g_tp_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_tp_cond_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_tp_cond_space = PTHREAD_COND_INITIALIZER;

typedef struct {
    uint32_t layer;
    uint32_t gate;   /* 0 attn / 1 ffn for row gates; 1 for batch/big */
    uint32_t rows;   /* verify-block rows for batch gates */
    uint64_t seq;
    uint32_t big;    /* prefill big gate */
    uint64_t big_bytes;
} ds4_rocm_tp_req;

static ds4_rocm_tp_req g_tp_queue[DS4_ROCM_TP_QUEUE];
static uint32_t g_tp_qhead;
static uint32_t g_tp_qcount;

static uint64_t g_tp_seq;        /* row gate sequence */
static uint64_t g_tp_batch_seq;  /* batch + big gate sequence (shared, like Metal) */

static uint32_t g_tp_split_rank = 1;
static uint32_t g_tp_split_world = 1;
static int g_tp_shard_suspended;
static volatile int g_tp_failed;

static void *g_tp_slab_dev;    /* device alias of the slab base */
static void *g_tp_slab_host;  /* host alias of the slab base */
static uint64_t g_tp_gpu_flags_off;
static uint64_t g_tp_in_flags_off;

/* Big-gate staging: pinned, host-mapped pair plus a global monotonic
 * release word (covers all earlier kicks of the shared batch sequence). */
static float *g_tp_stage_out;
static float *g_tp_stage_in;
static void *g_tp_stage_out_dev;
static void *g_tp_stage_in_dev;
static uint64_t g_tp_stage_bytes;
static volatile unsigned long long *g_tp_big_release_host;
static void *g_tp_big_release_dev;

static double ds4_rocm_tp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* Gate kernels.                                                        */
/* ------------------------------------------------------------------ */

__global__ static void tp_flag_set_kernel(volatile unsigned int *flag,
                                          unsigned int value) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        /* Make every earlier kernel's slab writes CPU-visible before the
         * arrival word publishes. */
        __threadfence_system();
        *flag = value;
    }
}

__global__ static void tp_gate_wait_kernel(
        volatile const unsigned long long *release,
        unsigned long long expected) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        const unsigned long long tag = expected & 0xC000000000000000ull;
        const unsigned long long seq = expected & 0x3FFFFFFFFFFFFFFFull;
        for (;;) {
            const unsigned long long v = *release;
            if ((v & 0xC000000000000000ull) == tag &&
                (v & 0x3FFFFFFFFFFFFFFFull) >= seq) break;
        }
        __threadfence_system();
    }
}

__global__ static void tp_big_gate_wait_kernel(
        volatile const unsigned long long *release,
        unsigned long long expected_seq,
        const float *src,
        float *dst,
        unsigned long long n) {
    if (threadIdx.x == 0u) {
        for (;;) {
            if (*release >= expected_seq) break;
        }
        __threadfence_system();
    }
    __syncthreads();
    for (unsigned long long i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ static void tp_stage_copy_kernel(const float *src, float *dst,
                                            unsigned long long n) {
    const unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
    /* The arrival flag kernel that follows in stream order fences the
     * staged payload before publishing; nothing to do here. */
}

/* ------------------------------------------------------------------ */
/* Service thread.                                                      */
/* ------------------------------------------------------------------ */

static void ds4_rocm_tp_store_release(uint32_t slot, unsigned long long value) {
    volatile unsigned long long *word =
        (volatile unsigned long long *)((char *)g_tp_slab_host +
                                        g_tp_in_flags_off +
                                        (uint64_t)slot * 8ull);
    __atomic_store_n(word, value, __ATOMIC_RELEASE);
}

static int ds4_rocm_tp_wait_arrival(uint32_t slot, unsigned int expected) {
    volatile unsigned int *flag =
        (volatile unsigned int *)((char *)g_tp_slab_host +
                                  g_tp_gpu_flags_off +
                                  (uint64_t)slot * 4ull);
    const unsigned int tag = expected & ROCM_TP_TAG_MASK;
    const unsigned int seq = expected & ROCM_TP_SEQ_MASK;
    const double deadline = ds4_rocm_tp_now_sec() + DS4_ROCM_TP_TIMEOUT_SEC;
    uint64_t spins = 0;
    for (;;) {
        const unsigned int v = __atomic_load_n(flag, __ATOMIC_ACQUIRE);
        if ((v & ROCM_TP_TAG_MASK) == tag && (v & ROCM_TP_SEQ_MASK) >= seq) {
            return 1;
        }
        /* The flag normally publishes within microseconds of the service
         * thread picking up the request (GPU is already executing).  Busy
         * spin for the first ~5ms, then sleep-poll to keep the CPU quiet
         * while the peer catches up. */
        spins++;
        if (spins < 20000u) {
            __builtin_ia32_pause();
        } else {
            usleep(50);
        }
        if (ds4_rocm_tp_now_sec() > deadline) return 0;
    }
}

static void *ds4_rocm_tp_service_thread(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_tp_mutex);
        while (g_tp_qcount == 0u && g_tp_thread_running) {
            pthread_cond_wait(&g_tp_cond_work, &g_tp_mutex);
        }
        if (g_tp_qcount == 0u && !g_tp_thread_running) {
            pthread_mutex_unlock(&g_tp_mutex);
            break;
        }
        const ds4_rocm_tp_req req = g_tp_queue[g_tp_qhead];
        g_tp_qhead = (g_tp_qhead + 1u) % DS4_ROCM_TP_QUEUE;
        g_tp_qcount--;
        pthread_cond_signal(&g_tp_cond_space);
        pthread_mutex_unlock(&g_tp_mutex);

        const uint32_t slot = req.layer * 2u + (req.big ? 1u : req.gate);
        unsigned int tag = ROCM_TP_TAG_ROW;
        if (req.big) tag = ROCM_TP_TAG_BIG;
        else if (req.rows != 0u) tag = ROCM_TP_TAG_BATCH;
        const unsigned int arrival = tag | (unsigned int)(req.seq & ROCM_TP_SEQ_MASK);

        int ok = ds4_rocm_tp_wait_arrival(slot, arrival);
        if (ok) {
            if (req.big) {
                if (g_tp_big_exchange_fn) {
                    ok = g_tp_big_exchange_fn(g_tp_exchange_ud, req.layer,
                                              req.seq, g_tp_stage_out,
                                              g_tp_stage_in, req.big_bytes);
                } else {
                    ok = 0;
                }
                /* Publish the big release last so the staged peer payload is
                 * visible before any waiter observes the counter. */
                __atomic_store_n(g_tp_big_release_host,
                                 (unsigned long long)(req.seq & ROCM_TP_SEQ_MASK),
                                 __ATOMIC_RELEASE);
            } else if (req.rows != 0u) {
                ok = g_tp_batch_exchange_fn ? g_tp_batch_exchange_fn(
                        g_tp_exchange_ud, req.layer, req.rows, req.seq) : 0;
                ds4_rocm_tp_store_release(
                        slot, 0x8000000000000000ull |
                              (req.seq & ROCM_TP_SEQ_MASK));
            } else {
                ok = g_tp_exchange_fn ? g_tp_exchange_fn(
                        g_tp_exchange_ud, req.layer, req.gate, req.seq) : 0;
                ds4_rocm_tp_store_release(
                        slot, req.seq & ROCM_TP_SEQ_MASK);
            }
        }
        if (!ok) {
            g_tp_failed = 1;
            /* Unblock the GPU regardless; the garbage combine aborts at the
             * next gate encode through the failure flag. */
            if (req.big) {
                __atomic_store_n(g_tp_big_release_host,
                                 (unsigned long long)(req.seq & ROCM_TP_SEQ_MASK),
                                 __ATOMIC_RELEASE);
            } else {
                ds4_rocm_tp_store_release(
                        slot, (req.rows != 0u ? 0x8000000000000000ull : 0ull) |
                              (req.seq & ROCM_TP_SEQ_MASK));
            }
        }
    }
    return NULL;
}

static int ds4_rocm_tp_enqueue(const ds4_rocm_tp_req *req) {
    pthread_mutex_lock(&g_tp_mutex);
    while (g_tp_qcount == DS4_ROCM_TP_QUEUE && g_tp_thread_running) {
        pthread_cond_wait(&g_tp_cond_space, &g_tp_mutex);
    }
    if (!g_tp_thread_running) {
        pthread_mutex_unlock(&g_tp_mutex);
        return 0;
    }
    const uint32_t tail = (g_tp_qhead + g_tp_qcount) % DS4_ROCM_TP_QUEUE;
    g_tp_queue[tail] = *req;
    g_tp_qcount++;
    pthread_cond_signal(&g_tp_cond_work);
    pthread_mutex_unlock(&g_tp_mutex);
    return 1;
}

/* ------------------------------------------------------------------ */
/* Init / shutdown / state.                                             */
/* ------------------------------------------------------------------ */

extern "C" void ds4_gpu_tp_set_slab_layout(uint64_t in_flags_off) {
    g_tp_in_flags_off = in_flags_off;
}

extern "C" int ds4_gpu_tp_init(uint32_t rank,
                    ds4_gpu_tensor *slab,
                    uint64_t gpu_flags_off,
                    ds4_gpu_tp_exchange_fn fn,
                    void *ud) {
    if (!slab || !slab->host_alias || !slab->ptr) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP slab is not host-mapped\n");
        return 0;
    }
    g_tp_slab_dev = slab->ptr;
    g_tp_slab_host = slab->host_alias;
    g_tp_gpu_flags_off = gpu_flags_off;
    g_tp_split_rank = rank;
    g_tp_split_world = 2u;
    g_tp_shard_suspended = 0;
    g_tp_failed = 0;
    g_tp_seq = 0;
    g_tp_batch_seq = 0;
    g_tp_qhead = 0;
    g_tp_qcount = 0;
    g_tp_exchange_fn = fn;
    g_tp_batch_exchange_fn = NULL;
    g_tp_big_exchange_fn = NULL;
    g_tp_exchange_ud = ud;

    /* Pinned staging pair for big gates + the global big release word. */
    if (!g_tp_stage_out) {
        void *host_out = NULL;
        void *host_in = NULL;
        void *big_rel = NULL;
        if (cudaHostAlloc(&host_out, 4u * 1024u * 1024u,
                          cudaHostAllocMapped) != cudaSuccess ||
            cudaHostAlloc(&host_in, 4u * 1024u * 1024u,
                          cudaHostAllocMapped) != cudaSuccess ||
            cudaHostAlloc(&big_rel, 8u, cudaHostAllocMapped) != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "TP staging alloc failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            return 0;
        }
        memset(host_out, 0, 4u * 1024u * 1024u);
        memset(host_in, 0, 4u * 1024u * 1024u);
        memset(big_rel, 0, 8u);
        void *dev_out = host_out;
        void *dev_in = host_in;
        void *dev_rel = big_rel;
        if (cudaHostGetDevicePointer(&dev_out, host_out, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(&dev_in, host_in, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(&dev_rel, big_rel, 0) != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "TP staging map failed: %s\n",
                    cudaGetErrorString(cudaGetLastError()));
            return 0;
        }
        g_tp_stage_out = (float *)host_out;
        g_tp_stage_in = (float *)host_in;
        g_tp_stage_out_dev = dev_out;
        g_tp_stage_in_dev = dev_in;
        g_tp_stage_bytes = 4u * 1024u * 1024u;
        g_tp_big_release_host = (volatile unsigned long long *)big_rel;
        g_tp_big_release_dev = dev_rel;
    }

    g_tp_thread_running = 1;
    if (pthread_create(&g_tp_thread, NULL, ds4_rocm_tp_service_thread, NULL) != 0) {
        g_tp_thread_running = 0;
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP service thread create failed\n");
        return 0;
    }
    return 1;
}

extern "C" void ds4_gpu_tp_shutdown(void) {
    if (g_tp_thread_running) {
        pthread_mutex_lock(&g_tp_mutex);
        g_tp_thread_running = 0;
        pthread_cond_broadcast(&g_tp_cond_work);
        pthread_cond_broadcast(&g_tp_cond_space);
        pthread_mutex_unlock(&g_tp_mutex);
        pthread_join(g_tp_thread, NULL);
    }
    g_tp_split_rank = 1;
    g_tp_split_world = 1;
    g_tp_exchange_fn = NULL;
    g_tp_batch_exchange_fn = NULL;
    g_tp_big_exchange_fn = NULL;
    g_tp_exchange_ud = NULL;
    g_tp_slab_dev = NULL;
    g_tp_slab_host = NULL;
    if (g_tp_stage_out) {
        (void)cudaFreeHost(g_tp_stage_out);
        (void)cudaFreeHost(g_tp_stage_in);
        (void)cudaFreeHost((void *)g_tp_big_release_host);
        g_tp_stage_out = NULL;
        g_tp_stage_in = NULL;
        g_tp_stage_out_dev = NULL;
        g_tp_stage_in_dev = NULL;
        g_tp_big_release_host = NULL;
        g_tp_big_release_dev = NULL;
        g_tp_stage_bytes = 0;
    }
}

extern "C" void ds4_gpu_tp_set_session_batch_mode(int enabled) {
    /* Flag gates with per-tag monotonic sequences stay correct across
     * shared slab slots on the eager stream; nothing to switch. */
    (void)enabled;
}

extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    g_tp_shard_suspended = suspend;
}

extern "C" void ds4_gpu_tp_keepalive_pause(int paused) {
    /* DVFS keep-alive is a Metal-only concern. */
    (void)paused;
}

extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled) {
    /* Head splitting is expressed through graph-level arguments on this
     * backend (compact q/head tensors plus shifted sinks). */
    (void)enabled;
}

extern "C" int ds4_gpu_tp_failed(void) {
    return g_tp_failed;
}

extern "C" int ds4_gpu_tp_world_is_two(void) {
    return g_tp_split_world == 2u && !g_tp_shard_suspended;
}

extern "C" void ds4_gpu_tp_expert_range(uint32_t n_total_expert,
                             uint32_t *first_expert,
                             uint32_t *n_bind_expert) {
    if (!first_expert || !n_bind_expert) return;
    if (g_tp_split_world < 2u || g_tp_shard_suspended || n_total_expert < 2u) {
        *first_expert = 0;
        *n_bind_expert = n_total_expert;
        return;
    }
    const uint32_t half = n_total_expert / 2u;
    *first_expert = g_tp_split_rank * half;
    *n_bind_expert = g_tp_split_rank + 1u == g_tp_split_world
        ? n_total_expert - *first_expert : half;
}

/* ------------------------------------------------------------------ */
/* Gate encoders.                                                       */
/* ------------------------------------------------------------------ */

static volatile unsigned int *ds4_rocm_tp_gpu_flag_dev(uint32_t slot) {
    return (volatile unsigned int *)((char *)g_tp_slab_dev +
                                     g_tp_gpu_flags_off +
                                     (uint64_t)slot * 4ull);
}

static volatile unsigned long long *ds4_rocm_tp_release_dev(uint32_t slot) {
    return (volatile unsigned long long *)((char *)g_tp_slab_dev +
                                           g_tp_in_flags_off +
                                           (uint64_t)slot * 8ull);
}

extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    if (!g_tp_thread_running || g_tp_failed) return 0;
    const uint64_t seq = ++g_tp_seq;
    const uint32_t slot = layer * 2u + gate;
    tp_flag_set_kernel<<<1, 1>>>(
            ds4_rocm_tp_gpu_flag_dev(slot),
            ROCM_TP_TAG_ROW | (unsigned int)(seq & ROCM_TP_SEQ_MASK));
    ds4_rocm_tp_req req;
    memset(&req, 0, sizeof(req));
    req.layer = layer;
    req.gate = gate;
    req.seq = seq;
    if (!ds4_rocm_tp_enqueue(&req)) return 0;
    tp_gate_wait_kernel<<<1, 1>>>(
            ds4_rocm_tp_release_dev(slot),
            (unsigned long long)(seq & ROCM_TP_SEQ_MASK));
    return cuda_ok(cudaGetLastError(), "tp row gate encode");
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    if (!g_tp_thread_running || g_tp_failed || rows == 0u) return 0;
    const uint64_t seq = ++g_tp_batch_seq;
    const uint32_t slot = layer * 2u + 1u;
    tp_flag_set_kernel<<<1, 1>>>(
            ds4_rocm_tp_gpu_flag_dev(slot),
            ROCM_TP_TAG_BATCH | (unsigned int)(seq & ROCM_TP_SEQ_MASK));
    ds4_rocm_tp_req req;
    memset(&req, 0, sizeof(req));
    req.layer = layer;
    req.gate = 1u;
    req.rows = rows;
    req.seq = seq;
    if (!ds4_rocm_tp_enqueue(&req)) return 0;
    tp_gate_wait_kernel<<<1, 1>>>(
            ds4_rocm_tp_release_dev(slot),
            0x8000000000000000ull | (unsigned long long)(seq & ROCM_TP_SEQ_MASK));
    return cuda_ok(cudaGetLastError(), "tp batch gate encode");
}

static int ds4_rocm_tp_stage_grow(uint64_t bytes) {
    if (bytes <= g_tp_stage_bytes) return 1;
    uint64_t want = g_tp_stage_bytes ? g_tp_stage_bytes : (4u << 20);
    while (want < bytes) want *= 2u;
    void *host_out = NULL;
    void *host_in = NULL;
    if (cudaHostAlloc(&host_out, (size_t)want, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc(&host_in, (size_t)want, cudaHostAllocMapped) != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "TP staging grow to %llu bytes failed: %s\n",
                (unsigned long long)want,
                cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    memset(host_out, 0, (size_t)want);
    memset(host_in, 0, (size_t)want);
    void *dev_out = host_out;
    void *dev_in = host_in;
    if (cudaHostGetDevicePointer(&dev_out, host_out, 0) != cudaSuccess ||
        cudaHostGetDevicePointer(&dev_in, host_in, 0) != cudaSuccess) {
        (void)cudaFreeHost(host_out);
        (void)cudaFreeHost(host_in);
        return 0;
    }
    (void)cudaFreeHost(g_tp_stage_out);
    (void)cudaFreeHost(g_tp_stage_in);
    g_tp_stage_out = (float *)host_out;
    g_tp_stage_in = (float *)host_in;
    g_tp_stage_out_dev = dev_out;
    g_tp_stage_in_dev = dev_in;
    g_tp_stage_bytes = want;
    return 1;
}

extern "C" uint64_t ds4_gpu_tp_big_gate_kick(uint32_t layer, uint32_t rows,
                                  const ds4_gpu_tensor *out_t,
                                  ds4_gpu_tensor *in_t,
                                  uint64_t bytes) {
    if (!g_tp_thread_running || g_tp_failed || !out_t || !in_t || bytes == 0u) {
        return 0;
    }
    if ((bytes & 3ull) != 0ull || !ds4_rocm_tp_stage_grow(bytes)) return 0;
    const uint64_t seq = ++g_tp_batch_seq;
    const uint64_t n = bytes / 4ull;
    /* Stage the payload into mapped memory, publish arrival, queue the
     * exchange, and park the stream on the global big-release word; the
     * spin kernel itself copies the staged peer data into in_t so every
     * kernel enqueued after the kick observes the swapped rows. */
    tp_stage_copy_kernel<<<(unsigned)((n + 255ull) / 256ull), 256>>>(
            (const float *)out_t->ptr, (float *)g_tp_stage_out_dev, n);
    tp_flag_set_kernel<<<1, 1>>>(
            ds4_rocm_tp_gpu_flag_dev(layer * 2u + 1u),
            ROCM_TP_TAG_BIG | (unsigned int)(seq & ROCM_TP_SEQ_MASK));
    ds4_rocm_tp_req req;
    memset(&req, 0, sizeof(req));
    req.layer = layer;
    req.gate = 1u;
    req.rows = rows;
    req.seq = seq;
    req.big = 1;
    req.big_bytes = bytes;
    if (!ds4_rocm_tp_enqueue(&req)) return 0;
    tp_big_gate_wait_kernel<<<1, 256>>>(
            (volatile unsigned long long *)g_tp_big_release_dev,
            (unsigned long long)(seq & ROCM_TP_SEQ_MASK),
            (const float *)g_tp_stage_in_dev,
            (float *)in_t->ptr,
            n);
    if (!cuda_ok(cudaGetLastError(), "tp big gate kick")) return 0;
    return seq;
}

extern "C" int ds4_gpu_tp_big_gate_wait(uint64_t seq) {
    /* The release wait (and the staged copy into in_t) is already encoded
     * at kick time; kicks are processed strictly in sequence order, so
     * waiting on the last kick covers all earlier ones. */
    return seq != 0u && !g_tp_failed;
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                               const ds4_gpu_tensor *out_t,
                               ds4_gpu_tensor *in_t,
                               uint64_t bytes) {
    return ds4_gpu_tp_big_gate_kick(layer, rows, out_t, in_t, bytes) != 0;
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) {
    g_tp_batch_exchange_fn = fn;
}

extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) {
    g_tp_big_exchange_fn = fn;
}

/* ------------------------------------------------------------------ */
/* K-sliced Q8_0 matvec (TP shared-expert down / attention expand).      */
/* ------------------------------------------------------------------ */

__global__ static void matmul_q8_0_kslice_warp8_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t k_blocks,
        uint64_t row_bytes,
        uint64_t out_dim) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * row_bytes;
    float acc = 0.0f;
    for (uint64_t b = 0; b < k_blocks; b++) {
        const unsigned char *blk = wr + b * 34u;
        const float d = q8_0_scale_broadcast_w32(blk);
        const int8_t q = ((const int8_t *)(blk + 2u))[lane];
        acc += d * (float)q * x[b * 32u + lane];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) out[row] = acc;
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (!out || !x || !model_map || full_in_dim == 0u || k_cnt == 0u ||
        (k_off & 31ull) != 0ull || (k_cnt & 31ull) != 0ull ||
        k_off + k_cnt > full_in_dim ||
        out->bytes < out_dim * sizeof(float) ||
        x->bytes < (x_elem_off + k_cnt) * sizeof(float)) {
        return 0;
    }
    const uint64_t full_blocks = full_in_dim / 32u;
    const uint64_t k_blocks = k_cnt / 32u;
    const uint64_t row_bytes = full_blocks * 34u;
    if (weight_offset > model_size ||
        (uint64_t)out_dim * row_bytes > model_size - weight_offset) {
        return 0;
    }
    const unsigned char *w = (const unsigned char *)cuda_model_range_ptr(
            model_map, weight_offset, out_dim * row_bytes, "q8_0_kslice");
    if (!w) return 0;
    w += (k_off / 32u) * 34u;
    matmul_q8_0_kslice_warp8_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr,
            w,
            (const float *)x->ptr + x_elem_off,
            k_blocks,
            row_bytes,
            out_dim);
    return cuda_ok(cudaGetLastError(), "matmul q8_0 kslice launch");
}

extern "C" int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t full_in_dim,
        uint64_t k_off, uint64_t k_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t x_elem_off) {
    if (weight_type == DS4_ROCM_TP_TENSOR_Q8_0) {
        return ds4_gpu_matmul_q8_0_kslice_tensor(out, model_map, model_size,
                                                 weight_offset, full_in_dim,
                                                 k_off, k_cnt, out_dim, x,
                                                 x_elem_off);
    }
    fprintf(stderr, DS4_GPU_LOG_PREFIX
            "tensor-parallel k-slice for expert type %u is not implemented\n",
            weight_type);
    return 0;
}

/* ------------------------------------------------------------------ */
/* TP attention output projection (single token).                       */
/* ------------------------------------------------------------------ */

extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups_total,
        uint32_t group0, uint32_t group_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *heads) {
    if (!out || !low || !heads || !model_map || group_cnt == 0u ||
        group0 + group_cnt > n_groups_total ||
        (group_dim & 31ull) != 0ull ||
        ((uint64_t)group_cnt * rank & 31ull) != 0ull) {
        return 0;
    }
    /* (a) low projection for the owned groups: heads carries the owned
     * half compactly at its base, so the generic grouped-Q8 entry point
     * works with n_groups = group_cnt and the weight window shifted to
     * this rank's groups. */
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    const uint64_t a_group_bytes = rank * blocks_a * 34u;
    if (out_a_offset > model_size ||
        (uint64_t)group_cnt * a_group_bytes > model_size - out_a_offset) {
        return 0;
    }
    if (!ds4_gpu_attention_output_low_q8_tensor(
                low, model_map, model_size,
                out_a_offset + (uint64_t)group0 * a_group_bytes,
                group_dim, rank, group_cnt, heads)) {
        return 0;
    }
    /* (b) expand projection: full width, K sliced to the owned groups. */
    return ds4_gpu_matmul_q8_0_kslice_tensor(
            out, model_map, model_size, out_b_offset,
            (uint64_t)n_groups_total * rank,
            (uint64_t)group0 * rank, (uint64_t)group_cnt * rank,
            out_dim, low, 0u);
}

/* ------------------------------------------------------------------ */
/* Fused TP combine into the HC expand (mirror of the CUDA backend).    */
/* ------------------------------------------------------------------ */

extern "C" int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
        uint32_t n_embd, uint32_t n_hc) {
    uint64_t n_tokens64 = 0, flat_bytes = 0, hc_bytes = 0, post_bytes = 0;
    uint64_t comb_bytes = 0, comb_stride = 0;
    if (!out_hc || !block_out || !block_add || !residual_hc || !post ||
        !comb || n_hc == 0u ||
        !cuda_hc_hc_token_count(out_hc, n_embd, n_hc, &n_tokens64) ||
        !cuda_u64_mul3_checked(n_tokens64, n_embd, sizeof(float), &flat_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, (uint64_t)n_hc * n_embd, sizeof(float), &hc_bytes) ||
        !cuda_u64_mul3_checked(n_tokens64, n_hc, sizeof(float), &post_bytes) ||
        !cuda_u64_mul_checked(n_hc, n_hc, &comb_stride) || comb_stride > UINT32_MAX ||
        !cuda_u64_mul3_checked(n_tokens64, comb_stride, sizeof(float), &comb_bytes) ||
        block_out->bytes < flat_bytes || block_add->bytes < flat_bytes ||
        residual_hc->bytes < hc_bytes || post->bytes < post_bytes ||
        comb->bytes < comb_bytes) {
        return 0;
    }
    uint32_t n_tokens = (uint32_t)n_tokens64;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>(
            (float *)out_hc->ptr,
            (const float *)block_out->ptr,
            (const float *)block_add->ptr,
            (const float *)residual_hc->ptr,
            (const float *)post->ptr,
            (const float *)comb->ptr,
            n_embd, n_hc, n_tokens, n_hc, (uint32_t)comb_stride, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add launch");
}

/* ------------------------------------------------------------------ */
/* TP prefill attention row ranges.                                     */
/* ------------------------------------------------------------------ */

extern "C" int ds4_gpu_attention_prefill_raw_heads_range_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t q_row0, uint32_t n_q,
        uint32_t n_kv, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_q * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_q * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_kv * head_dim * sizeof(float) ||
        n_q == 0u || window > 256u) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "prefill raw range validation failed "
                "(q_row0=%u n_q=%u n_kv=%u window=%u n_head=%u head_dim=%u)\n",
                q_row0, n_q, n_kv, window, n_head, head_dim);
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (n_q > 1u && head_dim == 512u &&
        !g_quality_mode &&
        ((window != 0u ? window : n_kv) <= 768u)) {
        dim3 grid(n_q, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>(
                (float *)heads->ptr,
                sinks,
                (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                (const float *)raw_kv->ptr,
                n_kv,
                0u,
                window,
                1u,
                q_row0,
                n_head,
                head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw range window launch");
    }
    dim3 grid(n_q, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128>>>(
            (float *)heads->ptr,
            sinks,
            (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_kv, window, q_row0, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention_prefill_raw range launch");
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, uint32_t q_row0, uint32_t n_q,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !comp_kv || !model_map ||
        sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_q * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_q * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim *
            (uint64_t)(comp_kv_f16 ? sizeof(__half) : sizeof(float)) ||
        n_q == 0u) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "prefill mixed range validation failed "
                "(comp_f16=%u q_row0=%u n_q=%u n_tokens=%u n_comp=%u)\n",
                comp_kv_f16, q_row0, n_q, n_tokens, n_comp);
        return 0;
    }
    if (comp_kv_f16) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "prefill mixed range: f16 compressed KV "
                "not supported on the TP row-split path\n");
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    dim3 grid(n_q, (n_head + 7u) / 8u, 1);
    attention_static_mixed_heads8_online_kernel<<<grid, 256>>>(
            (float *)heads->ptr,
            sinks,
            (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            (const float *)comp_kv->ptr,
            n_tokens,
            n_comp,
            window,
            ratio,
            q_row0,
            n_head,
            head_dim);
    return cuda_ok(cudaGetLastError(), "attention static mixed range launch");
}
