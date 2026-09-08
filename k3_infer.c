#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "k3_infer.h"

#include "ds4_gpu.h"
#include "k3_metal.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <mach/mach.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

enum {
    K3_MLA_CACHE_WIDTH = K3_KV_LORA + K3_QK_ROPE,
    K3_MLA_Q_WIDTH = K3_N_HEAD * K3_QK_HEAD,
    K3_MLA_LATENT_WIDTH = K3_N_HEAD * K3_KV_LORA,
    K3_EXPERT_PREFETCH_THREADS_MAX = 16,
    K3_STATIC_IO_THREADS_MAX = 16,
    K3_STREAMED_EXPERT_SLOTS = 64,
    K3_SPECULATIVE_BANKS = 2,
    K3_Q8_MAX_MATRICES = 16,
};

static const float K3_RMS_EPS = 1e-5f;
static const float K3_GATE_LOWER_BOUND = -5.0f;
static const uint64_t K3_LAYER_MAX_TENSOR_BYTES =
    512ull * 1024ull * 1024ull;
static const uint64_t K3_PREAD_CHUNK_BYTES =
    64ull * 1024ull * 1024ull;
static const uint64_t K3_IO_SPLIT_ALIGNMENT = 16ull * 1024ull;

typedef struct {
    ds4_gpu_tensor *conv;
    ds4_gpu_tensor *recurrent;
} k3_kda_state;

typedef struct {
    ds4_gpu_tensor *cache;
} k3_mla_state;

typedef struct {
    ds4_gpu_tensor *embedding_rows;
    ds4_gpu_tensor *prefix;
    ds4_gpu_tensor *delta;
    ds4_gpu_tensor *blocks;
    ds4_gpu_tensor *attn_in;
    ds4_gpu_tensor *attn_heads;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *ffn_in;
    ds4_gpu_tensor *final_hidden;
    ds4_gpu_tensor *final_norm;
    ds4_gpu_tensor *logits;

    ds4_gpu_tensor *q;
    ds4_gpu_tensor *k;
    ds4_gpu_tensor *v;
    ds4_gpu_tensor *raw_gate;
    ds4_gpu_tensor *beta;
    ds4_gpu_tensor *output_gate;
    ds4_gpu_tensor *low_rank;
    ds4_gpu_tensor *low_rank_norm;
    ds4_gpu_tensor *kv;
    ds4_gpu_tensor *q_latent;
    ds4_gpu_tensor *partial_values;
    ds4_gpu_tensor *partial_stats;

    ds4_gpu_tensor *dense_gate;
    ds4_gpu_tensor *dense_up;
    ds4_gpu_tensor *dense_mid;

    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *selected_ids;
    ds4_gpu_tensor *selected_weights;
    ds4_gpu_tensor *predicted_router_logits;
    ds4_gpu_tensor *predicted_ids;
    ds4_gpu_tensor *predicted_weights;
    ds4_gpu_tensor *routed_latent;
    ds4_gpu_tensor *routed_sum;
    ds4_gpu_tensor *routed_norm;
    ds4_gpu_tensor *routed_out;
    ds4_gpu_tensor *shared_gate;
    ds4_gpu_tensor *shared_up;
    ds4_gpu_tensor *shared_mid;
    ds4_gpu_tensor *shared_out;
    ds4_gpu_tensor *shared_input_values;
    ds4_gpu_tensor *shared_input_scales;
    ds4_gpu_tensor *shared_mid_values;
    ds4_gpu_tensor *shared_mid_scales;

    ds4_gpu_tensor *group_row_ids;
    ds4_gpu_tensor *group_weights;
    ds4_gpu_tensor *group_input;
    ds4_gpu_tensor *group_input_values;
    ds4_gpu_tensor *group_input_scales;
    ds4_gpu_tensor *expert_gate;
    ds4_gpu_tensor *expert_up;
    ds4_gpu_tensor *expert_mid;
    ds4_gpu_tensor *expert_mid_values;
    ds4_gpu_tensor *expert_mid_scales;
    ds4_gpu_tensor *expert_out;
} k3_scratch;

typedef struct {
    ds4_gpu_tensor *weights;
    uint64_t w1_packed;
    uint64_t w1_scale;
    uint64_t w3_packed;
    uint64_t w3_scale;
    uint64_t w2_packed;
    uint64_t w2_scale;
    uint64_t bytes;
} k3_streamed_expert;

typedef struct k3_q8_layer_cache k3_q8_layer_cache;

typedef struct {
    const void *base;
    uint64_t size;
    uint64_t bias;
    const k3_q8_layer_cache *q8;
} k3_weight_source;

typedef struct {
    void *data;
    uint64_t capacity;
    uint64_t bytes;
} k3_static_slot;

typedef struct {
    uint32_t expert;
    uint32_t begin;
    uint32_t count;
} k3_expert_group;

typedef struct k3_stream_batch k3_stream_batch;

enum {
    K3_EXPERT_IO_QUEUE_CAPACITY =
        4 * K3_STREAMED_EXPERT_SLOTS,
};

typedef struct {
    k3_stream_batch *batch;
    uint32_t index;
} k3_expert_io_task;

typedef struct {
    pthread_t threads[K3_EXPERT_PREFETCH_THREADS_MAX];
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    k3_expert_io_task high[K3_EXPERT_IO_QUEUE_CAPACITY];
    k3_expert_io_task low[K3_EXPERT_IO_QUEUE_CAPACITY];
    uint32_t high_count;
    uint32_t low_count;
    uint32_t thread_count;
    uint32_t started;
    int initialized;
    int stopping;
} k3_expert_io_pool;

typedef struct {
    const k3_st_tensor *tensor;
    uint64_t offset;
    uint64_t bytes;
} k3_q8_weight;

struct k3_q8_layer_cache {
    ds4_gpu_tensor *weights;
    k3_static_slot native;
    k3_q8_weight matrices[K3_Q8_MAX_MATRICES];
    uint32_t matrix_count;
    uint64_t bytes;
};

struct k3_context {
    /*
     * Cloned DwarfStar sessions share immutable weights and one transient
     * execution workspace. Their KDA/MLA state below remains private. The
     * common engine serializes graph execution already; this lock also makes
     * the direct runtime API's sharing contract explicit.
     */
    struct k3_context *shared_owner;
    pthread_mutex_t execution_mutex;
    int execution_mutex_initialized;
    int owns_execution;
    k3_model *model;
    k3_global_weights globals;
    k3_layer *layers[K3_N_LAYER];
    int expert_fds[K3_N_LAYER];
    uint32_t context_capacity;
    uint32_t max_prefill_tokens;
    uint32_t position;
    uint32_t layer_start;
    uint32_t layer_end;
    int has_output;
    uint64_t expert_stream_bytes;
    double expert_stream_seconds;
    uint64_t static_stream_bytes;
    double static_stream_seconds;
    uint64_t route_overlap;
    uint64_t route_total;
    uint64_t route_prediction_hits;
    uint64_t route_prediction_total;
    uint64_t route_prediction_rank_hits[K3_N_EXPERT_USED];
    uint64_t route_prediction_transitions;
    uint64_t speculative_stream_bytes;
    uint64_t speculative_route_hits;
    uint64_t speculative_route_total;
    double speculative_stream_seconds;
    int32_t previous_route[K3_N_LAYER][K3_N_EXPERT_USED];
    uint8_t previous_route_valid[K3_N_LAYER];
    int32_t predicted_route[K3_N_LAYER][K3_N_EXPERT_USED];
    uint8_t predicted_route_valid[K3_N_LAYER];
    int static_staging;
    int resident_static;
    int selective_shared;
    int pinned_static;
    int q8_static;
    int gguf_resident;
    int gguf_static_resident;
    int gguf_readahead_done;
    int speculative_enabled;
    k3_hidden_hook_fn hidden_hook;
    void *hidden_hook_ud;
    uint64_t resident_static_bytes;
    k3_static_slot static_slots[2];
    k3_static_slot resident_static_slots[K3_N_LAYER];
    k3_static_slot resident_shared_slots[K3_N_LAYER];
    k3_static_slot resident_lm_head;
    k3_q8_layer_cache q8_layers[K3_N_LAYER];
    k3_expert_io_pool expert_io_pool;
    k3_scratch scratch;
    k3_streamed_expert streamed_experts[K3_STREAMED_EXPERT_SLOTS];
    k3_streamed_expert
        speculative_experts[K3_SPECULATIVE_BANKS][K3_STREAMED_EXPERT_SLOTS];
    k3_expert_group
        speculative_groups[K3_SPECULATIVE_BANKS][K3_STREAMED_EXPERT_SLOTS];
    k3_stream_batch *speculative_batch[K3_SPECULATIVE_BANKS];
    uint32_t speculative_layer[K3_SPECULATIVE_BANKS];
    uint32_t speculative_count[K3_SPECULATIVE_BANKS];
    double speculative_started[K3_SPECULATIVE_BANKS];
    k3_kda_state kda[K3_N_LAYER];
    k3_mla_state mla[K3_N_LAYER];
};

static k3_context *execution_owner(k3_context *ctx) {
    return ctx && ctx->shared_owner ? ctx->shared_owner : ctx;
}

static void hidden_hook_emit(k3_context *ctx,
                             const char *stream,
                             uint32_t layer,
                             ds4_gpu_tensor *tensor,
                             uint32_t rows) {
    k3_context *owner = execution_owner(ctx);
    if (!owner || !owner->hidden_hook) return;
    owner->hidden_hook(owner->hidden_hook_ud,
                       stream,
                       layer,
                       ctx->position,
                       tensor,
                       (uint64_t)rows * K3_N_EMBD);
}

typedef struct {
    k3_layer *layer;
    int fd;
    const k3_expert_group *groups;
    k3_streamed_expert *slots;
    uint32_t count;
    uint32_t remaining;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    uint8_t completed[K3_STREAMED_EXPERT_SLOTS];
    uint8_t canceled[K3_STREAMED_EXPERT_SLOTS];
    uint64_t bytes_read;
    int ok;
    char error[256];
} k3_stream_work;

struct k3_stream_batch {
    k3_stream_work work;
    int initialized;
};

typedef struct {
    pthread_t thread;
    k3_st_shard *shard;
    const k3_st_tensor *tensor;
    int started;
    int ok;
    char error[256];
} k3_output_prefetch;

typedef struct {
    pthread_t thread;
    k3_layer *layer;
    k3_static_slot *slot;
    uint64_t offset;
    uint64_t size;
    uint64_t skip_offset;
    uint64_t skip_size;
    int started;
    int ok;
    double seconds;
    char error[256];
} k3_static_load;

static void set_error(char *err, size_t cap, const char *fmt, ...) {
    if (!err || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int trace_enabled(void) {
    const char *value = getenv("K3_TRACE");
    return value && value[0] && strcmp(value, "0") != 0;
}

static int io_profile_enabled(void) {
    const char *value = getenv("K3_IO_PROFILE");
    return value && value[0] && strcmp(value, "0") != 0;
}

static int speculative_experts_enabled(void) {
    const char *value = getenv("K3_SPECULATIVE_EXPERTS");
    return value && value[0] && strcmp(value, "0") != 0;
}

static int route_prediction_enabled(const k3_context *ctx) {
    const char *value = getenv("K3_ROUTE_PREDICT");
    return (ctx && ctx->speculative_enabled) ||
           (value && value[0] && strcmp(value, "0") != 0);
}

static int layer_pagein_enabled(void) {
    const char *value = getenv("K3_LAYER_PAGEIN");
    return value && value[0] && strcmp(value, "0") != 0;
}

static int static_staging_requested(void) {
    const char *value = getenv("K3_STATIC_STAGING");
    return !value || !value[0] || strcmp(value, "0") != 0;
}

static uint64_t macos_physical_memory(void) {
    uint64_t bytes = 0;
    size_t size = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &size, NULL, 0) != 0 ||
        size != sizeof(bytes)) {
        return 0;
    }
    return bytes;
}

typedef struct {
    uint64_t reclaimable;
    uint64_t free;
    uint64_t file_backed;
    uint64_t purgeable;
    uint64_t compressed;
} k3_memory_status;

static int macos_memory_status(k3_memory_status *status) {
    if (!status) return 0;
    memset(status, 0, sizeof(*status));
    vm_statistics64_data_t vm = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const mach_port_t host = mach_host_self();
    if (host_statistics64(
            host, HOST_VM_INFO64,
            (host_info64_t)&vm, &count) != KERN_SUCCESS) {
        mach_port_deallocate(mach_task_self(), host);
        return 0;
    }
    vm_size_t page_size = 0;
    if (host_page_size(host, &page_size) != KERN_SUCCESS ||
        page_size == 0) {
        mach_port_deallocate(mach_task_self(), host);
        return 0;
    }
    mach_port_deallocate(mach_task_self(), host);

    /* XNU includes speculative pages in free_count and also classifies them as
     * file-backed. Count them once. Inactive pages are deliberately excluded:
     * that queue can contain dirty anonymous memory which would require
     * compression or swap rather than cheap file-cache eviction. */
    const uint64_t free_pages =
        vm.free_count > vm.speculative_count
            ? (uint64_t)vm.free_count - vm.speculative_count : 0;
    const uint64_t file_pages = vm.external_page_count;
    const uint64_t purgeable_pages = vm.purgeable_count;
    uint64_t reclaimable_pages = free_pages;
    if (file_pages > UINT64_MAX - reclaimable_pages) return 0;
    reclaimable_pages += file_pages;
    if (purgeable_pages > UINT64_MAX - reclaimable_pages) return 0;
    reclaimable_pages += purgeable_pages;
    if (reclaimable_pages > UINT64_MAX / (uint64_t)page_size) return 0;

    status->reclaimable =
        reclaimable_pages * (uint64_t)page_size;
    status->free = free_pages * (uint64_t)page_size;
    status->file_backed = file_pages * (uint64_t)page_size;
    status->purgeable = purgeable_pages * (uint64_t)page_size;
    status->compressed =
        (uint64_t)vm.compressor_page_count * (uint64_t)page_size;
    return 1;
}

static uint64_t resident_memory_reserve(void) {
    const char *value = getenv("K3_MEMORY_RESERVE_GIB");
    if (value && value[0]) {
        char *end = NULL;
        errno = 0;
        const double gib = strtod(value, &end);
        if (!errno && end != value && *end == '\0' &&
            isfinite(gib) && gib >= 4.0 && gib <= 64.0) {
            return (uint64_t)(gib * 1073741824.0);
        }
        fprintf(stderr,
                "k3: ignoring invalid K3_MEMORY_RESERVE_GIB=%s "
                "(expected 4..64)\n",
                value);
    }
    uint64_t reserve = macos_physical_memory() / 8u;
    const uint64_t minimum = 12ull * 1024ull * 1024ull * 1024ull;
    const uint64_t maximum = 16ull * 1024ull * 1024ull * 1024ull;
    if (reserve < minimum) reserve = minimum;
    if (reserve > maximum) reserve = maximum;
    return reserve;
}

static uint64_t resident_static_budget(uint64_t allocated_bytes) {
    const uint64_t recommended =
        ds4_gpu_recommended_working_set_size();
    const uint64_t reserve = resident_memory_reserve();
    k3_memory_status memory = {0};
    const int have_memory = macos_memory_status(&memory);
    const uint64_t reclaimable = memory.reclaimable;
    const uint64_t safe_available =
        have_memory && reclaimable > reserve
            ? reclaimable - reserve : 0;
    const uint64_t recommended_available =
        recommended > allocated_bytes
            ? recommended - allocated_bytes : 0;
    uint64_t safety_cap = have_memory
        ? safe_available : recommended_available;
    if (recommended != 0 &&
        recommended_available < safety_cap) {
        safety_cap = recommended_available;
    }

    const char *value = getenv("K3_RESIDENT_STATIC_GIB");
    if (!value || !value[0]) {
        value = getenv("K3_RESIDENT_STATIC");
        if (value && strcmp(value, "0") == 0) return 0;
        if (!value || !value[0] || strcmp(value, "1") == 0 ||
            strcmp(value, "auto") == 0) {
            if (safety_cap != 0) {
                fprintf(stderr,
                        "k3: automatic resident static budget %.2f GiB "
                        "(%.2f GiB free, %.2f GiB file-backed, "
                        "%.2f GiB reclaimable, %.2f GiB process reserve, "
                        "%.2f GiB already allocated)\n",
                        (double)safety_cap / 1073741824.0,
                        (double)memory.free / 1073741824.0,
                        (double)memory.file_backed / 1073741824.0,
                        (double)reclaimable / 1073741824.0,
                        (double)reserve / 1073741824.0,
                        (double)allocated_bytes / 1073741824.0);
            } else {
                fprintf(stderr,
                        "k3: resident static cache disabled by live memory "
                        "limit (%.2f GiB reclaimable, %.2f GiB reserve, "
                        "%.2f GiB already allocated)\n",
                        (double)reclaimable / 1073741824.0,
                        (double)reserve / 1073741824.0,
                        (double)allocated_bytes / 1073741824.0);
            }
            return safety_cap;
        }
    }
    char *end = NULL;
    errno = 0;
    const double gib = strtod(value, &end);
    if (errno || end == value || *end != '\0' || !isfinite(gib) ||
        gib <= 0.0 || gib > 120.0) {
        fprintf(stderr,
                "k3: ignoring invalid resident static budget %s "
                "(expected 0..120 GiB)\n",
                value);
        return 0;
    }
    const uint64_t requested =
        (uint64_t)(gib * 1073741824.0);
    const char *unsafe = getenv("K3_UNSAFE_RESIDENT_STATIC");
    if (requested > safety_cap &&
        (!unsafe || !unsafe[0] || !strcmp(unsafe, "0"))) {
        fprintf(stderr,
                "k3: capping requested resident static budget %.2f GiB "
                "to %.2f GiB (%.2f GiB reclaimable, %.2f GiB reserve); "
                "set K3_UNSAFE_RESIDENT_STATIC=1 only for controlled tests\n",
                (double)requested / 1073741824.0,
                (double)safety_cap / 1073741824.0,
                (double)reclaimable / 1073741824.0,
                (double)reserve / 1073741824.0);
        return safety_cap;
    }
    return requested;
}

static int q8_static_requested(void) {
    const char *value = getenv("K3_Q8_STATIC");
    return value && value[0] && strcmp(value, "0") != 0;
}

static uint32_t expert_io_threads(void) {
    static uint32_t cached = 0;
    if (cached != 0) return cached;
    cached = 8;
    const char *value = getenv("K3_EXPERT_IO_THREADS");
    if (!value || !value[0]) return cached;
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (!errno && end != value && *end == '\0' &&
        parsed >= 1 && parsed <= K3_EXPERT_PREFETCH_THREADS_MAX) {
        cached = (uint32_t)parsed;
    } else {
        fprintf(stderr,
                "k3: ignoring invalid K3_EXPERT_IO_THREADS=%s "
                "(expected 1..%u)\n",
                value, (uint32_t)K3_EXPERT_PREFETCH_THREADS_MAX);
    }
    return cached;
}

static uint32_t expert_pipeline_batch(void) {
    static int initialized = 0;
    static uint32_t cached = 0;
    if (initialized) return cached;
    initialized = 1;
    const char *value = getenv("K3_EXPERT_PIPELINE_BATCH");
    if (!value || !value[0]) return cached;
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (!errno && end != value && *end == '\0' &&
        parsed <= K3_STREAMED_EXPERT_SLOTS) {
        cached = (uint32_t)parsed;
    } else {
        fprintf(stderr,
                "k3: ignoring invalid K3_EXPERT_PIPELINE_BATCH=%s "
                "(expected 0..%u)\n",
                value, (uint32_t)K3_STREAMED_EXPERT_SLOTS);
    }
    return cached;
}

static uint32_t speculative_top_k(void) {
    static uint32_t cached = 0;
    if (cached != 0) return cached;
    cached = 8;
    const char *value = getenv("K3_SPECULATIVE_TOP_K");
    if (!value || !value[0]) return cached;
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (!errno && end != value && *end == '\0' &&
        parsed >= 1 && parsed <= K3_N_EXPERT_USED) {
        cached = (uint32_t)parsed;
    } else {
        fprintf(stderr,
                "k3: ignoring invalid K3_SPECULATIVE_TOP_K=%s "
                "(expected 1..%u)\n",
                value, (uint32_t)K3_N_EXPERT_USED);
    }
    return cached;
}

static int speculative_early_enabled(void) {
    const char *value = getenv("K3_SPECULATIVE_EARLY");
    return !value || !value[0] || strcmp(value, "0") != 0;
}

static uint32_t static_io_threads(void) {
    static uint32_t cached = 0;
    if (cached != 0) return cached;
    cached = 16;
    const char *value = getenv("K3_STATIC_IO_THREADS");
    if (!value || !value[0]) return cached;
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (!errno && end != value && *end == '\0' &&
        parsed >= 1 && parsed <= K3_STATIC_IO_THREADS_MAX) {
        cached = (uint32_t)parsed;
    } else {
        fprintf(stderr,
                "k3: ignoring invalid K3_STATIC_IO_THREADS=%s "
                "(expected 1..%u)\n",
                value, (uint32_t)K3_STATIC_IO_THREADS_MAX);
    }
    return cached;
}

static int checked_bytes(uint64_t rows,
                         uint64_t width,
                         uint64_t element_size,
                         uint64_t *out) {
    if (!out || rows == 0 || width == 0 || element_size == 0 ||
        rows > UINT64_MAX / width ||
        rows * width > UINT64_MAX / element_size) {
        return 0;
    }
    *out = rows * width * element_size;
    return 1;
}

static ds4_gpu_tensor *alloc_tensor(uint64_t rows,
                                    uint64_t width,
                                    uint64_t element_size,
                                    int managed) {
    uint64_t bytes = 0;
    if (!checked_bytes(rows, width, element_size, &bytes)) return NULL;
    return managed ? ds4_gpu_tensor_alloc_managed(bytes)
                   : ds4_gpu_tensor_alloc(bytes);
}

static void free_tensor(ds4_gpu_tensor **tensor) {
    if (!tensor || !*tensor) return;
    ds4_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static int alloc_scratch(k3_context *ctx, char *err, size_t errcap) {
    k3_scratch *s = &ctx->scratch;
    const uint64_t rows = ctx->max_prefill_tokens;
    const uint64_t pairs = rows * K3_N_EXPERT_USED;
    const uint64_t max_partials =
        ((uint64_t)K3_MAX_CONTEXT + 1023ull) / 1024ull;
    const uint64_t q_latent_rows =
        rows > max_partials ? rows : max_partials;

#define ALLOC(field_, rows_, width_, size_, managed_) do { \
    s->field_ = alloc_tensor((rows_), (width_), (size_), (managed_)); \
    if (!s->field_) { \
        set_error(err, errcap, "GPU allocation failed for K3 scratch " #field_); \
        return 0; \
    } \
} while (0)

    ALLOC(embedding_rows, rows, K3_N_EMBD, sizeof(uint16_t), 1);
    ALLOC(prefix, rows, K3_N_EMBD, sizeof(float), 0);
    ALLOC(delta, rows, K3_N_EMBD, sizeof(float), 0);
    ALLOC(blocks, rows, K3_ATTN_RES_BLOCKS * K3_N_EMBD, sizeof(float), 0);
    ALLOC(attn_in, rows, K3_N_EMBD, sizeof(float), 0);
    ALLOC(attn_heads, rows, K3_KDA_PROJ, sizeof(float), 0);
    ALLOC(attn_out, rows, K3_N_EMBD, sizeof(float), 0);
    ALLOC(ffn_in, rows, K3_N_EMBD, sizeof(float), 0);
    ALLOC(final_hidden, rows, K3_N_EMBD, sizeof(float), 0);
    ALLOC(final_norm, 1, K3_N_EMBD, sizeof(float), 0);
    ALLOC(logits, 1, K3_N_VOCAB, sizeof(float), 1);

    ALLOC(q, rows, K3_MLA_Q_WIDTH, sizeof(float), 0);
    ALLOC(k, rows, K3_KDA_PROJ, sizeof(float), 0);
    ALLOC(v, rows, K3_KDA_PROJ, sizeof(float), 0);
    ALLOC(raw_gate, rows, K3_KDA_PROJ, sizeof(float), 0);
    ALLOC(beta, rows, K3_N_HEAD, sizeof(float), 0);
    ALLOC(output_gate, rows, K3_KDA_PROJ, sizeof(float), 0);
    ALLOC(low_rank, rows, K3_Q_LORA, sizeof(float), 0);
    ALLOC(low_rank_norm, rows, K3_Q_LORA, sizeof(float), 0);
    ALLOC(kv, rows, K3_MLA_CACHE_WIDTH, sizeof(float), 0);
    ALLOC(q_latent, q_latent_rows, K3_MLA_LATENT_WIDTH, sizeof(float), 0);
    ALLOC(partial_values, max_partials, K3_MLA_LATENT_WIDTH, sizeof(float), 0);
    ALLOC(partial_stats, max_partials, K3_N_HEAD * 2u, sizeof(float), 0);

    ALLOC(dense_gate, rows, K3_DENSE_FF, sizeof(float), 0);
    ALLOC(dense_up, rows, K3_DENSE_FF, sizeof(float), 0);
    ALLOC(dense_mid, rows, K3_DENSE_FF, sizeof(float), 0);

    ALLOC(router_logits, rows, K3_N_EXPERT, sizeof(float), 0);
    ALLOC(selected_ids, pairs, 1, sizeof(int32_t), 1);
    ALLOC(selected_weights, pairs, 1, sizeof(float), 1);
    ALLOC(predicted_router_logits, 1, K3_N_EXPERT, sizeof(float), 0);
    ALLOC(predicted_ids, K3_N_EXPERT_USED, 1, sizeof(int32_t), 1);
    ALLOC(predicted_weights, K3_N_EXPERT_USED, 1, sizeof(float), 1);
    ALLOC(routed_latent, rows, K3_ROUTED_HIDDEN, sizeof(float), 0);
    ALLOC(routed_sum, rows, K3_ROUTED_HIDDEN, sizeof(float), 0);
    ALLOC(routed_norm, rows, K3_ROUTED_HIDDEN, sizeof(float), 0);
    ALLOC(routed_out, rows, K3_N_EMBD, sizeof(float), 0);
    ALLOC(shared_gate, rows, K3_SHARED_FF, sizeof(float), 0);
    ALLOC(shared_up, rows, K3_SHARED_FF, sizeof(float), 0);
    ALLOC(shared_mid, rows, K3_SHARED_FF, sizeof(float), 0);
    ALLOC(shared_out, rows, K3_N_EMBD, sizeof(float), 0);
    ALLOC(shared_input_values, 1, K3_N_EMBD, sizeof(uint8_t), 0);
    ALLOC(shared_input_scales, 1, K3_N_EMBD / 32u,
          sizeof(uint8_t), 0);
    ALLOC(shared_mid_values, 1, K3_SHARED_FF, sizeof(uint8_t), 0);
    ALLOC(shared_mid_scales, 1, K3_SHARED_FF / 32u,
          sizeof(uint8_t), 0);

    ALLOC(group_row_ids, pairs, 1, sizeof(int32_t), 1);
    ALLOC(group_weights, pairs, 1, sizeof(float), 1);
    ALLOC(group_input, rows, K3_ROUTED_HIDDEN, sizeof(float), 0);
    ALLOC(group_input_values, rows, K3_ROUTED_HIDDEN, sizeof(uint8_t), 0);
    ALLOC(group_input_scales, rows, K3_ROUTED_HIDDEN / 32u,
          sizeof(uint8_t), 0);
    const uint64_t expert_rows = ctx->model && ctx->model->gguf
        ? pairs : rows;
    ALLOC(expert_gate, expert_rows, K3_EXPERT_FF, sizeof(float), 0);
    ALLOC(expert_up, expert_rows, K3_EXPERT_FF, sizeof(float), 0);
    ALLOC(expert_mid, expert_rows, K3_EXPERT_FF, sizeof(float), 0);
    ALLOC(expert_mid_values, rows, K3_EXPERT_FF, sizeof(uint8_t), 0);
    ALLOC(expert_mid_scales, rows, K3_EXPERT_FF / 32u,
          sizeof(uint8_t), 0);
    ALLOC(expert_out, expert_rows, K3_ROUTED_HIDDEN, sizeof(float), 0);
#undef ALLOC
    return 1;
}

static void free_scratch(k3_scratch *s) {
#define FREE(field_) free_tensor(&s->field_)
    FREE(embedding_rows);
    FREE(prefix);
    FREE(delta);
    FREE(blocks);
    FREE(attn_in);
    FREE(attn_heads);
    FREE(attn_out);
    FREE(ffn_in);
    FREE(final_hidden);
    FREE(final_norm);
    FREE(logits);
    FREE(q);
    FREE(k);
    FREE(v);
    FREE(raw_gate);
    FREE(beta);
    FREE(output_gate);
    FREE(low_rank);
    FREE(low_rank_norm);
    FREE(kv);
    FREE(q_latent);
    FREE(partial_values);
    FREE(partial_stats);
    FREE(dense_gate);
    FREE(dense_up);
    FREE(dense_mid);
    FREE(router_logits);
    FREE(selected_ids);
    FREE(selected_weights);
    FREE(predicted_router_logits);
    FREE(predicted_ids);
    FREE(predicted_weights);
    FREE(routed_latent);
    FREE(routed_sum);
    FREE(routed_norm);
    FREE(routed_out);
    FREE(shared_gate);
    FREE(shared_up);
    FREE(shared_mid);
    FREE(shared_out);
    FREE(shared_input_values);
    FREE(shared_input_scales);
    FREE(shared_mid_values);
    FREE(shared_mid_scales);
    FREE(group_row_ids);
    FREE(group_weights);
    FREE(group_input);
    FREE(group_input_values);
    FREE(group_input_scales);
    FREE(expert_gate);
    FREE(expert_up);
    FREE(expert_mid);
    FREE(expert_mid_values);
    FREE(expert_mid_scales);
    FREE(expert_out);
#undef FREE
}

static uint64_t streamed_expert_bytes(void) {
    const uint64_t w1_packed =
        (uint64_t)K3_EXPERT_FF * (K3_ROUTED_HIDDEN / 2u);
    const uint64_t w1_scale =
        (uint64_t)K3_EXPERT_FF * (K3_ROUTED_HIDDEN / 32u);
    const uint64_t w2_packed =
        (uint64_t)K3_ROUTED_HIDDEN * (K3_EXPERT_FF / 2u);
    const uint64_t w2_scale =
        (uint64_t)K3_ROUTED_HIDDEN * (K3_EXPERT_FF / 32u);
    return 2u * (w1_packed + w1_scale) + w2_packed + w2_scale;
}

static void set_streamed_expert_layout(k3_streamed_expert *slot) {
    const uint64_t w1_packed_bytes =
        (uint64_t)K3_EXPERT_FF * (K3_ROUTED_HIDDEN / 2u);
    const uint64_t w1_scale_bytes =
        (uint64_t)K3_EXPERT_FF * (K3_ROUTED_HIDDEN / 32u);
    const uint64_t w2_packed_bytes =
        (uint64_t)K3_ROUTED_HIDDEN * (K3_EXPERT_FF / 2u);
    const uint64_t w2_scale_bytes =
        (uint64_t)K3_ROUTED_HIDDEN * (K3_EXPERT_FF / 32u);
    slot->w1_packed = 0;
    slot->w1_scale = slot->w1_packed + w1_packed_bytes;
    slot->w2_packed = slot->w1_scale + w1_scale_bytes;
    slot->w2_scale = slot->w2_packed + w2_packed_bytes;
    slot->w3_packed = slot->w2_scale + w2_scale_bytes;
    slot->w3_scale = slot->w3_packed + w1_packed_bytes;
    slot->bytes = slot->w3_scale + w1_scale_bytes;
}

static int alloc_streamed_expert_bank(
        k3_streamed_expert slots[K3_STREAMED_EXPERT_SLOTS],
        uint32_t count,
        const char *label,
        char *err,
        size_t errcap) {
    if (!slots || count == 0 || count > K3_STREAMED_EXPERT_SLOTS) {
        set_error(err, errcap, "invalid K3 streamed expert bank");
        return 0;
    }
    const uint64_t bytes = streamed_expert_bytes();
    for (uint32_t i = 0; i < count; i++) {
        k3_streamed_expert *slot = &slots[i];
        set_streamed_expert_layout(slot);
        if (slot->bytes != bytes) {
            set_error(err, errcap, "K3 streamed expert layout mismatch");
            return 0;
        }
        slot->weights = ds4_gpu_tensor_alloc(bytes);
        if (!slot->weights) {
            set_error(
                err, errcap,
                "GPU allocation failed for K3 %s expert slot %u",
                label ? label : "streamed", i);
            return 0;
        }
    }
    return 1;
}

static void free_streamed_expert_bank(
        k3_streamed_expert slots[K3_STREAMED_EXPERT_SLOTS]) {
    for (uint32_t i = 0; i < K3_STREAMED_EXPERT_SLOTS; i++) {
        free_tensor(&slots[i].weights);
    }
}

static int layer_shared_span(const k3_layer *layer,
                             uint64_t *offset,
                             uint64_t *bytes) {
    if (!layer || layer->index == 0 || !layer->shared_gate ||
        !layer->shared_up || !layer->shared_down || !offset || !bytes) {
        return 0;
    }
    const k3_st_tensor *tensors[] = {
        layer->shared_gate,
        layer->shared_up,
        layer->shared_down,
    };
    uint64_t begin = UINT64_MAX;
    uint64_t end = 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < 3; i++) {
        const k3_st_tensor *tensor = tensors[i];
        if (tensor->offset < begin) begin = tensor->offset;
        if (tensor->size > UINT64_MAX - tensor->offset) {
            return 0;
        }
        const uint64_t tensor_end = tensor->offset + tensor->size;
        if (tensor_end > end) end = tensor_end;
        if (tensor->size > UINT64_MAX - total) return 0;
        total += tensor->size;
    }
    if (layer->static_size > UINT64_MAX - layer->static_offset ||
        begin < layer->static_offset ||
        end > layer->static_offset + layer->static_size ||
        end - begin != total) {
        return 0;
    }
    *offset = begin;
    *bytes = end - begin;
    return 1;
}

static uint64_t layer_stream_static_size(const k3_context *ctx,
                                         const k3_layer *layer) {
    if (!ctx || !layer) return 0;
    if (!ctx->selective_shared ||
        !ctx->resident_shared_slots[layer->index].data) {
        return layer->static_size;
    }
    uint64_t shared_offset = 0;
    uint64_t shared_bytes = 0;
    if (!layer_shared_span(layer, &shared_offset, &shared_bytes) ||
        shared_bytes > layer->static_size) {
        return 0;
    }
    return layer->static_size - shared_bytes;
}

static void layer_static_skip(const k3_context *ctx,
                              const k3_layer *layer,
                              uint64_t *offset,
                              uint64_t *bytes) {
    if (offset) *offset = 0;
    if (bytes) *bytes = 0;
    if (!ctx || !layer || !offset || !bytes ||
        !ctx->selective_shared ||
        !ctx->resident_shared_slots[layer->index].data) {
        return;
    }
    (void)layer_shared_span(layer, offset, bytes);
}

static uint64_t static_slots_capacity(const k3_context *ctx) {
    const long page_value = sysconf(_SC_PAGESIZE);
    if (!ctx || page_value <= 0) return 0;
    uint64_t max_static[2] = {0};
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        const uint32_t slot = i & 1u;
        const uint64_t bytes = ctx->layers[i]->static_size;
        if (bytes > max_static[slot]) max_static[slot] = bytes;
    }
    const uint64_t page = (uint64_t)page_value;
    uint64_t total = 0;
    for (uint32_t i = 0; i < 2; i++) {
        if (max_static[i] == 0 ||
            max_static[i] > UINT64_MAX - (page - 1u)) {
            return 0;
        }
        const uint64_t capacity =
            (max_static[i] + page - 1u) & ~(page - 1u);
        if (capacity > UINT64_MAX - total) return 0;
        total += capacity;
    }
    return total;
}

static int alloc_static_slots(
        k3_context *ctx,
        char *err,
        size_t errcap) {
    uint64_t max_static[2] = {0};
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        const uint32_t slot = i & 1u;
        const uint64_t bytes = ctx->layers[i]->static_size;
        if (bytes > max_static[slot]) {
            max_static[slot] = bytes;
        }
    }
    const long page_value = sysconf(_SC_PAGESIZE);
    if (max_static[0] == 0 || max_static[1] == 0 || page_value <= 0) {
        set_error(err, errcap, "invalid K3 static staging size");
        return 0;
    }
    const uint64_t page = (uint64_t)page_value;
    for (uint32_t i = 0; i < 2; i++) {
        if (max_static[i] > UINT64_MAX - (page - 1u)) {
            set_error(err, errcap, "K3 static staging size overflow");
            return 0;
        }
        const uint64_t capacity =
            (max_static[i] + page - 1u) & ~(page - 1u);
        if (capacity > SIZE_MAX) {
            set_error(err, errcap, "K3 static staging exceeds address space");
            return 0;
        }
        void *data = mmap(NULL,
                          (size_t)capacity,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON,
                          -1,
                          0);
        if (data == MAP_FAILED) {
            set_error(err, errcap,
                      "mmap K3 static slot %u (%.2f GiB): %s",
                      i,
                      (double)capacity / 1073741824.0,
                      strerror(errno));
            return 0;
        }
#if defined(MADV_SEQUENTIAL)
        (void)madvise(data, (size_t)capacity, MADV_SEQUENTIAL);
#endif
        ctx->static_slots[i] = (k3_static_slot){
            .data = data,
            .capacity = capacity,
        };
    }
    ctx->static_staging = 1;
    return 1;
}

static int alloc_resident_static_slots(
        k3_context *ctx,
        uint64_t budget,
        char *err,
        size_t errcap) {
    const long page_value = sysconf(_SC_PAGESIZE);
    if (!ctx || budget == 0 || page_value <= 0) {
        set_error(err, errcap, "invalid K3 resident static allocation");
        return 0;
    }
    const uint64_t page = (uint64_t)page_value;
    uint64_t full_sizes[K3_N_LAYER] = {0};
    uint64_t shared_sizes[K3_N_LAYER] = {0};
    uint64_t full_capacity = 0;
    uint64_t shared_capacity = 0;
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        const uint64_t bytes = ctx->layers[i]->static_size;
        if (bytes == 0 || bytes > UINT64_MAX - (page - 1u)) {
            set_error(err, errcap,
                      "invalid K3 resident static size for layer %u", i);
            return 0;
        }
        const uint64_t capacity = (bytes + page - 1u) & ~(page - 1u);
        full_sizes[i] = capacity;
        if (capacity > UINT64_MAX - full_capacity) {
            set_error(err, errcap, "K3 resident static size overflow");
            return 0;
        }
        full_capacity += capacity;
        if (i != 0) {
            uint64_t shared_offset = 0;
            uint64_t shared_bytes = 0;
            if (!layer_shared_span(
                    ctx->layers[i], &shared_offset, &shared_bytes) ||
                shared_bytes > UINT64_MAX - (page - 1u)) {
                set_error(err, errcap,
                          "invalid K3 shared span for layer %u", i);
                return 0;
            }
            const uint64_t shared_aligned =
                (shared_bytes + page - 1u) & ~(page - 1u);
            shared_sizes[i] = shared_aligned;
            if (shared_aligned > UINT64_MAX - shared_capacity) {
                set_error(err, errcap, "K3 shared resident size overflow");
                return 0;
            }
            shared_capacity += shared_aligned;
        }
    }
    uint8_t full_selected[K3_N_LAYER] = {0};
    uint8_t shared_selected[K3_N_LAYER] = {0};
    const char *shared_first_env = getenv("K3_RESIDENT_SHARED_FIRST");
    const int shared_first =
        shared_first_env && shared_first_env[0] &&
        strcmp(shared_first_env, "0") != 0 &&
        budget < full_capacity && budget >= shared_capacity;
    uint64_t allocated = 0;
    if (shared_first) {
        allocated = shared_capacity;
        for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
            if (i == 0) continue;
            shared_selected[i] = 1;
        }
        for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
            const uint64_t incremental =
                full_sizes[i] - shared_sizes[i];
            if (incremental > budget - allocated) continue;
            full_selected[i] = 1;
            shared_selected[i] = 0;
            allocated += incremental;
        }
    } else {
        for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
            if (full_sizes[i] > budget - allocated) continue;
            full_selected[i] = 1;
            allocated += full_sizes[i];
        }
        for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
            if (i == 0) continue;
            if (full_selected[i] ||
                shared_sizes[i] > budget - allocated) {
                continue;
            }
            shared_selected[i] = 1;
            allocated += shared_sizes[i];
        }
    }
    ctx->selective_shared = 0;
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        if (i == 0) continue;
        if (shared_selected[i]) {
            ctx->selective_shared = 1;
            break;
        }
    }
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        if (!full_selected[i]) continue;
        if (full_sizes[i] > SIZE_MAX) {
            set_error(err, errcap,
                      "K3 resident static layer %u exceeds address space", i);
            return 0;
        }
        void *data = mmap(NULL,
                          (size_t)full_sizes[i],
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON,
                          -1,
                          0);
        if (data == MAP_FAILED) {
            set_error(err, errcap,
                      "mmap K3 resident static layer %u (%.2f GiB): %s",
                      i,
                      (double)full_sizes[i] / 1073741824.0,
                      strerror(errno));
            return 0;
        }
        ctx->resident_static_slots[i] = (k3_static_slot){
            .data = data,
            .capacity = full_sizes[i],
        };
    }
    if (ctx->selective_shared) {
        for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
            if (i == 0) continue;
            if (!shared_selected[i]) continue;
            void *data = mmap(NULL,
                              (size_t)shared_sizes[i],
                              PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANON,
                              -1,
                              0);
            if (data == MAP_FAILED) {
                set_error(err, errcap,
                          "mmap K3 resident shared layer %u (%.2f GiB): %s",
                          i,
                          (double)shared_sizes[i] / 1073741824.0,
                          strerror(errno));
                return 0;
            }
            ctx->resident_shared_slots[i] = (k3_static_slot){
                .data = data,
                .capacity = shared_sizes[i],
            };
        }
    }
    if (allocated == 0) {
        return 1;
    }
    ctx->resident_static = 1;
    ctx->resident_static_bytes = allocated;
    return 1;
}

static void free_static_slots(k3_context *ctx) {
    if (!ctx) return;
    for (uint32_t i = 0; i < 2; i++) {
        k3_static_slot *slot = &ctx->static_slots[i];
        if (slot->data) {
            munmap(slot->data, (size_t)slot->capacity);
        }
        memset(slot, 0, sizeof(*slot));
    }
    ctx->static_staging = 0;
}

static void free_resident_static_slots(k3_context *ctx) {
    if (!ctx) return;
    if (ctx->resident_lm_head.data) {
        munmap(ctx->resident_lm_head.data,
               (size_t)ctx->resident_lm_head.capacity);
    }
    memset(&ctx->resident_lm_head, 0, sizeof(ctx->resident_lm_head));
    for (uint32_t i = 0; i < K3_N_LAYER; i++) {
        k3_static_slot *slot = &ctx->resident_static_slots[i];
        if (slot->data) {
            munmap(slot->data, (size_t)slot->capacity);
        }
        memset(slot, 0, sizeof(*slot));
        slot = &ctx->resident_shared_slots[i];
        if (slot->data) {
            munmap(slot->data, (size_t)slot->capacity);
        }
        memset(slot, 0, sizeof(*slot));
    }
    ctx->resident_static = 0;
    ctx->selective_shared = 0;
    ctx->resident_static_bytes = 0;
}

static int alloc_states(k3_context *ctx, char *err, size_t errcap) {
    const uint64_t conv_count =
        3ull * (K3_CONV_WIDTH - 1u) * K3_KDA_PROJ;
    const uint64_t recurrent_count =
        (uint64_t)K3_KDA_PROJ * K3_HEAD_DIM;
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        if (k3_layer_is_mla(i)) {
            ctx->mla[i].cache = alloc_tensor(
                ctx->context_capacity,
                K3_MLA_CACHE_WIDTH,
                sizeof(uint16_t),
                0);
            if (!ctx->mla[i].cache) {
                set_error(err, errcap,
                          "GPU allocation failed for MLA cache layer %u", i);
                return 0;
            }
        } else {
            ctx->kda[i].conv = alloc_tensor(
                conv_count, 1, sizeof(float), 0);
            ctx->kda[i].recurrent = alloc_tensor(
                recurrent_count, 1, sizeof(float), 0);
            if (!ctx->kda[i].conv || !ctx->kda[i].recurrent ||
                !ds4_gpu_tensor_fill_f32(
                    ctx->kda[i].conv, 0.0f, conv_count) ||
                !ds4_gpu_tensor_fill_f32(
                    ctx->kda[i].recurrent, 0.0f, recurrent_count)) {
                set_error(err, errcap,
                          "GPU allocation failed for KDA state layer %u", i);
                return 0;
            }
        }
    }
    return 1;
}

static void free_states(k3_context *ctx) {
    for (uint32_t i = 0; i < K3_N_LAYER; i++) {
        free_tensor(&ctx->kda[i].conv);
        free_tensor(&ctx->kda[i].recurrent);
        free_tensor(&ctx->mla[i].cache);
    }
}

static int require_no_layer_mappings(
        const k3_context *ctx,
        const char *phase,
        char *err,
        size_t errcap) {
    if (!ctx) return 0;
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        if (ctx->layers[i] && k3_st_map_base(ctx->layers[i]->shard)) {
            set_error(err, errcap,
                      "K3 layer %u remained mapped at %s boundary",
                      i, phase ? phase : "evaluation");
            return 0;
        }
    }
    return 1;
}

static int map_range(const k3_st_shard *shard,
                     uint64_t offset,
                     uint64_t size,
                     uint64_t max_tensor,
                     char *err,
                     size_t errcap) {
    const void *base = k3_st_map_base(shard);
    const uint64_t file_size = k3_st_file_size(shard);
    if (max_tensor > size) max_tensor = size;
    if (!base || !ds4_gpu_set_model_fd_for_map(k3_st_fd(shard), base) ||
        !ds4_gpu_set_model_map_range(
            base, file_size, offset, size, max_tensor)) {
        set_error(err, errcap, "Metal could not map %s", k3_st_path(shard));
        return 0;
    }
    return 1;
}

static int map_static_slot_span(
        const k3_static_slot *slot,
        const k3_layer *layer,
        uint64_t bytes,
        char *err,
        size_t errcap) {
    const long page_value = sysconf(_SC_PAGESIZE);
    if (!slot || !layer || !slot->data || page_value <= 0 ||
        bytes == 0 ||
        bytes > UINT64_MAX - ((uint64_t)page_value - 1u)) {
        set_error(err, errcap, "invalid K3 staged layer mapping");
        return 0;
    }
    const uint64_t page = (uint64_t)page_value;
    const uint64_t mapped_bytes =
        (bytes + page - 1u) & ~(page - 1u);
    if (slot->bytes != bytes ||
        slot->capacity < mapped_bytes ||
        !ds4_gpu_set_model_map_range(
            slot->data,
            bytes,
            0,
            bytes,
            K3_LAYER_MAX_TENSOR_BYTES)) {
        set_error(err, errcap,
                  "Metal could not map K3 staged layer %u",
                  layer ? layer->index : UINT32_MAX);
        return 0;
    }
    return 1;
}

static int map_spans(const k3_st_shard *shard,
                     const uint64_t *offsets,
                     const uint64_t *sizes,
                     uint32_t count,
                     uint64_t max_tensor,
                     char *err,
                     size_t errcap) {
    const void *base = k3_st_map_base(shard);
    const uint64_t file_size = k3_st_file_size(shard);
    if (!base || !ds4_gpu_set_model_fd_for_map(k3_st_fd(shard), base) ||
        !ds4_gpu_set_model_map_spans(
            base, file_size, offsets, sizes, count, max_tensor)) {
        set_error(err, errcap, "Metal could not map tensor spans in %s",
                  k3_st_path(shard));
        return 0;
    }
    return 1;
}

static int map_gguf_layer_spans(const k3_context *ctx,
                                const k3_layer *layer,
                                int include_static,
                                char *err,
                                size_t errcap) {
    if (!ctx || !ctx->globals.shard || !layer || !layer->shard) return 0;
    const k3_st_tensor *experts[] = {
        layer->expert_gate,
        layer->expert_up,
        layer->expert_down,
    };
    uint64_t offsets[4] = {0};
    uint64_t sizes[4] = {0};
    uint32_t count = 0;
    uint64_t max_tensor = 1;
    if (include_static) {
        offsets[count] = layer->static_offset;
        sizes[count] = layer->static_size;
        if (sizes[count] > max_tensor) max_tensor = sizes[count];
        count++;
    }
    if (layer->index != 0) {
        for (uint32_t i = 0; i < 3; i++) {
            if (!experts[i] || experts[i]->size == 0) {
                set_error(err, errcap,
                          "K3 layer %u GGUF expert span is missing",
                          layer->index);
                return 0;
            }
            offsets[count] = experts[i]->offset;
            sizes[count] = experts[i]->size;
            if (sizes[count] > max_tensor) max_tensor = sizes[count];
            count++;
        }
    }
    if (count == 0) return 1;
    return map_spans(
        ctx->globals.shard,
        offsets,
        sizes,
        count,
        max_tensor,
        err,
        errcap);
}

static int prefetch_gguf_selected_experts(
        k3_context *ctx,
        const k3_layer *layer,
        uint32_t rows,
        char *err,
        size_t errcap) {
    if (!ctx || !layer || !layer->shard || rows == 0) return 0;
    const int32_t *ids =
        ds4_gpu_tensor_contents(ctx->scratch.selected_ids);
    if (!ids) {
        set_error(err, errcap,
                  "K3 layer %u selected experts are unavailable",
                  layer->index);
        return 0;
    }
    bool selected[K3_N_EXPERT] = {false};
    const uint64_t count = (uint64_t)rows * K3_N_EXPERT_USED;
    for (uint64_t i = 0; i < count; i++) {
        if (ids[i] < 0 || ids[i] >= K3_N_EXPERT) {
            set_error(err, errcap,
                      "K3 layer %u selected invalid expert %d",
                      layer->index, ids[i]);
            return 0;
        }
        selected[(uint32_t)ids[i]] = true;
    }

    const k3_st_tensor *stacks[] = {
        layer->expert_gate,
        layer->expert_up,
        layer->expert_down,
    };
    if (rows == 1u) {
        for (uint32_t i = 0; i < 3; i++) {
            if (!stacks[i] ||
                stacks[i]->size % K3_N_EXPERT != 0) {
                set_error(err, errcap,
                          "K3 layer %u GGUF expert stack is invalid",
                          layer->index);
                return 0;
            }
        }
        const ds4_gpu_stream_expert_table table = {
            .model_map = k3_st_map_base(ctx->globals.shard),
            .model_size = k3_st_file_size(ctx->globals.shard),
            .layer = layer->index,
            .n_total_expert = K3_N_EXPERT,
            .gate_offset = stacks[0]->offset,
            .up_offset = stacks[1]->offset,
            .down_offset = stacks[2]->offset,
            .gate_expert_bytes =
                stacks[0]->size / K3_N_EXPERT,
            .down_expert_bytes =
                stacks[2]->size / K3_N_EXPERT,
        };
        if (!table.model_map ||
            !ds4_gpu_stream_expert_cache_begin_selected_load(
                &table, ids, K3_N_EXPERT_USED)) {
            set_error(err, errcap,
                      "K3 layer %u selected expert staging failed",
                      layer->index);
            return 0;
        }
        return 1;
    }
    for (uint32_t stack = 0;
         stack < sizeof(stacks) / sizeof(stacks[0]);
         stack++) {
        const k3_st_tensor *tensor = stacks[stack];
        if (!tensor || tensor->size % K3_N_EXPERT != 0) {
            set_error(err, errcap,
                      "K3 layer %u GGUF expert stack is invalid",
                      layer->index);
            return 0;
        }
        const uint64_t expert_bytes = tensor->size / K3_N_EXPERT;
        for (uint32_t begin = 0; begin < K3_N_EXPERT;) {
            while (begin < K3_N_EXPERT && !selected[begin]) begin++;
            if (begin == K3_N_EXPERT) break;
            uint32_t end = begin + 1u;
            while (end < K3_N_EXPERT && selected[end]) end++;
            const uint64_t run_count = end - begin;
            if (expert_bytes > UINT64_MAX / run_count ||
                expert_bytes * begin > UINT64_MAX - tensor->offset ||
                !k3_st_prefetch(
                    layer->shard,
                    tensor->offset + expert_bytes * begin,
                    expert_bytes * run_count,
                    false,
                    err,
                    errcap)) {
                return 0;
            }
            begin = end;
        }
    }
    return 1;
}

static int prefetch_tensor(k3_st_shard *shard,
                           const k3_st_tensor *tensor,
                           int touch,
                           char *err,
                           size_t errcap) {
    return tensor &&
           k3_st_prefetch(
               shard, tensor->offset, tensor->size, touch != 0, err, errcap);
}

static int load_embedding_rows(
        k3_context *ctx,
        const int32_t *token_ids,
        uint32_t rows,
        char *err,
        size_t errcap) {
    const k3_st_tensor *embedding = ctx->globals.token_embedding;
    const uint64_t row_bytes = (uint64_t)K3_N_EMBD * sizeof(uint16_t);
    const uint8_t *base = k3_st_map_base(ctx->globals.shard);
    uint8_t *dst = ds4_gpu_tensor_contents(ctx->scratch.embedding_rows);
    if (!embedding || !base || !dst ||
        embedding->offset > k3_st_file_size(ctx->globals.shard) ||
        embedding->size >
            k3_st_file_size(ctx->globals.shard) - embedding->offset ||
        (uint64_t)rows * row_bytes >
            ds4_gpu_tensor_bytes(ctx->scratch.embedding_rows)) {
        set_error(err, errcap, "invalid K3 embedding row staging");
        return 0;
    }
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t token = (uint32_t)token_ids[row];
        const uint64_t offset =
            embedding->offset + (uint64_t)token * row_bytes;
        memcpy(dst + (uint64_t)row * row_bytes, base + offset, row_bytes);
    }
    return 1;
}

static void *output_prefetch_main(void *arg) {
    k3_output_prefetch *job = arg;
    job->ok = prefetch_tensor(
        job->shard, job->tensor, 0, job->error, sizeof(job->error));
    return NULL;
}

static int output_prefetch_start(k3_output_prefetch *job,
                                 k3_st_shard *shard,
                                 const k3_st_tensor *tensor,
                                 char *err,
                                 size_t errcap) {
    if (!job || job->started || !shard || !tensor) {
        set_error(err, errcap, "invalid K3 output prefetch");
        return 0;
    }
    job->shard = shard;
    job->tensor = tensor;
    job->ok = 0;
    job->error[0] = '\0';
    const int rc = pthread_create(
        &job->thread, NULL, output_prefetch_main, job);
    if (rc != 0) {
        set_error(err, errcap, "pthread_create for K3 output prefetch: %s",
                  strerror(rc));
        return 0;
    }
    job->started = 1;
    return 1;
}

static int output_prefetch_join(k3_output_prefetch *job,
                                char *err,
                                size_t errcap) {
    if (!job || !job->started) return 1;
    const int rc = pthread_join(job->thread, NULL);
    job->started = 0;
    if (rc != 0) {
        set_error(err, errcap, "pthread_join for K3 output prefetch: %s",
                  strerror(rc));
        return 0;
    }
    if (!job->ok) {
        set_error(err, errcap, "%s",
                  job->error[0] ? job->error : "K3 output prefetch failed");
        return 0;
    }
    return 1;
}

static void output_prefetch_discard(k3_output_prefetch *job) {
    if (!job || !job->started) return;
    (void)pthread_join(job->thread, NULL);
    job->started = 0;
}

static int pread_streamed_range(
        int fd,
        uint64_t offset,
        uint64_t size,
        uint8_t *dst,
        char *err,
        size_t errcap) {
    if (fd < 0 || !dst || size == 0 ||
        offset > (uint64_t)LLONG_MAX ||
        size > (uint64_t)LLONG_MAX - offset) {
        set_error(err, errcap, "invalid K3 streamed expert tensor");
        return 0;
    }
    uint64_t done = 0;
    while (done < size) {
        const uint64_t remaining = size - done;
        uint64_t chunk = remaining;
        if (chunk > K3_PREAD_CHUNK_BYTES) chunk = K3_PREAD_CHUNK_BYTES;
        if (chunk > (uint64_t)SSIZE_MAX) chunk = (uint64_t)SSIZE_MAX;
        const size_t want = (size_t)chunk;
        ssize_t nread;
        do {
            nread = pread(
                fd, dst + done, want, (off_t)(offset + done));
        } while (nread < 0 && errno == EINTR);
        if (nread <= 0) {
            set_error(err, errcap, "pread K3 weights: %s",
                      nread < 0 ? strerror(errno) : "unexpected end of file");
            return 0;
        }
        done += (uint64_t)nread;
    }
    return 1;
}

static int open_uncached_stream_fd(const k3_st_shard *shard,
                                   char *err,
                                   size_t errcap) {
    if (!shard || !k3_st_path(shard)) {
        set_error(err, errcap, "invalid K3 streaming shard");
        return -1;
    }
    const int fd = open(k3_st_path(shard), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        set_error(err, errcap, "open K3 streaming shard %s: %s",
                  k3_st_path(shard), strerror(errno));
        return -1;
    }
#if defined(F_NOCACHE)
    if (fcntl(fd, F_NOCACHE, 1) != 0) {
        set_error(err, errcap, "F_NOCACHE for %s: %s",
                  k3_st_path(shard), strerror(errno));
        close(fd);
        return -1;
    }
#endif
    return fd;
}

static uint64_t resident_lm_head_capacity(const k3_context *ctx) {
    const long page_value = sysconf(_SC_PAGESIZE);
    if (!ctx || !ctx->globals.lm_head || page_value <= 0 ||
        ctx->globals.lm_head->size >
            UINT64_MAX - ((uint64_t)page_value - 1u)) {
        return 0;
    }
    const uint64_t page = (uint64_t)page_value;
    return (ctx->globals.lm_head->size + page - 1u) & ~(page - 1u);
}

static uint64_t resident_all_layers_capacity(const k3_context *ctx) {
    const long page_value = sysconf(_SC_PAGESIZE);
    if (!ctx || page_value <= 0) return 0;
    const uint64_t page = (uint64_t)page_value;
    uint64_t total = 0;
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        const uint64_t bytes = ctx->layers[i]->static_size;
        if (bytes == 0 || bytes > UINT64_MAX - (page - 1u)) return 0;
        const uint64_t capacity = (bytes + page - 1u) & ~(page - 1u);
        if (capacity > UINT64_MAX - total) return 0;
        total += capacity;
    }
    return total;
}

static int gguf_resident_spans(
        const k3_context *ctx,
        int              include_experts,
        uint64_t         *offsets,
        uint64_t         *sizes,
        uint32_t         *count,
        uint64_t         *total) {
    const long page_value = sysconf(_SC_PAGESIZE);
    if (!ctx || !ctx->model || !ctx->model->gguf ||
        !offsets || !sizes || !count || !total || page_value <= 0) {
        return 0;
    }
    const uint64_t page = (uint64_t)page_value;
    const uint64_t file_size = k3_st_file_size(ctx->globals.shard);
    uint32_t n = 0;
    uint64_t bytes = 0;
#define ADD_GGUF_SPAN(offset_, size_) do { \
    const uint64_t span_offset_ = (offset_); \
    const uint64_t span_size_ = (size_); \
    if (span_offset_ > file_size || span_size_ == 0 || \
        span_size_ > file_size - span_offset_) return 0; \
    const uint64_t aligned_start_ = span_offset_ & ~(page - 1u); \
    const uint64_t leading_ = span_offset_ - aligned_start_; \
    if (span_size_ > UINT64_MAX - leading_ || \
        leading_ + span_size_ > UINT64_MAX - (page - 1u)) return 0; \
    uint64_t aligned_size_ = \
        (leading_ + span_size_ + page - 1u) & ~(page - 1u); \
    if (aligned_size_ > file_size - aligned_start_) \
        aligned_size_ = file_size - aligned_start_; \
    if (aligned_size_ > UINT64_MAX - bytes) return 0; \
    offsets[n] = span_offset_; \
    sizes[n] = span_size_; \
    n++; \
    bytes += aligned_size_; \
} while (0)
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        ADD_GGUF_SPAN(ctx->layers[i]->static_offset,
                      include_experts ? ctx->layers[i]->layer_size
                                      : ctx->layers[i]->static_size);
    }
    /* Streaming reads only the input token rows, not the 2 GiB table. */
    if (include_experts && ctx->layer_start == 0u) {
        ADD_GGUF_SPAN(ctx->globals.token_embedding->offset,
                      ctx->globals.token_embedding->size);
    }
    if (ctx->has_output) {
        ADD_GGUF_SPAN(ctx->globals.output_attn_res_norm->offset,
                      ctx->globals.output_attn_res_norm->size);
        ADD_GGUF_SPAN(ctx->globals.output_attn_res_proj->offset,
                      ctx->globals.output_attn_res_proj->size);
        ADD_GGUF_SPAN(ctx->globals.norm->offset,
                      ctx->globals.norm->size);
        ADD_GGUF_SPAN(ctx->globals.lm_head->offset,
                      ctx->globals.lm_head->size);
    }
#undef ADD_GGUF_SPAN
    *count = n;
    *total = bytes;
    return n != 0;
}

static int try_pin_resident_gguf(k3_context *ctx,
                                 uint64_t budget,
                                 char *err,
                                 size_t errcap) {
    if (!ctx || !ctx->model || !ctx->model->gguf) return 1;
    const char *value = getenv("K3_GGUF_RESIDENT");
    if (value && value[0] && strcmp(value, "0") == 0) return 1;
    const int explicit_request =
        value && value[0] && strcmp(value, "auto") != 0;
    const uint64_t automatic_memory_floor =
        256ull * 1024ull * 1024ull * 1024ull;
    if (!explicit_request &&
        macos_physical_memory() < automatic_memory_floor) {
        return 1;
    }
    /*
     * File-backed GGUF spans are cheap to evict and the two 512 GiB target
     * nodes deliberately run above Metal's conservative recommendation.
     * Keep the live-memory reserve, but do not let the recommendation strand
     * an otherwise safe 400 GiB local slice.
     */
    k3_memory_status memory = {0};
    if (macos_memory_status(&memory)) {
        const uint64_t reserve = resident_memory_reserve();
        const uint64_t file_backed_budget =
            memory.reclaimable > reserve
                ? memory.reclaimable - reserve : 0;
        if (file_backed_budget > budget) budget = file_backed_budget;
    }

    uint64_t offsets[K3_N_LAYER + 5] = {0};
    uint64_t sizes[K3_N_LAYER + 5] = {0};
    uint32_t count = 0;
    uint64_t total = 0;
    if (!gguf_resident_spans(
            ctx, 1, offsets, sizes, &count, &total)) {
        set_error(err, errcap, "invalid K3 GGUF residency spans");
        return 0;
    }
    if (budget < total) {
        fprintf(stderr,
                "k3: GGUF local slice needs %.2f GiB for full residency; "
                "%.2f GiB is available, using layer-major streaming\n",
                (double)total / 1073741824.0,
                (double)budget / 1073741824.0);
        return 1;
    }
    const void *base = k3_st_map_base(ctx->globals.shard);
    const uint64_t file_size = k3_st_file_size(ctx->globals.shard);
    if (!base ||
        !k3_gpu_pin_model_spans(
            base, file_size, offsets, sizes, count)) {
        if (explicit_request) {
            set_error(err, errcap,
                      "Metal could not pin the K3 GGUF local slice");
            return 0;
        }
        fprintf(stderr,
                "k3: Metal GGUF residency is unavailable; "
                "using layer-major streaming\n");
        return 1;
    }
    ctx->gguf_resident = 1;
    ctx->resident_static_bytes = total;
    fprintf(stderr,
            "k3: pinned %u GGUF spans for layers %u:%u%s, %.3f GiB "
            "in one Metal residency set\n",
            count,
            ctx->layer_start,
            ctx->layer_end + 1u,
            ctx->has_output ? ":output" : "",
            (double)total / 1073741824.0);
    return 1;
}

/* Keep the weights touched by every token out of the SSD working set. Charge
 * the entire lazy expert cache before pinning, not just its current contents. */
static void try_pin_gguf_static(k3_context *ctx, uint64_t budget,
                                uint64_t expert_cache_bytes) {
    if (!ctx->model->gguf || ctx->gguf_resident ||
        getenv("K3_DISABLE_GGUF_STATIC_RESIDENCY") != NULL ||
        budget <= expert_cache_bytes) return;
    uint64_t offsets[K3_N_LAYER + 5] = {0};
    uint64_t sizes[K3_N_LAYER + 5] = {0};
    uint32_t count = 0;
    uint64_t total = 0;
    if (!gguf_resident_spans(ctx, 0, offsets, sizes, &count, &total) ||
        total > budget - expert_cache_bytes) return;
    const void *base = k3_st_map_base(ctx->globals.shard);
    if (!base || !k3_gpu_pin_model_spans(base,
            k3_st_file_size(ctx->globals.shard), offsets, sizes, count)) return;
    ctx->gguf_static_resident = 1;
    ctx->resident_static_bytes = total;
    fprintf(stderr, "k3: pinned %.3f GiB of static GGUF weights; "
                    "%.3f GiB reserved for streamed experts\n",
            (double)total / 1073741824.0,
            (double)expert_cache_bytes / 1073741824.0);
}

static int load_resident_lm_head(k3_context *ctx,
                                 char *err,
                                 size_t errcap) {
    const uint64_t capacity = resident_lm_head_capacity(ctx);
    if (!ctx || capacity == 0 || capacity > SIZE_MAX) {
        set_error(err, errcap, "invalid K3 resident output-head size");
        return 0;
    }
    void *data = mmap(NULL,
                      (size_t)capacity,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANON,
                      -1,
                      0);
    if (data == MAP_FAILED) {
        set_error(err, errcap,
                  "mmap K3 resident output head (%.2f GiB): %s",
                  (double)capacity / 1073741824.0,
                  strerror(errno));
        return 0;
    }
    const double start = now_seconds();
    const int fd = open_uncached_stream_fd(ctx->globals.shard, err, errcap);
    if (fd < 0 ||
        !pread_streamed_range(
            fd,
            ctx->globals.lm_head->offset,
            ctx->globals.lm_head->size,
            data,
            err,
            errcap)) {
        if (fd >= 0) close(fd);
        munmap(data, (size_t)capacity);
        return 0;
    }
    close(fd);
    ctx->resident_lm_head = (k3_static_slot){
        .data = data,
        .capacity = capacity,
        .bytes = ctx->globals.lm_head->size,
    };
    fprintf(stderr,
            "k3: resident BF16 output head ready: %.3f GiB in %.3f s\n",
            (double)ctx->globals.lm_head->size / 1073741824.0,
            now_seconds() - start);
    return 1;
}

typedef struct {
    int fd;
    uint64_t offset[2];
    uint64_t size[2];
    uint8_t *dst[2];
    uint32_t count;
    int ok;
    char error[256];
} k3_static_worker;

static void *static_worker_main(void *arg) {
    k3_static_worker *worker = arg;
    worker->ok = 1;
    for (uint32_t i = 0; i < worker->count; i++) {
        if (!pread_streamed_range(
                worker->fd,
                worker->offset[i],
                worker->size[i],
                worker->dst[i],
                worker->error,
                sizeof(worker->error))) {
            worker->ok = 0;
            break;
        }
    }
    return NULL;
}

static void discard_unused_static_pages(k3_static_slot *slot,
                                        uint64_t next_bytes) {
#if defined(MADV_DONTNEED)
    if (!slot || !slot->data || slot->bytes <= next_bytes) return;
    const long page_value = sysconf(_SC_PAGESIZE);
    if (page_value <= 0) return;
    const uint64_t page = (uint64_t)page_value;
    if (next_bytes > UINT64_MAX - (page - 1u)) return;
    const uint64_t begin = (next_bytes + page - 1u) & ~(page - 1u);
    if (begin >= slot->bytes || begin >= slot->capacity) return;
    const uint64_t used =
        slot->bytes < slot->capacity ? slot->bytes : slot->capacity;
    const uint64_t end = used & ~(page - 1u);
    if (end > begin && end - begin <= SIZE_MAX) {
        (void)madvise(
            (uint8_t *)slot->data + begin,
            (size_t)(end - begin),
            MADV_DONTNEED);
    }
#else
    (void)slot;
    (void)next_bytes;
#endif
}

static void *static_load_main(void *arg) {
    k3_static_load *load = arg;
    if (!load || !load->layer || !load->slot ||
        !load->slot->data ||
        load->size == 0 || load->size > load->slot->capacity) {
        if (load) {
            snprintf(load->error, sizeof(load->error),
                     "invalid K3 static staging load");
            load->ok = 0;
        }
        return NULL;
    }
    discard_unused_static_pages(load->slot, load->size);
    load->slot->bytes = 0;
    const double start = now_seconds();
    const int fd = open_uncached_stream_fd(
        load->layer->shard, load->error, sizeof(load->error));
    if (fd < 0) {
        load->ok = 0;
        load->seconds = now_seconds() - start;
        return NULL;
    }
    const uint32_t requested_threads = static_io_threads();
    const uint64_t min_thread_bytes = K3_PREAD_CHUNK_BYTES;
    const uint64_t logical_size = load->size - load->skip_size;
    const uint64_t prefix_size = load->skip_size
        ? load->skip_offset - load->offset : logical_size;
    uint32_t thread_count = requested_threads;
    const uint64_t useful_threads =
        (logical_size + min_thread_bytes - 1u) /
        min_thread_bytes;
    if (thread_count > useful_threads) thread_count = (uint32_t)useful_threads;
    if (thread_count == 0) thread_count = 1;

    pthread_t threads[K3_STATIC_IO_THREADS_MAX];
    k3_static_worker workers[K3_STATIC_IO_THREADS_MAX] = {0};
    uint32_t started = 0;
    load->ok = 1;
    for (; started < thread_count; started++) {
        uint64_t begin = logical_size * started / thread_count;
        uint64_t end = logical_size * (started + 1u) / thread_count;
        if (started != 0) {
            begin -= begin % K3_IO_SPLIT_ALIGNMENT;
        }
        if (started + 1u != thread_count) {
            end -= end % K3_IO_SPLIT_ALIGNMENT;
        }
        k3_static_worker *worker = &workers[started];
        worker->fd = fd;
        if (begin < prefix_size) {
            const uint64_t segment_end =
                end < prefix_size ? end : prefix_size;
            worker->offset[worker->count] = load->offset + begin;
            worker->size[worker->count] = segment_end - begin;
            worker->dst[worker->count] =
                (uint8_t *)load->slot->data + begin;
            worker->count++;
        }
        if (end > prefix_size) {
            const uint64_t logical_begin =
                begin > prefix_size ? begin : prefix_size;
            const uint64_t physical_begin =
                logical_begin + load->skip_size;
            worker->offset[worker->count] =
                load->offset + physical_begin;
            worker->size[worker->count] = end - logical_begin;
            worker->dst[worker->count] =
                (uint8_t *)load->slot->data + physical_begin;
            worker->count++;
        }
        if (thread_count == 1) {
            (void)static_worker_main(worker);
        } else {
            const int rc = pthread_create(
                &threads[started],
                NULL,
                static_worker_main,
                worker);
            if (rc != 0) {
                load->ok = 0;
                snprintf(load->error, sizeof(load->error),
                         "pthread_create for K3 static worker: %s",
                         strerror(rc));
                break;
            }
        }
    }
    if (thread_count != 1) {
        for (uint32_t i = 0; i < started; i++) {
            const int rc = pthread_join(threads[i], NULL);
            if (rc != 0 && load->ok) {
                load->ok = 0;
                snprintf(load->error, sizeof(load->error),
                         "pthread_join for K3 static worker: %s",
                         strerror(rc));
            } else if (!workers[i].ok && load->ok) {
                load->ok = 0;
                snprintf(load->error, sizeof(load->error), "%s",
                         workers[i].error[0]
                             ? workers[i].error : "K3 static read failed");
            }
        }
    }
    for (uint32_t i = 0; i < started && load->ok; i++) {
        if (!workers[i].ok) {
            load->ok = 0;
            snprintf(load->error, sizeof(load->error), "%s",
                     workers[i].error[0]
                         ? workers[i].error : "K3 static read failed");
        }
    }
    close(fd);
    load->seconds = now_seconds() - start;
    if (load->ok) load->slot->bytes = load->size;
    return NULL;
}

static int static_span_load_start(
        k3_static_load *load,
        k3_layer *layer,
        k3_static_slot *slot,
        uint64_t offset,
        uint64_t size,
        uint64_t skip_offset,
        uint64_t skip_size,
        char *err,
        size_t errcap) {
    if (!load || load->started || !layer || !slot || !slot->data ||
        size == 0 || size > slot->capacity || skip_size > size ||
        (skip_size != 0 &&
         (skip_offset < offset ||
          skip_offset - offset > size - skip_size))) {
        set_error(err, errcap, "invalid K3 static staging request");
        return 0;
    }
    *load = (k3_static_load){
        .layer = layer,
        .slot = slot,
        .offset = offset,
        .size = size,
        .skip_offset = skip_offset,
        .skip_size = skip_size,
        .ok = 0,
    };
    const int rc = pthread_create(
        &load->thread, NULL, static_load_main, load);
    if (rc != 0) {
        set_error(err, errcap,
                  "pthread_create for K3 static staging: %s",
                  strerror(rc));
        return 0;
    }
    load->started = 1;
    return 1;
}

static int static_load_start(
        k3_static_load *load,
        k3_layer *layer,
        k3_static_slot *slot,
        char *err,
        size_t errcap) {
    return static_span_load_start(
        load,
        layer,
        slot,
        layer ? layer->static_offset : 0,
        layer ? layer->static_size : 0,
        0,
        0,
        err,
        errcap);
}

static int static_load_join(
        k3_static_load *load,
        char *err,
        size_t errcap) {
    if (!load || !load->started) {
        set_error(err, errcap, "invalid K3 static staging join");
        return 0;
    }
    const int rc = pthread_join(load->thread, NULL);
    load->started = 0;
    if (rc != 0) {
        set_error(err, errcap,
                  "pthread_join for K3 static staging: %s",
                  strerror(rc));
        return 0;
    }
    if (!load->ok) {
        set_error(err, errcap, "%s",
                  load->error[0]
                      ? load->error : "K3 static staging failed");
        return 0;
    }
    return 1;
}

static void static_load_cancel(k3_static_load *load) {
    if (!load || !load->started) return;
    (void)pthread_join(load->thread, NULL);
    load->started = 0;
}

static int unsafe_residency_enabled(void) {
    const char *value = getenv("K3_UNSAFE_RESIDENT_STATIC");
    return value && value[0] && strcmp(value, "0") != 0;
}

static int resident_load_has_headroom(uint64_t next_capacity,
                                      uint64_t staging_capacity,
                                      k3_memory_status *memory) {
    if (memory) memset(memory, 0, sizeof(*memory));
    if (unsafe_residency_enabled()) return 1;
    k3_memory_status current = {0};
    if (!macos_memory_status(&current)) {
        /* The startup budget already used Metal's working-set cap when host
         * statistics were unavailable. Do not discard a valid plan because a
         * later, rate-limited statistics query failed. */
        return 1;
    }
    if (memory) *memory = current;
    uint64_t floor = resident_memory_reserve();
    if (staging_capacity > UINT64_MAX - floor) return 0;
    floor += staging_capacity;
    return current.reclaimable > floor &&
           next_capacity <= current.reclaimable - floor;
}

static void discard_unloaded_resident_slots(k3_context *ctx) {
    if (!ctx) return;
    uint64_t resident_bytes = 0;
    int resident_static = 0;
    int selective_shared = 0;
    for (uint32_t i = 0; i < K3_N_LAYER; i++) {
        k3_static_slot *slot = &ctx->resident_static_slots[i];
        if (slot->data && slot->bytes == 0) {
            munmap(slot->data, (size_t)slot->capacity);
            memset(slot, 0, sizeof(*slot));
        }
        if (slot->data) {
            resident_static = 1;
            resident_bytes += slot->capacity;
        }

        slot = &ctx->resident_shared_slots[i];
        if (slot->data && slot->bytes == 0) {
            munmap(slot->data, (size_t)slot->capacity);
            memset(slot, 0, sizeof(*slot));
        }
        if (slot->data) {
            resident_static = 1;
            selective_shared = 1;
            resident_bytes += slot->capacity;
        }
    }
    ctx->resident_static = resident_static;
    ctx->selective_shared = selective_shared;
    ctx->resident_static_bytes = resident_bytes;
}

static void demote_resident_slots_for_staging(
        k3_context *ctx,
        const k3_memory_status *memory,
        uint64_t staging_capacity) {
    if (!ctx || !memory || unsafe_residency_enabled()) return;
    uint64_t floor = resident_memory_reserve();
    if (staging_capacity > UINT64_MAX - floor) return;
    floor += staging_capacity;
    if (memory->reclaimable >= floor) return;

    const uint64_t needed = floor - memory->reclaimable;
    uint64_t released = 0;
    uint32_t layers = 0;
    for (uint32_t i = ctx->layer_end + 1u;
         i > ctx->layer_start && released < needed;
         i--) {
        k3_static_slot *slot = &ctx->resident_shared_slots[i - 1u];
        if (!slot->data) continue;
        released += slot->capacity;
        munmap(slot->data, (size_t)slot->capacity);
        memset(slot, 0, sizeof(*slot));
        layers++;
    }
    for (uint32_t i = ctx->layer_end + 1u;
         i > ctx->layer_start && released < needed;
         i--) {
        k3_static_slot *slot = &ctx->resident_static_slots[i - 1u];
        if (!slot->data) continue;
        released += slot->capacity;
        munmap(slot->data, (size_t)slot->capacity);
        memset(slot, 0, sizeof(*slot));
        layers++;
    }
    discard_unloaded_resident_slots(ctx);
    if (released != 0) {
        fprintf(stderr,
                "k3: demoted %u resident layers (%.2f GiB) to preserve "
                "streaming-slot headroom\n",
                layers,
                (double)released / 1073741824.0);
    }
}

static int load_resident_static_slots(k3_context *ctx,
                                      char *err,
                                      size_t errcap) {
    if (!ctx) {
        set_error(err, errcap, "invalid K3 resident static load");
        return 0;
    }
    uint64_t total_bytes = 0;
    uint32_t resident_full_layers = 0;
    uint32_t resident_shared_layers = 0;
    for (uint32_t i = 0; i < K3_N_LAYER; i++) {
        if (ctx->resident_static_slots[i].data) {
            const uint64_t bytes =
                ctx->layers[i]->static_size;
            if (bytes > UINT64_MAX - total_bytes) {
                set_error(err, errcap, "K3 resident static byte overflow");
                return 0;
            }
            total_bytes += bytes;
            resident_full_layers++;
        }
        if (ctx->resident_shared_slots[i].data) {
            uint64_t shared_offset = 0;
            uint64_t bytes = 0;
            if (!layer_shared_span(
                    ctx->layers[i], &shared_offset, &bytes) ||
                bytes > UINT64_MAX - total_bytes) {
                set_error(err, errcap, "K3 resident shared byte overflow");
                return 0;
            }
            total_bytes += bytes;
            resident_shared_layers++;
        }
    }
    fprintf(stderr,
            "k3: loading %.3f GiB resident static weights "
            "(full %u/%u, shared-only %u/%u) with %u I/O threads\n",
            (double)total_bytes / 1073741824.0,
            resident_full_layers,
            ctx->layer_end - ctx->layer_start + 1u,
            resident_shared_layers,
            ctx->layer_end - ctx->layer_start + (ctx->layer_start == 0 ? 0u : 1u),
            static_io_threads());
    const double start = now_seconds();
    uint64_t loaded_bytes = 0;
    uint64_t staging_capacity = 0;
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        if (!ctx->resident_static_slots[i].data) {
            staging_capacity = static_slots_capacity(ctx);
            break;
        }
    }
    int pressure_stop = 0;
    k3_memory_status pressure_memory = {0};
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        if (!ctx->resident_static_slots[i].data) continue;
        if (!resident_load_has_headroom(
                ctx->resident_static_slots[i].capacity,
                staging_capacity,
                &pressure_memory)) {
            fprintf(stderr,
                    "k3: stopping resident promotion before full layer %u "
                    "(%.2f GiB reclaimable, %.2f GiB reserve, "
                    "%.2f GiB staging reserve)\n",
                    i,
                    (double)pressure_memory.reclaimable / 1073741824.0,
                    (double)resident_memory_reserve() / 1073741824.0,
                    (double)staging_capacity / 1073741824.0);
            pressure_stop = 1;
            break;
        }
        k3_static_load load = {0};
        const uint64_t bytes =
            ctx->layers[i]->static_size;
        if (!static_span_load_start(
                &load, ctx->layers[i], &ctx->resident_static_slots[i],
                ctx->layers[i]->static_offset, bytes, 0, 0, err, errcap) ||
            !static_load_join(&load, err, errcap)) {
            static_load_cancel(&load);
            return 0;
        }
        loaded_bytes += bytes;
    }
    for (uint32_t i = ctx->layer_start;
         !pressure_stop && i <= ctx->layer_end;
         i++) {
        if (i == 0) continue;
        if (!ctx->resident_shared_slots[i].data) continue;
        uint64_t offset = 0;
        uint64_t bytes = 0;
        k3_static_load load = {0};
        if (!layer_shared_span(ctx->layers[i], &offset, &bytes)) {
            set_error(err, errcap,
                      "invalid K3 shared span for layer %u", i);
            return 0;
        }
        if (!resident_load_has_headroom(
                ctx->resident_shared_slots[i].capacity,
                staging_capacity,
                &pressure_memory)) {
            fprintf(stderr,
                    "k3: stopping resident promotion before shared layer %u "
                    "(%.2f GiB reclaimable, %.2f GiB reserve, "
                    "%.2f GiB staging reserve)\n",
                    i,
                    (double)pressure_memory.reclaimable / 1073741824.0,
                    (double)resident_memory_reserve() / 1073741824.0,
                    (double)staging_capacity / 1073741824.0);
            pressure_stop = 1;
            break;
        }
        if (
            !static_span_load_start(
                &load, ctx->layers[i], &ctx->resident_shared_slots[i],
                offset, bytes, 0, 0, err, errcap) ||
            !static_load_join(&load, err, errcap)) {
            static_load_cancel(&load);
            return 0;
        }
        loaded_bytes += bytes;
    }
    if (pressure_stop) {
        if (staging_capacity == 0) {
            staging_capacity = static_slots_capacity(ctx);
        }
        discard_unloaded_resident_slots(ctx);
        demote_resident_slots_for_staging(
            ctx, &pressure_memory, staging_capacity);
    }
    loaded_bytes = 0;
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        if (ctx->resident_static_slots[i].data) {
            loaded_bytes += ctx->layers[i]->static_size;
        }
        if (ctx->resident_shared_slots[i].data) {
            uint64_t offset = 0;
            uint64_t bytes = 0;
            if (layer_shared_span(ctx->layers[i], &offset, &bytes)) {
                loaded_bytes += bytes;
            }
        }
    }
    const double seconds = now_seconds() - start;
    fprintf(stderr,
            "k3: resident static weights ready: %.3f GiB in %.3f s "
            "(%.2f GiB/s)\n",
            (double)loaded_bytes / 1073741824.0,
            seconds,
            seconds > 0.0
                ? ((double)loaded_bytes / 1073741824.0) / seconds
                : 0.0);
    return 1;
}

static int pin_resident_static_slots(k3_context *ctx,
                                     char *err,
                                     size_t errcap) {
    const void *maps[2 * K3_N_LAYER + 1];
    uint64_t sizes[2 * K3_N_LAYER + 1];
    uint32_t count = 0;
    if (ctx->resident_lm_head.data) {
        maps[count] = ctx->resident_lm_head.data;
        sizes[count] = ctx->resident_lm_head.capacity;
        count++;
    }
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        const k3_static_slot *slot = &ctx->resident_static_slots[i];
        if (!slot->data) continue;
        maps[count] = slot->data;
        sizes[count] = slot->capacity;
        count++;
    }
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        if (i == 0) continue;
        const k3_static_slot *slot = &ctx->resident_shared_slots[i];
        if (!slot->data) continue;
        maps[count] = slot->data;
        sizes[count] = slot->capacity;
        count++;
    }
    if (count == 0 ||
        !k3_gpu_pin_static_maps(maps, sizes, count)) {
        set_error(err, errcap,
                  "Metal could not pin K3 resident BF16 layers");
        return 0;
    }
    ctx->pinned_static = 1;
    fprintf(stderr,
            "k3: pinned %u resident BF16 spans in one Metal residency set"
            "%s\n",
            count,
            ctx->resident_lm_head.data ? " (including output head)" : "");
    return 1;
}

static uint32_t q8_layer_matrices(
        const k3_layer *layer,
        const k3_st_tensor *out[K3_Q8_MAX_MATRICES]) {
    uint32_t count = 0;
#define ADD_Q8(tensor_) do { \
    if ((tensor_) != NULL && count < K3_Q8_MAX_MATRICES) { \
        out[count++] = (tensor_); \
    } \
} while (0)
    ADD_Q8(layer->g_proj);
    ADD_Q8(layer->o_proj);
    if (layer->attention == K3_LAYER_KDA) {
        ADD_Q8(layer->q_proj);
        ADD_Q8(layer->k_proj);
        ADD_Q8(layer->v_proj);
        ADD_Q8(layer->f_a_proj);
        ADD_Q8(layer->f_b_proj);
        ADD_Q8(layer->b_proj);
    } else {
        ADD_Q8(layer->q_a_proj);
        ADD_Q8(layer->q_b_proj);
        ADD_Q8(layer->kv_a_proj);
    }
    if (layer->index == 0) {
        ADD_Q8(layer->dense_gate);
        ADD_Q8(layer->dense_up);
        ADD_Q8(layer->dense_down);
    } else {
        ADD_Q8(layer->router);
        ADD_Q8(layer->routed_down);
        ADD_Q8(layer->routed_up);
        ADD_Q8(layer->shared_gate);
        ADD_Q8(layer->shared_up);
        ADD_Q8(layer->shared_down);
    }
#undef ADD_Q8
    return count;
}

static const k3_q8_weight *find_q8_weight(
        const k3_q8_layer_cache *cache,
        const k3_st_tensor *tensor) {
    if (!cache || !cache->weights || !tensor) return NULL;
    for (uint32_t i = 0; i < cache->matrix_count; i++) {
        if (cache->matrices[i].tensor == tensor) return &cache->matrices[i];
    }
    return NULL;
}

static int alloc_q8_layer_cache(k3_context *ctx,
                                uint32_t index,
                                char *err,
                                size_t errcap) {
    if (!ctx || index >= K3_N_LAYER) {
        set_error(err, errcap, "invalid K3 Q8 cache layer");
        return 0;
    }
    k3_layer *layer = ctx->layers[index];
    k3_q8_layer_cache *cache = &ctx->q8_layers[index];
    const k3_st_tensor *matrices[K3_Q8_MAX_MATRICES] = {0};
    const uint32_t count = q8_layer_matrices(layer, matrices);
    uint64_t q8_bytes = 0;
    for (uint32_t i = 0; i < count; i++) {
        const k3_st_tensor *tensor = matrices[i];
        if (tensor->dtype != K3_ST_DTYPE_BF16 || tensor->n_dims != 2 ||
            tensor->shape[1] == 0 || (tensor->shape[1] % 32u) != 0u) {
            set_error(err, errcap,
                      "K3 Q8 cache cannot quantize %s", tensor->name);
            return 0;
        }
        const uint64_t row_bytes = (tensor->shape[1] / 32u) * 34u;
        if (tensor->shape[0] > UINT64_MAX / row_bytes) {
            set_error(err, errcap, "K3 Q8 cache size overflow");
            return 0;
        }
        const uint64_t bytes = tensor->shape[0] * row_bytes;
        const uint64_t aligned = (q8_bytes + 255u) & ~255ull;
        if (aligned < q8_bytes || bytes > UINT64_MAX - aligned) {
            set_error(err, errcap, "K3 Q8 cache layout overflow");
            return 0;
        }
        cache->matrices[i] = (k3_q8_weight){
            .tensor = tensor,
            .offset = aligned,
            .bytes = bytes,
        };
        q8_bytes = aligned + bytes;
    }
    if (q8_bytes == 0) {
        set_error(err, errcap, "empty K3 Q8 layer cache");
        return 0;
    }
    cache->weights = ds4_gpu_tensor_alloc(q8_bytes);
    if (!cache->weights) {
        set_error(err, errcap,
                  "Metal allocation failed for K3 Q8 layer %u (%.2f GiB)",
                  index, (double)q8_bytes / 1073741824.0);
        return 0;
    }
    const long page_value = sysconf(_SC_PAGESIZE);
    if (page_value <= 0 ||
        layer->static_size > UINT64_MAX - ((uint64_t)page_value - 1u)) {
        set_error(err, errcap, "invalid K3 Q8 native layer size");
        return 0;
    }
    const uint64_t page = (uint64_t)page_value;
    const uint64_t capacity =
        (layer->static_size + page - 1u) & ~(page - 1u);
    void *native = mmap(NULL,
                        (size_t)capacity,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON,
                        -1,
                        0);
    if (native == MAP_FAILED) {
        set_error(err, errcap,
                  "mmap K3 Q8 native layer %u: %s",
                  index, strerror(errno));
        return 0;
    }
    cache->native = (k3_static_slot){
        .data = native,
        .capacity = capacity,
        .bytes = layer->static_size,
    };
    cache->matrix_count = count;
    cache->bytes = q8_bytes;
    return 1;
}

static inline float k3_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static inline uint16_t k3_f32_to_f16(float value) {
    const __fp16 half = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

typedef struct {
    const uint16_t *src;
    uint8_t *dst;
    uint64_t cols;
    uint64_t row_begin;
    uint64_t row_end;
} k3_q8_quant_worker;

static void *quantize_q8_worker(void *arg) {
    k3_q8_quant_worker *work = arg;
    const uint64_t blocks = work->cols / 32u;
    const uint64_t row_bytes = blocks * 34u;
    for (uint64_t row = work->row_begin; row < work->row_end; row++) {
        const uint16_t *src = work->src + row * work->cols;
        uint8_t *dst = work->dst + row * row_bytes;
        for (uint64_t block = 0; block < blocks; block++) {
            const uint16_t *values = src + block * 32u;
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32u; i++) {
                const float absolute = fabsf(k3_bf16_to_f32(values[i]));
                if (absolute > amax) amax = absolute;
            }
            const float scale = amax / 127.0f;
            const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
            const uint16_t half_scale = k3_f32_to_f16(scale);
            memcpy(dst, &half_scale, sizeof(half_scale));
            int8_t *quants = (int8_t *)(dst + sizeof(half_scale));
            for (uint32_t i = 0; i < 32u; i++) {
                int value = (int)roundf(
                    k3_bf16_to_f32(values[i]) * inverse);
                if (value > 127) value = 127;
                if (value < -127) value = -127;
                quants[i] = (int8_t)value;
            }
            dst += 34u;
        }
    }
    return NULL;
}

static int quantize_q8_matrix(const uint16_t *src,
                              uint8_t *dst,
                              uint64_t rows,
                              uint64_t cols,
                              char *err,
                              size_t errcap) {
    uint32_t thread_count = static_io_threads();
    if (thread_count > rows) thread_count = (uint32_t)rows;
    if (thread_count == 0) return 0;
    pthread_t threads[K3_STATIC_IO_THREADS_MAX];
    k3_q8_quant_worker workers[K3_STATIC_IO_THREADS_MAX];
    uint32_t started = 0;
    for (; started < thread_count; started++) {
        workers[started] = (k3_q8_quant_worker){
            .src = src,
            .dst = dst,
            .cols = cols,
            .row_begin = rows * started / thread_count,
            .row_end = rows * (started + 1u) / thread_count,
        };
        const int rc = pthread_create(
            &threads[started], NULL, quantize_q8_worker, &workers[started]);
        if (rc != 0) {
            set_error(err, errcap,
                      "pthread_create for K3 Q8 quantization: %s",
                      strerror(rc));
            break;
        }
    }
    int ok = started == thread_count;
    for (uint32_t i = 0; i < started; i++) {
        if (pthread_join(threads[i], NULL) != 0) ok = 0;
    }
    if (!ok && (!err || !err[0])) {
        set_error(err, errcap, "K3 Q8 quantization worker failed");
    }
    return ok;
}

static int q8_tensor_is_matrix(const k3_q8_layer_cache *cache,
                               const k3_st_tensor *tensor) {
    return find_q8_weight(cache, tensor) != NULL;
}

static int populate_q8_layer_cache(k3_context *ctx,
                                   uint32_t index,
                                   k3_static_slot *staging,
                                   char *err,
                                   size_t errcap) {
    k3_layer *layer = ctx->layers[index];
    k3_q8_layer_cache *cache = &ctx->q8_layers[index];
    uint8_t *q8 = ds4_gpu_tensor_contents(cache->weights);
    if (!q8 || !staging || staging->bytes != layer->static_size) {
        set_error(err, errcap, "invalid K3 Q8 cache population state");
        return 0;
    }
    for (uint32_t i = 0; i < cache->matrix_count; i++) {
        const k3_q8_weight *weight = &cache->matrices[i];
        const k3_st_tensor *tensor = weight->tensor;
        const uint64_t relative = tensor->offset - layer->static_offset;
        if (relative > staging->bytes ||
            tensor->size > staging->bytes - relative ||
            !quantize_q8_matrix(
                (const uint16_t *)((const uint8_t *)staging->data + relative),
                q8 + weight->offset,
                tensor->shape[0],
                tensor->shape[1],
                err,
                errcap)) {
            if (!err || !err[0]) {
                set_error(err, errcap,
                          "K3 Q8 quantization failed for %s", tensor->name);
            }
            return 0;
        }
    }
    const uint32_t tensor_count = k3_st_tensor_count(layer->shard);
    for (uint32_t i = 0; i < tensor_count; i++) {
        const k3_st_tensor *tensor = k3_st_tensor_at(layer->shard, i);
        if (!tensor || tensor->offset < layer->static_offset ||
            tensor->offset >= layer->static_offset + layer->static_size ||
            tensor->size >
                layer->static_offset + layer->static_size - tensor->offset ||
            q8_tensor_is_matrix(cache, tensor)) {
            continue;
        }
        const uint64_t relative = tensor->offset - layer->static_offset;
        memcpy((uint8_t *)cache->native.data + relative,
               (const uint8_t *)staging->data + relative,
               (size_t)tensor->size);
    }
#if defined(MADV_DONTNEED)
    const long page_value = sysconf(_SC_PAGESIZE);
    if (page_value > 0) {
        const uint64_t page = (uint64_t)page_value;
        const uint64_t bytes =
            (staging->bytes + page - 1u) & ~(page - 1u);
        if (bytes <= staging->capacity) {
            (void)madvise(staging->data, (size_t)bytes, MADV_DONTNEED);
        }
    }
#endif
    staging->bytes = 0;
    return 1;
}

static int load_q8_static_cache(k3_context *ctx,
                                char *err,
                                size_t errcap) {
    uint64_t total_q8 = 0;
    const double start = now_seconds();
    fprintf(stderr,
            "k3: building resident Q8_0 static cache from SafeTensors\n");
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        if (!alloc_q8_layer_cache(ctx, i, err, errcap)) return 0;
        k3_static_slot *staging = &ctx->static_slots[i & 1u];
        k3_static_load load = {0};
        if (!static_load_start(
                &load, ctx->layers[i], staging, err, errcap) ||
            !static_load_join(&load, err, errcap) ||
            !populate_q8_layer_cache(ctx, i, staging, err, errcap)) {
            static_load_cancel(&load);
            return 0;
        }
        total_q8 += ctx->q8_layers[i].bytes;
        const uint32_t loaded_layers = i - ctx->layer_start + 1u;
        const uint32_t owned_layers =
            ctx->layer_end - ctx->layer_start + 1u;
        if (loaded_layers % 8u == 0u || i == ctx->layer_end) {
            fprintf(stderr,
                    "k3: Q8 static cache %u/%u layers, %.2f GiB\n",
                    loaded_layers,
                    owned_layers,
                    (double)total_q8 / 1073741824.0);
        }
    }
    ds4_gpu_tensor *resident_weights[K3_N_LAYER] = {0};
    uint32_t resident_count = 0;
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        resident_weights[resident_count++] = ctx->q8_layers[i].weights;
    }
    if (!k3_gpu_request_static_residency(
            resident_weights, resident_count)) {
        set_error(err, errcap,
                  "Metal could not make the K3 Q8 static cache resident");
        return 0;
    }
    free_static_slots(ctx);
    ctx->q8_static = 1;
    fprintf(stderr,
            "k3: resident Q8 static cache ready: %.3f GiB in %.3f s\n",
            (double)total_q8 / 1073741824.0,
            now_seconds() - start);
    return 1;
}

static void free_q8_static_cache(k3_context *ctx) {
    if (!ctx) return;
    k3_gpu_clear_static_residency();
    ctx->pinned_static = 0;
    for (uint32_t i = 0; i < K3_N_LAYER; i++) {
        k3_q8_layer_cache *cache = &ctx->q8_layers[i];
        free_tensor(&cache->weights);
        if (cache->native.data) {
            munmap(cache->native.data, (size_t)cache->native.capacity);
        }
        memset(cache, 0, sizeof(*cache));
    }
    ctx->q8_static = 0;
}

static int streamed_tensor_matches(
        const k3_st_tensor *tensor,
        uint64_t source_base,
        uint64_t destination_offset,
        uint64_t expected_size) {
    return tensor &&
           destination_offset <= UINT64_MAX - source_base &&
           tensor->offset == source_base + destination_offset &&
           tensor->size == expected_size;
}

static int load_streamed_expert(
        k3_layer *layer,
        int fd,
        uint32_t expert_id,
        k3_streamed_expert *slot,
        char *err,
        size_t errcap) {
    if (!layer || expert_id >= K3_N_EXPERT || !slot || !slot->weights) {
        set_error(err, errcap, "invalid K3 streamed expert load");
        return 0;
    }
    uint8_t *dst = ds4_gpu_tensor_contents(slot->weights);
    if (!dst || ds4_gpu_tensor_bytes(slot->weights) != slot->bytes) {
        set_error(err, errcap, "K3 streamed expert buffer is not writable");
        return 0;
    }
    const k3_expert_weights *expert = &layer->experts[expert_id];
    const uint64_t w1_packed_bytes = slot->w1_scale - slot->w1_packed;
    const uint64_t w1_scale_bytes = slot->w2_packed - slot->w1_scale;
    const uint64_t w2_packed_bytes = slot->w2_scale - slot->w2_packed;
    const uint64_t w2_scale_bytes = slot->w3_packed - slot->w2_scale;
    const uint64_t source_base = expert->w1_packed->offset;
    if (!streamed_tensor_matches(
            expert->w1_packed, source_base,
            slot->w1_packed, w1_packed_bytes) ||
        !streamed_tensor_matches(
            expert->w1_scale, source_base,
            slot->w1_scale, w1_scale_bytes) ||
        !streamed_tensor_matches(
            expert->w2_packed, source_base,
            slot->w2_packed, w2_packed_bytes) ||
        !streamed_tensor_matches(
            expert->w2_scale, source_base,
            slot->w2_scale, w2_scale_bytes) ||
        !streamed_tensor_matches(
            expert->w3_packed, source_base,
            slot->w3_packed, w1_packed_bytes) ||
        !streamed_tensor_matches(
            expert->w3_scale, source_base,
            slot->w3_scale, w1_scale_bytes)) {
        set_error(err, errcap,
                  "K3 expert %u tensors are not one contiguous range",
                  expert_id);
        return 0;
    }
    return pread_streamed_range(
        fd, source_base, slot->bytes, dst, err, errcap);
}

static void *expert_io_pool_worker(void *arg) {
    k3_expert_io_pool *pool = arg;
    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (!pool->stopping &&
               pool->high_count == 0 &&
               pool->low_count == 0) {
            pthread_cond_wait(&pool->ready, &pool->mutex);
        }
        if (pool->stopping) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        k3_expert_io_task task;
        if (pool->high_count != 0) {
            task = pool->high[0];
            pool->high_count--;
            if (pool->high_count != 0) {
                memmove(
                    &pool->high[0],
                    &pool->high[1],
                    pool->high_count * sizeof(pool->high[0]));
            }
        } else {
            task = pool->low[0];
            pool->low_count--;
            if (pool->low_count != 0) {
                memmove(
                    &pool->low[0],
                    &pool->low[1],
                    pool->low_count * sizeof(pool->low[0]));
            }
        }
        pthread_mutex_unlock(&pool->mutex);

        k3_stream_work *work = &task.batch->work;
        pthread_mutex_lock(&work->mutex);
        const int should_load = work->ok;
        pthread_mutex_unlock(&work->mutex);

        char local_error[256] = {0};
        const int loaded = should_load && load_streamed_expert(
                work->layer,
                work->fd,
                work->groups[task.index].expert,
                &work->slots[task.index],
                local_error,
                sizeof(local_error));
        pthread_mutex_lock(&work->mutex);
        if (loaded) {
            work->completed[task.index] = 1;
            if (work->bytes_read <=
                UINT64_MAX - work->slots[task.index].bytes) {
                work->bytes_read += work->slots[task.index].bytes;
            } else {
                work->bytes_read = UINT64_MAX;
            }
        } else if (should_load) {
            if (work->ok) {
                work->ok = 0;
                snprintf(work->error, sizeof(work->error), "%s",
                         local_error[0] ? local_error
                                        : "streamed expert load failed");
            }
        }
        if (work->remaining != 0) work->remaining--;
        pthread_cond_broadcast(&work->cond);
        pthread_mutex_unlock(&work->mutex);
    }
    return NULL;
}

static int expert_io_pool_init(k3_context *ctx,
                               char *err,
                               size_t errcap) {
    if (!ctx) {
        set_error(err, errcap, "invalid K3 expert I/O pool");
        return 0;
    }
    k3_expert_io_pool *pool = &ctx->expert_io_pool;
    if (pool->initialized) return 1;
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        set_error(err, errcap,
                  "could not initialize K3 expert I/O pool");
        return 0;
    }
    if (pthread_cond_init(&pool->ready, NULL) != 0) {
        pthread_mutex_destroy(&pool->mutex);
        set_error(err, errcap,
                  "could not initialize K3 expert I/O pool");
        return 0;
    }
    pool->thread_count = expert_io_threads();
    pool->initialized = 1;
    for (; pool->started < pool->thread_count; pool->started++) {
        const int rc = pthread_create(
            &pool->threads[pool->started],
            NULL,
            expert_io_pool_worker,
            pool);
        if (rc != 0) {
            set_error(err, errcap,
                      "pthread_create for K3 expert I/O pool: %s",
                      strerror(rc));
            pthread_mutex_lock(&pool->mutex);
            pool->stopping = 1;
            pthread_cond_broadcast(&pool->ready);
            pthread_mutex_unlock(&pool->mutex);
            for (uint32_t i = 0; i < pool->started; i++) {
                (void)pthread_join(pool->threads[i], NULL);
            }
            pthread_cond_destroy(&pool->ready);
            pthread_mutex_destroy(&pool->mutex);
            memset(pool, 0, sizeof(*pool));
            return 0;
        }
    }
    return 1;
}

static void expert_io_pool_shutdown(k3_context *ctx) {
    if (!ctx || !ctx->expert_io_pool.initialized) return;
    k3_expert_io_pool *pool = &ctx->expert_io_pool;
    pthread_mutex_lock(&pool->mutex);
    pool->stopping = 1;
    pthread_cond_broadcast(&pool->ready);
    pthread_mutex_unlock(&pool->mutex);
    for (uint32_t i = 0; i < pool->started; i++) {
        (void)pthread_join(pool->threads[i], NULL);
    }
    pthread_cond_destroy(&pool->ready);
    pthread_mutex_destroy(&pool->mutex);
    memset(pool, 0, sizeof(*pool));
}

static int expert_io_pool_enqueue(
        k3_expert_io_pool *pool,
        k3_stream_batch *batch,
        uint32_t count,
        int low_priority,
        char *err,
        size_t errcap) {
    if (!pool || !pool->initialized || !batch || count == 0) {
        set_error(err, errcap, "invalid K3 expert I/O submission");
        return 0;
    }
    pthread_mutex_lock(&pool->mutex);
    k3_expert_io_task *queue =
        low_priority ? pool->low : pool->high;
    uint32_t *queued =
        low_priority ? &pool->low_count : &pool->high_count;
    if (count > K3_EXPERT_IO_QUEUE_CAPACITY - *queued) {
        pthread_mutex_unlock(&pool->mutex);
        set_error(err, errcap, "K3 expert I/O queue is full");
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) {
        queue[(*queued)++] = (k3_expert_io_task){
            .batch = batch,
            .index = i,
        };
    }
    pthread_cond_broadcast(&pool->ready);
    pthread_mutex_unlock(&pool->mutex);
    return 1;
}

static int expert_io_pool_resolve_speculation(
        k3_context *ctx,
        k3_stream_batch *batch,
        const uint8_t needed[K3_STREAMED_EXPERT_SLOTS],
        uint32_t count,
        char *err,
        size_t errcap) {
    if (!ctx || !batch || !batch->initialized || !needed ||
        count > batch->work.count) {
        set_error(err, errcap,
                  "invalid K3 speculative I/O resolution");
        return 0;
    }
    k3_expert_io_pool *pool = &execution_owner(ctx)->expert_io_pool;
    uint32_t canceled_count = 0;
    uint8_t canceled[K3_STREAMED_EXPERT_SLOTS] = {0};
    pthread_mutex_lock(&pool->mutex);
    uint32_t promote_count = 0;
    for (uint32_t i = 0; i < pool->low_count; i++) {
        const k3_expert_io_task task = pool->low[i];
        if (task.batch == batch && task.index < count &&
            needed[task.index]) {
            promote_count++;
        }
    }
    if (promote_count >
        K3_EXPERT_IO_QUEUE_CAPACITY - pool->high_count) {
        pthread_mutex_unlock(&pool->mutex);
        set_error(err, errcap,
                  "K3 expert I/O priority queue is full");
        return 0;
    }
    for (uint32_t i = 0; i < pool->low_count;) {
        const k3_expert_io_task task = pool->low[i];
        if (task.batch != batch || task.index >= count) {
            i++;
            continue;
        }
        pool->low_count--;
        if (i < pool->low_count) {
            memmove(
                &pool->low[i],
                &pool->low[i + 1u],
                (pool->low_count - i) * sizeof(pool->low[0]));
        }
        if (needed[task.index]) {
            pool->high[pool->high_count++] = task;
        } else {
            canceled[task.index] = 1;
            canceled_count++;
        }
    }
    pthread_cond_broadcast(&pool->ready);
    pthread_mutex_unlock(&pool->mutex);

    if (canceled_count != 0) {
        pthread_mutex_lock(&batch->work.mutex);
        if (batch->work.remaining < canceled_count) {
            fprintf(stderr,
                    "k3: speculative cancellation accounting underflow\n");
            abort();
        }
        batch->work.remaining -= canceled_count;
        for (uint32_t i = 0; i < count; i++) {
            if (canceled[i]) batch->work.canceled[i] = 1;
        }
        pthread_cond_broadcast(&batch->work.cond);
        pthread_mutex_unlock(&batch->work.mutex);
    }
    return 1;
}

static int start_streamed_experts(
        k3_context *ctx,
        k3_layer *layer,
        const k3_expert_group *groups,
        uint32_t count,
        k3_streamed_expert slots[K3_STREAMED_EXPERT_SLOTS],
        int low_priority,
        k3_stream_batch *batch,
        char *err,
        size_t errcap) {
    if (!ctx || !layer || !groups || !slots || !batch || count == 0 ||
        count > K3_STREAMED_EXPERT_SLOTS) {
        set_error(err, errcap, "invalid K3 streamed expert batch");
        return 0;
    }
    const int fd = ctx->expert_fds[layer->index];
    if (fd < 0) {
        set_error(err, errcap,
                  "K3 layer %u has no expert streaming descriptor",
                  layer->index);
        return 0;
    }
    memset(batch, 0, sizeof(*batch));
    batch->work = (k3_stream_work){
        .layer = layer,
        .fd = fd,
        .groups = groups,
        .slots = slots,
        .count = count,
        .remaining = count,
        .ok = 1,
    };
    if (pthread_mutex_init(&batch->work.mutex, NULL) != 0) {
        set_error(err, errcap, "could not initialize K3 stream mutex");
        return 0;
    }
    if (pthread_cond_init(&batch->work.cond, NULL) != 0) {
        pthread_mutex_destroy(&batch->work.mutex);
        set_error(err, errcap, "could not initialize K3 stream condition");
        return 0;
    }
    batch->initialized = 1;
    if (!expert_io_pool_enqueue(
            &execution_owner(ctx)->expert_io_pool,
            batch,
            count,
            low_priority,
            err,
            errcap)) {
        pthread_cond_destroy(&batch->work.cond);
        pthread_mutex_destroy(&batch->work.mutex);
        batch->initialized = 0;
        return 0;
    }
    return 1;
}

static int wait_streamed_expert(k3_stream_batch *batch,
                                uint32_t index,
                                char *err,
                                size_t errcap) {
    if (!batch || !batch->initialized || index >= batch->work.count) {
        set_error(err, errcap, "invalid K3 streamed expert wait");
        return 0;
    }
    pthread_mutex_lock(&batch->work.mutex);
    while (batch->work.ok && !batch->work.completed[index] &&
           !batch->work.canceled[index]) {
        const int rc =
            pthread_cond_wait(&batch->work.cond, &batch->work.mutex);
        if (rc != 0) {
            batch->work.ok = 0;
            snprintf(batch->work.error, sizeof(batch->work.error),
                     "pthread_cond_wait for K3 expert stream: %s",
                     strerror(rc));
        }
    }
    const int ok = batch->work.ok && batch->work.completed[index] &&
                   !batch->work.canceled[index];
    if (!ok) {
        set_error(err, errcap, "%s",
                  batch->work.error[0]
                      ? batch->work.error : "K3 expert stream failed");
    }
    pthread_mutex_unlock(&batch->work.mutex);
    return ok;
}

static int finish_streamed_experts(k3_stream_batch *batch,
                                   char *err,
                                   size_t errcap) {
    if (!batch || !batch->initialized) return 1;
    pthread_mutex_lock(&batch->work.mutex);
    while (batch->work.remaining != 0) {
        const int rc =
            pthread_cond_wait(&batch->work.cond, &batch->work.mutex);
        if (rc != 0) {
            batch->work.ok = 0;
            snprintf(batch->work.error, sizeof(batch->work.error),
                     "pthread_cond_wait for K3 expert stream: %s",
                     strerror(rc));
            /*
             * Do not destroy this synchronization state while pool workers
             * still own batch pointers. Mark queued work failed, then poll
             * until every claimed or queued task has retired.
             */
            pthread_mutex_unlock(&batch->work.mutex);
            const struct timespec delay = {
                .tv_sec = 0,
                .tv_nsec = 1000000,
            };
            (void)nanosleep(&delay, NULL);
            pthread_mutex_lock(&batch->work.mutex);
        }
    }
    const int ok = batch->work.ok && batch->work.remaining == 0;
    if (!ok && (!err || !err[0])) {
        set_error(err, errcap, "%s",
                  batch->work.error[0]
                      ? batch->work.error : "K3 expert stream join failed");
    }
    pthread_mutex_unlock(&batch->work.mutex);
    pthread_cond_destroy(&batch->work.cond);
    pthread_mutex_destroy(&batch->work.mutex);
    batch->initialized = 0;
    return ok;
}

static int finish_speculative_bank(k3_context *ctx,
                                   uint32_t bank,
                                   char *err,
                                   size_t errcap) {
    if (!ctx || bank >= K3_SPECULATIVE_BANKS ||
        !ctx->speculative_batch[bank] ||
        !ctx->speculative_batch[bank]->initialized) {
        return 1;
    }
    const int ok = finish_streamed_experts(
        ctx->speculative_batch[bank], err, errcap);
    if (ctx->speculative_stream_bytes <=
        UINT64_MAX - ctx->speculative_batch[bank]->work.bytes_read) {
        ctx->speculative_stream_bytes +=
            ctx->speculative_batch[bank]->work.bytes_read;
    } else {
        ctx->speculative_stream_bytes = UINT64_MAX;
    }
    if (ctx->speculative_started[bank] != 0.0) {
        ctx->speculative_stream_seconds +=
            now_seconds() - ctx->speculative_started[bank];
    }
    ctx->speculative_started[bank] = 0.0;
    return ok;
}

static int finish_all_speculative_experts(k3_context *ctx,
                                          char *err,
                                          size_t errcap) {
    int ok = 1;
    for (uint32_t bank = 0; bank < K3_SPECULATIVE_BANKS; bank++) {
        if (!finish_speculative_bank(ctx, bank, err, errcap)) ok = 0;
        ctx->speculative_layer[bank] = UINT32_MAX;
        ctx->speculative_count[bank] = 0;
    }
    return ok;
}

static int start_speculative_experts(k3_context *ctx,
                                     uint32_t layer_index,
                                     char *err,
                                     size_t errcap) {
    const uint32_t bank = layer_index % K3_SPECULATIVE_BANKS;
    if (!ctx || !ctx->speculative_enabled ||
        !ctx->speculative_batch[bank] || layer_index >= K3_N_LAYER ||
        !ctx->predicted_route_valid[layer_index]) {
        return 1;
    }
    if (ctx->speculative_batch[bank]->initialized) {
        set_error(err, errcap,
                  "K3 speculative expert bank %u is still active", bank);
        return 0;
    }
    /*
     * The target bank was consumed by the previous layer. Shared-event router
     * readback is not a resource-lifetime fence for asynchronously flushed
     * Metal command buffers, so drain them before the CPU overwrites the bank.
     * A new command buffer is opened before I/O starts; the future read still
     * overlaps all exact expert work in the current layer.
     */
    if (!ds4_gpu_synchronize() || !ds4_gpu_begin_commands()) {
        set_error(err, errcap,
                  "K3 speculative expert refill fence failed");
        return 0;
    }
    const uint32_t count = speculative_top_k();
    for (uint32_t i = 0; i < count; i++) {
        const int32_t expert = ctx->predicted_route[layer_index][i];
        if (expert < 0 || expert >= K3_N_EXPERT) {
            set_error(err, errcap,
                      "K3 speculative route contains invalid expert %d",
                      expert);
            return 0;
        }
        ctx->speculative_groups[bank][i] = (k3_expert_group){
            .expert = (uint32_t)expert,
            .begin = 0,
            .count = 1,
        };
    }
    if (!start_streamed_experts(
            ctx,
            ctx->layers[layer_index],
            ctx->speculative_groups[bank],
            count,
            ctx->speculative_experts[bank],
            1,
            ctx->speculative_batch[bank],
            err,
            errcap)) {
        return 0;
    }
    ctx->speculative_layer[bank] = layer_index;
    ctx->speculative_count[bank] = count;
    ctx->speculative_started[bank] = now_seconds();
    return 1;
}

static int map_global_tensors(k3_context *ctx,
                              int embedding,
                              char *err,
                              size_t errcap) {
    if (!ctx) {
        set_error(err, errcap, "invalid K3 global tensor mapping request");
        return 0;
    }
    if (ctx->gguf_resident || ctx->gguf_static_resident) return 1;
    const k3_global_weights *g = &ctx->globals;
    if (embedding) {
        return map_range(
            g->shard,
            g->token_embedding->offset,
            g->token_embedding->size,
            g->token_embedding->size,
            err,
            errcap);
    }
    const k3_st_tensor *tensors[] = {
        g->output_attn_res_norm,
        g->output_attn_res_proj,
        g->norm,
        g->lm_head,
    };
    uint64_t offsets[4], sizes[4], max_tensor = 1;
    const uint32_t count = ctx->resident_lm_head.data ? 3u : 4u;
    for (uint32_t i = 0; i < count; i++) {
        offsets[i] = tensors[i]->offset;
        sizes[i] = tensors[i]->size;
        if (sizes[i] > max_tensor) max_tensor = sizes[i];
    }
    return map_spans(
        g->shard, offsets, sizes, count, max_tensor, err, errcap);
}

static int matmul(ds4_gpu_tensor *out,
                  const k3_st_tensor *weight,
                  const ds4_gpu_tensor *x,
                  uint32_t in_dim,
                  uint32_t out_dim,
                  uint32_t rows,
                  const k3_weight_source *source) {
    if (!weight || !source) {
        return 0;
    }
    const k3_q8_layer_cache *q8 = source->q8;
    if (q8) {
        const k3_q8_weight *cached = find_q8_weight(q8, weight);
        if (cached) {
            return k3_gpu_matmul_q8_0_tensor(
                out,
                q8->weights,
                cached->offset,
                in_dim,
                out_dim,
                x,
                rows);
        }
    }
    if (!source->base || weight->offset < source->bias) return 0;
    const uint64_t offset = weight->offset - source->bias;
    switch (weight->dtype) {
    case K3_ST_DTYPE_BF16:
        return k3_gpu_matmul_bf16(
            out, source->base, source->size, offset,
            in_dim, out_dim, x, rows);
    case K3_ST_DTYPE_Q8_0:
        return ds4_gpu_matmul_quant_tensor(
            out, source->base, source->size, offset,
            DS4_GPU_GGML_TYPE_Q8_0,
            in_dim, out_dim, x, rows);
    case K3_ST_DTYPE_MXFP4:
        return ds4_gpu_matmul_quant_tensor(
            out, source->base, source->size, offset,
            DS4_GPU_GGML_TYPE_MXFP4,
            in_dim, out_dim, x, rows);
    default:
        return 0;
    }
}

static int rms_norm(ds4_gpu_tensor *out,
                    const ds4_gpu_tensor *x,
                    const k3_st_tensor *weight,
                    uint32_t width,
                    uint32_t rows,
                    const k3_weight_source *source) {
    if (!weight || !source || !source->base ||
        weight->offset < source->bias) {
        return 0;
    }
    return k3_gpu_rms_norm_bf16(
        out,
        x,
        source->base,
        source->size,
        weight->offset - source->bias,
        width,
        rows,
        K3_RMS_EPS);
}

static uint64_t source_offset(
        const k3_weight_source *source,
        const k3_st_tensor *tensor) {
    if (!source || !tensor || tensor->offset < source->bias) {
        return UINT64_MAX;
    }
    return tensor->offset - source->bias;
}

static int predict_next_route(k3_context *ctx,
                              const k3_layer *layer,
                              uint32_t rows,
                              char *err,
                              size_t errcap) {
    if (!route_prediction_enabled(ctx) || rows != 1u ||
        !ctx || !layer || layer->index + 1u >= K3_N_LAYER) {
        return 1;
    }
    const uint32_t next_index = layer->index + 1u;
    /*
     * At an AttnRes block boundary the next attention output replaces prefix,
     * so the pre-attention state is not a useful router predictor.
     */
    if ((next_index % K3_ATTN_RES_BLOCK) == 0u) return 1;
    const k3_static_slot *slot =
        &ctx->resident_static_slots[next_index];
    if (!ctx->pinned_static || !slot->data) return 1;

    const k3_layer *next = ctx->layers[next_index];
    const k3_weight_source source = {
        .base = slot->data,
        .size = slot->capacity,
        .bias = next->static_offset,
    };
    const uint32_t valid_blocks =
        (next_index + K3_ATTN_RES_BLOCK - 1u) / K3_ATTN_RES_BLOCK;
    uint64_t ready = 0;
    /*
     * Prefix and all saved blocks are current here. Approximate the next
     * ffn_in with the next layer's exact AttnRes mixture, omitting only the
     * current FFN delta and the next attention delta. attn_in is dead after
     * the current attention and will be overwritten by the next layer.
     */
    if (!k3_gpu_attn_res(
            ctx->scratch.attn_in,
            ctx->scratch.prefix,
            NULL,
            ctx->scratch.blocks,
            source.base,
            source.size,
            source_offset(&source, next->mlp_res_norm),
            source_offset(&source, next->mlp_res_proj),
            source_offset(&source, next->post_attention_norm),
            K3_N_EMBD,
            1u,
            K3_ATTN_RES_BLOCKS,
            valid_blocks,
            -1,
            1,
            K3_RMS_EPS,
            K3_RMS_EPS) ||
        !matmul(
            ctx->scratch.predicted_router_logits,
            next->router,
            ctx->scratch.attn_in,
            K3_N_EMBD,
            K3_N_EXPERT,
            1u,
            &source) ||
        !k3_gpu_router_topk(
            ctx->scratch.predicted_ids,
            ctx->scratch.predicted_weights,
            ctx->scratch.predicted_router_logits,
            source.base,
            source.size,
            source_offset(&source, next->router_bias),
            1u,
            K3_N_EXPERT,
            K3_N_EXPERT_USED) ||
        !ds4_gpu_signal_selected_readback_ready(&ready) ||
        !ds4_gpu_commit_and_wait_selected_readback(
            ready, "K3 speculative router readback")) {
        set_error(err, errcap,
                  "K3 layer %u speculative next-route evaluation failed",
                  layer->index);
        return 0;
    }
    const int32_t *ids =
        ds4_gpu_tensor_contents(ctx->scratch.predicted_ids);
    if (!ids) {
        set_error(err, errcap,
                  "K3 speculative route is not CPU-visible");
        return 0;
    }
    for (uint32_t i = 0; i < K3_N_EXPERT_USED; i++) {
        if (ids[i] < 0 || ids[i] >= K3_N_EXPERT) {
            set_error(err, errcap,
                      "K3 speculative router produced invalid expert %d",
                      ids[i]);
            return 0;
        }
    }
    memcpy(ctx->predicted_route[next_index],
           ids,
           K3_N_EXPERT_USED * sizeof(*ids));
    ctx->predicted_route_valid[next_index] = 1;
    return 1;
}

static int run_attention(k3_context *ctx,
                         const k3_layer *layer,
                         const k3_weight_source *source,
                         uint32_t rows,
                         int prefill) {
    k3_scratch *s = &ctx->scratch;
    const void *base = source->base;
    const uint64_t size = source->size;

    if (layer->attention == K3_LAYER_KDA) {
        int qkvg_ok = 0;
        if (rows <= 8u && !source->q8 &&
            layer->g_proj->dtype == K3_ST_DTYPE_BF16 &&
            layer->q_proj->dtype == K3_ST_DTYPE_BF16 &&
            layer->k_proj->dtype == K3_ST_DTYPE_BF16 &&
            layer->v_proj->dtype == K3_ST_DTYPE_BF16 &&
            layer->g_proj->offset >= source->bias &&
            layer->q_proj->offset >= source->bias &&
            layer->k_proj->offset >= source->bias &&
            layer->v_proj->offset >= source->bias &&
            k3_gpu_matmul4_bf16_enabled()) {
            ds4_gpu_tensor *outputs[4] = {
                s->output_gate, s->q, s->k, s->v,
            };
            const uint64_t offsets[4] = {
                layer->g_proj->offset - source->bias,
                layer->q_proj->offset - source->bias,
                layer->k_proj->offset - source->bias,
                layer->v_proj->offset - source->bias,
            };
            qkvg_ok = k3_gpu_matmul4_bf16(
                outputs, base, size, offsets,
                K3_N_EMBD, K3_KDA_PROJ, s->attn_in, rows);
            if (!qkvg_ok) return 0;
        } else {
            qkvg_ok =
                matmul(s->output_gate, layer->g_proj, s->attn_in,
                       K3_N_EMBD, K3_KDA_PROJ, rows, source) &&
                matmul(s->q, layer->q_proj, s->attn_in,
                       K3_N_EMBD, K3_KDA_PROJ, rows, source) &&
                matmul(s->k, layer->k_proj, s->attn_in,
                       K3_N_EMBD, K3_KDA_PROJ, rows, source) &&
                matmul(s->v, layer->v_proj, s->attn_in,
                       K3_N_EMBD, K3_KDA_PROJ, rows, source);
        }
        if (!qkvg_ok ||
            !matmul(s->low_rank, layer->f_a_proj, s->attn_in,
                    K3_N_EMBD, K3_HEAD_DIM, rows, source) ||
            !matmul(s->raw_gate, layer->f_b_proj, s->low_rank,
                    K3_HEAD_DIM, K3_KDA_PROJ, rows, source) ||
            !matmul(s->beta, layer->b_proj, s->attn_in,
                    K3_N_EMBD, K3_N_HEAD, rows, source)) {
            return 0;
        }
        const int ok = prefill
            ? k3_gpu_kda_prefill(
                  s->attn_heads,
                  ctx->kda[layer->index].conv,
                  ctx->kda[layer->index].recurrent,
                  s->q,
                  s->k,
                  s->v,
                  s->raw_gate,
                  s->beta,
                  s->output_gate,
                  base,
                  size,
                  source_offset(source, layer->q_conv),
                  source_offset(source, layer->k_conv),
                  source_offset(source, layer->v_conv),
                  source_offset(source, layer->a_log),
                  source_offset(source, layer->dt_bias),
                  source_offset(source, layer->o_norm),
                  K3_N_HEAD,
                  rows,
                  K3_GATE_LOWER_BOUND,
                  K3_RMS_EPS)
            : k3_gpu_kda_decode(
                  s->attn_heads,
                  ctx->kda[layer->index].conv,
                  ctx->kda[layer->index].recurrent,
                  s->q,
                  s->k,
                  s->v,
                  s->raw_gate,
                  s->beta,
                  s->output_gate,
                  base,
                  size,
                  source_offset(source, layer->q_conv),
                  source_offset(source, layer->k_conv),
                  source_offset(source, layer->v_conv),
                  source_offset(source, layer->a_log),
                  source_offset(source, layer->dt_bias),
                  source_offset(source, layer->o_norm),
                  K3_N_HEAD,
                  rows,
                  K3_GATE_LOWER_BOUND,
                  K3_RMS_EPS);
        if (!ok) return 0;
    } else {
        if (!matmul(s->output_gate, layer->g_proj, s->attn_in,
                    K3_N_EMBD, K3_KDA_PROJ, rows, source) ||
            !matmul(s->low_rank, layer->q_a_proj, s->attn_in,
                    K3_N_EMBD, K3_Q_LORA, rows, source) ||
            !rms_norm(s->low_rank_norm, s->low_rank, layer->q_a_norm,
                      K3_Q_LORA, rows, source) ||
            !matmul(s->q, layer->q_b_proj, s->low_rank_norm,
                    K3_Q_LORA, K3_MLA_Q_WIDTH, rows, source) ||
            !matmul(s->kv, layer->kv_a_proj, s->attn_in,
                    K3_N_EMBD, K3_MLA_CACHE_WIDTH, rows, source) ||
            !k3_gpu_rms_norm_bf16_strided(
                s->kv,
                s->kv,
                base,
                size,
                source_offset(source, layer->kv_a_norm),
                K3_KV_LORA,
                rows,
                K3_MLA_CACHE_WIDTH,
                K3_RMS_EPS)) {
            return 0;
        }
        const uint32_t max_partials = (uint32_t)(
            ((uint64_t)ctx->context_capacity + 1023ull) / 1024ull);
        const int ok = prefill
            ? k3_gpu_mla_prefill(
                  s->attn_heads,
                  ctx->mla[layer->index].cache,
                  s->q_latent,
                  s->q,
                  s->kv,
                  s->output_gate,
                  base,
                  size,
                  source_offset(source, layer->kv_b_proj),
                  K3_N_HEAD,
                  ctx->position,
                  rows,
                  ctx->context_capacity)
            : k3_gpu_mla_decode(
                  s->attn_heads,
                  ctx->mla[layer->index].cache,
                  s->q_latent,
                  s->partial_values,
                  s->partial_stats,
                  s->q,
                  s->kv,
                  s->output_gate,
                  base,
                  size,
                  source_offset(source, layer->kv_b_proj),
                  K3_N_HEAD,
                  ctx->position,
                  ctx->context_capacity,
                  max_partials);
        if (!ok) return 0;
    }

    return matmul(s->attn_out, layer->o_proj, s->attn_heads,
                  K3_KDA_PROJ, K3_N_EMBD, rows, source);
}

static int run_dense_ffn(k3_context *ctx,
                         const k3_layer *layer,
                         const k3_weight_source *source,
                         uint32_t rows) {
    k3_scratch *s = &ctx->scratch;
    if (!matmul(s->dense_gate, layer->dense_gate, s->ffn_in,
                K3_N_EMBD, K3_DENSE_FF, rows, source) ||
        !matmul(s->dense_up, layer->dense_up, s->ffn_in,
                K3_N_EMBD, K3_DENSE_FF, rows, source) ||
        !k3_gpu_situ(s->dense_mid, s->dense_gate, s->dense_up,
                     K3_DENSE_FF, rows, 4.0f, 25.0f) ||
        !matmul(s->delta, layer->dense_down, s->dense_mid,
                K3_DENSE_FF, K3_N_EMBD, rows, source)) {
        return 0;
    }
    return 1;
}

static int build_expert_groups(
        k3_context *ctx,
        uint32_t layer_index,
        uint32_t rows,
        k3_expert_group groups[K3_N_EXPERT],
        uint32_t *group_count,
        char *err,
        size_t errcap) {
    const uint64_t pairs = (uint64_t)rows * K3_N_EXPERT_USED;
    int32_t *selected =
        ds4_gpu_tensor_contents(ctx->scratch.selected_ids);
    float *weights =
        ds4_gpu_tensor_contents(ctx->scratch.selected_weights);
    int32_t *group_rows =
        ds4_gpu_tensor_contents(ctx->scratch.group_row_ids);
    float *group_weights =
        ds4_gpu_tensor_contents(ctx->scratch.group_weights);
    uint32_t counts[K3_N_EXPERT] = {0};
    uint32_t cursor[K3_N_EXPERT] = {0};
    if (!selected || !weights || !group_rows || !group_weights) {
        set_error(err, errcap,
                  "K3 router buffers are not CPU-visible");
        return 0;
    }
    for (uint64_t i = 0; i < pairs; i++) {
        if (selected[i] < 0 || selected[i] >= K3_N_EXPERT ||
            !isfinite(weights[i])) {
            set_error(err, errcap, "K3 router produced invalid expert data");
            return 0;
        }
        counts[(uint32_t)selected[i]]++;
    }
    if (rows == 1u && layer_index < K3_N_LAYER &&
        ctx->predicted_route_valid[layer_index]) {
        for (uint32_t j = 0; j < K3_N_EXPERT_USED; j++) {
            for (uint32_t i = 0; i < K3_N_EXPERT_USED; i++) {
                if (ctx->predicted_route[layer_index][j] == selected[i]) {
                    ctx->route_prediction_rank_hits[j]++;
                    break;
                }
            }
        }
        ctx->route_prediction_transitions++;
        for (uint32_t i = 0; i < K3_N_EXPERT_USED; i++) {
            for (uint32_t j = 0; j < K3_N_EXPERT_USED; j++) {
                if (selected[i] == ctx->predicted_route[layer_index][j]) {
                    ctx->route_prediction_hits++;
                    break;
                }
            }
        }
        ctx->route_prediction_total += K3_N_EXPERT_USED;
        ctx->predicted_route_valid[layer_index] = 0;
    }
    if (rows == 1u && layer_index < K3_N_LAYER) {
        if (ctx->previous_route_valid[layer_index]) {
            for (uint32_t i = 0; i < K3_N_EXPERT_USED; i++) {
                for (uint32_t j = 0; j < K3_N_EXPERT_USED; j++) {
                    if (selected[i] ==
                        ctx->previous_route[layer_index][j]) {
                        ctx->route_overlap++;
                        break;
                    }
                }
            }
            ctx->route_total += K3_N_EXPERT_USED;
        }
        memcpy(ctx->previous_route[layer_index],
               selected,
               K3_N_EXPERT_USED * sizeof(*selected));
        ctx->previous_route_valid[layer_index] = 1;
    }
    uint32_t n_groups = 0;
    uint32_t offset = 0;
    for (uint32_t e = 0; e < K3_N_EXPERT; e++) {
        if (!counts[e]) continue;
        groups[n_groups++] = (k3_expert_group){
            .expert = e,
            .begin = offset,
            .count = counts[e],
        };
        cursor[e] = offset;
        offset += counts[e];
    }
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t k = 0; k < K3_N_EXPERT_USED; k++) {
            const uint64_t pair = (uint64_t)row * K3_N_EXPERT_USED + k;
            const uint32_t expert = (uint32_t)selected[pair];
            const uint32_t dst = cursor[expert]++;
            group_rows[dst] = (int32_t)row;
            group_weights[dst] = weights[pair];
        }
    }
    *group_count = n_groups;
    return 1;
}

static ds4_gpu_tensor *view_rows(const ds4_gpu_tensor *base,
                                 uint64_t row_begin,
                                 uint64_t row_count,
                                 uint64_t width,
                                 uint64_t element_size) {
    uint64_t offset = 0, bytes = 0;
    if (row_begin > UINT64_MAX / width ||
        row_begin * width > UINT64_MAX / element_size ||
        !checked_bytes(row_count, width, element_size, &bytes)) {
        return NULL;
    }
    offset = row_begin * width * element_size;
    return ds4_gpu_tensor_view(base, offset, bytes);
}

static int add_tensors(ds4_gpu_tensor *out,
                       const ds4_gpu_tensor *a,
                       const ds4_gpu_tensor *b,
                       uint64_t count) {
    const uint64_t elements_per_dispatch = 1ull << 30;
    uint64_t offset = 0;
    while (offset < count) {
        uint64_t chunk = count - offset;
        if (chunk > elements_per_dispatch) chunk = elements_per_dispatch;
        ds4_gpu_tensor *out_view =
            view_rows(out, offset, chunk, 1, sizeof(float));
        ds4_gpu_tensor *a_view =
            view_rows(a, offset, chunk, 1, sizeof(float));
        ds4_gpu_tensor *b_view =
            view_rows(b, offset, chunk, 1, sizeof(float));
        const int ok = out_view && a_view && b_view &&
            ds4_gpu_add_tensor(
                out_view, a_view, b_view, (uint32_t)chunk);
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(a_view);
        ds4_gpu_tensor_free(b_view);
        if (!ok) return 0;
        offset += chunk;
    }
    return 1;
}

static int run_shared_experts(
        k3_context *ctx,
        const k3_layer *layer,
        const k3_weight_source *source,
        uint32_t rows) {
    k3_scratch *s = &ctx->scratch;
    const int gguf_a8 =
        ctx->model->gguf && rows == 1u &&
        getenv("K3_GGUF_SHARED_A8") != NULL &&
        layer->shared_gate->dtype == K3_ST_DTYPE_MXFP4 &&
        layer->shared_up->dtype == K3_ST_DTYPE_MXFP4 &&
        layer->shared_down->dtype == K3_ST_DTYPE_MXFP4;
    const int gguf_pair_f32 =
        ctx->model->gguf && rows == 1u && !gguf_a8 &&
        getenv("K3_DISABLE_GGUF_SHARED_PAIR") == NULL &&
        layer->shared_gate->dtype == K3_ST_DTYPE_MXFP4 &&
        layer->shared_up->dtype == K3_ST_DTYPE_MXFP4 &&
        layer->shared_down->dtype == K3_ST_DTYPE_MXFP4;
    if (gguf_pair_f32) {
        const uint64_t gate_offset =
            source_offset(source, layer->shared_gate);
        const uint64_t up_offset =
            source_offset(source, layer->shared_up);
        return gate_offset != UINT64_MAX &&
               up_offset != UINT64_MAX &&
               k3_gpu_gguf_mxfp4_pair_situ_f32(
                   s->shared_mid,
                   source->base,
                   source->size,
                   gate_offset,
                   up_offset,
                   s->ffn_in,
                   K3_N_EMBD,
                   K3_SHARED_FF,
                   4.0f,
                   25.0f) &&
               matmul(s->shared_out, layer->shared_down, s->shared_mid,
                      K3_SHARED_FF, K3_N_EMBD, rows, source);
    }
    if (!gguf_a8) {
        return matmul(s->shared_gate, layer->shared_gate, s->ffn_in,
                      K3_N_EMBD, K3_SHARED_FF, rows, source) &&
               matmul(s->shared_up, layer->shared_up, s->ffn_in,
                      K3_N_EMBD, K3_SHARED_FF, rows, source) &&
               k3_gpu_situ(s->shared_mid, s->shared_gate, s->shared_up,
                           K3_SHARED_FF, rows, 4.0f, 25.0f) &&
               matmul(s->shared_out, layer->shared_down, s->shared_mid,
                      K3_SHARED_FF, K3_N_EMBD, rows, source);
    }
    const uint64_t gate_offset = source_offset(source, layer->shared_gate);
    const uint64_t up_offset = source_offset(source, layer->shared_up);
    const uint64_t down_offset = source_offset(source, layer->shared_down);
    return gate_offset != UINT64_MAX &&
           up_offset != UINT64_MAX &&
           down_offset != UINT64_MAX &&
           k3_gpu_mxfp8_quantize(
               s->shared_input_values,
               s->shared_input_scales,
               s->ffn_in,
               K3_N_EMBD,
               1u) &&
           k3_gpu_gguf_mxfp4_pair_situ(
               s->shared_mid,
               source->base,
               source->size,
               gate_offset,
               up_offset,
               s->shared_input_values,
               s->shared_input_scales,
               K3_N_EMBD,
               K3_SHARED_FF,
               1u,
               4.0f,
               25.0f) &&
           k3_gpu_mxfp8_quantize(
               s->shared_mid_values,
               s->shared_mid_scales,
               s->shared_mid,
               K3_SHARED_FF,
               1u) &&
           k3_gpu_gguf_mxfp4_down(
               s->shared_out,
               source->base,
               source->size,
               down_offset,
               s->shared_mid_values,
               s->shared_mid_scales,
               K3_SHARED_FF,
               K3_N_EMBD,
               1u);
}

static int run_one_expert(k3_context *ctx,
                          const k3_layer *layer,
                          const k3_expert_group *group,
                          int input_prequantized,
                          const k3_streamed_expert *streamed) {
    k3_scratch *s = &ctx->scratch;
    const k3_expert_weights *expert =
        &layer->experts[group->expert];
    const uint32_t rows = group->count;
    const int direct_decode_accum = input_prequantized && rows == 1u;
    const float *group_weights = direct_decode_accum
        ? ds4_gpu_tensor_contents(s->group_weights) : NULL;
    const float route_weight = group_weights
        ? group_weights[group->begin] : 1.0f;
    ds4_gpu_tensor *row_ids = direct_decode_accum ? NULL : view_rows(
        s->group_row_ids, group->begin, rows, 1, sizeof(int32_t));
    ds4_gpu_tensor *row_weights = direct_decode_accum ? NULL : view_rows(
        s->group_weights, group->begin, rows, 1, sizeof(float));
    ds4_gpu_tensor *group_input = input_prequantized ? NULL : view_rows(
        s->group_input, 0, rows, K3_ROUTED_HIDDEN, sizeof(float));
    ds4_gpu_tensor *input_values = view_rows(
        s->group_input_values, 0, rows, K3_ROUTED_HIDDEN, sizeof(uint8_t));
    ds4_gpu_tensor *input_scales = view_rows(
        s->group_input_scales, 0, rows, K3_ROUTED_HIDDEN / 32u,
        sizeof(uint8_t));
    ds4_gpu_tensor *gate = view_rows(
        s->expert_gate, 0, rows, K3_EXPERT_FF, sizeof(float));
    ds4_gpu_tensor *up = view_rows(
        s->expert_up, 0, rows, K3_EXPERT_FF, sizeof(float));
    ds4_gpu_tensor *mid = view_rows(
        s->expert_mid, 0, rows, K3_EXPERT_FF, sizeof(float));
    ds4_gpu_tensor *mid_values = view_rows(
        s->expert_mid_values, 0, rows, K3_EXPERT_FF, sizeof(uint8_t));
    ds4_gpu_tensor *mid_scales = view_rows(
        s->expert_mid_scales, 0, rows, K3_EXPERT_FF / 32u,
        sizeof(uint8_t));
    ds4_gpu_tensor *expert_out = direct_decode_accum ? NULL : view_rows(
        s->expert_out, 0, rows, K3_ROUTED_HIDDEN, sizeof(float));
    if ((!direct_decode_accum && (!row_ids || !row_weights || !expert_out)) ||
        (direct_decode_accum && !group_weights) ||
        (!input_prequantized && !group_input) || !input_values ||
        !input_scales || !gate || !up || !mid || !mid_values ||
        !mid_scales) {
        goto fail;
    }
    const void *base = k3_st_map_base(layer->shard);
    const uint64_t size = k3_st_file_size(layer->shard);
    const int ok =
        (input_prequantized ||
         (k3_gpu_gather_rows(
              group_input, s->routed_latent, row_ids,
              rows, K3_ROUTED_HIDDEN) &&
          k3_gpu_mxfp8_quantize(
              input_values, input_scales, group_input,
              K3_ROUTED_HIDDEN, rows))) &&
        (streamed
            ? k3_gpu_mxfp4_pair_situ_tensor(
                  gate, up, mid, streamed->weights,
                  streamed->w1_packed, streamed->w1_scale,
                  streamed->w3_packed, streamed->w3_scale,
                  input_values, input_scales,
                  K3_ROUTED_HIDDEN, K3_EXPERT_FF, rows, 4.0f, 25.0f)
            : k3_gpu_mxfp4_pair_situ(
                  gate, up, mid, base, size,
                  expert->w1_packed->offset, expert->w1_scale->offset,
                  expert->w3_packed->offset, expert->w3_scale->offset,
                  input_values, input_scales,
                  K3_ROUTED_HIDDEN, K3_EXPERT_FF, rows, 4.0f, 25.0f)) &&
        k3_gpu_mxfp8_quantize(
            mid_values, mid_scales, mid, K3_EXPERT_FF, rows) &&
        (streamed
            ? k3_gpu_mxfp4_down_tensor(
                  direct_decode_accum ? s->routed_sum : expert_out,
                  streamed->weights,
                  streamed->w2_packed, streamed->w2_scale,
                  mid_values, mid_scales,
                  K3_EXPERT_FF, K3_ROUTED_HIDDEN, rows,
                  direct_decode_accum ? route_weight : 1.0f,
                  direct_decode_accum)
            : k3_gpu_mxfp4_down(
                  direct_decode_accum ? s->routed_sum : expert_out,
                  base, size,
                  expert->w2_packed->offset, expert->w2_scale->offset,
                  mid_values, mid_scales,
                  K3_EXPERT_FF, K3_ROUTED_HIDDEN, rows,
                  direct_decode_accum ? route_weight : 1.0f,
                  direct_decode_accum)) &&
        (direct_decode_accum ||
         k3_gpu_scatter_add_rows(
             s->routed_sum, expert_out, row_ids, row_weights,
             rows, K3_ROUTED_HIDDEN));
    ds4_gpu_tensor_free(row_ids);
    ds4_gpu_tensor_free(row_weights);
    ds4_gpu_tensor_free(group_input);
    ds4_gpu_tensor_free(input_values);
    ds4_gpu_tensor_free(input_scales);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(mid_values);
    ds4_gpu_tensor_free(mid_scales);
    ds4_gpu_tensor_free(expert_out);
    return ok;

fail:
    ds4_gpu_tensor_free(row_ids);
    ds4_gpu_tensor_free(row_weights);
    ds4_gpu_tensor_free(group_input);
    ds4_gpu_tensor_free(input_values);
    ds4_gpu_tensor_free(input_scales);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(mid_values);
    ds4_gpu_tensor_free(mid_scales);
    ds4_gpu_tensor_free(expert_out);
    return 0;
}

static int run_moe_ffn(k3_context *ctx,
                       k3_layer *layer,
                       const k3_weight_source *source,
                       uint32_t rows,
                       k3_output_prefetch *output_prefetch,
                       char *err,
                       size_t errcap) {
    k3_scratch *s = &ctx->scratch;
    const void *base = source->base;
    const uint64_t size = source->size;
    k3_weight_source shared_resident = {0};
    const k3_weight_source *shared_source = source;
    if (ctx->selective_shared &&
        ctx->resident_shared_slots[layer->index].data) {
        uint64_t shared_offset = 0;
        uint64_t shared_bytes = 0;
        if (!layer_shared_span(layer, &shared_offset, &shared_bytes)) {
            set_error(err, errcap,
                      "K3 layer %u shared resident span is invalid",
                      layer->index);
            return 0;
        }
        shared_resident = (k3_weight_source){
            .base = ctx->resident_shared_slots[layer->index].data,
            .size = ctx->resident_shared_slots[layer->index].capacity,
            .bias = shared_offset,
        };
        shared_source = &shared_resident;
    }
    const uint64_t routed_elements = (uint64_t)rows * K3_ROUTED_HIDDEN;
    if ((!ctx->gguf_resident &&
         !k3_gpu_zero_f32_ordered(s->routed_sum, routed_elements)) ||
        !matmul(s->router_logits, layer->router, s->ffn_in,
                K3_N_EMBD, K3_N_EXPERT, rows, source) ||
        !k3_gpu_router_topk(
            s->selected_ids, s->selected_weights, s->router_logits,
            base, size, source_offset(source, layer->router_bias),
            rows, K3_N_EXPERT, K3_N_EXPERT_USED)) {
        set_error(err, errcap, "K3 layer %u router failed", layer->index);
        return 0;
    }

    const int needs_router_readback =
        !ctx->model->gguf || !ctx->gguf_resident;
    uint64_t router_event = 0;
    if ((needs_router_readback &&
         !ds4_gpu_signal_selected_readback_ready(&router_event)) ||
        !matmul(s->routed_latent, layer->routed_down, s->ffn_in,
                K3_N_EMBD, K3_ROUTED_HIDDEN, rows, source) ||
        !run_shared_experts(ctx, layer, shared_source, rows) ||
        (needs_router_readback &&
         !ds4_gpu_commit_and_wait_selected_readback(
             router_event, "K3 router readback"))) {
        set_error(err, errcap, "K3 layer %u static MoE path failed",
                  layer->index);
        return 0;
    }
    if (ctx->model->gguf) {
        if (!ctx->gguf_resident &&
            !prefetch_gguf_selected_experts(
                ctx, layer, rows, err, errcap)) {
            return 0;
        }
        const k3_st_tensor *gate_weights = layer->expert_gate;
        const k3_st_tensor *up_weights = layer->expert_up;
        const k3_st_tensor *down_weights = layer->expert_down;
        if (!gate_weights || !up_weights || !down_weights ||
            gate_weights->size % K3_N_EXPERT != 0 ||
            up_weights->size % K3_N_EXPERT != 0 ||
            up_weights->size != gate_weights->size ||
            down_weights->size % K3_N_EXPERT != 0) {
            set_error(err, errcap,
                      "K3 layer %u GGUF expert layout is invalid",
                      layer->index);
            return 0;
        }
        k3_weight_source mapped_experts = *source;
        const uint64_t gate_expert_bytes =
            gate_weights->size / K3_N_EXPERT;
        const uint64_t down_expert_bytes =
            down_weights->size / K3_N_EXPERT;
        const uint64_t gate_row_bytes =
            gate_expert_bytes / K3_EXPERT_FF;
        const uint64_t down_row_bytes =
            down_expert_bytes / K3_ROUTED_HIDDEN;
        const uint64_t source_end =
            source->bias <= UINT64_MAX - source->size
                ? source->bias + source->size : UINT64_MAX;
        const int experts_in_source =
            gate_weights->offset >= source->bias &&
            gate_weights->offset <= source_end &&
            gate_weights->size <= source_end - gate_weights->offset &&
            up_weights->offset >= source->bias &&
            up_weights->offset <= source_end &&
            up_weights->size <= source_end - up_weights->offset &&
            down_weights->offset >= source->bias &&
            down_weights->offset <= source_end &&
            down_weights->size <= source_end - down_weights->offset;
        if (!experts_in_source) {
            if (ctx->model->gguf) {
                mapped_experts = (k3_weight_source){
                    .base = k3_st_map_base(ctx->globals.shard),
                    .size = k3_st_file_size(ctx->globals.shard),
                    .bias = 0,
                };
            } else {
                if (!k3_st_map(layer->shard, err, errcap)) return 0;
                mapped_experts = (k3_weight_source){
                    .base = k3_st_map_base(layer->shard),
                    .size = k3_st_file_size(layer->shard),
                    .bias = 0,
                };
            }
        }
        if (!mapped_experts.base ||
            gate_weights->offset < mapped_experts.bias ||
            up_weights->offset < mapped_experts.bias ||
            down_weights->offset < mapped_experts.bias ||
            gate_row_bytes * K3_EXPERT_FF != gate_expert_bytes ||
            down_row_bytes * K3_ROUTED_HIDDEN != down_expert_bytes) {
            set_error(err, errcap,
                      "K3 layer %u GGUF expert source is invalid",
                      layer->index);
            return 0;
        }
        if (output_prefetch && !ctx->resident_lm_head.data &&
            !ctx->gguf_static_resident &&
            !ctx->gguf_resident &&
            !output_prefetch_start(
                output_prefetch,
                ctx->globals.shard,
                ctx->globals.lm_head,
                err,
                errcap)) {
            return 0;
        }
        const uint64_t expert_bias = mapped_experts.bias;
        const int routed_ok = rows == 1u
            ? ds4_gpu_k3_routed_moe_one_tensor(
                  s->routed_sum,
                  s->expert_gate,
                  s->expert_up,
                  s->expert_mid,
                  s->expert_out,
                  mapped_experts.base,
                  mapped_experts.size,
                  gate_weights->offset - expert_bias,
                  up_weights->offset - expert_bias,
                  down_weights->offset - expert_bias,
                  gate_expert_bytes,
                  gate_row_bytes,
                  down_expert_bytes,
                  down_row_bytes,
                  K3_ROUTED_HIDDEN,
                  K3_EXPERT_FF,
                  K3_ROUTED_HIDDEN,
                  s->selected_ids,
                  s->selected_weights,
                  K3_N_EXPERT,
                  K3_N_EXPERT_USED,
                  s->routed_latent,
                  layer->index,
                  ctx->gguf_resident)
            : ds4_gpu_k3_routed_moe_batch_tensor(
                  s->routed_sum,
                  s->expert_gate,
                  s->expert_up,
                  s->expert_mid,
                  s->expert_out,
                  mapped_experts.base,
                  mapped_experts.size,
                  gate_weights->offset - expert_bias,
                  up_weights->offset - expert_bias,
                  down_weights->offset - expert_bias,
                  gate_expert_bytes,
                  gate_row_bytes,
                  down_expert_bytes,
                  down_row_bytes,
                  K3_ROUTED_HIDDEN,
                  K3_EXPERT_FF,
                  K3_ROUTED_HIDDEN,
                  s->selected_ids,
                  s->selected_weights,
                  K3_N_EXPERT,
                  K3_N_EXPERT_USED,
                  s->routed_latent,
                  layer->index,
                  rows,
                  ctx->gguf_resident);
        if (!routed_ok) {
            set_error(err, errcap,
                      "K3 layer %u GGUF routed MoE failed",
                      layer->index);
            return 0;
        }
        if (!rms_norm(s->routed_norm, s->routed_sum, layer->routed_norm,
                      K3_ROUTED_HIDDEN, rows, source) ||
            !matmul(s->routed_out, layer->routed_up, s->routed_norm,
                    K3_ROUTED_HIDDEN, K3_N_EMBD, rows, source) ||
            !add_tensors(s->delta, s->shared_out, s->routed_out,
                         (uint64_t)rows * K3_N_EMBD)) {
            set_error(err, errcap,
                      "K3 layer %u GGUF routed MoE failed",
                      layer->index);
            return 0;
        }
        return 1;
    }

    k3_expert_group groups[K3_N_EXPERT];
    uint32_t n_groups = 0;
    if (!build_expert_groups(
            ctx, layer->index, rows, groups, &n_groups, err, errcap)) {
        return 0;
    }

    const int batched_prefill =
        n_groups > K3_STREAMED_EXPERT_SLOTS;
    const uint32_t speculative_bank =
        layer->index % K3_SPECULATIVE_BANKS;
    const int has_speculative_batch =
        !batched_prefill && rows == 1u &&
        ctx->speculative_batch[speculative_bank] &&
        ctx->speculative_batch[speculative_bank]->initialized &&
        ctx->speculative_layer[speculative_bank] == layer->index;
    if (ctx->speculative_batch[speculative_bank] &&
        ctx->speculative_batch[speculative_bank]->initialized &&
        !has_speculative_batch) {
        set_error(err, errcap,
                  "K3 speculative expert batch targets layer %u, not %u",
                  ctx->speculative_layer[speculative_bank], layer->index);
        (void)finish_all_speculative_experts(ctx, NULL, 0);
        return 0;
    }
    int32_t speculative_slot[K3_STREAMED_EXPERT_SLOTS];
    int32_t exact_slot[K3_STREAMED_EXPERT_SLOTS];
    k3_expert_group exact_groups[K3_STREAMED_EXPERT_SLOTS];
    uint8_t speculative_needed[K3_STREAMED_EXPERT_SLOTS] = {0};
    uint32_t exact_count = 0;
    uint32_t speculative_hits = 0;
    if (!batched_prefill) {
        for (uint32_t i = 0; i < n_groups; i++) {
            speculative_slot[i] = -1;
            exact_slot[i] = -1;
            if (has_speculative_batch) {
                for (uint32_t j = 0;
                     j < ctx->speculative_count[speculative_bank];
                     j++) {
                    if (groups[i].expert ==
                        ctx->speculative_groups[speculative_bank][j].expert) {
                        speculative_slot[i] = (int32_t)j;
                        speculative_needed[j] = 1;
                        speculative_hits++;
                        break;
                    }
                }
            }
            if (speculative_slot[i] < 0) {
                exact_slot[i] = (int32_t)exact_count;
                exact_groups[exact_count++] = groups[i];
            }
        }
    }
    if (has_speculative_batch) {
        ctx->speculative_route_hits += speculative_hits;
        ctx->speculative_route_total += n_groups;
        if (!expert_io_pool_resolve_speculation(
                ctx,
                ctx->speculative_batch[speculative_bank],
                speculative_needed,
                ctx->speculative_count[speculative_bank],
                err,
                errcap)) {
            (void)finish_all_speculative_experts(ctx, NULL, 0);
            return 0;
        }
    }
    k3_stream_batch stream_batch = {0};
    const int profile_io = io_profile_enabled();
    const double io_start = profile_io ? now_seconds() : 0.0;
    const uint32_t first_batch_count = batched_prefill
        ? K3_STREAMED_EXPERT_SLOTS : exact_count;
    const k3_expert_group *first_batch_groups = batched_prefill
        ? groups : exact_groups;
    /*
     * The fixed expert bank is allocated before resident-static budgeting.
     * Large prefills reuse it in bounded batches instead of creating hundreds
     * of no-copy Metal allocations on top of the static residency set.
     */
    if (first_batch_count != 0 && !start_streamed_experts(
            ctx,
            layer,
            first_batch_groups,
            first_batch_count,
            ctx->streamed_experts,
            0,
            &stream_batch,
            err,
            errcap)) {
        if (!err || !err[0]) {
            set_error(err, errcap,
                      "K3 layer %u expert streaming failed",
                      layer->index);
        }
        (void)finish_all_speculative_experts(ctx, NULL, 0);
        return 0;
    }
    if (!predict_next_route(ctx, layer, rows, err, errcap)) {
        (void)finish_streamed_experts(&stream_batch, NULL, 0);
        (void)finish_all_speculative_experts(ctx, NULL, 0);
        return 0;
    }
    if (speculative_early_enabled() &&
        !start_speculative_experts(
            ctx, layer->index + 1u, err, errcap)) {
        (void)finish_streamed_experts(&stream_batch, NULL, 0);
        (void)finish_all_speculative_experts(ctx, NULL, 0);
        return 0;
    }
    if (output_prefetch && !ctx->resident_lm_head.data &&
        !ctx->gguf_resident &&
        !output_prefetch_start(
            output_prefetch,
            ctx->globals.shard,
            ctx->globals.lm_head,
            err,
            errcap)) {
        (void)finish_streamed_experts(&stream_batch, NULL, 0);
        (void)finish_all_speculative_experts(ctx, NULL, 0);
        return 0;
    }

    const int input_prequantized = rows == 1u;
    if (input_prequantized &&
        !k3_gpu_mxfp8_quantize(
            s->group_input_values,
            s->group_input_scales,
            s->routed_latent,
            K3_ROUTED_HIDDEN,
            1u)) {
        set_error(err, errcap,
                  "K3 layer %u decode activation quantization failed",
                  layer->index);
        (void)finish_streamed_experts(&stream_batch, NULL, 0);
        (void)finish_all_speculative_experts(ctx, NULL, 0);
        return 0;
    }
    const uint32_t pipeline_batch = expert_pipeline_batch();
    if (batched_prefill) {
        uint32_t offset = 0;
        while (offset < n_groups) {
            uint32_t batch_count = n_groups - offset;
            if (batch_count > K3_STREAMED_EXPERT_SLOTS) {
                batch_count = K3_STREAMED_EXPERT_SLOTS;
            }
            if (offset != 0) {
                if (!ds4_gpu_synchronize() ||
                    !start_streamed_experts(
                        ctx,
                        layer,
                        &groups[offset],
                        batch_count,
                        ctx->streamed_experts,
                        0,
                        &stream_batch,
                        err,
                        errcap) ||
                    !ds4_gpu_begin_commands()) {
                    (void)finish_streamed_experts(
                        &stream_batch, NULL, 0);
                    if (!err || !err[0]) {
                        set_error(
                            err, errcap,
                            "K3 layer %u expert batch setup failed",
                            layer->index);
                    }
                    (void)finish_all_speculative_experts(ctx, NULL, 0);
                    return 0;
                }
            }
            for (uint32_t j = 0; j < batch_count; j++) {
                const uint32_t i = offset + j;
                if (!wait_streamed_expert(
                        &stream_batch, j, err, errcap) ||
                    !run_one_expert(
                        ctx,
                        layer,
                        &groups[i],
                        input_prequantized,
                        &ctx->streamed_experts[j])) {
                    if (!err || !err[0]) {
                        set_error(
                            err, errcap,
                            "K3 layer %u expert %u failed",
                            layer->index, groups[i].expert);
                    }
                    (void)finish_streamed_experts(
                        &stream_batch, NULL, 0);
                    (void)finish_all_speculative_experts(ctx, NULL, 0);
                    return 0;
                }
                if (pipeline_batch != 0 &&
                    (j + 1u) < batch_count &&
                    (j + 1u) % pipeline_batch == 0 &&
                    !ds4_gpu_flush_commands()) {
                    set_error(
                        err, errcap,
                        "K3 layer %u expert pipeline commit failed",
                        layer->index);
                    (void)finish_streamed_experts(
                        &stream_batch, NULL, 0);
                    (void)finish_all_speculative_experts(ctx, NULL, 0);
                    return 0;
                }
            }
            if (!finish_streamed_experts(
                    &stream_batch, err, errcap)) {
                if (!err || !err[0]) {
                    set_error(err, errcap,
                              "K3 layer %u expert streaming failed",
                              layer->index);
                }
                (void)finish_all_speculative_experts(ctx, NULL, 0);
                return 0;
            }
            offset += batch_count;
        }
    } else {
        for (uint32_t i = 0; i < n_groups; i++) {
            k3_stream_batch *read_batch = NULL;
            uint32_t read_slot = 0;
            const k3_streamed_expert *weights = NULL;
            if (speculative_slot[i] >= 0) {
                read_batch = ctx->speculative_batch[speculative_bank];
                read_slot = (uint32_t)speculative_slot[i];
                weights =
                    &ctx->speculative_experts[speculative_bank][read_slot];
            } else {
                read_batch = &stream_batch;
                read_slot = (uint32_t)exact_slot[i];
                weights = &ctx->streamed_experts[read_slot];
            }
            if ((read_batch &&
                 !wait_streamed_expert(
                     read_batch, read_slot, err, errcap)) ||
                !run_one_expert(
                    ctx,
                    layer,
                    &groups[i],
                    input_prequantized,
                    weights)) {
                if (!err || !err[0]) {
                    set_error(err, errcap, "K3 layer %u expert %u failed",
                              layer->index, groups[i].expert);
                }
                (void)finish_streamed_experts(&stream_batch, NULL, 0);
                (void)finish_all_speculative_experts(ctx, NULL, 0);
                return 0;
            }
            if (pipeline_batch != 0 &&
                (i + 1u) < n_groups &&
                (i + 1u) % pipeline_batch == 0 &&
                !ds4_gpu_flush_commands()) {
                set_error(err, errcap,
                          "K3 layer %u expert pipeline commit failed",
                          layer->index);
                (void)finish_streamed_experts(&stream_batch, NULL, 0);
                (void)finish_all_speculative_experts(ctx, NULL, 0);
                return 0;
            }
        }
        if (!finish_streamed_experts(&stream_batch, err, errcap)) {
            if (!err || !err[0]) {
                set_error(err, errcap,
                          "K3 layer %u expert streaming failed",
                          layer->index);
            }
            (void)finish_all_speculative_experts(ctx, NULL, 0);
            return 0;
        }
    }
    if (!finish_speculative_bank(
            ctx, speculative_bank, err, errcap)) {
        if (!err || !err[0]) {
            set_error(err, errcap,
                      "K3 layer %u speculative expert streaming failed",
                      layer->index);
        }
        return 0;
    }
    if (profile_io) {
        const uint32_t streamed_count =
            batched_prefill ? n_groups : exact_count;
        ctx->expert_stream_bytes +=
            (uint64_t)streamed_count * ctx->streamed_experts[0].bytes;
        ctx->expert_stream_seconds += now_seconds() - io_start;
    }
    if (!speculative_early_enabled() &&
        !start_speculative_experts(
            ctx, layer->index + 1u, err, errcap)) {
        return 0;
    }
    if (!rms_norm(s->routed_norm, s->routed_sum, layer->routed_norm,
                  K3_ROUTED_HIDDEN, rows, source) ||
        !matmul(s->routed_out, layer->routed_up, s->routed_norm,
                K3_ROUTED_HIDDEN, K3_N_EMBD, rows, source) ||
        !add_tensors(s->delta, s->shared_out, s->routed_out,
                     (uint64_t)rows * K3_N_EMBD)) {
        set_error(err, errcap, "K3 layer %u routed output failed",
                  layer->index);
        (void)finish_all_speculative_experts(ctx, NULL, 0);
        return 0;
    }
    return 1;
}

static int run_layer(k3_context *ctx,
                     k3_layer *layer,
                     const k3_weight_source *source,
                     uint32_t rows,
                     int prefill,
                     int has_delta,
                     k3_output_prefetch *output_prefetch,
                     char *err,
                     size_t errcap) {
    k3_scratch *s = &ctx->scratch;
    const void *base = source->base;
    const uint64_t size = source->size;
    const uint32_t valid_blocks =
        (layer->index + K3_ATTN_RES_BLOCK - 1u) / K3_ATTN_RES_BLOCK;
    const int write_block =
        (layer->index % K3_ATTN_RES_BLOCK) == 0u;

    if (!k3_gpu_attn_res(
            s->attn_in,
            s->prefix,
            has_delta ? s->delta : NULL,
            s->blocks,
            base,
            size,
            source_offset(source, layer->self_attention_res_norm),
            source_offset(source, layer->self_attention_res_proj),
            source_offset(source, layer->input_norm),
            K3_N_EMBD,
            rows,
            K3_ATTN_RES_BLOCKS,
            valid_blocks,
            write_block ? (int32_t)(layer->index / K3_ATTN_RES_BLOCK) : -1,
            0,
            K3_RMS_EPS,
            K3_RMS_EPS)) {
        set_error(err, errcap, "K3 layer %u attention failed", layer->index);
        return 0;
    }
    hidden_hook_emit(
        ctx, "hc_attn_pre", layer->index, s->attn_in, rows);
    if (!run_attention(ctx, layer, source, rows, prefill)) {
        set_error(err, errcap, "K3 layer %u attention failed", layer->index);
        return 0;
    }
    hidden_hook_emit(
        ctx, "attn_out", layer->index, s->attn_out, rows);
    const ds4_gpu_tensor *attention_delta = s->attn_out;
    if (write_block) {
        if (!ds4_gpu_tensor_copy(
                s->prefix, 0, s->attn_out, 0,
                (uint64_t)rows * K3_N_EMBD * sizeof(float))) {
            set_error(err, errcap, "K3 layer %u prefix update failed",
                      layer->index);
            return 0;
        }
        attention_delta = NULL;
    }
    if (!k3_gpu_attn_res(
            s->ffn_in,
            s->prefix,
            attention_delta,
            s->blocks,
            base,
            size,
            source_offset(source, layer->mlp_res_norm),
            source_offset(source, layer->mlp_res_proj),
            source_offset(source, layer->post_attention_norm),
            K3_N_EMBD,
            rows,
            K3_ATTN_RES_BLOCKS,
            valid_blocks + (write_block ? 1u : 0u),
            -1,
            0,
            K3_RMS_EPS,
            K3_RMS_EPS)) {
        set_error(err, errcap, "K3 layer %u post-attention failed",
                  layer->index);
        return 0;
    }
    hidden_hook_emit(
        ctx, "hc_ffn_pre", layer->index, s->ffn_in, rows);
    const int ffn_ok = layer->index == 0
        ? run_dense_ffn(ctx, layer, source, rows)
        : run_moe_ffn(
              ctx, layer, source, rows, output_prefetch, err, errcap);
    if (!ffn_ok) return 0;
    hidden_hook_emit(
        ctx, "ffn_out", layer->index, s->delta, rows);
    return 1;
}

static int run_output(k3_context *ctx,
                      uint32_t rows,
                      float *logits,
                      k3_output_prefetch *output_prefetch,
                      char *err,
                      size_t errcap) {
    k3_scratch *s = &ctx->scratch;
    k3_global_weights *g = &ctx->globals;
    if (!output_prefetch_join(output_prefetch, err, errcap) ||
        (!ctx->gguf_resident && !ds4_gpu_synchronize()) ||
        !map_global_tensors(ctx, 0, err, errcap) ||
        !ds4_gpu_begin_commands()) {
        if (!err || !err[0]) set_error(err, errcap, "K3 output setup failed");
        return 0;
    }
    const void *base = k3_st_map_base(g->shard);
    const uint64_t size = k3_st_file_size(g->shard);
    const k3_weight_source source = {
        .base = base,
        .size = size,
        .bias = 0,
    };
    const k3_weight_source head_source = ctx->resident_lm_head.data
        ? (k3_weight_source){
              .base = ctx->resident_lm_head.data,
              .size = ctx->resident_lm_head.capacity,
              .bias = g->lm_head->offset,
          }
        : source;
    if (!k3_gpu_attn_res(
            s->final_hidden,
            s->prefix,
            s->delta,
            s->blocks,
            base,
            size,
            g->output_attn_res_norm->offset,
            g->output_attn_res_proj->offset,
            UINT64_MAX,
            K3_N_EMBD,
            rows,
            K3_ATTN_RES_BLOCKS,
            K3_ATTN_RES_BLOCKS,
            -1,
            0,
            K3_RMS_EPS,
            0.0f)) {
        set_error(err, errcap, "K3 final AttnRes failed");
        return 0;
    }
    ds4_gpu_tensor *last = view_rows(
        s->final_hidden, rows - 1u, 1, K3_N_EMBD, sizeof(float));
    if (!last ||
        !rms_norm(s->final_norm, last, g->norm,
                  K3_N_EMBD, 1, &source) ||
        !matmul(s->logits, g->lm_head, s->final_norm,
                K3_N_EMBD, K3_N_VOCAB, 1, &head_source) ||
        !ds4_gpu_end_commands() ||
        !ds4_gpu_tensor_read(
            s->logits, 0, logits,
            (uint64_t)K3_N_VOCAB * sizeof(float))) {
        ds4_gpu_tensor_free(last);
        set_error(err, errcap, "K3 output projection failed");
        return 0;
    }
    ds4_gpu_tensor_free(last);
    return 1;
}

static int import_boundary(k3_context *ctx,
                           const float *boundary,
                           uint32_t rows) {
    if (!ctx || !boundary || rows == 0) return 0;
    const uint64_t vector_bytes =
        (uint64_t)rows * K3_N_EMBD * sizeof(float);
    const uint64_t blocks_bytes =
        vector_bytes * K3_ATTN_RES_BLOCKS;
    return ds4_gpu_tensor_write(
               ctx->scratch.prefix, 0, boundary, vector_bytes) &&
           ds4_gpu_tensor_write(
               ctx->scratch.delta,
               0,
               (const uint8_t *)boundary + vector_bytes,
               vector_bytes) &&
           ds4_gpu_tensor_write(
               ctx->scratch.blocks,
               0,
               (const uint8_t *)boundary + 2u * vector_bytes,
               blocks_bytes);
}

static int export_boundary(k3_context *ctx,
                           float *boundary,
                           uint32_t rows) {
    if (!ctx || !boundary || rows == 0) return 0;
    const uint64_t vector_bytes =
        (uint64_t)rows * K3_N_EMBD * sizeof(float);
    const uint64_t blocks_bytes =
        vector_bytes * K3_ATTN_RES_BLOCKS;
    return ds4_gpu_tensor_read(
               ctx->scratch.prefix, 0, boundary, vector_bytes) &&
           ds4_gpu_tensor_read(
               ctx->scratch.delta,
               0,
               (uint8_t *)boundary + vector_bytes,
               vector_bytes) &&
           ds4_gpu_tensor_read(
               ctx->scratch.blocks,
               0,
               (uint8_t *)boundary + 2u * vector_bytes,
               blocks_bytes);
}

static int evaluate(k3_context *ctx,
                    const int32_t *token_ids,
                    uint32_t rows,
                    int prefill,
                    const float *input_boundary,
                    float *output_boundary,
                    int output_logits,
                    float *logits,
                    char *err,
                    size_t errcap) {
    if (!ctx || !token_ids || rows == 0 ||
        rows > ctx->max_prefill_tokens ||
        ctx->position > ctx->context_capacity ||
        rows > ctx->context_capacity - ctx->position ||
        (ctx->layer_start != 0 && !input_boundary) ||
        (ctx->layer_start == 0 && input_boundary) ||
        (output_logits &&
         (!logits || !ctx->has_output ||
          ctx->layer_end + 1u != K3_N_LAYER))) {
        set_error(err, errcap, "invalid K3 evaluation request");
        return 0;
    }
    for (uint32_t i = 0; i < rows; i++) {
        if (token_ids[i] < 0 || token_ids[i] >= K3_N_VOCAB) {
            set_error(err, errcap, "K3 token id %d is outside the vocabulary",
                      token_ids[i]);
            return 0;
        }
    }
    if (!finish_all_speculative_experts(ctx, err, errcap)) {
        return 0;
    }

    k3_layer_load *current_load = NULL;
    k3_static_load static_load = {0};
    k3_output_prefetch output_prefetch = {0};
    const int trace = trace_enabled();
    const int profile = trace || io_profile_enabled();
    const int pagein = layer_pagein_enabled();
    k3_context *owner = execution_owner(ctx);
    const int resident_readahead =
        ctx->gguf_resident && !owner->gguf_readahead_done;
    const int stage_static =
        !ctx->model->gguf && ctx->static_staging && rows == 1u;
    const uint32_t first_layer = ctx->layer_start;
    const int first_resident =
        ctx->q8_static ||
        ctx->resident_static_slots[first_layer].data != NULL;
    const double evaluate_start = profile ? now_seconds() : 0.0;
    double layer_ready_seconds = 0.0;
    double layer_execute_seconds = 0.0;
    ctx->expert_stream_bytes = 0;
    ctx->expert_stream_seconds = 0.0;
    ctx->static_stream_bytes = 0;
    ctx->static_stream_seconds = 0.0;
    ctx->route_overlap = 0;
    ctx->route_total = 0;
    ctx->route_prediction_hits = 0;
    ctx->route_prediction_total = 0;
    memset(ctx->route_prediction_rank_hits, 0,
           sizeof(ctx->route_prediction_rank_hits));
    ctx->route_prediction_transitions = 0;
    ctx->speculative_stream_bytes = 0;
    ctx->speculative_stream_seconds = 0.0;
    ctx->speculative_route_hits = 0;
    ctx->speculative_route_total = 0;
    memset(ctx->predicted_route_valid, 0,
           sizeof(ctx->predicted_route_valid));
    if (!require_no_layer_mappings(ctx, "start", err, errcap)) {
        return 0;
    }
    const int first_load_started = ctx->gguf_resident
        ? (!resident_readahead ||
           k3_layer_readahead_start(
               &current_load,
               ctx->layers[first_layer],
               true,
               err,
               errcap))
        : ctx->gguf_static_resident
        ? 1
        : ctx->model->gguf
        ? k3_layer_prefetch_start(
              &current_load,
              ctx->layers[first_layer],
              false,
              pagein,
              err,
              errcap)
        : first_resident
        ? 1
        : (stage_static
            ? static_span_load_start(
                  &static_load,
                  ctx->layers[first_layer],
                  &ctx->static_slots[first_layer & 1u],
                  ctx->layers[first_layer]->static_offset,
                  ctx->layers[first_layer]->static_size,
                  0,
                  0,
                  err,
                  errcap)
            : k3_layer_prefetch_start(
                  &current_load,
                  ctx->layers[first_layer],
                  false,
                  pagein,
                  err,
                  errcap));
    int boundary_ready = first_load_started && ds4_gpu_synchronize();
    if (boundary_ready && input_boundary) {
        boundary_ready = import_boundary(ctx, input_boundary, rows);
    } else if (boundary_ready) {
        boundary_ready =
            load_embedding_rows(ctx, token_ids, rows, err, errcap) &&
            ds4_gpu_tensor_fill_f32(
                ctx->scratch.blocks,
                0.0f,
                (uint64_t)rows * K3_ATTN_RES_BLOCKS * K3_N_EMBD) &&
            ds4_gpu_begin_commands() &&
            k3_gpu_expand_bf16(
                ctx->scratch.prefix,
                ctx->scratch.embedding_rows,
                (uint64_t)rows * K3_N_EMBD) &&
            (ctx->gguf_resident || ds4_gpu_end_commands());
    }
    if (!boundary_ready) {
        static_load_cancel(&static_load);
        k3_layer_load_cancel(current_load);
        if (!err || !err[0]) {
            set_error(err, errcap,
                      input_boundary
                          ? "K3 boundary import failed"
                          : "K3 embedding failed");
        }
        return 0;
    }

    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        const double layer_start = profile ? now_seconds() : 0.0;
        k3_layer *layer = ctx->layers[i];
        const int layer_resident =
            ctx->q8_static || ctx->resident_static_slots[i].data != NULL;
        if (ctx->model->gguf &&
            ((!ctx->gguf_resident && !ctx->gguf_static_resident) ||
             resident_readahead)) {
            k3_layer *loaded = NULL;
            if (!k3_layer_load_join(
                    current_load, &loaded, err, errcap) ||
                loaded != layer) {
                (void)finish_all_speculative_experts(ctx, NULL, 0);
                output_prefetch_discard(&output_prefetch);
                return 0;
            }
            current_load = NULL;
        } else if (!ctx->model->gguf &&
                   !layer_resident && stage_static) {
            if (!static_load_join(&static_load, err, errcap)) {
                (void)finish_all_speculative_experts(ctx, NULL, 0);
                output_prefetch_discard(&output_prefetch);
                return 0;
            }
            ctx->static_stream_bytes +=
                layer_stream_static_size(ctx, layer);
            ctx->static_stream_seconds += static_load.seconds;
        } else if (!ctx->model->gguf && !layer_resident) {
            layer = NULL;
            if (!k3_layer_load_join(current_load, &layer, err, errcap)) {
                (void)finish_all_speculative_experts(ctx, NULL, 0);
                output_prefetch_discard(&output_prefetch);
                return 0;
            }
            current_load = NULL;
        }
        const double layer_ready = profile ? now_seconds() : 0.0;
        if (i < ctx->layer_end) {
            uint64_t next_skip_offset = 0;
            uint64_t next_skip_bytes = 0;
            layer_static_skip(
                ctx, ctx->layers[i + 1u],
                &next_skip_offset, &next_skip_bytes);
            const int next_resident =
                ctx->q8_static ||
                ctx->resident_static_slots[i + 1u].data != NULL;
            const int need_next_load = ctx->model->gguf
                ? ((!ctx->gguf_resident && !ctx->gguf_static_resident) ||
                   resident_readahead)
                : !next_resident;
            if (need_next_load) {
                const int next_started = ctx->gguf_resident
                    ? k3_layer_readahead_start(
                          &current_load,
                          ctx->layers[i + 1u],
                          true,
                          err,
                          errcap)
                    : ctx->model->gguf
                    ? k3_layer_prefetch_start(
                          &current_load,
                          ctx->layers[i + 1u],
                          false,
                          pagein,
                          err,
                          errcap)
                    : stage_static
                    ? static_span_load_start(
                          &static_load,
                          ctx->layers[i + 1u],
                          &ctx->static_slots[(i + 1u) & 1u],
                          ctx->layers[i + 1u]->static_offset,
                          ctx->layers[i + 1u]->static_size,
                          next_skip_offset,
                          next_skip_bytes,
                          err,
                          errcap)
                    : k3_layer_prefetch_start(
                          &current_load,
                          ctx->layers[i + 1u],
                          false,
                          pagein,
                          err,
                          errcap);
                if (!next_started) {
                    if (ctx->model->gguf ||
                        (!layer_resident && !stage_static)) {
                        k3_st_unmap(layer->shard);
                    }
                    (void)finish_all_speculative_experts(ctx, NULL, 0);
                    return 0;
                }
            }
        }
        const uint64_t map_size =
            layer->static_size;
        const k3_static_slot *static_slot = ctx->q8_static
            ? &ctx->q8_layers[i].native
            : (layer_resident
                ? &ctx->resident_static_slots[i]
                : (stage_static ? &ctx->static_slots[i & 1u] : NULL));
        const int use_static_slot = layer_resident || stage_static;
        const int next_layer_resident =
            i < ctx->layer_end &&
            (ctx->q8_static ||
             ctx->resident_static_slots[i + 1u].data != NULL);
        const int pipeline_commands =
            i < ctx->layer_end &&
            (ctx->gguf_resident ||
             (!ctx->model->gguf &&
              ctx->pinned_static && layer_resident && !ctx->q8_static &&
              next_layer_resident));
        const k3_weight_source source = ctx->gguf_resident
            ? (k3_weight_source){
                  .base = k3_st_map_base(ctx->globals.shard),
                  .size = k3_st_file_size(ctx->globals.shard),
                  .bias = 0,
              }
            : use_static_slot
            ? (k3_weight_source){
                  .base = static_slot->data,
                  .size = ctx->pinned_static && layer_resident &&
                          !ctx->q8_static
                      ? static_slot->capacity
                      : map_size,
                  .bias = layer->static_offset,
                  .q8 = ctx->q8_static ? &ctx->q8_layers[i] : NULL,
              }
            : ctx->model->gguf
            ? (k3_weight_source){
                  .base = k3_st_map_base(ctx->globals.shard),
                  .size = k3_st_file_size(ctx->globals.shard),
                  .bias = 0,
              }
            : (k3_weight_source){
                  .base = k3_st_map_base(layer->shard),
                  .size = k3_st_file_size(layer->shard),
                  .bias = 0,
              };
        const int mapped =
            ctx->gguf_resident
            ? 1
            : ctx->model->gguf
            ? map_gguf_layer_spans(
                  ctx, layer, !use_static_slot && !ctx->gguf_static_resident,
                  err, errcap)
            : ctx->pinned_static && layer_resident && !ctx->q8_static
            ? 1
            : use_static_slot
            ? map_static_slot_span(
                  static_slot, layer, map_size, err, errcap)
            : map_range(
                  layer->shard,
                  layer->static_offset,
                  map_size,
                  K3_LAYER_MAX_TENSOR_BYTES,
                  err,
                  errcap);
        if (!mapped ||
            (!ds4_gpu_commands_active() && !ds4_gpu_begin_commands()) ||
            !run_layer(
                ctx,
                layer,
                &source,
                rows,
                prefill,
                i != 0,
                output_logits && i + 1u == K3_N_LAYER
                    ? &output_prefetch
                    : NULL,
                err,
                errcap) ||
            !(pipeline_commands
                ? ds4_gpu_flush_commands()
                : ds4_gpu_end_commands())) {
            if (ds4_gpu_commands_active()) (void)ds4_gpu_synchronize();
            if (ds4_gpu_release_model_views()) {
                if (ctx->model->gguf || !use_static_slot) {
                    k3_st_unmap(layer->shard);
                }
            }
            static_load_cancel(&static_load);
            k3_layer_load_cancel(current_load);
            (void)finish_all_speculative_experts(ctx, NULL, 0);
            output_prefetch_discard(&output_prefetch);
            return 0;
        }
        if (!pipeline_commands && !ds4_gpu_release_model_views()) {
            set_error(err, errcap,
                      "K3 layer %u Metal view release failed", i);
            static_load_cancel(&static_load);
            k3_layer_load_cancel(current_load);
            (void)finish_all_speculative_experts(ctx, NULL, 0);
            output_prefetch_discard(&output_prefetch);
            return 0;
        }
        /*
         * Each layer lives in a separate shard. Keeping all 93 mappings alive
         * retains their faulted pages in this process even after Metal drops
         * its no-copy views, so make the mapping lifetime layer-major too.
         */
        if (ctx->model->gguf || !use_static_slot) {
            k3_st_unmap(layer->shard);
        }
        if (profile) {
            const double layer_done = now_seconds();
            layer_ready_seconds += layer_ready - layer_start;
            layer_execute_seconds += layer_done - layer_ready;
            if (trace) {
                fprintf(stderr,
                        "k3: layer %u/%u %s rows=%u ready=%.3f ms "
                        "execute=%.3f ms total=%.3f ms\n",
                        i + 1u,
                        (uint32_t)K3_N_LAYER,
                        layer->attention == K3_LAYER_MLA ? "MLA" : "KDA",
                        rows,
                        (layer_ready - layer_start) * 1000.0,
                        (layer_done - layer_ready) * 1000.0,
                        (layer_done - layer_start) * 1000.0);
            }
        }
    }
    if (resident_readahead) owner->gguf_readahead_done = 1;
    const double output_start = profile ? now_seconds() : 0.0;
    if (!require_no_layer_mappings(ctx, "output", err, errcap) ||
        (output_boundary &&
         !export_boundary(ctx, output_boundary, rows)) ||
        (output_logits &&
         !run_output(
             ctx, rows, logits, &output_prefetch, err, errcap))) {
        if ((!err || !err[0]) && output_boundary) {
            set_error(err, errcap, "K3 boundary export failed");
        }
        output_prefetch_discard(&output_prefetch);
        return 0;
    }
    const double output_done = profile ? now_seconds() : 0.0;
    if (io_profile_enabled()) {
        if (ctx->expert_stream_seconds > 0.0) {
            const double gib =
                (double)ctx->expert_stream_bytes / 1073741824.0;
            fprintf(stderr,
                    "k3: expert I/O threads=%u bytes=%.3f GiB "
                    "read=%.3f s throughput=%.2f GiB/s\n",
                    expert_io_threads(),
                    gib,
                    ctx->expert_stream_seconds,
                    gib / ctx->expert_stream_seconds);
        }
        if (stage_static && ctx->static_stream_seconds > 0.0) {
            const double static_gib =
                (double)ctx->static_stream_bytes / 1073741824.0;
            fprintf(stderr,
                    "k3: static I/O threads=%u bytes=%.3f GiB "
                    "read=%.3f s throughput=%.2f GiB/s\n",
                    static_io_threads(),
                    static_gib,
                    ctx->static_stream_seconds,
                    static_gib / ctx->static_stream_seconds);
        }
        if (ctx->route_total != 0) {
            fprintf(stderr,
                    "k3: adjacent-token expert overlap=%" PRIu64
                    "/%" PRIu64 " (%.1f%%)\n",
                    ctx->route_overlap,
                    ctx->route_total,
                    100.0 * (double)ctx->route_overlap /
                        (double)ctx->route_total);
        }
        if (ctx->route_prediction_total != 0) {
            fprintf(stderr,
                    "k3: next-layer route prediction=%" PRIu64
                    "/%" PRIu64 " (%.1f%%)\n",
                    ctx->route_prediction_hits,
                    ctx->route_prediction_total,
                    100.0 * (double)ctx->route_prediction_hits /
                        (double)ctx->route_prediction_total);
            uint64_t cumulative = 0;
            fprintf(stderr, "k3: prediction precision");
            for (uint32_t i = 0; i < K3_N_EXPERT_USED; i++) {
                cumulative += ctx->route_prediction_rank_hits[i];
                if ((i + 1u) % 4u == 0u) {
                    const uint64_t total =
                        ctx->route_prediction_transitions * (i + 1u);
                    fprintf(stderr,
                            " @%u=%.1f%%",
                            i + 1u,
                            total != 0
                                ? 100.0 * (double)cumulative /
                                      (double)total
                                : 0.0);
                }
            }
            fprintf(stderr, "\n");
        }
        if (ctx->speculative_route_total != 0) {
            fprintf(stderr,
                    "k3: speculative top-%u expert hits=%" PRIu64
                    "/%" PRIu64 " (%.1f%%), read %.3f GiB in %.3f s\n",
                    speculative_top_k(),
                    ctx->speculative_route_hits,
                    ctx->speculative_route_total,
                    100.0 * (double)ctx->speculative_route_hits /
                        (double)ctx->speculative_route_total,
                    (double)ctx->speculative_stream_bytes / 1073741824.0,
                    ctx->speculative_stream_seconds);
        }
        fprintf(stderr,
                "k3: evaluation timing static=%s pagein=%s "
                "layers_ready=%.6f s "
                "layers_execute=%.6f s output=%.6f s total=%.6f s\n",
                ctx->gguf_resident
                    ? "gguf-resident"
                    : ctx->gguf_static_resident
                    ? "gguf-static-resident"
                    : ctx->q8_static
                    ? "q8-resident"
                    : ctx->resident_static
                    ? (ctx->static_staging ? "resident+staged" : "resident")
                    : (stage_static ? "staged" : "mmap"),
                pagein ? "on" : "off",
                layer_ready_seconds,
                layer_execute_seconds,
                output_done - output_start,
                output_done - evaluate_start);
    }
    ctx->position += rows;
    return 1;
}

int k3_context_create(k3_context **out,
                      const k3_context_config *config,
                      char *err,
                      size_t errcap) {
    if (out) *out = NULL;
    const uint32_t layer_start =
        config && config->layer_slice ? config->layer_start : 0u;
    const uint32_t layer_end =
        config && config->layer_slice
            ? config->layer_end
            : (uint32_t)K3_N_LAYER - 1u;
    const int has_output =
        config && config->layer_slice ? config->has_output : 1;
    if (!out || !config || !config->model_directory ||
        config->context_capacity == 0 ||
        config->context_capacity > K3_MAX_CONTEXT ||
        config->max_prefill_tokens == 0 ||
        config->max_prefill_tokens > config->context_capacity ||
        layer_start > layer_end || layer_end >= K3_N_LAYER ||
        (has_output && layer_end + 1u != K3_N_LAYER)) {
        set_error(err, errcap, "invalid K3 context configuration");
        return 0;
    }
    k3_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        set_error(err, errcap, "out of memory creating K3 context");
        return 0;
    }
    ctx->shared_owner = ctx;
    ctx->owns_execution = 1;
    if (pthread_mutex_init(&ctx->execution_mutex, NULL) != 0) {
        set_error(err, errcap, "could not initialize K3 execution mutex");
        free(ctx);
        return 0;
    }
    ctx->execution_mutex_initialized = 1;
    for (uint32_t i = 0; i < K3_N_LAYER; i++) {
        ctx->expert_fds[i] = -1;
    }
    for (uint32_t bank = 0; bank < K3_SPECULATIVE_BANKS; bank++) {
        ctx->speculative_layer[bank] = UINT32_MAX;
    }
    ctx->context_capacity = config->context_capacity;
    ctx->max_prefill_tokens = config->max_prefill_tokens;
    ctx->layer_start = layer_start;
    ctx->layer_end = layer_end;
    ctx->has_output = has_output;
    ds4_gpu_set_ssd_streaming(true);
    if (!ds4_gpu_init() ||
        !k3_model_open(
            &ctx->model, config->model_directory, err, errcap) ||
        !k3_model_open_globals(
            ctx->model, &ctx->globals, err, errcap)) {
        k3_context_destroy(ctx);
        return 0;
    }
    for (uint32_t i = 0; i < K3_N_LAYER; i++) {
        if (!k3_model_open_layer(
                ctx->model, i, &ctx->layers[i], err, errcap)) {
            k3_context_destroy(ctx);
            return 0;
        }
        if (!ctx->model->gguf) {
            ctx->expert_fds[i] =
                open_uncached_stream_fd(ctx->layers[i]->shard, err, errcap);
            if (ctx->expert_fds[i] < 0) {
                k3_context_destroy(ctx);
                return 0;
            }
        }
        k3_st_unmap(ctx->layers[i]->shard);
    }
    if (ctx->model->gguf &&
        !ds4_gpu_set_model_fd(k3_st_fd(ctx->globals.shard))) {
        set_error(err, errcap,
                  "Metal could not register the K3 GGUF descriptor");
        k3_context_destroy(ctx);
        return 0;
    }
    const int want_q8_static =
        !ctx->model->gguf && q8_static_requested();
    if (!alloc_scratch(ctx, err, errcap) ||
        !alloc_states(ctx, err, errcap) ||
        (!ctx->model->gguf &&
         (!alloc_streamed_expert_bank(
              ctx->streamed_experts,
              K3_STREAMED_EXPERT_SLOTS,
              "streamed",
              err,
              errcap) ||
          !expert_io_pool_init(ctx, err, errcap)))) {
        k3_context_destroy(ctx);
        return 0;
    }
    const uint64_t resident_budget = want_q8_static
        ? 0 : resident_static_budget(ds4_gpu_tensor_allocated_bytes());
    if (!try_pin_resident_gguf(
            ctx, resident_budget, err, errcap)) {
        k3_context_destroy(ctx);
        return 0;
    }
    uint64_t gguf_expert_cache_bytes = 0;
    if (ctx->model->gguf && !ctx->gguf_resident) {
        const k3_st_tensor *gate = ctx->layers[1]->expert_gate;
        const k3_st_tensor *down = ctx->layers[1]->expert_down;
        if (!gate || !down ||
            gate->size % K3_N_EXPERT != 0 ||
            down->size % K3_N_EXPERT != 0) {
            set_error(err, errcap,
                      "K3 GGUF routed expert layout is invalid");
            k3_context_destroy(ctx);
            return 0;
        }
        const uint64_t gate_expert_bytes =
            gate->size / K3_N_EXPERT;
        const uint64_t down_expert_bytes =
            down->size / K3_N_EXPERT;
        if (gate_expert_bytes >
            (UINT64_MAX - down_expert_bytes) / 2u) {
            set_error(err, errcap,
                      "K3 GGUF routed expert size overflow");
            k3_context_destroy(ctx);
            return 0;
        }
        const uint64_t expert_bytes =
            2u * gate_expert_bytes + down_expert_bytes;
        const uint32_t first_routed =
            ctx->layer_start < 1u ? 1u : ctx->layer_start;
        const uint32_t routed_layers =
            first_routed <= ctx->layer_end
                ? ctx->layer_end - first_routed + 1u : 0u;
        uint32_t cache_slots =
            routed_layers * K3_N_EXPERT_USED;
        /*
         * Batched prefill may route to every expert in a layer. Keep enough
         * reusable slabs for one complete layer so the selected-address path
         * never falls back to faulting multi-GiB tensor views. Slabs are
         * allocated lazily, and decode retains the larger per-layer top-k
         * budget when this process owns enough layers.
         */
        if (routed_layers != 0 && cache_slots < K3_N_EXPERT) {
            cache_slots = K3_N_EXPERT;
        }
        ds4_gpu_set_streaming_expert_cache_expert_bytes(expert_bytes);
        ds4_gpu_set_streaming_expert_cache_budget(cache_slots);
        gguf_expert_cache_bytes = (uint64_t)cache_slots * expert_bytes;
        fprintf(stderr,
                "k3: GGUF expert staging/cache enabled: %u slots "
                "(at least one full routed layer), up to %.2f GiB\n",
                cache_slots,
                (double)cache_slots * (double)expert_bytes /
                    1073741824.0);
    } else {
        ds4_gpu_set_streaming_expert_cache_budget(0);
    }
    try_pin_gguf_static(ctx, resident_budget, gguf_expert_cache_bytes);
    const uint64_t head_capacity =
        ctx->has_output ? resident_lm_head_capacity(ctx) : 0;
    const uint64_t all_layers_capacity =
        resident_all_layers_capacity(ctx);
    const uint64_t streaming_slots_capacity =
        static_slots_capacity(ctx);
    const uint64_t speculative_capacity =
        (uint64_t)K3_SPECULATIVE_BANKS *
        speculative_top_k() * streamed_expert_bytes();
    const char *speculative_env = getenv("K3_SPECULATIVE_EXPERTS");
    const int speculative_explicit =
        speculative_env && speculative_env[0];
    const int speculative_fits_full =
        head_capacity != 0 && all_layers_capacity != 0 &&
        head_capacity <= UINT64_MAX - all_layers_capacity &&
        head_capacity + all_layers_capacity <=
            UINT64_MAX - speculative_capacity &&
        resident_budget >=
            head_capacity + all_layers_capacity + speculative_capacity;
    const int want_speculative =
        !ctx->model->gguf && !want_q8_static &&
        (speculative_explicit
            ? speculative_experts_enabled()
            : speculative_fits_full);
    ctx->speculative_enabled = want_speculative;
    if (want_speculative) {
        for (uint32_t bank = 0; bank < K3_SPECULATIVE_BANKS; bank++) {
            ctx->speculative_batch[bank] =
                calloc(1, sizeof(*ctx->speculative_batch[bank]));
            if (!ctx->speculative_batch[bank]) {
                set_error(err, errcap,
                          "out of memory creating K3 speculative I/O bank");
                k3_context_destroy(ctx);
                return 0;
            }
            if (!alloc_streamed_expert_bank(
                    ctx->speculative_experts[bank],
                    speculative_top_k(),
                    "speculative",
                    err,
                    errcap)) {
                k3_context_destroy(ctx);
                return 0;
            }
        }
        fprintf(stderr,
                "k3: next-layer top-%u speculative expert prefetch %s\n",
                speculative_top_k(),
                speculative_explicit ? "enabled" : "enabled automatically");
    }
    uint64_t layer_resident_budget = resident_budget;
    if (want_speculative &&
        layer_resident_budget >= speculative_capacity) {
        layer_resident_budget -= speculative_capacity;
    }
    if (!want_q8_static && !ctx->gguf_resident && !ctx->gguf_static_resident &&
        head_capacity != 0 &&
        layer_resident_budget >= head_capacity) {
        if (!load_resident_lm_head(ctx, err, errcap)) {
            k3_context_destroy(ctx);
            return 0;
        }
        layer_resident_budget -= head_capacity;
    }
    if (!ctx->model->gguf &&
        !want_q8_static && static_staging_requested() &&
        layer_resident_budget != 0 &&
        all_layers_capacity != 0 &&
        layer_resident_budget < all_layers_capacity) {
        if (streaming_slots_capacity >= layer_resident_budget) {
            layer_resident_budget = 0;
        } else {
            layer_resident_budget -= streaming_slots_capacity;
        }
        fprintf(stderr,
                "k3: reserving %.2f GiB for two layer-major streaming slots\n",
                (double)streaming_slots_capacity / 1073741824.0);
    }
    if (want_q8_static &&
        (!alloc_static_slots(ctx, err, errcap) ||
         !load_q8_static_cache(ctx, err, errcap))) {
        k3_context_destroy(ctx);
        return 0;
    }
    if (!ctx->model->gguf &&
        !want_q8_static && !ctx->gguf_resident &&
        layer_resident_budget != 0) {
        if (!alloc_resident_static_slots(
                ctx, layer_resident_budget, err, errcap) ||
            (ctx->resident_static &&
             !load_resident_static_slots(ctx, err, errcap))) {
            k3_context_destroy(ctx);
            return 0;
        }
    }
    if (!want_q8_static && !ctx->gguf_resident &&
        (ctx->resident_lm_head.data || ctx->resident_static) &&
        !pin_resident_static_slots(ctx, err, errcap)) {
        k3_context_destroy(ctx);
        return 0;
    }
    int has_nonresident = 0;
    for (uint32_t i = ctx->layer_start; i <= ctx->layer_end; i++) {
        if (!ctx->q8_static && !ctx->resident_static_slots[i].data) {
            has_nonresident = 1;
            break;
        }
    }
    if (!ctx->model->gguf && has_nonresident &&
        static_staging_requested() &&
        !alloc_static_slots(ctx, err, errcap)) {
        k3_context_destroy(ctx);
        return 0;
    }
    *out = ctx;
    return 1;
}

int k3_context_clone(k3_context **out,
                     k3_context *shared,
                     uint32_t context_capacity,
                     char *err,
                     size_t errcap) {
    if (out) *out = NULL;
    k3_context *owner = execution_owner(shared);
    if (!out || !owner || !owner->owns_execution ||
        context_capacity == 0 || context_capacity > K3_MAX_CONTEXT ||
        owner->max_prefill_tokens == 0) {
        set_error(err, errcap, "invalid K3 shared-context configuration");
        return 0;
    }

    k3_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        set_error(err, errcap, "out of memory cloning K3 context");
        return 0;
    }
    ctx->shared_owner = owner;
    ctx->model = owner->model;
    ctx->globals = owner->globals;
    memcpy(ctx->layers, owner->layers, sizeof(ctx->layers));
    memcpy(ctx->expert_fds, owner->expert_fds, sizeof(ctx->expert_fds));
    ctx->context_capacity = context_capacity;
    ctx->max_prefill_tokens =
        owner->max_prefill_tokens < context_capacity
            ? owner->max_prefill_tokens : context_capacity;
    ctx->layer_start = owner->layer_start;
    ctx->layer_end = owner->layer_end;
    ctx->has_output = owner->has_output;
    ctx->static_staging = owner->static_staging;
    ctx->resident_static = owner->resident_static;
    ctx->selective_shared = owner->selective_shared;
    ctx->pinned_static = owner->pinned_static;
    ctx->q8_static = owner->q8_static;
    ctx->gguf_resident = owner->gguf_resident;
    ctx->gguf_static_resident = owner->gguf_static_resident;
    ctx->speculative_enabled = owner->speculative_enabled;
    ctx->resident_static_bytes = owner->resident_static_bytes;
    memcpy(ctx->static_slots, owner->static_slots, sizeof(ctx->static_slots));
    memcpy(ctx->resident_static_slots, owner->resident_static_slots,
           sizeof(ctx->resident_static_slots));
    memcpy(ctx->resident_shared_slots, owner->resident_shared_slots,
           sizeof(ctx->resident_shared_slots));
    ctx->resident_lm_head = owner->resident_lm_head;
    memcpy(ctx->q8_layers, owner->q8_layers, sizeof(ctx->q8_layers));
    ctx->scratch = owner->scratch;
    memcpy(ctx->streamed_experts, owner->streamed_experts,
           sizeof(ctx->streamed_experts));
    memcpy(ctx->speculative_experts, owner->speculative_experts,
           sizeof(ctx->speculative_experts));
    memcpy(ctx->speculative_batch, owner->speculative_batch,
           sizeof(ctx->speculative_batch));
    for (uint32_t bank = 0; bank < K3_SPECULATIVE_BANKS; bank++) {
        ctx->speculative_layer[bank] = UINT32_MAX;
    }
    pthread_mutex_lock(&owner->execution_mutex);
    const int states_ok = alloc_states(ctx, err, errcap);
    if (!states_ok) {
        free_states(ctx);
    }
    pthread_mutex_unlock(&owner->execution_mutex);
    if (!states_ok) {
        free(ctx);
        return 0;
    }
    *out = ctx;
    return 1;
}

void k3_context_destroy(k3_context *ctx) {
    if (!ctx) return;
    k3_context *owner = execution_owner(ctx);
    if (!ctx->owns_execution) {
        if (owner && owner->execution_mutex_initialized) {
            pthread_mutex_lock(&owner->execution_mutex);
        }
        (void)finish_all_speculative_experts(ctx, NULL, 0);
        (void)ds4_gpu_synchronize();
        free_states(ctx);
        if (owner && owner->execution_mutex_initialized) {
            pthread_mutex_unlock(&owner->execution_mutex);
        }
        free(ctx);
        return;
    }
    (void)finish_all_speculative_experts(ctx, NULL, 0);
    (void)ds4_gpu_synchronize();
    expert_io_pool_shutdown(ctx);
    free_states(ctx);
    for (uint32_t bank = 0; bank < K3_SPECULATIVE_BANKS; bank++) {
        free_streamed_expert_bank(ctx->speculative_experts[bank]);
    }
    free_streamed_expert_bank(ctx->streamed_experts);
    for (uint32_t bank = 0; bank < K3_SPECULATIVE_BANKS; bank++) {
        free(ctx->speculative_batch[bank]);
        ctx->speculative_batch[bank] = NULL;
    }
    free_scratch(&ctx->scratch);
    free_q8_static_cache(ctx);
    ds4_gpu_cleanup();
    free_static_slots(ctx);
    free_resident_static_slots(ctx);
    for (uint32_t i = 0; i < K3_N_LAYER; i++) {
        if (ctx->expert_fds[i] >= 0) close(ctx->expert_fds[i]);
        k3_layer_close(ctx->layers[i]);
    }
    k3_global_weights_close(&ctx->globals);
    k3_model_close(ctx->model);
    if (ctx->execution_mutex_initialized) {
        pthread_mutex_destroy(&ctx->execution_mutex);
    }
    free(ctx);
}

int k3_context_reset(k3_context *ctx, char *err, size_t errcap) {
    if (!ctx) {
        set_error(err, errcap, "invalid K3 reset request");
        return 0;
    }
    k3_context *owner = execution_owner(ctx);
    pthread_mutex_lock(&owner->execution_mutex);
    int ok = finish_all_speculative_experts(ctx, err, errcap) &&
             ds4_gpu_synchronize();
    const uint64_t conv_count =
        3ull * (K3_CONV_WIDTH - 1u) * K3_KDA_PROJ;
    const uint64_t recurrent_count =
        (uint64_t)K3_KDA_PROJ * K3_HEAD_DIM;
    for (uint32_t i = ctx->layer_start;
         ok && i <= ctx->layer_end;
         i++) {
        if (k3_layer_is_mla(i)) continue;
        ok = ds4_gpu_tensor_fill_f32(ctx->kda[i].conv, 0.0f, conv_count) &&
             ds4_gpu_tensor_fill_f32(
                 ctx->kda[i].recurrent, 0.0f, recurrent_count);
    }
    if (ok) {
        ctx->position = 0;
        memset(ctx->previous_route_valid, 0,
               sizeof(ctx->previous_route_valid));
        memset(ctx->predicted_route_valid, 0,
               sizeof(ctx->predicted_route_valid));
        for (uint32_t bank = 0; bank < K3_SPECULATIVE_BANKS; bank++) {
            ctx->speculative_layer[bank] = UINT32_MAX;
            ctx->speculative_count[bank] = 0;
        }
    } else if (!err || !err[0]) {
        set_error(err, errcap, "K3 state reset failed");
    }
    pthread_mutex_unlock(&owner->execution_mutex);
    return ok;
}

int k3_context_prefill(k3_context *ctx,
                       const int32_t *token_ids,
                       uint32_t n_tokens,
                       float *logits,
                       char *err,
                       size_t errcap) {
    if (!ctx) {
        set_error(err, errcap, "invalid K3 prefill context");
        return 0;
    }
    k3_context *owner = execution_owner(ctx);
    pthread_mutex_lock(&owner->execution_mutex);
    const int ok = evaluate(
        ctx, token_ids, n_tokens, 1,
        NULL, NULL, 1, logits, err, errcap);
    pthread_mutex_unlock(&owner->execution_mutex);
    return ok;
}

int k3_context_decode(k3_context *ctx,
                      int32_t token_id,
                      float *logits,
                      char *err,
                      size_t errcap) {
    if (!ctx) {
        set_error(err, errcap, "invalid K3 decode context");
        return 0;
    }
    k3_context *owner = execution_owner(ctx);
    pthread_mutex_lock(&owner->execution_mutex);
    const int ok = evaluate(
        ctx, &token_id, 1, 0,
        NULL, NULL, 1, logits, err, errcap);
    pthread_mutex_unlock(&owner->execution_mutex);
    return ok;
}

int k3_context_eval_slice(k3_context *ctx,
                          const int32_t *token_ids,
                          uint32_t n_tokens,
                          uint32_t pos0,
                          uint32_t layer_start,
                          uint32_t layer_end,
                          const float *input_boundary,
                          float *output_boundary,
                          bool output_logits,
                          float *logits,
                          char *err,
                          size_t errcap) {
    if (!ctx || layer_start != ctx->layer_start ||
        layer_end != ctx->layer_end || pos0 != ctx->position) {
        set_error(err, errcap, "invalid K3 layer-slice request");
        return 0;
    }
    k3_context *owner = execution_owner(ctx);
    pthread_mutex_lock(&owner->execution_mutex);
    const int prefill = n_tokens > 1u || pos0 == 0u;
    const int ok = evaluate(
        ctx, token_ids, n_tokens, prefill,
        input_boundary, output_boundary,
        output_logits ? 1 : 0, logits, err, errcap);
    pthread_mutex_unlock(&owner->execution_mutex);
    return ok;
}

uint32_t k3_context_position(const k3_context *ctx) {
    return ctx ? ctx->position : 0;
}

void k3_context_set_hidden_hook(k3_context *ctx,
                                k3_hidden_hook_fn fn,
                                void *ud) {
    k3_context *owner = execution_owner(ctx);
    if (!owner) return;
    pthread_mutex_lock(&owner->execution_mutex);
    owner->hidden_hook = fn;
    owner->hidden_hook_ud = fn ? ud : NULL;
    pthread_mutex_unlock(&owner->execution_mutex);
}

uint32_t k3_context_capacity(const k3_context *ctx) {
    return ctx ? ctx->context_capacity : 0;
}

uint32_t k3_context_prefill_capacity(const k3_context *ctx) {
    return ctx ? ctx->max_prefill_tokens : 0;
}
