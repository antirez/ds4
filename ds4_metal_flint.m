/*
 * Metal glue for the Recursive Flint family (fork-owned).
 *
 * Owns the decode megakernel's library (metal/flint.metal compiled with
 * strict IEEE math, never into ds4's fast-math library), the model buffer
 * (the mapped GGUF wrapped without a copy), the descriptor and expert
 * address tables, and per-session KV blocks and scratch.  Built on the
 * narrow ds4_metal_internal.h surface, so upstream changes to ds4_metal.m
 * internals do not touch this file.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#include "ds4_metal_internal.h"
#include "ds4_flint_gpu.h"
#include "ds4_flint_stream.h"

#define FLINT_WORKERS 36u
#define FLINT_THREADS 256u
#define FLINT_WORKSPACE_ELEMS 78848u
#define FLINT_PARTIALS_ELEMS (32u * 1024u * 128u / 2u)
#define FLINT_STATISTICS_ELEMS (32u * 1024u * 2u)
#define FLINT_ROUTING_WORDS 1280u
#define FLINT_CONTROL_WORDS 1024u
#define FLINT_N_GLOBAL 14u
#define FLINT_N_SLIDING 35u
#define FLINT_TOKEN_ELEMS 1024u   /* 2 (K,V) x 4 heads x 128 per position */
#define FLINT_PACK_BYTES 1769472ull

static id<MTLDevice> g_flint_device;
static id<MTLLibrary> g_flint_library;
static id<MTLComputePipelineState> g_flint_decode;
static id<MTLBuffer> g_flint_model;
static uint64_t g_flint_model_inner;
static uint64_t g_flint_model_size;
static id<MTLBuffer> g_flint_experts;
static id<MTLBuffer> g_flint_silu;
static uint32_t g_flint_meta[DS4_FLINT_GPU_META_WORDS];
static const char *g_flint_profile;
/* SSD streaming (ds4_flint_stream.c): the cache and staging buffers are
 * reached through GPU addresses, so every encoder marks them resident. */
static ds4_flint_stream *g_flint_stream;
static NSMutableArray<id<MTLBuffer>> *g_flint_stream_bufs;
#define FLINT_REQUEST_WORDS 393u
#define FLINT_MISS_WORD 392u

/* Keep-alive for streaming decode (flint_keepalive in flint.metal, after
 * ds4's TP keep-alive): one threadgroup of 2000 FMA pairs (about 30 us)
 * dispatched back-to-back on its own queue while a streaming session
 * decodes.  Every relaunch waits 0.3 to 1 ms for its fetch, which makes the
 * GPU look idle to the power manager; without this every launch runs
 * 1.7x slower and commit-to-start and completion latencies double.
 * Measured 2026-11-11 (M4 Pro 48 GB, 300 tokens): 23 -> 36 tok/s with an
 * 8 GB cache, 27 -> 41 at 14 GB, 22 -> 38 at 22 GB.  Long dispatches hurt
 * (30000 iterations: 31 tok/s) because command buffers on two queues do not
 * overlap here, so a launch waits for the keep-alive in flight; more
 * threadgroups gain nothing.  DS4_FLINT_STREAM_KEEPALIVE=0 turns it off,
 * =N uses N threadgroups, DS4_FLINT_STREAM_KEEPALIVE_ITERS sets the loop. */
static id<MTLComputePipelineState> g_flint_keepalive;
static id<MTLCommandQueue> g_flint_keepalive_queue;
static id<MTLBuffer> g_flint_keepalive_buffer;
static pthread_t g_flint_keepalive_thread;
static int g_flint_keepalive_running;
static volatile int g_flint_keepalive_stop, g_flint_keepalive_active;
static uint32_t g_flint_keepalive_tgs = 1, g_flint_keepalive_iters = 2000;

static void *flint_keepalive_thread(void *arg) {
    (void)arg;
    while (!g_flint_keepalive_stop) {
        if (!g_flint_keepalive_active) { usleep(200); continue; }
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_flint_keepalive_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g_flint_keepalive];
            [enc setBuffer:g_flint_keepalive_buffer offset:0 atIndex:0];
            [enc setBytes:&g_flint_keepalive_iters length:sizeof(g_flint_keepalive_iters) atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(g_flint_keepalive_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    return NULL;
}

/* Called at the start of every streaming decode step; the thread spins up
 * once and stays until the engine is released.  Prefill marks it inactive. */
static void flint_keepalive_set_active(int active) {
    if (!g_flint_keepalive_running) {
        if (!active || !g_flint_keepalive) return;
        const char *env = getenv("DS4_FLINT_STREAM_KEEPALIVE");
        if (env && atoi(env) <= 0) return;
        if (env) g_flint_keepalive_tgs = (uint32_t)atoi(env) > 64u ? 64u : (uint32_t)atoi(env);
        if (getenv("DS4_FLINT_STREAM_KEEPALIVE_ITERS")) g_flint_keepalive_iters = (uint32_t)atoi(getenv("DS4_FLINT_STREAM_KEEPALIVE_ITERS"));
        g_flint_keepalive_queue = [g_flint_device newCommandQueue];
        g_flint_keepalive_buffer = [g_flint_device newBufferWithLength:(NSUInteger)g_flint_keepalive_tgs * 256u * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
        if (!g_flint_keepalive_queue || !g_flint_keepalive_buffer) return;
        memset([g_flint_keepalive_buffer contents], 0, (size_t)g_flint_keepalive_tgs * 256u * sizeof(float));
        g_flint_keepalive_stop = 0;
        g_flint_keepalive_active = 1;
        if (pthread_create(&g_flint_keepalive_thread, NULL, flint_keepalive_thread, NULL) != 0) return;
        g_flint_keepalive_running = 1;
        return;
    }
    g_flint_keepalive_active = active;
}

static void flint_keepalive_release(void) {
    if (!g_flint_keepalive_running) return;
    g_flint_keepalive_stop = 1;
    pthread_join(g_flint_keepalive_thread, NULL);
    g_flint_keepalive_running = 0;
    g_flint_keepalive_queue = nil;
    g_flint_keepalive_buffer = nil;
}

static void flint_use_resources(id<MTLComputeCommandEncoder> enc) {
    [enc useResource:g_flint_model usage:MTLResourceUsageRead];
    for (id<MTLBuffer> b in g_flint_stream_bufs) [enc useResource:b usage:MTLResourceUsageRead];
}

/* An owned bank for the resident weights (streaming mode copies every
 * non-expert tensor into it and drops the file mapping). */
void *ds4_flint_gpu_bank_alloc(uint64_t bytes) {
    @autoreleasepool {
        if (!g_flint_device) g_flint_device = ds4_metal_device();
        if (!g_flint_device || g_flint_model) return NULL;
        id<MTLBuffer> b = [g_flint_device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
        if (!b) return NULL;
        b.label = @"ds4_flint_bank";
        g_flint_model = b;
        g_flint_model_inner = 0;
        g_flint_model_size = bytes;
        return [b contents];
    }
}

int ds4_flint_gpu_region_alloc(void *ud, uint64_t bytes, const char *label, ds4_flint_stream_region *out) {
    (void)ud;
    @autoreleasepool {
        if (!g_flint_device || !out) return 0;
        id<MTLBuffer> b = [g_flint_device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
        if (!b) return 0;
        b.label = [NSString stringWithUTF8String:label ? label : "ds4_flint_region"];
        uint64_t address = 0;
#if TARGET_OS_OSX
        if (@available(macOS 13.0, *)) address = (uint64_t)[b gpuAddress];
#endif
        if (!address) return 0;
        if (!g_flint_stream_bufs) g_flint_stream_bufs = [NSMutableArray array];
        [g_flint_stream_bufs addObject:b];
        out->ptr = [b contents];
        out->gpu = address;
        return 1;
    }
}

uint64_t *ds4_flint_gpu_expert_table(void) {
    return g_flint_experts ? (uint64_t *)[g_flint_experts contents] : NULL;
}

void ds4_flint_gpu_set_stream(ds4_flint_stream *st) {
    g_flint_stream = st;
}

/* rec's profile table: chip name by Metal device name and architecture. */
const char *ds4_flint_gpu_device_profile(const char **why) {
    if (g_flint_profile) return g_flint_profile;
    id<MTLDevice> dev = ds4_metal_device();
    if (!dev) {
        if (why) *why = "Metal is not initialized";
        return NULL;
    }
    const char *name = [[dev name] UTF8String];
    const char *arch = "";
    if (@available(macOS 14.0, *)) {
        arch = [[[dev architecture] name] UTF8String];
    }
    static char reason[256];
    if (!strcmp(name, "Apple M4 Pro") && !strcmp(arch, "applegpu_g16s")) {
        g_flint_profile = "M4 Pro";
    } else if (!strcmp(name, "Apple M4 Max") && !strcmp(arch, "applegpu_g16s")) {
        g_flint_profile = "M4 Max";
    } else {
        snprintf(reason, sizeof(reason),
                 "no resident decode profile for %s (%s); the kernel is tuned for the M4 Pro launch envelope",
                 name, arch);
        if (why) *why = reason;
        return NULL;
    }
    return g_flint_profile;
}

static NSString *flint_read_source(const char **why) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSMutableArray<NSString *> *paths = [NSMutableArray array];
    const char *override_path = getenv("DS4_METAL_FLINT_SOURCE");
    if (override_path && override_path[0]) [paths addObject:[NSString stringWithUTF8String:override_path]];
    [paths addObject:@"metal/flint.metal"];
    [paths addObject:@"./metal/flint.metal"];
    for (NSString *path in paths) {
        if (![fm fileExistsAtPath:path]) continue;
        NSError *error = nil;
        NSString *loaded = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
        if (!loaded) {
            if (why) *why = "could not read metal/flint.metal";
            return nil;
        }
        return loaded;
    }
    if (why) *why = "metal/flint.metal not found (set DS4_METAL_FLINT_SOURCE)";
    return nil;
}

static int flint_compile(const char **why) {
    if (g_flint_decode) return 1;
    @autoreleasepool {
        NSString *source = flint_read_source(why);
        if (!source) return 0;
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            /* DS4_FLINT_FAST_MATH=1 is a diagnostic: rec compiles safe. */
            options.mathMode = getenv("DS4_FLINT_FAST_MATH") ? MTLMathModeFast : MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        /* DS4_FLINT_KV_PREFETCH=N overrides the kernel's KV prefetch depth
         * (metal/flint.metal, FLINT_KV_PREFETCH) for timing sweeps. */
        const char *pf_env = getenv("DS4_FLINT_KV_PREFETCH");
        if (pf_env && atoi(pf_env) > 0) options.preprocessorMacros = @{ @"FLINT_KV_PREFETCH": @(atoi(pf_env)) };
        NSError *error = nil;
        g_flint_library = [g_flint_device newLibraryWithSource:source options:options error:&error];
        if (!g_flint_library) {
            fprintf(stderr, "ds4: Flint Metal library failed to compile: %s\n",
                    [[error localizedDescription] UTF8String]);
            if (why) *why = "metal/flint.metal does not compile";
            return 0;
        }
        id<MTLFunction> fn = [g_flint_library newFunctionWithName:@"flint_decode"];
        if (!fn) {
            if (why) *why = "flint_decode is missing from the library";
            return 0;
        }
        g_flint_decode = [g_flint_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_flint_decode) {
            fprintf(stderr, "ds4: Flint decode pipeline failed: %s\n", [[error localizedDescription] UTF8String]);
            if (why) *why = "flint_decode pipeline creation failed";
            return 0;
        }
        g_flint_keepalive = nil;
        id<MTLFunction> ka = [g_flint_library newFunctionWithName:@"flint_keepalive"];
        if (ka) g_flint_keepalive = [g_flint_device newComputePipelineStateWithFunction:ka error:&error];
        if (g_flint_decode.maxTotalThreadsPerThreadgroup < FLINT_THREADS) {
            if (why) *why = "flint_decode cannot run 256 threads per threadgroup";
            return 0;
        }
        if (getenv("DS4_FLINT_DECODE_PROFILE") || getenv("DS4_FLINT_KV_PREFETCH"))
            fprintf(stderr, "ds4: flint_decode pipeline: max %lu threads per threadgroup, %lu bytes of threadgroup memory\n",
                    (unsigned long)g_flint_decode.maxTotalThreadsPerThreadgroup,
                    (unsigned long)g_flint_decode.staticThreadgroupMemoryLength);
        return 1;
    }
}

static int flint_wrap_model(const void *model_map, uint64_t model_size, const char **why) {
    const uint64_t page = (uint64_t)getpagesize();
    const uintptr_t base = (uintptr_t)model_map;
    const uint64_t leading = base & (page - 1);
    uint64_t bytes = (leading + model_size + page - 1) / page * page;
    if (bytes > leading + model_size) bytes = leading + model_size;
    if (bytes > (uint64_t)[g_flint_device maxBufferLength]) {
        if (why) *why = "the model does not fit one Metal buffer on this device";
        return 0;
    }
    g_flint_model = [g_flint_device newBufferWithBytesNoCopy:(void *)(base - leading)
                                                       length:(NSUInteger)bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    if (!g_flint_model) {
        if (why) *why = "Metal could not wrap the mapped model";
        return 0;
    }
    g_flint_model.label = @"ds4_flint_model";
    g_flint_model_inner = leading;
    g_flint_model_size = model_size;
    return 1;
}

int ds4_flint_gpu_engine_open(const void *model_map, uint64_t model_size,
                              const uint32_t *metadata,
                              const uint64_t *expert_pack_offsets,
                              const uint16_t *silu_bf16, const char **why) {
    @autoreleasepool {
        g_flint_device = ds4_metal_device();
        if (!g_flint_device) {
            if (why) *why = "Metal is not initialized";
            return 0;
        }
        if (!ds4_flint_gpu_device_profile(why)) return 0;
        if (!flint_compile(why)) return 0;
        if (!g_flint_model && !flint_wrap_model(model_map, model_size, why)) return 0;
        memcpy(g_flint_meta, metadata, sizeof(g_flint_meta));
        /* The model buffer starts `inner` bytes before the map; descriptor
         * offsets are relative to the map, so shift them (all 256-aligned). */
        if (g_flint_model_inner % 256u) {
            if (why) *why = "the model map is not 256-byte aligned";
            return 0;
        }
        const uint32_t shift = (uint32_t)(g_flint_model_inner >> 8);
        for (uint32_t row = 0; row < 50; row++) {
            uint32_t *d = g_flint_meta + row * 64;
            static const uint8_t offset_words[] = {3, 4, 5, 6, 7, 60};
            for (size_t i = 0; i < sizeof(offset_words); i++) {
                if (d[offset_words[i]]) d[offset_words[i]] += shift;
            }
            for (uint32_t base = 10; base <= 55; base += 5) {
                if (d[base + 3] == 0) continue;   /* unused block */
                d[base] += shift; d[base + 1] += shift; d[base + 2] += shift;
            }
        }
        const uint64_t table_words = (uint64_t)DS4_FLINT_GPU_N_LAYER * DS4_FLINT_GPU_N_EXPERT;
        g_flint_experts = [g_flint_device newBufferWithLength:table_words * sizeof(uint64_t)
                                                      options:MTLResourceStorageModeShared];
        g_flint_silu = [g_flint_device newBufferWithLength:DS4_FLINT_GPU_SILU_ENTRIES * sizeof(uint16_t)
                                                   options:MTLResourceStorageModeShared];
        if (!g_flint_experts || !g_flint_silu) {
            if (why) *why = "out of Metal memory for the Flint tables";
            return 0;
        }
        g_flint_experts.label = @"ds4_flint_experts";
        g_flint_silu.label = @"ds4_flint_silu";
        uint64_t model_address = 0;
#if TARGET_OS_OSX
        if (@available(macOS 13.0, *)) model_address = (uint64_t)[g_flint_model gpuAddress];
#endif
        if (!model_address) {
            if (why) *why = "Metal buffers have no GPU address on this macOS";
            return 0;
        }
        uint64_t *table = (uint64_t *)[g_flint_experts contents];
        for (uint64_t i = 0; i < table_words; i++) {
            table[i] = expert_pack_offsets[i] ? model_address + g_flint_model_inner + expert_pack_offsets[i] : 0;
        }
        memcpy([g_flint_silu contents], silu_bf16, DS4_FLINT_GPU_SILU_ENTRIES * sizeof(uint16_t));
        return 1;
    }
}

static void flint_prefill_scratch_release(void);
void ds4_flint_gpu_engine_release(void) {
    flint_keepalive_release();
    g_flint_keepalive = nil;
    flint_prefill_scratch_release();
    g_flint_stream = NULL;
    g_flint_stream_bufs = nil;
    g_flint_experts = nil;
    g_flint_silu = nil;
    g_flint_model = nil;
    g_flint_model_inner = 0;
    g_flint_model_size = 0;
}

struct ds4_flint_gpu_session {
    uint32_t ctx_cap;
    uint32_t cap_global;
    uint32_t n_blocks;
    uint64_t block_elems[DS4_FLINT_GPU_MAX_BLOCKS];
    uint32_t kv_offset[DS4_FLINT_GPU_N_LAYER];   /* elements, from the block start */
    uint32_t kv_cap[DS4_FLINT_GPU_N_LAYER];
    uint8_t  kv_global[DS4_FLINT_GPU_N_LAYER];
    id<MTLBuffer> kv[DS4_FLINT_GPU_MAX_BLOCKS];
    id<MTLBuffer> metadata;
    id<MTLBuffer> workspace, partials, statistics, routing;
    id<MTLBuffer> logits, trace, diagnostic, control, requests;
    int broken;
};

static void flint_session_layout(uint32_t ctx_cap, struct ds4_flint_gpu_session *s) {
    s->ctx_cap = ctx_cap;
    s->cap_global = ctx_cap < DS4_FLINT_GPU_BLOCK_TOKENS ? ctx_cap : DS4_FLINT_GPU_BLOCK_TOKENS;
    s->n_blocks = (ctx_cap + DS4_FLINT_GPU_BLOCK_TOKENS - 1) / DS4_FLINT_GPU_BLOCK_TOKENS;
    if (s->n_blocks == 0) s->n_blocks = 1;
    const uint64_t global_elems = (uint64_t)FLINT_N_GLOBAL * s->cap_global * FLINT_TOKEN_ELEMS;
    uint32_t gi = 0, li = 0;
    for (uint32_t il = 0; il < DS4_FLINT_GPU_N_LAYER; il++) {
        const uint32_t *d = g_flint_meta + il * 64;
        s->kv_global[il] = d[1] != 0;
        if (d[1]) {
            s->kv_offset[il] = (uint32_t)((uint64_t)gi * s->cap_global * FLINT_TOKEN_ELEMS);
            s->kv_cap[il] = s->cap_global;
            gi++;
        } else {
            s->kv_offset[il] = (uint32_t)(global_elems + (uint64_t)li * DS4_FLINT_GPU_SWA * FLINT_TOKEN_ELEMS);
            s->kv_cap[il] = DS4_FLINT_GPU_SWA;
            li++;
        }
    }
    s->block_elems[0] = global_elems + (uint64_t)FLINT_N_SLIDING * DS4_FLINT_GPU_SWA * FLINT_TOKEN_ELEMS;
    for (uint32_t b = 1; b < DS4_FLINT_GPU_MAX_BLOCKS; b++) s->block_elems[b] = b < s->n_blocks ? global_elems : 0;
}

uint64_t ds4_flint_gpu_session_bytes(uint32_t ctx_cap) {
    struct ds4_flint_gpu_session s = {0};
    flint_session_layout(ctx_cap, &s);
    uint64_t bytes = 0;
    for (uint32_t b = 0; b < DS4_FLINT_GPU_MAX_BLOCKS; b++) bytes += s.block_elems[b] * 2u;
    bytes += (uint64_t)FLINT_WORKSPACE_ELEMS * 2 + (uint64_t)FLINT_PARTIALS_ELEMS * 4 +
             (uint64_t)FLINT_STATISTICS_ELEMS * 4 + (uint64_t)FLINT_ROUTING_WORDS * 4 +
             (uint64_t)DS4_FLINT_GPU_N_VOCAB * 2 + (uint64_t)DS4_FLINT_GPU_TRACE_ELEMS * 2 +
             (uint64_t)DS4_FLINT_GPU_META_WORDS * 4 + FLINT_CONTROL_WORDS * 4 + FLINT_WORKERS * 4 +
             FLINT_REQUEST_WORDS * 4;
    return bytes;
}

static double ds4_flint_now_s(void) {
    return (double)clock_gettime_nsec_np(CLOCK_MONOTONIC) * 1e-9;
}

static id<MTLBuffer> flint_buffer(uint64_t bytes, const char *label, int zero) {
    id<MTLBuffer> b = [g_flint_device newBufferWithLength:(NSUInteger)(bytes ? bytes : 16)
                                                  options:MTLResourceStorageModeShared];
    if (!b) {
        fprintf(stderr, "ds4: Flint could not allocate %s (%.1f MiB)\n", label, (double)bytes / 1048576.0);
        return nil;
    }
    b.label = [NSString stringWithUTF8String:label];
    if (zero) memset([b contents], 0, (size_t)(bytes ? bytes : 16));
    return b;
}

ds4_flint_gpu_session *ds4_flint_gpu_session_create(uint32_t ctx_cap) {
    if (!g_flint_decode || !g_flint_model) {
        fprintf(stderr, "ds4: Flint GPU engine is not open\n");
        return NULL;
    }
    if (ctx_cap == 0 || ctx_cap > DS4_FLINT_GPU_MAX_CTX) {
        fprintf(stderr, "ds4: Flint context capacity %u is outside 1..%u\n", ctx_cap, DS4_FLINT_GPU_MAX_CTX);
        return NULL;
    }
    @autoreleasepool {
        struct ds4_flint_gpu_session *s = calloc(1, sizeof(*s));
        if (!s) return NULL;
        flint_session_layout(ctx_cap, s);
        for (uint32_t b = 0; b < DS4_FLINT_GPU_MAX_BLOCKS; b++) {
            char label[32];
            snprintf(label, sizeof(label), "ds4_flint_kv%u", b);
            s->kv[b] = flint_buffer(s->block_elems[b] * 2u, label, 1);
            if (!s->kv[b]) goto fail;
        }
        s->metadata = flint_buffer(DS4_FLINT_GPU_META_WORDS * 4, "ds4_flint_metadata", 0);
        s->workspace = flint_buffer((uint64_t)FLINT_WORKSPACE_ELEMS * 2, "ds4_flint_workspace", 1);
        s->partials = flint_buffer((uint64_t)FLINT_PARTIALS_ELEMS * 4, "ds4_flint_partials", 1);
        s->statistics = flint_buffer((uint64_t)FLINT_STATISTICS_ELEMS * 4, "ds4_flint_statistics", 1);
        s->routing = flint_buffer((uint64_t)FLINT_ROUTING_WORDS * 4, "ds4_flint_routing", 1);
        s->logits = flint_buffer((uint64_t)DS4_FLINT_GPU_N_VOCAB * 2, "ds4_flint_logits", 1);
        s->trace = flint_buffer((uint64_t)DS4_FLINT_GPU_TRACE_ELEMS * 2, "ds4_flint_trace", 1);
        s->diagnostic = flint_buffer(FLINT_WORKERS * 4, "ds4_flint_diagnostic", 1);
        s->control = flint_buffer(FLINT_CONTROL_WORDS * 4, "ds4_flint_control", 1);
        s->requests = flint_buffer(FLINT_REQUEST_WORDS * 4, "ds4_flint_requests", 1);
        if (!s->metadata || !s->workspace || !s->partials || !s->statistics || !s->routing ||
            !s->logits || !s->trace || !s->diagnostic || !s->control || !s->requests) {
            goto fail;
        }
        uint32_t *meta = (uint32_t *)[s->metadata contents];
        memcpy(meta, g_flint_meta, sizeof(g_flint_meta));
        for (uint32_t il = 0; il < DS4_FLINT_GPU_N_LAYER; il++) {
            meta[il * 64 + 61] = s->kv_offset[il];
            meta[il * 64 + 62] = s->kv_cap[il];
        }
        return s;
    fail:
        ds4_flint_gpu_session_free(s);
        return NULL;
    }
}

void ds4_flint_gpu_session_free(ds4_flint_gpu_session *s) {
    if (!s) return;
    for (uint32_t b = 0; b < DS4_FLINT_GPU_MAX_BLOCKS; b++) s->kv[b] = nil;
    s->metadata = nil; s->workspace = nil; s->partials = nil; s->statistics = nil; s->routing = nil;
    s->logits = nil; s->trace = nil; s->diagnostic = nil; s->control = nil; s->requests = nil;
    free(s);
}

uint32_t ds4_flint_gpu_session_ctx_cap(const ds4_flint_gpu_session *s) {
    return s ? s->ctx_cap : 0;
}

/* Element address of one KV row: block and offset within it. */
static int flint_kv_locate(const ds4_flint_gpu_session *s, uint32_t layer, int key, uint32_t head,
                           uint32_t row, uint32_t *block, uint64_t *elem) {
    if (!s || layer >= DS4_FLINT_GPU_N_LAYER || head >= 4) return 0;
    const uint32_t cap = s->kv_cap[layer];
    uint32_t b = 0;
    uint32_t local = row;
    if (s->kv_global[layer]) {
        b = row / DS4_FLINT_GPU_BLOCK_TOKENS;
        local = row % DS4_FLINT_GPU_BLOCK_TOKENS;
        if (b >= s->n_blocks || local >= cap) return 0;
    } else if (row >= cap) {
        return 0;
    }
    *block = b;
    *elem = (uint64_t)s->kv_offset[layer] + (key ? 0u : (uint64_t)cap * 512u) +
            (uint64_t)head * cap * 128u + (uint64_t)local * 128u;
    return 1;
}

int ds4_flint_gpu_kv_write(ds4_flint_gpu_session *s, uint32_t layer, int key, uint32_t head,
                           uint32_t row0, uint32_t n_rows, const uint16_t *rows) {
    for (uint32_t r = 0; r < n_rows; r++) {
        uint32_t block; uint64_t elem;
        if (!flint_kv_locate(s, layer, key, head, row0 + r, &block, &elem)) return 0;
        memcpy((uint16_t *)[s->kv[block] contents] + elem, rows + (size_t)r * 128u, 128u * sizeof(uint16_t));
    }
    return 1;
}

int ds4_flint_gpu_kv_read(const ds4_flint_gpu_session *s, uint32_t layer, int key, uint32_t head,
                          uint32_t row0, uint32_t n_rows, uint16_t *rows) {
    for (uint32_t r = 0; r < n_rows; r++) {
        uint32_t block; uint64_t elem;
        if (!flint_kv_locate(s, layer, key, head, row0 + r, &block, &elem)) return 0;
        memcpy(rows + (size_t)r * 128u, (const uint16_t *)[s->kv[block] contents] + elem, 128u * sizeof(uint16_t));
    }
    return 1;
}

int ds4_flint_gpu_decode(ds4_flint_gpu_session *s, uint32_t token, uint32_t position,
                         uint32_t initial_position, uint16_t *logits_bf16, uint16_t *trace_bf16,
                         char *err, size_t errlen) {
    if (!s || s->broken) {
        if (err && errlen) snprintf(err, errlen, "Flint session is unusable after a GPU synchronization failure");
        return 0;
    }
    if (position >= s->ctx_cap) {
        if (err && errlen) snprintf(err, errlen, "position %u exceeds the session capacity %u", position, s->ctx_cap);
        return 0;
    }
    @autoreleasepool {
        /* Diagnostics for the trace test: DS4_FLINT_LAYER_COUNT runs the first N
         * layers only; DS4_FLINT_WORKSPACE_DUMP appends the workspace after each
         * step; DS4_FLINT_TRACE_DUMP appends the residual trace and
         * DS4_FLINT_ROUTE_LOG the position and the 49 x 8 request words (the
         * routing of every sparse layer) per token. */
        const char *lc_env = getenv("DS4_FLINT_LAYER_COUNT");
        const uint32_t layer_count = lc_env ? (uint32_t)atoi(lc_env) : DS4_FLINT_GPU_N_LAYER;
        const uint32_t streaming = g_flint_stream ? 1u : 0u;
        uint32_t layer_start = 0;
        uint32_t relaunches = 0;
        uint32_t ids_seen[DS4_FLINT_GPU_N_LAYER * 8];
        memset(ids_seen, 0, sizeof(ids_seen));
        /* DS4_FLINT_DECODE_PROFILE=1: GPU time of the launch against the wall
         * time of the step (command buffer creation to completion), averaged
         * over 64 steps and printed with the position. */
        static int dprof = -1;
        static double dprof_gpu = 0, dprof_wall = 0;
        static uint32_t dprof_n = 0;
        if (dprof < 0) dprof = getenv("DS4_FLINT_DECODE_PROFILE") ? 1 : 0;
        /* DS4_FLINT_STREAM_PROFILE=1: per token (all launches), averaged over
         * 64 tokens: wall ms, launches, misses and read ms. */
        static int sprof = -1;
        static double sprof_wall = 0, sprof_read = 0;
        static double sprof_prep = 0, sprof_queue = 0, sprof_gpu = 0, sprof_wake = 0, sprof_fetch = 0, sprof_host = 0;
        static double sprof_gpu_abort = 0, sprof_gpu_final = 0;
        static uint64_t sprof_layers_abort = 0, sprof_layers_final = 0, sprof_n_abort = 0;
        static uint64_t sprof_launch = 0, sprof_miss = 0, sprof_n = 0, sprof_last_fetch = 0;
        if (sprof < 0) sprof = getenv("DS4_FLINT_STREAM_PROFILE") && g_flint_stream ? atoi(getenv("DS4_FLINT_STREAM_PROFILE")) : 0;
        ds4_flint_stream_stats sst0;
        const double t_tok0 = sprof ? ds4_flint_now_s() : 0;
        sprof_last_fetch = 0;
        if (sprof) ds4_flint_stream_get_stats(g_flint_stream, &sst0);
        double t_prev_end = sprof ? ds4_flint_now_s() : 0;
        if (g_flint_stream) flint_keepalive_set_active(1);
        for (;;) {
            const double t_wall0 = (dprof || sprof) ? ds4_flint_now_s() : 0;
            memset([s->control contents], 0, FLINT_CONTROL_WORDS * 4);
            memset([s->diagnostic contents], 0, FLINT_WORKERS * 4);
            memset([s->requests contents], 0, FLINT_REQUEST_WORDS * 4);
            const uint32_t parameters[8] = { token, position, initial_position, layer_count, layer_start, streaming, 0, 0 };
            int owned = 0;
            id<MTLCommandBuffer> cb = ds4_metal_command_buffer(&owned);
            if (!cb) {
                if (err && errlen) snprintf(err, errlen, "no Metal command buffer");
                return 0;
            }
            id<MTLComputeCommandEncoder> enc = ds4_metal_compute_encoder(cb);
            if (!enc) {
                if (err && errlen) snprintf(err, errlen, "no Metal compute encoder");
                return 0;
            }
            [enc setComputePipelineState:g_flint_decode];
            flint_use_resources(enc);
            [enc setBuffer:g_flint_model offset:0 atIndex:0];
            [enc setBuffer:s->metadata offset:0 atIndex:1];
            for (uint32_t b = 0; b < DS4_FLINT_GPU_MAX_BLOCKS; b++) [enc setBuffer:s->kv[b] offset:0 atIndex:2 + b];
            [enc setBytes:parameters length:sizeof(parameters) atIndex:6];
            [enc setBuffer:g_flint_silu offset:0 atIndex:7];
            [enc setBuffer:s->control offset:0 atIndex:8];
            [enc setBuffer:g_flint_experts offset:0 atIndex:9];
            [enc setBuffer:s->logits offset:0 atIndex:10];
            [enc setBuffer:s->trace offset:0 atIndex:11];
            [enc setBuffer:s->diagnostic offset:0 atIndex:12];
            [enc setBuffer:s->workspace offset:0 atIndex:13];
            [enc setBuffer:s->partials offset:0 atIndex:14];
            [enc setBuffer:s->statistics offset:0 atIndex:15];
            [enc setBuffer:s->routing offset:0 atIndex:16];
            [enc setBuffer:s->requests offset:0 atIndex:17];
            [enc dispatchThreadgroups:MTLSizeMake(FLINT_WORKERS, 1, 1) threadsPerThreadgroup:MTLSizeMake(FLINT_THREADS, 1, 1)];
            ds4_metal_end_compute_encoder(cb, enc);
            const double t_commit = sprof ? ds4_flint_now_s() : 0;
            if (!ds4_metal_finish_command_buffer(cb, owned, "flint_decode")) {
                if (err && errlen) snprintf(err, errlen, "flint_decode command buffer failed");
                return 0;
            }
            if (sprof) {
                /* GPUStartTime/GPUEndTime share mach_absolute_time's base; so does CLOCK_UPTIME_RAW */
                const double t_done = ds4_flint_now_s();
                const double up_now = (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9;
                const double skew = up_now - t_done;
                sprof_host += t_wall0 - t_prev_end;
                sprof_prep += t_commit - t_wall0;
                sprof_queue += (cb.GPUStartTime - skew) - t_commit;
                sprof_gpu += cb.GPUEndTime - cb.GPUStartTime;
                sprof_wake += t_done - (cb.GPUEndTime - skew);
                t_prev_end = t_done;
            }
            if (dprof) {
                dprof_gpu += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
                dprof_wall += (ds4_flint_now_s() - t_wall0) * 1e3;
                if (++dprof_n == 64) {
                    fprintf(stderr, "ds4: Flint decode profile at position %u: GPU %.2f ms, wall %.2f ms per step (64 steps)\n",
                            position, dprof_gpu / 64, dprof_wall / 64);
                    dprof_gpu = dprof_wall = 0;
                    dprof_n = 0;
                }
            }
            const uint32_t *req = (const uint32_t *)[s->requests contents];
            const uint32_t miss = req[FLINT_MISS_WORD];
            const uint32_t ran_to = miss ? miss - 1 : layer_count;
            if (sprof) {
                const double g = cb.GPUEndTime - cb.GPUStartTime;
                if (miss) { sprof_gpu_abort += g; sprof_layers_abort += ran_to + 1 - layer_start; sprof_n_abort++; }
                else { sprof_gpu_final += g; sprof_layers_final += ran_to - layer_start; }
                if (sprof >= 3) {
                    const double up2 = (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9 - ds4_flint_now_s();
                    fprintf(stderr, "ds4: L pos %u start %u ran_to %u miss %u gpu %.4f queue %.4f wake %.4f prev_fetch %u\n",
                            position, layer_start, ran_to, miss ? __builtin_popcount(req[(miss - 1) * 8 + 6]) : 0,
                            g * 1e3, ((cb.GPUStartTime - up2) - t_commit) * 1e3, (t_prev_end - (cb.GPUEndTime - up2)) * 1e3,
                            (unsigned)sprof_last_fetch);
                }
            }
            for (uint32_t il = layer_start; il < ran_to && il < DS4_FLINT_GPU_N_LAYER; il++) {
                memcpy(ids_seen + il * 8, req + il * 8, 8 * 4);
                if (g_flint_stream && g_flint_meta[il * 64 + 2]) ds4_flint_stream_touch(g_flint_stream, il, req + il * 8, 6);
            }
            if (miss) {
                const uint32_t il = miss - 1;
                if (!g_flint_stream || il >= DS4_FLINT_GPU_N_LAYER || il < layer_start || ++relaunches > DS4_FLINT_GPU_N_LAYER + 8) {
                    s->broken = 1;
                    if (err && errlen) snprintf(err, errlen, "expert cache miss handling failed at layer %u", il);
                    return 0;
                }
                memcpy(ids_seen + il * 8, req + il * 8, 8 * 4);
                const double t_f0 = sprof ? ds4_flint_now_s() : 0;
                if (!ds4_flint_stream_fetch(g_flint_stream, il, req + il * 8, 6, err, errlen)) return 0;
                if (sprof) { sprof_fetch += ds4_flint_now_s() - t_f0; sprof_last_fetch = (uint32_t)__builtin_popcount(req[il * 8 + 6] & 63u); }
                layer_start = il;
                continue;
            }
            if (sprof) {
                ds4_flint_stream_stats sst1;
                ds4_flint_stream_get_stats(g_flint_stream, &sst1);
                sprof_wall += (ds4_flint_now_s() - t_tok0) * 1e3;
                sprof_launch += relaunches + 1;
                sprof_miss += sst1.decode_misses - sst0.decode_misses;
                sprof_read += sst1.decode_read_ms - sst0.decode_read_ms;
                if (++sprof_n == 64) {
                    fprintf(stderr, "ds4: Flint streaming profile at position %u: %.2f ms/token, %.1f launches, %.1f misses, "
                            "%.2f ms reading per token (64 tokens)\n",
                            position, sprof_wall / 64, sprof_launch / 64.0, sprof_miss / 64.0, sprof_read / 64);
                    if (sprof > 1) {
                        fprintf(stderr, "ds4:   per launch: host %.3f ms, encode %.3f, queue %.3f, GPU %.3f, wake %.3f; fetch %.3f ms per launch\n",
                                sprof_host * 1e3 / sprof_launch, sprof_prep * 1e3 / sprof_launch, sprof_queue * 1e3 / sprof_launch,
                                sprof_gpu * 1e3 / sprof_launch, sprof_wake * 1e3 / sprof_launch, sprof_fetch * 1e3 / sprof_launch);
                        fprintf(stderr, "ds4:   GPU per launch: aborted %.3f ms over %.1f layers (%.3f ms/layer, incl. the aborted one); "
                                "final %.3f ms over %.1f layers (%.3f ms/layer)\n",
                                sprof_gpu_abort * 1e3 / (sprof_n_abort ? sprof_n_abort : 1), (double)sprof_layers_abort / (sprof_n_abort ? sprof_n_abort : 1),
                                sprof_gpu_abort * 1e3 / (sprof_layers_abort ? sprof_layers_abort : 1),
                                sprof_gpu_final * 1e3 / 64, (double)sprof_layers_final / 64, sprof_gpu_final * 1e3 / (sprof_layers_final ? sprof_layers_final : 1));
                    }
                    sprof_wall = sprof_read = sprof_prep = sprof_queue = sprof_gpu = sprof_wake = sprof_fetch = sprof_host = 0;
                    sprof_gpu_abort = sprof_gpu_final = 0;
                    sprof_layers_abort = sprof_layers_final = sprof_n_abort = 0;
                    sprof_launch = sprof_miss = sprof_n = 0;
                }
            }
            const uint32_t *diag = (const uint32_t *)[s->diagnostic contents];
            for (uint32_t g = 0; g < FLINT_WORKERS; g++) {
                if (diag[g]) {
                    s->broken = 1;
                    if (err && errlen) {
                        snprintf(err, errlen, "GPU synchronization failed: group %u timed out in phase %u", g, diag[g]);
                    }
                    return 0;
                }
            }
            break;
        }
        const char *ws_env = getenv("DS4_FLINT_WORKSPACE_DUMP");
        if (ws_env) {
            FILE *f = fopen(ws_env, "ab");
            if (f) { fwrite([s->workspace contents], 2, FLINT_WORKSPACE_ELEMS, f); fclose(f); }
        }
        const char *td_env = getenv("DS4_FLINT_TRACE_DUMP");   /* the 50 x 2048 bf16 residual trace per token */
        if (td_env) {
            FILE *f = fopen(td_env, "ab");
            if (f) { fwrite([s->trace contents], 2, DS4_FLINT_GPU_TRACE_ELEMS, f); fclose(f); }
        }
        const char *rl_env = getenv("DS4_FLINT_ROUTE_LOG");
        if (rl_env) {
            FILE *f = fopen(rl_env, "ab");
            if (f) { fwrite(&position, 4, 1, f); fwrite(ids_seen, 4, DS4_FLINT_GPU_N_LAYER * 8, f); fclose(f); }
        }
        if (logits_bf16) memcpy(logits_bf16, [s->logits contents], DS4_FLINT_GPU_N_VOCAB * sizeof(uint16_t));
        if (trace_bf16) memcpy(trace_bf16, [s->trace contents], DS4_FLINT_GPU_TRACE_ELEMS * sizeof(uint16_t));
        return 1;
    }
}

/* ======================================================================
 * Prefill (metal/flint_prefill.metal): plain kernels over chunks of up to
 * 2048 tokens, following rec's MLX graph operation by operation.  One
 * engine-wide scratch serves every session (sessions are serialized on the
 * queue); it is allocated at the first prefill.
 * ====================================================================== */

#define FLINT_PF_CAP DS4_FLINT_GPU_PREFILL_CAP
#define FLINT_PF_WIN (512u + FLINT_PF_CAP)          /* sliding window rows: previous 512 + chunk */
#define FLINT_PF_MAX_TILES (FLINT_PF_CAP * 6u / 64u + 256u)   /* 64-row tiles */
#define FLINT_PF_TILES_TOTAL (FLINT_PF_MAX_TILES + 3u * 256u)  /* plus one 32/16/8-row remainder tile per expert */

static id<MTLLibrary> g_flint_prefill_lib;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_flint_pipes;
static struct {
    id<MTLBuffer> tokens, x, xn, q, k, v, attn, act, act_sh, router;
    id<MTLBuffer> ids, wts, counts, starts, cursors, tiles, dispatch, rows, act_sorted, y6;
    id<MTLBuffer> win_k, win_v, logits, logits_all, dummy, exp_table;
    uint32_t logits_all_rows;
    int ready;
} g_pf;

typedef struct { uint32_t M, N, K, lda, ldc, ldr; } FpGemmArgs;
typedef struct { uint32_t M, q_pos0, key_pos0, n_keys, kv_head_stride, window, ldq, ldo; } FpAttnArgs;
typedef struct { uint32_t M, pos0, rotary, dst_row0, dst_head_stride; } FpHeadArgs;
typedef struct { uint32_t n_rows, src_head_stride, dst_head_stride, first_pos, ring_initial, ring_side, linear_row0; } FpRingArgs;

static NSString *flint_read_prefill_source(const char **why) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSMutableArray<NSString *> *paths = [NSMutableArray array];
    const char *override_path = getenv("DS4_METAL_FLINT_PREFILL_SOURCE");
    if (override_path && override_path[0]) [paths addObject:[NSString stringWithUTF8String:override_path]];
    [paths addObject:@"metal/flint_prefill.metal"];
    [paths addObject:@"./metal/flint_prefill.metal"];
    for (NSString *path in paths) {
        if (![fm fileExistsAtPath:path]) continue;
        NSString *loaded = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
        if (loaded) return loaded;
    }
    if (why) *why = "metal/flint_prefill.metal not found (set DS4_METAL_FLINT_PREFILL_SOURCE)";
    return nil;
}

static int flint_prefill_compile(const char **why) {
    if (g_flint_prefill_lib) return 1;
    @autoreleasepool {
        NSString *source = flint_read_prefill_source(why);
        if (!source) return 0;
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *error = nil;
        g_flint_prefill_lib = [g_flint_device newLibraryWithSource:source options:options error:&error];
        if (!g_flint_prefill_lib) {
            fprintf(stderr, "ds4: Flint prefill library failed to compile: %s\n", [[error localizedDescription] UTF8String]);
            if (why) *why = "metal/flint_prefill.metal does not compile";
            return 0;
        }
        g_flint_pipes = [NSMutableDictionary dictionary];
        return 1;
    }
}

static id<MTLComputePipelineState> flint_pipe(const char *name) {
    NSString *key = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> p = g_flint_pipes[key];
    if (p) return p;
    id<MTLFunction> fn = [g_flint_prefill_lib newFunctionWithName:key];
    NSError *error = nil;
    p = fn ? [g_flint_device newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (!p) {
        fprintf(stderr, "ds4: Flint prefill pipeline %s failed: %s\n", name,
                error ? [[error localizedDescription] UTF8String] : "function missing");
        return nil;
    }
    g_flint_pipes[key] = p;
    return p;
}

uint64_t ds4_flint_gpu_prefill_scratch_bytes(void) {
    const uint64_t cap = FLINT_PF_CAP;
    return cap * 4 + cap * 2048 * 2 * 2 + cap * 4096 * 2 * 2 + cap * 512 * 2 * 2 + cap * 6144 * 2 + cap * 1024 * 2 +
           cap * 256 * 2 + cap * 6 * (4 + 2 + 4) + 257 * 4 * 3 + FLINT_PF_TILES_TOTAL * 16 + 96 + cap * 6 * 512 * 2 +
           cap * 6 * 2048 * 2 + 2ull * 4 * FLINT_PF_WIN * 128 * 2 + DS4_FLINT_GPU_N_VOCAB * 2;
}

static int flint_prefill_scratch(void) {
    if (g_pf.ready) return 1;
    const uint64_t cap = FLINT_PF_CAP;
    g_pf.tokens = flint_buffer(cap * 4, "ds4_flint_pf_tokens", 0);
    g_pf.x = flint_buffer(cap * 2048 * 2, "ds4_flint_pf_x", 1);
    g_pf.xn = flint_buffer(cap * 2048 * 2, "ds4_flint_pf_xn", 1);
    g_pf.q = flint_buffer(cap * 4096 * 2, "ds4_flint_pf_q", 1);
    g_pf.k = flint_buffer(cap * 512 * 2, "ds4_flint_pf_k", 1);
    g_pf.v = flint_buffer(cap * 512 * 2, "ds4_flint_pf_v", 1);
    g_pf.attn = flint_buffer(cap * 4096 * 2, "ds4_flint_pf_attn", 1);
    g_pf.act = flint_buffer(cap * 6144 * 2, "ds4_flint_pf_act", 1);
    g_pf.act_sh = flint_buffer(cap * 1024 * 2, "ds4_flint_pf_act_sh", 1);
    g_pf.router = flint_buffer(cap * 256 * 2, "ds4_flint_pf_router", 1);
    g_pf.ids = flint_buffer(cap * 6 * 4, "ds4_flint_pf_ids", 1);
    g_pf.wts = flint_buffer(cap * 6 * 2, "ds4_flint_pf_wts", 1);
    g_pf.counts = flint_buffer(256 * 4, "ds4_flint_pf_counts", 1);
    g_pf.starts = flint_buffer(257 * 4, "ds4_flint_pf_starts", 1);
    g_pf.cursors = flint_buffer(256 * 4, "ds4_flint_pf_cursors", 1);
    g_pf.tiles = flint_buffer(FLINT_PF_TILES_TOTAL * 16, "ds4_flint_pf_tiles", 1);
    g_pf.dispatch = flint_buffer(96, "ds4_flint_pf_dispatch", 1);
    g_pf.rows = flint_buffer(cap * 6 * 4, "ds4_flint_pf_rows", 1);
    g_pf.act_sorted = flint_buffer(cap * 6 * 512 * 2, "ds4_flint_pf_act_sorted", 1);
    g_pf.y6 = flint_buffer(cap * 6 * 2048 * 2, "ds4_flint_pf_y6", 1);
    g_pf.win_k = flint_buffer(4ull * FLINT_PF_WIN * 128 * 2, "ds4_flint_pf_win_k", 1);
    g_pf.win_v = flint_buffer(4ull * FLINT_PF_WIN * 128 * 2, "ds4_flint_pf_win_v", 1);
    g_pf.logits = flint_buffer(DS4_FLINT_GPU_N_VOCAB * 2, "ds4_flint_pf_logits", 1);
    g_pf.dummy = flint_buffer(256, "ds4_flint_pf_dummy", 1);
    g_pf.exp_table = flint_buffer(2 * 256 * 8, "ds4_flint_pf_exp_table", 1);
    if (!g_pf.tokens || !g_pf.x || !g_pf.xn || !g_pf.q || !g_pf.k || !g_pf.v || !g_pf.attn || !g_pf.act ||
        !g_pf.act_sh || !g_pf.router || !g_pf.ids || !g_pf.wts || !g_pf.counts || !g_pf.starts || !g_pf.cursors ||
        !g_pf.tiles || !g_pf.dispatch || !g_pf.rows || !g_pf.act_sorted || !g_pf.y6 || !g_pf.win_k || !g_pf.win_v ||
        !g_pf.logits || !g_pf.dummy || !g_pf.exp_table) {
        return 0;
    }
    g_pf.ready = 1;
    return 1;
}

static void flint_prefill_scratch_release(void) {
    g_pf.tokens = g_pf.x = g_pf.xn = g_pf.q = g_pf.k = g_pf.v = g_pf.attn = g_pf.act = g_pf.act_sh = g_pf.router = nil;
    g_pf.ids = g_pf.wts = g_pf.counts = g_pf.starts = g_pf.cursors = g_pf.tiles = g_pf.dispatch = g_pf.rows = nil;
    g_pf.act_sorted = g_pf.y6 = g_pf.win_k = g_pf.win_v = g_pf.logits = g_pf.logits_all = g_pf.dummy = g_pf.exp_table = nil;
    g_pf.logits_all_rows = 0;
    g_pf.ready = 0;
    g_flint_pipes = nil;
    g_flint_prefill_lib = nil;
}

/* Encoding helpers.  Weight descriptors are rec's five words {w, s, b, n, k}
 * with offsets in 256-byte units into the model buffer. */
static inline NSUInteger flint_moff(uint32_t units) { return (NSUInteger)units << 8; }

static void flint_enc_gemm(id<MTLComputeCommandEncoder> enc, const char *pipe, id<MTLBuffer> A, uint32_t lda,
                           const uint32_t *w, id<MTLBuffer> C, uint32_t ldc, id<MTLBuffer> R, uint32_t ldr,
                           const uint32_t *w2, id<MTLBuffer> Y6, uint32_t M) {
    const FpGemmArgs args = { M, w[3], w[4], lda, ldc, ldr };
    [enc setComputePipelineState:flint_pipe(pipe)];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:g_flint_model offset:flint_moff(w[0]) atIndex:1];
    [enc setBuffer:g_flint_model offset:flint_moff(w[1] ? w[1] : w[0]) atIndex:2];
    [enc setBuffer:g_flint_model offset:flint_moff(w[2] ? w[2] : w[0]) atIndex:3];
    [enc setBuffer:C offset:0 atIndex:4];
    [enc setBytes:&args length:sizeof(args) atIndex:5];
    [enc setBuffer:R ? R : g_pf.dummy offset:0 atIndex:6];
    [enc setBuffer:g_flint_model offset:flint_moff(w2 ? w2[0] : w[0]) atIndex:7];
    [enc setBuffer:g_flint_model offset:flint_moff(w2 ? w2[1] : w[0]) atIndex:8];
    [enc setBuffer:g_flint_model offset:flint_moff(w2 ? w2[2] : w[0]) atIndex:9];
    [enc setBuffer:g_flint_silu offset:0 atIndex:10];
    [enc setBuffer:Y6 ? Y6 : g_pf.dummy offset:0 atIndex:11];
    [enc dispatchThreadgroups:MTLSizeMake((w[3] + 63) / 64, (M + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void flint_enc_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, uint32_t w_units, id<MTLBuffer> out,
                              uint32_t row0, uint32_t rows) {
    [enc setComputePipelineState:flint_pipe("flint_rmsnorm")];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:g_flint_model offset:flint_moff(w_units) atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&row0 length:4 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void flint_enc_ring_copy(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> K, NSUInteger k_off, id<MTLBuffer> V,
                                NSUInteger v_off, id<MTLBuffer> Kd, NSUInteger kd_off, id<MTLBuffer> Vd, NSUInteger vd_off,
                                const FpRingArgs *args) {
    if (args->n_rows == 0) return;
    [enc setComputePipelineState:flint_pipe("flint_ring_copy")];
    [enc setBuffer:K offset:k_off atIndex:0];
    [enc setBuffer:V offset:v_off atIndex:1];
    [enc setBuffer:Kd offset:kd_off atIndex:2];
    [enc setBuffer:Vd offset:vd_off atIndex:3];
    [enc setBytes:args length:sizeof(*args) atIndex:4];
    const uint32_t threads = 4u * args->n_rows * 16u;
    [enc dispatchThreadgroups:MTLSizeMake((threads + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static int flint_pf_finish(id<MTLCommandBuffer> cb, int owned, const char *what, char *err, size_t errlen) {
    if (!ds4_metal_finish_command_buffer(cb, owned, what)) {
        if (err && errlen) snprintf(err, errlen, "%s command buffer failed", what);
        return 0;
    }
    return 1;
}

/* DS4_FLINT_PREFILL_PROFILE=1: GPU time per kernel group.  Every group then
 * runs in its own command buffer (a breakdown, not a speed: the gaps between
 * buffers are not counted) and the running table is printed after each chunk. */
enum { PFP_NORM, PFP_QKV, PFP_HEAD, PFP_ATTN, PFP_OPROJ, PFP_DENSE, PFP_ROUTE, PFP_GATEUP, PFP_DOWN, PFP_SHARED, PFP_FINAL, PFP_N };
static const char *const g_pfp_name[PFP_N] = {
    "rmsnorm x2", "q/k/v gemm", "head norm, rope, ring", "attention", "o proj gemm", "dense glu + down",
    "router gemm, topk, scatter", "moe gate/up", "moe down", "shared glu + combine", "final norm + head"
};
static double g_pfp_ms[PFP_N], g_pfp_attn_global_ms;
static uint64_t g_pfp_tokens;
static int g_pfp_on = -1;
/* DS4_FLINT_MOE_TILES=64: every expert remainder in a 64-row tile (the
 * pre-2026-11-11 behaviour, for A/B runs); default: 64/32/16/8-row tiles. */
static int g_pf_moe_classes = -1;

static int flint_pf_mark(id<MTLCommandBuffer> __strong *cb, id<MTLComputeCommandEncoder> __strong *enc, int *owned,
                         int cat, char *err, size_t errlen) {
    if (g_pfp_on <= 0 || !*owned) return 1;
    ds4_metal_end_compute_encoder(*cb, *enc);
    if (!flint_pf_finish(*cb, *owned, "flint_prefill_profile", err, errlen)) return 0;
    g_pfp_ms[cat] += ((*cb).GPUEndTime - (*cb).GPUStartTime) * 1e3;
    *cb = ds4_metal_command_buffer(owned);
    *enc = ds4_metal_compute_encoder(*cb);
    flint_use_resources(*enc);
    return 1;
}

static void flint_pf_profile_print(uint32_t n, uint32_t position) {
    double total = 0;
    for (int i = 0; i < PFP_N; i++) total += g_pfp_ms[i];
    fprintf(stderr, "ds4: Flint prefill profile after %llu tokens (last chunk %u at %u): %.1f ms GPU, %.3f ms/token\n",
            (unsigned long long)g_pfp_tokens, n, position, total, total / (double)g_pfp_tokens);
    for (int i = 0; i < PFP_N; i++)
        fprintf(stderr, "ds4:   %-28s %9.1f ms %5.1f%%%s\n", g_pfp_name[i], g_pfp_ms[i], 100.0 * g_pfp_ms[i] / (total > 0 ? total : 1),
                i == PFP_ATTN ? [[NSString stringWithFormat:@"  (global layers %.1f ms)", g_pfp_attn_global_ms] UTF8String] : "");
}

int ds4_flint_gpu_prefill(ds4_flint_gpu_session *s, const int32_t *tokens, uint32_t n,
                          uint32_t position, uint32_t ring_initial, uint16_t *last_logits_bf16,
                          uint16_t *hidden_bf16, uint16_t *all_logits_bf16, char *err, size_t errlen) {
    if (!s || s->broken) {
        if (err && errlen) snprintf(err, errlen, "Flint session is unusable");
        return 0;
    }
    if (n == 0 || n > FLINT_PF_CAP) {
        if (err && errlen) snprintf(err, errlen, "prefill chunk of %u tokens is outside 1..%u", n, FLINT_PF_CAP);
        return 0;
    }
    flint_keepalive_set_active(0);   /* prefill keeps the GPU busy by itself */
    if ((uint64_t)position + n > s->ctx_cap) {
        if (err && errlen) snprintf(err, errlen, "prefill to position %u exceeds the session capacity %u", position + n, s->ctx_cap);
        return 0;
    }
    @autoreleasepool {
        const char *why = NULL;
        if (!flint_prefill_compile(&why) || !flint_prefill_scratch()) {
            if (err && errlen) snprintf(err, errlen, "%s", why ? why : "Flint prefill scratch allocation failed");
            return 0;
        }
        if (all_logits_bf16 && g_pf.logits_all_rows < n) {
            g_pf.logits_all = nil;
            g_pf.logits_all = flint_buffer((uint64_t)n * DS4_FLINT_GPU_N_VOCAB * 2, "ds4_flint_pf_logits_all", 0);
            if (!g_pf.logits_all) { if (err && errlen) snprintf(err, errlen, "no memory for all-row logits"); return 0; }
            g_pf.logits_all_rows = n;
        }
        memcpy([g_pf.tokens contents], tokens, (size_t)n * 4);
        if (g_pfp_on < 0) g_pfp_on = getenv("DS4_FLINT_PREFILL_PROFILE") ? 1 : 0;
        if (g_pf_moe_classes < 0) {
            const char *mt = getenv("DS4_FLINT_MOE_TILES");
            g_pf_moe_classes = (mt && atoi(mt) == 64) ? 0 : 1;
        }
        const uint32_t n_prev = position < 512u ? position : 512u;
        const uint32_t *final_row = g_flint_meta + 49 * 64;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_metal_command_buffer(&owned);
        id<MTLComputeCommandEncoder> enc = cb ? ds4_metal_compute_encoder(cb) : nil;
        if (!enc) { if (err && errlen) snprintf(err, errlen, "no Metal encoder"); return 0; }
        flint_use_resources(enc);

        /* embedding */
        [enc setComputePipelineState:flint_pipe("flint_embed")];
        [enc setBuffer:g_pf.tokens offset:0 atIndex:0];
        [enc setBuffer:g_flint_model offset:flint_moff(final_row[10]) atIndex:1];
        [enc setBuffer:g_flint_model offset:flint_moff(final_row[11]) atIndex:2];
        [enc setBuffer:g_flint_model offset:flint_moff(final_row[12]) atIndex:3];
        [enc setBuffer:g_pf.x offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(2048 / 256, n, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (hidden_bf16) {
            ds4_metal_end_compute_encoder(cb, enc);
            if (!flint_pf_finish(cb, owned, "flint_prefill", err, errlen)) return 0;
            memcpy(hidden_bf16, [g_pf.x contents], (size_t)n * 2048 * 2);
            cb = ds4_metal_command_buffer(&owned);
            enc = ds4_metal_compute_encoder(cb);
            flint_use_resources(enc);
        }

        for (uint32_t il = 0; il < DS4_FLINT_GPU_N_LAYER; il++) {
            const uint32_t *d = g_flint_meta + il * 64;
            const bool global = d[1] != 0, sparse = d[2] != 0;
            const NSUInteger kv_k_off = (NSUInteger)s->kv_offset[il] * 2;
            const NSUInteger kv_v_off = kv_k_off + (NSUInteger)s->kv_cap[il] * 512 * 2;

            /* attention */
            flint_enc_rmsnorm(enc, g_pf.x, d[3], g_pf.xn, 0, n);
            if (!flint_pf_mark(&cb, &enc, &owned, PFP_NORM, err, errlen)) return 0;
            flint_enc_gemm(enc, "flint_gemm_q4_store", g_pf.xn, 2048, d + 10, g_pf.q, 4096, nil, 0, NULL, nil, n);
            flint_enc_gemm(enc, "flint_gemm_q4_store", g_pf.xn, 2048, d + 15, g_pf.k, 512, nil, 0, NULL, nil, n);
            flint_enc_gemm(enc, "flint_gemm_q4_store", g_pf.xn, 2048, d + 20, g_pf.v, 512, nil, 0, NULL, nil, n);
            if (!flint_pf_mark(&cb, &enc, &owned, PFP_QKV, err, errlen)) return 0;
            if (!global) {
                /* previous positions from the ring into the window */
                const FpRingArgs gather = { n_prev, 513u * 128u, FLINT_PF_WIN * 128u, position - n_prev, ring_initial, 0, 0 };
                flint_enc_ring_copy(enc, s->kv[0], kv_k_off, s->kv[0], kv_v_off, g_pf.win_k, 0, g_pf.win_v, 0, &gather);
            }
            {
                FpHeadArgs ha = { n, position, global ? 0u : 1u, global ? position : n_prev,
                                  global ? s->kv_cap[il] * 128u : FLINT_PF_WIN * 128u };
                [enc setComputePipelineState:flint_pipe("flint_head_norm_rope")];
                [enc setBuffer:g_pf.q offset:0 atIndex:0];
                [enc setBuffer:g_pf.k offset:0 atIndex:1];
                [enc setBuffer:g_pf.v offset:0 atIndex:2];
                [enc setBuffer:g_flint_model offset:flint_moff(d[5]) atIndex:3];
                [enc setBuffer:g_flint_model offset:flint_moff(d[6]) atIndex:4];
                if (global) {
                    const uint32_t b0 = position >> 16, b1 = (position + n - 1) >> 16;
                    [enc setBuffer:s->kv[b0] offset:kv_k_off atIndex:5];
                    [enc setBuffer:s->kv[b1] offset:kv_k_off atIndex:6];
                    [enc setBuffer:s->kv[b0] offset:kv_v_off atIndex:7];
                    [enc setBuffer:s->kv[b1] offset:kv_v_off atIndex:8];
                } else {
                    [enc setBuffer:g_pf.win_k offset:0 atIndex:5];
                    [enc setBuffer:g_pf.win_k offset:0 atIndex:6];
                    [enc setBuffer:g_pf.win_v offset:0 atIndex:7];
                    [enc setBuffer:g_pf.win_v offset:0 atIndex:8];
                }
                [enc setBytes:&ha length:sizeof(ha) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake((n * 40u + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            if (!flint_pf_mark(&cb, &enc, &owned, PFP_HEAD, err, errlen)) return 0;
            {
                FpAttnArgs aa;
                aa.M = n; aa.q_pos0 = position; aa.ldq = 4096; aa.ldo = 4096;
                if (global) {
                    aa.key_pos0 = 0; aa.n_keys = position + n; aa.kv_head_stride = s->kv_cap[il] * 128u; aa.window = 0;
                } else {
                    aa.key_pos0 = position - n_prev; aa.n_keys = n_prev + n; aa.kv_head_stride = FLINT_PF_WIN * 128u; aa.window = 513u;
                }
                [enc setComputePipelineState:flint_pipe("flint_attention")];
                [enc setBuffer:g_pf.q offset:0 atIndex:0];
                for (uint32_t b = 0; b < DS4_FLINT_GPU_MAX_BLOCKS; b++) {
                    /* blocks past n_blocks are never read; bind block 0 so the offset stays in range */
                    id<MTLBuffer> kb = global ? s->kv[b < s->n_blocks ? b : 0] : g_pf.win_k;
                    id<MTLBuffer> vb = global ? s->kv[b < s->n_blocks ? b : 0] : g_pf.win_v;
                    [enc setBuffer:kb offset:global ? kv_k_off : 0 atIndex:1 + b];
                    [enc setBuffer:vb offset:global ? kv_v_off : 0 atIndex:5 + b];
                }
                [enc setBuffer:g_flint_model offset:flint_moff(d[7]) atIndex:9];
                [enc setBuffer:g_pf.attn offset:0 atIndex:10];
                [enc setBytes:&aa length:sizeof(aa) atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake((n + 63) / 64, 32, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            if (g_pfp_on > 0) {
                const double before = g_pfp_ms[PFP_ATTN];
                if (!flint_pf_mark(&cb, &enc, &owned, PFP_ATTN, err, errlen)) return 0;
                if (global) g_pfp_attn_global_ms += g_pfp_ms[PFP_ATTN] - before;
            }
            flint_enc_gemm(enc, "flint_gemm_q4_residual", g_pf.attn, 4096, d + 25, g_pf.x, 2048, g_pf.x, 2048, NULL, nil, n);
            if (!global) {
                /* the window's last min(513, position + n) rows become the ring, laid out for position + n */
                const uint32_t total = position + n;
                const uint32_t rows = total < 513u ? total : 513u;
                const uint32_t first_pos = total - rows;
                const FpRingArgs scatter = { rows, FLINT_PF_WIN * 128u, 513u * 128u, first_pos, total, 1,
                                             first_pos - (position - n_prev) };
                flint_enc_ring_copy(enc, g_pf.win_k, 0, g_pf.win_v, 0, s->kv[0], kv_k_off, s->kv[0], kv_v_off, &scatter);
            }
            if (!flint_pf_mark(&cb, &enc, &owned, PFP_OPROJ, err, errlen)) return 0;

            /* feed-forward */
            flint_enc_rmsnorm(enc, g_pf.x, d[4], g_pf.xn, 0, n);
            if (!flint_pf_mark(&cb, &enc, &owned, PFP_NORM, err, errlen)) return 0;
            if (!sparse) {
                flint_enc_gemm(enc, "flint_gemm_q4_glu", g_pf.xn, 2048, d + 30, g_pf.act, 6144, nil, 0, d + 35, nil, n);
                flint_enc_gemm(enc, "flint_gemm_q4_residual", g_pf.act, 6144, d + 40, g_pf.x, 2048, g_pf.x, 2048, NULL, nil, n);
                if (!flint_pf_mark(&cb, &enc, &owned, PFP_DENSE, err, errlen)) return 0;
            } else {
                const uint32_t router[5] = { d[60], 0, 0, 256u, 2048u };
                flint_enc_gemm(enc, "flint_gemm_bf16_store", g_pf.xn, 2048, router, g_pf.router, 256, nil, 0, NULL, nil, n);
                [enc setComputePipelineState:flint_pipe("flint_route_topk")];
                [enc setBuffer:g_pf.router offset:0 atIndex:0];
                [enc setBuffer:g_pf.ids offset:0 atIndex:1];
                [enc setBuffer:g_pf.wts offset:0 atIndex:2];
                [enc setBuffer:g_pf.counts offset:0 atIndex:3];
                [enc setBytes:&n length:4 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((n + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc setComputePipelineState:flint_pipe("flint_route_offsets")];
                [enc setBuffer:g_pf.counts offset:0 atIndex:0];
                [enc setBuffer:g_pf.starts offset:0 atIndex:1];
                [enc setBuffer:g_pf.tiles offset:0 atIndex:2];
                [enc setBuffer:g_pf.dispatch offset:0 atIndex:3];
                [enc setBuffer:g_pf.cursors offset:0 atIndex:4];
                const uint32_t tile_cfg[2] = {FLINT_PF_MAX_TILES, (uint32_t)g_pf_moe_classes};
                [enc setBytes:tile_cfg length:8 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc setComputePipelineState:flint_pipe("flint_route_scatter")];
                [enc setBuffer:g_pf.ids offset:0 atIndex:0];
                [enc setBuffer:g_pf.starts offset:0 atIndex:1];
                [enc setBuffer:g_pf.cursors offset:0 atIndex:2];
                [enc setBuffer:g_pf.rows offset:0 atIndex:3];
                [enc setBytes:&n length:4 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((n * 6 + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                if (!flint_pf_mark(&cb, &enc, &owned, PFP_ROUTE, err, errlen)) return 0;
                id<MTLBuffer> experts_buf = g_flint_experts;
                NSUInteger experts_off = (NSUInteger)il * 256 * 8;
                if (g_flint_stream) {
                    /* Streaming: the routing decides which packs the layer needs.
                     * Wait for it, stage or cache them, and point this layer's
                     * table (two, by layer parity) at them.  The wait also means
                     * the previous layer's kernels are done with the other
                     * staging buffer before it is refilled. */
                    ds4_metal_end_compute_encoder(cb, enc);
                    if (!flint_pf_finish(cb, owned, "flint_prefill", err, errlen)) return 0;
                    const uint32_t *starts = (const uint32_t *)[g_pf.starts contents];
                    uint8_t need[256];
                    for (uint32_t e = 0; e < 256; e++) need[e] = starts[e + 1] > starts[e];
                    uint64_t *table = (uint64_t *)[g_pf.exp_table contents] + (il & 1u) * 256;
                    if (!ds4_flint_stream_prefill_layer(g_flint_stream, il, need, n, table, err, errlen)) return 0;
                    cb = ds4_metal_command_buffer(&owned);
                    enc = ds4_metal_compute_encoder(cb);
                    flint_use_resources(enc);
                    experts_buf = g_pf.exp_table;
                    experts_off = (NSUInteger)(il & 1u) * 256 * 8;
                }
                /* The four tile heights: table k at tile offset tile_base[k],
                 * dispatch arguments at word 3k (gate/up) and 12 + 3k (down). */
                static const char *const gateup_kernels[4] = {"flint_moe_gateup_64", "flint_moe_gateup_32", "flint_moe_gateup_16", "flint_moe_gateup_8"};
                static const char *const down_kernels[4] = {"flint_moe_down_64", "flint_moe_down_32", "flint_moe_down_16", "flint_moe_down_8"};
                const NSUInteger tile_base[4] = {0, FLINT_PF_MAX_TILES, FLINT_PF_MAX_TILES + 256, FLINT_PF_MAX_TILES + 512};
                [enc setBuffer:g_pf.xn offset:0 atIndex:0];
                [enc setBuffer:g_pf.rows offset:0 atIndex:1];
                [enc setBuffer:experts_buf offset:experts_off atIndex:3];
                [enc setBuffer:g_flint_silu offset:0 atIndex:4];
                [enc setBuffer:g_pf.act_sorted offset:0 atIndex:5];
                for (int k = 0; k < (g_pf_moe_classes ? 4 : 1); k++) {
                    [enc setComputePipelineState:flint_pipe(gateup_kernels[k])];
                    [enc setBuffer:g_pf.tiles offset:tile_base[k] * 16 atIndex:2];
                    [enc dispatchThreadgroupsWithIndirectBuffer:g_pf.dispatch indirectBufferOffset:(NSUInteger)k * 12 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                if (!flint_pf_mark(&cb, &enc, &owned, PFP_GATEUP, err, errlen)) return 0;
                [enc setBuffer:g_pf.act_sorted offset:0 atIndex:0];
                [enc setBuffer:g_pf.rows offset:0 atIndex:1];
                [enc setBuffer:experts_buf offset:experts_off atIndex:3];
                [enc setBuffer:g_pf.wts offset:0 atIndex:4];
                [enc setBuffer:g_pf.y6 offset:0 atIndex:5];
                for (int k = 0; k < (g_pf_moe_classes ? 4 : 1); k++) {
                    [enc setComputePipelineState:flint_pipe(down_kernels[k])];
                    [enc setBuffer:g_pf.tiles offset:tile_base[k] * 16 atIndex:2];
                    [enc dispatchThreadgroupsWithIndirectBuffer:g_pf.dispatch indirectBufferOffset:48 + (NSUInteger)k * 12 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                if (!flint_pf_mark(&cb, &enc, &owned, PFP_DOWN, err, errlen)) return 0;
                flint_enc_gemm(enc, "flint_gemm_q4_glu", g_pf.xn, 2048, d + 45, g_pf.act_sh, 1024, nil, 0, d + 50, nil, n);
                flint_enc_gemm(enc, "flint_gemm_q4_combine", g_pf.act_sh, 1024, d + 55, g_pf.x, 2048, g_pf.x, 2048, NULL, g_pf.y6, n);
                if (!flint_pf_mark(&cb, &enc, &owned, PFP_SHARED, err, errlen)) return 0;
            }
            if (hidden_bf16) {
                ds4_metal_end_compute_encoder(cb, enc);
                if (!flint_pf_finish(cb, owned, "flint_prefill", err, errlen)) return 0;
                memcpy(hidden_bf16 + (size_t)(il + 1) * n * 2048, [g_pf.x contents], (size_t)n * 2048 * 2);
                /* Diagnostic: DS4_FLINT_ROUTER_DUMP=FILE appends each sparse layer's
                 * router logits (n x 256 bf16) and chosen ids (n x 6 u32). */
                const char *rd = getenv("DS4_FLINT_ROUTER_DUMP");
                if (rd && sparse) {
                    FILE *f = fopen(rd, "ab");
                    if (f) {
                        fwrite([g_pf.router contents], 2, (size_t)n * 256, f);
                        fwrite([g_pf.ids contents], 4, (size_t)n * 6, f);
                        fclose(f);
                    }
                }
                cb = ds4_metal_command_buffer(&owned);
                enc = ds4_metal_compute_encoder(cb);
                flint_use_resources(enc);
            }
        }
        if (g_flint_stream) ds4_flint_stream_prefill_end(g_flint_stream);

        /* final norm and head */
        if (all_logits_bf16) {
            flint_enc_rmsnorm(enc, g_pf.x, final_row[3], g_pf.xn, 0, n);
            flint_enc_gemm(enc, "flint_gemm_q4_store", g_pf.xn, 2048, final_row + 30, g_pf.logits_all, DS4_FLINT_GPU_N_VOCAB,
                           nil, 0, NULL, nil, n);
        } else {
            flint_enc_rmsnorm(enc, g_pf.x, final_row[3], g_pf.xn, n - 1, 1);
        }
        {
            const uint32_t N = DS4_FLINT_GPU_N_VOCAB;
            [enc setComputePipelineState:flint_pipe("flint_head_row")];
            [enc setBuffer:g_pf.xn offset:(NSUInteger)(n - 1) * 2048 * 2 atIndex:0];
            [enc setBuffer:g_flint_model offset:flint_moff(final_row[30]) atIndex:1];
            [enc setBuffer:g_flint_model offset:flint_moff(final_row[31]) atIndex:2];
            [enc setBuffer:g_flint_model offset:flint_moff(final_row[32]) atIndex:3];
            [enc setBuffer:g_pf.logits offset:0 atIndex:4];
            [enc setBytes:&N length:4 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_metal_end_compute_encoder(cb, enc);
        if (!flint_pf_finish(cb, owned, "flint_prefill", err, errlen)) return 0;
        if (g_pfp_on > 0 && owned) {
            g_pfp_ms[PFP_FINAL] += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            g_pfp_tokens += n;
            flint_pf_profile_print(n, position);
        }
        if (last_logits_bf16) memcpy(last_logits_bf16, [g_pf.logits contents], DS4_FLINT_GPU_N_VOCAB * 2);
        if (all_logits_bf16) memcpy(all_logits_bf16, [g_pf.logits_all contents], (size_t)n * DS4_FLINT_GPU_N_VOCAB * 2);
        return 1;
    }
}
