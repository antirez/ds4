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

#include "ds4_metal_internal.h"
#include "ds4_flint_gpu.h"

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
        if (g_flint_decode.maxTotalThreadsPerThreadgroup < FLINT_THREADS) {
            if (why) *why = "flint_decode cannot run 256 threads per threadgroup";
            return 0;
        }
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

void ds4_flint_gpu_engine_release(void) {
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
    id<MTLBuffer> logits, trace, diagnostic, control;
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
             (uint64_t)DS4_FLINT_GPU_META_WORDS * 4 + FLINT_CONTROL_WORDS * 4 + FLINT_WORKERS * 4;
    return bytes;
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
        if (!s->metadata || !s->workspace || !s->partials || !s->statistics || !s->routing ||
            !s->logits || !s->trace || !s->diagnostic || !s->control) {
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
    s->logits = nil; s->trace = nil; s->diagnostic = nil; s->control = nil;
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
        memset([s->control contents], 0, FLINT_CONTROL_WORDS * 4);
        memset([s->diagnostic contents], 0, FLINT_WORKERS * 4);
        /* Diagnostics for the trace test: DS4_FLINT_LAYER_COUNT runs the first N
         * layers only; DS4_FLINT_WORKSPACE_DUMP appends the workspace after each step. */
        const char *lc_env = getenv("DS4_FLINT_LAYER_COUNT");
        const uint32_t layer_count = lc_env ? (uint32_t)atoi(lc_env) : DS4_FLINT_GPU_N_LAYER;
        const uint32_t parameters[4] = { token, position, initial_position, layer_count };
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
        [enc dispatchThreadgroups:MTLSizeMake(FLINT_WORKERS, 1, 1) threadsPerThreadgroup:MTLSizeMake(FLINT_THREADS, 1, 1)];
        ds4_metal_end_compute_encoder(cb, enc);
        if (!ds4_metal_finish_command_buffer(cb, owned, "flint_decode")) {
            if (err && errlen) snprintf(err, errlen, "flint_decode command buffer failed");
            return 0;
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
        const char *ws_env = getenv("DS4_FLINT_WORKSPACE_DUMP");
        if (ws_env) {
            FILE *f = fopen(ws_env, "ab");
            if (f) { fwrite([s->workspace contents], 2, FLINT_WORKSPACE_ELEMS, f); fclose(f); }
        }
        if (logits_bf16) memcpy(logits_bf16, [s->logits contents], DS4_FLINT_GPU_N_VOCAB * sizeof(uint16_t));
        if (trace_bf16) memcpy(trace_bf16, [s->trace contents], DS4_FLINT_GPU_TRACE_ELEMS * sizeof(uint16_t));
        return 1;
    }
}
