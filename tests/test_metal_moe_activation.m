// SPDX-License-Identifier: MIT
// Native oracle for the actual IQ2 pair, route-map and activation-copy shaders.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne11;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne20, ne21, ne0, ne1;
    int16_t r2, r3;
    int32_t tp_rank, tp_world, tp_expert_base;
} MM;
typedef struct {
    int32_t ne02, ne10, ne11;
    uint64_t nb11, nb12;
    int32_t ne21, ne20;
    uint64_t nb21;
} Map;
typedef struct {
    uint32_t width, rows;
    uint64_t gate_row_stride, up_row_stride, mid_row_stride, weight_stride;
    uint32_t write_clamped;
    float clamp_value;
} Activation;
typedef struct { uint16_t d; uint8_t qs[64]; } IQ2;
_Static_assert(sizeof(MM) == 104 && sizeof(Map) == 48 &&
               sizeof(Activation) == 48 && sizeof(IQ2) == 66, "Metal MoE ABI");

enum { GUARD = 256, TOPK = 6, SMEM = 16384 };
static const uint16_t poison = 0x7e55;
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pair[2], copy_pipeline, map_pipeline;
static unsigned cases;

static void require(bool ok, const char *message) {
    if (!ok) { fprintf(stderr, "Metal MoE activation FAIL: %s\n", message); exit(1); }
}
static uint32_t random32(uint32_t *seed) { return *seed = *seed * 1664525u + 1013904223u; }
static int compare_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static void *data(id<MTLBuffer> b) { return (char *)b.contents + GUARD; }
static size_t bytes(id<MTLBuffer> b) { return b.length - 2u*GUARD; }
static id<MTLBuffer> buffer(size_t size) {
    id<MTLBuffer> b = [device newBufferWithLength:size + 2u*GUARD options:MTLResourceStorageModeShared];
    require(b != nil, "buffer allocation");
    memset(b.contents, 0xa5, b.length);
    return b;
}
static void guards(id<MTLBuffer> b) {
    const uint8_t *p = b.contents;
    for (unsigned i = 0; i < GUARD; ++i)
        require(p[i] == 0xa5 && p[b.length-1-i] == 0xa5, "buffer guard overwritten");
}
static uint64_t hash(id<MTLBuffer> b) {
    const uint8_t *p = b.contents;
    uint64_t value = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < b.length; ++i) value = (value ^ p[i])*UINT64_C(1099511628211);
    return value;
}
static void bind(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> b, unsigned index) {
    [enc setBuffer:b offset:GUARD atIndex:index];
}
static double finish(id<MTLCommandBuffer> cb) {
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
        fprintf(stderr, "%s\n", cb.error.description.UTF8String);
    require(cb.status == MTLCommandBufferStatusCompleted, "GPU execution");
    return (cb.GPUEndTime - cb.GPUStartTime)*1000.0;
}
static id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib, NSString *name) {
    NSError *error = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    id<MTLComputePipelineState> p = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (!p) fprintf(stderr, "%s: %s\n", name.UTF8String, error.description.UTF8String);
    require(p != nil, "pipeline creation");
    return p;
}
static void compile(NSString *source, bool fast) {
    NSError *error = nil;
    MTLCompileOptions *options = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = fast;
#pragma clang diagnostic pop
    id<MTLLibrary> lib = [device newLibraryWithSource:source options:options error:&error];
    if (!lib) fprintf(stderr, "%s\n", error.description.UTF8String);
    require(lib != nil, "compile production shader");
    pair[0] = pipeline(lib, @"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16");
    pair[1] = pipeline(lib, @"kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_rhs");
    copy_pipeline = pipeline(lib, @"kernel_cpy_contig_f32_f16_4");
    map_pipeline = pipeline(lib, @"kernel_mul_mm_id_map0_ne20_6");
    require(pair[0].threadExecutionWidth == 32 && pair[1].threadExecutionWidth == 32 &&
            pair[0].maxTotalThreadsPerThreadgroup >= 128 &&
            pair[1].maxTotalThreadsPerThreadgroup >= 128, "SIMDgroup shape");
}

// Checked verbatim against ds4_metal.m by the Python driver.
static NSUInteger ds4_gpu_cpy_threads(uint32_t n, id<MTLComputePipelineState> pipeline) {
    NSUInteger nth = 32u;
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    while (nth < (NSUInteger)n && nth < max_threads) nth *= 2u;
    if (nth > max_threads) nth = max_threads;
    if (nth > (NSUInteger)n) nth = (NSUInteger)n;
    return nth ? nth : 1u;
}

static void encode_copy(id<MTLCommandBuffer> cb, id<MTLBuffer> x, id<MTLBuffer> xhalf, uint32_t count) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:copy_pipeline];
    [enc setBytes:&count length:sizeof(count) atIndex:0];
    bind(enc, x, 1); bind(enc, xhalf, 2);
    const NSUInteger items = ((NSUInteger)count+3u)/4u;
    const NSUInteger nth = ds4_gpu_cpy_threads((uint32_t)items, copy_pipeline);
    [enc dispatchThreadgroups:MTLSizeMake((items+nth-1u)/nth, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
    [enc endEncoding];
}
static void encode_pair(id<MTLCommandBuffer> cb, unsigned arm, MM args, Activation act,
                        id<MTLBuffer> wg, id<MTLBuffer> wu, id<MTLBuffer> x,
                        id<MTLBuffer> counts, id<MTLBuffer> ids, id<MTLBuffer> out,
                        id<MTLBuffer> weights, id<MTLBuffer> work, unsigned work_cap) {
    if (arm) { args.nb10 /= 2; args.nb11 /= 2; args.nb12 /= 2; args.nb13 /= 2; }
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pair[arm]];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBytes:&act length:sizeof(act) atIndex:1];
    bind(enc, wg, 2); bind(enc, wu, 3); bind(enc, x, 4); bind(enc, counts, 5);
    bind(enc, ids, 6); bind(enc, out, 7); bind(enc, weights, 8); bind(enc, work, 9);
    [enc setThreadgroupMemoryLength:SMEM atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(work_cap, (args.ne0+63u)/64u, 1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
}

static void run(unsigned k, unsigned m, unsigned n, unsigned experts, unsigned routing,
                unsigned padding, bool fast, bool bench) {
    @autoreleasepool {
        const size_t row_bytes = (k/256u)*sizeof(IQ2), expert_bytes = m*row_bytes;
        const unsigned work_cap = (TOPK*n + 31u*experts + 31u)/32u;
        const unsigned stride = m + padding;
        id<MTLBuffer> wg = buffer(experts*expert_bytes), wu = buffer(experts*expert_bytes);
        id<MTLBuffer> x = buffer((size_t)n*k*4), xhalf = buffer((size_t)n*k*2);
        id<MTLBuffer> selected = buffer((size_t)n*TOPK*4), weights = buffer((size_t)n*TOPK*4);
        id<MTLBuffer> counts = buffer(experts*4), ids = buffer((size_t)experts*n*4);
        id<MTLBuffer> work = buffer(8u + work_cap*8u);
        id<MTLBuffer> out[2] = {buffer((size_t)(n*TOPK+1u)*stride*2),
                               buffer((size_t)(n*TOPK+1u)*stride*2)};
        uint32_t seed = k + m*19u + n*37u + routing;
        for (id<MTLBuffer> b in @[wg, wu]) {
            IQ2 *p = data(b);
            for (size_t i = 0; i < bytes(b)/sizeof(IQ2); ++i) {
                p[i].d = (i%19u == 0) ? 0 : 0x0800u + (random32(&seed)&0x7ffu);
                for (unsigned j = 0; j < 64; ++j) p[i].qs[j] = random32(&seed) >> 24;
            }
        }
        const float edge[] = {0.f, -0.f, 0x1p-24f, -0x1p-24f, 0x1p-25f,
                              0x1.002p0f, -0x1.002p0f, 0x1.ffep-1f, 0x1p-14f};
        float *xp = data(x);
        for (size_t i = 0; i < (size_t)n*k; ++i)
            xp[i] = (i%31u < sizeof(edge)/sizeof(edge[0])) ? edge[i%31u] :
                    ((int)(random32(&seed)%2049u)-1024)/8192.f;
        int32_t *sp = data(selected);
        float *wp = data(weights);
        for (unsigned t = 0; t < n; ++t) for (unsigned s = 0; s < TOPK; ++s) {
            unsigned e = routing == 1 ? s :
                routing == 2 ? (s == 0 ? 0 : 1u + (t*7u+s*11u)%(experts-1u)) :
                (t*7u+s*41u)%experts;
            sp[t*TOPK+s] = (int32_t)e;
            wp[t*TOPK+s] = s%3u == 0 ? 0.f : (float)(s+1u)/32.f;
        }
        for (unsigned arm = 0; arm < 2; ++arm) {
            uint16_t *p = data(out[arm]);
            for (size_t i = 0; i < bytes(out[arm])/2; ++i) p[i] = poison;
        }
        NSArray<id<MTLBuffer>> *inputs = @[wg, wu, x, selected, weights];
        uint64_t hashes[5];
        for (unsigned i = 0; i < 5; ++i) hashes[i] = hash(inputs[i]);
        Map map = {.ne02 = (int32_t)experts, .ne10 = (int32_t)k, .ne11 = 1,
                   .nb11 = k*4u, .nb12 = k*4u, .ne21 = (int32_t)n,
                   .ne20 = TOPK, .nb21 = TOPK*4};
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:map_pipeline];
        [enc setBytes:&map length:sizeof(map) atIndex:0];
        bind(enc, selected, 1); bind(enc, counts, 2); bind(enc, ids, 3); bind(enc, work, 4);
        [enc setThreadgroupMemoryLength:experts*TOPK*2 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(experts, 1, 1)];
        [enc endEncoding]; finish(cb);
        const uint32_t *cp = data(counts), *workp = data(work);
        const int32_t *ip = data(ids);
        unsigned tiles = 0, max_count = 0;
        for (unsigned e = 0; e < experts; ++e) {
            unsigned count = 0;
            for (unsigned t = 0; t < n; ++t) for (unsigned s = 0; s < TOPK; ++s)
                if (sp[t*TOPK+s] == (int32_t)e)
                    require(ip[e*n + count++] == (int32_t)(t*TOPK+s), "production route ID/order");
            require(cp[e] == count, "production route count");
            if (count > max_count) max_count = count;
            for (unsigned row = 0; row < count; row += 32) {
                require(workp[2+2*tiles] == e && workp[3+2*tiles] == row, "production work tile");
                ++tiles;
            }
        }
        require(workp[0] == tiles && tiles < work_cap, "work count or padded-grid coverage");
        MM args = {.ne00 = (int32_t)k, .ne02 = (int32_t)experts, .nb01 = row_bytes,
                   .nb02 = expert_bytes, .nb03 = experts*expert_bytes, .ne11 = 1,
                   .nb10 = 4, .nb11 = k*4u, .nb12 = k*4u, .nb13 = (uint64_t)n*k*4u,
                   .ne20 = TOPK, .ne21 = (int32_t)n, .ne0 = (int32_t)m,
                   .ne1 = TOPK, .r2 = 1, .r3 = 1};
        Activation act = {.width = m, .rows = n*TOPK, .gate_row_stride = m*4u,
                          .up_row_stride = m*4u, .mid_row_stride = stride*2u,
                          .weight_stride = 4, .clamp_value = routing%2u ? 0.f : 7.f};
        cb = [queue commandBuffer];
        encode_pair(cb, 0, args, act, wg, wu, x, counts, ids, out[0], weights, work, work_cap);
        encode_copy(cb, x, xhalf, n*k);
        encode_pair(cb, 1, args, act, wg, wu, xhalf, counts, ids, out[1], weights, work, work_cap);
        finish(cb);
        const uint16_t *ref = data(out[0]), *got = data(out[1]), *xh = data(xhalf);
        for (size_t i = 0; i < bytes(out[0])/2; ++i) {
            const bool active = i/stride < n*TOPK && i%stride < m;
            if ((active && (ref[i] != got[i] || (got[i]&0x7c00u) == 0x7c00u)) ||
                (!active && (ref[i] != poison || got[i] != poison))) {
                fprintf(stderr, "K=%u M=%u N=%u E=%u route=%u fast=%d i=%zu active=%d ref=%04x got=%04x\n",
                        k, m, n, experts, routing, fast, i, active, ref[i], got[i]);
                require(false, "bitwise finite active outputs / inactive poison");
            }
        }
        for (size_t i = 0; i < (size_t)n*k; ++i) {
            const _Float16 expected = (_Float16)xp[i]; uint16_t bits;
            memcpy(&bits, &expected, sizeof(bits));
            require(xh[i] == bits, "copy rounding differs from host IEEE half");
        }
        for (unsigned i = 0; i < 5; ++i) require(hash(inputs[i]) == hashes[i], "read-only input mutated");
        for (id<MTLBuffer> b in @[wg, wu, x, xhalf, selected, weights, counts, ids, work, out[0], out[1]]) guards(b);
        ++cases;
        if (!bench) return;
        // Warm, fixed input. Include the extra copy and encoder dependency in
        // every candidate; the identical, already-validated route map is common.
        double samples[2][14]; unsigned used[2] = {0, 0};
        for (unsigned trial = 0; trial < 8; ++trial) for (unsigned step = 0; step < 4; ++step) {
            const unsigned arm = (step == 1 || step == 2);
            cb = [queue commandBuffer];
            if (arm) encode_copy(cb, x, xhalf, n*k);
            encode_pair(cb, arm, args, act, wg, wu, arm ? xhalf : x, counts, ids,
                        out[arm], weights, work, work_cap);
            const double ms = finish(cb);
            require(ms > 0, "GPU timestamps unavailable");
            if (trial) samples[arm][used[arm]++] = ms;
        }
        double avg[2] = {0, 0}, lo[2] = {INFINITY, INFINITY}, hi[2] = {0, 0};
        for (unsigned arm = 0; arm < 2; ++arm) for (unsigned i = 0; i < used[arm]; ++i) {
            const double value = samples[arm][i]; avg[arm] += value/used[arm];
            if (value < lo[arm]) lo[arm] = value;
            if (value > hi[arm]) hi[arm] = value;
        }
        double median[2];
        for (unsigned arm = 0; arm < 2; ++arm) {
            printf("SAMPLES N=%u arm=%s ms=", n, arm ? "copy_F16_pair" : "F32_pair");
            for (unsigned i = 0; i < used[arm]; ++i)
                printf("%s%.6f", i ? "," : "", samples[arm][i]);
            putchar('\n');
            qsort(samples[arm], used[arm], sizeof(double), compare_double);
            median[arm] = (samples[arm][used[arm]/2-1] + samples[arm][used[arm]/2])*0.5;
        }
        printf("BENCH K=%u M=%u N=%u E=%u topk=6 route=%s tiles=%u max_count=%u "
               "F32_pair_ms=%.4f [%.4f,%.4f] copy_F16_pair_ms=%.4f [%.4f,%.4f] "
               "mean_speedup_pct=%.2f median_F32_ms=%.4f median_F16_ms=%.4f "
               "median_speedup_pct=%.2f samples=14/arm\n", k, m, n, experts,
               routing == 2 ? "skew" : "uniform", tiles, max_count,
               avg[0], lo[0], hi[0], avg[1], lo[1], hi[1], 100.0*(avg[0]/avg[1]-1.0),
               median[0], median[1], 100.0*(median[0]/median[1]-1.0));
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        require(argc == 2 || (argc == 3 && !strcmp(argv[2], "--bench")),
                "usage: test_metal_moe_activation SOURCE.metal [--bench]");
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "no Metal device; run with GPU access");
        queue = [device newCommandQueue]; require(queue != nil, "command queue");
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                    encoding:NSUTF8StringEncoding error:&error];
        require(source != nil, "read extracted production shader");
        printf("Metal MoE activation device=%s\n", device.name.UTF8String); fflush(stdout);
        const unsigned ns[] = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 129};
        const unsigned ms[] = {1, 17, 63, 64, 65, 128};
        for (unsigned fast = 0; fast < 2; ++fast) {
            compile(source, fast);
            for (unsigned i = 0; i < sizeof(ns)/sizeof(ns[0]); ++i)
                for (unsigned route = 0; route < 3; ++route)
                    run(256, ms[i%6], ns[i], 16, route, i%2u ? 5u : 0u, fast, false);
            run(4096, 65, 128, 256, 0, 5, fast, false);
            run(4096, 128, 129, 256, 2, 0, fast, false);
            run(512, 128, 512, 256, 0, 5, fast, false);
            printf("PASS: %u cumulative native cases; %s math.\n", cases, fast ? "fast" : "strict"); fflush(stdout);
        }
        if (argc == 3) {
            puts("Warm fixed-input GPU stage A/B; includes candidate copy + pair; excludes common map, allocations, down projection, model I/O. ABBA, one warmup block, seven measured blocks.");
            for (unsigned i = 0; i < 3; ++i) run(4096, 2048, (unsigned[]){128, 512, 2048}[i], 256, 0, 0, true, true);
        }
        printf("PASS: %u native IQ2 paired-MoE activation reuse cases.\n", cases);
    }
    return 0;
}
