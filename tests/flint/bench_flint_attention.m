/*
 * Micro-benchmark for the Flint prefill attention kernels in
 * metal/flint_prefill.metal (docs/RECURSIVE_FLINT.md, prefill speed work).
 *
 * Times each named kernel alone on one chunk of M queries against n_keys
 * keys (the shape of one global layer in a 2048-token prefill chunk at a
 * given position) with GPU timestamps, prints ms and TFLOPS over the
 * unmasked pairs, and compares every kernel's output with the first one's.
 *
 *   clang -fobjc-arc -O2 -framework Metal -framework Foundation -o tests/flint/bench_flint_attention tests/flint/bench_flint_attention.m
 *   tests/flint/bench_flint_attention [--m 2048] [--keys 16384] [--window 0|513] [--iters 5] [--shader PATH]
 *                                     [--variants tests/flint/flint_attention_variants.metal] kernel [kernel ...]
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t M, q_pos0, key_pos0, n_keys, kv_head_stride, window, ldq, ldo; } AttnArgs;

static float bf2f(uint16_t b) { union { uint32_t u; float f; } v; v.u = (uint32_t)b << 16; return v.f; }
static uint16_t f2bf(float f) {
    union { uint32_t u; float f; } v; v.f = f;
    const uint32_t r = 0x7fffu + ((v.u >> 16) & 1u);
    return (uint16_t)((v.u + r) >> 16);
}
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}
static float rndn(void) {
    float s = 0; for (int i = 0; i < 4; i++) s += (float)(rnd32() & 0xffff) / 65535.0f - 0.5f;
    return s * 1.7f;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        uint32_t M = 2048, n_keys = 16384, window = 0, iters = 5;
        const char *shader = "metal/flint_prefill.metal", *variants = NULL;
        const char *names[16]; int n_names = 0;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--m") && i + 1 < argc) M = (uint32_t)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--keys") && i + 1 < argc) n_keys = (uint32_t)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = (uint32_t)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = (uint32_t)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--shader") && i + 1 < argc) shader = argv[++i];
            else if (!strcmp(argv[i], "--variants") && i + 1 < argc) variants = argv[++i];
            else if (n_names < 16) names[n_names++] = argv[i];
        }
        if (!n_names) names[n_names++] = "flint_attention";
        if (n_keys < M) n_keys = M;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:shader] encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "cannot read %s\n", shader); return 1; }
        if (variants) {
            NSString *extra = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:variants] encoding:NSUTF8StringEncoding error:&err];
            if (!extra) { fprintf(stderr, "cannot read %s\n", variants); return 1; }
            src = [src stringByAppendingFormat:@"\n%@", extra];
        }
        MTLCompileOptions *opt = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) opt.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }

        /* The chunk is the last M positions of a context of n_keys (global
         * layer: key_pos0 = 0); with a window the keys are the previous 512
         * plus the chunk, as the host lays the window buffer out. */
        AttnArgs a;
        a.M = M; a.ldq = 4096; a.ldo = 4096; a.window = window;
        if (window) { a.key_pos0 = n_keys - M - 512u; a.q_pos0 = n_keys - M; a.n_keys = 512u + M; }
        else { a.key_pos0 = 0; a.q_pos0 = n_keys - M; a.n_keys = n_keys; }
        a.kv_head_stride = a.n_keys * 128u;
        const uint32_t cap_rows = (M + 63) / 64 * 64;
        id<MTLBuffer> Q = [dev newBufferWithLength:(size_t)cap_rows * 4096 * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> K = [dev newBufferWithLength:(size_t)4 * a.n_keys * 128 * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> V = [dev newBufferWithLength:(size_t)4 * a.n_keys * 128 * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> sinks = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        id<MTLBuffer> args = [dev newBufferWithLength:sizeof a options:MTLResourceStorageModeShared];
        memcpy(args.contents, &a, sizeof a);
        uint16_t *p = Q.contents; for (size_t i = 0; i < (size_t)cap_rows * 4096; i++) p[i] = f2bf(rndn());
        p = K.contents; for (size_t i = 0; i < (size_t)4 * a.n_keys * 128; i++) p[i] = f2bf(rndn());
        p = V.contents; for (size_t i = 0; i < (size_t)4 * a.n_keys * 128; i++) p[i] = f2bf(rndn());
        p = sinks.contents; for (int i = 0; i < 32; i++) p[i] = f2bf(2.0f * rndn());
        /* unmasked (query, key) pairs */
        double pairs = 0;
        for (uint32_t i = 0; i < M; i++) {
            const uint32_t qpos = a.q_pos0 + i;
            uint32_t lo = 0;
            if (window && qpos + 1 > window) lo = qpos + 1 - window;
            pairs += (double)(qpos + 1 - lo);
        }
        const double flops = pairs * 128.0 * 4.0 * 32.0;
        printf("M=%u keys=%u window=%u: %.3g unmasked pairs, %.3g GFLOP per run, best of %u\n", M, a.n_keys, window, pairs, flops / 1e9, iters);

        uint16_t *ref = NULL;
        for (int k = 0; k < n_names; k++) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:names[k]]];
            if (!fn) { fprintf(stderr, "%s: no such kernel\n", names[k]); continue; }
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) { fprintf(stderr, "%s: %s\n", names[k], err.localizedDescription.UTF8String); continue; }
            const NSUInteger tg_threads = pso.maxTotalThreadsPerThreadgroup;
            /* every kernel declares its threadgroup size with
             * max_total_threads_per_threadgroup (128 or 256) */
            NSUInteger threads = tg_threads < 256 ? 128 : tg_threads == 256 ? 256 : 128;
            const uint32_t bq = strstr(names[k], "bq32") ? 32 : 64;
            id<MTLBuffer> O = [dev newBufferWithLength:(size_t)cap_rows * 4096 * 2 options:MTLResourceStorageModeShared];
            double best = 1e30, total = 0;
            for (uint32_t it = 0; it <= iters; it++) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:Q offset:0 atIndex:0];
                for (int b = 0; b < 4; b++) { [enc setBuffer:K offset:0 atIndex:1 + b]; [enc setBuffer:V offset:0 atIndex:5 + b]; }
                [enc setBuffer:sinks offset:0 atIndex:9];
                [enc setBuffer:O offset:0 atIndex:10];
                [enc setBuffer:args offset:0 atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake((M + bq - 1) / bq, 32, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.error) { fprintf(stderr, "%s: %s\n", names[k], cb.error.localizedDescription.UTF8String); return 1; }
                const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
                if (it == 0) continue;   /* warm-up */
                if (ms < best) best = ms;
                total += ms;
            }
            double maxd = 0; size_t bad = 0;
            const uint16_t *o = O.contents;
            if (!ref) {
                ref = malloc((size_t)M * 4096 * 2);
                memcpy(ref, o, (size_t)M * 4096 * 2);
            } else {
                for (size_t i = 0; i < (size_t)M * 4096; i++) {
                    const double d = fabs((double)bf2f(o[i]) - bf2f(ref[i]));
                    if (d > maxd) maxd = d;
                    if (d > 0.01 + 0.01 * fabs(bf2f(ref[i]))) bad++;
                }
            }
            printf("%-28s best %8.3f ms  avg %8.3f ms  %6.2f TFLOPS  threads %lu  regs %lu", names[k], best, total / iters,
                   flops / best / 1e9, (unsigned long)threads, (unsigned long)tg_threads);
            if (k) printf("  vs %s: max|d| %.3g, %zu outside 1%%", names[0], maxd, bad);
            printf("\n");
        }
        free(ref);
    }
    return 0;
}
