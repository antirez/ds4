/*
 * Micro-benchmark for the Flint prefill GEMM kernels in
 * metal/flint_prefill.metal: the dense projections of one 2048-token chunk
 * (q/k/v, o, the dense GLU and down), timed alone with GPU timestamps.
 *
 *   clang -fobjc-arc -O2 -framework Metal -framework Foundation -o tests/flint/bench_flint_gemm tests/flint/bench_flint_gemm.m
 *   tests/flint/bench_flint_gemm [--m 2048] [--iters 5] [--shader PATH] [--variants FILE] [suffix]
 *
 * With a suffix the kernels flint_gemm_q4_store<suffix> etc. are timed and
 * compared with the plain ones (variants appended from FILE).
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t M, N, K, lda, ldc, ldr; } GemmArgs;

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
static float rndu(void) { return (float)(rnd32() & 0xffffff) / 16777215.0f; }

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> lib;

static id<MTLBuffer> buf(size_t bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModeShared];
    memset(b.contents, 0, bytes ? bytes : 16);
    return b;
}
static id<MTLBuffer> bf16_random(size_t n) {
    id<MTLBuffer> b = buf(n * 2);
    uint16_t *p = b.contents;
    for (size_t i = 0; i < n; i++) p[i] = f2bf(rndn());
    return b;
}
typedef struct { id<MTLBuffer> w, s, b; } Q4;
static Q4 q4_random(uint32_t N, uint32_t K) {
    Q4 m = { buf((size_t)N * K / 2), buf((size_t)N * K / 64 * 2), buf((size_t)N * K / 64 * 2) };
    uint8_t *w = m.w.contents; uint16_t *s = m.s.contents, *bz = m.b.contents;
    for (size_t i = 0; i < (size_t)N * K / 2; i++) w[i] = (uint8_t)(rnd32() & 0xff);
    for (size_t i = 0; i < (size_t)N * K / 64; i++) { s[i] = f2bf(0.01f + 0.04f * rndu()); bz[i] = f2bf(-0.3f + 0.6f * rndu() - 8.0f * bf2f(s[i])); }
    return m;
}

/* times one GEMM; returns best ms over iters (after one warm-up); writes C */
static double time_gemm(const char *name, id<MTLBuffer> A, uint32_t lda, Q4 w, Q4 w2, id<MTLBuffer> C, uint32_t ldc,
                        id<MTLBuffer> R, id<MTLBuffer> silu, uint32_t M, uint32_t N, uint32_t K, uint32_t iters) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { fprintf(stderr, "%s: no such kernel\n", name); return -1; }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "%s: %s\n", name, err.localizedDescription.UTF8String); return -1; }
    const NSUInteger threads = pso.maxTotalThreadsPerThreadgroup >= 256 && strstr(name, "t256") ? 256 : 128;
    GemmArgs a = { M, N, K, lda, ldc, ldc };
    id<MTLBuffer> args = buf(sizeof a); memcpy(args.contents, &a, sizeof a);
    double best = 1e30;
    for (uint32_t it = 0; it <= iters; it++) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:A offset:0 atIndex:0];
        [enc setBuffer:w.w offset:0 atIndex:1];
        [enc setBuffer:w.s offset:0 atIndex:2];
        [enc setBuffer:w.b offset:0 atIndex:3];
        [enc setBuffer:C offset:0 atIndex:4];
        [enc setBuffer:args offset:0 atIndex:5];
        [enc setBuffer:R offset:0 atIndex:6];
        [enc setBuffer:w2.w offset:0 atIndex:7];
        [enc setBuffer:w2.s offset:0 atIndex:8];
        [enc setBuffer:w2.b offset:0 atIndex:9];
        [enc setBuffer:silu offset:0 atIndex:10];
        [enc setBuffer:R offset:0 atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "%s: %s\n", name, cb.error.localizedDescription.UTF8String); exit(1); }
        const double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
        if (it && ms < best) best = ms;
    }
    return best;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        uint32_t M = 2048, iters = 5;
        const char *shader = "metal/flint_prefill.metal", *variants = NULL, *suffix = "";
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--m") && i + 1 < argc) M = (uint32_t)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = (uint32_t)atoi(argv[++i]);
            else if (!strcmp(argv[i], "--shader") && i + 1 < argc) shader = argv[++i];
            else if (!strcmp(argv[i], "--variants") && i + 1 < argc) variants = argv[++i];
            else suffix = argv[i];
        }
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
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
        lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }

        id<MTLBuffer> silu = buf(65536 * 2);
        {
            uint16_t *t = silu.contents;
            for (uint32_t i = 0; i < 65536; i++) { const float x = bf2f((uint16_t)i); t[i] = f2bf(x / (1.0f + expf(-x))); }
        }
        struct { const char *what, *kernel; uint32_t N, K; int two; } shapes[] = {
            { "q proj  [2048]->[4096]", "flint_gemm_q4_store", 4096, 2048, 0 },
            { "k proj  [2048]->[512]", "flint_gemm_q4_store", 512, 2048, 0 },
            { "o proj  [4096]->[2048]+R", "flint_gemm_q4_residual", 2048, 4096, 0 },
            { "dense GLU [2048]->[6144]x2", "flint_gemm_q4_glu", 6144, 2048, 1 },
            { "dense down [6144]->[2048]+R", "flint_gemm_q4_residual", 2048, 6144, 0 },
        };
        printf("M=%u, best of %u\n", M, iters);
        for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
            const uint32_t N = shapes[i].N, K = shapes[i].K;
            id<MTLBuffer> A = bf16_random((size_t)M * K), C = buf((size_t)M * N * 2), R = bf16_random((size_t)M * N);
            Q4 w = q4_random(N, K), w2 = shapes[i].two ? q4_random(N, K) : w;
            char name[128];
            snprintf(name, sizeof name, "%s%s", shapes[i].kernel, suffix);
            const double ms = time_gemm(name, A, K, w, w2, C, N, R, silu, M, N, K, iters);
            if (ms < 0) continue;
            const double flops = 2.0 * M * N * K * (shapes[i].two ? 2 : 1);
            printf("  %-30s %-28s %8.3f ms  %6.2f TFLOPS", shapes[i].what, name, ms, flops / ms / 1e9);
            if (suffix[0]) {   /* compare with the plain kernel */
                id<MTLBuffer> C0 = buf((size_t)M * N * 2);
                time_gemm(shapes[i].kernel, A, K, w, w2, C0, N, R, silu, M, N, K, 1);
                const uint16_t *p = C.contents, *q = C0.contents;
                size_t diff = 0;
                for (size_t j = 0; j < (size_t)M * N; j++) if (p[j] != q[j]) diff++;
                printf("  %zu of %zu elements differ from %s", diff, (size_t)M * N, shapes[i].kernel);
            }
            printf("\n");
        }
    }
    return 0;
}
