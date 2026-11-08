/* Compare the production Qwen HC-down F16 tile with the generic F16 GEMM.
 *
 * Build:
 *   clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
 *       tests/test_metal_qwen4_hc.m -o /tmp/test_metal_qwen4_hc
 * Run from the repository root, or pass --repo PATH.
 * Default/--test: 92 fixtures, default/safe math, bitwise GPU parity, guards,
 * immutable inputs, and independent double-reference samples.
 * --compile-only: compile all 16 pipelines without creating a GPU queue.
 * --bench: warm repeated-kernel ABBA/BAAB comparison; not model throughput.
 * --output DIR: optionally write machine-readable results; default is stdout.
 *
 * Both kernels, wrappers, argument layout, and dequantization helpers are
 * extracted from metal/dense.metal. Neither oracle nor candidate is generated
 * from the other. This is a kernel test; host dispatch selection is separate.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1;
    int16_t r2, r3;
} mm_args;
_Static_assert(sizeof(mm_args) == 88, "production matmul argument ABI");
_Static_assert(offsetof(mm_args, nb10) == 40, "production stride alignment");
_Static_assert(offsetof(mm_args, ne0) == 72, "production output dimensions");

typedef struct { uint32_t k, n, tokens; } shape;
static const uint32_t kCanary = UINT32_C(0x7f24d13b);
static const NSUInteger kOffset = 16;
static const shape kShapes[] = {
    {10240, 320, 9},   {10240, 320, 29},  {10240, 320, 63},  {10240, 320, 128},
    {10240, 320, 1},   {10240, 320, 2},   {10240, 320, 15},  {10240, 320, 16},
    {10240, 320, 17},  {10240, 320, 31},  {10240, 320, 32},  {10240, 320, 33},
    {10240, 320, 64},  {10240, 320, 127}, {10240, 320, 129},
    {10243, 316, 29},  {10239, 324, 63},  {31, 28, 9},       {33, 36, 17},
    {65, 68, 33},     {8, 4, 1},        {96, 32, 16},      {128, 64, 32},
};

static void fail(NSString *message) {
    fprintf(stderr, "FAIL Qwen HC: %s\n", message.UTF8String);
    exit(1);
}

static NSRange unique_range(NSString *source, NSString *anchor) {
    if ([source componentsSeparatedByString:anchor].count != 2) {
        fail([NSString stringWithFormat:@"missing/non-unique source anchor: %@", anchor]);
    }
    return [source rangeOfString:anchor];
}

static NSString *between(NSString *source, NSString *begin, NSString *end) {
    NSRange a = unique_range(source, begin), b = unique_range(source, end);
    if (b.location <= NSMaxRange(a)) fail(@"invalid production source range");
    return [source substringWithRange:NSMakeRange(a.location, b.location - a.location)];
}

static NSString *source_line(NSString *source, NSString *anchor) {
    NSRange a = unique_range(source, anchor);
    NSRange rest = NSMakeRange(a.location, source.length - a.location);
    NSRange newline = [source rangeOfString:@"\n" options:0 range:rest];
    if (newline.location == NSNotFound) fail(@"unterminated production source line");
    return [source substringWithRange:NSMakeRange(a.location, newline.location + 1 - a.location)];
}

static NSString *kernel_template(NSString *source, NSString *name, NSString *end) {
    NSRange function = unique_range(source, name);
    NSRange start = [source rangeOfString:@"template<typename " options:NSBackwardsSearch
                                   range:NSMakeRange(0, function.location)];
    NSRange stop = unique_range(source, end);
    if (start.location == NSNotFound || stop.location <= function.location) {
        fail(@"invalid production kernel template range");
    }
    return [source substringWithRange:NSMakeRange(start.location, stop.location - start.location)];
}

static NSString *production_source(NSString *repo) {
    NSError *error = nil;
    NSString *path = [repo stringByAppendingPathComponent:@"metal/dense.metal"];
    NSString *dense = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (!dense) fail([NSString stringWithFormat:@"cannot read %@: %@", path, error]);
    NSMutableString *source = [NSMutableString stringWithString:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "#define FC_MUL_MM 700\n"
         "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"];
    [source appendString:between(dense, @"struct ds4_metal_args_mul_mm {",
                                       @"struct ds4_metal_args_mul_mv_ext {")];
    [source appendString:between(dense, @"template <typename type4x4>\nvoid dequantize_f32(",
                                       @"template <typename type4x4>\nvoid dequantize_q8_0(")];
    [source appendString:source_line(dense, @"constant bool FC_mul_mm_bc_inp ")];
    [source appendString:source_line(dense, @"constant bool FC_mul_mm_bc_out ")];
    [source appendString:kernel_template(dense, @"kernel void kernel_mul_mm(",
                                                @"/* Narrow HC-down prefill on M1 Max:")];
    [source appendString:source_line(dense, @"typedef decltype(kernel_mul_mm<")];
    [source appendString:source_line(dense, @"template [[host_name(\"kernel_mul_mm_f16_f32\")]]")];
    [source appendString:kernel_template(dense, @"kernel void kernel_qwen4_hc_down_mm32x16(",
                                                @"// Legacy F16-weight/F32-RHS prefill matmul")];
    return source;
}

static uint32_t mix_bits(uint32_t x) {
    x ^= x >> 16; x *= UINT32_C(0x7feb352d);
    x ^= x >> 15; x *= UINT32_C(0x846ca68b);
    return x ^ (x >> 16);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes) {
    bytes = (bytes + 3) / 4 * 4;
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes + 2 * kOffset
                                            options:MTLResourceStorageModeShared];
    if (!buffer) fail(@"buffer allocation");
    uint32_t *words = buffer.contents;
    for (size_t i = 0; i < buffer.length / 4; ++i) words[i] = kCanary;
    return buffer;
}

static void check_guards(NSArray<id<MTLBuffer>> *buffers) {
    for (id<MTLBuffer> buffer in buffers) {
        const uint32_t *words = buffer.contents;
        size_t count = buffer.length / 4;
        for (size_t i = 0; i < kOffset / 4; ++i) {
            if (words[i] != kCanary || words[count - kOffset / 4 + i] != kCanary) {
                fail(@"buffer guard overwritten");
            }
        }
    }
}

static void poison_output(id<MTLBuffer> output) {
    uint32_t *words = output.contents;
    for (size_t i = 0; i < output.length / 4; ++i) words[i] = kCanary;
}

static NSArray<id<MTLBuffer>> *fixture(id<MTLDevice> device, shape s, int pattern) {
    id<MTLBuffer> weights = make_buffer(device, (size_t)s.k * s.n * sizeof(_Float16));
    id<MTLBuffer> input = make_buffer(device, (size_t)s.k * s.tokens * sizeof(float));
    id<MTLBuffer> output = make_buffer(device, (size_t)s.n * s.tokens * sizeof(float));
    _Float16 *w = (_Float16 *)((char *)weights.contents + kOffset);
    float *x = (float *)((char *)input.contents + kOffset);
    for (size_t i = 0; i < (size_t)s.k * s.n; ++i) {
        float value = ((int)(mix_bits((uint32_t)i + 123) % 2049) - 1024) * 0.0001220703125f;
        if (pattern) value = (i % 2 ? -1.f : 1.f) * (i % 13 == 0 ? 0.5f : 0.000030517578125f);
        w[i] = value;
    }
    for (size_t i = 0; i < (size_t)s.k * s.tokens; ++i) {
        float value = ((int)(mix_bits((uint32_t)i + 941) % 4097) - 2048) * 0.00048828125f + 0.00000017f;
        if (pattern) value = (i % 2 ? -1.f : 1.f) * (i % 7 == 0 ? 0.0625f : 0x1p-24f);
        if (i < s.k && i % 3 == 0) {
            uint32_t zero = i % 2 ? UINT32_C(0x80000000) : 0;
            memcpy(&value, &zero, sizeof(value));
        }
        x[i] = value;
    }
    return @[weights, input, output];
}

static NSDictionary *dispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                              NSArray<id<MTLBuffer>> *buffers, shape s, bool candidate, int repeats) {
    mm_args args = {
        .ne00 = (int)s.k, .ne02 = 1, .nb01 = (uint64_t)s.k * 2,
        .nb02 = (uint64_t)s.k * s.n * 2, .nb03 = (uint64_t)s.k * s.n * 2,
        .ne12 = 1, .nb10 = 4, .nb11 = (uint64_t)s.k * 4,
        .nb12 = (uint64_t)s.k * s.tokens * 4, .nb13 = (uint64_t)s.k * s.tokens * 4,
        .ne0 = (int)s.n, .ne1 = (int)s.tokens, .r2 = 1, .r3 = 1,
    };
    const uint32_t rows = candidate ? 32 : 64, tokens = candidate ? 16 : 32;
    const bool tail = s.n % rows || s.tokens % tokens;
    const double start = now();
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) fail(@"command allocation");
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:&args length:sizeof(args) atIndex:0];
    for (int i = 0; i < 3; ++i) [encoder setBuffer:buffers[i] offset:kOffset atIndex:i + 1];
    [encoder setThreadgroupMemoryLength:candidate ? 3072 : tail ? 8192 : 6144 atIndex:0];
    for (int i = 0; i < repeats; ++i) {
        [encoder dispatchThreadgroups:MTLSizeMake((s.tokens + tokens - 1) / tokens,
                                                  (s.n + rows - 1) / rows, 1)
                 threadsPerThreadgroup:MTLSizeMake(candidate ? 64 : 128, 1, 1)];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) fail(command.error.description);
    const double host = now() - start, gpu = command.GPUEndTime - command.GPUStartTime;
    if (!isfinite(gpu) || gpu <= 0 || !isfinite(host)) fail(@"invalid command timing");
    return @{@"gpu_us": @(gpu * 1e6 / repeats), @"host_us": @(host * 1e6 / repeats)};
}

static NSArray<NSData *> *snapshot(NSArray<id<MTLBuffer>> *buffers) {
    NSMutableArray *result = [NSMutableArray new];
    for (id<MTLBuffer> buffer in buffers) {
        [result addObject:[NSData dataWithBytes:buffer.contents length:buffer.length]];
    }
    return result;
}

static void check_exact(NSArray<id<MTLBuffer>> *buffers, NSArray<NSData *> *reference) {
    check_guards(buffers);
    for (int b = 0; b < 3; ++b) {
        const uint32_t *expected = reference[b].bytes, *actual = buffers[b].contents;
        if (reference[b].length != buffers[b].length) fail(@"reference buffer size changed");
        for (size_t i = 0; i < reference[b].length / 4; ++i) {
            if (expected[i] != actual[i]) {
                fail([NSString stringWithFormat:@"buffer %d word %zu: expected %08x, got %08x",
                                                b, i, expected[i], actual[i]]);
            }
        }
    }
}

/* Independent double accumulation, including the production FP32->FP16 input
 * rounding. A conservative error bound checks gross math/layout errors; exact
 * GPU parity above is the numerical acceptance criterion. */
static double check_cpu_samples(NSArray<id<MTLBuffer>> *buffers, shape s) {
    const _Float16 *w = (const _Float16 *)((const char *)buffers[0].contents + kOffset);
    const float *x = (const float *)((const char *)buffers[1].contents + kOffset);
    const float *y = (const float *)((const char *)buffers[2].contents + kOffset);
    double maximum = 0;
    for (int sample = 0; sample < 8; ++sample) {
        uint32_t row = sample == 0 ? 0 : sample == 1 ? s.n - 1 : mix_bits(sample + 372) % s.n;
        uint32_t tok = sample == 0 ? 0 : sample == 1 ? s.tokens - 1 : mix_bits(sample + 863) % s.tokens;
        double sum = 0, sum_abs = 0;
        for (uint32_t k = 0; k < s.k; ++k) {
            double product = (double)w[(size_t)row * s.k + k] * (double)(_Float16)x[(size_t)tok * s.k + k];
            sum += product;
            sum_abs += fabs(product);
        }
        double actual = y[(size_t)tok * s.n + row], error = fabs(actual - sum);
        if (!isfinite(actual) || error > 0.0005 + 0.00005 * sum_abs) fail(@"sampled CPU reference mismatch");
        maximum = fmax(maximum, error);
    }
    return maximum;
}

static id<MTLLibrary> make_library(id<MTLDevice> device, NSString *source, bool safe) {
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (safe) {
        if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
    }
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) fail(error.description);
    return library;
}

static NSArray *make_pipelines(id<MTLDevice> device, id<MTLLibrary> library, int math,
                              NSMutableArray *resources) {
    NSMutableArray *variants = [NSMutableArray new];
    for (int candidate = 0; candidate < 2; ++candidate) {
        NSMutableArray *pipelines = [NSMutableArray new];
        for (int flags = 0; flags < 4; ++flags) {
            bool input_bounds = flags & 1, output_bounds = flags & 2;
            MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
            [values setConstantValue:&input_bounds type:MTLDataTypeBool atIndex:700];
            [values setConstantValue:&output_bounds type:MTLDataTypeBool atIndex:701];
            NSError *error = nil;
            NSString *name = candidate ? @"kernel_qwen4_hc_down_mm_f16" : @"kernel_mul_mm_f16_f32";
            id<MTLFunction> function = [library newFunctionWithName:name constantValues:values error:&error];
            if (!function) fail(error.description);
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) fail(error.description);
            if (pipeline.threadExecutionWidth != 32 ||
                pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)(candidate ? 64 : 128)) {
                fail(@"unsupported threadgroup/SIMD size");
            }
            [pipelines addObject:pipeline];
            [resources addObject:@{@"name": name, @"math": @(math), @"flags": @(flags),
                                   @"static_tg_bytes": @(pipeline.staticThreadgroupMemoryLength),
                                   @"max_threads": @(pipeline.maxTotalThreadsPerThreadgroup)}];
        }
        [variants addObject:pipelines];
    }
    return variants;
}

static void write_results(NSString *directory, NSString *mode, id results) {
    if (!directory) return;
    NSError *error = nil;
    if (![[NSFileManager defaultManager] createDirectoryAtPath:directory withIntermediateDirectories:YES
                                                   attributes:nil error:&error]) fail(error.description);
    NSData *data = [NSJSONSerialization dataWithJSONObject:results options:NSJSONWritingPrettyPrinted error:&error];
    NSString *path = [directory stringByAppendingPathComponent:[mode stringByAppendingString:@".json"]];
    if (!data || ![data writeToFile:path options:NSDataWritingAtomic error:&error]) fail(error.description);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        NSString *repo = @".", *output = nil, *mode = @"test";
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "--test")) mode = @"test";
            else if (!strcmp(argv[i], "--bench")) mode = @"bench";
            else if (!strcmp(argv[i], "--compile-only")) mode = @"compile";
            else if (!strcmp(argv[i], "--repo") && i + 1 < argc) repo = @(argv[++i]);
            else if (!strcmp(argv[i], "--output") && i + 1 < argc) output = @(argv[++i]);
            else {
                fprintf(stderr, "Usage: %s [--test|--bench|--compile-only] [--repo PATH] [--output DIR]\n", argv[0]);
                return 2;
            }
        }
        const bool compile_only = [mode isEqualToString:@"compile"], bench = [mode isEqualToString:@"bench"];
        NSString *source = production_source(repo);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) fail(@"no Metal device available");
        NSMutableArray *resources = [NSMutableArray new], *tests = [NSMutableArray new], *runs = [NSMutableArray new];
        NSMutableArray *math_pipelines = [NSMutableArray new];
        for (int math = 0; math < (bench ? 1 : 2); ++math) {
            id<MTLLibrary> library = make_library(device, source, math != 0);
            [math_pipelines addObject:make_pipelines(device, library, math, resources)];
        }
        /* All compilation happens before any GPU queue or data allocation. */
        if (!compile_only) {
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!queue) fail(@"command queue allocation");
            for (NSUInteger math = 0; math < math_pipelines.count; ++math) {
                NSArray *pipelines = math_pipelines[math];
                for (size_t si = 0; si < (bench ? 4 : sizeof(kShapes) / sizeof(kShapes[0])); ++si) {
                    for (int pattern = 0; pattern < (bench ? 1 : 2); ++pattern) {
                        @autoreleasepool {
                            shape s = kShapes[si];
                            NSArray<id<MTLBuffer>> *buffers = fixture(device, s, pattern);
                            int flags[2] = {
                                (s.k % 32 ? 1 : 0) | ((s.n % 64 || s.tokens % 32) ? 2 : 0),
                                (s.k % 32 ? 1 : 0) | ((s.n % 32 || s.tokens % 16) ? 2 : 0),
                            };
                            NSArray *inputs = snapshot(buffers);
                            (void)dispatch(queue, pipelines[0][flags[0]], buffers, s, false, 1);
                            check_guards(buffers);
                            /* Validate both inputs against the pre-dispatch snapshot. */
                            for (int i = 0; i < 2; ++i) {
                                NSData *frozen = inputs[i];
                                if (memcmp(frozen.bytes, buffers[i].contents, frozen.length)) fail(@"baseline modified its input");
                            }
                            double cpu_error = check_cpu_samples(buffers, s);
                            NSArray *reference = snapshot(buffers);
                            poison_output(buffers[2]);
                            (void)dispatch(queue, pipelines[1][flags[1]], buffers, s, true, 1);
                            check_exact(buffers, reference);
                            [tests addObject:@{@"math": @(math), @"K": @(s.k), @"N": @(s.n), @"T": @(s.tokens),
                                               @"pattern": @(pattern), @"cpu_max_abs": @(cpu_error), @"pass": @YES}];
                            printf("PASS HC-down %s K=%u N=%u T=%u pattern=%d: exact output, guards, frozen inputs\n",
                                   math ? "safe" : "default", s.k, s.n, s.tokens, pattern);
                            fflush(stdout);
                            if (!bench) continue;
                            const int warmup[] = {0, 1, 1, 0};
                            for (size_t i = 0; i < sizeof(warmup) / sizeof(warmup[0]); ++i) {
                                int candidate = warmup[i];
                                poison_output(buffers[2]);
                                (void)dispatch(queue, pipelines[candidate][flags[candidate]], buffers, s, candidate, 16);
                                check_exact(buffers, reference);
                            }
                            const int order[] = {0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1};
                            for (size_t i = 0; i < sizeof(order) / sizeof(order[0]); ++i) {
                                int candidate = order[i];
                                poison_output(buffers[2]);
                                NSMutableDictionary *timing = [dispatch(queue, pipelines[candidate][flags[candidate]],
                                                                        buffers, s, candidate, 16) mutableCopy];
                                check_exact(buffers, reference);
                                [timing addEntriesFromDictionary:@{@"candidate": @(candidate), @"K": @(s.k),
                                                                   @"N": @(s.n), @"T": @(s.tokens), @"order": @(i)}];
                                [runs addObject:timing];
                                printf("BENCH K=%u N=%u T=%u sample=%zu %s GPU=%.3f us host=%.3f us\n",
                                       s.k, s.n, s.tokens, i, candidate ? "candidate" : "baseline",
                                       [timing[@"gpu_us"] doubleValue], [timing[@"host_us"] doubleValue]);
                                fflush(stdout);
                            }
                        }
                    }
                }
            }
        }
        write_results(output, mode, @{@"device": device.name, @"mode": mode, @"pipelines": resources,
                                      @"tests": tests, @"runs": runs, @"dispatches_per_timing": @16});
        printf("PASS Qwen HC %s: %lu pipelines, %lu fixtures, %lu timing samples%s\n", mode.UTF8String,
               (unsigned long)resources.count, (unsigned long)tests.count, (unsigned long)runs.count,
               compile_only ? "; no queue, buffers, command buffers, or dispatches created" : "");
    }
    return 0;
}
