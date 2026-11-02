/* Compare the production Q8 matvec reduction against a frozen legacy oracle.
 *
 * Build:
 *   clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
 *       tests/test_metal_q8_reduction.m -o /tmp/test_metal_q8_reduction
 * Run from the repository root, or pass --repo PATH.
 * --compile-only builds both libraries and all 32 specialized pipelines, then
 * exits before creating a queue, buffers, command buffers, or GPU dispatches.
 *
 * This tests only reduction and bounds, not quantization or complete GEMV.
 * Both stages of simd_sum are intentionally present in the frozen oracle,
 * including NSG=1. NaN classification must match, but NaN payload bits are not
 * prescribed by Metal and are counted rather than asserted. All non-NaN
 * outputs (including signed zero, subnormals, and infinities) must match bits.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    kRows = 2,
    kWidth = 32,
    kLayouts = 6,
    kPatterns = 32,
    kSamples = kLayouts * kPatterns,
    kOutputStride = 16,
    kGuard = 4,
};
static const uint32_t kCanary = UINT32_C(0x7f24d13b);

/* Independent fixture: copied from helper_mv_reduce_and_write at 5cbdb6e.
 * Never derive this body from the candidate or the current generic helper. */
static NSString *const kLegacyReduction =
@"template<short NR0>\n"
 "static inline void legacy_mv_reduce_and_write(\n"
 "        device float * dst_f32,\n"
 "        float sumf[NR0],\n"
 "        const int r0,\n"
 "        const int ne01,\n"
 "        ushort tiisg,\n"
 "        ushort sgitg,\n"
 "        threadgroup char * shmem) {\n"
 "    constexpr short NW = N_SIMDWIDTH;\n"
 "    threadgroup float * shmem_f32[NR0];\n"
 "    for (short row = 0; row < NR0; ++row) {\n"
 "        shmem_f32[row] = (threadgroup float *) shmem + NW*row;\n"
 "        if (sgitg == 0) {\n"
 "            shmem_f32[row][tiisg] = 0.0f;\n"
 "        }\n"
 "        sumf[row] = simd_sum(sumf[row]);\n"
 "    }\n"
 "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
 "    for (short row = 0; row < NR0; ++row) {\n"
 "        if (tiisg == 0) {\n"
 "            shmem_f32[row][sgitg] = sumf[row];\n"
 "        }\n"
 "    }\n"
 "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
 "    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {\n"
 "        float tot = simd_sum(shmem_f32[row][tiisg]);\n"
 "        if (tiisg == 0 && sgitg == 0) {\n"
 "            dst_f32[r0 + row] = tot;\n"
 "        }\n"
 "    }\n"
 "}\n";

static void fail(const char *message) {
    fprintf(stderr, "FAIL Q8 reduction: %s\n", message);
    exit(1);
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool is_nan_bits(uint32_t bits) {
    return (bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000);
}

static NSString *production_reduction(NSString *repo) {
    NSError *error = nil;
    NSString *path = [repo stringByAppendingPathComponent:@"metal/dense.metal"];
    NSString *dense = [NSString stringWithContentsOfFile:path
                                              encoding:NSUTF8StringEncoding
                                                 error:&error];
    if (!dense) {
        fprintf(stderr, "Cannot read %s: %s\n", path.UTF8String,
                error.localizedDescription.UTF8String);
        exit(1);
    }
    NSString *begin = @"template<short NR0>\nstatic inline void helper_q8_mv_reduce_and_write(";
    NSString *end = @"template<short NR0, typename args_t>\nvoid kernel_mul_mv_q8_0_f32_plain_impl(";
    if ([[dense componentsSeparatedByString:begin] count] != 2 ||
        [[dense componentsSeparatedByString:end] count] != 2) {
        fail("production helper anchors missing or non-unique; update extraction explicitly");
    }
    NSRange start = [dense rangeOfString:begin];
    NSRange stop = [dense rangeOfString:end];
    if (stop.location <= NSMaxRange(start)) fail("invalid production helper source range");
    return [dense substringWithRange:NSMakeRange(start.location, stop.location - start.location)];
}

static NSString *wrapper(NSString *name, NSString *helper) {
    return [NSString stringWithFormat:
        @"kernel void %@(\n"
         "    device const float *partials [[buffer(0)]],\n"
         "    device const uint2 *ranges [[buffer(1)]],\n"
         "    device float *outputs [[buffer(2)]],\n"
         "    threadgroup char *shmem [[threadgroup(0)]],\n"
         "    uint3 group [[threadgroup_position_in_grid]],\n"
         "    ushort lane [[thread_index_in_simdgroup]],\n"
         "    ushort simd [[simdgroup_index_in_threadgroup]]) {\n"
         "    const uint sample = group.x;\n"
         "    const uint nsg = FC_mul_mv_nsg;\n"
         "    // Poison padding so missing zero lanes cannot pass by accident.\n"
         "    threadgroup uint *poison = (threadgroup uint *)shmem;\n"
         "    for (uint i = simd*32 + lane; i < 64; i += nsg*32) {\n"
         "        poison[i] = 0x7fc04a00u + i;\n"
         "    }\n"
         "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "    float sums[2];\n"
         "    for (uint row = 0; row < 2; ++row) {\n"
         "        sums[row] = partials[((sample*2 + row)*nsg + simd)*32 + lane];\n"
         "    }\n"
         "    %@(outputs + 4 + sample*16, sums, (int)ranges[sample].x,\n"
         "         (int)ranges[sample].y, lane, simd, shmem);\n"
         "}\n", name, [helper stringByAppendingString:@"<2>"]];
}

static id<MTLLibrary> make_library(id<MTLDevice> device, NSString *source, bool safe) {
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (safe) {
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
    }
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        fprintf(stderr, "Metal library (%s): %s\n", safe ? "safe" : "default",
                error.localizedDescription.UTF8String);
        exit(1);
    }
    return library;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                                id<MTLLibrary> library,
                                                NSString *name, short nsg) {
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:0];
    NSError *error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name constantValues:constants error:&error];
    id<MTLComputePipelineState> pipeline = function ?
        [device newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!pipeline) {
        fprintf(stderr, "Metal pipeline %s NSG=%d: %s\n", name.UTF8String, nsg,
                error.localizedDescription.UTF8String);
        exit(1);
    }
    if (pipeline.threadExecutionWidth != kWidth || pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)(nsg*kWidth)) {
        fail("device cannot execute the required SIMD width or threadgroup size");
    }
    return pipeline;
}

static uint32_t random_word(uint32_t value) {
    value ^= value >> 16;
    value *= UINT32_C(0x7feb352d);
    value ^= value >> 15;
    value *= UINT32_C(0x846ca68b);
    return value ^ (value >> 16);
}

static uint32_t input_word(unsigned pattern, unsigned row, unsigned simd, unsigned lane) {
    const unsigned index = simd*kWidth + lane;
    const uint32_t sign = (index + row) & 1 ? UINT32_C(0x80000000) : 0;
    switch (pattern) {
        case 0: return 0;                                      /* Positive zero. */
        case 1: return UINT32_C(0x80000000);                     /* Negative zero. */
        case 2: return row ? UINT32_C(0xbf800000) : UINT32_C(0x3f800000);
        case 3: return sign;                                   /* Mixed zeros. */
        case 4: return sign | (1u + index % 31);                /* Tiny subnormals. */
        case 5: return sign | UINT32_C(0x007fffff);             /* Largest subnormal. */
        case 6: return sign | UINT32_C(0x00800000);             /* Smallest normal. */
        case 7: return float_bits((lane & 1 ? -1.0f : 1.0f) * 0x1p24f);
        case 8: {                                              /* Lossy cancellation. */
            const float values[] = {0x1p60f, 1.0f, -0x1p60f, -0.5f};
            return float_bits(values[(index + row) % 4]);
        }
        case 9: return sign | UINT32_C(0x7f7fffff);             /* Extremes/overflow. */
        case 10: return float_bits((float)(simd + 1) * (row ? -1.0f : 1.0f));
        case 11: return float_bits((float)((int)(index % 17) - 8));
        case 12: return index == 0 ? UINT32_C(0x7f800000) : 0;  /* +Inf. */
        case 13: return index == 0 ? UINT32_C(0xff800000) : 0;  /* -Inf. */
        case 14: return lane == 0 ? (sign | UINT32_C(0x7f800000)) : float_bits(0.25f);
        case 15: return index == row ? UINT32_C(0x7fc00125) : float_bits(1.0f);
        case 16: return lane == 0 ? (UINT32_C(0xffc00000) | (simd + 1)) : 0;
        case 17: return lane == 0 ? (simd & 1 ? UINT32_C(0xff800000) : UINT32_C(0x7f800000)) : 0;
        case 18: return lane == 0 ? float_bits((float)(simd + 1)) : UINT32_C(0x80000000);
        default: {
            const uint32_t random = random_word(pattern*977u + row*65537u + index*7919u + 1);
            /* Finite, wide exponent distribution; reductions may overflow. */
            return (random & UINT32_C(0x807fffff)) | (((random >> 23) % 254u) << 23);
        }
    }
}

static bool written_row(unsigned sample, unsigned column, const uint32_t *ranges) {
    const unsigned r0 = ranges[2*sample], ne01 = ranges[2*sample + 1];
    return column >= r0 && column < r0 + kRows && column < ne01;
}

static void encode(id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> partials, id<MTLBuffer> ranges, id<MTLBuffer> outputs,
                   short nsg) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) fail("cannot create command encoder");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:partials offset:0 atIndex:0];
    [encoder setBuffer:ranges offset:0 atIndex:1];
    [encoder setBuffer:outputs offset:0 atIndex:2];
    /* Both helpers receive the production NR2 scratch contract, including
     * NSG=1. Optimizing away its use does not change the host ABI. */
    [encoder setThreadgroupMemoryLength:kRows*kWidth*sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(kSamples, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kWidth*nsg, 1, 1)];
    [encoder endEncoding];
}

static void run_cases(id<MTLDevice> device, id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> legacy,
                      id<MTLComputePipelineState> candidate, short nsg, const char *mode,
                      size_t *total_rows, size_t *nan_payload_differences) {
    const NSUInteger input_count = kSamples*kRows*nsg*kWidth;
    const NSUInteger output_count = kGuard + kSamples*kOutputStride + kGuard;
    id<MTLBuffer> partials = [device newBufferWithLength:input_count*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> ranges = [device newBufferWithLength:kSamples*2*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> reference = [device newBufferWithLength:output_count*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> actual = [device newBufferWithLength:output_count*sizeof(float) options:MTLResourceStorageModeShared];
    if (!partials || !ranges || !reference || !actual) fail("buffer allocation failed");
    uint32_t *input = partials.contents, *range = ranges.contents;
    const uint32_t layouts[kLayouts][2] = {{2, 4}, {2, 3}, {2, 2}, {0, 2}, {13, 15}, {13, 14}};
    for (unsigned sample = 0; sample < kSamples; ++sample) {
        memcpy(range + 2*sample, layouts[sample % kLayouts], 2*sizeof(uint32_t));
        for (unsigned row = 0; row < kRows; ++row) {
            for (unsigned simd = 0; simd < (unsigned)nsg; ++simd) {
                for (unsigned lane = 0; lane < kWidth; ++lane) {
                    input[((sample*kRows + row)*nsg + simd)*kWidth + lane] =
                        input_word(sample/kLayouts, row, simd, lane);
                }
            }
        }
    }
    uint32_t *expected = reference.contents, *observed = actual.contents;
    for (NSUInteger i = 0; i < output_count; ++i) expected[i] = observed[i] = kCanary;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!command) fail("cannot create command buffer");
    encode(command, legacy, partials, ranges, reference, nsg);
    encode(command, candidate, partials, ranges, actual, nsg);
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "GPU command failed: %s\n", command.error.localizedDescription.UTF8String);
        exit(1);
    }
    size_t rows = 0;
    for (NSUInteger i = 0; i < output_count; ++i) {
        bool active = false;
        unsigned sample = 0, column = 0;
        if (i >= kGuard && i < output_count - kGuard) {
            sample = (unsigned)(i - kGuard)/kOutputStride;
            column = (unsigned)(i - kGuard)%kOutputStride;
            active = written_row(sample, column, range);
        }
        if (!active) {
            if (expected[i] != kCanary || observed[i] != kCanary) fail("output guard or tail row overwritten");
            continue;
        }
        ++rows;
        const bool reference_nan = is_nan_bits(expected[i]);
        const bool candidate_nan = is_nan_bits(observed[i]);
        if (reference_nan && candidate_nan) {
            *nan_payload_differences += expected[i] != observed[i];
        } else if (expected[i] != observed[i]) {
            fprintf(stderr, "FAIL %s NSG=%d pattern=%u layout=%u column=%u: "
                    "legacy=%08x (%g), candidate=%08x (%g)\n", mode, nsg,
                    sample/kLayouts, sample%kLayouts, column, expected[i], bits_float(expected[i]),
                    observed[i], bits_float(observed[i]));
            exit(1);
        }
        /* Independent analytical check: exactly representable integer sums. */
        const unsigned pattern = sample/kLayouts, row = column - range[2*sample];
        if (pattern == 2 || pattern == 10) {
            float total = pattern == 2 ? (float)(kWidth*nsg) : (float)(kWidth*nsg*(nsg + 1)/2);
            if (row) total = -total;
            if (observed[i] != float_bits(total) || expected[i] != float_bits(total)) {
                fail("analytical integer-sum oracle failed");
            }
        }
    }
    *total_rows += rows;
    printf("PASS %s NSG=%d NR=2: %d cases, %zu output rows, guards intact\n", mode, nsg, kSamples, rows);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        bool compile_only = false;
        NSString *repo = @".";
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "--compile-only")) compile_only = true;
            else if (!strcmp(argv[i], "--repo") && i + 1 < argc) repo = @(argv[++i]);
            else {
                fprintf(stderr, "Usage: %s [--compile-only] [--repo PATH]\n", argv[0]);
                return 2;
            }
        }
        NSString *production = production_reduction(repo);
        NSString *source = [@"#include <metal_stdlib>\nusing namespace metal;\n"
                             "constant short FC_mul_mv_nsg [[function_constant(0)]];\n"
                             "constant short N_SIMDWIDTH = 32;\n"
                            stringByAppendingString:kLegacyReduction];
        source = [source stringByAppendingString:production];
        source = [source stringByAppendingString:wrapper(@"reduce_legacy", @"legacy_mv_reduce_and_write")];
        source = [source stringByAppendingString:wrapper(@"reduce_candidate", @"helper_q8_mv_reduce_and_write")];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) fail("no Metal device available");
        NSMutableArray<id<MTLComputePipelineState>> *pipelines = [NSMutableArray new];
        for (int mode = 0; mode < 2; ++mode) {
            id<MTLLibrary> library = make_library(device, source, mode != 0);
            for (short nsg = 1; nsg <= 8; ++nsg) {
                [pipelines addObject:make_pipeline(device, library, @"reduce_legacy", nsg)];
                [pipelines addObject:make_pipeline(device, library, @"reduce_candidate", nsg)];
            }
            printf("PASS compile %s: 16 pipelines, NSG=1..8, NR=2\n", mode ? "safe" : "default");
        }
        if (compile_only) {
            printf("PASS compile-only: 32 pipelines; no queue, buffers, command buffers, or dispatches created\n");
            return 0;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) fail("cannot create Metal command queue");
        size_t total_rows = 0, nan_payload_differences = 0;
        for (int mode = 0; mode < 2; ++mode) {
            for (short nsg = 1; nsg <= 8; ++nsg) {
                const NSUInteger index = mode*16 + (nsg - 1)*2;
                run_cases(device, queue, pipelines[index], pipelines[index + 1], nsg,
                          mode ? "safe" : "default", &total_rows, &nan_payload_differences);
            }
        }
        printf("PASS Q8 reduction: %d cases, %zu output rows; non-NaN bits and NaN classification match; "
               "%zu NaN payload differences ignored\n", 2*8*kSamples, total_rows, nan_payload_differences);
    }
    return 0;
}
