/* MXFP4 prefill staging oracle: production stage16 versus two unchanged
 * stage8 calls. Covers every exponent byte and nibble code at every block
 * alignment, for half/float storage and generic/specialized dispatch.
 * Build: clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
 *   tests/test_metal_qwen4_mxfp4_stage.m -o /tmp/test_metal_qwen4_mxfp4_stage
 * Run from the repository root. --source FILE selects a candidate Metal
 * source; --compile-only creates pipelines without GPU dispatches.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const NSUInteger kGuard = 256;
static const unsigned kBlocks = 256 * 16;
static uint64_t compared, nan_payload_differences;

static void fail(NSString *s) {
    fprintf(stderr, "FAIL MXFP4 staging: %s\n", s.UTF8String); exit(1);
}
static NSString *read_source(NSString *path) {
    NSError *error = nil;
    NSString *s = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (!s) fail([NSString stringWithFormat:@"read %@: %@", path, error]);
    return s;
}
static NSString *between(NSString *s, NSString *begin, NSString *end) {
    if ([s componentsSeparatedByString:begin].count != 2 ||
        [s componentsSeparatedByString:end].count != 2) fail(@"source anchors missing/nonunique");
    NSRange a = [s rangeOfString:begin], b = [s rangeOfString:end];
    if (b.location <= a.location) fail(@"invalid source range");
    return [s substringWithRange:NSMakeRange(a.location, b.location - a.location)];
}
static NSString *kernel_name(bool half, bool specialized, bool candidate) {
    return [NSString stringWithFormat:@"stage_%@_%u_%u", half ? @"half" : @"float", specialized, candidate];
}
static NSString *production_source(NSString *repo, NSString *path) {
    NSString *q = read_source(path), *m = read_source([repo stringByAppendingPathComponent:@"metal/moe.metal"]);
    NSMutableString *s = [NSMutableString stringWithString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
    [s appendString:between(m, @"static constant float ds4_metal_mxfp4_values[16]", @"// BEGIN GENERATED MXFP4 HALF LUT")];
    [s appendString:between(m, @"static constant uchar ds4_metal_ksigns_iq2xs[128]", @"#define kmask_iq2xs")];
    [s appendString:between(q, @"/* dequantize 8 consecutive values", @"/* Raw words of one 32-block")];
    for (unsigned h = 0; h < 2; h++) for (unsigned sp = 0; sp < 2; sp++) for (unsigned arm = 0; arm < 2; arm++) {
        NSString *type = h ? @"half" : @"float";
        [s appendFormat:@"\nkernel void %@(device const char *input [[buffer(0)]], "
            "device %@ *output [[buffer(1)]], constant uint &runtime_type [[buffer(2)]], "
            "uint gid [[thread_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {\n"
            "  threadgroup %@ scratch[128 * 16];\n"
            "  threadgroup %@ *dst = scratch + uint(tid) * 16;\n"
            "  const uint b = gid / 2, q0 = (gid & 1u) * 2;\n"
            "  const uint type = %@;\n", kernel_name(h, sp, arm), type, type, type, sp ? @"39u" : @"runtime_type"];
        if (arm) [s appendString:@"  qwen4_mm_stage16(input + 1, b, q0, type, dst);\n"];
        else [s appendString:@"  qwen4_mm_stage8(input + 1, b, q0, type, dst);\n"
                               "  qwen4_mm_stage8(input + 1, b, q0 + 1, type, dst + 8);\n"];
        [s appendString:@"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                         "  for (uint i = 0; i < 16; i++) output[gid * 16 + i] = dst[i];\n}\n"];
    }
    return s;
}
static id<MTLBuffer> buffer(id<MTLDevice> device, size_t bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:bytes + 2 * kGuard options:MTLResourceStorageModeShared];
    if (!b) fail(@"buffer allocation");
    memset(b.contents, 0xa5, b.length); return b;
}
static void *payload(id<MTLBuffer> b) { return (char *)b.contents + kGuard; }
static void check_guard(id<MTLBuffer> b) {
    const uint8_t *p = b.contents;
    for (NSUInteger i = 0; i < kGuard; i++)
        if (p[i] != 0xa5 || p[b.length - 1 - i] != 0xa5) fail(@"output guard overwritten");
}
static NSArray<id<MTLComputePipelineState>> *pipelines(id<MTLDevice> device, NSString *source, bool safe) {
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
    id<MTLLibrary> lib = [device newLibraryWithSource:source options:options error:&error];
    if (!lib) fail(error.description);
    NSMutableArray *ps = [NSMutableArray new];
    for (unsigned h = 0; h < 2; h++) for (unsigned sp = 0; sp < 2; sp++) for (unsigned arm = 0; arm < 2; arm++) {
        id<MTLFunction> f = [lib newFunctionWithName:kernel_name(h, sp, arm)];
        if (!f) fail(@"kernel unavailable");
        id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:f error:&error];
        if (!p) fail(error.description);
        if (p.maxTotalThreadsPerThreadgroup < 128) fail(@"128-thread staging unavailable");
        [ps addObject:p];
    }
    return ps;
}
static void dispatch_pair(id<MTLCommandQueue> queue, NSArray *ps, unsigned first,
                          id<MTLBuffer> input, id<MTLBuffer> ref, id<MTLBuffer> got) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    if (!cb || !en) fail(@"command allocation");
    const uint32_t type = 39;
    for (unsigned arm = 0; arm < 2; arm++) {
        [en setComputePipelineState:ps[first + arm]];
        [en setBuffer:input offset:kGuard atIndex:0];
        [en setBuffer:arm ? got : ref offset:kGuard atIndex:1];
        [en setBytes:&type length:sizeof(type) atIndex:2];
        [en dispatchThreadgroups:MTLSizeMake(kBlocks * 2 / 128, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) fail(cb.error.description);
}
static void compare(id<MTLBuffer> ref, id<MTLBuffer> got, bool half) {
    for (size_t i = 0; i < (size_t)kBlocks * 32; i++) {
        const uint32_t r = half ? ((uint16_t *)payload(ref))[i] : ((uint32_t *)payload(ref))[i];
        const uint32_t g = half ? ((uint16_t *)payload(got))[i] : ((uint32_t *)payload(got))[i];
        const uint32_t signless = half ? 0x7fffu : 0x7fffffffu, inf = half ? 0x7c00u : 0x7f800000u;
        if (r == (half ? 0xa5a5u : 0xa5a5a5a5u) || g == (half ? 0xa5a5u : 0xa5a5a5a5u)) fail(@"output untouched");
        if ((r & signless) > inf && (g & signless) > inf) nan_payload_differences += r != g;
        else if (r != g) fail([NSString stringWithFormat:@"%@ block=%zu value=%zu: %08x != %08x",
            half ? @"half" : @"float", i / 32, i % 32, r, g]);
        compared++;
    }
    check_guard(ref); check_guard(got);
}
int main(int argc, const char **argv) { @autoreleasepool {
    NSString *repo = [[NSFileManager defaultManager] currentDirectoryPath], *path = nil;
    bool compileOnly = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--source") && i + 1 < argc) path = @(argv[++i]);
        else if (!strcmp(argv[i], "--repo") && i + 1 < argc) repo = @(argv[++i]);
        else if (!strcmp(argv[i], "--compile-only")) compileOnly = true;
        else fail(@"usage: [--repo PATH] [--source FILE] [--compile-only]");
    }
    path = path ?: [repo stringByAppendingPathComponent:@"metal/qwen4.metal"];
    NSString *source = production_source(repo, path);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) fail(@"Metal device unavailable");
    for (unsigned safe = 0; safe < 2; safe++) @autoreleasepool {
        NSArray *ps = pipelines(device, source, safe);
        if (compileOnly) continue;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) fail(@"command queue unavailable");
        id<MTLBuffer> input = buffer(device, (size_t)kBlocks * 17 + 1);
        uint8_t *p = (uint8_t *)payload(input) + 1;
        /* The odd base and 17-byte stride exercise all sixteen alignments.
         * Each exponent sees every low/high code at every payload position. */
        for (unsigned b = 0; b < kBlocks; b++) {
            const unsigned pattern = b % 16;
            p[b * 17] = b / 16;
            for (unsigned j = 0; j < 16; j++)
                p[b * 17 + 1 + j] = ((pattern + j) & 15u) | (((pattern * 7u + j * 3u) & 15u) << 4);
        }
        NSData *before = [NSData dataWithBytes:input.contents length:input.length];
        for (unsigned h = 0; h < 2; h++) for (unsigned sp = 0; sp < 2; sp++) {
            const size_t bytes = (size_t)kBlocks * 32 * (h ? 2 : 4);
            id<MTLBuffer> ref = buffer(device, bytes), got = buffer(device, bytes);
            dispatch_pair(queue, ps, h * 4 + sp * 2, input, ref, got); compare(ref, got, h);
            if (memcmp(input.contents, before.bytes, input.length)) fail(@"input or its guards modified");
            printf("PASS MXFP4 stage8/stage16 safe=%u storage=%s specialized=%u: %u values\n",
                safe, h ? "half" : "float", sp, kBlocks * 32);
        }
    }
    printf("PASS MXFP4 staging: %llu comparisons, %llu NaN payload differences%s\n",
        (unsigned long long)compared, (unsigned long long)nan_payload_differences,
        compileOnly ? " (compile only)" : "");
    return 0;
} }
