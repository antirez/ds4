/* Model-free large-slab residency reproducer; see speed-bench/README.md. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static uint64_t number(const char *text) {
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    return text[0] && text[0] != '-' && end && !*end ? value : 0;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s none|queue|toggle SLABS SLAB_MIB FILLED_MIB ITERATIONS SELECTED\n",
                argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    const uint64_t count64 = number(argv[2]), mib = number(argv[3]);
    const uint64_t filled_mib = number(argv[4]);
    const uint64_t iterations64 = number(argv[5]), selected64 = number(argv[6]);
    if ((strcmp(mode, "none") && strcmp(mode, "queue") && strcmp(mode, "toggle")) ||
        !count64 || count64 > UINT32_MAX || !mib || mib > UINT64_MAX / 1048576 ||
        filled_mib < 4 || filled_mib > mib || !iterations64 || iterations64 > UINT32_MAX ||
        !selected64 || selected64 > count64 || selected64 > UINT32_MAX / 1024) return 2;
    const uint32_t count = (uint32_t)count64, iterations = (uint32_t)iterations64;
    const uint32_t selected = (uint32_t)selected64;
    const uint64_t bytes = mib * 1048576, filled = filled_mib * 1048576;
    const BOOL alternate = getenv("DS4_SLAB_BENCH_ALTERNATE_SMALL") != NULL;

    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            id<MTLCommandQueue> queue = [dev newCommandQueue];
            if (!dev || !queue || bytes > dev.maxBufferLength) return 1;
            fprintf(stderr, "device=%s mode=%s slabs=%u slab_MiB=%llu filled_MiB=%llu "
                    "selected=%u alternate_small=%d\n", dev.name.UTF8String, mode,
                    count, mib, filled_mib, selected, alternate);
            NSString *source = @"#include <metal_stdlib>\n"
                "using namespace metal;\n"
                "kernel void probe(device const ulong *a [[buffer(0)]], "
                "device uint *out [[buffer(1)]], uint i [[thread_position_in_grid]]) { "
                "device const uint *b = reinterpret_cast<device const uint *>(a[i/1024]); "
                "out[i] = b[(i%1024)*1024]; }";
            NSError *error = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:source options:nil error:&error];
            if (!lib) {
                fprintf(stderr, "Metal library: %s\n", error.description.UTF8String);
                return 1;
            }
            id<MTLComputePipelineState> pipeline = [dev
                newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe"] error:&error];
            if (!pipeline) return 1;

            NSMutableArray<id<MTLBuffer>> *pool = [NSMutableArray array];
            for (uint32_t i = 0; i < count; i++) {
                id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!b) return 1;
                memset(b.contents, i % 127 + 1, filled);
                /* Match the Q2 expert-slot lock granularity. No model file IO. */
                const uint64_t slot_bytes = 9961472;
                for (uint64_t off = 0; off < filled; off += slot_bytes) {
                    const uint64_t n = filled - off < slot_bytes ? filled - off : slot_bytes;
                    if (mlock((char *)b.contents + off, n)) {
                        perror("mlock");
                        return 1;
                    }
                }
                [pool addObject:b];
            }
            id<MTLBuffer> small = [dev newBufferWithLength:4194304 options:MTLResourceStorageModeShared];
            id<MTLBuffer> addresses = [dev newBufferWithLength:selected * sizeof(uint64_t)
                                                     options:MTLResourceStorageModeShared];
            id<MTLBuffer> out = [dev newBufferWithLength:selected * 1024 * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
            if (!small || !addresses || !out) return 1;
            memset(small.contents, 1, 4194304);
            id<MTLResidencySet> set = nil;
            BOOL queued = NO;
            if (strcmp(mode, "none")) {
                MTLResidencySetDescriptor *desc = [MTLResidencySetDescriptor new];
                desc.initialCapacity = count;
                set = [dev newResidencySetWithDescriptor:desc error:&error];
                if (!set) return 1;
                for (id<MTLBuffer> b in pool) [set addAllocation:b];
                [set commit];
                if (!strcmp(mode, "queue")) {
                    [queue addResidencySet:set];
                    queued = YES;
                }
            }
            printf("phase,iteration,kind,wall_ms,gpu_ms,driver_ms,checksum\n");
            const uint32_t phases = !strcmp(mode, "toggle") ? 4 : 1;
            for (uint32_t phase = 0; phase < phases; phase++) {
                if (phases > 1) {
                    if (phase % 2) {
                        [queue addResidencySet:set];
                        queued = YES;
                    } else if (queued) {
                        [queue removeResidencySet:set];
                        queued = NO;
                    }
                }
                for (uint32_t it = 0; it < iterations; it++) {
                    @autoreleasepool {
                        const BOOL use_small = alternate && (it % 2);
                        const double begin = now_ms();
                        id<MTLCommandBuffer> cb = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        uint64_t expected = 0;
                        for (uint32_t j = 0; j < selected; j++) {
                            const uint32_t k = ((uint64_t)it * selected + j) % count;
                            id<MTLBuffer> b = use_small ? small : pool[k];
                            ((uint64_t *)addresses.contents)[j] = b.gpuAddress;
                            [enc useResource:b usage:MTLResourceUsageRead];
                            expected += (uint64_t)(0x01010101u * (use_small ? 1 : k % 127 + 1)) * 1024;
                        }
                        [enc setComputePipelineState:pipeline];
                        [enc setBuffer:addresses offset:0 atIndex:0];
                        [enc setBuffer:out offset:0 atIndex:1];
                        [enc dispatchThreadgroups:MTLSizeMake(selected * 4, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        [enc endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        const double end = now_ms();
                        if (cb.status != MTLCommandBufferStatusCompleted) {
                            fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String);
                            return 1;
                        }
                        uint64_t checksum = 0;
                        for (uint32_t j = 0; j < selected * 1024; j++) checksum += ((uint32_t *)out.contents)[j];
                        if (checksum != expected) {
                            fprintf(stderr, "checksum mismatch\n");
                            return 1;
                        }
                        printf("%u,%u,%s,%.6f,%.6f,%.6f,%llu\n", phase, it,
                            use_small ? "small" : "slabs", end - begin,
                            (cb.GPUEndTime - cb.GPUStartTime) * 1000,
                            (cb.kernelEndTime - cb.kernelStartTime) * 1000, checksum);
                    }
                }
                fflush(stdout);
            }
            if (queued) [queue removeResidencySet:set];
            for (id<MTLBuffer> b in pool) munlock(b.contents, filled);
        }
    } else {
        fprintf(stderr, "requires macOS 15 or later\n");
        return 1;
    }
    return 0;
}
