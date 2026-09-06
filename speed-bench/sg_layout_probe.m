#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static void print_mapping(const char *label, const float *base,
                          uint stride, uint count) {
    printf("%s mapping:\n", label);
    for (uint lane = 0; lane < 32; lane++) {
        printf("  lane %2u:", lane);
        for (uint e = 0; e < count; e++) {
            const float v = base[lane * stride + e];
            if (v < 0.0f || v > 63.0f) {
                printf(" elem%u=INVALID(%.1f)", e, v);
            } else {
                printf(" elem%u=(%u,%u)", e, (uint)v / 8u, (uint)v % 8u);
            }
        }
        printf("\n");
    }
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSString *src = [NSString stringWithContentsOfFile:
            @"speed-bench/sg_layout_probe.metal"
            encoding:NSUTF8StringEncoding error:nil];
        if (!src) { fprintf(stderr, "read source failed\n"); return 1; }
        id<MTLLibrary> lib = [device newLibraryWithSource:src
            options:nil error:&error];
        if (!lib) {
            fprintf(stderr, "compile failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"sg_layout_probe"];
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) {
            fprintf(stderr, "pso failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLBuffer> out = [device newBufferWithLength:8192
            options:MTLResourceStorageModeShared];
        memset(out.contents, 0, 8192);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setThreadgroupMemoryLength:1024 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const float *v = out.contents;
        print_mapping("float8x8 accumulator", v + 1, 3, 2);
        print_mapping("half8x8 operand", v + 97, 8, 8);
    }
    return 0;
}
