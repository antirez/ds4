/* Whole-layer fusion feasibility prototype for the Qwen HC-read chain.
 * Compares the production three-dispatch structure (norm, down+SiLU,
 * up+mix) against one persistent kernel with a software grid barrier, at
 * the real verify shapes (hidden 2560, 4 streams, lowrank 320).  Both GPU
 * paths are validated against a CPU reference before any timing is
 * reported.  Standalone: builds its own Metal library from an embedded
 * source string; the production tree is untouched. */
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HIDDEN = 2560, STREAMS = 4, LOWRANK = 320, ROWS_MAX = 8, PGROUPS = 40 };

static const char *proto_source =
#include "hc_chain_proto_kernels.h"
;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;

static uint32_t rng_state = 0x243f6a88u;
static float frand(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return ((float)(int32_t)(x >> 8) / 8388608.0f);
}

static float b16(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

int main(int argc, char **argv) {
    int rows = 4, iterations = 200, groups = PGROUPS;
    if (argc >= 2) rows = atoi(argv[1]);
    if (argc >= 3) iterations = atoi(argv[2]);
    if (argc >= 4) groups = atoi(argv[3]);
    if (rows < 1 || rows > ROWS_MAX || iterations < 1 || iterations > 100000 ||
        groups < 1 || groups > 640 || (320 % groups) != 0 || (2560 % groups) != 0) {
        fprintf(stderr, "usage: %s [rows 1-8] [iterations] [groups; "
                "must divide 320 and 2560]\n", argv[0]);
        return 2;
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "no Metal device\n"); return 1; }
        NSError *error = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:
            [NSString stringWithUTF8String:proto_source] options:nil error:&error];
        if (!lib) {
            fprintf(stderr, "compile failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLComputePipelineState> pn = [device newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"k_norm"] error:&error];
        id<MTLComputePipelineState> pd = [device newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"k_down_silu"] error:&error];
        id<MTLComputePipelineState> pu = [device newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"k_up_mix"] error:&error];
        id<MTLComputePipelineState> pp = [device newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"hc_read_persistent"] error:&error];
        if (!pn || !pd || !pu || !pp) {
            fprintf(stderr, "pipeline failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }
        const uint32_t hc_dim = STREAMS * HIDDEN;
        const size_t streams_bytes = (size_t)rows * hc_dim * sizeof(float);
        const size_t normw_bytes = (size_t)STREAMS * HIDDEN * sizeof(uint16_t);
        const size_t down_bytes = (size_t)LOWRANK * (hc_dim / 32) * sizeof(block_q8_0);
        const size_t up_bytes = (size_t)hc_dim * (LOWRANK / 32) * sizeof(block_q8_0);
        id<MTLBuffer> streams_b = [device newBufferWithLength:streams_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> normw_b = [device newBufferWithLength:normw_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> downw_b = [device newBufferWithLength:down_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> upw_b = [device newBufferWithLength:up_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_ref = [device newBufferWithLength:(size_t)rows * HIDDEN * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_per = [device newBufferWithLength:(size_t)rows * HIDDEN * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> norm_scr = [device newBufferWithLength:streams_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> low_scr = [device newBufferWithLength:(size_t)rows * LOWRANK * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> barrier_b = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        if (!streams_b || !normw_b || !downw_b || !upw_b || !out_ref ||
            !out_per || !norm_scr || !low_scr || !barrier_b) return 1;
        /* deterministic fill */
        float *sp = (float *)streams_b.contents;
        for (uint32_t i = 0; i < (uint32_t)(streams_bytes / 4); i++)
            sp[i] = frand() * 0.5f;
        uint16_t *np = (uint16_t *)normw_b.contents;
        for (uint32_t i = 0; i < (uint32_t)(normw_bytes / 2); i++) {
            float w = 0.75f + 0.5f * frand();
            uint32_t bits;
            memcpy(&bits, &w, 4);
            np[i] = (uint16_t)(bits >> 16);
        }
        block_q8_0 *dp = (block_q8_0 *)downw_b.contents;
        for (size_t i = 0; i < down_bytes / sizeof(block_q8_0); i++) {
            float vals[32], maxv = 0.0f;
            for (int j = 0; j < 32; j++) {
                vals[j] = frand() * 0.05f;
                if (fabsf(vals[j]) > maxv) maxv = fabsf(vals[j]);
            }
            float d = fmaxf(maxv / 127.0f, 1e-8f);
            uint32_t bits;
            memcpy(&bits, &d, 4);
            dp[i].d = (uint16_t)(bits >> 16);
            for (int j = 0; j < 32; j++) {
                long q = lrintf(vals[j] / b16(dp[i].d));
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                dp[i].qs[j] = (int8_t)q;
            }
        }
        block_q8_0 *up = (block_q8_0 *)upw_b.contents;
        for (size_t i = 0; i < up_bytes / sizeof(block_q8_0); i++) {
            float vals[32], maxv = 0.0f;
            for (int j = 0; j < 32; j++) {
                vals[j] = frand() * 0.08f;
                if (fabsf(vals[j]) > maxv) maxv = fabsf(vals[j]);
            }
            float d = fmaxf(maxv / 127.0f, 1e-8f);
            uint32_t bits;
            memcpy(&bits, &d, 4);
            up[i].d = (uint16_t)(bits >> 16);
            for (int j = 0; j < 32; j++) {
                long q = lrintf(vals[j] / b16(up[i].d));
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                up[i].qs[j] = (int8_t)q;
            }
        }
        struct { uint32_t rows, hidden, streams, lowrank, groups; } args =
            { (uint32_t)rows, HIDDEN, STREAMS, LOWRANK, (uint32_t)groups };
        id<MTLCommandQueue> queue = device.newCommandQueue;

        void (^run_ref)(void) = ^{
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pn];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:streams_b offset:0 atIndex:1];
            [enc setBuffer:normw_b offset:0 atIndex:2];
            [enc setBuffer:norm_scr offset:0 atIndex:7];
            [enc setThreadgroupMemoryLength:64 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows * STREAMS, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc setComputePipelineState:pd];
            [enc setBuffer:downw_b offset:0 atIndex:4];
            [enc setBuffer:low_scr offset:0 atIndex:8];
            [enc dispatchThreadgroups:MTLSizeMake((LOWRANK + 3) / 4, rows, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc setComputePipelineState:pu];
            [enc setBuffer:upw_b offset:0 atIndex:5];
            [enc setBuffer:out_ref offset:0 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(HIDDEN, rows, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        };
        void (^run_per)(void) = ^{
            memset(barrier_b.contents, 0, 4);
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pp];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:streams_b offset:0 atIndex:1];
            [enc setBuffer:normw_b offset:0 atIndex:2];
            [enc setBuffer:downw_b offset:0 atIndex:4];
            [enc setBuffer:upw_b offset:0 atIndex:5];
            [enc setBuffer:out_per offset:0 atIndex:6];
            [enc setBuffer:norm_scr offset:0 atIndex:7];
            [enc setBuffer:low_scr offset:0 atIndex:8];
            [enc setBuffer:barrier_b offset:0 atIndex:9];
            [enc setThreadgroupMemoryLength:128 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        };

        /* CPU reference of the whole chain */
        float *cpu = malloc((size_t)rows * HIDDEN * sizeof(float));
        float *norm = malloc(streams_bytes);
        float *lowv = malloc((size_t)rows * LOWRANK * sizeof(float));
        for (int t = 0; t < rows; t++)
            for (uint32_t s = 0; s < STREAMS; s++) {
                double ss = 0.0;
                for (uint32_t h = 0; h < HIDDEN; h++) {
                    const float v = sp[(size_t)t * hc_dim + s * HIDDEN + h];
                    ss += (double)v * v;
                }
                const float scale = 1.0f / sqrtf((float)(ss / HIDDEN) + 1e-6f);
                for (uint32_t h = 0; h < HIDDEN; h++)
                    norm[(size_t)t * hc_dim + s * HIDDEN + h] =
                        sp[(size_t)t * hc_dim + s * HIDDEN + h] * scale *
                        b16(np[s * HIDDEN + h]);
            }
        for (int t = 0; t < rows; t++)
            for (uint32_t o = 0; o < LOWRANK; o++) {
                double dot = 0.0;
                const block_q8_0 *rowp = dp + (size_t)o * (hc_dim / 32);
                for (uint32_t blk = 0; blk < hc_dim / 32; blk++) {
                    double dq = 0.0;
                    for (int j = 0; j < 32; j++)
                        dq += (double)rowp[blk].qs[j] *
                              norm[(size_t)t * hc_dim + blk * 32 + j];
                    dot += dq * b16(rowp[blk].d);
                }
                const float v = (float)dot;
                lowv[(size_t)t * LOWRANK + o] = v / (1.0f + expf(-v));
            }
        for (int t = 0; t < rows; t++)
            for (uint32_t h = 0; h < HIDDEN; h++) {
                double value = 0.0;
                for (uint32_t s = 0; s < STREAMS; s++) {
                    double dot = 0.0;
                    const block_q8_0 *rowp =
                        up + ((size_t)s * HIDDEN + h) * (LOWRANK / 32);
                    for (uint32_t blk = 0; blk < LOWRANK / 32; blk++) {
                        double dq = 0.0;
                        for (int j = 0; j < 32; j++)
                            dq += (double)rowp[blk].qs[j] *
                                  lowv[(size_t)t * LOWRANK + blk * 32 + j];
                        dot += dq * b16(rowp[blk].d);
                    }
                    value += (double)norm[(size_t)t * hc_dim + s * HIDDEN + h] *
                             (1.0 / (1.0 + exp(-(double)dot)));
                }
                cpu[(size_t)t * HIDDEN + h] = (float)(value / STREAMS);
            }

        run_ref();
        run_per();
        const float *rf = (const float *)out_ref.contents;
        const float *pf = (const float *)out_per.contents;
        double maxrel_ref = 0.0, maxrel_per = 0.0;
        for (uint32_t i = 0; i < (uint32_t)rows * HIDDEN; i++) {
            const double denom = fmax(fabs(cpu[i]), 1e-3);
            const double er = fabs(rf[i] - cpu[i]) / denom;
            const double ep = fabs(pf[i] - cpu[i]) / denom;
            if (er > maxrel_ref) maxrel_ref = er;
            if (ep > maxrel_per) maxrel_per = ep;
        }
        printf("validation rows=%d: ref-vs-cpu=%.2e persistent-vs-cpu=%.2e\n",
               rows, maxrel_ref, maxrel_per);
        if (!(maxrel_ref < 3e-3 && maxrel_per < 3e-3)) {
            fprintf(stderr, "PROTOTYPE INVALID: tolerances exceeded\n");
            return 1;
        }
        double t_ref = 0.0, t_per = 0.0;
        for (int i = 0; i < iterations; i++) {
            double t0 = now_ms(); run_ref(); t_ref += now_ms() - t0;
            double t1 = now_ms(); run_per(); t_per += now_ms() - t1;
        }
        printf("hc-chain rows=%d iters=%d: ref(3-dispatch)=%.4f ms  "
               "persistent=%d-group=%.4f ms  speedup=%.3fx\n",
               rows, iterations, t_ref / iterations, groups,
               t_per / iterations, t_ref / t_per);
        free(cpu); free(norm); free(lowv);
    }
    return 0;
}
