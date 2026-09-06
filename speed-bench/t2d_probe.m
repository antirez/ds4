/* Host driver for speed-bench/t2d_probe.metal.
 *
 * Validates the three operand-type variants (float/float, float/half,
 * half/half) against an exact double-precision CPU reference at the
 * gathered-QSA attention geometry, then times the float/half and
 * float/float variants at a production-shaped dispatch (8192 x 2
 * threadgroups, broadcast inputs, no gather) to estimate the raw
 * chunked-matmul rate.  Prints the device threadgroup-memory limit.
 *
 * Build:  cc -O2 -std=c99 -framework Foundation -framework Metal \
 *             -o speed-bench/t2d_probe speed-bench/t2d_probe.m
 * Run from the repo root.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { M = 16, D = 256, BK = 64 };

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static uint32_t lcg = 0x811c9dc5u;
static float lcg_float(float scale) {
    lcg = lcg * 1664525u + 1013904223u;
    return ((float)(lcg >> 8) / (float)(1u << 24) - 0.5f) * scale;
}

typedef struct {
    double s_max_abs, s_rms, o_max_abs, o_rms;
    double s_ref_scale, o_ref_scale;
} verdict;

/* fp16 rounding on host (round-to-nearest-even via _Float16). */
static float f16_round(float v) { return (float)(_Float16)v; }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    @autoreleasepool {
        uint32_t groups_x = 8192;
        if (argc > 1) groups_x = (uint32_t)strtoul(argv[1], NULL, 0);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("device %s\n", device.name.UTF8String);
        printf("maxThreadgroupMemoryLength %llu\n",
               (unsigned long long)device.maxThreadgroupMemoryLength);

        NSError *error = nil;
        NSString *src = [NSString stringWithContentsOfFile:
            @"speed-bench/t2d_probe.metal"
            encoding:NSUTF8StringEncoding error:nil];
        if (!src) { fprintf(stderr, "read source failed\n"); return 1; }
        id<MTLLibrary> lib = [device newLibraryWithSource:src
            options:nil error:&error];
        if (!lib) {
            fprintf(stderr, "compile failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        const int marker = argc > 2 && strcmp(argv[2], "marker") == 0;
        /* Host data with production-class magnitudes (Q/K dot products
         * land around +-20 like the real scores). */
        float *q = malloc((size_t)M * D * sizeof(float));
        float *kc = malloc((size_t)BK * D * sizeof(float));
        float *p = malloc((size_t)M * BK * sizeof(float));
        float *vc = malloc((size_t)BK * D * sizeof(float));
        for (int i = 0; i < M * D; i++) q[i] = lcg_float(0.5f);
        for (int i = 0; i < BK * D; i++) kc[i] = lcg_float(0.5f);
        for (int i = 0; i < M * BK; i++) p[i] = fabsf(lcg_float(1.0f)) * 0.02f;
        for (int i = 0; i < BK * D; i++) vc[i] = lcg_float(0.5f);
        if (marker) {
            /* S[h][t] must equal (h+1) + 100*(t+1); P/V encode the
             * register/coord map through the PV epilogue marker. */
            for (int h = 0; h < M; h++)
                for (int d = 0; d < D; d++)
                    q[h * D + d] = d == 0 ? (float)(h + 1) : (d == 1 ? 1.0f : 0.0f);
            for (int t = 0; t < BK; t++)
                for (int d = 0; d < D; d++)
                    kc[t * D + d] = d == 0 ? 1.0f : (d == 1 ? 100.0f * (t + 1) : 0.0f);
            for (int i = 0; i < M * BK; i++) p[i] = 1.0f;
            for (int i = 0; i < BK * D; i++) vc[i] = 1.0f;
        }

        /* Exact CPU references in double. */
        double *s_ref = malloc((size_t)M * BK * sizeof(double));
        double *o_ref = malloc((size_t)M * D * sizeof(double));
        for (int m = 0; m < M; m++)
            for (int t = 0; t < BK; t++) {
                double acc = 0.0;
                for (int d = 0; d < D; d++)
                    acc += (double)q[m * D + d] * kc[t * D + d];
                s_ref[m * BK + t] = acc;
            }
        for (int m = 0; m < M; m++)
            for (int d = 0; d < D; d++) {
                double acc = 0.0;
                for (int t = 0; t < BK; t++)
                    acc += (double)p[m * BK + t] * vc[t * D + d];
                o_ref[m * D + d] = acc;
            }

        id<MTLBuffer> qb = [device newBufferWithBytes:q
            length:(size_t)M * D * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> kb = [device newBufferWithBytes:kc
            length:(size_t)BK * D * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> pb = [device newBufferWithBytes:p
            length:(size_t)M * BK * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> vb = [device newBufferWithBytes:vc
            length:(size_t)BK * D * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> sob = [device newBufferWithLength:(size_t)M * BK * 4
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> oob = [device newBufferWithLength:(size_t)M * D * 4
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> dbgb = [device newBufferWithLength:16
            options:MTLResourceStorageModeShared];

        const char *names[2] = {"t2d_qkpv_ff", "t2d_qkpv_hh"};
        const size_t ff_bytes =
            (size_t)M * 32 * 4 + (size_t)BK * 32 * 4 +
            (size_t)M * 32 * 4 + (size_t)128 * 32 * 4 +
            (size_t)M * BK * 4;
        const size_t hh_bytes =
            (size_t)M * 32 * 2 + (size_t)BK * 32 * 2 +
            (size_t)M * 32 * 2 + (size_t)128 * 32 * 2 +
            (size_t)M * BK * 4;
        printf("threadgroup bytes: float %zu  half %zu\n",
               ff_bytes, hh_bytes);

        for (int v = 0; v < 2; v++) {
            id<MTLFunction> fn = [lib newFunctionWithName:
                [NSString stringWithUTF8String:names[v]]];
            if (!fn) { fprintf(stderr, "missing %s\n", names[v]); return 1; }
            id<MTLComputePipelineState> pso =
                [device newComputePipelineStateWithFunction:fn error:&error];
            if (!pso) {
                fprintf(stderr, "pso %s failed: %s\n", names[v],
                        error.localizedDescription.UTF8String);
                return 1;
            }
            const size_t tg = v == 0 ? ff_bytes : hh_bytes;
            memset(sob.contents, 0, (size_t)M * BK * 4);
            memset(oob.contents, 0, (size_t)M * D * 4);
            ((float *)dbgb.contents)[0] = marker ? 1.0f : 0.0f;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:qb offset:0 atIndex:0];
            [enc setBuffer:kb offset:0 atIndex:1];
            [enc setBuffer:pb offset:0 atIndex:2];
            [enc setBuffer:vb offset:0 atIndex:3];
            [enc setBuffer:sob offset:0 atIndex:4];
            [enc setBuffer:oob offset:0 atIndex:5];
            [enc setBuffer:dbgb offset:0 atIndex:6];
            [enc setThreadgroupMemoryLength:tg atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                fprintf(stderr, "%s dispatch failed: %s\n", names[v],
                        cb.error.localizedDescription.UTF8String);
                return 1;
            }
            const float *sg = sob.contents, *og = oob.contents;
            if (marker && v == 0) {
                printf("QK store map (address i -> (h,t) via value):\n");
                for (int i = 0; i < 32; i++) {
                    const int v0 = (int)(sg[i] + 0.5f);
                    printf("  addr %3d: val %6d -> h=%d t=%d\n",
                           i, v0, v0 % 100 - 1, v0 / 100 - 1);
                }
                printf("PV coord map (linear reg i -> (h,d)):\n");
                for (int i = 0; i < M * D; i++) {
                    const int v0 = (int)og[i];
                    if (v0 >= 1000 && i < 96)
                        printf("  oo[%3d] (h=%d d=%d): reg %d\n",
                               i, i / D, i % D, v0 - 1000);
                    if (i >= 96) break;
                }
                continue;
            }
            if (v == 0) {
                printf("sample so:");
                for (int i = 0; i < 6; i++)
                    printf(" %.4f/%.4f", sg[i], s_ref[i]);
                printf("\n        oo:");
                for (int i = 0; i < 4; i++)
                    printf(" %.4f/%.4f", og[i], o_ref[i]);
                printf("\n");
            }
            double s_max = 0, s_rms = 0, o_max = 0, o_rms = 0;
            double s_scale = 0, o_scale = 0;
            for (int i = 0; i < M * BK; i++) {
                const double e = fabs(sg[i] - s_ref[i]);
                if (e > s_max) s_max = e;
                s_rms += e * e;
                s_scale += fabs(s_ref[i]);
            }
            for (int i = 0; i < M * D; i++) {
                const double e = fabs(og[i] - o_ref[i]);
                if (e > o_max) o_max = e;
                o_rms += e * e;
                o_scale += fabs(o_ref[i]);
            }
            s_rms = sqrt(s_rms / (M * BK));
            o_rms = sqrt(o_rms / (M * D));
            s_scale /= M * BK;
            o_scale /= M * D;
            printf("%-14s QK max_abs %.3e rms %.3e (ref mean %.3f rel %.2e)"
                   "  PV max_abs %.3e rms %.3e (ref mean %.4f rel %.2e)\n",
                   names[v], s_max, s_rms, s_scale, s_rms / s_scale,
                   o_max, o_rms, o_scale, o_rms / o_scale);
        }

        /* Timing at the production-shaped grid: every threadgroup runs
         * one QK + one PV sequence; per-layer work is 32 gathered
         * tokens-tiles of 64 per (query, kv_head) x 16384 groups, of
         * which this runs 1/32nd. */
        static const int skips[3] = {0, 1, 2}; /* both, QK-only, PV-only */
        static const char *skipn[3] = {"qk+pv", "qk   ", "pv   "};
        for (int v = 0; v < 2; v++) {
          id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:
              [lib newFunctionWithName:
                [NSString stringWithUTF8String:names[v]]] error:nil];
          const size_t tg = v == 0 ? ff_bytes : hh_bytes;
          for (int sk = 0; sk < 3; sk++) {
            ((float *)dbgb.contents)[1] = (float)skips[sk];
            double samples[5];
            for (int iter = 0; iter < 7; iter++) {
                const double t0 = now_ms();
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:qb offset:0 atIndex:0];
                [enc setBuffer:kb offset:0 atIndex:1];
                [enc setBuffer:pb offset:0 atIndex:2];
                [enc setBuffer:vb offset:0 atIndex:3];
                [enc setBuffer:sob offset:0 atIndex:4];
                [enc setBuffer:oob offset:0 atIndex:5];
                [enc setBuffer:dbgb offset:0 atIndex:6];
                [enc setThreadgroupMemoryLength:tg atIndex:0];
                [enc dispatchThreadgroups:MTLSizeMake(groups_x, 2, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (iter >= 2) samples[iter - 2] = now_ms() - t0;
            }
            for (int i = 1; i < 5; i++) {
                const double key = samples[i];
                int j = i - 1;
                while (j >= 0 && samples[j] > key) {
                    samples[j + 1] = samples[j];
                    j--;
                }
                samples[j + 1] = key;
            }
            /* One iteration = 1/32 of one layer's QK+PV at 8192 queries
             * (2048 selected tokens = 32 tiles of 64). */
            printf("%-14s %s %8.2f ms for 1/32 layer-equiv"
                   " -> est %.1f ms/layer\n",
                   names[v], skipn[sk], samples[2], samples[2] * 32.0);
          }
        }
    }
    return 0;
}
