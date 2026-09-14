// SPDX-License-Identifier: MIT
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ABI mirrors the production structs extracted by test_metal_iq2_signs.py. */
typedef struct {
    int32_t nei0, nei1;
    uint64_t nbi1;
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12, ne13;
    uint64_t nb10, nb11, nb12;
    int32_t ne0, ne1;
    uint64_t nb1;
    int32_t nr0, tp_rank, tp_world, tp_addend, tp_expert_base;
} MVId;
typedef struct {
    uint32_t width, rows;
    uint64_t gate_row_stride, up_row_stride, mid_row_stride, weight_stride;
    uint32_t write_clamped;
    float clamp_value;
} Activation;
typedef struct { uint32_t active_mask, accumulate; } Split;
typedef struct { uint16_t d; uint8_t qs[64]; } IQ2;
_Static_assert(sizeof(MVId) == 136 && sizeof(Activation) == 48 && sizeof(IQ2) == 66,
               "production IQ2 wrapper ABI changed");

enum { E = 6, GUARD = 256, NR = 4, NSG = 2, SCRATCH = 2176 };
static const uint32_t poison = 0x7fc12345u;
static NSString *const names[] = {
    @"kernel_mul_mv_id_iq2_xxs_pair_f32",
    @"kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32",
    @"kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32",
    @"kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32",
};
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pipelines[2][4];
static unsigned cases;

static void require(bool ok, const char *message) {
    if (!ok) { fprintf(stderr, "IQ2 sign oracle FAIL: %s\n", message); exit(1); }
}
static uint32_t random32(uint32_t *state) {
    return *state = *state * 1664525u + 1013904223u;
}
static void *data(id<MTLBuffer> b) { return (char *)b.contents + GUARD; }
static id<MTLBuffer> buffer(size_t bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:bytes + 2u * GUARD
                                        options:MTLResourceStorageModeShared];
    require(b != nil, "buffer allocation");
    memset(b.contents, 0xa5, b.length);
    return b;
}
static void poison_output(id<MTLBuffer> b) {
    uint32_t *p = data(b);
    for (size_t i = 0; i < (b.length - 2u * GUARD) / 4u; ++i) p[i] = poison;
}
static void check_guards(id<MTLBuffer> b) {
    const uint8_t *p = b.contents;
    for (size_t i = 0; i < GUARD; ++i)
        require(p[i] == 0xa5 && p[b.length - 1u - i] == 0xa5,
                "buffer guard overwritten");
}
static uint64_t hash(id<MTLBuffer> b) {
    const uint8_t *p = b.contents;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < b.length; ++i) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static void bind(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> b, unsigned index) {
    [enc setBuffer:b offset:GUARD atIndex:index];
}
static void finish(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
        fprintf(stderr, "%s\n", cb.error.description.UTF8String);
    require(cb.status == MTLCommandBufferStatusCompleted, "GPU execution");
}

static void compile(NSString *source, bool fast) {
    for (unsigned arm = 0; arm < 2; ++arm) {
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.preprocessorMacros = @{@"DS4_METAL_IQ2_PAIR_POPCOUNT": @(arm)};
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = fast;
#pragma clang diagnostic pop
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) fprintf(stderr, "%s\n", error.description.UTF8String);
        require(library != nil, "compile current production shader");
        for (unsigned kind = 0; kind < 4; ++kind) {
            MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
            const short nsg = NSG;
            [values setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
            id<MTLFunction> fn = [library newFunctionWithName:names[kind]
                                             constantValues:values error:&error];
            pipelines[arm][kind] = fn ?
                [device newComputePipelineStateWithFunction:fn error:&error] : nil;
            if (!pipelines[arm][kind]) fprintf(stderr, "%s: %s\n",
                                              names[kind].UTF8String, error.description.UTF8String);
            require(pipelines[arm][kind] != nil, "wrapper pipeline creation");
            require(pipelines[arm][kind].threadExecutionWidth == 32 &&
                    pipelines[arm][kind].maxTotalThreadsPerThreadgroup >= 32 * NSG,
                    "production threadgroup shape unavailable");
        }
    }
}

static void check_output(id<MTLBuffer> reference, id<MTLBuffer> candidate,
                         unsigned kind, unsigned which, uint32_t mask,
                         unsigned h, unsigned tokens) {
    const size_t count = (reference.length - 2u * GUARD) / sizeof(uint32_t);
    const uint32_t *ref = data(reference), *got = data(candidate);
    const unsigned stride = which == 2 ? h + 5u : h;
    for (size_t i = 0; i < count; ++i) {
        const size_t pair = i / stride;
        const bool active = pair < tokens * E && i % stride < h &&
            (kind != 3 || (mask & (1u << (pair % E)))) &&
            (which != 2 || kind != 0);
        const bool ok = active
            ? ref[i] == got[i] && (got[i] & 0x7f800000u) != 0x7f800000u
            : ref[i] == poison && got[i] == poison;
        if (!ok) {
            fprintf(stderr, "kind=%s tensor=%u mask=%02x index=%zu active=%d "
                    "ref=%08x candidate=%08x\n", names[kind].UTF8String,
                    which, mask, i, active, ref[i], got[i]);
            require(false, "exact outputs or inactive-row preservation");
        }
    }
    check_guards(reference);
    check_guards(candidate);
}

static void run_shape(unsigned d, unsigned h, unsigned tokens, bool fast) {
    @autoreleasepool {
        const size_t row_bytes = d / 256u * sizeof(IQ2);
        const size_t expert_bytes = h * row_bytes;
        id<MTLBuffer> wg = buffer(E * expert_bytes), wu = buffer(E * expert_bytes);
        id<MTLBuffer> x = buffer(tokens * (d + 4u) * sizeof(float));
        id<MTLBuffer> ids = buffer(tokens * (E + 3u) * sizeof(int32_t));
        id<MTLBuffer> weights = buffer(tokens * E * 3u * sizeof(float));
        id<MTLBuffer> gate_addrs = buffer(E * sizeof(uint64_t));
        id<MTLBuffer> up_addrs = buffer(E * sizeof(uint64_t));
        id<MTLBuffer> output[2][3];
        for (unsigned arm = 0; arm < 2; ++arm) for (unsigned j = 0; j < 3; ++j)
            output[arm][j] = buffer((tokens * E * (j == 2 ? h + 5u : h) + h) * sizeof(float));
        uint32_t seed = 0x5a1398d7u;
        for (unsigned arm = 0; arm < 2; ++arm) {
            IQ2 *blocks = data(arm ? wu : wg);
            for (size_t i = 0; i < E * expert_bytes / sizeof(IQ2); ++i) {
                blocks[i].d = i % 29u ? 0x1400u + (random32(&seed) & 0xfffu) : 0;
                for (unsigned sub = 0; sub < 8; ++sub) {
                    uint32_t metadata = (uint32_t)((i + sub) & 15u) << 28;
                    for (unsigned l = 0; l < 4; ++l) {
                        blocks[i].qs[sub * 8u + l] = random32(&seed) >> 24;
                        const uint32_t code = (uint32_t)(i * 32u + sub * 4u + l + arm) & 127u;
                        metadata |= code << (7u * l);
                    }
                    memcpy(blocks[i].qs + sub * 8u + 4u, &metadata, sizeof(metadata));
                }
            }
        }
        float *input = data(x), *routes = data(weights);
        int32_t *selected = data(ids);
        const int32_t order[E] = {5, 1, 4, 0, 3, 2};
        for (unsigned t = 0; t < tokens; ++t) {
            for (unsigned i = 0; i < d; ++i)
                input[t * (d + 4u) + i] = i % 17u ?
                    ((int)(random32(&seed) % 2049u) - 1024) / 1001.0f : -0.0f;
            for (unsigned s = 0; s < E; ++s) {
                selected[t * (E + 3u) + s] = order[s];
                routes[(t * E + s) * 3u] = s == 2 ? 0.0f : (s + 1u) / 21.0f;
            }
        }
        for (unsigned e = 0; e < E; ++e) {
            ((uint64_t *)data(gate_addrs))[e] = wg.gpuAddress + GUARD + e * expert_bytes;
            ((uint64_t *)data(up_addrs))[e] = wu.gpuAddress + GUARD + e * expert_bytes;
        }
        id<MTLBuffer> readonly[] = {wg, wu, x, ids, weights, gate_addrs, up_addrs};
        uint64_t snapshots[7];
        for (unsigned i = 0; i < 7; ++i) snapshots[i] = hash(readonly[i]);
        MVId args = {
            .nei0 = E, .nei1 = tokens, .nbi1 = (E + 3u) * sizeof(int32_t),
            .ne00 = d, .ne01 = h, .ne02 = E, .nb00 = 2,
            .nb01 = row_bytes, .nb02 = expert_bytes,
            .ne10 = d, .ne11 = 1, .ne12 = tokens, .ne13 = 1,
            .nb10 = sizeof(float), .nb11 = (d + 4u) * sizeof(float),
            .nb12 = (d + 4u) * sizeof(float), .ne0 = h, .ne1 = E,
            .nb1 = h * sizeof(float), .nr0 = NR, .tp_world = 1,
        };
        Activation act = {
            .width = h, .rows = tokens * E,
            .gate_row_stride = h * sizeof(float), .up_row_stride = h * sizeof(float),
            .mid_row_stride = (h + 5u) * sizeof(float), .weight_stride = 3u * sizeof(float),
        };
        const uint32_t masks[] = {0u, 0x15u, 0x3fu};
        for (unsigned clamp = 0; clamp < 2; ++clamp) {
            act.clamp_value = clamp ? 7.0f : 0.0f;
            for (unsigned kind = 0; kind < 4; ++kind) {
                for (unsigned mi = 0; mi < (kind == 3 ? 3u : 1u); ++mi) {
                    Split split = {.active_mask = kind == 3 ? masks[mi] : 0x3fu};
                    for (unsigned arm = 0; arm < 2; ++arm)
                        for (unsigned j = 0; j < 3; ++j) poison_output(output[arm][j]);
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    for (unsigned arm = 0; arm < 2; ++arm) {
                        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                        [enc setComputePipelineState:pipelines[arm][kind]];
                        [enc setBytes:&args length:sizeof(args) atIndex:0];
                        if (kind == 0) {
                            bind(enc, wg, 1); bind(enc, wu, 2); bind(enc, x, 3);
                            bind(enc, output[arm][0], 4); bind(enc, output[arm][1], 5);
                            bind(enc, ids, 6);
                        } else {
                            [enc setBytes:&act length:sizeof(act) atIndex:1];
                            unsigned first = 2;
                            if (kind == 3) {
                                [enc setBytes:&split length:sizeof(split) atIndex:2];
                                first = 3;
                            }
                            if (kind == 1) {
                                for (unsigned s = 0; s < E; ++s) {
                                    [enc setBuffer:wg offset:GUARD + order[s] * expert_bytes atIndex:2 + s];
                                    [enc setBuffer:wu offset:GUARD + order[s] * expert_bytes atIndex:8 + s];
                                }
                                first = 14;
                            } else {
                                bind(enc, gate_addrs, first++); bind(enc, up_addrs, first++);
                                [enc useResource:wg usage:MTLResourceUsageRead];
                                [enc useResource:wu usage:MTLResourceUsageRead];
                            }
                            bind(enc, x, first++);
                            for (unsigned j = 0; j < 3; ++j) bind(enc, output[arm][j], first++);
                            if (kind != 1) bind(enc, ids, first++);
                            bind(enc, weights, first);
                        }
                        [enc setThreadgroupMemoryLength:SCRATCH atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(h / (NR * NSG), 1, tokens * E)
                           threadsPerThreadgroup:MTLSizeMake(32, NSG, 1)];
                        [enc endEncoding];
                    }
                    finish(cb);
                    for (unsigned j = 0; j < 3; ++j)
                        check_output(output[0][j], output[1][j], kind, j,
                                     split.active_mask, h, tokens);
                    ++cases;
                }
            }
        }
        for (unsigned i = 0; i < 7; ++i) {
            require(hash(readonly[i]) == snapshots[i], "readonly tensor modified");
            check_guards(readonly[i]);
        }
        fprintf(stderr, "PASS IQ2 wrappers D=%u H=%u E=%u tokens=%u fast=%d "
                "gate/up/mid bitwise, masks, strides and guards.\n", d, h, E, tokens, fast);
    }
}

int main(int argc, char **argv) {
    require(argc == 2, "usage: test_metal_iq2_signs SHADER.metal");
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, "Metal device unavailable; run with GPU access");
        queue = [device newCommandQueue];
        require(queue != nil, "Metal command queue");
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                    encoding:NSUTF8StringEncoding error:&error];
        require(source != nil, "read extracted production shader");
        for (unsigned fast = 0; fast < 2; ++fast) {
            compile(source, fast != 0);
            run_shape(256, 256, 2, fast != 0);
            run_shape(4096, 2048, 1, fast != 0);
        }
        printf("PASS: %u IQ2 wrapper cases on %s, strict/fast macro on/off.\n",
               cases, device.name.UTF8String);
    }
    return 0;
}
