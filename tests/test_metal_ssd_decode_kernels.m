#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// GGUF-free comparison of the production legacy and SSD-specialized kernels.
// Synthetic buffers mimic streamed cache slots, including nonzero offsets and
// indirect addresses. This is a correctness test, not an SSD throughput test.
enum { GUARD = 256, SLOTS = 6 };
static const uint32_t poison = 0x7fc12345u;
typedef struct {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1, nr0;
    int16_t r2, r3;
} mv_args;
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
} id_args;
typedef struct {
    uint32_t width, rows;
    uint64_t gate_row_stride, up_row_stride, mid_row_stride, weight_stride;
    uint32_t write_clamped;
    float clamp_value;
} act_args;
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } q4_block;
typedef struct { uint16_t d; int8_t qs[32]; } q8_block;
typedef struct { uint16_t d, dmin; uint8_t scales[12], qh[32], qs[128]; } q5_block;
typedef struct { uint8_t ql[128], qh[64]; int8_t scales[16]; uint16_t d; } q6_block;
typedef struct {
    uint32_t in_dim, mid_dim, out_dim, n_total_expert, n_expert_used, n_tokens;
    uint32_t mid_token_stride, down_type;
    float swiglu_clamp;
    int32_t tp_rank, tp_world, tp_expert_base;
    uint64_t gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes;
    uint64_t down_expert_bytes, down_row_bytes;
} glm_args;
typedef struct {
    int32_t ne02, ne10, ne11;
    uint64_t nb11, nb12;
    int32_t ne21, ne20;
    uint64_t nb21;
} map_args;
typedef struct {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne11;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne20, ne21, ne0, ne1;
    int16_t r2, r3;
    int32_t tp_rank, tp_world, tp_expert_base;
} mm_id_args;
_Static_assert(sizeof(mv_args) == 112, "mul_mv ABI");
_Static_assert(sizeof(id_args) == 136, "mul_mv_id ABI");
_Static_assert(sizeof(act_args) == 48, "SwiGLU ABI");
_Static_assert(sizeof(q4_block) == 144 && sizeof(q8_block) == 34, "quant ABI");
_Static_assert(sizeof(q5_block) == 176 && sizeof(q6_block) == 210, "GLM quant ABI");
_Static_assert(sizeof(glm_args) == 96 && sizeof(map_args) == 48 &&
               sizeof(mm_id_args) == 104, "GLM/map ABI");

static void require(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "SSD kernel test: %s\n", message); exit(1); }
}

static NSString *metal_prelude(void) {
    return @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
            "#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
            "#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
            "#define QK8_0 32\n"
            "#ifndef QK_K\n#define QK_K 256\n#endif\n"
            "#define N_SIMDWIDTH 32\n"
            "#define N_R0_Q8_0 2\n"
            "#define N_SG_Q8_0 4\n"
            "#define FC_MUL_MV 600\n"
            "#define FC_MUL_MM 700\n"
            "#define FC_BIN 1300\n"
            "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
            "#define M_PI_F 3.14159265358979323846f\n"
            "enum ds4_sort_order { DS4_SORT_ORDER_ASC, DS4_SORT_ORDER_DESC };\n"
            "struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n"
            "struct block_q8_K { float d; int8_t qs[QK_K]; "
            "int16_t bsums[QK_K / 16]; };\n";
}

/* Keep the same concatenation order as ds4_metal.m. The test specializes
 * and dispatches the checked-in production kernels; it carries no kernel copy. */
static NSString *load_metal_source(void) {
    static const char *paths[] = {
        "metal/activations.metal",
        "metal/flash_attn.metal",
        "metal/dense.metal",
        "metal/moe.metal",
        "metal/dsv4_hc.metal",
        "metal/unary.metal",
        "metal/dsv4_kv.metal",
        "metal/dsv4_rope.metal",
        "metal/dsv4_misc.metal",
        "metal/argsort.metal",
        "metal/cpy.metal",
        "metal/concat.metal",
        "metal/get_rows.metal",
        "metal/sum_rows.metal",
        "metal/softmax.metal",
        "metal/repeat.metal",
        "metal/glu.metal",
        "metal/norm.metal",
        "metal/bin.metal",
        "metal/set_rows.metal",
    };
    const char *root_env = getenv("DS4_SOURCE_ROOT");
    NSString *root = root_env && root_env[0]
        ? [NSString stringWithUTF8String:root_env]
        : @".";
    NSMutableString *source = [NSMutableString stringWithString:metal_prelude()];
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        NSString *relative = [NSString stringWithUTF8String:paths[i]];
        NSString *path = [root stringByAppendingPathComponent:relative];
        NSError *error = nil;
        NSString *part = [NSString stringWithContentsOfFile:path
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
        if (!part) {
            fprintf(stderr, "test-metal-ssd-decode-kernels: cannot read %s: %s\n",
                    [path fileSystemRepresentation],
                    [[error localizedDescription] UTF8String]);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", relative, part];
    }
    return source;
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, id<MTLLibrary> lib,
                                           NSString *name, int16_t nsg) {
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    NSError *error = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name constantValues:constants error:&error];
    if (!fn) fprintf(stderr, "%s: %s\n", name.UTF8String, error.description.UTF8String);
    require(fn != nil, "function specialization failed");
    id<MTLComputePipelineState> result = [dev newComputePipelineStateWithFunction:fn error:&error];
    if (!result) fprintf(stderr, "%s: %s\n", name.UTF8String, error.description.UTF8String);
    require(result != nil, "pipeline compilation failed");
    return result;
}
static uint32_t random_u32(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}
static void *data(id<MTLBuffer> b) { return (char *)b.contents + GUARD; }
static id<MTLBuffer> buffer(id<MTLDevice> dev, NSUInteger bytes) {
    id<MTLBuffer> b = [dev newBufferWithLength:GUARD * 2 + bytes options:MTLResourceStorageModeShared];
    require(b != nil, "buffer allocation failed");
    memset(b.contents, 0xa5, b.length);
    return b;
}
static id<MTLBuffer> output(id<MTLDevice> dev, NSUInteger count) {
    id<MTLBuffer> b = buffer(dev, count * sizeof(float));
    uint32_t *p = data(b);
    for (NSUInteger i = 0; i < count; ++i) p[i] = poison;
    return b;
}
static uint64_t hash(id<MTLBuffer> b) {
    const uint8_t *p = b.contents;
    uint64_t h = 14695981039346656037ull;
    for (NSUInteger i = 0; i < b.length; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
static void equal_output(id<MTLBuffer> a, id<MTLBuffer> b, NSUInteger rows,
                         NSUInteger width, NSUInteger stride) {
    require(a.length == b.length, "output lengths differ");
    const uint8_t *ap = a.contents, *bp = b.contents;
    for (NSUInteger i = 0; i < GUARD; ++i)
        require(ap[i] == 0xa5 && bp[i] == 0xa5 &&
                ap[a.length - GUARD + i] == 0xa5 && bp[b.length - GUARD + i] == 0xa5,
                "output guard overwritten");
    const uint32_t *av = data(a), *bv = data(b);
    for (NSUInteger r = 0; r < rows; ++r) {
        for (NSUInteger c = 0; c < stride; ++c) {
            NSUInteger i = r * stride + c;
            if (c < width) {
                // Integer checks remain valid when host is built with fast-math.
                require((av[i] & 0x7f800000u) != 0x7f800000u &&
                        (bv[i] & 0x7f800000u) != 0x7f800000u, "nonfinite/unwritten output");
            } else require(av[i] == poison && bv[i] == poison, "stride padding overwritten");
            if (av[i] != bv[i]) {
                fprintf(stderr, "row=%lu col=%lu old=%08x new=%08x\n",
                        (unsigned long)r, (unsigned long)c, av[i], bv[i]);
                require(0, "output is not bitwise equal");
            }
        }
    }
}
static void finish(id<MTLCommandBuffer> cb) {
    [cb commit]; [cb waitUntilCompleted];
    if (cb.error) fprintf(stderr, "%s\n", cb.error.description.UTF8String);
    require(cb.status == MTLCommandBufferStatusCompleted, "GPU execution failed");
}
static void fill_x(id<MTLBuffer> x, NSUInteger count, uint32_t *seed) {
    float *p = data(x);
    for (NSUInteger i = 0; i < count; ++i)
        p[i] = i % 11 == 0 ? 0.f : ((int)(random_u32(seed) % 8193) - 4096) / 4096.f;
}
static void test_q4(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> oldp, id<MTLComputePipelineState> newp,
                    BOOL addr, int k, int rows, int tokens, float clamp) {
    uint32_t seed = 17u + k + rows + tokens;
    NSUInteger wb = (NSUInteger)(k / 256) * 144 * rows;
    id<MTLBuffer> gate[SLOTS], up[SLOTS];
    uint64_t original[2 * SLOTS];
    for (int s = 0; s < SLOTS; ++s) {
        gate[s] = buffer(dev, wb); up[s] = buffer(dev, wb);
        for (int arm = 0; arm < 2; ++arm) {
            q4_block *w = data(arm ? up[s] : gate[s]);
            for (NSUInteger i = 0; i < wb / 144; ++i) {
                w[i].d = 0x1800 + (random_u32(&seed) & 0x3ff);
                w[i].dmin = 0x1400 + (random_u32(&seed) & 0x3ff);
                for (int c = 0; c < 12; ++c) w[i].scales[c] = random_u32(&seed) >> 24;
                for (int c = 0; c < 128; ++c) w[i].qs[c] = random_u32(&seed) >> 24;
            }
        }
    }
    // Repeated routing id: two slots must address the same expert.
    gate[3] = gate[1]; up[3] = up[1];
    for (int s = 0; s < SLOTS; ++s) { original[s] = hash(gate[s]); original[6+s] = hash(up[s]); }
    int32_t expert[SLOTS] = {0, 17, 383, 17, 255, 1};
    id<MTLBuffer> ids = buffer(dev, tokens * SLOTS * 4);
    id<MTLBuffer> ga = buffer(dev, 384 * 8), ua = buffer(dev, 384 * 8);
    memset(data(ga), 0, 384 * 8); memset(data(ua), 0, 384 * 8);
    for (int s = 0; s < SLOTS; ++s) {
        ((uint64_t *)data(ga))[expert[s]] = gate[s].gpuAddress + GUARD;
        ((uint64_t *)data(ua))[expert[s]] = up[s].gpuAddress + GUARD;
        for (int t = 0; t < tokens; ++t) ((int32_t *)data(ids))[t*SLOTS+s] = expert[s];
    }
    id<MTLBuffer> x = buffer(dev, k * tokens * 4), weights = buffer(dev, SLOTS * tokens * 16);
    fill_x(x, k * tokens, &seed);
    for (int r = 0; r < tokens*SLOTS; ++r) ((float *)data(weights))[r*4] = (r%6) / 7.f;
    uint64_t xhash = hash(x), whash = hash(weights), ihash = hash(ids);
    const NSUInteger mid_stride = rows + 8;
    id<MTLBuffer> out[2][3];
    for (int arm = 0; arm < 2; ++arm)
        for (int d = 0; d < 3; ++d)
            out[arm][d] = output(dev, tokens * SLOTS * (d == 2 ? mid_stride : rows));
    id_args args = {
        .nei0 = SLOTS, .nei1 = tokens, .nbi1 = SLOTS * 4,
        .ne00 = k, .ne01 = rows, .ne02 = 384, .nb00 = 144,
        .nb01 = (uint64_t)k/256*144, .nb02 = wb,
        .ne10 = k, .ne11 = 1, .ne12 = tokens, .ne13 = 1,
        .nb10 = 4, .nb11 = (uint64_t)k*4, .nb12 = (uint64_t)k*4,
        .ne0 = rows, .ne1 = SLOTS, .nb1 = (uint64_t)rows*4, .nr0 = 2, .tp_world = 1,
    };
    act_args act = {.width = rows, .rows = tokens*SLOTS,
        .gate_row_stride = (uint64_t)rows*4, .up_row_stride = (uint64_t)rows*4,
        .mid_row_stride = mid_stride*4, .weight_stride = 16, .clamp_value = clamp};
    for (int arm = 0; arm < 2; ++arm) {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:arm ? newp : oldp];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBytes:&act length:sizeof(act) atIndex:1];
        if (addr) {
            [enc setBuffer:ga offset:GUARD atIndex:2];
            [enc setBuffer:ua offset:GUARD atIndex:3];
            [enc setBuffer:x offset:GUARD atIndex:4];
            for (int d = 0; d < 3; ++d) [enc setBuffer:out[arm][d] offset:GUARD atIndex:5+d];
            [enc setBuffer:ids offset:GUARD atIndex:8];
            [enc setBuffer:weights offset:GUARD atIndex:9];
            for (int s = 0; s < SLOTS; ++s) {
                [enc useResource:gate[s] usage:MTLResourceUsageRead];
                [enc useResource:up[s] usage:MTLResourceUsageRead];
            }
        } else {
            for (int s = 0; s < SLOTS; ++s) {
                [enc setBuffer:gate[s] offset:GUARD atIndex:2+s];
                [enc setBuffer:up[s] offset:GUARD atIndex:8+s];
            }
            [enc setBuffer:x offset:GUARD atIndex:14];
            for (int d = 0; d < 3; ++d) [enc setBuffer:out[arm][d] offset:GUARD atIndex:15+d];
            [enc setBuffer:weights offset:GUARD atIndex:18];
        }
        [enc setThreadgroupMemoryLength:256 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows/4, 1, tokens*SLOTS)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        [enc endEncoding]; finish(cb);
    }
    for (int d = 0; d < 3; ++d)
        equal_output(out[0][d], out[1][d], tokens*SLOTS, rows, d == 2 ? mid_stride : rows);
    for (int s = 0; s < SLOTS; ++s)
        require(hash(gate[s]) == original[s] && hash(up[s]) == original[6+s], "Q4 weight mutated");
    require(hash(x) == xhash && hash(weights) == whash && hash(ids) == ihash, "Q4 input mutated");
}

static void test_q8(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                    id<MTLComputePipelineState> oldp, id<MTLComputePipelineState> newp,
                    int nsg, BOOL store, int k, int rows, float clamp) {
    uint32_t seed = 41u + k + rows;
    NSUInteger wb = (NSUInteger)(k / 32) * 34 * rows;
    id<MTLBuffer> gate = buffer(dev, wb), up = buffer(dev, wb), x = buffer(dev, k*4);
    for (int arm = 0; arm < 2; ++arm) {
        q8_block *w = data(arm ? up : gate);
        for (NSUInteger i = 0; i < wb/34; ++i) {
            w[i].d = 0x1800 + (random_u32(&seed) & 0x3ff);
            for (int c = 0; c < 32; ++c) w[i].qs[c] = (int)(random_u32(&seed) >> 24) - 128;
        }
    }
    fill_x(x, k, &seed);
    uint64_t gh = hash(gate), uh = hash(up);
    mv_args args = {.ne00=k, .ne01=rows, .ne02=1, .nb00=34,
        .nb01=(uint64_t)k/32*34, .nb02=wb, .nb03=wb,
        .ne10=k, .ne11=1, .ne12=1, .nb10=4, .nb11=(uint64_t)k*4,
        .nb12=(uint64_t)k*4, .nb13=(uint64_t)k*4, .ne0=rows, .ne1=1,
        .nr0=2, .r2=1, .r3=1};
    id<MTLBuffer> out[2][3];
    for (int arm = 0; arm < 2; ++arm)
        for (int d = 0; d < 3; ++d) out[arm][d] = output(dev, rows);
    // Repeated dispatches exercise recycled threadgroup memory. In mid-only
    // mode both unused diagnostic outputs alias mid, just like production.
    for (int repeat = 0; repeat < 5; ++repeat) {
        fill_x(x, k, &seed);
        uint64_t xh = hash(x);
        for (int arm = 0; arm < 2; ++arm)
            for (int d = 0; d < 3; ++d)
                for (int row = 0; row < rows; ++row)
                    ((uint32_t *)data(out[arm][d]))[row] = poison;
        for (int arm = 0; arm < 2; ++arm) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:arm ? newp : oldp];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:gate offset:GUARD atIndex:1];
            [enc setBuffer:up offset:GUARD atIndex:2];
            [enc setBuffer:x offset:GUARD atIndex:3];
            for (int d = 0; d < 3; ++d)
                [enc setBuffer:out[arm][store ? d : 2] offset:GUARD atIndex:4+d];
            [enc setBytes:&clamp length:sizeof(clamp) atIndex:7];
            [enc setThreadgroupMemoryLength:512 atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(rows/2, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
            [enc endEncoding]; finish(cb);
        }
        for (int d = store ? 0 : 2; d < 3; ++d)
            equal_output(out[0][d], out[1][d], 1, rows, rows);
        require(hash(x) == xh, "Q8 input mutated");
    }
    require(hash(gate) == gh && hash(up) == uh, "Q8 weight mutated");
}

static void encode_q4_dense(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pipeline, const mv_args args[2],
                           id<MTLBuffer> __strong w[2], id<MTLBuffer> x,
                           id<MTLBuffer> __strong out[2], int arm, NSUInteger set) {
    [enc setComputePipelineState:pipeline];
    [enc setThreadgroupMemoryLength:32 atIndex:0];
    const int tokens = args[0].ne11;
    if (arm == 0 || arm == 2 || arm == 4) {
        [enc setBuffer:x offset:GUARD atIndex:2];
        for (int m = 0; m < 2; ++m) {
            [enc setBytes:&args[m] length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:w[m] offset:GUARD + set*args[m].nb03 atIndex:1];
            [enc setBuffer:out[m] offset:GUARD + set*args[m].ne0*tokens*sizeof(float) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((args[m].ne0+3)/4, arm == 2 ? (tokens+1)/2 : tokens, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        }
    } else {
        for (int m = 0; m < 2; ++m) {
            [enc setBytes:&args[m] length:sizeof(mv_args) atIndex:m];
            [enc setBuffer:w[m] offset:GUARD + set*args[m].nb03 atIndex:2+m];
            [enc setBuffer:out[m] offset:GUARD + set*args[m].ne0*tokens*sizeof(float) atIndex:5+m];
        }
        [enc setBuffer:x offset:GUARD atIndex:4];
        int max_rows = args[0].ne0 > args[1].ne0 ? args[0].ne0 : args[1].ne0;
        [enc dispatchThreadgroups:MTLSizeMake((max_rows+3)/4, arm == 3 ? (tokens+1)/2 : tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    }
}

static int compare_double(const void *a, const void *b);
static void test_q4_dense_pair(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                              id<MTLComputePipelineState> __strong p[5],
                              int k, int rows_a, int rows_b, int tokens, int padding, int timing) {
    // Standalone oracle and sequential pair. Pad physical weight rows through
    // a full threadgroup for the unchanged classic two-row loader, while
    // keeping logical/output extents exact. This padding is fixture-only.
    int rows[2] = {rows_a, rows_b};
    const int sets = timing == 2 ? 8 : 1;
    id<MTLBuffer> w[2], out[5][2];
    mv_args args[2];
    uint64_t wh[2];
    uint32_t seed = 7919u + k + rows_a + rows_b + tokens + padding;
    for (int m = 0; m < 2; ++m) {
        NSUInteger stride = (k / 256 + padding * (m+1)) * sizeof(q4_block);
        NSUInteger bytes = ((rows[m] + 3u) & ~3u) * stride;
        w[m] = buffer(dev, bytes*sets);
        q4_block *blocks = data(w[m]);
        for (NSUInteger b = 0; b < bytes*sets / sizeof(*blocks); ++b) {
            blocks[b].d = b % 17 ? 0x2400 + (random_u32(&seed) & 0x3ff) : 0;
            blocks[b].dmin = b % 13 ? 0x1c00 + (random_u32(&seed) & 0x3ff) : 0;
            for (int i = 0; i < 12; ++i) blocks[b].scales[i] = random_u32(&seed) >> 24;
            for (int i = 0; i < 128; ++i) blocks[b].qs[i] = random_u32(&seed) >> 24;
        }
        wh[m] = hash(w[m]);
        args[m] = (mv_args){.ne00=k, .ne01=rows[m], .ne02=1,
            .nb00=1, .nb01=stride, .nb02=bytes, .nb03=bytes,
            .ne10=k, .ne11=tokens, .ne12=1, .nb10=4, .nb11=(uint64_t)k*4,
            .nb12=(uint64_t)k*tokens*4, .nb13=(uint64_t)k*tokens*4,
            .ne0=rows[m], .ne1=tokens, .nr0=2, .r2=1, .r3=1};
        for (int arm = 0; arm < 5; ++arm) out[arm][m] = output(dev, rows[m]*tokens*sets);
    }
    id<MTLBuffer> x = buffer(dev, k*tokens*sizeof(float));
    for (int repeat = 0; repeat < 3; ++repeat) {
        fill_x(x, k*tokens, &seed);
        uint64_t xh = hash(x);
        for (int arm = 0; arm < 5; ++arm) {
            for (int m = 0; m < 2; ++m) for (int i = 0; i < rows[m]*tokens*sets; ++i)
                ((uint32_t *)data(out[arm][m]))[i] = poison;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int set = 0; set < sets; ++set)
                encode_q4_dense(enc, p[arm], args, w, x, out[arm], arm, set);
            [enc endEncoding]; finish(cb);
        }
        for (int arm = 1; arm < 5; ++arm) for (int m = 0; m < 2; ++m)
            equal_output(out[0][m], out[arm][m], tokens*sets, rows[m], rows[m]);
        require(hash(x) == xh, "Q4 dense pair input mutated");
    }
    require(hash(w[0]) == wh[0] && hash(w[1]) == wh[1], "Q4 dense pair weight mutated");
    const uint64_t timed_x_hash = hash(x);
    if (timing == 2) {
        enum { SAMPLES = 16, REPEATS = 64 };
        double times[2][SAMPLES];
        for (int round = -2; round < SAMPLES; ++round) for (int pos = 0; pos < 2; ++pos) {
            const int arm = (round & 1) ? 1-pos : pos;
            const int index = arm ? 4 : 0;
            // Poison outside GPU timing. Every rotating set must be written
            // by this sample, not inherited from the untimed oracle above.
            for (int m = 0; m < 2; ++m)
                for (int i = 0; i < rows[m]*tokens*sets; ++i)
                    ((uint32_t *)data(out[index][m]))[i] = poison;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int repeat = 0; repeat < REPEATS; ++repeat)
                encode_q4_dense(enc, p[index], args, w, x, out[index], index, repeat % sets);
            [enc endEncoding]; finish(cb);
            // The legacy paired oracle is independent and never timed here.
            for (int m = 0; m < 2; ++m)
                equal_output(out[1][m], out[index][m], tokens*sets, rows[m], rows[m]);
            if (round >= 0) {
                require(cb.GPUEndTime > cb.GPUStartTime, "GPU timestamps unavailable");
                times[arm][round] = (cb.GPUEndTime - cb.GPUStartTime) * 1.e6 / REPEATS;
                printf("Q4 single-vector sample=%d arm=%c us=%.6f\n", round, arm ? 'B' : 'A', times[arm][round]);
            }
        }
        double median[2];
        for (int arm = 0; arm < 2; ++arm) {
            qsort(times[arm], SAMPLES, sizeof(double), compare_double);
            median[arm] = (times[arm][SAMPLES/2-1] + times[arm][SAMPLES/2]) * .5;
        }
        for (int m = 0; m < 2; ++m)
            equal_output(out[0][m], out[4][m], tokens*sets, rows[m], rows[m]);
        printf("Q4 single-vector kernel K=%d M=%d+%d N=%d sets=%d: %.3f -> %.3f us (%+.2f%% time); rotating resident weights, not model t/s.\n",
            k, rows_a, rows_b, tokens, sets, median[0], median[1], 100.*(median[1]/median[0]-1.));
        require(hash(x) == timed_x_hash && hash(w[0]) == wh[0] && hash(w[1]) == wh[1],
                "Q4 single-vector timing mutated input/weights");
    } else if (timing) {
        enum { SAMPLES = 8, REPEATS = 64 };
        double times[4][SAMPLES];
        for (int round = -2; round < SAMPLES; ++round) for (int position = 0; position < 4; ++position) {
            const int order[] = {0,2,1,3}; // adjacent legacy/candidate, reversed every round
            const int arm = order[(round & 1) ? 3-position : position];
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int repeat = 0; repeat < REPEATS; ++repeat)
                encode_q4_dense(enc, p[arm], args, w, x, out[arm], arm, 0);
            [enc endEncoding]; finish(cb);
            if (round >= 0) {
                require(cb.GPUEndTime > cb.GPUStartTime, "GPU timestamps unavailable");
                times[arm][round] = (cb.GPUEndTime - cb.GPUStartTime) * 1.e6 / REPEATS;
            }
        }
        double median[4];
        for (int a = 0; a < 4; ++a) {
            qsort(times[a], SAMPLES, sizeof(double), compare_double);
            median[a] = (times[a][SAMPLES/2-1] + times[a][SAMPLES/2]) * .5;
        }
        for (int arm = 1; arm < 4; ++arm) for (int m = 0; m < 2; ++m)
            equal_output(out[0][m], out[arm][m], tokens, rows[m], rows[m]);
        printf("Q4 two-token kernel K=%d M=%d+%d N=%d: separate %.3f -> %.3f us (%+.2f%%), pair %.3f -> %.3f us (%+.2f%% time).\n",
            k, rows_a, rows_b, tokens, median[0], median[2], 100.*(median[2]/median[0]-1.),
            median[1], median[3], 100.*(median[3]/median[1]-1.));
    }
}

typedef struct {
    __strong id<MTLComputePipelineState> p[4]; // old/new single, old/new pair
    __strong id<MTLBuffer> w[2], x, out[4][2];
    mv_args args[2];
    int rows[2], nsg, k;
} projection_fixture;

static void init_projection(projection_fixture *f, id<MTLDevice> dev,
                            id<MTLComputePipelineState> __strong p[4], int nsg,
                            int k, int rows_a, int rows_b, int padding) {
    *f = (projection_fixture){0};
    f->rows[0] = rows_a; f->rows[1] = rows_b; f->nsg = nsg; f->k = k;
    uint32_t seed = 71u + k + rows_a + rows_b + padding;
    for (int arm = 0; arm < 4; ++arm) f->p[arm] = p[arm];
    for (int m = 0; m < 2; ++m) {
        // The legacy single kernel reads two weight rows even on odd tails.
        // Pad only its fixture allocation, not args.ne01 or the output. The
        // paired kernels must independently guard the true row extents.
        NSUInteger stride = (k/32 + padding*(m+1)) * sizeof(q8_block);
        NSUInteger bytes = ((f->rows[m] + 1u) & ~1u) * stride;
        f->w[m] = buffer(dev, bytes);
        q8_block *w = data(f->w[m]);
        for (NSUInteger b = 0; b < bytes / sizeof(*w); ++b) {
            w[b].d = b % 17 ? 0x1800 + (random_u32(&seed) & 0x3ff) : 0;
            for (int i = 0; i < 32; ++i) w[b].qs[i] = (int)(random_u32(&seed) >> 24) - 128;
        }
        f->args[m] = (mv_args){.ne00=k, .ne01=f->rows[m], .ne02=1,
            .nb00=34, .nb01=stride, .nb02=bytes, .nb03=bytes,
            .ne10=k, .ne11=1, .ne12=1, .nb10=4, .nb11=(uint64_t)k*4,
            .nb12=(uint64_t)k*4, .nb13=(uint64_t)k*4,
            .ne0=f->rows[m], .ne1=1, .nr0=2, .r2=1, .r3=1};
        for (int arm = 0; arm < 4; ++arm) f->out[arm][m] = output(dev, f->rows[m]);
    }
    f->x = buffer(dev, k * sizeof(float));
    fill_x(f->x, k, &seed);
}

static void encode_projection(projection_fixture *f, id<MTLComputeCommandEncoder> enc, int arm) {
    [enc setComputePipelineState:f->p[arm]];
    if (arm < 2) {
        [enc setBuffer:f->x offset:GUARD atIndex:2];
        [enc setThreadgroupMemoryLength:256 atIndex:0];
        for (int m = 0; m < 2; ++m) {
            [enc setBytes:&f->args[m] length:sizeof(mv_args) atIndex:0];
            [enc setBuffer:f->w[m] offset:GUARD atIndex:1];
            [enc setBuffer:f->out[arm][m] offset:GUARD atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((f->rows[m]+1)/2, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, f->nsg, 1)];
        }
    } else {
        [enc setBytes:&f->args[0] length:sizeof(mv_args) atIndex:0];
        [enc setBytes:&f->args[1] length:sizeof(mv_args) atIndex:1];
        [enc setBuffer:f->w[0] offset:GUARD atIndex:2];
        [enc setBuffer:f->w[1] offset:GUARD atIndex:3];
        [enc setBuffer:f->x offset:GUARD atIndex:4];
        [enc setBuffer:f->out[arm][0] offset:GUARD atIndex:5];
        [enc setBuffer:f->out[arm][1] offset:GUARD atIndex:6];
        [enc setThreadgroupMemoryLength:512 atIndex:0];
        int rows = f->rows[0] > f->rows[1] ? f->rows[0] : f->rows[1];
        [enc dispatchThreadgroups:MTLSizeMake((rows+1)/2, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, f->nsg, 1)];
    }
}

static void compare_projection(projection_fixture *f) {
    for (int m = 0; m < 2; ++m) for (int arm = 1; arm < 4; ++arm)
        equal_output(f->out[0][m], f->out[arm][m], 1, f->rows[m], f->rows[m]);
}

static void test_projection(projection_fixture *f, id<MTLCommandQueue> queue) {
    uint64_t wh[2] = {hash(f->w[0]), hash(f->w[1])};
    uint32_t seed = 513;
    for (int repeat = 0; repeat < 3; ++repeat) {
        fill_x(f->x, f->k, &seed);
        uint64_t xh = hash(f->x);
        for (int arm = 0; arm < 4; ++arm) {
            for (int m = 0; m < 2; ++m) for (int row = 0; row < f->rows[m]; ++row)
                ((uint32_t *)data(f->out[arm][m]))[row] = poison;
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode_projection(f, enc, arm);
            [enc endEncoding]; finish(cb);
        }
        compare_projection(f);
        require(hash(f->x) == xh, "Q8 projection input mutated");
    }
    require(hash(f->w[0]) == wh[0] && hash(f->w[1]) == wh[1], "Q8 projection weight mutated");
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void bench_projection(projection_fixture *f, id<MTLCommandQueue> queue) {
    enum { SAMPLES = 12, REPEATS = 128 };
    double times[4][SAMPLES];
    // Alternate all four arms and reverse the order every round; two warmups.
    for (int round = -2; round < SAMPLES; ++round) for (int position = 0; position < 4; ++position) {
        int arm = (round & 1) ? 3-position : position;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (int r = 0; r < REPEATS; ++r) encode_projection(f, enc, arm);
        [enc endEncoding]; finish(cb);
        if (round >= 0) {
            require(cb.GPUEndTime > cb.GPUStartTime, "GPU timestamps unavailable");
            times[arm][round] = (cb.GPUEndTime - cb.GPUStartTime) * 1.e6 / REPEATS;
        }
    }
    double median[4];
    for (int arm = 0; arm < 4; ++arm) {
        qsort(times[arm], SAMPLES, sizeof(double), compare_double);
        median[arm] = (times[arm][SAMPLES/2-1] + times[arm][SAMPLES/2]) * .5;
    }
    compare_projection(f);
    printf("Q8 MV kernel-only K=%d M=%d+%d NSG=%d: separate %.3f -> %.3f us (%+.2f%%), pair %.3f -> %.3f us (%+.2f%% time).\n",
           f->k, f->rows[0], f->rows[1], f->nsg, median[0], median[1],
           100.*(median[1]/median[0]-1.), median[2], median[3], 100.*(median[3]/median[2]-1.));
}

/* A full expert buffer with remote ids masked out is the oracle for a
 * rank-local binding. Poisoned remote mid rows catch missing ownership checks
 * without requiring a second machine or a distributed session. */
static void test_glm_down_tp(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                             id<MTLLibrary> lib, BOOL q6, int tokens) {
    enum { E = 8, K = 256, ROWS = 17, TOP = 6, STRIDE = TOP*K + 8 };
    const NSUInteger row_bytes = q6 ? sizeof(q6_block) : sizeof(q5_block);
    const NSUInteger expert_bytes = ROWS * row_bytes;
    id<MTLBuffer> full = buffer(dev, E * expert_bytes);
    uint32_t seed = 197;
    for (int b = 0; b < E*ROWS; b++) {
        if (q6) {
            q6_block *w = (q6_block *)data(full) + b;
            for (int i = 0; i < 128; i++) w->ql[i] = random_u32(&seed) >> 24;
            for (int i = 0; i < 64; i++) w->qh[i] = random_u32(&seed) >> 24;
            for (int i = 0; i < 16; i++) w->scales[i] = (int)(random_u32(&seed)%31)-15;
            w->d = 0x2000 + b;
        } else {
            q5_block *w = (q5_block *)data(full) + b;
            for (int i = 0; i < 128; i++) w->qs[i] = random_u32(&seed) >> 24;
            for (int i = 0; i < 32; i++) w->qh[i] = random_u32(&seed) >> 24;
            for (int i = 0; i < 12; i++) w->scales[i] = random_u32(&seed) >> 24;
            w->d = 0x2000 + b; w->dmin = 0x1800 + b;
        }
    }
    id<MTLComputePipelineState> p = pipeline(dev, lib,
        q6 ? @"kernel_glm_q6_K_down_f32" : @"kernel_glm_q5_K_down_f32", 2);
    const int32_t selected[TOP] = {0, 4, 7, 3, -1, E};
    for (int rank = 0; rank < 2; rank++) {
        id<MTLBuffer> local = buffer(dev, E/2 * expert_bytes);
        memcpy(data(local), (char *)data(full) + rank*E/2*expert_bytes, E/2*expert_bytes);
        id<MTLBuffer> ids[2] = {buffer(dev, tokens*TOP*4), buffer(dev, tokens*TOP*4)};
        id<MTLBuffer> mid = output(dev, tokens*STRIDE);
        for (int t = 0; t < tokens; t++) for (int s = 0; s < TOP; s++) {
            const int e = selected[(s+t)%TOP];
            const BOOL owned = e >= rank*E/2 && e < (rank+1)*E/2;
            ((int32_t *)data(ids[0]))[t*TOP+s] = owned ? e : -1;
            ((int32_t *)data(ids[1]))[t*TOP+s] = e;
            if (owned) for (int k = 0; k < K; k++)
                ((float *)data(mid))[t*STRIDE+s*K+k] = ((k*13+s*7+t)%97-48)/64.f;
        }
        id<MTLBuffer> out[2] = {output(dev, tokens*ROWS), output(dev, tokens*ROWS)};
        const uint64_t full_hash = hash(full), local_hash = hash(local), mid_hash = hash(mid);
        for (int arm = 0; arm < 2; arm++) {
            glm_args args = {.in_dim=K, .mid_dim=K, .out_dim=ROWS,
                .n_total_expert=E, .n_expert_used=TOP, .n_tokens=tokens,
                .mid_token_stride=STRIDE, .down_type=q6 ? 14 : 13,
                .tp_rank=arm ? rank : 0, .tp_world=arm ? 2 : 1,
                .tp_expert_base=arm ? rank*E/2 : 0,
                .down_expert_bytes=expert_bytes, .down_row_bytes=row_bytes};
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:p];
            [enc setBytes:&args length:sizeof(args) atIndex:0];
            [enc setBuffer:arm ? local : full offset:GUARD atIndex:1];
            [enc setBuffer:ids[arm] offset:GUARD atIndex:2];
            [enc setBuffer:mid offset:GUARD atIndex:3];
            [enc setBuffer:out[arm] offset:GUARD atIndex:4];
            const int rows_per_group = q6 ? 4 : 8;
            [enc dispatchThreadgroups:MTLSizeMake((ROWS+rows_per_group-1)/rows_per_group, tokens, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
            [enc endEncoding]; finish(cb);
        }
        equal_output(out[0], out[1], tokens, ROWS, ROWS);
        require(hash(full) == full_hash && hash(local) == local_hash && hash(mid) == mid_hash,
                "GLM down input mutated");
    }
}

/* Check the production map and pack kernels against token/slot identities,
 * including duplicate routes, empty experts, partial tiles and both RHS types.
 * Packing itself uses no TensorOps, so this runs on pre-M5 hardware as well. */
static void test_moe_pack(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                          id<MTLLibrary> lib, BOOL f16, int rhs_slots) {
    enum { E = 16, T = 37, TOP = 6, K = 40, PAIRS = T*TOP };
    const NSUInteger tpe_bytes = E*2*4, ids_bytes = PAIRS*4;
    const NSUInteger work_off = (tpe_bytes + ids_bytes + 7) & ~7ul;
    const NSUInteger tiles_cap = (PAIRS + 31*E + 31)/32;
    id<MTLBuffer> map = buffer(dev, work_off + 8 + tiles_cap*8);
    id<MTLBuffer> ids = buffer(dev, PAIRS*4);
    int32_t *selected = data(ids);
    for (int t = 0; t < T; t++) for (int s = 0; s < TOP; s++)
        selected[t*TOP+s] = s < 3 ? 9 : (t*7+s*3)%13;
    map_args ma = {.ne02=E, .ne10=K, .ne11=1, .ne21=T, .ne20=TOP, .nb21=TOP*4};
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline(dev, lib, @"kernel_mul_mm_id_map0_ne20_6", 2)];
    [enc setBytes:&ma length:sizeof(ma) atIndex:0];
    [enc setBuffer:ids offset:GUARD atIndex:1];
    [enc setBuffer:map offset:GUARD atIndex:2];
    [enc setBuffer:map offset:GUARD+tpe_bytes atIndex:3];
    [enc setBuffer:map offset:GUARD+work_off atIndex:4];
    [enc setThreadgroupMemoryLength:E*TOP*2 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(E, 1, 1)];
    [enc endEncoding]; finish(cb);
    uint32_t counts[E] = {0}, bases[E], route_ids[PAIRS], tile_experts[PAIRS], tile_starts[PAIRS];
    uint32_t route = 0, tiles = 0;
    const uint32_t *tpe = data(map);
    const int32_t *mapped_ids = (const int32_t *)((const char *)data(map) + tpe_bytes);
    const uint32_t *work = (const uint32_t *)((const char *)data(map) + work_off);
    for (int e = 0; e < E; e++) {
        bases[e] = route;
        for (int i = 0; i < PAIRS; i++) if (selected[i] == e) {
            counts[e]++;
            require(mapped_ids[route] == i, "map lost token/slot identity");
            route_ids[route++] = i;
        }
        require(tpe[2*e] == counts[e] && tpe[2*e+1] == bases[e], "map count/base mismatch");
        for (uint32_t r = 0; r < counts[e]; r += 32) {
            require(work[2+2*tiles] == (uint32_t)e && work[3+2*tiles] == r, "map tile mismatch");
            tile_experts[tiles] = e; tile_starts[tiles++] = r;
        }
    }
    require(route == PAIRS && work[0] == tiles, "map total mismatch");
    const NSUInteger element_bytes = f16 ? 2 : 4;
    id<MTLBuffer> src = buffer(dev, T*rhs_slots*K*element_bytes);
    for (int i = 0; i < T*rhs_slots*K; i++) {
        const float v = (i%997-498)/97.f;
        if (f16) ((_Float16 *)data(src))[i] = (_Float16)v;
        else ((float *)data(src))[i] = v;
    }
    const uint64_t src_hash = hash(src), map_hash = hash(map);
    id<MTLComputePipelineState> p = pipeline(dev, lib,
        f16 ? @"kernel_moe_pack_rhs_f16" : @"kernel_moe_pack_rhs_f32", 2);
    for (int rank = -1; rank < 2; rank++) {
        id<MTLBuffer> packed = buffer(dev, tiles_cap*32*K*2);
        mm_id_args args = {.ne00=K, .ne02=E, .ne11=rhs_slots,
            .nb10=element_bytes, .nb11=K*element_bytes, .nb12=rhs_slots*K*element_bytes,
            .ne20=TOP, .ne21=T, .tp_rank=rank < 0 ? 0 : rank, .tp_world=rank < 0 ? 1 : 2};
        cb = [queue commandBuffer]; enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:src offset:GUARD atIndex:1];
        [enc setBuffer:map offset:GUARD atIndex:2];
        [enc setBuffer:map offset:GUARD+tpe_bytes atIndex:3];
        [enc setBuffer:map offset:GUARD+work_off atIndex:4];
        [enc setBuffer:packed offset:GUARD atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(tiles_cap*32, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding]; finish(cb);
        for (NSUInteger tile = 0; tile < tiles_cap; tile++) for (int row = 0; row < 32; row++) {
            const BOOL owned = tile < tiles && (rank < 0 || tile_experts[tile]/(E/2) == (uint32_t)rank);
            for (int k = 0; k < K; k++) {
                uint16_t expected = 0xa5a5;
                if (owned) {
                    const uint32_t e = tile_experts[tile], r = tile_starts[tile]+row;
                    const uint32_t id = route_ids[bases[e] + (r < counts[e] ? r : counts[e]-1)];
                    const NSUInteger index = (id/TOP*rhs_slots + id%TOP%rhs_slots)*K+k;
                    const _Float16 value = f16 ? ((_Float16 *)data(src))[index] : (_Float16)((float *)data(src))[index];
                    memcpy(&expected, &value, sizeof(expected));
                }
                require(((uint16_t *)data(packed))[(tile*32+row)*K+k] == expected,
                        "packed RHS row/slot/tail mismatch");
            }
        }
        const uint8_t *bytes = packed.contents;
        for (int i = 0; i < GUARD; i++)
            require(bytes[i] == 0xa5 && bytes[packed.length-GUARD+i] == 0xa5, "pack guard overwritten");
    }
    require(hash(src) == src_hash && hash(map) == map_hash, "pack input mutated");
}

int main(int argc, char **argv) {
    BOOL bench = argc == 2 && strcmp(argv[1], "--bench-q8-mv") == 0;
    BOOL bench_q4 = argc == 2 && strcmp(argv[1], "--bench-q4-token-pair") == 0;
    BOOL bench_single = argc == 2 && strcmp(argv[1], "--bench-q4-single-vec") == 0;
    BOOL moe_contracts = argc == 2 && strcmp(argv[1], "--moe-contracts") == 0;
    if (argc > 1 && !bench && !bench_q4 && !bench_single && !moe_contracts) {
        fprintf(stderr, "usage: %s [--bench-q8-mv|--bench-q4-token-pair|--bench-q4-single-vec|--moe-contracts]\n", argv[0]);
        return 2;
    }
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        require(dev != nil, "no Metal device (run on a Mac with GPU access)");
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        require(queue != nil, "command queue creation failed");
        NSError *error = nil;
        NSString *source = load_metal_source();
        require(source != nil, "shader source loading failed");
        id<MTLLibrary> lib = [dev newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
        if (!lib) fprintf(stderr, "%s\n", error.description.UTF8String);
        require(lib != nil, "Metal library compilation failed");
        if (argc == 1 || moe_contracts) {
            for (int q6 = 0; q6 < 2; q6++) for (int tokens = 1; tokens <= 3; tokens += 2)
                test_glm_down_tp(dev, queue, lib, q6, tokens);
            for (int f16 = 0; f16 < 2; f16++) for (int slots = 1; slots <= SLOTS; slots += SLOTS-1)
                test_moe_pack(dev, queue, lib, f16, slots);
            printf("MoE contracts: Q5/Q6 rank-local down and compact-map F32/F16 packing passed.\n");
            if (moe_contracts) return 0;
        }
        int cases = 0;
        for (int addr = 0; addr < 2; ++addr) {
            NSString *base = addr ? @"kernel_mul_mv_addr_q4_K_pair_swiglu_f32" :
                                    @"kernel_mul_mv_slots6_q4_K_pair_swiglu_f32";
            id<MTLComputePipelineState> oldp = pipeline(dev, lib, base, 2);
            id<MTLComputePipelineState> newp = pipeline(dev, lib, [base stringByAppendingString:@"_shared_x"], 2);
            const int shapes[][3] = {{256,4,1}, {1024,32,3}, {4096,2048,1}, {7168,32,1}};
            for (unsigned s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s)
                for (int clamp = 0; clamp < 2; ++clamp) {
                    test_q4(dev, queue, oldp, newp, addr, shapes[s][0], shapes[s][1], shapes[s][2], clamp*4.f);
                    ++cases;
                }
        }
        printf("Q4: 16 bitwise cases passed (slots/address, offsets, routing, clamp).\n");
        id<MTLComputePipelineState> q4p[5] = {
            pipeline(dev, lib, @"kernel_mul_mv_q4_K_dense_f32", 2),
            pipeline(dev, lib, @"kernel_mul_mv_q4_K_dense_pair_f32", 2),
            pipeline(dev, lib, @"kernel_mul_mv_q4_K_dense_token_pair_f32", 2),
            pipeline(dev, lib, @"kernel_mul_mv_q4_K_dense_pair_token_pair_f32", 2),
            pipeline(dev, lib, @"kernel_mul_mv_q4_K_dense_single_vec_f32", 2),
        };
        const int q4_shapes[][3] = {
            {256,1,3}, {512,6,4}, {768,5,9}, {1024,32,32},
            {4096,1024,512}, {4096,512,1024}, {7168,66,34}, {256,65538,65540},
        };
        const int q4_tokens[] = {1,2,3,4,5,6,7,8};
        for (unsigned s = 0; s < sizeof(q4_shapes)/sizeof(q4_shapes[0]); ++s)
            for (unsigned t = 0; t < sizeof(q4_tokens)/sizeof(q4_tokens[0]); ++t)
                for (int padding = 0; padding <= 2; padding += 2) {
                    test_q4_dense_pair(dev, queue, q4p, q4_shapes[s][0],
                                       q4_shapes[s][1], q4_shapes[s][2], q4_tokens[t], padding,
                                       NO);
                    ++cases;
                }
        printf("Q4 dense/token pair: 128 bitwise cases passed (unequal/odd rows, strides, 1..8 tokens).\n");
        if (bench_q4) {
            for (unsigned t = 1; t < sizeof(q4_tokens)/sizeof(q4_tokens[0]); ++t) {
                test_q4_dense_pair(dev, queue, q4p, 1024, 32768, 32768, q4_tokens[t], 0, YES);
                ++cases;
            }
        }
        if (bench_single) {
            test_q4_dense_pair(dev, queue, q4p, 1024, 32768, 32768, 1, 0, 2);
            ++cases;
        }
        for (int nsg = 4; nsg <= 8; nsg += 4) for (int store = 0; store < 2; ++store) {
            NSString *base = store ? @"kernel_dsv4_shared_gate_up_swiglu_q8_0" :
                                     @"kernel_dsv4_shared_mid_swiglu_q8_0";
            id<MTLComputePipelineState> oldp = pipeline(dev, lib, base, nsg);
            id<MTLComputePipelineState> newp = pipeline(dev, lib, [base stringByAppendingString:@"_single_barrier"], nsg);
            const int shapes[][2] = {{32,2}, {256,32}, {4096,2048}, {7168,32}};
            for (unsigned s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s)
                for (int clamp = 0; clamp < 2; ++clamp) {
                    test_q8(dev, queue, oldp, newp, nsg, store, shapes[s][0], shapes[s][1], clamp*4.f);
                    ++cases;
                }
        }
        printf("Q8: 32 bitwise cases passed (4/8 simdgroups, mid-only alias, clamp, repeated dispatch).\n");
        for (int nsg = 4; nsg <= 8; nsg += 4) {
            id<MTLComputePipelineState> p[4] = {
                pipeline(dev, lib, @"kernel_mul_mv_q8_0_f32", nsg),
                pipeline(dev, lib, @"kernel_mul_mv_q8_0_f32_single_barrier", nsg),
                pipeline(dev, lib, @"kernel_mul_mv_q8_0_f32_pair", nsg),
                pipeline(dev, lib, @"kernel_mul_mv_q8_0_f32_pair_single_barrier", nsg),
            };
            const int shapes[][3] = {
                {32,2,2}, {32,1,3}, {256,3,1}, {1024,65,33},
                {4096,1024,512}, {4096,512,1024}, {7168,66,34},
                {4096,2048,2048}, {32,65538,65540},
            };
            for (unsigned s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s)
                for (int padding = 0; padding <= 2; padding += 2) {
                    projection_fixture f;
                    init_projection(&f, dev, p, nsg, shapes[s][0], shapes[s][1], shapes[s][2], padding);
                    test_projection(&f, queue);
                    ++cases;
                }
            if (bench) {
                projection_fixture f;
                init_projection(&f, dev, p, nsg, 4096, 1024, 512, 0);
                test_projection(&f, queue);
                bench_projection(&f, queue);
                if (nsg != 4) printf("NSG=%d candidate is benchmark-only; production keeps the legacy kernel.\n", nsg);
            }
        }
        printf("Q8 projections: 36 bitwise cases passed (single/pair, unequal/odd rows, independent strides).\n");
        printf("PASS: %d Metal decode kernel cases on %s; no model or SSD I/O measured.\n",
               cases, dev.name.UTF8String);
    }
    return 0;
}
