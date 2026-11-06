/* Full production Q8 GEMV against the independently frozen e37f185 kernel.
 *
 * Build:
 *   clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
 *       tests/test_metal_q8_gemv_reference.m -o /tmp/test_metal_q8_gemv_reference
 * Run from the repository root, or pass --repo PATH.
 * --compile-only creates libraries and specialized pipelines, never a queue
 * or dispatch. --dense-source FILE replaces only the candidate source.
 * --output DIR writes a JSON report; no model or model weights are required.
 * --benchmark additionally times NR0=2 against NR0=4 on Qwen projections,
 * alternating execution order and keeping NSG=4 and arithmetic unchanged.
 *
 * This supplements the reduction-only oracle. It exercises the complete
 * packed-weight dot, K walk, scale multiplication, reduction and output tail.
 * The reference below is a literal prefix of metal/dense.metal at
 * e37f18576cd0c06217b0f1745021f1fe16f665cc; never generate it from the candidate.
 * Both arms receive the same immutable buffers and NSG specialization. They
 * compile as separate libraries with identical default or safe math options.
 * Tail output counts allocate nonzero weights through the next four-row
 * boundary, matching both specializations' unconditional weight reads;
 * only valid output rows may be written. Production NR4 dispatch must require
 * rows divisible by four unless the weight allocation explicitly has padding.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1, nr0;
    int16_t r2, r3;
} mv_args;
_Static_assert(sizeof(mv_args) == 112, "production Q8 matvec argument ABI");
_Static_assert(offsetof(mv_args, nb10) == 64, "production activation stride alignment");
typedef struct { _Float16 d; int8_t qs[32]; } q8_block;
_Static_assert(sizeof(q8_block) == 34, "GGUF Q8_0 block ABI");
typedef struct { uint32_t k, rows, tokens; } shape;
static const shape kShapes[] = {
    {32, 1, 1}, {64, 3, 3}, {256, 17, 1}, {320, 33, 2},
    {640, 65, 1}, {2560, 48, 1}, {2560, 65, 3}, {6144, 17, 1},
    {10240, 320, 1}, {10240, 3, 2},
};
/* Actual wide Qwen projections, excluding the vocabulary. Keep the full
 * adversarial matrix on the small fixtures and sample these larger shapes
 * under NSG=4, the non-vocabulary decode specialization. */
static const shape kQwenShapes[] = {
    {2560, 10240, 1}, /* GDN qkv */
    {2560, 6144, 1},  /* GDN gate */
    {6144, 2560, 1},  /* GDN/attention output */
    {2560, 12288, 1}, /* Attention query + gate */
};
static const short kGroups[] = {1, 2, 4, 8};
enum { kGroupCount = sizeof(kGroups) / sizeof(kGroups[0]) };
static const uint32_t kCanary = UINT32_C(0x7f24d13b);
static const NSUInteger kOffset = 68; /* Four-byte aligned, not float4 aligned. */
static const int kPatterns = 6;

static NSString *const kPreamble =
@"#include <metal_stdlib>\nusing namespace metal;\n"
 "#define FC_MUL_MV 600\n#define N_SIMDWIDTH 32\n#define N_R0_Q8_0 2\n#define QK8_0 32\n"
 "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
 "struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n";

// Frozen prefix SHA-256: 2e55b02a9cc0bbee026b08c04826c7faa4d2297ecbef42cf163b22aa74ef195d
static NSString *const kFrozenE37 =
@"// DS4 Metal matvec kernels used by generation.\n"
 "\n"
 "constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];\n"
 "constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];\n"
 "\n"
 "struct ds4_metal_args_mul_mv {\n"
 "    int ne00;\n"
 "    int ne01;\n"
 "    int ne02;\n"
 "    ulong nb00;\n"
 "    ulong nb01;\n"
 "    ulong nb02;\n"
 "    ulong nb03;\n"
 "    int ne10;\n"
 "    int ne11;\n"
 "    int ne12;\n"
 "    ulong nb10;\n"
 "    ulong nb11;\n"
 "    ulong nb12;\n"
 "    ulong nb13;\n"
 "    int ne0;\n"
 "    int ne1;\n"
 "    int nr0;\n"
 "    short r2;\n"
 "    short r3;\n"
 "};\n"
 "\n"
 "struct ds4_metal_args_compressor_pair_store {\n"
 "    uint32_t width;\n"
 "    uint32_t ratio;\n"
 "    uint32_t pos;\n"
 "    uint32_t ape_type;\n"
 "};\n"
 "\n"
 "struct ds4_metal_args_mul_mm {\n"
 "    int32_t ne00;\n"
 "    int32_t ne02;\n"
 "    uint64_t nb01;\n"
 "    uint64_t nb02;\n"
 "    uint64_t nb03;\n"
 "    int32_t ne12;\n"
 "    uint64_t nb10;\n"
 "    uint64_t nb11;\n"
 "    uint64_t nb12;\n"
 "    uint64_t nb13;\n"
 "    int32_t ne0;\n"
 "    int32_t ne1;\n"
 "    int16_t r2;\n"
 "    int16_t r3;\n"
 "};\n"
 "\n"
 "struct ds4_metal_args_mul_mv_ext {\n"
 "    int32_t ne00;\n"
 "    int32_t ne01;\n"
 "    int32_t ne02;\n"
 "    uint64_t nb00;\n"
 "    uint64_t nb01;\n"
 "    uint64_t nb02;\n"
 "    uint64_t nb03;\n"
 "    int32_t ne10;\n"
 "    int32_t ne11;\n"
 "    int32_t ne12;\n"
 "    uint64_t nb10;\n"
 "    uint64_t nb11;\n"
 "    uint64_t nb12;\n"
 "    uint64_t nb13;\n"
 "    int32_t ne0;\n"
 "    int32_t ne1;\n"
 "    int16_t r2;\n"
 "    int16_t r3;\n"
 "};\n"
 "\n"
 "template<short NR0>\n"
 "static inline void helper_mv_reduce_and_write(\n"
 "        device float * dst_f32,\n"
 "        float sumf[NR0],\n"
 "        const int r0,\n"
 "        const int ne01,\n"
 "        ushort tiisg,\n"
 "        ushort sgitg,\n"
 "        threadgroup char * shmem) {\n"
 "    constexpr short NW = N_SIMDWIDTH;\n"
 "\n"
 "    threadgroup float * shmem_f32[NR0];\n"
 "\n"
 "    for (short row = 0; row < NR0; ++row) {\n"
 "        shmem_f32[row] = (threadgroup float *) shmem + NW*row;\n"
 "\n"
 "        if (sgitg == 0) {\n"
 "            shmem_f32[row][tiisg] = 0.0f;\n"
 "        }\n"
 "\n"
 "        sumf[row] = simd_sum(sumf[row]);\n"
 "    }\n"
 "\n"
 "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
 "\n"
 "    for (short row = 0; row < NR0; ++row) {\n"
 "        if (tiisg == 0) {\n"
 "            shmem_f32[row][sgitg] = sumf[row];\n"
 "        }\n"
 "    }\n"
 "\n"
 "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
 "\n"
 "    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {\n"
 "        float tot = simd_sum(shmem_f32[row][tiisg]);\n"
 "\n"
 "        if (tiisg == 0 && sgitg == 0) {\n"
 "            dst_f32[r0 + row] = tot;\n"
 "        }\n"
 "    }\n"
 "}\n"
 "\n"
 "template<short NR0, typename args_t>\n"
 "void kernel_mul_mv_q8_0_f32_impl(\n"
 "        args_t args,\n"
 "        device const char * src0,\n"
 "        device const char * src1,\n"
 "        device       char * dst,\n"
 "        threadgroup  char * shmem,\n"
 "        uint3  tgpig,\n"
 "        ushort tiisg,\n"
 "        ushort sgitg) {\n"
 "    const short NSG = FC_mul_mv_nsg;\n"
 "\n"
 "    constexpr short NW = N_SIMDWIDTH;\n"
 "    constexpr short NQ = 8;\n"
 "\n"
 "    const int nb = args.ne00/QK8_0;\n"
 "\n"
 "    const int r0 = tgpig.x*NR0;\n"
 "    const int r1 = tgpig.y;\n"
 "    const int im = tgpig.z;\n"
 "\n"
 "    const uint i12 = im%args.ne12;\n"
 "    const uint i13 = im/args.ne12;\n"
 "\n"
 "    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;\n"
 "\n"
 "    device const float * y = (device const float *) (src1 + offset1);\n"
 "\n"
 "    device const block_q8_0 * ax[NR0];\n"
 "    FOR_UNROLL (short row = 0; row < NR0; ++row) {\n"
 "        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;\n"
 "\n"
 "        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);\n"
 "    }\n"
 "\n"
 "    float sumf[NR0] = { 0.f };\n"
 "\n"
 "    const short ix = tiisg/(NW/NQ);\n"
 "    const short il = tiisg%(NW/NQ);\n"
 "\n"
 "    const int ib0 = sgitg*NQ + ix;\n"
 "\n"
 "    float yl[NQ];\n"
 "\n"
 "    device const float * yb = y + ib0*QK8_0 + il*NQ;\n"
 "\n"
 "    for (int ib = ib0; ib < nb; ib += NSG*NQ) {\n"
 "        for (short i = 0; i < NQ; ++i) {\n"
 "            yl[i] = yb[i];\n"
 "        }\n"
 "\n"
 "        for (short row = 0; row < NR0; row++) {\n"
 "            device const int8_t * qs = ax[row][ib].qs + il*NQ;\n"
 "\n"
 "            float sumq = 0.f;\n"
 "            FOR_UNROLL (short i = 0; i < NQ; ++i) {\n"
 "                sumq += qs[i] * yl[i];\n"
 "            }\n"
 "\n"
 "            sumf[row] += sumq*ax[row][ib].d;\n"
 "        }\n"
 "\n"
 "        yb += NSG*NQ*QK8_0;\n"
 "    }\n"
 "\n"
 "    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;\n"
 "\n"
 "    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);\n"
 "}\n"
 "\n"
 "// Decode-time Q8_0 matrix-vector multiply. DS4 uses this for Q8_0 dense\n"
 "// projections such as shared experts and output-side small matvecs.\n"
 "[[host_name(\"kernel_mul_mv_q8_0_f32\")]]\n"
 "kernel void kernel_mul_mv_q8_0_f32(\n"
 "        constant ds4_metal_args_mul_mv & args,\n"
 "        device const char * src0,\n"
 "        device const char * src1,\n"
 "        device       char * dst,\n"
 "        threadgroup  char * shmem [[threadgroup(0)]],\n"
 "        uint3  tgpig[[threadgroup_position_in_grid]],\n"
 "        ushort tiisg[[thread_index_in_simdgroup]],\n"
 "        ushort sgitg[[simdgroup_index_in_threadgroup]]) {\n"
 "    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);\n"
 "}\n"
 "\n";

static void fail(NSString *message) {
    fprintf(stderr, "FAIL Q8 GEMV reference: %s\n", message.UTF8String);
    exit(1);
}

static NSString *candidate_source(NSString *repo, NSString *override) {
    NSString *path = override ?: [repo stringByAppendingPathComponent:@"metal/dense.metal"];
    NSError *error = nil;
    NSString *text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (!text) fail([NSString stringWithFormat:@"cannot read %@: %@", path, error]);
    /* Select only the production plain Q8 implementation and its NR2/NR4
     * entry points. The adjacent BF16/RoPE/HC variants have their own
     * arithmetic and need not become dependencies of this frozen oracle. */
    NSString *declarationsEnd = @"template<short NR0, bool ROUND = false>\nstatic inline void helper_mv_reduce_and_write(";
    NSString *plainBegin = @"template<short NR0>\nstatic inline void helper_q8_mv_reduce_and_write(";
    NSString *plainImpl = @"template<short NR0, typename args_t>\nvoid kernel_mul_mv_q8_0_f32_plain_impl(";
    NSString *plainEnd = @"// Rope folded into a matvec:";
    NSString *kernelsBegin = @"// Decode-time Q8_0 matrix-vector multiply.";
    NSString *kernelsEnd = @"[[host_name(\"kernel_mul_mv_q8_0_f32_bf16\")]]";
    for (NSString *anchor in @[declarationsEnd, plainBegin, plainImpl, plainEnd,
                               kernelsBegin, kernelsEnd,
                               @"[[host_name(\"kernel_mul_mv_q8_0_f32\")]]",
                               @"[[host_name(\"kernel_mul_mv_q8_0_f32_nr4\")]]"]) {
        if ([text componentsSeparatedByString:anchor].count != 2)
            fail(@"candidate source anchors missing/non-unique; update extraction explicitly");
    }
    const NSUInteger declarationsStop = [text rangeOfString:declarationsEnd].location;
    const NSUInteger plainStart = [text rangeOfString:plainBegin].location;
    const NSUInteger implStart = [text rangeOfString:plainImpl].location;
    const NSUInteger plainStop = [text rangeOfString:plainEnd].location;
    const NSUInteger kernelsStart = [text rangeOfString:kernelsBegin].location;
    const NSUInteger kernelsStop = [text rangeOfString:kernelsEnd].location;
    if (!(declarationsStop < plainStart && plainStart < implStart && implStart < plainStop &&
          plainStop < kernelsStart && kernelsStart < kernelsStop))
        fail(@"invalid candidate declarations/plain/kernel source ranges");
    return [NSString stringWithFormat:@"%@%@%@",
        [text substringToIndex:declarationsStop],
        [text substringWithRange:NSMakeRange(plainStart, plainStop - plainStart)],
        [text substringWithRange:NSMakeRange(kernelsStart, kernelsStop - kernelsStart)]];
}

static id<MTLLibrary> make_library(id<MTLDevice> device, NSString *body, bool safe) {
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
    id<MTLLibrary> lib = [device newLibraryWithSource:[kPreamble stringByAppendingString:body]
                                             options:options error:&error];
    if (!lib) fail([NSString stringWithFormat:@"Metal compile (%@): %@", safe ? @"safe" : @"default", error]);
    return lib;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, short nsg, short nr0) {
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    [values setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    NSError *error = nil;
    NSString *name = nr0 == 4 ? @"kernel_mul_mv_q8_0_f32_nr4" : @"kernel_mul_mv_q8_0_f32";
    id<MTLFunction> fn = [library newFunctionWithName:name constantValues:values error:&error];
    id<MTLComputePipelineState> pipeline = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
    if (!pipeline) fail([NSString stringWithFormat:@"Metal pipeline NR0=%d NSG=%d: %@", nr0, nsg, error]);
    if (pipeline.threadExecutionWidth != 32 || pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)(32*nsg))
        fail(@"device cannot execute the required Q8 SIMD geometry");
    return pipeline;
}

static uint32_t mix_bits(uint32_t x) {
    x ^= x >> 16; x *= UINT32_C(0x7feb352d);
    x ^= x >> 15; x *= UINT32_C(0x846ca68b);
    return x ^ (x >> 16);
}

static id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes) {
    bytes = (bytes + 3) / 4 * 4;
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes + 2*kOffset options:MTLResourceStorageModeShared];
    if (!buffer) fail(@"buffer allocation");
    uint32_t *p = buffer.contents;
    for (size_t i = 0; i < buffer.length / 4; ++i) p[i] = kCanary;
    return buffer;
}

static void check_output_guards(id<MTLBuffer> buffer, size_t n) {
    const uint32_t *words = buffer.contents;
    for (size_t i = 0; i < buffer.length / 4; ++i) {
        if ((i < kOffset/4 || i >= kOffset/4 + n) && words[i] != kCanary)
            fail(@"output prefix, suffix or odd-row tail overwritten");
    }
}

static void fixture(id<MTLBuffer> weights, id<MTLBuffer> input, shape s, int pattern) {
    q8_block *w = (q8_block *)((char *)weights.contents + kOffset);
    float *x = (float *)((char *)input.contents + kOffset);
    const uint32_t blocks = s.k / 32, physical_rows = (s.rows + 3u) & ~3u;
    for (uint32_t r = 0; r < physical_rows; ++r) for (uint32_t b = 0; b < blocks; ++b) {
        q8_block *p = &w[(uint64_t)r*blocks + b];
        const uint32_t random = mix_bits(r*65537u + b*313u + 471u);
        p->d = (_Float16)ldexpf(0.75f + (random % 997u)/997.0f, (int)(random % 9u) - 10);
        if (pattern == 0) p->d = (_Float16)0.5f;
        if (pattern == 3) p->d = (_Float16)ldexpf(1.f + (random % 113u)/127.f, (int)(b % 20u) - 14);
        if (pattern == 4) p->d = (_Float16)(b % 2 ? -0.0f : 0.0f);
        for (uint32_t q = 0; q < 32; ++q) {
            p->qs[q] = (int8_t)((int)(mix_bits(random + q*811u) % 256u) - 128);
            if (pattern == 0 || pattern == 2 || pattern == 3) p->qs[q] = (int8_t)(1 + r % 7);
            if (pattern == 5) p->qs[q] = q % 2 ? INT8_MIN : INT8_MAX;
        }
    }
    for (uint32_t t = 0; t < s.tokens; ++t) for (uint32_t k = 0; k < s.k; ++k) {
        const uint32_t random = mix_bits(t*65537u + k*7919u + 123u);
        float v = ((int)(random % 65521u) - 32760) / 16384.0f + 0.00000017f;
        if (pattern == 0) v = 0.125f;
        if (pattern == 1 && k % 11u == 0u) v *= 16.f; /* Model-like outliers. */
        if (pattern == 2) {
            const float cancellation[] = {0x1p20f, 0.06250001f, -0x1p20f, 1.00000012f,
                                          -0x1p18f, -0.25000003f, 0x1p18f, -0.50000006f};
            v = cancellation[k % 8u] * (t + 1u);
        }
        if (pattern == 3) v = (k / 32u % 2u ? -1.f : 1.f) * (1.f + (k % 31u)/31.f);
        if (pattern == 4) v = (k % 2 ? -1.f : 1.f) * (k % 5 ? 0.f : 0x1p-140f);
        if (pattern == 5) v = ldexpf(v, (int)(k % 25u) - 12);
        x[(uint64_t)t*s.k + k] = v;
    }
}

static void encode(id<MTLCommandBuffer> command, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> weights, id<MTLBuffer> input, id<MTLBuffer> output,
                   shape s, short nsg, short nr0, uint32_t iterations) {
    const uint64_t row_bytes = (uint64_t)(s.k / 32u)*34u;
    mv_args args = {
        .ne00 = s.k, .ne01 = s.rows, .ne02 = 1,
        .nb00 = 34, .nb01 = row_bytes, .nb02 = row_bytes*s.rows, .nb03 = row_bytes*s.rows,
        .ne10 = s.k, .ne11 = s.tokens, .ne12 = 1,
        .nb10 = 4, .nb11 = (uint64_t)s.k*4u,
        .nb12 = (uint64_t)s.k*s.tokens*4u, .nb13 = (uint64_t)s.k*s.tokens*4u,
        .ne0 = s.rows, .ne1 = s.tokens, .nr0 = nr0, .r2 = 1, .r3 = 1,
    };
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) fail(@"command encoder allocation");
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:&args length:sizeof(args) atIndex:0];
    [encoder setBuffer:weights offset:kOffset atIndex:1];
    [encoder setBuffer:input offset:kOffset atIndex:2];
    [encoder setBuffer:output offset:kOffset atIndex:3];
    [encoder setThreadgroupMemoryLength:(NSUInteger)nr0*32u*sizeof(float) atIndex:0];
    for (uint32_t i = 0; i < iterations; ++i) {
        [encoder dispatchThreadgroups:MTLSizeMake((s.rows + nr0 - 1u)/nr0, s.tokens, 1)
                 threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    }
    [encoder endEncoding];
}

static NSDictionary *run_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
                             id<MTLComputePipelineState> reference, id<MTLComputePipelineState> candidate,
                             shape s, short nsg, short nr0, bool safe, int pattern) {
    const uint64_t n = (uint64_t)s.rows*s.tokens;
    const uint32_t physical_rows = (s.rows + 3u) & ~3u;
    id<MTLBuffer> weights = make_buffer(device, (uint64_t)physical_rows*(s.k/32u)*sizeof(q8_block));
    id<MTLBuffer> input = make_buffer(device, (uint64_t)s.k*s.tokens*sizeof(float));
    id<MTLBuffer> expected = make_buffer(device, n*sizeof(float)), actual = make_buffer(device, n*sizeof(float));
    fixture(weights, input, s, pattern);
    NSData *weight_snapshot = [NSData dataWithBytes:weights.contents length:weights.length];
    NSData *input_snapshot = [NSData dataWithBytes:input.contents length:input.length];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!command) fail(@"command buffer allocation");
    encode(command, reference, weights, input, expected, s, nsg, 2, 1);
    encode(command, candidate, weights, input, actual, s, nsg, nr0, 1);
    [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        fail([NSString stringWithFormat:@"GPU execution: %@", command.error]);
    if (memcmp(weights.contents, weight_snapshot.bytes, weights.length) ||
        memcmp(input.contents, input_snapshot.bytes, input.length)) fail(@"input or weight buffer changed");
    check_output_guards(expected, n); check_output_guards(actual, n);
    const uint32_t *want = (const uint32_t *)((const char *)expected.contents + kOffset);
    const uint32_t *got = (const uint32_t *)((const char *)actual.contents + kOffset);
    const float *ef = (const float *)want, *af = (const float *)got;
    uint64_t mismatches = 0;
    double max_abs = 0, sum_sq = 0;
    for (uint64_t i = 0; i < n; ++i) {
        if (!isfinite(ef[i]) || !isfinite(af[i])) fail(@"finite fixture produced non-finite output");
        if (want[i] != got[i]) {
            if (mismatches++ == 0) fprintf(stderr,
                "DIFF %s NR0=%d NSG=%d K=%u rows=%u T=%u pattern=%d at=%llu e37=%08x (%.9g) current=%08x (%.9g)\n",
                safe ? "safe" : "default", nr0, nsg, s.k, s.rows, s.tokens, pattern,
                (unsigned long long)i, want[i], ef[i], got[i], af[i]);
        }
        double delta = fabs((double)ef[i] - af[i]);
        if (delta > max_abs) max_abs = delta;
        sum_sq += delta*delta;
        if (pattern == 0) {
            const float exact = (float)s.k * (1u + (i % s.rows) % 7u) / 16.f;
            if (ef[i] != exact || af[i] != exact) fail(@"independent exact integer-valued GEMV oracle failed");
        }
    }
    /* Independent double dot samples prevent two kernels with the same ABI
     * mistake from passing. This is a conservative FP32 rounding bound; exact
     * reference parity above, not the bound, decides the test result. */
    const q8_block *w = (const q8_block *)((const char *)weights.contents + kOffset);
    const float *x = (const float *)((const char *)input.contents + kOffset);
    for (uint32_t t = 0; t < s.tokens; ++t) for (uint32_t r = 0; r < s.rows; r += s.rows > 7 ? s.rows/7 : 1) {
        double dot = 0, absolute = 0;
        for (uint32_t k = 0; k < s.k; ++k) {
            const q8_block *b = &w[(uint64_t)r*(s.k/32u) + k/32u];
            double term = (double)b->d * b->qs[k%32u] * x[(uint64_t)t*s.k + k];
            dot += term; absolute += fabs(term);
        }
        const double bound = 4.0*FLT_EPSILON*(32u + s.k/(32u*nsg))*absolute + 1e-8;
        if (fabs(ef[(uint64_t)t*s.rows + r] - dot) > bound ||
            fabs(af[(uint64_t)t*s.rows + r] - dot) > bound) fail(@"independent double dot exceeds rounding bound");
    }
    return @{@"math":safe ? @"safe" : @"default", @"nsg":@(nsg), @"nr0":@(nr0), @"k":@(s.k), @"rows":@(s.rows),
             @"tokens":@(s.tokens), @"pattern":@(pattern), @"values":@(n), @"different_bits":@(mismatches),
             @"max_abs":@(max_abs), @"rmse":@(sqrt(sum_sq/n))};
}

static double bench_sample(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> weights, id<MTLBuffer> input, id<MTLBuffer> output,
                           shape s, short nr0, uint32_t iterations) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    if (!command) fail(@"benchmark command buffer allocation");
    encode(command, pipeline, weights, input, output, s, 4, nr0, iterations);
    [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
        fail([NSString stringWithFormat:@"benchmark GPU execution: %@", command.error]);
    const double elapsed = command.GPUEndTime - command.GPUStartTime;
    if (!(elapsed > 0.0) || !isfinite(elapsed)) fail(@"GPU timestamps unavailable");
    return elapsed * 1e6 / iterations;
}

static double median(NSArray<NSNumber *> *values) {
    NSArray<NSNumber *> *sorted = [values sortedArrayUsingSelector:@selector(compare:)];
    const NSUInteger n = sorted.count;
    return n % 2u ? sorted[n/2u].doubleValue :
        0.5 * (sorted[n/2u - 1u].doubleValue + sorted[n/2u].doubleValue);
}

static NSDictionary *benchmark_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                    id<MTLComputePipelineState> nr2, id<MTLComputePipelineState> nr4,
                                    shape s, uint32_t rounds, uint32_t iterations) {
    const uint32_t physical_rows = (s.rows + 3u) & ~3u;
    id<MTLBuffer> weights = make_buffer(device, (uint64_t)physical_rows*(s.k/32u)*sizeof(q8_block));
    id<MTLBuffer> input = make_buffer(device, (uint64_t)s.k*sizeof(float));
    id<MTLBuffer> out2 = make_buffer(device, (uint64_t)s.rows*sizeof(float));
    id<MTLBuffer> out4 = make_buffer(device, (uint64_t)s.rows*sizeof(float));
    fixture(weights, input, s, 1);
    /* Warm both pipelines and allocations before timing. Repeated dispatches
     * intentionally measure warm resident dense projections, not SSD reads. */
    (void)bench_sample(queue, nr2, weights, input, out2, s, 2, 8);
    (void)bench_sample(queue, nr4, weights, input, out4, s, 4, 8);
    NSMutableArray<NSNumber *> *times2 = [NSMutableArray new], *times4 = [NSMutableArray new];
    NSMutableArray<NSNumber *> *ratios = [NSMutableArray new];
    for (uint32_t round = 0; round < rounds; ++round) @autoreleasepool {
        double t2, t4;
        if (round % 2u == 0u) {
            t2 = bench_sample(queue, nr2, weights, input, out2, s, 2, iterations);
            t4 = bench_sample(queue, nr4, weights, input, out4, s, 4, iterations);
        } else {
            t4 = bench_sample(queue, nr4, weights, input, out4, s, 4, iterations);
            t2 = bench_sample(queue, nr2, weights, input, out2, s, 2, iterations);
        }
        [times2 addObject:@(t2)]; [times4 addObject:@(t4)]; [ratios addObject:@(t2/t4)];
    }
    check_output_guards(out2, s.rows); check_output_guards(out4, s.rows);
    const bool exact = memcmp((char *)out2.contents + kOffset, (char *)out4.contents + kOffset,
                              (size_t)s.rows*sizeof(float)) == 0;
    if (!exact) fail(@"benchmark NR2/NR4 output mismatch");
    const double time2 = median(times2), time4 = median(times4), speedup = median(ratios);
    printf("BENCH Q8 NSG=4 K=%u rows=%u: NR2 %.3f us, NR4 %.3f us, paired median %.4fx\n",
           s.k, s.rows, time2, time4, speedup);
    return @{@"k":@(s.k), @"rows":@(s.rows), @"tokens":@1, @"nsg":@4,
             @"rounds":@(rounds), @"iterations_per_sample":@(iterations),
             @"nr2_us":times2, @"nr4_us":times4, @"paired_speedup":ratios,
             @"median_nr2_us":@(time2), @"median_nr4_us":@(time4),
             @"median_paired_speedup":@(speedup), @"bit_exact":@(exact)};
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        bool compile_only = false, benchmark = false;
        uint32_t rounds = 9, iterations = 32;
        NSString *repo = @".", *override = nil, *output = nil;
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "--compile-only")) compile_only = true;
            else if (!strcmp(argv[i], "--test")) compile_only = false;
            else if (!strcmp(argv[i], "--repo") && i + 1 < argc) repo = @(argv[++i]);
            else if (!strcmp(argv[i], "--dense-source") && i + 1 < argc) override = @(argv[++i]);
            else if (!strcmp(argv[i], "--output") && i + 1 < argc) output = @(argv[++i]);
            else if (!strcmp(argv[i], "--benchmark")) benchmark = true;
            else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (!strcmp(argv[i], "--iterations") && i + 1 < argc) iterations = (uint32_t)strtoul(argv[++i], NULL, 10);
            else {
                fprintf(stderr, "Usage: %s [--test|--compile-only] [--repo PATH] [--dense-source FILE] [--output DIR] [--benchmark] [--rounds N] [--iterations N]\n", argv[0]);
                return 2;
            }
        }
        if (!rounds || rounds > 1000u || !iterations || iterations > 10000u) {
            fprintf(stderr, "rounds must be 1..1000; iterations must be 1..10000\n");
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) fail(@"Metal device unavailable");
        NSString *candidate = candidate_source(repo, override);
        NSMutableArray<id<MTLComputePipelineState>> *pipelines = [NSMutableArray new];
        for (int mode = 0; mode < 2; ++mode) {
            id<MTLLibrary> legacy = make_library(device, kFrozenE37, mode != 0);
            id<MTLLibrary> current = make_library(device, candidate, mode != 0);
            for (size_t group = 0; group < kGroupCount; ++group) {
                [pipelines addObject:make_pipeline(device, legacy, kGroups[group], 2)];
                [pipelines addObject:make_pipeline(device, current, kGroups[group], 2)];
                [pipelines addObject:make_pipeline(device, current, kGroups[group], 4)];
            }
            printf("PASS compile %s: frozen e37 NR2 and candidate NR2/NR4, NSG=1/2/4/8\n", mode ? "safe" : "default");
        }
        if (compile_only) {
            printf("PASS compile-only: %lu pipelines; no GPU queue, buffers or dispatches created\n",
                   (unsigned long)pipelines.count);
            return 0;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) fail(@"GPU command queue allocation");
        NSMutableArray<NSDictionary *> *results = [NSMutableArray new];
        uint64_t mismatches = 0, values = 0;
        for (int mode = 0; mode < 2; ++mode) for (size_t group = 0; group < kGroupCount; ++group) {
            const NSUInteger pi = (mode*kGroupCount + group)*3;
            for (short nr0 = 2; nr0 <= 4; nr0 += 2)
            for (size_t shape_id = 0; shape_id < sizeof(kShapes)/sizeof(kShapes[0]); ++shape_id)
                for (int pattern = 0; pattern < kPatterns; ++pattern) @autoreleasepool {
                    NSDictionary *result = run_case(device, queue, pipelines[pi], pipelines[pi + nr0/2],
                                                   kShapes[shape_id], kGroups[group], nr0, mode != 0, pattern);
                    [results addObject:result];
                    mismatches += [result[@"different_bits"] unsignedLongLongValue];
                    values += [result[@"values"] unsignedLongLongValue];
                }
            if (kGroups[group] == 4) {
                const int patterns[] = {0, 1, 2, 5};
                for (short nr0 = 2; nr0 <= 4; nr0 += 2)
                for (size_t shape_id = 0; shape_id < sizeof(kQwenShapes)/sizeof(kQwenShapes[0]); ++shape_id)
                    for (size_t p = 0; p < sizeof(patterns)/sizeof(patterns[0]); ++p) @autoreleasepool {
                        NSDictionary *result = run_case(device, queue, pipelines[pi], pipelines[pi + nr0/2],
                                                       kQwenShapes[shape_id], 4, nr0, mode != 0, patterns[p]);
                        [results addObject:result];
                        mismatches += [result[@"different_bits"] unsignedLongLongValue];
                        values += [result[@"values"] unsignedLongLongValue];
                    }
            }
        }
        NSMutableArray<NSDictionary *> *benchmarks = [NSMutableArray new];
        if (benchmark && mismatches == 0) {
            const NSUInteger pi = 2u * 3u; /* default math, NSG=4 */
            for (size_t shape_id = 0; shape_id < sizeof(kQwenShapes)/sizeof(kQwenShapes[0]); ++shape_id) @autoreleasepool {
                [benchmarks addObject:benchmark_case(device, queue, pipelines[pi + 1], pipelines[pi + 2],
                                                     kQwenShapes[shape_id], rounds, iterations)];
            }
        }
        NSDictionary *report = @{@"reference":@"e37f18576cd0c06217b0f1745021f1fe16f665cc", @"device":device.name,
                                 @"cases":@(results.count), @"values":@(values), @"different_bits":@(mismatches),
                                 @"passed":@(mismatches == 0), @"results":results, @"benchmarks":benchmarks,
                                 @"activation_offset_bytes":@(kOffset),
                                 @"benchmark_scope":@"warm resident dense weights; GPU command timestamps; alternating NR2/NR4 order"};
        if (output) {
            NSError *error = nil;
            if (![[NSFileManager defaultManager] createDirectoryAtPath:output withIntermediateDirectories:YES attributes:nil error:&error])
                fail([NSString stringWithFormat:@"report directory: %@", error]);
            NSData *data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
            if (!data || ![data writeToFile:[output stringByAppendingPathComponent:@"q8-gemv-reference.json"] options:NSDataWritingAtomic error:&error])
                fail([NSString stringWithFormat:@"report write: %@", error]);
        }
        printf("%s full Q8 GEMV: %lu fixtures, %llu values, %llu bit differences; input immutability and output guards checked\n",
               mismatches ? "FAIL" : "PASS", (unsigned long)results.count,
               (unsigned long long)values, (unsigned long long)mismatches);
        return mismatches ? 1 : 0;
    }
}
