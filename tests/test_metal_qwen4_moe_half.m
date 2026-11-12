/* Simdgroup MoE half-storage regression and benchmark, independent of host
 * dispatch policy. IQ2XXS gate/up and padded Q2_K F640 down use production
 * kernels. A separately compiled --baseline-source freezes the float oracle.
 * Build: clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
 *   tests/test_metal_qwen4_moe_half.m -o /tmp/test_metal_qwen4_moe_half
 * Run from the repository root. --compile-only creates pipelines but no queue
 * or GPU dispatch. --bench includes x conversion and the dual mid write in
 * every candidate timing; alternating ABBA/BAAB rounds share the same fixture.
 * --q4-k32 adds Q4_K/MXFP4 float and half K32 NT4 kernels, while checking
 * that every existing K64 entry point still matches the frozen baseline.
 * K32 binds quantization constants as the runtime does.
 * Generic weight types (function constant 900 = 0) match M1 large-prefill
 * production dispatch. --specialized-types retains the specialized comparison.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n_tokens, n_slots, n_out, in_dim, out_rows, weight_type, row_bytes, list_cap;
    uint64_t expert_bytes;
    uint32_t n_expert, tiles_per_launch, tail_base, expert_major, n_active_expert;
    uint32_t active_expert[512];
} mm_args;
_Static_assert(sizeof(mm_args) == 2112, "production MoE MM ABI");
_Static_assert(offsetof(mm_args, active_expert) == 60, "production active expert ABI");
static const NSUInteger kOffset = 256;
static const unsigned kNT[] = {1, 2, 4, 8};
static uint64_t comparisons, half_comparisons, fixtures;
static bool specialized_types, q4_k32;

static void fail(NSString *message) {
    fprintf(stderr, "FAIL Qwen MoE half: %s\n", message.UTF8String);
    exit(1);
}
static NSString *read_source(NSString *path) {
    NSError *error = nil;
    NSString *s = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (!s) fail([NSString stringWithFormat:@"read %@: %@", path, error]);
    return s;
}
static NSString *between(NSString *s, NSString *begin, NSString *end) {
    if ([s componentsSeparatedByString:begin].count != 2 ||
        [s componentsSeparatedByString:end].count != 2) fail(@"source anchor missing/nonunique");
    NSRange a = [s rangeOfString:begin], b = [s rangeOfString:end];
    if (b.location <= a.location) fail(@"invalid source range");
    return [s substringWithRange:NSMakeRange(a.location, b.location - a.location)];
}
static NSString *production_source(NSString *repo, NSString *qwenPath) {
    NSString *q = read_source(qwenPath);
    NSString *m = read_source([repo stringByAppendingPathComponent:@"metal/moe.metal"]);
    NSMutableString *s = [NSMutableString stringWithString:
        @"#include <metal_stdlib>\nusing namespace metal;\n#define QWEN4_ROUTER_MAX_EXPERT 512\n"];
    [s appendString:between(m, @"static constant float ds4_metal_mxfp4_values[16]", @"// BEGIN GENERATED MXFP4 HALF LUT")];
    [s appendString:between(m, @"static constant uchar ds4_metal_ksigns_iq2xs[128]", @"#define kmask_iq2xs")];
    [s appendString:between(q, @"static inline float qwen4_sigmoid(", @"static inline float qwen4_softplus(")];
    [s appendString:between(q, @"static inline float qwen4_silu(", @"/* --- hyper-connections")];
    [s appendString:between(q, @"constant bool qwen4_expert_addresses ", @"#define QWEN4_MOE_NSG")];
    [s appendString:between(q, @"struct ds4_metal_args_qwen4_moe_mm {", @"#ifdef DS4_METAL_HAS_TENSOR")];
    return s;
}
static uint64_t source_hash(NSString *s) {
    NSData *data = [s dataUsingEncoding:NSUTF8StringEncoding];
    const uint8_t *p = data.bytes;
    uint64_t h = UINT64_C(14695981039346656037);
    for (NSUInteger i = 0; i < data.length; i++) h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= UINT32_C(0x7feb352d);
    x ^= x >> 15; x *= UINT32_C(0x846ca68b);
    return x ^ (x >> 16);
}
static void *data(id<MTLBuffer> b) { return (char *)b.contents + kOffset; }
static id<MTLBuffer> buffer(id<MTLDevice> device, size_t bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:bytes + 2 * kOffset options:MTLResourceStorageModeShared];
    if (!b) fail([NSString stringWithFormat:@"allocate %zu bytes", bytes]);
    memset(b.contents, 0xA5, b.length);
    return b;
}
static void guard(id<MTLBuffer> b) {
    const uint8_t *p = b.contents;
    for (NSUInteger i = 0; i < kOffset; i++)
        if (p[i] != 0xA5 || p[b.length - 1 - i] != 0xA5) fail(@"buffer guard overwritten");
}
static void fill_weights(id<MTLBuffer> b, unsigned type, unsigned seed) {
    uint8_t *p = data(b);
    const size_t bytes = b.length - 2 * kOffset;
    for (size_t i = 0; i < bytes; i++) p[i] = (uint8_t)mix((uint32_t)i + seed);
    if (type == 39u) {
        for (size_t i = 0; i < bytes; i += 17u) p[i] = 116u + mix((uint32_t)i + seed) % 5u;
        return;
    }
    const unsigned block = type == 12 ? 144 : type == 16 ? 66 : 84;
    for (size_t i = 0; i < bytes; i += block) {
        /* Small normal scales keep the complete fixture finite. Alternating
         * signs and very different magnitudes still exercise cancellation. */
        uint16_t h = (uint16_t)(0x1000u | (mix((uint32_t)i + seed) & 0x83ffu));
        memcpy(p + i + (type == 10 ? 80 : 0), &h, 2);
        if (type == 10 || type == 12) { h ^= 0x0195u; memcpy(p + i + (type == 10 ? 82 : 2), &h, 2); }
    }
}

@interface Fixture : NSObject
@property unsigned tokens, dim, ff, experts, slots;
@property mm_args midArgs, downArgs;
@property(strong) id<MTLBuffer> gate, up, down, lists, counts, x, xHalf;
@property(strong) id<MTLBuffer> refMid, gotMid, midHalf, refPart, gotPart;
@end
@implementation Fixture
@end

static Fixture *fixture(id<MTLDevice> device, unsigned tokens, unsigned dim,
                        unsigned experts, bool sparse, bool expertMajor) {
    Fixture *f = [Fixture new];
    f.tokens = tokens; f.dim = dim; f.ff = 640; f.experts = experts; f.slots = 10;
    mm_args m = {.n_tokens=tokens, .n_slots=f.slots, .n_out=f.slots,
        .in_dim=dim, .out_rows=f.ff, .weight_type=q4_k32?12u:16u, .row_bytes=dim/256*(q4_k32?144u:66u),
        .list_cap=tokens, .n_expert=experts, .tiles_per_launch=MIN(8u,(tokens+31u)/32u),
        .expert_major=expertMajor};
    m.expert_bytes = (uint64_t)m.row_bytes * m.out_rows;
    mm_args d = m;
    d.in_dim = f.ff; d.out_rows = dim + (q4_k32 && sparse ? 4u : 0u);
    d.weight_type = q4_k32 ? 39u : 10u; d.row_bytes = q4_k32 ? f.ff/32u*17u : 252u;
    d.expert_bytes = (uint64_t)d.row_bytes * d.out_rows;
    if (sparse) {
        m.n_active_expert = d.n_active_expert = experts - 2;
        for (unsigned e = 0; e < m.n_active_expert; e++) m.active_expert[e] = d.active_expert[e] = e;
    }
    f.midArgs = m; f.downArgs = d;
    f.gate = buffer(device, m.expert_bytes * experts); f.up = buffer(device, m.expert_bytes * experts);
    f.down = buffer(device, d.expert_bytes * experts);
    fill_weights(f.gate, m.weight_type, 17); fill_weights(f.up, m.weight_type, 59); fill_weights(f.down, d.weight_type, 117);
    f.lists = buffer(device, (size_t)experts * tokens * sizeof(int32_t));
    f.counts = buffer(device, experts * sizeof(int32_t));
    int32_t *lists = data(f.lists), *counts = data(f.counts);
    memset(counts, 0, experts * sizeof(int32_t));
    const unsigned active = sparse ? experts - 2 : experts;
    for (unsigned t = 0; t < tokens; t++) for (unsigned slot = 0; slot < f.slots; slot++) {
        /* Distinct slots per token: consecutive IDs avoid duplicate expert
         * selection even in the small 11/13-expert regression fixtures. */
        unsigned e = (t * 7 + slot) % active;
        lists[(size_t)e * tokens + (unsigned)counts[e]++] = (int32_t)(t * f.slots + slot);
    }
    f.x = buffer(device, (size_t)tokens * dim * 4); f.xHalf = buffer(device, (size_t)tokens * dim * 2);
    float *x = data(f.x);
    for (size_t i = 0; i < (size_t)tokens * dim; i++)
        x[i] = ((int)(mix((uint32_t)i + 31) % 16385) - 8192) / 8192.f;
    const size_t nm = (size_t)tokens * f.slots * f.ff, np = (size_t)tokens * f.slots * d.out_rows;
    f.refMid = buffer(device, nm * 4); f.gotMid = buffer(device, nm * 4); f.midHalf = buffer(device, nm * 2);
    f.refPart = buffer(device, np * 4); f.gotPart = buffer(device, np * 4);
    return f;
}

static id<MTLLibrary> library(id<MTLDevice> device, NSString *source, bool safe) {
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
    return lib;
}
static id<MTLComputePipelineState> pipeline_with_type(id<MTLDevice> device, id<MTLLibrary> lib,
                                                     NSString *name, unsigned bound_type) {
    NSError *error = nil;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    unsigned tail = 0; bool addressed = false;
    [constants setConstantValue:&bound_type type:MTLDataTypeUInt atIndex:900];
    [constants setConstantValue:&tail type:MTLDataTypeUInt atIndex:905];
    [constants setConstantValue:&addressed type:MTLDataTypeBool atIndex:906];
    id<MTLFunction> fn = [lib newFunctionWithName:name constantValues:constants error:&error];
    if (!fn) fail([NSString stringWithFormat:@"%@: %@", name, error]);
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!p) fail(error.description);
    if (p.threadExecutionWidth != 32 || p.maxTotalThreadsPerThreadgroup < 128) fail(@"unsupported execution geometry");
    return p;
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> lib,
                                           NSString *name, unsigned type) {
    return pipeline_with_type(device, lib, name, specialized_types ? type : 0u);
}
static NSString *kernel_name(bool mid, bool half, unsigned nt) {
    NSString *kind = mid ? @"mid" : @"down";
    if (half) return [NSString stringWithFormat:@"kernel_qwen4_moe_mm_%@_f16_nt%u", kind, nt];
    return [NSString stringWithFormat:@"kernel_qwen4_moe_mm_%@%@", kind,
        nt == 4 ? @"" : [NSString stringWithFormat:@"_nt%u", nt]];
}
static NSArray *pair(id<MTLDevice> device, id<MTLLibrary> lib, bool half, unsigned midNT, unsigned downNT) {
    return @[pipeline(device, lib, kernel_name(true, half, midNT), q4_k32?12u:16u),
             pipeline(device, lib, kernel_name(false, half, downNT), q4_k32?39u:10u)];
}
static NSArray *pair_k32(id<MTLDevice> device, id<MTLLibrary> lib, bool half) {
    NSString *mid = half ? @"kernel_qwen4_moe_mm_mid_f16_k32_nt4" : @"kernel_qwen4_moe_mm_mid_k32_nt4";
    NSString *down = half ? @"kernel_qwen4_moe_mm_down_f16_k32_nt4" : @"kernel_qwen4_moe_mm_down_k32_nt4";
    return @[pipeline_with_type(device, lib, mid, 12u), pipeline_with_type(device, lib, down, 39u)];
}
static void bind(id<MTLComputeCommandEncoder> e, id<MTLBuffer> b, unsigned index) {
    [e setBuffer:b offset:kOffset atIndex:index];
}
static void encode_convert(id<MTLComputeCommandEncoder> en, id<MTLComputePipelineState> p, Fixture *f) {
    const uint32_t n4 = f.tokens * f.dim / 4;
    [en setComputePipelineState:p]; [en setBytes:&n4 length:sizeof(n4) atIndex:0];
    bind(en, f.x, 1); bind(en, f.xHalf, 2);
    [en dispatchThreadgroups:MTLSizeMake(((n4+3)/4+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static void encode_mm(id<MTLComputeCommandEncoder> en, id<MTLComputePipelineState> p,
                       mm_args args, NSArray<id<MTLBuffer>> *buffers) {
    [en setComputePipelineState:p]; [en setBytes:&args length:sizeof(args) atIndex:0];
    for (NSUInteger i = 0; i < buffers.count; i++) bind(en, buffers[i], (unsigned)i + 1);
    unsigned rb = (args.out_rows + 31) / 32, ne = args.n_active_expert ?: args.n_expert;
    MTLSize grid = args.expert_major ? MTLSizeMake((NSUInteger)rb*args.tiles_per_launch,ne,1)
                                   : MTLSizeMake(rb,ne,args.tiles_per_launch);
    [en dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}
static double run(id<MTLCommandQueue> queue, NSArray *ps, id<MTLComputePipelineState> convert,
                    Fixture *f, bool half, bool reference) {
    id<MTLBuffer> mid = reference ? f.refMid : f.gotMid, part = reference ? f.refPart : f.gotPart;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    if (!cb || !en) fail(@"command allocation");
    if (half) encode_convert(en, convert, f);
    NSMutableArray *in = [@[f.gate,f.up,f.lists,f.counts,half?f.xHalf:f.x,mid] mutableCopy];
    if (half) [in addObject:f.midHalf];
    encode_mm(en, ps[0], f.midArgs, in);
    encode_mm(en, ps[1], f.downArgs, @[f.down,f.lists,f.counts,half?f.midHalf:mid,part]);
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) fail(cb.error.description);
    const double elapsed = (cb.GPUEndTime - cb.GPUStartTime) * 1000.;
    if (!(elapsed > 0)) fail(@"GPU timestamps unavailable");
    return elapsed;
}
static void check(id<MTLBuffer> reference, id<MTLBuffer> candidate, NSString *label) {
    const size_t n = (reference.length - 2 * kOffset) / 4;
    const uint32_t *r = data(reference), *c = data(candidate);
    for (size_t i = 0; i < n; i++) {
        if (r[i] == UINT32_C(0xa5a5a5a5) || c[i] == UINT32_C(0xa5a5a5a5))
            fail([NSString stringWithFormat:@"%@ output untouched at %zu", label, i]);
        if ((r[i]&0x7f800000u) == 0x7f800000u || (c[i]&0x7f800000u) == 0x7f800000u)
            fail([NSString stringWithFormat:@"%@ nonfinite at %zu", label, i]);
        if (r[i] != c[i]) fail([NSString stringWithFormat:@"%@ mismatch at %zu: %08x != %08x", label, i, r[i], c[i]]);
    }
    comparisons += n; guard(reference); guard(candidate);
}
static void poison(id<MTLBuffer> b) {
    /* Payload only: pre-existing boundary guards must remain observable. */
    memset(data(b), 0xA5, b.length - 2 * kOffset);
}
static uint16_t half_bits(float value) {
    const _Float16 rounded = (_Float16)value;
    uint16_t bits;
    _Static_assert(sizeof(rounded) == sizeof(bits), "CPU IEEE half storage");
    memcpy(&bits, &rounded, sizeof(bits));
    return bits;
}
static void check_half_payloads(Fixture *f) {
    const float *x = data(f.x), *mid = data(f.refMid);
    const uint16_t *xh = data(f.xHalf), *mh = data(f.midHalf);
    const size_t nx = (size_t)f.tokens * f.dim;
    for (size_t i = 0; i < nx; ++i) {
        const uint16_t expected = half_bits(x[i]);
        if (xh[i] != expected)
            fail([NSString stringWithFormat:@"x half payload at %zu: %04x != %04x", i, xh[i], expected]);
    }
    /* The routing list defines the mid rows actually written by the kernel.
     * Do not treat unselected expert/list padding as an activation result. */
    const mm_args args = f.midArgs;
    const size_t pairs = (size_t)args.n_tokens * args.n_out;
    uint8_t *written = calloc(pairs, sizeof(*written));
    if (!written) fail(@"half payload routing mask allocation");
    const int32_t *lists = data(f.lists), *counts = data(f.counts);
    const unsigned ne = args.n_active_expert ?: args.n_expert;
    for (unsigned grid_e = 0; grid_e < ne; ++grid_e) {
        const unsigned e = args.n_active_expert ? args.active_expert[grid_e] : grid_e;
        if (e >= args.n_expert || counts[e] < 0 || (unsigned)counts[e] > args.list_cap)
            fail(@"invalid half payload fixture routing");
        for (unsigned i = 0; i < (unsigned)counts[e]; ++i) {
            const int32_t pair = lists[(size_t)e * args.list_cap + i];
            if (pair < 0 || (uint64_t)pair >= (uint64_t)args.n_tokens * args.n_slots)
                fail(@"half payload fixture pair outside token slots");
            const unsigned token = (unsigned)pair / args.n_slots, slot = (unsigned)pair % args.n_slots;
            if (slot >= args.n_out) fail(@"half payload fixture output slot");
            const size_t row = (size_t)token * args.n_out + slot;
            if (written[row]++) fail(@"half payload fixture duplicate output row");
        }
    }
    for (size_t pair = 0; pair < pairs; ++pair) {
        for (unsigned r = 0; r < args.out_rows; ++r) {
            const size_t i = pair * args.out_rows + r;
            const uint16_t expected = written[pair] ? half_bits(mid[i]) : UINT16_C(0xa5a5);
            if (mh[i] != expected)
                fail([NSString stringWithFormat:@"mid half payload at %zu: %04x != %04x", i, mh[i], expected]);
        }
        if (written[pair]) half_comparisons += args.out_rows;
    }
    half_comparisons += nx;
    free(written);
    guard(f.xHalf); guard(f.midHalf);
}
static void validate(id<MTLCommandQueue> queue, NSArray *baseline, NSArray *current,
                        NSArray *half, id<MTLComputePipelineState> convert, Fixture *f) {
    poison(f.refMid); poison(f.refPart);
    run(queue, baseline, convert, f, false, true);
    poison(f.gotMid); poison(f.gotPart);
    run(queue, current, convert, f, false, false);
    check(f.refMid, f.gotMid, @"current float mid"); check(f.refPart, f.gotPart, @"current float down");
    poison(f.gotMid); poison(f.gotPart); poison(f.xHalf); poison(f.midHalf);
    run(queue, half, convert, f, true, false);
    check(f.refMid, f.gotMid, @"half-storage float mid"); check(f.refPart, f.gotPart, @"half-storage down");
    check_half_payloads(f);
    for (id<MTLBuffer> b in @[f.gate,f.up,f.down,f.lists,f.counts,f.x,f.xHalf,f.midHalf]) guard(b);
    fixtures++;
}
static double median(NSArray<NSNumber *> *a) {
    NSArray *s = [a sortedArrayUsingSelector:@selector(compare:)];
    NSUInteger n = s.count;
    return n&1 ? [s[n/2] doubleValue] : ([s[n/2-1] doubleValue]+[s[n/2] doubleValue])/2.;
}
static NSDictionary *benchmark(id<MTLCommandQueue> queue, NSArray *baseline, NSArray *current,
                                  NSArray *half, id<MTLComputePipelineState> convert, Fixture *f,
                                  unsigned midNT, unsigned downNT, unsigned reps) {
    validate(queue, baseline, current, half, convert, f);
    for (unsigned i = 0; i < 2; i++) { run(queue, baseline, convert, f, false, true); run(queue, half, convert, f, true, false); }
    NSMutableArray *a = [NSMutableArray new], *b = [NSMutableArray new];
    /* Each round has two samples of each arm with equal mean position in
     * the sequence. Reversing successive rounds also balances adjacency. */
    for (unsigned i = 0; i < reps; i++) for (unsigned j = 0; j < 4; j++) {
        const bool candidate = ((j == 1 || j == 2) != ((i & 1u) != 0));
        if (candidate) [b addObject:@(run(queue, half, convert, f, true, false))];
        else [a addObject:@(run(queue, baseline, convert, f, false, true))];
    }
    check(f.refMid, f.gotMid, @"benchmark mid"); check(f.refPart, f.gotPart, @"benchmark down");
    const double ma = median(a), mb = median(b);
    printf("T=%u E=%u F=%u experts=%u slots=%u midNT=%u downNT=%u: float %.3f ms half %.3f ms speedup %.2f%%\n",
        f.tokens,f.dim,f.ff,f.experts,f.slots,midNT,downNT,ma,mb,(ma/mb-1)*100);
    fflush(stdout);
    return @{@"tokens":@(f.tokens),@"dim":@(f.dim),@"ff":@(f.ff),@"experts":@(f.experts),
        @"slots":@(f.slots),@"mid_nt":@(midNT),@"down_nt":@(downNT),
        @"float_gpu_ms":a,@"half_gpu_ms":b,@"float_median_ms":@(ma),@"half_median_ms":@(mb),
        @"speedup_percent":@((ma/mb-1)*100),@"sample_order":@"alternating ABBA/BAAB",@"rounds":@(reps),
        @"warmup_runs_per_arm":@2,@"includes_x_conversion":@YES,@"includes_dual_mid_write":@YES};
}
static void write_json(NSString *path, NSDictionary *report) {
    if (!path) return;
    NSError *error = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
    if (!json || ![json writeToFile:path options:NSDataWritingAtomic error:&error]) fail(error.description);
}
int main(int argc, const char **argv) { @autoreleasepool {
    NSString *repo = [[NSFileManager defaultManager] currentDirectoryPath], *candidatePath = nil, *baselinePath = nil, *output = nil;
    bool compileOnly = false, bench = false, safe = false;
    unsigned reps = 9;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--compile-only")) compileOnly = true;
        else if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--q4-k32")) q4_k32 = true;
        else if (!strcmp(argv[i], "--safe-math")) safe = true;
        else if (!strcmp(argv[i], "--generic-types")) specialized_types = false;
        else if (!strcmp(argv[i], "--specialized-types")) specialized_types = true;
        else if (!strcmp(argv[i], "--repo") && i+1 < argc) repo = [NSString stringWithUTF8String:argv[++i]];
        else if (!strcmp(argv[i], "--qwen-source") && i+1 < argc) candidatePath = [NSString stringWithUTF8String:argv[++i]];
        else if (!strcmp(argv[i], "--baseline-source") && i+1 < argc) baselinePath = [NSString stringWithUTF8String:argv[++i]];
        else if (!strcmp(argv[i], "--output") && i+1 < argc) output = [NSString stringWithUTF8String:argv[++i]];
        else if (!strcmp(argv[i], "--reps") && i+1 < argc) reps = (unsigned)strtoul(argv[++i],NULL,10);
        else { fprintf(stderr,"usage: %s [--compile-only] [--bench] [--safe-math] [--q4-k32] [--generic-types|--specialized-types] [--reps N] [--repo PATH] [--qwen-source FILE] [--baseline-source FILE] [--output JSON]\n",argv[0]); return 2; }
    }
    if (reps < 3 || reps > 100) fail(@"--reps must be in [3,100]");
    candidatePath = candidatePath ?: [repo stringByAppendingPathComponent:@"metal/qwen4.metal"];
    const bool explicit_baseline = baselinePath != nil;
    baselinePath = baselinePath ?: candidatePath;
    NSString *candidateSource = production_source(repo,candidatePath), *baselineSource = production_source(repo,baselinePath);
    const bool independent_baseline = explicit_baseline && ![baselineSource isEqualToString:candidateSource];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) fail(@"Metal device unavailable");
    id<MTLLibrary> candidateLib = library(device,candidateSource,safe), baselineLib = library(device,baselineSource,safe);
    NSMutableArray *bases = [NSMutableArray new], *currents = [NSMutableArray new], *halves = [NSMutableArray new];
    for (unsigned i = 0; i < 4; i++) {
        [bases addObject:pair(device,baselineLib,false,kNT[i],kNT[i])];
        [currents addObject:pair(device,candidateLib,false,kNT[i],kNT[i])];
        [halves addObject:pair(device,candidateLib,true,kNT[i],kNT[i])];
    }
    NSArray *k32Float = q4_k32 ? pair_k32(device,candidateLib,false) : nil;
    NSArray *k32Half = q4_k32 ? pair_k32(device,candidateLib,true) : nil;
    id<MTLComputePipelineState> convert = pipeline(device,candidateLib,@"kernel_qwen4_rows_f32_to_f16",0);
    NSMutableDictionary *report = [@{@"device":device.name,@"safe_math":@(safe),@"compile_only":@(compileOnly),
        @"baseline_source":baselinePath,@"candidate_source":candidatePath,
        @"explicit_baseline":@(explicit_baseline),@"independent_baseline":@(independent_baseline),
        @"weight_type_mode":specialized_types ? @"specialized" : @"generic",
        @"function_constant_900_mid":@(specialized_types ? (q4_k32?12u:16u) : 0u),
        @"function_constant_900_down":@(specialized_types ? (q4_k32?39u:10u) : 0u),
        @"q4_k32":@(q4_k32),@"k32_specialized":@(q4_k32),
        @"poisoned_validation_outputs":@YES,@"half_payload_cpu_reference":@YES,
        @"baseline_fnv1a64":[NSString stringWithFormat:@"%016llx",(unsigned long long)source_hash(baselineSource)],
        @"candidate_fnv1a64":[NSString stringWithFormat:@"%016llx",(unsigned long long)source_hash(candidateSource)]} mutableCopy];
    if (!compileOnly) {
        id<MTLCommandQueue> queue = [device newCommandQueue]; if (!queue) fail(@"command queue");
        const unsigned ts[] = {1,7,9,31,33,67,131};
        for (unsigned s = 0; s < sizeof(ts)/sizeof(ts[0]); s++) @autoreleasepool {
            Fixture *f = fixture(device,ts[s],q4_k32 && s%3==2 ? 768u : 256u,13,(s&1)!=0,(s&2)!=0);
            for (unsigned n = 0; n < 4; n++) validate(queue,bases[n],currents[n],halves[n],convert,f);
            if (q4_k32) validate(queue,bases[2],k32Float,k32Half,convert,f);
            printf("PASS T=%u NT=1/2/4/8%s sparse=%u expert-major=%u\n",ts[s],
                q4_k32 ? " plus K32 NT4" : "",s&1,(s&2)!=0); fflush(stdout);
        }
        if (q4_k32) {
            const unsigned sizes[] = {439,8191,8192};
            for (unsigned i = 0; i < 3; i++) @autoreleasepool {
                Fixture *f = fixture(device,sizes[i],256,13,true,(i&1)!=0);
                validate(queue,bases[2],k32Float,k32Half,convert,f);
                printf("PASS Q4 K32 float/half T=%u, row and token tails\n",sizes[i]); fflush(stdout);
            }
        }
        NSMutableArray *times = [NSMutableArray new];
        if (bench) {
            const unsigned bt[] = {128,2048,8192};
            for (unsigned i = 0; i < 3; i++) @autoreleasepool {
                const unsigned midNT = !q4_k32 && i==0 ? 1 : 4, downNT = !q4_k32 && i==0 ? 1 : 4;
                Fixture *f = fixture(device,bt[i],2560,512,false,false);
                NSArray *bp = pair(device,baselineLib,false,midNT,downNT), *cp = pair(device,candidateLib,false,midNT,downNT),
                        *hp = pair(device,candidateLib,true,midNT,downNT);
                if (q4_k32) { cp = k32Float; hp = k32Half; }
                [times addObject:benchmark(queue,bp,cp,hp,convert,f,midNT,downNT,reps)];
            }
        }
        report[@"benchmarks"] = times;
    }
    report[@"fixtures"] = @(fixtures); report[@"compared_floats"] = @(comparisons);
    report[@"compared_half_values"] = @(half_comparisons); report[@"status"] = @"PASS";
    write_json(output,report);
    printf("PASS Qwen MoE half: %llu fixtures, %llu float comparisons%s\n",
        (unsigned long long)fixtures,(unsigned long long)comparisons,compileOnly?" (compile only)":"");
    return 0;
} }
