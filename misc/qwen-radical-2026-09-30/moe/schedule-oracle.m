/* Isolated Q4_K/MXFP4 prefill candidate oracle. Compile baseline and candidate
 * source into separate libraries, compare float intermediates bit-for-bit,
 * and measure paired ABBA/BAAB GPU command-buffer durations. No SSD I/O here.
 * clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
 *   misc/qwen-gain10-v2-2026-09-27/test-prefill-oracle.m -o /tmp/test-prefill-oracle
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t n_tokens, n_slots, n_out, in_dim, out_rows, weight_type, row_bytes, list_cap;
    uint64_t expert_bytes;
    uint32_t n_expert, tiles_per_launch, tail_base, expert_major, n_active_expert;
    uint32_t active_expert[512];
} mm_args;
_Static_assert(sizeof(mm_args) == 2112, "production MoE MM ABI");
_Static_assert(offsetof(mm_args, active_expert) == 60, "production active expert ABI");
static const NSUInteger kOffset = 256;
static uint64_t comparisons, fixtures;
static bool specialized_types=true, skewed_routing, use_real_routing;
static unsigned candidate_mid_rows=32, candidate_mid_threads=128, timing_stage;
static NSString *routing_path;
static NSString *schedule_mode = @"compact";
static bool encoding_candidate;
static double worklist_ms;
static uint32_t compact_work_count;
static unsigned routing_layer;
static NSData *routing_selected, *routing_exact_lists;
static uint32_t routing_header[8];
static NSMutableSet *candidate_mid_pipelines;
static id<MTLComputePipelineState> baseline_convert, candidate_convert;
static void fail(NSString *message) {
    fprintf(stderr, "FAIL Qwen prefill frozen oracle: %s\n", message.UTF8String);
    exit(1);
}
/* QWR2 appends the exact packed expert-major GPU lists. QWR1 is accepted
 * for old captures, with the reconstructed-order limitation recorded. */
static void load_routing(void) {
    NSError *error=nil;
    NSData *blob=[NSData dataWithContentsOfFile:routing_path options:0 error:&error];
    if(!blob)fail([NSString stringWithFormat:@"routing read: %@",error]);
    const uint8_t *bytes=blob.bytes; NSUInteger at=0; bool found=false;
    while(at<blob.length) {
        if(blob.length-at<32u)fail(@"routing truncated header");
        uint32_t h[8];memcpy(h,bytes+at,32);at+=32;
        if((h[0]!=0x51575231u&&h[0]!=0x51575232u)||!h[2]||!h[3]||h[3]>512u||h[4]!=10u||h[5]!=2560u||h[6]!=640u||h[7]!=(uint64_t)h[2]*h[4])
            fail(@"routing header/shape mismatch");
        const bool exactLists=h[0]==0x51575232u;
        const uint64_t payload=((uint64_t)h[3]+(uint64_t)h[7]*(exactLists?2u:1u))*4u;
        if(payload>blob.length-at)fail(@"routing truncated payload");
        if(h[1]==routing_layer) {
            if(found)fail(@"duplicate requested routing layer");
            found=true;memcpy(routing_header,h,sizeof(h));
            uint32_t recorded[512],counts[512]={0};memcpy(recorded,bytes+at,h[3]*4u);
            routing_selected=[NSData dataWithBytes:bytes+at+h[3]*4u length:(NSUInteger)h[7]*4u];
            const int32_t *ids=routing_selected.bytes;
            for(uint32_t t=0;t<h[2];t++)for(uint32_t slot=0;slot<h[4];slot++) {
                const int32_t e=ids[(size_t)t*h[4]+slot];
                if(e<0||(uint32_t)e>=h[3])fail(@"routing expert outside range");
                for(uint32_t prev=0;prev<slot;prev++)if(ids[(size_t)t*h[4]+prev]==e)fail(@"routing duplicate expert within token");
                counts[e]++;
            }
            uint64_t total=0;for(uint32_t e=0;e<h[3];e++) {
                if(counts[e]!=recorded[e]||counts[e]>h[2])fail(@"routing frequency mismatch");
                total+=counts[e];
            }
            if(total!=h[7])fail(@"routing frequency total");
            if(exactLists) {
                routing_exact_lists=[NSData dataWithBytes:bytes+at+(h[3]+(uint64_t)h[7])*4u length:(NSUInteger)h[7]*4u];
                const int32_t *pairs=routing_exact_lists.bytes;
                NSMutableData *seenData=[NSMutableData dataWithLength:h[7]];uint8_t *seen=seenData.mutableBytes;
                uint32_t offset=0;
                for(uint32_t e=0;e<h[3];e++)for(uint32_t j=0;j<recorded[e];j++) {
                    int32_t pair=pairs[offset++];
                    if(pair<0||(uint32_t)pair>=h[7]||seen[pair]||ids[pair]!=(int32_t)e)fail(@"routing exact list pair mismatch");
                    seen[pair]=1;
                }
                for(uint32_t pair=0;pair<h[7];pair++)if(!seen[pair])fail(@"routing exact list omitted pair");
            }
        }
        at+=(NSUInteger)payload;
    }
    if(!found)fail(@"requested routing layer missing");
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
    (void)repo;
    NSString *m = read_source([qwenPath.stringByDeletingLastPathComponent stringByAppendingPathComponent:@"baseline-moe.metal"]);
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
    if (type == 12u) {
        for (size_t i = 0; i < bytes; i += 144u) {
            uint16_t h = (uint16_t)(0x1000u | (mix((uint32_t)i + seed) & 0x83ffu));
            memcpy(p+i,&h,2); h ^= 0x0195u; memcpy(p+i+2,&h,2);
        }
    } else if (type == 39u) {
        for (size_t i = 0; i < bytes; i += 17u) p[i] = 116u + mix((uint32_t)i + seed) % 5u;
    } else fail(@"unsupported fixture weight type");
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
        .in_dim=dim, .out_rows=f.ff, .weight_type=12, .row_bytes=dim/256*144,
        .list_cap=tokens, .n_expert=experts, .tiles_per_launch=MIN(8u,(tokens+31u)/32u),
        .expert_major=expertMajor};
    m.expert_bytes = (uint64_t)m.row_bytes * m.out_rows;
    mm_args d = m;
    d.in_dim = f.ff; d.out_rows = dim; d.weight_type = 39; d.row_bytes = f.ff/32*17;
    d.expert_bytes = (uint64_t)d.row_bytes * d.out_rows;
    if (sparse) {
        m.n_active_expert = d.n_active_expert = experts - 2;
        for (unsigned e = 0; e < m.n_active_expert; e++) m.active_expert[e] = d.active_expert[e] = e;
    }
    f.midArgs = m; f.downArgs = d;
    f.gate = buffer(device, m.expert_bytes * experts); f.up = buffer(device, m.expert_bytes * experts);
    f.down = buffer(device, d.expert_bytes * experts);
    fill_weights(f.gate, 12, 17); fill_weights(f.up, 12, 59); fill_weights(f.down, 39, 117);
    f.lists = buffer(device, ((size_t)experts * tokens + (size_t)experts * ((tokens+31u)/32u)) * sizeof(int32_t));
    f.counts = buffer(device, experts * sizeof(int32_t));
    int32_t *lists = data(f.lists), *counts = data(f.counts);
    memset(counts, 0, experts * sizeof(int32_t));
    const unsigned active = sparse ? experts - 2 : experts;
    for (unsigned t = 0; t < tokens; t++) for (unsigned slot = 0; slot < f.slots; slot++) {
        /* Distinct slots per token: consecutive IDs avoid duplicate expert
         * selection even in the small 11/13-expert regression fixtures. */
        unsigned width = skewed_routing && t % 4 != 0 ? MIN(active,32u) : active;
        unsigned e = use_real_routing ? (unsigned)((const int32_t *)routing_selected.bytes)[(size_t)t*f.slots+slot] : (t * 7 + slot) % width;
        lists[(size_t)e * tokens + (unsigned)counts[e]++] = (int32_t)(t * f.slots + slot);
    }
    if(use_real_routing) {
        if(tokens>routing_header[2]||experts!=routing_header[3])fail(@"routing fixture dimensions");
        if(routing_exact_lists) {
            const int32_t *pairs=routing_exact_lists.bytes;
            const int32_t *ids=routing_selected.bytes;
            uint32_t full_counts[512]={0};
            for(uint32_t pair=0;pair<routing_header[7];pair++)full_counts[ids[pair]]++;
            size_t offset=0;
            for(unsigned e=0;e<experts;e++) {
                uint32_t n=0;
                for(uint32_t j=0;j<full_counts[e];j++) {
                    const int32_t pair=pairs[offset++];
                    if((uint32_t)pair<tokens*f.slots)lists[(size_t)e*tokens+n++]=pair;
                }
                if(n!=(uint32_t)counts[e])fail(@"captured-order prefix frequency mismatch");
            }
        }
    }
    /* Match production SSD: ascending original expert IDs. */
    m.n_active_expert=d.n_active_expert=0;
    for(unsigned e=0;e<experts;e++)if(counts[e]) {
        m.active_expert[m.n_active_expert++]=e;
        d.active_expert[d.n_active_expert++]=e;
    }
    if(!m.n_active_expert)fail(@"empty fixture route");
    f.midArgs=m;f.downArgs=d;
    // Frequencies are already present on CPU in the SSD path. Include measured
    // schedule construction cost in the paired screen instead of hiding it.
    struct timespec wl_start,wl_end;clock_gettime(CLOCK_MONOTONIC,&wl_start);
    uint32_t order[512];unsigned ne=0;
    for(unsigned e=0;e<experts;e++)if(counts[e])order[ne++]=e;
    if([schedule_mode isEqualToString:@"compact-heavy"]) {
        for(unsigned i=1;i<ne;i++) { uint32_t e=order[i];unsigned j=i;
            while(j && counts[order[j-1]]<counts[e]) {order[j]=order[j-1];j--;}
            order[j]=e;
        }
    }
    uint32_t *work=(uint32_t *)(lists+(size_t)experts*tokens);compact_work_count=0;
    for(unsigned i=0;i<ne;i++)for(unsigned tile=0;tile<((unsigned)counts[order[i]]+31u)/32u;tile++)
        work[compact_work_count++]=(tile<<9u)|order[i];
    clock_gettime(CLOCK_MONOTONIC,&wl_end);
    worklist_ms=(wl_end.tv_sec-wl_start.tv_sec)*1000.+(wl_end.tv_nsec-wl_start.tv_nsec)*1e-6;
    f.x = buffer(device, (size_t)tokens * dim * 4); f.xHalf = buffer(device, (size_t)tokens * dim * 2);
    float *x = data(f.x);
    for (size_t i = 0; i < (size_t)tokens * dim; i++)
        x[i] = ((int)(mix((uint32_t)i + 31) % 16385) - 8192) / 8192.f;
    const size_t nm = (size_t)tokens * f.slots * f.ff, np = (size_t)tokens * f.slots * dim;
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
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> lib,
                                           NSString *name, unsigned type) {
    NSError *error = nil;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    unsigned tail = 0; bool addressed = false;
    const unsigned bound_type = specialized_types ? type : 0u;
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
static NSArray *pair(id<MTLDevice> device,id<MTLLibrary> lib,bool half) {
    return @[pipeline(device,lib,half?@"kernel_qwen4_moe_mm_mid_f16_k32_nt4":@"kernel_qwen4_moe_mm_mid_k32_nt4",12),
             pipeline(device,lib,half?@"kernel_qwen4_moe_mm_down_f16_k32_nt4":@"kernel_qwen4_moe_mm_down_k32_nt4",39)];
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
    if(encoding_candidate && [schedule_mode isEqualToString:@"major"]) args.expert_major=1u;
    const bool compact=encoding_candidate && [schedule_mode hasPrefix:@"compact"];
    if(compact) {args.expert_major=2u;args.tiles_per_launch=args.n_tokens;args.tail_base=0u;}
    [en setComputePipelineState:p]; [en setBytes:&args length:sizeof(args) atIndex:0];
    for (NSUInteger i = 0; i < buffers.count; i++) bind(en, buffers[i], (unsigned)i + 1);
    unsigned rows = [candidate_mid_pipelines containsObject:p] ? candidate_mid_rows : 32;
    unsigned rb = (args.out_rows + rows - 1) / rows, ne = args.n_active_expert ?: args.n_expert;
    MTLSize grid = args.expert_major ? MTLSizeMake((NSUInteger)rb*args.tiles_per_launch,ne,1)
                                   : MTLSizeMake(rb,ne,args.tiles_per_launch);
    if(compact) grid=MTLSizeMake(rb,compact_work_count,1);
    if(!grid.width || !grid.height || !grid.depth) return;
    [en dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake([candidate_mid_pipelines containsObject:p]?candidate_mid_threads:128,1,1)];
}
static double run(id<MTLCommandQueue> queue, NSArray *ps, id<MTLComputePipelineState> convert,
                    Fixture *f, bool half, bool reference) {
    encoding_candidate=!reference;
    id<MTLBuffer> mid = reference ? f.refMid : f.gotMid, part = reference ? f.refPart : f.gotPart;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    if (!cb || !en) fail(@"command allocation");
    if (half && (!timing_stage || timing_stage==1)) encode_convert(en, convert, f);
    NSMutableArray *in = [@[f.gate,f.up,f.lists,f.counts,half?f.xHalf:f.x,mid] mutableCopy];
    if (half) [in addObject:f.midHalf];
    if(!timing_stage || timing_stage==2) encode_mm(en, ps[0], f.midArgs, in);
    if(!timing_stage || timing_stage==3) encode_mm(en, ps[1], f.downArgs, @[f.down,f.lists,f.counts,half?f.midHalf:mid,part]);
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) fail(cb.error.description);
    const double elapsed = (cb.GPUEndTime - cb.GPUStartTime) * 1000.;
    if (!(elapsed > 0)) fail(@"GPU timestamps unavailable");
    return elapsed + (!reference && [schedule_mode hasPrefix:@"compact"] && !timing_stage ? worklist_ms : 0.);
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

static uint64_t hash_inputs(Fixture *f) {
    uint64_t h=UINT64_C(14695981039346656037);
    for(id<MTLBuffer> b in @[f.x,f.lists,f.counts]) {
        const uint8_t *p=data(b);
        for(NSUInteger i=0;i<b.length-2*kOffset;i++)h=(h^p[i])*UINT64_C(1099511628211);
    }
    return h;
}
static void check_halves(Fixture *f) {
    const float *x=data(f.x),*mid=data(f.gotMid);
    const uint16_t *xh=data(f.xHalf),*mh=data(f.midHalf);
    for(NSUInteger i=0;i<(f.x.length-2*kOffset)/4;i++) {
        _Float16 v=(_Float16)x[i];uint16_t bits;memcpy(&bits,&v,2);
        if(bits!=xh[i])fail(@"input half conversion mismatch");
    }
    for(NSUInteger i=0;i<(f.gotMid.length-2*kOffset)/4;i++) {
        _Float16 v=(_Float16)mid[i];uint16_t bits;memcpy(&bits,&v,2);
        if(bits!=mh[i])fail(@"half shadow mismatch");
    }
}
static void validate(id<MTLCommandQueue> queue,NSArray *baseline,NSArray *candidate,Fixture *f,bool half) {
    const uint64_t before=hash_inputs(f);
    poison(f.refMid);poison(f.refPart);poison(f.gotMid);poison(f.gotPart);poison(f.xHalf);poison(f.midHalf);
    run(queue,baseline,baseline_convert,f,half,true);
    run(queue,candidate,candidate_convert,f,half,false);
    check(f.refMid,f.gotMid,@"mid");check(f.refPart,f.gotPart,@"part");
    if(half)check_halves(f);
    if(before!=hash_inputs(f))fail(@"immutable inputs changed");
    for(id<MTLBuffer> b in @[f.gate,f.up,f.down,f.lists,f.counts,f.x,f.xHalf,f.midHalf])guard(b);
    fixtures++;
}
static double median(NSArray<NSNumber *> *a) {
    NSArray *s=[a sortedArrayUsingSelector:@selector(compare:)];NSUInteger n=s.count;
    return n&1?[s[n/2] doubleValue]:([s[n/2-1] doubleValue]+[s[n/2] doubleValue])/2.;
}
static NSDictionary *benchmark(id<MTLCommandQueue> queue,NSArray *baseline,NSArray *candidate,Fixture *f,bool half,unsigned reps,bool stages) {
    validate(queue,baseline,candidate,f,half);
    for(unsigned i=0;i<2;i++){run(queue,baseline,baseline_convert,f,half,true);run(queue,candidate,candidate_convert,f,half,false);}
    NSMutableArray *a=[NSMutableArray new],*b=[NSMutableArray new];
    for(unsigned i=0;i<reps;i++)for(unsigned j=0;j<4;j++) {
        bool fast=((j==1||j==2)!=((i&1u)!=0));
        [(fast?b:a) addObject:@(run(queue,fast?candidate:baseline,fast?candidate_convert:baseline_convert,f,half,!fast))];
    }
    check(f.refMid,f.gotMid,@"bench mid");check(f.refPart,f.gotPart,@"bench part");
    uint64_t allocation=0;for(id<MTLBuffer> buf in @[f.gate,f.up,f.down,f.lists,f.counts,f.x,f.xHalf,f.midHalf,f.refMid,f.gotMid,f.refPart,f.gotPart])allocation+=buf.length;
    NSMutableArray *separate=[NSMutableArray new];
    if(stages)for(timing_stage=half?1u:2u;timing_stage<=3u;timing_stage++) {
        NSMutableArray *sa=[NSMutableArray new],*sb=[NSMutableArray new];
        for(unsigned i=0;i<3;i++)for(unsigned j=0;j<4;j++) {
            bool fast=((j==1||j==2)!=((i&1u)!=0));
            [(fast?sb:sa) addObject:@(run(queue,fast?candidate:baseline,fast?candidate_convert:baseline_convert,f,half,!fast))];
        }
        [separate addObject:@{@"stage":@(timing_stage),@"baseline_gpu_ms":sa,@"candidate_gpu_ms":sb,@"baseline_median_ms":@(median(sa)),@"candidate_median_ms":@(median(sb))}];
    }
    timing_stage=0;
    const double ma=median(a),mb=median(b);
    printf("BENCH T=%u D=%u E=%u baseline=%.4f candidate=%.4f gain=%.2f%%\n",f.tokens,f.dim,f.experts,ma,mb,(ma/mb-1)*100);fflush(stdout);
    NSData *countData=[NSData dataWithBytes:data(f.counts) length:f.experts*4u];
    NSMutableArray *counts=[NSMutableArray new];const int32_t *cp=countData.bytes;for(unsigned e=0;e<f.experts;e++)[counts addObject:@(cp[e])];
    return @{@"tokens":@(f.tokens),@"dim":@(f.dim),@"ff":@(f.ff),@"experts":@(f.experts),@"active_experts":@(f.midArgs.n_active_expert),@"counts":counts,@"half":@(half),
        @"baseline_gpu_ms":a,@"candidate_gpu_ms":b,@"baseline_median_ms":@(ma),@"candidate_median_ms":@(mb),@"speedup_percent":@((ma/mb-1)*100),
        @"baseline_mid_tg_bytes":@([baseline[0] staticThreadgroupMemoryLength]),@"candidate_mid_tg_bytes":@([candidate[0] staticThreadgroupMemoryLength]),@"baseline_down_tg_bytes":@([baseline[1] staticThreadgroupMemoryLength]),@"candidate_down_tg_bytes":@([candidate[1] staticThreadgroupMemoryLength]),@"fixture_buffer_bytes":@(allocation),@"extra_candidate_global_bytes":@0,@"sample_order":@"alternating ABBA/BAAB",@"rounds":@(reps),@"stages":separate};
}
int main(int argc,const char **argv){@autoreleasepool{
    NSString *repo=[[NSFileManager defaultManager]currentDirectoryPath],*basePath=nil,*candidatePath=nil,*output=nil;
    bool bench=false,safe=false,compileOnly=false,stages=false;unsigned reps=3,benchTokens=8192,benchExperts=512;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--bench"))bench=true;
        else if(!strcmp(argv[i],"--safe-math"))safe=true;
        else if(!strcmp(argv[i],"--compile-only"))compileOnly=true;
        else if(!strcmp(argv[i],"--stages"))stages=true;
        else if(!strcmp(argv[i],"--schedule")&&i+1<argc)schedule_mode=[NSString stringWithUTF8String:argv[++i]];
        else if(!strcmp(argv[i],"--skewed"))skewed_routing=true;
        else if(!strcmp(argv[i],"--routing-file")&&i+1<argc)routing_path=[NSString stringWithUTF8String:argv[++i]];
        else if(!strcmp(argv[i],"--routing-layer")&&i+1<argc)routing_layer=(unsigned)strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--baseline-source")&&i+1<argc)basePath=[NSString stringWithUTF8String:argv[++i]];
        else if(!strcmp(argv[i],"--qwen-source")&&i+1<argc)candidatePath=[NSString stringWithUTF8String:argv[++i]];
        else if(!strcmp(argv[i],"--output")&&i+1<argc)output=[NSString stringWithUTF8String:argv[++i]];
        else if(!strcmp(argv[i],"--reps")&&i+1<argc)reps=(unsigned)strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--bench-tokens")&&i+1<argc)benchTokens=(unsigned)strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--bench-experts")&&i+1<argc)benchExperts=(unsigned)strtoul(argv[++i],NULL,10);
        else{fprintf(stderr,"bad argument %s\n",argv[i]);return 2;}
    }
    if(![@[@"baseline",@"major",@"compact",@"compact-heavy"] containsObject:schedule_mode])fail(@"invalid schedule");
    if(!basePath||!candidatePath||reps<3||reps>20||benchExperts<10||benchExperts>512)fail(@"explicit sources and valid dimensions required");
    if(routing_path){load_routing();benchExperts=routing_header[3];if(benchTokens>routing_header[2])fail(@"routing prefix too long");}
    NSString *bs=production_source(repo,basePath),*cs=production_source(repo,candidatePath);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();if(!device)fail(@"Metal device unavailable");
    id<MTLLibrary> bl=library(device,bs,safe),cl=library(device,cs,safe);
    NSArray *bases=@[pair(device,bl,false),pair(device,bl,true)],*cands=@[pair(device,cl,false),pair(device,cl,true)];
    baseline_convert=pipeline(device,bl,@"kernel_qwen4_rows_f32_to_f16",0);
    candidate_convert=pipeline(device,cl,@"kernel_qwen4_rows_f32_to_f16",0);
    candidate_mid_pipelines=[NSMutableSet setWithObjects:cands[0][0],cands[1][0],nil];
    NSMutableArray *times=[NSMutableArray new];
    if(!compileOnly){
        id<MTLCommandQueue> queue=[device newCommandQueue];if(!queue)fail(@"command queue");
        const unsigned ts[]={1,7,9,30,33,67,131,201,439};
        for(unsigned k=0;k<sizeof(ts)/sizeof(*ts);k++)@autoreleasepool{
            Fixture *f=fixture(device,ts[k],k==8?768:256,13,(k&1)!=0,(k&2)!=0);
            for(unsigned h=0;h<2;h++)validate(queue,bases[h],cands[h],f,h);
            printf("PASS small T%u float/half\n",ts[k]);fflush(stdout);
        }
        if(bench){use_real_routing=routing_path!=nil;Fixture *f=fixture(device,benchTokens,2560,benchExperts,false,false);bool half=benchTokens>=8192;
            [times addObject:benchmark(queue,bases[half],cands[half],f,half,reps,stages)];}
    }
    NSDictionary *result=@{@"status":@"PASS",@"device":device.name,@"safe_math":@(safe),@"compile_only":@(compileOnly),@"baseline_source":basePath,@"candidate_source":candidatePath,
        @"baseline_fnv1a64":[NSString stringWithFormat:@"%016llx",(unsigned long long)source_hash(bs)],@"candidate_fnv1a64":[NSString stringWithFormat:@"%016llx",(unsigned long long)source_hash(cs)],
        @"schedule":schedule_mode,@"worklist_cpu_ms":@(worklist_ms),@"compact_work_count":@(compact_work_count),@"active_order":@"ascending expert IDs, matching production SSD",@"routing_file":routing_path?:@"synthetic",@"routing_layer":@(routing_layer),@"routing_order":routing_exact_lists?@"captured GPU order filtered to token prefix":@"CPU ascending pair IDs",@"output_stride":@10,
        @"fixtures":@(fixtures),@"bit_exact_float_comparisons":@(comparisons),@"benchmarks":times};
    if(output){NSError *e=nil;NSData *d=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&e];if(!d||![d writeToFile:output options:NSDataWritingAtomic error:&e])fail(e.description);}
    printf("PASS prefill oracle %llu fixtures %llu exact FP32 comparisons\n",(unsigned long long)fixtures,(unsigned long long)comparisons);return 0;
}}
