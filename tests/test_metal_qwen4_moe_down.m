/* Exact Q2_K F640 down regression: generic row dot vs the optimized export
 * selected automatically on M1 Max. The production host policy is separate.
 * Build: clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
 *            tests/test_metal_qwen4_moe_down.m -o /tmp/test_metal_qwen4_moe_down
 * Run from the repository root, or use --repo PATH. --compile-only creates
 * no queue/buffers/dispatches. --qwen-source FILE checks a temporary candidate.
 * Kernel bodies, argument layout, helpers and tables come from current sources.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n_tokens, n_slots, in_dim, out_rows, weight_type, row_bytes;
    uint64_t expert_bytes;
    uint32_t has_shared, shared_type, shared_row_bytes, n_total_expert;
    uint32_t slot_mask[3], masked_tokens;
    uint32_t list_cap, pad0;
} moe_args;
_Static_assert(sizeof(moe_args)==72, "production MoE argument ABI");
_Static_assert(offsetof(moe_args,expert_bytes)==24, "expert stride alignment");
_Static_assert(offsetof(moe_args,masked_tokens)==60, "slot mask ABI");
_Static_assert(offsetof(moe_args, list_cap) == 64, "grouped list capacity ABI");
typedef struct { uint32_t tokens, rows, slots; } shape;
static const shape kShapes[]={{1,1,1},{2,7,2},{3,9,10},{1,2560,10},{3,2561,10}};
static const NSUInteger kOffset=512; /* Covers a wrongly read 128-float input tail. */
static const uint32_t kCanary=UINT32_C(0x4bd13579);
static const unsigned kExperts=12;
static uint64_t fixtures, compared, nan_payload_differences;

typedef NS_ENUM(unsigned, Layout) {
    Resident, Addressed, CompactAll, CompactSparse, CompactShared, CompactRouted,
    CompactShort, CompactEmpty, LayoutCount
};
static void fail(NSString *s) { fprintf(stderr,"FAIL Qwen MoE down: %s\n",s.UTF8String); exit(1); }
static NSString *read_source(NSString *path) {
    NSError *error=nil;
    NSString *s=[NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (!s) fail([NSString stringWithFormat:@"read %@: %@",path,error]); return s;
}
static NSString *between(NSString *s, NSString *begin, NSString *end) {
    if ([s componentsSeparatedByString:begin].count!=2 || [s componentsSeparatedByString:end].count!=2)
        fail(@"production source anchor missing/nonunique");
    NSRange a=[s rangeOfString:begin], b=[s rangeOfString:end];
    if (b.location<=NSMaxRange(a)) fail(@"invalid source range");
    return [s substringWithRange:NSMakeRange(a.location,b.location-a.location)];
}
static NSString *production_source(NSString *repo, NSString *override) {
    NSString *q=read_source(override ?: [repo stringByAppendingPathComponent:@"metal/qwen4.metal"]);
    NSString *m=read_source([repo stringByAppendingPathComponent:@"metal/moe.metal"]);
    NSMutableString *s=[NSMutableString stringWithString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
    [s appendString:between(m,@"static constant float ds4_metal_mxfp4_values[16]",@"// BEGIN GENERATED MXFP4 HALF LUT")];
    [s appendString:between(m,@"static constant uchar ds4_metal_ksigns_iq2xs[128]",@"#define kmask_iq2xs")];
    [s appendString:between(q,@"constant bool qwen4_expert_addresses ",@"/* --- small multi-output GEMV")];
    [s appendString:between(q,@"constant uint qwen4_mv_type ",@"/* mid[t][s][r] =")];
    [s appendString:between(q,@"/* part[t][s][r] = down_row",@"/* MXFP4 routed down rows")];
    return s;
}
static uint64_t source_hash(NSString *s) {
    NSData *data=[s dataUsingEncoding:NSUTF8StringEncoding]; const uint8_t *bytes=data.bytes;
    uint64_t h=UINT64_C(14695981039346656037);
    for (NSUInteger i=0;i<data.length;++i) h=(h^bytes[i])*UINT64_C(1099511628211); return h;
}
static uint32_t mix_bits(uint32_t x) {
    x^=x>>16; x*=UINT32_C(0x7feb352d); x^=x>>15; x*=UINT32_C(0x846ca68b); return x^(x>>16);
}
static void *contents(id<MTLBuffer> b) { return (char *)b.contents+kOffset; }
/* Exercise packed loads at their promised minimum alignment, including an
 * expert stride of 252 bytes and odd-row tails. */
static void *weight_contents(id<MTLBuffer> b) { return (char *)contents(b)+2; }
static void *activation_contents(id<MTLBuffer> b) { return (char *)contents(b)+4; }
static id<MTLBuffer> make_buffer(id<MTLDevice> d, size_t bytes) {
    id<MTLBuffer> b=[d newBufferWithLength:(bytes+3)/4*4+2*kOffset options:MTLResourceStorageModeShared];
    if (!b) fail(@"buffer allocation"); uint32_t *p=b.contents;
    for (NSUInteger i=0;i<b.length/4;++i) p[i]=kCanary; return b;
}
static void fill_weights(id<MTLBuffer> b, unsigned rows, bool q2, unsigned seed, unsigned pattern) {
    const unsigned stride=q2?252:680, block=q2?84:34;
    uint8_t *p=weight_contents(b); size_t bytes=(size_t)rows*stride;
    for (size_t i=0;i<bytes;++i) p[i]=(uint8_t)mix_bits((uint32_t)i+seed);
    const uint16_t edge[]={0,0x8000,1,0x8001,0x03ff,0x0400,0x3c00,0xbc00,0x7bff,0xfbff};
    for (size_t i=0;i<bytes;i+=block) {
        uint16_t h=(uint16_t)(0x1800u|(mix_bits((uint32_t)i+seed)&0x3ffu));
        if (pattern==4) h=edge[(i/block+seed)%10];
        memcpy(p+i+(q2?80:0),&h,2);
        if (q2) { h=pattern==4?edge[(i/block+seed+3)%10]:(uint16_t)(h^0x81a5u); memcpy(p+i+82,&h,2); }
    }
    if (q2) for (unsigned r=0;r<rows;++r) {
        /* Unused half of block 2 is deliberately nonzero: group 8..15 scale
         * bytes and their packed codes. Live group 0..7 bytes stay untouched. */
        uint8_t *tail=p+(size_t)r*stride+2*84;
        for (unsigned i=8;i<16;++i) tail[i]=(uint8_t)(0x81u|(mix_bits(r+i+seed)&0x7eu));
        for (unsigned i=48;i<80;++i) tail[i]=(uint8_t)(0x55u|(mix_bits(r+i+seed)&0xaau));
    }
}
static moe_args arguments(shape s, unsigned shared_type, Layout layout) {
    bool shared=shared_type!=UINT32_MAX;
    moe_args a={.n_tokens=s.tokens,.n_slots=s.slots,.in_dim=640,.out_rows=s.rows,
        .weight_type=10,.row_bytes=252,.has_shared=shared,.shared_type=shared?shared_type:0,
        .shared_row_bytes=shared?(shared_type==8?680:1280):0,.n_total_expert=0};
    a.expert_bytes=(uint64_t)a.row_bytes*a.out_rows;
    if (layout>=CompactAll) {
        a.masked_tokens=layout==CompactShort?MAX(1u,a.n_tokens-1):a.n_tokens;
        const uint32_t routed=(1u<<a.n_slots)-1u, sh=shared?1u<<a.n_slots:0;
        for (unsigned t=0;t<a.masked_tokens;++t) {
            if (layout==CompactAll || layout==CompactShort) a.slot_mask[t]=routed|sh;
            else if (layout==CompactSparse) a.slot_mask[t]=t==1?0:((0x155u>>t)&routed)|sh;
            else if (layout==CompactShared) a.slot_mask[t]=sh;
            else if (layout==CompactRouted) a.slot_mask[t]=routed;
        }
    }
    return a;
}
/* in = resident weights, selected, mid, shared, address table, separate SSD
 * payloads. The resident generic oracle does not depend on address-table lookup. */
static NSArray<id<MTLBuffer>> *fixture(id<MTLDevice> d, moe_args a, unsigned pattern) {
    id<MTLBuffer> w=make_buffer(d,a.expert_bytes*kExperts+4);
    id<MTLBuffer> ids=make_buffer(d,a.n_tokens*a.n_slots*4);
    id<MTLBuffer> mid=make_buffer(d,(size_t)a.n_tokens*(a.n_slots+a.has_shared)*a.in_dim*4+4);
    id<MTLBuffer> shared=make_buffer(d,(size_t)a.out_rows*a.shared_row_bytes+4);
    id<MTLBuffer> addresses=make_buffer(d,kExperts*8);
    fill_weights(w,a.out_rows*kExperts,true,17,pattern);
    if (a.has_shared && a.shared_type==8) fill_weights(shared,a.out_rows,false,211,pattern);
    else if (a.has_shared) {
        uint16_t *p=weight_contents(shared);
        for (size_t i=0;i<(size_t)a.out_rows*a.in_dim;++i) p[i]=(uint16_t)(0x1800u|(mix_bits((uint32_t)i+211)&0x83ffu));
    }
    int32_t *selected=contents(ids);
    for (unsigned t=0;t<a.n_tokens;++t) for (unsigned s=0;s<a.n_slots;++s)
        selected[t*a.n_slots+s]=(int32_t)((t*7+s*5)%kExperts);
    const uint32_t finite[]={0,0x80000000u,1,0x80000001u,0x007fffffu,0x00800000u,0x80800000u,
        0x3f800000u,0xbf800000u,0x3f800001u,0x5f123456u,0xdf123456u,0x7f7fffffu,0xff7fffffu};
    const uint32_t extreme[]={0x7f800000u,0xff800000u,0x7fc12345u,0xffc54321u,0x7f7fffffu,0};
    float *x=activation_contents(mid); size_t n=(size_t)a.n_tokens*(a.n_slots+a.has_shared)*a.in_dim;
    for (size_t i=0;i<n;++i) {
        x[i]=((int)(mix_bits((uint32_t)i+31)%2049)-1024)/512.f; if (i%7==0) x[i]*=8.f;
        if (pattern==1) memcpy(x+i,&finite[i%14],4);
        if (pattern==2) x[i]=ldexpf((i&1)?-1.f:1.f,(int)(mix_bits((uint32_t)i+47)%61)-30);
        if (pattern==3) memcpy(x+i,&extreme[i%6],4);
    }
    NSMutableArray *in=[@[w,ids,mid,shared,addresses] mutableCopy]; uint64_t *table=contents(addresses);
    /* Separate allocations and a nonzero inner offset prevent contiguous
     * arithmetic from accidentally satisfying an address-table fixture. */
    for (unsigned e=0;e<kExperts;++e) {
        id<MTLBuffer> payload=make_buffer(d,a.expert_bytes+4);
        memcpy(weight_contents(payload),(char *)weight_contents(w)+e*a.expert_bytes,a.expert_bytes);
        table[e]=payload.gpuAddress+kOffset+2; [in addObject:payload];
    }
    return in;
}
static NSArray<NSData *> *snapshot(NSArray<id<MTLBuffer>> *in) {
    NSMutableArray *out=[NSMutableArray new];
    for (id<MTLBuffer> b in in) [out addObject:[NSData dataWithBytes:b.contents length:b.length]]; return out;
}
static void check_inputs(NSArray<id<MTLBuffer>> *in, NSArray<NSData *> *before) {
    for (NSUInteger i=0;i<in.count;++i)
        if (memcmp(in[i].contents,before[i].bytes,in[i].length)) fail(@"input or input guard modified");
}
static void dispatch(id<MTLCommandQueue> q, id<MTLComputePipelineState> p, moe_args a,
                     NSArray<id<MTLBuffer>> *in, id<MTLBuffer> out, bool addressed) {
    unsigned slots=a.n_slots+a.has_shared;
    if (addressed && a.masked_tokens) {
        slots=0;
        for (unsigned t=0;t<a.masked_tokens;++t) slots=MAX(slots,(unsigned)__builtin_popcount(a.slot_mask[t]));
    }
    if (!slots) return; /* Production returns before encoding an empty pass. */
    id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
    if (!cb || !en) fail(@"command allocation");
    [en setComputePipelineState:p]; [en setBytes:&a length:sizeof(a) atIndex:0];
    [en setBuffer:in[addressed?4:0] offset:addressed?kOffset:kOffset+2 atIndex:1];
    [en setBuffer:in[1] offset:kOffset atIndex:2]; [en setBuffer:in[2] offset:kOffset+4 atIndex:3];
    [en setBuffer:out offset:kOffset atIndex:4]; [en setBuffer:in[3] offset:kOffset+2 atIndex:5];
    if (addressed) for (NSUInteger i=5;i<in.count;++i) [en useResource:in[i] usage:MTLResourceUsageRead];
    /* M1 Max automatic down geometry: NR2, four SIMD groups. */
    [en dispatchThreadgroups:MTLSizeMake((a.out_rows+7)/8,slots,a.n_tokens)
           threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status!=MTLCommandBufferStatusCompleted) fail(cb.error.description);
}
static void check_outputs(id<MTLBuffer> reference, id<MTLBuffer> candidate, moe_args a,
                          bool addressed, NSString *label) {
    const uint32_t *ref=contents(reference), *got=contents(candidate); unsigned slots=a.n_slots+a.has_shared;
    for (unsigned t=0;t<a.n_tokens;++t) for (unsigned s=0;s<slots;++s) for (unsigned r=0;r<a.out_rows;++r) {
        size_t i=((size_t)t*slots+s)*a.out_rows+r;
        bool active=!addressed || !a.masked_tokens || (t<a.masked_tokens && (a.slot_mask[t]&(1u<<s)));
        if (!active) { if (got[i]!=kCanary) fail(@"inactive output written"); continue; }
        if (ref[i]==kCanary || got[i]==kCanary) fail(@"active output untouched"); ++compared;
        bool rn=(ref[i]&0x7fffffffu)>0x7f800000u, gn=(got[i]&0x7fffffffu)>0x7f800000u;
        if (rn && gn) { nan_payload_differences+=ref[i]!=got[i]; continue; }
        if (ref[i]!=got[i]) fail([NSString stringWithFormat:@"%@ t%u slot%u row%u: %08x != %08x",label,t,s,r,ref[i],got[i]]);
    }
    for (id<MTLBuffer> b in @[reference,candidate]) {
        const uint32_t *p=b.contents;
        for (NSUInteger i=0;i<kOffset/4;++i)
            if (p[i]!=kCanary || p[b.length/4-1-i]!=kCanary) fail(@"output guard overwritten");
    }
}
static NSArray<id<MTLComputePipelineState>> *pipelines(id<MTLDevice> d, NSString *source, NSMutableArray *metadata) {
    NSMutableArray *result=[NSMutableArray new];
    for (unsigned safe=0;safe<2;++safe) {
        MTLCompileOptions *options=[MTLCompileOptions new];
        if (safe) {
            if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
            else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled=NO;
#pragma clang diagnostic pop
            }
        }
        NSError *error=nil; id<MTLLibrary> lib=[d newLibraryWithSource:source options:options error:&error];
        if (!lib) fail(error.description);
        for (unsigned address=0;address<2;++address) for (unsigned arm=0;arm<2;++arm) {
            NSString *name=arm?@"kernel_qwen4_moe_down_q2k":@"kernel_qwen4_moe_down";
            bool addressed=address!=0; MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
            [constants setConstantValue:&addressed type:MTLDataTypeBool atIndex:906];
            id<MTLFunction> f=[lib newFunctionWithName:name constantValues:constants error:&error]; if (!f) fail(error.description);
            id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&error]; if (!p) fail(error.description);
            if (p.threadExecutionWidth!=32 || p.maxTotalThreadsPerThreadgroup<128) fail(@"unsupported execution geometry");
            [result addObject:p]; [metadata addObject:@{@"kernel":name,@"safe_math":@(safe),@"addresses":@(addressed),
                @"execution_width":@(p.threadExecutionWidth),@"max_threads":@(p.maxTotalThreadsPerThreadgroup)}];
        }
    }
    return result;
}
static void write_json(NSString *directory, NSDictionary *report) {
    if (!directory) return; NSError *error=nil;
    if (![[NSFileManager defaultManager] createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:&error]) fail(error.description);
    NSData *data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
    if (!data || ![data writeToFile:[directory stringByAppendingPathComponent:@"moe-down-tests.json"] options:NSDataWritingAtomic error:&error]) fail(error.description);
}
int main(int argc, const char **argv) { @autoreleasepool {
    NSString *repo=[[NSFileManager defaultManager] currentDirectoryPath], *override=nil, *output=nil; bool test=true;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"--test")) test=true;
        else if (!strcmp(argv[i],"--compile-only")) test=false;
        else if (!strcmp(argv[i],"--repo") && i+1<argc) repo=[NSString stringWithUTF8String:argv[++i]];
        else if (!strcmp(argv[i],"--qwen-source") && i+1<argc) override=[NSString stringWithUTF8String:argv[++i]];
        else if (!strcmp(argv[i],"--output") && i+1<argc) output=[NSString stringWithUTF8String:argv[++i]];
        else { fprintf(stderr,"usage: %s [--test|--compile-only] [--repo PATH] [--qwen-source FILE] [--output DIR]\n",argv[0]); return 2; }
    }
    NSString *source=production_source(repo,override); id<MTLDevice> d=MTLCreateSystemDefaultDevice(); if (!d) fail(@"Metal device unavailable");
    NSMutableArray *metadata=[NSMutableArray new]; NSArray *ps=pipelines(d,source,metadata);
    NSMutableDictionary *report=[@{@"schema_version":@1,@"device":d.name,@"pipelines":metadata,
        @"source_fnv1a64":[NSString stringWithFormat:@"%016llx",(unsigned long long)source_hash(source)],
        @"qwen_source":override ?: [repo stringByAppendingPathComponent:@"metal/qwen4.metal"]} mutableCopy];
    if (!test) { report[@"status"]=@"compile_only_pass"; write_json(output,report); puts("PASS Qwen MoE down: eight pipelines; no queue/buffers/dispatches"); return 0; }
    id<MTLCommandQueue> q=[d newCommandQueue]; if (!q) fail(@"queue allocation");
    for (unsigned safe=0;safe<2;++safe) for (unsigned si=0;si<sizeof(kShapes)/sizeof(kShapes[0]);++si)
        for (unsigned shared=0;shared<3;++shared) for (unsigned pattern=0;pattern<5;++pattern) { @autoreleasepool {
            /* Half shared fallback needs one odd-row T3 shape; Q8 and no-shared
             * cover every shape. Down receives n_total_expert=0, like its host wrapper; selected IDs are valid. */
            if (shared==2 && si!=2) continue;
            unsigned st=shared==0?UINT32_MAX:shared==1?8u:1u;
            moe_args full=arguments(kShapes[si],st,Resident);
            NSArray *in=fixture(d,full,pattern), *before=snapshot(in);
            size_t bytes=(size_t)full.n_tokens*(full.n_slots+full.has_shared)*full.out_rows*4;
            id<MTLBuffer> ref=make_buffer(d,bytes); dispatch(q,ps[safe*4],full,in,ref,false);
            for (unsigned layout=0;layout<LayoutCount;++layout) { @autoreleasepool {
                moe_args a=arguments(kShapes[si],st,(Layout)layout); bool addressed=layout!=Resident;
                for (unsigned arm=addressed?0u:1u;arm<2;++arm) {
                    id<MTLBuffer> got=make_buffer(d,bytes);
                    dispatch(q,ps[safe*4+addressed*2+arm],a,in,got,addressed);
                    check_outputs(ref,got,a,addressed,[NSString stringWithFormat:@"safe%u T%u R%u shared%u pattern%u layout%u arm%u",safe,a.n_tokens,a.out_rows,st,pattern,layout,arm]);
                    ++fixtures;
                }
            }}
            check_inputs(in,before);
        }}
    report[@"status"]=@"pass"; report[@"fixtures"]=@(fixtures); report[@"values_compared"]=@(compared);
    report[@"nan_payload_differences"]=@(nan_payload_differences);
    report[@"contract"]=@"Non-NaN bitwise; NaN classification; unmasked resident oracle; separate SSD payloads; immutable inputs; output/inactive-slot canaries; nonzero Q2 padding";
    write_json(output,report);
    printf("PASS Qwen MoE down: %llu fixtures, %llu comparisons, %llu NaN payload differences\n",(unsigned long long)fixtures,(unsigned long long)compared,(unsigned long long)nan_payload_differences);
    return 0;
}}
