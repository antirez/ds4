/* Compare the production Qwen IQ2 mid NR1 and NR2 exports.
 * Build: clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
 *            tests/test_metal_qwen4_moe_mid.m -o /tmp/test_metal_qwen4_moe_mid
 * Run from the repository root, or use --repo PATH. Default/--test checks
 * default/safe math, resident/SSD addressing, masks, shared rows and tails.
 * --compile-only creates no queue/buffers/dispatches. --bench adds a warmed
 * kernel ABBA/BAAB comparison; --output DIR saves JSON with every observation.
 * Kernels, argument layout, helpers and tables are extracted from production
 * Metal files. Shared results also use a separate-row-dot oracle; host dispatch policy is separate.
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
#include <time.h>

typedef struct {
    uint32_t n_tokens, n_slots, in_dim, out_rows, weight_type, row_bytes;
    uint64_t expert_bytes;
    uint32_t has_shared, shared_type, shared_row_bytes, n_total_expert;
    uint32_t slot_mask[3], masked_tokens;
    uint32_t list_cap, pad0;
} moe_args;
_Static_assert(sizeof(moe_args) == 72, "production MoE argument ABI");
_Static_assert(offsetof(moe_args, expert_bytes) == 24, "expert stride alignment");
_Static_assert(offsetof(moe_args, masked_tokens) == 60, "slot mask ABI");
_Static_assert(offsetof(moe_args, list_cap) == 64, "grouped list capacity ABI");
typedef struct { uint32_t tokens, width, rows, slots; } shape;
static const shape kShapes[] = {
    {1, 256, 1, 2}, {2, 512, 7, 2}, {2, 768, 9, 10},
    {1, 2560, 640, 10}, {2, 2560, 641, 10},
};
static const NSUInteger kOffset = 16;
static const uint32_t kCanary = UINT32_C(0x7f24d13b);
static uint64_t compared, nan_payload_differences, fixtures, shared_oracle_fixtures;

static void fail(NSString *message) {
    fprintf(stderr, "FAIL Qwen MoE mid: %s\n", message.UTF8String);
    exit(1);
}
static NSString *read_source(NSString *repo, NSString *relative) {
    NSError *error = nil;
    NSString *path = [repo stringByAppendingPathComponent:relative];
    NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (!source) fail([NSString stringWithFormat:@"read %@: %@", path, error]);
    return source;
}
static NSString *between(NSString *source, NSString *begin, NSString *end) {
    if ([source componentsSeparatedByString:begin].count != 2 ||
        [source componentsSeparatedByString:end].count != 2) fail(@"production source anchor missing/nonunique");
    NSRange a = [source rangeOfString:begin], b = [source rangeOfString:end];
    if (b.location <= NSMaxRange(a)) fail(@"invalid production source range");
    return [source substringWithRange:NSMakeRange(a.location, b.location - a.location)];
}
static NSString *production_source(NSString *repo) {
    NSString *qwen = read_source(repo, @"metal/qwen4.metal");
    NSString *moe = read_source(repo, @"metal/moe.metal");
    NSMutableString *source = [NSMutableString stringWithString:@"#include <metal_stdlib>\nusing namespace metal;\n"];
    [source appendString:between(moe, @"static constant float ds4_metal_mxfp4_values[16]",
                                     @"// BEGIN GENERATED MXFP4 HALF LUT")];
    [source appendString:between(moe, @"static constant uchar ds4_metal_ksigns_iq2xs[128]",
                                     @"#define kmask_iq2xs")];
    [source appendString:between(qwen, @"static inline float qwen4_sigmoid(",
                                      @"/* --- hyper-connections")];
    [source appendString:between(qwen, @"constant bool qwen4_expert_addresses ",
                                      @"/* --- small multi-output GEMV")];
    [source appendString:between(qwen, @"/* IQ2XXS gate/up input reuse,",
                                      @"/* Q4_K gate/up input reuse")];
    /* Independent shared-slot oracle: the unchanged row dot still performs
     * two separate projection loops, even when both IQ2 exports fuse Q8. */
    [source appendString:
        @"kernel void kernel_qwen4_moe_shared_dot_oracle(\n"
        @" constant ds4_metal_args_qwen4_moe &args, device const char *gate,\n"
        @" device const char *up, device const int32_t *selected, device const float *x,\n"
        @" device float *mid, device const char *sh_gate, device const char *sh_up,\n"
        @" uint3 pos [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],\n"
        @" ushort sg [[simdgroup_index_in_threadgroup]], ushort3 ntg [[threads_per_threadgroup]]) {\n"
        @" const uint slot=qwen4_moe_slot(args,pos.y,pos.z), tok=pos.z;\n"
        @" const uint row=pos.x*(ntg.x/32u)+sg, slots=args.n_slots+args.has_shared;\n"
        @" if (!args.has_shared || slot!=args.n_slots || tok>=args.n_tokens || row>=args.out_rows) return;\n"
        @" const uint64_t off=(uint64_t)row*args.shared_row_bytes;\n"
        @" device const float *xt=x+(uint64_t)tok*args.in_dim;\n"
        @" const float g=qwen4_row_dot(sh_gate+off,xt,args.shared_type,args.in_dim,lane);\n"
        @" const float u=qwen4_row_dot(sh_up+off,xt,args.shared_type,args.in_dim,lane);\n"
        @" if (lane==0) mid[((uint64_t)tok*slots+slot)*args.out_rows+row]=qwen4_silu(g)*u;\n"
        @"}\n"];
    return source;
}
static uint64_t source_hash(NSString *source) {
    NSData *data = [source dataUsingEncoding:NSUTF8StringEncoding];
    const uint8_t *bytes = data.bytes;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (NSUInteger i = 0; i < data.length; ++i) hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    return hash;
}
static uint32_t mix_bits(uint32_t x) {
    x ^= x >> 16; x *= UINT32_C(0x7feb352d);
    x ^= x >> 15; x *= UINT32_C(0x846ca68b); return x ^ (x >> 16);
}
static void *contents(id<MTLBuffer> buffer) { return (char *)buffer.contents + kOffset; }
static id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes) {
    bytes = (bytes + 3) / 4 * 4;
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes + 2 * kOffset options:MTLResourceStorageModeShared];
    if (!buffer) fail(@"buffer allocation");
    uint32_t *words = buffer.contents;
    for (NSUInteger i = 0; i < buffer.length / 4; ++i) words[i] = kCanary;
    return buffer;
}
static void fill_weights(id<MTLBuffer> buffer, uint32_t rows, uint32_t width, bool iq2, uint32_t seed) {
    const uint32_t block_bytes = iq2 ? 66 : 34, block_width = iq2 ? 256 : 32;
    const size_t bytes = (size_t)rows * (width / block_width) * block_bytes;
    uint8_t *weights = contents(buffer);
    for (size_t i = 0; i < bytes; ++i) weights[i] = (uint8_t)mix_bits((uint32_t)i + seed);
    for (size_t i = 0; i < bytes; i += block_bytes) {
        uint16_t scale = (uint16_t)(0x1800u | (mix_bits((uint32_t)i + seed) & 0x3ffu));
        memcpy(weights + i, &scale, sizeof(scale));
    }
}
static moe_args arguments(shape s, bool shared, bool masked) {
    moe_args a = { .n_tokens=s.tokens, .n_slots=s.slots, .in_dim=s.width, .out_rows=s.rows,
        .weight_type=16, .row_bytes=s.width/256*66, .has_shared=shared,
        .shared_type=8, .shared_row_bytes=s.width/32*34, .n_total_expert=12 };
    a.expert_bytes = (uint64_t)a.row_bytes * a.out_rows;
    if (masked) {
        a.masked_tokens = a.n_tokens;
        a.slot_mask[0] = 1u | (shared ? 1u << a.n_slots : 0u);
        if (a.n_slots > 2) a.slot_mask[0] |= 1u << 2;
        /* A zero second-token mask must leave its entire output untouched. */
    }
    return a;
}
static NSArray<id<MTLBuffer>> *fixture(id<MTLDevice> device, moe_args a, unsigned pattern) {
    id<MTLBuffer> gate = make_buffer(device, a.expert_bytes*a.n_total_expert);
    id<MTLBuffer> up = make_buffer(device, a.expert_bytes*a.n_total_expert);
    id<MTLBuffer> selected = make_buffer(device, a.n_tokens*a.n_slots*4);
    id<MTLBuffer> input = make_buffer(device, a.n_tokens*a.in_dim*4);
    id<MTLBuffer> shared_gate = make_buffer(device, a.out_rows*a.shared_row_bytes);
    id<MTLBuffer> shared_up = make_buffer(device, a.out_rows*a.shared_row_bytes);
    id<MTLBuffer> gate_addresses = make_buffer(device, a.n_total_expert*8);
    id<MTLBuffer> up_addresses = make_buffer(device, a.n_total_expert*8);
    fill_weights(gate, a.out_rows*a.n_total_expert, a.in_dim, true, 17);
    fill_weights(up, a.out_rows*a.n_total_expert, a.in_dim, true, 9701);
    fill_weights(shared_gate, a.out_rows, a.in_dim, false, 211);
    fill_weights(shared_up, a.out_rows, a.in_dim, false, 407);
    int32_t *ids = contents(selected);
    for (uint32_t t=0; t<a.n_tokens; ++t) for (uint32_t s=0; s<a.n_slots; ++s)
        ids[t*a.n_slots+s] = (int32_t)((t*7+s*5)%a.n_total_expert);
    if (pattern == 2) { ids[0] = -1; ids[1] = (int32_t)a.n_total_expert; }
    float *x = contents(input);
    const uint32_t finite[] = {0,0x80000000u,1,0x80000001u,0x00800000u,0x80800000u,0x3f800000u,0xbf800000u};
    const uint32_t extreme[] = {0x7f800000u,0xff800000u,0x7fc12345u,0x7f7fffffu,0xff7fffffu,0};
    for (uint32_t i=0; i<a.n_tokens*a.in_dim; ++i) {
        x[i] = ((int)(mix_bits(i+31)%2049)-1024)/512.f;
        if (i%7 == 0) x[i] *= 8.f;
        if (pattern) memcpy(x+i, pattern == 1 ? &finite[i%8] : &extreme[i%6], 4);
    }
    uint64_t *ga=contents(gate_addresses), *ua=contents(up_addresses);
    for (uint32_t e=0; e<a.n_total_expert; ++e) {
        ga[e]=gate.gpuAddress+kOffset+e*a.expert_bytes;
        ua[e]=up.gpuAddress+kOffset+e*a.expert_bytes;
    }
    return @[gate,up,selected,input,shared_gate,shared_up,gate_addresses,up_addresses];
}
static NSArray<NSData *> *snapshot(NSArray<id<MTLBuffer>> *buffers) {
    NSMutableArray *result=[NSMutableArray new];
    for (id<MTLBuffer> buffer in buffers) [result addObject:[NSData dataWithBytes:buffer.contents length:buffer.length]];
    return result;
}
static void check_inputs(NSArray<id<MTLBuffer>> *buffers, NSArray<NSData *> *before) {
    for (NSUInteger i=0; i<buffers.count; ++i)
        if (memcmp(buffers[i].contents,before[i].bytes,buffers[i].length)) fail(@"input or input guard modified");
}
static NSDictionary *dispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
        moe_args a, NSArray<id<MTLBuffer>> *in, id<MTLBuffer> output, bool addresses, unsigned nr, unsigned repeats) {
    unsigned slots=a.n_slots+a.has_shared;
    if (addresses && a.masked_tokens) {
        slots=0;
        for (unsigned t=0;t<a.masked_tokens;++t) slots=MAX(slots,(unsigned)__builtin_popcount(a.slot_mask[t]));
    }
    struct timespec start,end; clock_gettime(CLOCK_MONOTONIC,&start);
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if (!command || !encoder) fail(@"command allocation");
    [encoder setComputePipelineState:pipeline]; [encoder setBytes:&a length:sizeof(a) atIndex:0];
    [encoder setBuffer:in[addresses?6:0] offset:kOffset atIndex:1];
    [encoder setBuffer:in[addresses?7:1] offset:kOffset atIndex:2];
    [encoder setBuffer:in[2] offset:kOffset atIndex:3]; [encoder setBuffer:in[3] offset:kOffset atIndex:4];
    [encoder setBuffer:output offset:kOffset atIndex:5];
    [encoder setBuffer:in[4] offset:kOffset atIndex:6]; [encoder setBuffer:in[5] offset:kOffset atIndex:7];
    if (addresses) { [encoder useResource:in[0] usage:MTLResourceUsageRead]; [encoder useResource:in[1] usage:MTLResourceUsageRead]; }
    for (unsigned i=0; i<repeats; ++i)
        [encoder dispatchThreadgroups:MTLSizeMake((a.out_rows+4*nr-1)/(4*nr),slots,a.n_tokens)
                 threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) fail(command.error.description);
    clock_gettime(CLOCK_MONOTONIC,&end);
    double host=(end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)*1e-9;
    double gpu=command.GPUEndTime-command.GPUStartTime;
    if (!isfinite(gpu) || gpu<=0) fail(@"invalid GPU timing");
    return @{@"gpu_us":@(gpu*1e6/repeats),@"host_us":@(host*1e6/repeats)};
}
static void check_outputs(id<MTLBuffer> reference, id<MTLBuffer> candidate, moe_args a,
                          const int32_t *selected, NSString *label) {
    const uint32_t *ref=contents(reference), *got=contents(candidate);
    const unsigned slots=a.n_slots+a.has_shared;
    for (unsigned t=0; t<a.n_tokens; ++t) for (unsigned s=0; s<slots; ++s) for (unsigned r=0; r<a.out_rows; ++r) {
        size_t i=((size_t)t*slots+s)*a.out_rows+r;
        bool active=!a.masked_tokens || (a.slot_mask[t] & (1u<<s));
        bool invalid=s<a.n_slots && (selected[t*a.n_slots+s]<0 || (uint32_t)selected[t*a.n_slots+s]>=a.n_total_expert);
        if (!active && (ref[i]!=kCanary || got[i]!=kCanary)) fail(@"masked output was written");
        if (active && invalid && (ref[i]!=0 || got[i]!=0)) fail(@"invalid expert output was not zero");
        if (active && (ref[i]==kCanary || got[i]==kCanary)) fail(@"active output was not written");
        bool rn=(ref[i]&0x7fffffffu)>0x7f800000u, gn=(got[i]&0x7fffffffu)>0x7f800000u;
        ++compared;
        if (rn && gn) { nan_payload_differences += ref[i]!=got[i]; continue; }
        if (ref[i]!=got[i]) fail([NSString stringWithFormat:@"%@ output %lu: %08x != %08x",label,(unsigned long)i,ref[i],got[i]]);
    }
    for (id<MTLBuffer> output in @[reference,candidate]) {
        const uint32_t *words=output.contents;
        for (NSUInteger i=0; i<kOffset/4; ++i)
            if (words[i]!=kCanary || words[output.length/4-1-i]!=kCanary) fail(@"output guard overwritten");
    }
}
static NSArray<id<MTLComputePipelineState>> *pipelines(id<MTLDevice> device, NSString *source, NSMutableArray *metadata) {
    NSMutableArray *result=[NSMutableArray new];
    for (unsigned safe=0; safe<2; ++safe) {
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
        NSError *error=nil;
        id<MTLLibrary> library=[device newLibraryWithSource:source options:options error:&error];
        if (!library) fail(error.description);
        for (unsigned address=0; address<2; ++address) for (unsigned arm=0; arm<3; ++arm) {
            NSString *name=arm==2?@"kernel_qwen4_moe_shared_dot_oracle":
                arm?@"kernel_qwen4_moe_mid_iq2_nr1":@"kernel_qwen4_moe_mid_iq2";
            bool addresses=address!=0;
            MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
            [constants setConstantValue:&addresses type:MTLDataTypeBool atIndex:906];
            id<MTLFunction> function=[library newFunctionWithName:name constantValues:constants error:&error];
            if (!function) fail(error.description);
            id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) fail(error.description);
            [result addObject:pipeline];
            [metadata addObject:@{@"kernel":name,@"safe_math":@(safe),@"addresses":@(addresses),
                @"execution_width":@(pipeline.threadExecutionWidth),@"max_threads":@(pipeline.maxTotalThreadsPerThreadgroup),
                @"static_threadgroup_bytes":@(pipeline.staticThreadgroupMemoryLength)}];
        }
    }
    return result;
}
static void write_json(NSString *directory, NSString *name, id value) {
    if (!directory) return;
    NSError *error=nil;
    if (![[NSFileManager defaultManager] createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:&error]) fail(error.description);
    NSData *data=[NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
    if (!data || ![data writeToFile:[directory stringByAppendingPathComponent:name] options:NSDataWritingAtomic error:&error]) fail(error.description);
}
static double median(NSArray<NSNumber *> *values) {
    NSArray<NSNumber *> *sorted=[values sortedArrayUsingSelector:@selector(compare:)];
    NSUInteger n=sorted.count; return (sorted[(n-1)/2].doubleValue+sorted[n/2].doubleValue)/2;
}
int main(int argc, const char **argv) { @autoreleasepool {
    NSString *repo=[[NSFileManager defaultManager] currentDirectoryPath], *output=nil;
    bool test=true, bench=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"--test")) test=true;
        else if (!strcmp(argv[i],"--compile-only")) test=false;
        else if (!strcmp(argv[i],"--bench")) bench=true;
        else if (!strcmp(argv[i],"--repo") && i+1<argc) repo=[NSString stringWithUTF8String:argv[++i]];
        else if (!strcmp(argv[i],"--output") && i+1<argc) output=[NSString stringWithUTF8String:argv[++i]];
        else { fprintf(stderr,"usage: %s [--test|--compile-only] [--bench] [--repo PATH] [--output DIR]\n",argv[0]); return 2; }
    }
    NSString *source=production_source(repo);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice(); if (!device) fail(@"Metal device unavailable");
    NSMutableArray *pipeline_metadata=[NSMutableArray new];
    NSArray *ps=pipelines(device,source,pipeline_metadata);
    NSMutableDictionary *report=[@{@"schema_version":@1,@"device":device.name,@"source_fnv1a64":[NSString stringWithFormat:@"%016llx",(unsigned long long)source_hash(source)],@"pipelines":pipeline_metadata} mutableCopy];
    if (!test && !bench) {
        report[@"status"]=@"compile_only_pass"; write_json(output,@"moe-mid-tests.json",report);
        printf("PASS Qwen MoE mid: twelve pipelines, no queues/buffers/dispatches\n"); return 0;
    }
    id<MTLCommandQueue> queue=[device newCommandQueue]; if (!queue) fail(@"queue allocation");
    if (test) for (unsigned safe=0;safe<2;++safe) for (unsigned si=0;si<sizeof(kShapes)/sizeof(kShapes[0]);++si)
        for (unsigned shared=0;shared<2;++shared) for (unsigned pattern=0;pattern<3;++pattern) for (unsigned layout=0;layout<3;++layout) { @autoreleasepool {
            moe_args a=arguments(kShapes[si],shared,layout==2);
            NSArray *in=fixture(device,a,pattern), *before=snapshot(in);
            size_t bytes=(size_t)a.n_tokens*(a.n_slots+a.has_shared)*a.out_rows*4;
            id<MTLBuffer> ref=make_buffer(device,bytes), got=make_buffer(device,bytes);
            bool addresses=layout!=0; unsigned index=safe*6+addresses*3;
            dispatch(queue,ps[index],a,in,ref,addresses,2,1); dispatch(queue,ps[index+1],a,in,got,addresses,1,1);
            check_outputs(ref,got,a,contents(in[2]),[NSString stringWithFormat:@"safe%u shape%u shared%u pattern%u layout%u",safe,si,shared,pattern,layout]);
            if (shared) {
                /* Copy only as a routed-output placeholder; every shared row
                 * is poisoned before the separate-dot oracle writes it. */
                id<MTLBuffer> oracle=make_buffer(device,bytes);
                memcpy(oracle.contents,got.contents,got.length);
                uint32_t *values=contents(oracle);
                for (unsigned t=0; t<a.n_tokens; ++t) for (unsigned r=0; r<a.out_rows; ++r)
                    values[((size_t)t*(a.n_slots+1)+a.n_slots)*a.out_rows+r]=kCanary;
                dispatch(queue,ps[index+2],a,in,oracle,addresses,1,1);
                check_outputs(got,oracle,a,contents(in[2]),@"independent shared row-dot oracle");
                ++shared_oracle_fixtures;
            }
            check_inputs(in,before); ++fixtures;
        }}
    if (bench) {
        NSMutableArray *records=[NSMutableArray new], *summaries=[NSMutableArray new];
        for (unsigned count=8;count<=11;count+=3) { @autoreleasepool {
            moe_args a=arguments((shape){1,2560,640,10},true,true);
            a.slot_mask[0]=((1u<<(count-1))-1u)|(1u<<a.n_slots);
            NSArray *in=fixture(device,a,0), *before=snapshot(in);
            size_t bytes=(a.n_slots+1)*a.out_rows*4;
            NSArray *out=@[make_buffer(device,bytes),make_buffer(device,bytes)];
            NSMutableArray<NSNumber *> *samples[2]={[NSMutableArray new],[NSMutableArray new]};
            for (unsigned arm=0;arm<2;++arm) dispatch(queue,ps[3+arm],a,in,out[arm],true,arm?1:2,128);
            const unsigned order[]={0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1};
            for (unsigned step=0;step<16;++step) {
                unsigned arm=order[step];
                NSMutableDictionary *row=[dispatch(queue,ps[3+arm],a,in,out[arm],true,arm?1:2,128) mutableCopy];
                check_outputs(out[0],out[1],a,contents(in[2]),@"benchmark");
                row[@"nr"]=@(arm?1:2); row[@"dispatch_slots"]=@(count); row[@"step"]=@(step); row[@"repeats"]=@128;
                [records addObject:row]; [samples[arm] addObject:row[@"gpu_us"]];
            }
            check_inputs(in,before);
            double base=median(samples[0]), candidate=median(samples[1]);
            [summaries addObject:@{@"dispatch_slots":@(count),@"nr2_median_gpu_us":@(base),@"nr1_median_gpu_us":@(candidate),@"speedup_pct":@((base/candidate-1)*100)}];
            printf("MoE mid slots=%u: NR2 %.3f us, NR1 %.3f us, %+.2f%%\n",count,base,candidate,(base/candidate-1)*100);
        }}
        write_json(output,@"moe-mid-bench-runs.json",records); write_json(output,@"moe-mid-bench-summary.json",summaries);
    }
    report[@"status"]=@"pass"; report[@"fixtures"]=@(fixtures); report[@"values_compared"]=@(compared);
    report[@"shared_oracle_fixtures"]=@(shared_oracle_fixtures);
    report[@"nan_payload_differences"]=@(nan_payload_differences); report[@"contract"]=@"Non-NaN bitwise; NaN classification; immutable inputs; output and inactive-slot canaries";
    write_json(output,@"moe-mid-tests.json",report);
    printf("PASS Qwen MoE mid: %llu fixtures, %llu comparisons, %llu NaN payload differences\n",(unsigned long long)fixtures,(unsigned long long)compared,(unsigned long long)nan_payload_differences);
    return 0;
}}
