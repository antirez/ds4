// SPDX-License-Identifier: MIT
// Model-free Q-b oracle: the same M64/N32/K32 kernel consumes F32 or a once
// converted F16 activation. Benchmark times include conversion and both encoders.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { K = 1024, MAX_M = 32768, MAX_N = 2048, GUARD = 256 };
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } Q4;
typedef struct {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1;
    int16_t r2, r3;
} MM;
_Static_assert(sizeof(Q4) == 144, "Q4_K ABI");
_Static_assert(sizeof(MM) == 88, "mul_mm ABI");
static const uint32_t canary = 0x7fc12345u;
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> mm32[2], mm16[2], convert;
static id<MTLBuffer> weights, input, rhs, baseline, candidate;
static size_t weight_stride;
static uint32_t sets = 8, samples = 12, dispatches = 4;
static uint32_t rng_state = 731;
static void require(bool ok, const char *message) {
    if (!ok) { fprintf(stderr, "Metal Q4 activation FAIL: %s\n", message); exit(1); }
}
static uint32_t random32(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5; return rng_state;
}
static void *payload(id<MTLBuffer> b) { return (char *)b.contents + GUARD; }
static void poison(id<MTLBuffer> b, uint32_t value) {
    uint32_t *p = b.contents;
    for (size_t i = 0; i < b.length / 4; ++i) p[i] = value;
}
static id<MTLBuffer> buffer(size_t bytes) {
    id<MTLBuffer> b = [device newBufferWithLength:bytes + 2*GUARD
                                        options:MTLResourceStorageModeShared];
    require(b != nil, "buffer allocation"); poison(b, canary); return b;
}
static void guards(id<MTLBuffer> b, size_t active) {
    const uint32_t *p = b.contents;
    for (size_t i = 0; i < GUARD/4; ++i) require(p[i] == canary, "prefix guard");
    for (size_t i = (GUARD + active)/4; i < b.length/4; ++i)
        require(p[i] == canary, "suffix guard/inactive payload");
}
static NSString *source(void) {
    NSMutableString *s = [NSMutableString stringWithString:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
        "#define MAX(x,y) ((x)>(y)?(x):(y))\n#define MIN(x,y) ((x)<(y)?(x):(y))\n"
        "#define SWAP(x,y) { auto t=(x); (x)=(y); (y)=t; }\n"
        "#define QK8_0 32\n#define QK_K 256\n#define N_SIMDWIDTH 32\n"
        "#define N_R0_Q8_0 2\n#define N_SG_Q8_0 4\n"
        "#define FC_MUL_MV 600\n#define FC_MUL_MM 700\n#define FC_BIN 1300\n"
        "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
        "#define M_PI_F 3.14159265358979323846f\n"
        "enum ds4_sort_order { DS4_SORT_ORDER_ASC, DS4_SORT_ORDER_DESC };\n"
        "struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n"
        "struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };\n"];
    const char *root = getenv("DS4_SOURCE_ROOT");
    NSString *directory = root ? [NSString stringWithUTF8String:root] : @".";
    // These are the production sources in their normal concatenation order.
    for (NSString *name in @[@"flash_attn", @"dense", @"glm53_bf16", @"glm53_vision",
            @"deepseek4_vision", @"glm53_kda", @"moe",
            @"dsv4_hc", @"unary", @"dsv4_kv", @"dsv4_rope", @"dsv4_misc",
            @"argsort", @"cpy", @"concat", @"get_rows", @"sum_rows",
            @"softmax", @"repeat", @"glu", @"norm", @"bin", @"set_rows"]) {
        NSString *path = [directory stringByAppendingPathComponent:
                          [NSString stringWithFormat:@"metal/%@.metal", name]];
        NSError *error = nil;
        NSString *part = [NSString stringWithContentsOfFile:path
                            encoding:NSUTF8StringEncoding error:&error];
        require(part != nil, error.localizedDescription.UTF8String);
        [s appendFormat:@"\n%@\n", part];
    }
    return s;
}
static id<MTLComputePipelineState> pipeline(id<MTLLibrary> library,
        NSString *name, bool matmul, bool bc_out) {
    NSError *error = nil;
    id<MTLFunction> function;
    if (matmul) {
        bool no = false;
        MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
        [values setConstantValue:&no type:MTLDataTypeBool atIndex:700];
        [values setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];
        function = [library newFunctionWithName:name constantValues:values error:&error];
    } else function = [library newFunctionWithName:name];
    require(function != nil, error ? error.localizedDescription.UTF8String : name.UTF8String);
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:function error:&error];
    require(p != nil, error.localizedDescription.UTF8String);
    require(p.maxTotalThreadsPerThreadgroup >= 128, "threadgroup limit"); return p;
}
static void encode_convert(id<MTLCommandBuffer> cb, uint32_t n) {
    const uint32_t count = n*K;
    const NSUInteger work_items = (count+3u)/4u;
    // Mirror ds4_gpu_cpy_threads: the runtime selects by vector work items.
    NSUInteger threads = 32u;
    const NSUInteger max_threads = convert.maxTotalThreadsPerThreadgroup;
    while (threads < work_items && threads < max_threads) threads *= 2u;
    if (threads > max_threads) threads = max_threads;
    if (threads > work_items) threads = work_items;
    if (!threads) threads = 1u;
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:convert]; [e setBytes:&count length:4 atIndex:0];
    [e setBuffer:input offset:GUARD atIndex:1]; [e setBuffer:rhs offset:GUARD atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake((work_items+threads-1)/threads,1,1)
         threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [e endEncoding];
}
static void encode_mm(id<MTLCommandBuffer> cb, bool half, uint32_t m,
        uint32_t n, uint32_t set) {
    const uint64_t row = (K/256)*sizeof(Q4), elem = half ? 2 : 4;
    const MM a = {.ne00=K, .ne02=1, .nb01=row, .nb02=row*m, .nb03=row*m,
        .ne12=1, .nb10=elem, .nb11=K*elem, .nb12=K*n*elem, .nb13=K*n*elem,
        .ne0=(int32_t)m, .ne1=(int32_t)n, .r2=1, .r3=1};
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const bool bc_out = m%64 || n%32;
    [e setComputePipelineState:half ? mm16[bc_out] : mm32[bc_out]];
    [e setThreadgroupMemoryLength:bc_out ? 8192 : 6144 atIndex:0];
    [e setBytes:&a length:sizeof(a) atIndex:0];
    [e setBuffer:weights offset:GUARD + set*weight_stride atIndex:1];
    [e setBuffer:half ? rhs : input offset:GUARD atIndex:2];
    [e setBuffer:half ? candidate : baseline offset:GUARD atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake((n+31)/32,(m+63)/64,1)
         threadsPerThreadgroup:MTLSizeMake(128,1,1)]; [e endEncoding];
}
static double workload(bool half, uint32_t m, uint32_t n,
        uint32_t calls, uint32_t start) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        for (uint32_t i = 0; i < calls; ++i) {
            // A fresh conversion for every projection; do not amortize it away.
            if (half) encode_convert(cb, n);
            encode_mm(cb, half, m, n, (start+i)%sets);
        }
        [cb commit]; [cb waitUntilCompleted];
        require(cb.status == MTLCommandBufferStatusCompleted,
                cb.error ? cb.error.localizedDescription.UTF8String : "GPU submission");
        const double elapsed = cb.GPUEndTime - cb.GPUStartTime;
        require(elapsed > 0, "GPU timestamps unavailable"); return elapsed/calls;
    }
}
static void oracle(uint32_t m, uint32_t n, uint32_t set) {
    poison(baseline, canary); poison(candidate, canary); poison(rhs, canary);
    const size_t count = (size_t)m*n;
    uint32_t *a = payload(baseline), *b = payload(candidate);
    for (size_t i = 0; i < count; ++i) { a[i]=0x7fc0b001u; b[i]=0x7fc0c001u; }
    workload(false,m,n,1,set); workload(true,m,n,1,set);
    for (size_t i = 0; i < count; ++i) {
        require((a[i]&0x7f800000u) != 0x7f800000u &&
                (b[i]&0x7f800000u) != 0x7f800000u, "unwritten/nonfinite output");
        if (a[i] != b[i]) {
            fprintf(stderr,"M=%u N=%u set=%u word=%zu: %08x != %08x\n",m,n,set,i,a[i],b[i]);
            require(false,"output is not bitwise equal");
        }
    }
    const float *x = payload(input); const uint16_t *h = payload(rhs);
    for (size_t i = 0; i < (size_t)n*K; ++i) {
        _Float16 expected = (_Float16)x[i]; uint16_t bits; memcpy(&bits,&expected,2);
        require(h[i] == bits, "conversion differs from host IEEE F16 rounding");
    }
    guards(baseline,count*4); guards(candidate,count*4); guards(rhs,(size_t)n*K*2);
    printf("PASS bitwise M=%u N=%u set=%u (F16 conversion, output poison, tails, guards)\n",m,n,set);
    fflush(stdout);
}
static int compare_double(const void *a,const void *b) {
    double x=*(const double *)a,y=*(const double *)b; return (x>y)-(x<y);
}
static double median(double *x) {
    qsort(x,samples,sizeof(*x),compare_double); return (x[samples/2-1]+x[samples/2])*0.5;
}
static void benchmark(uint32_t n) {
    double a[64], b[64]; uint32_t na=0,nb=0;
    for (uint32_t i=0;i<4;++i) workload(i==1||i==2,MAX_M,n,dispatches,i*dispatches);
    for (uint32_t cycle=0;cycle<samples/2;++cycle) {
        const bool abba[] = {false,true,true,false}, baab[] = {true,false,false,true};
        const bool *order = cycle%2 ? baab : abba;
        for (uint32_t j=0;j<4;++j) {
            const bool half=order[j]; const uint32_t index=half ? nb : na;
            double elapsed=workload(half,MAX_M,n,dispatches,(index*dispatches)%sets)*1e6;
            if (half) b[nb++]=elapsed; else a[na++]=elapsed;
        }
    }
    const double am=median(a), bm=median(b);
    printf("BENCH M=%u N=%u baseline_f32_us=%.3f candidate_copy_f16_us=%.3f "
           "speedup_percent=%+.2f time_saved_percent=%+.2f "
           "baseline_range_us=%.3f:%.3f candidate_range_us=%.3f:%.3f\n",
           MAX_M,n,am,bm,(am/bm-1)*100,(1-bm/am)*100,a[0],a[samples-1],b[0],b[samples-1]);
    fflush(stdout);
}
static uint32_t option(const char *text,uint32_t min,uint32_t max) {
    char *end=NULL; unsigned long value=strtoul(text,&end,10);
    if (!text[0] || !end || *end || value<min || value>max) {
        fprintf(stderr,"invalid option value: %s\n",text); exit(2);
    }
    return (uint32_t)value;
}
int main(int argc,char **argv) { @autoreleasepool {
    bool bench=false, bench_small=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"--bench")) bench=true;
        else if (!strcmp(argv[i],"--bench-small")) bench=bench_small=true;
        else if (!strcmp(argv[i],"--sets") && i+1<argc) sets=option(argv[++i],1,32);
        else if (!strcmp(argv[i],"--samples") && i+1<argc) samples=option(argv[++i],2,64);
        else if (!strcmp(argv[i],"--dispatches") && i+1<argc) dispatches=option(argv[++i],1,64);
        else { fprintf(stderr,"usage: %s [--bench | --bench-small] [--sets 1..32] [--samples 2..64 even] [--dispatches 1..64]\n",argv[0]); return 2; }
    }
    if (samples%2) { fprintf(stderr,"--samples must be even\n"); return 2; }
    device=MTLCreateSystemDefaultDevice();
    if (!device) { fprintf(stderr,"SKIP: no Metal device; GPU access is required.\n"); return 77; }
    queue=[device newCommandQueue]; require(queue != nil,"queue allocation");
    printf("Metal Q4 activation: device=%s sets=%u calls/sample=%u samples/arm=%u\n",
           device.name.UTF8String,sets,dispatches,samples); fflush(stdout);
    NSError *error=nil;
    // Use the default production Metal compilation settings for both arms.
    id<MTLLibrary> library=[device newLibraryWithSource:source() options:nil error:&error];
    require(library != nil,error.localizedDescription.UTF8String);
    for (int bc_out=0;bc_out<2;++bc_out) {
        mm32[bc_out]=pipeline(library,@"kernel_mul_mm_q4_K_f32",true,bc_out);
        mm16[bc_out]=pipeline(library,@"kernel_mul_mm_q4_K_f16_rhs",true,bc_out);
    }
    convert=pipeline(library,@"kernel_cpy_contig_f32_f16_4",false,false);
    weight_stride=(size_t)MAX_M*(K/256)*sizeof(Q4);
    weights=buffer(sets*weight_stride); input=buffer((size_t)MAX_N*K*4);
    rhs=buffer((size_t)MAX_N*K*2);
    baseline=buffer((size_t)MAX_M*MAX_N*4); candidate=buffer((size_t)MAX_M*MAX_N*4);
    Q4 *w=payload(weights);
    for (size_t i=0;i<sets*weight_stride/sizeof(*w);++i) {
        w[i].d=0x2000u+(random32()&0x7ffu);
        w[i].dmin=i%7 ? 0x1800u+(random32()&0x7ffu) : 0;
        for (size_t j=0;j<12;++j) w[i].scales[j]=random32()>>24;
        for (size_t j=0;j<128;++j) w[i].qs[j]=random32()>>24;
    }
    float *x=payload(input);
    for (size_t i=0;i<(size_t)MAX_N*K;++i) {
        // Full finite mantissas and normal F16 ranges challenge rounding/order.
        uint32_t bits=(random32()&0x807fffffu) | ((114u+random32()%17u)<<23);
        if (i%97==0) bits=0x80000000u;
        if (i%193==0) bits=0x3f801000u; // halfway between adjacent half values
        memcpy(&x[i],&bits,4);
    }
    NSData *x_snapshot=[NSData dataWithBytes:input.contents length:input.length];
    NSData *w_snapshot=[NSData dataWithBytes:weights.contents length:weights.length];
    const uint32_t tokens[]={128,129,256,257,511,512,513,1024,2048};
    for (size_t i=0;i<sizeof(tokens)/sizeof(*tokens);++i) oracle(MAX_M,tokens[i],i%sets);
    oracle(MAX_M-1,129,0); oracle(MAX_M-1,257,(sets-1));
    if (bench) {
        puts("Timing: resident synthetic Q-b only; conversion and separate tracked encoders included; allocations, SSD, sidecars, runtime and CPU wall time excluded.");
        const uint32_t timed[]={128,256,512,1024,2048};
        for (size_t i=0;i<(bench_small ? 2 : sizeof(timed)/sizeof(*timed));++i) benchmark(timed[i]);
    }
    require(!memcmp(input.contents,x_snapshot.bytes,input.length),"activation input changed");
    require(!memcmp(weights.contents,w_snapshot.bytes,weights.length),"weight input changed");
    puts("PASS: production kernels, bitwise parity, conversion, guards and immutable inputs.");
    return 0;
} }
