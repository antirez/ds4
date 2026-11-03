/* Independent default-fast HC gate/mixer oracle.
 * Frozen generic shader from 0aaea5a2; current generic/PF/reuse compiled
 * separately with identical default Metal options. No model required.
 * clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal
 *   hc-reference.m -o hc-reference
 * Run: ./hc-reference /absolute/path/to/ds4
 * Fails on any finite-output bit mismatch, corrupted guard, or input mutation.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Frozen assembled shader body SHA256: ea6c1ea18f345ee1ff86bea1eddc0613c90a329b8d3a223d52f001959f7fb600
// Exact source fragments taken from commit 0aaea5a2 without arithmetic edits.
static NSString *const kFrozen =
@"static inline float qwen4_sigmoid(float x) {\n"
"    if (x >= 0.0f) {\n"
"        const float e = exp(-x);\n"
"        return 1.0f / (1.0f + e);\n"
"    }\n"
"    const float e = exp(x);\n"
"    return e / (1.0f + e);\n"
"}\n"
"\n"
"static inline float qwen4_softplus(float x) {\n"
"    if (x > 20.0f) return x;\n"
"    if (x < -20.0f) return exp(x);\n"
"    return log(1.0f + exp(x));\n"
"}\n"
"\n"
"static inline float qwen4_silu(float x) {\n"
"    return x * qwen4_sigmoid(x);\n"
"}\n"
"\n"
"\n"
"struct qwen4_w_f16 {\n"
"    device const half *p;\n"
"    qwen4_w_f16(device const char *base) : p((device const half *)base) {}\n"
"    float at(uint64_t i) const { return (float)p[i]; }\n"
"};\n"
"struct qwen4_w_f32 {\n"
"    device const float *p;\n"
"    qwen4_w_f32(device const char *base) : p((device const float *)base) {}\n"
"    float at(uint64_t i) const { return p[i]; }\n"
"};\n"
"struct qwen4_w_q8 {\n"
"    device const char *p;\n"
"    qwen4_w_q8(device const char *base) : p(base) {}\n"
"    float at(uint64_t i) const {\n"
"        device const char *b = p + (i >> 5) * 34;\n"
"        return (float)(*(device const half *)b) * (float)b[2 + (i & 31u)];\n"
"    }\n"
"};\n"
"\n"
"\n"
"struct ds4_metal_args_qwen4_hc_gate_mix {\n"
"    uint32_t n_tokens;\n"
"    uint32_t n_embd;\n"
"    uint32_t n_hc;\n"
"    uint32_t n_rank;\n"
"};\n"
"\n"
"/* mixed[d] = mean over streams of sigmoid(w_up[s*E+d] . silu(lo/hc)) *\n"
" * xn[s*E+d], lo being the raw low-rank projection.  One simdgroup per d:\n"
" * the four 8-lane groups stream the four stream rows (contiguous n_rank\n"
" * weights each) and shuffle-reduce, first within the group, then across\n"
" * the streams. */\n"
"template <typename W>\n"
"kernel void kernel_qwen4_hc_gate_mix(\n"
"        constant ds4_metal_args_qwen4_hc_gate_mix & args,\n"
"        device const float *xn,       /* [T][hc*E] */\n"
"        device const float *lo,       /* [T][n_rank] raw */\n"
"        device const char  *w_up,     /* [hc*E][n_rank] */\n"
"        device float       *mixed,    /* [T][E] */\n"
"        uint3 tgpig [[threadgroup_position_in_grid]],\n"
"        ushort3 ntg [[threads_per_threadgroup]],\n"
"        ushort sgitg [[simdgroup_index_in_threadgroup]],\n"
"        ushort tiisg [[thread_index_in_simdgroup]]) {\n"
"    const uint hc = args.n_hc;\n"
"    const uint E = args.n_embd;\n"
"    const uint nsg = ntg.x / 32;\n"
"    const uint d = tgpig.x * nsg + sgitg;\n"
"    const uint tok = tgpig.y;\n"
"    if (d >= E || tok >= args.n_tokens) return;\n"
"    const uint s = tiisg / 8, lane = tiisg % 8;\n"
"    device const float *l = lo + (uint64_t)tok * args.n_rank;\n"
"    const W w(w_up);\n"
"    const uint64_t row = (uint64_t)(s * E + d) * args.n_rank;\n"
"    float acc = 0.0f;\n"
"    for (uint r = lane; r < args.n_rank; r += 8) acc += w.at(row + r) * qwen4_silu(l[r] / (float)hc);\n"
"    acc += simd_shuffle_xor(acc, 1);\n"
"    acc += simd_shuffle_xor(acc, 2);\n"
"    acc += simd_shuffle_xor(acc, 4);\n"
"    float g = qwen4_sigmoid(acc) * xn[(uint64_t)tok * E * hc + s * E + d];\n"
"    g += simd_shuffle_xor(g, 8);\n"
"    g += simd_shuffle_xor(g, 16);\n"
"    if (tiisg == 0) mixed[(uint64_t)tok * E + d] = g / (float)hc;\n"
"}\n"
"\n"
"#define QWEN4_HC_MIX_INSTANCE(SUFFIX, W) \\\n"
"template [[host_name(\"kernel_qwen4_hc_gate_mix_\" #SUFFIX)]] \\\n"
"kernel void kernel_qwen4_hc_gate_mix<W>(constant ds4_metal_args_qwen4_hc_gate_mix &, device const float *, \\\n"
"        device const float *, device const char *, device float *, uint3, ushort3, ushort, ushort);\n"
"QWEN4_HC_MIX_INSTANCE(f16, qwen4_w_f16)\n"
"QWEN4_HC_MIX_INSTANCE(f32, qwen4_w_f32)\n"
"QWEN4_HC_MIX_INSTANCE(q8, qwen4_w_q8)\n"
"\n";

static NSString *const kPreamble =
@"#include <metal_stdlib>\nusing namespace metal;\n";
static const size_t kGuard = 64;
static const uint32_t kCanary = UINT32_C(0x7e71357b);
typedef struct { uint32_t T,E,hc,rank; } hc_args;
_Static_assert(sizeof(hc_args)==16,"HC argument ABI");
static void fail(const char *msg) {
    fprintf(stderr,"FAIL HC reference: %s\n",msg); exit(1);
}
static NSRange unique(NSString *s, NSString *anchor) {
    NSUInteger count=[s componentsSeparatedByString:anchor].count-1;
    if(count!=1) {
        fprintf(stderr,"source anchor occurrences=%lu: %s\n",(unsigned long)count,anchor.UTF8String);
        fail("source anchor missing/nonunique");
    }
    return [s rangeOfString:anchor];
}
static NSString *between(NSString *s, NSString *a, NSString *b) {
    NSRange x=unique(s,a),y=unique(s,b);
    if(y.location<=x.location)fail("invalid source range");
    return [s substringWithRange:NSMakeRange(x.location,y.location-x.location)];
}
static NSString *candidate(NSString *repo) {
    NSError *err=nil;
    NSString *s=[NSString stringWithContentsOfFile:[repo stringByAppendingPathComponent:@"metal/qwen4.metal"]
                                        encoding:NSUTF8StringEncoding error:&err];
    if(!s)fail("cannot read current qwen4.metal");
    NSMutableString *out=[NSMutableString stringWithString:kPreamble];
    [out appendString:between(s,@"static inline float qwen4_sigmoid(",
        @"static inline float qwen4_row_dot(device const char *row, device const float *x,\n"
         "                                  uint weight_type, uint in_dim, ushort tiisg);")];
    [out appendString:between(s,@"struct qwen4_w_f16 {",@"/* Grouped RMSNorm")];
    [out appendString:between(s,@"struct ds4_metal_args_qwen4_hc_gate_mix {",@"/* Two-token MTP verification")];
    return out;
}
static id<MTLLibrary> library(id<MTLDevice> d, NSString *s) {
    NSError *err=nil;
    // Leave fastMathEnabled/mathMode at the same defaults as production.
    MTLCompileOptions *o=[MTLCompileOptions new];
    id<MTLLibrary> l=[d newLibraryWithSource:s options:o error:&err];
    if(!l) { fprintf(stderr,"Metal compilation: %s\n",err.description.UTF8String); fail("library"); }
    return l;
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,id<MTLLibrary> l,NSString *name) {
    NSError *err=nil;
    id<MTLFunction> f=[l newFunctionWithName:name];
    if(!f)fail("missing shader function");
    id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:f error:&err];
    if(!p)fail("pipeline");
    if(p.threadExecutionWidth!=32 || p.maxTotalThreadsPerThreadgroup<128)fail("unsupported SIMD geometry");
    return p;
}
static uint32_t mix(uint32_t x) {
    x^=x>>16; x*=UINT32_C(0x7feb352d); x^=x>>15; x*=UINT32_C(0x846ca68b); return x^(x>>16);
}
static id<MTLBuffer> buffer(id<MTLDevice> d,size_t n) {
    if(n%4)fail("unaligned fixture size");
    id<MTLBuffer> b=[d newBufferWithLength:n+2*kGuard options:MTLResourceStorageModeShared];
    if(!b)fail("buffer allocation");
    uint32_t *w=b.contents;
    for(size_t i=0;i<b.length/4;i++)w[i]=kCanary;
    return b;
}
static void guards(id<MTLBuffer> b) {
    uint32_t *w=b.contents;
    for(size_t i=0;i<kGuard/4;i++)
        if(w[i]!=kCanary || w[b.length/4-1-i]!=kCanary)fail("guard corruption");
}
static void dispatch(id<MTLCommandQueue> q,id<MTLComputePipelineState> p,
                     hc_args a,NSArray<id<MTLBuffer>> *b,bool reuse,uint32_t nsg) {
    uint32_t *v=b[3].contents;
    for(size_t i=0;i<b[3].length/4;i++)v[i]=kCanary;
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    if(!e)fail("command encoder");
    [e setComputePipelineState:p]; [e setBytes:&a length:sizeof(a) atIndex:0];
    for(NSUInteger i=0;i<b.count;i++)[e setBuffer:b[i] offset:kGuard atIndex:i+1];
    if(reuse)[e setThreadgroupMemoryLength:(NSUInteger)a.rank*2*sizeof(float) atIndex:0];
    if(p.maxTotalThreadsPerThreadgroup<32*nsg)fail("pipeline cannot support requested NSG");
    [e dispatchThreadgroups:MTLSizeMake((a.E+nsg-1)/nsg,a.T,1)
        threadsPerThreadgroup:MTLSizeMake(32*nsg,1,1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if(cb.status!=MTLCommandBufferStatusCompleted)fail("GPU command failed");
    for(id<MTLBuffer> v in b)guards(v);
}
int main(int argc,char **argv) {
    @autoreleasepool {
        if(argc!=2) {fprintf(stderr,"usage: hc-reference REPO\n");return 2;}
        id<MTLDevice> d=MTLCreateSystemDefaultDevice();if(!d)fail("Metal device unavailable");
        id<MTLCommandQueue> q=[d newCommandQueue];if(!q)fail("queue");
        NSString *repo=[NSString stringWithUTF8String:argv[1]];
        id<MTLLibrary> old=library(d,[kPreamble stringByAppendingString:kFrozen]);
        id<MTLLibrary> now=library(d,candidate(repo));
        id<MTLComputePipelineState> oracle=pipeline(d,old,@"kernel_qwen4_hc_gate_mix_f16");
        NSArray<NSString*> *names=@[@"kernel_qwen4_hc_gate_mix_f16",@"kernel_qwen4_hc_gate_mix_f16_pf",@"kernel_qwen4_hc_gate_mix_f16_reuse"];
        NSArray<id<MTLComputePipelineState>> *ps=@[pipeline(d,now,names[0]),pipeline(d,now,names[1]),pipeline(d,now,names[2])];
        const hc_args shapes[]={{1,2560,4,320},{3,33,4,136}};
        uint64_t checked=0,differences=0;
        double worst=0;
        printf("HC frozen0aa reference: device=%s math=default frozen_sha256=ea6c1ea18f345ee1ff86bea1eddc0613c90a329b8d3a223d52f001959f7fb600\n",d.name.UTF8String);
        for(size_t si=0;si<sizeof(shapes)/sizeof(shapes[0]);si++) {
            hc_args a=shapes[si];
            for(uint32_t pattern=0;pattern<3;pattern++) {
                size_t nx=(size_t)a.T*a.hc*a.E,nl=(size_t)a.T*a.rank,nw=(size_t)a.hc*a.E*a.rank,no=(size_t)a.T*a.E;
                NSArray<id<MTLBuffer>> *bs=@[buffer(d,nx*4),buffer(d,nl*4),buffer(d,nw*2),buffer(d,no*4)];
                float *x=(float*)((char*)bs[0].contents+kGuard),*lo=(float*)((char*)bs[1].contents+kGuard);
                _Float16 *w=(_Float16*)((char*)bs[2].contents+kGuard);
                for(size_t i=0;i<nx;i++)x[i]=((int)(mix((uint32_t)i+41)%4097)-2048)*0.00048828125f;
                for(size_t i=0;i<nl;i++)lo[i]=((int)(mix((uint32_t)i+67)%4097)-2048)*(pattern==1?0.125f:0.0029296875f);
                for(size_t i=0;i<nw;i++)w[i]=(_Float16)(((int)(mix((uint32_t)i+137)%2049)-1024)*(pattern==2?0.000030517578125f:0.00029296875f));
                if(pattern==2)for(size_t i=0;i<nl;i++)lo[i]=(i&1?-1.f:1.f)*(i%7?0x1p-19f:6.f);
                NSMutableArray<NSData*> *frozen=[NSMutableArray new];
                for(NSUInteger i=0;i<3;i++)[frozen addObject:[NSData dataWithBytes:bs[i].contents length:bs[i].length]];
                float *ref=malloc(no*4);if(!ref)fail("reference allocation");
                const uint32_t nsgs[]={1,4,8};
                for(size_t gi=0;gi<sizeof(nsgs)/sizeof(nsgs[0]);gi++) {
                    uint32_t nsg=nsgs[gi];
                    dispatch(q,oracle,a,bs,false,nsg);
                    memcpy(ref,(char*)bs[3].contents+kGuard,no*4);
                    for(size_t j=0;j<no;j++) {
                        uint32_t bits;memcpy(&bits,ref+j,4);
                        if(bits==kCanary || !isfinite(ref[j]))fail("oracle omitted output or produced nonfinite value");
                    }
                    for(NSUInteger pi=0;pi<ps.count;pi++) {
                        dispatch(q,ps[pi],a,bs,pi==2,nsg);
                        const float *got=(float*)((char*)bs[3].contents+kGuard);
                        uint64_t changed=0; double delta=0;
                        for(size_t j=0;j<no;j++) {
                            if(!isfinite(ref[j])||!isfinite(got[j]))fail("nonfinite output");
                            uint32_t ob,nb;memcpy(&ob,ref+j,4);memcpy(&nb,got+j,4);
                            if(nb==kCanary)fail("candidate omitted output");
                            if(ob!=nb)changed++;
                            delta=fmax(delta,fabs((double)ref[j]-got[j]));
                        }
                        for(NSUInteger j=0;j<3;j++)if(memcmp(frozen[j].bytes,bs[j].contents,frozen[j].length))fail("immutable input changed");
                        printf("%s E=%u rank=%u T=%u pattern=%u NSG=%u %s: changed=%llu/%zu max_abs=%.9g\n",
                               changed?"DIFF":"PASS",a.E,a.rank,a.T,pattern,nsg,names[pi].UTF8String,
                               (unsigned long long)changed,no,delta);
                        checked+=no;differences+=changed;worst=fmax(worst,delta);
                    }
                }
                free(ref);
            }
        }
        printf("%s HC frozen0aa: comparisons=%llu different=%llu max_abs=%.9g\n",
               differences?"FAIL":"PASS",(unsigned long long)checked,(unsigned long long)differences,worst);
        return differences?1:0;
    }
}
