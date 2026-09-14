#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    QK_K = 256,
    Q4_BLOCK_BYTES = 144,
    THREADS = 128,
    SMEM = 6144,
    GUARD_BYTES = 256,
};

static const uint32_t k_guard = 0x7fc12345u;
static const uint32_t k_baseline_poison = 0x7fc0b001u;
static const uint32_t k_pair_poison = 0x7fc0c001u;
static uint32_t in_dim = 4096, out0_dim = 1024, out1_dim = 512, max_tokens = 128;
static uint32_t k_tokens[16] = {32u, 64u, 96u, 128u};
static size_t token_cases = 4;

typedef struct {
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
} block_q4_K_host;

typedef struct {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1;
    int16_t r2, r3;
} mul_mm_args;

_Static_assert(sizeof(block_q4_K_host) == Q4_BLOCK_BYTES, "Q4_K ABI");
_Static_assert(sizeof(mul_mm_args) == 88, "mul_mm ABI");

typedef struct {
    uint32_t sets, dispatches, warmup, samples;
    const char *reference_root;
    bool rhs_f32, legacy_tile, reference_f32;
} config;

typedef enum { ARM_BASELINE, ARM_PAIR_F16 } arm;

typedef struct {
    __strong id<MTLDevice> device;
    __strong id<MTLCommandQueue> queue;
    __strong id<MTLComputePipelineState> mm_f32;
    __strong id<MTLComputePipelineState> mm_f16;
    __strong id<MTLComputePipelineState> mm_reference;
    __strong id<MTLComputePipelineState> copy;
    __strong id<MTLBuffer> w0, w1, x, rhs;
    __strong id<MTLBuffer> base0, base1, pair0, pair1;
    NSUInteger w0_set_bytes, w1_set_bytes;
    NSUInteger x_off, rhs_off, out0_stride, out1_stride;
    uint32_t sets;
    bool rhs_f32, legacy_tile, reference_f32;
    uint8_t *x_snapshot;
} fixture;

static void usage(const char *argv0) {
    printf("usage: %s [--sets N] [--dispatches N] [--warmup N] [--samples N] [--reference-root DIR]\n",
           argv0);
    puts("       [--in-dim K] [--out0-dim M] [--out1-dim M] [--tokens N,N,...] [--legacy-tile] [--rhs-f32] [--reference-f32]");
}

static uint32_t parse_u32(const char *s, const char *name, uint32_t min) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !s[0] || !end || *end || v < min || v > UINT32_MAX) {
        fprintf(stderr, "metal-q4-prefill-pair-bench: invalid %s: %s\n", name, s);
        exit(2);
    }
    return (uint32_t)v;
}

static const char *arg_value(int *i, int argc, char **argv) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "metal-q4-prefill-pair-bench: %s needs a value\n", argv[*i]);
        exit(2);
    }
    return argv[++*i];
}

static config parse_options(int argc, char **argv) {
    config c = {.sets=64u, .dispatches=32u, .warmup=8u, .samples=12u};
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]); exit(0);
        } else if (!strcmp(argv[i], "--sets")) {
            c.sets = parse_u32(arg_value(&i, argc, argv), "--sets", 1u);
        } else if (!strcmp(argv[i], "--dispatches")) {
            c.dispatches = parse_u32(arg_value(&i, argc, argv), "--dispatches", 1u);
        } else if (!strcmp(argv[i], "--warmup")) {
            c.warmup = parse_u32(arg_value(&i, argc, argv), "--warmup", 0u);
        } else if (!strcmp(argv[i], "--samples")) {
            c.samples = parse_u32(arg_value(&i, argc, argv), "--samples", 2u);
        } else if (!strcmp(argv[i], "--reference-root")) {
            c.reference_root = arg_value(&i, argc, argv);
        } else if (!strcmp(argv[i], "--legacy-tile")) {
            c.legacy_tile = true;
        } else if (!strcmp(argv[i], "--rhs-f32")) {
            c.rhs_f32 = true;
        } else if (!strcmp(argv[i], "--reference-f32")) {
            c.reference_f32 = true;

        } else if (!strcmp(argv[i], "--in-dim")) {
            in_dim = parse_u32(arg_value(&i, argc, argv), "--in-dim", 256u);
        } else if (!strcmp(argv[i], "--out0-dim")) {
            out0_dim = parse_u32(arg_value(&i, argc, argv), "--out0-dim", 1u);
        } else if (!strcmp(argv[i], "--out1-dim")) {
            out1_dim = parse_u32(arg_value(&i, argc, argv), "--out1-dim", 0u);
        } else if (!strcmp(argv[i], "--tokens")) {
            const char *values = arg_value(&i, argc, argv);
            token_cases = 0; max_tokens = 0;
            while (*values && token_cases < sizeof(k_tokens)/sizeof(k_tokens[0])) {
                char *end = NULL;
                const unsigned long n = strtoul(values, &end, 10);
                if (end == values || n == 0 || n > 4096 || (*end && *end != ',')) {
                    fprintf(stderr, "invalid --tokens list\n"); exit(2);
                }
                k_tokens[token_cases++] = (uint32_t)n;
                if (n > max_tokens) max_tokens = (uint32_t)n;
                values = *end ? end+1 : end;
            }
            if (*values || !token_cases) { fprintf(stderr, "invalid --tokens list\n"); exit(2); }
        } else {
            usage(argv[0]); exit(2);
        }
    }
    if (c.samples & 1u) {
        fprintf(stderr, "metal-q4-prefill-pair-bench: --samples must be even\n");
        exit(2);
    }
    if (in_dim % 256u) { fprintf(stderr, "--in-dim must be divisible by 256\n"); exit(2); }
    if (in_dim > INT32_MAX || out0_dim > INT32_MAX || out1_dim > INT32_MAX ||
        (uint64_t)max_tokens*in_dim > UINT32_MAX ||
        (uint64_t)max_tokens*out0_dim > INT32_MAX ||
        (uint64_t)max_tokens*out1_dim > INT32_MAX) {
        fprintf(stderr, "dimensions exceed the kernel argument range\n"); exit(2);
    }
    if (c.rhs_f32 && !c.reference_root) { fprintf(stderr, "--rhs-f32 requires --reference-root\n"); exit(2); }
    if (c.reference_f32 && !c.reference_root) { fprintf(stderr, "--reference-f32 requires --reference-root\n"); exit(2); }
    return c;
}

static NSString *prelude(void) {
    return @"#include <metal_stdlib>\nusing namespace metal;\n"
            "#define MAX(x,y) ((x)>(y)?(x):(y))\n"
            "#define MIN(x,y) ((x)<(y)?(x):(y))\n"
            "#define SWAP(x,y) { auto t=(x); (x)=(y); (y)=t; }\n"
            "#define QK8_0 32\n#ifndef QK_K\n#define QK_K 256\n#endif\n"
            "#define N_SIMDWIDTH 32\n#define N_R0_Q8_0 2\n#define N_SG_Q8_0 4\n"
            "#define FC_MUL_MV 600\n#define FC_MUL_MM 700\n#define FC_BIN 1300\n"
            "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
            "#define M_PI_F 3.14159265358979323846f\n"
            "enum ds4_sort_order { DS4_SORT_ORDER_ASC, DS4_SORT_ORDER_DESC };\n"
            "struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n"
            "struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };\n";
}

static NSString *load_source(const char *root_override) {
    static const char *files[] = {
        "metal/activations.metal", "metal/flash_attn.metal", "metal/dense.metal",
        "metal/moe.metal", "metal/dsv4_hc.metal", "metal/unary.metal",
        "metal/dsv4_kv.metal", "metal/dsv4_rope.metal", "metal/dsv4_misc.metal",
        "metal/argsort.metal", "metal/cpy.metal", "metal/concat.metal",
        "metal/get_rows.metal", "metal/sum_rows.metal", "metal/softmax.metal",
        "metal/repeat.metal", "metal/glu.metal", "metal/norm.metal",
        "metal/bin.metal", "metal/set_rows.metal",
    };
    const char *root_path = root_override ? root_override : getenv("DS4_SOURCE_ROOT");
    NSString *root = root_path ? [NSString stringWithUTF8String:root_path] : @".";
    NSMutableString *s = [NSMutableString stringWithString:prelude()];
    for (size_t i = 0; i < sizeof(files)/sizeof(files[0]); i++) {
        NSString *p = [root stringByAppendingPathComponent:
            [NSString stringWithUTF8String:files[i]]];
        NSError *e = nil;
        NSString *part = [NSString stringWithContentsOfFile:p
                           encoding:NSUTF8StringEncoding error:&e];
        if (!part) {
            fprintf(stderr, "metal-q4-prefill-pair-bench: read %s: %s\n",
                    files[i], [[e localizedDescription] UTF8String]);
            return nil;
        }
        [s appendFormat:@"\n%@\n", part];
    }
    return s;
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> d,
        id<MTLLibrary> l, NSString *name, bool mm) {
    NSError *e = nil;
    id<MTLFunction> fn = nil;
    if (mm) {
        bool no = false, yes = true;
        MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
        [v setConstantValue:&no type:MTLDataTypeBool atIndex:700];
        [v setConstantValue:&yes type:MTLDataTypeBool atIndex:701];
        fn = [l newFunctionWithName:name constantValues:v error:&e];
    } else {
        fn = [l newFunctionWithName:name];
    }
    if (!fn) {
        fprintf(stderr, "metal-q4-prefill-pair-bench: function %s: %s\n",
                [name UTF8String], e ? [[e localizedDescription] UTF8String] : "missing");
        return nil;
    }
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:fn error:&e];
    if (!p) fprintf(stderr, "metal-q4-prefill-pair-bench: pipeline %s: %s\n",
                    [name UTF8String], [[e localizedDescription] UTF8String]);
    return p;
}

static uint32_t rng(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static void fill_q4(void *ptr, NSUInteger bytes, uint32_t seed) {
    block_q4_K_host *b = ptr;
    for (NSUInteger i = 0; i < bytes / sizeof(*b); i++) {
        b[i].d = (uint16_t)(0x2400u | (rng(&seed) & 0x3ffu));
        b[i].dmin = (uint16_t)(0x1c00u | (rng(&seed) & 0x3ffu));
        for (size_t j=0;j<sizeof(b[i].scales);j++) b[i].scales[j]=(uint8_t)(rng(&seed)>>24);
        for (size_t j=0;j<sizeof(b[i].qs);j++) b[i].qs[j]=(uint8_t)(rng(&seed)>>24);
    }
}
static void fill_guard(id<MTLBuffer> b) {
    uint32_t *p=b.contents; for (NSUInteger i=0;i<b.length/4;i++) p[i]=k_guard;
}
static void fill_payload(id<MTLBuffer> b, NSUInteger stride, uint32_t sets,
                         NSUInteger bytes, uint32_t poison) {
    for (uint32_t s = 0; s < sets; s++) {
        uint32_t *p = (uint32_t *)((uint8_t *)b.contents +
            (NSUInteger)s * stride + GUARD_BYTES);
        for (NSUInteger i = 0; i < bytes / sizeof(*p); i++) p[i] = poison;
    }
}
static void fill_x(float *x) {
    uint32_t s=0x243f6a88u;
    for (NSUInteger i=0;i<(NSUInteger)in_dim*max_tokens;i++)
        x[i]=((int32_t)(rng(&s)&0xffffu)-32768)/32768.0f;
}
static NSUInteger align_up(NSUInteger x, NSUInteger a) { return (x+a-1)/a*a; }
static id<MTLBuffer> alloc(id<MTLDevice> d, NSUInteger n, NSString *label) {
    if (!n || n>d.maxBufferLength) return nil;
    id<MTLBuffer> b=[d newBufferWithLength:n options:MTLResourceStorageModeShared];
    b.label=label; return b;
}

static bool is_pre_m5_apple_silicon_name(const char *name) {
    return name && !strncmp(name, "Apple M", 7) &&
           name[7] >= '1' && name[7] <= '4' &&
           (name[8] == '\0' || name[8] == ' ');
}

static bool init_fixture(fixture *f, const config *c, id<MTLDevice> device) {
    f->device=device;
    f->queue=[f->device newCommandQueue];
    NSString *src=load_source(NULL); if (!f->device || !f->queue || !src) return false;
    NSError *e=nil;
    id<MTLLibrary> l=[f->device newLibraryWithSource:src options:nil error:&e];
    if (!l) { fprintf(stderr,"metal-q4-prefill-pair-bench: compile: %s\n",
                      [[e localizedDescription] UTF8String]); return false; }
    f->mm_f32=pipeline(f->device,l,@"kernel_mul_mm_q4_K_f32",true);
    f->mm_f16=pipeline(f->device,l,c->legacy_tile?@"kernel_mul_mm_q4_K_f16_rhs":@"kernel_mul_mm_q4_K_f16_rhs_k64_m32",true);
    f->legacy_tile=c->legacy_tile; f->rhs_f32=c->rhs_f32; f->reference_f32=c->reference_f32;
    f->copy=pipeline(f->device,l,@"kernel_cpy_contig_f32_f16_4",false);
    if (c->reference_root) {
        NSString *reference=load_source(c->reference_root); if (!reference) return false;
        id<MTLLibrary> ref=[f->device newLibraryWithSource:reference options:nil error:&e];
        if (!ref) { fprintf(stderr,"reference compile: %s\n",[[e localizedDescription] UTF8String]); return false; }
        f->mm_reference=pipeline(f->device,ref,(c->rhs_f32||c->reference_f32)?@"kernel_mul_mm_q4_K_f32":@"kernel_mul_mm_q4_K_f16_rhs",true);
        if (!f->mm_reference) return false;
    }
    if (!f->mm_f32 || !f->mm_f16 || !f->copy ||
        f->mm_f32.maxTotalThreadsPerThreadgroup<THREADS ||
        f->mm_f16.maxTotalThreadsPerThreadgroup<THREADS) return false;
    f->sets=c->sets;
    f->w0_set_bytes=(NSUInteger)(in_dim/QK_K)*Q4_BLOCK_BYTES*out0_dim;
    f->w1_set_bytes=(NSUInteger)(in_dim/QK_K)*Q4_BLOCK_BYTES*out1_dim;
    f->x_off=f->rhs_off=GUARD_BYTES;
    f->out0_stride=align_up(GUARD_BYTES+(NSUInteger)max_tokens*out0_dim*4+GUARD_BYTES,256);
    f->out1_stride=align_up(GUARD_BYTES+(NSUInteger)max_tokens*out1_dim*4+GUARD_BYTES,256);
    if (f->w0_set_bytes>device.maxBufferLength/c->sets ||
        f->w1_set_bytes>device.maxBufferLength/c->sets ||
        f->out0_stride>device.maxBufferLength/c->sets ||
        f->out1_stride>device.maxBufferLength/c->sets) {
        fprintf(stderr, "fixture exceeds the Metal buffer limit\n"); return false;
    }
    f->w0=alloc(f->device,f->w0_set_bytes*c->sets,@"pair-w0");
    f->w1=alloc(f->device,out1_dim?f->w1_set_bytes*c->sets:Q4_BLOCK_BYTES,@"pair-w1");
    f->x=alloc(f->device,GUARD_BYTES+(NSUInteger)max_tokens*in_dim*4+GUARD_BYTES,@"pair-x");
    f->rhs=alloc(f->device,GUARD_BYTES+(NSUInteger)max_tokens*in_dim*2+GUARD_BYTES,@"pair-rhs");
    f->base0=alloc(f->device,f->out0_stride*c->sets,@"base0");
    f->base1=alloc(f->device,f->out1_stride*c->sets,@"base1");
    f->pair0=alloc(f->device,f->out0_stride*c->sets,@"pair0");
    f->pair1=alloc(f->device,f->out1_stride*c->sets,@"pair1");
    if (!f->w0||!f->w1||!f->x||!f->rhs||!f->base0||!f->base1||!f->pair0||!f->pair1) return false;
    fill_q4(f->w0.contents,f->w0.length,0x41c64e6du);
    fill_q4(f->w1.contents,f->w1.length,0x9e3779b9u);
    fill_guard(f->x); fill_x((float *)((uint8_t *)f->x.contents+f->x_off));
    f->x_snapshot=malloc(f->x.length); if (!f->x_snapshot) return false;
    memcpy(f->x_snapshot,f->x.contents,f->x.length);
    return true;
}

static mul_mm_args args(uint32_t out, uint32_t n, bool half_rhs) {
    uint64_t row=(in_dim/QK_K)*Q4_BLOCK_BYTES, elem=half_rhs?2u:4u;
    return (mul_mm_args){.ne00=in_dim,.ne02=1,.nb01=row,.nb02=row*out,.nb03=row*out,
        .ne12=1,.nb10=elem,.nb11=(uint64_t)in_dim*elem,
        .nb12=(uint64_t)in_dim*n*elem,.nb13=(uint64_t)in_dim*n*elem,
        .ne0=(int32_t)out,.ne1=(int32_t)n,.r2=1,.r3=1};
}

static void encode_mm(fixture *f,id<MTLComputeCommandEncoder> e,bool half,bool reference,
        uint32_t n,uint32_t set,id<MTLBuffer> o0,id<MTLBuffer> o1) {
    mul_mm_args a0=args(out0_dim,n,half), a1=args(out1_dim,n,half);
    const bool candidate_m32 = !reference && !f->legacy_tile && half;
    const uint32_t tile_m = candidate_m32 ? 32 : 64;
    [e setComputePipelineState:reference?f->mm_reference:(half?f->mm_f16:f->mm_f32)];
    [e setThreadgroupMemoryLength:candidate_m32?8192:
        ((n%32 || out0_dim%64 || out1_dim%64)?8192:SMEM) atIndex:0];
    [e setBytes:&a0 length:sizeof(a0) atIndex:0];
    [e setBuffer:f->w0 offset:set*f->w0_set_bytes atIndex:1];
    [e setBuffer:half?f->rhs:f->x offset:half?f->rhs_off:f->x_off atIndex:2];
    [e setBuffer:o0 offset:set*f->out0_stride+GUARD_BYTES atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake((n+31)/32,(out0_dim+tile_m-1)/tile_m,1)
         threadsPerThreadgroup:MTLSizeMake(THREADS,1,1)];
    if (!out1_dim) return;
    [e setBytes:&a1 length:sizeof(a1) atIndex:0];
    [e setBuffer:f->w1 offset:set*f->w1_set_bytes atIndex:1];
    [e setBuffer:o1 offset:set*f->out1_stride+GUARD_BYTES atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake((n+31)/32,(out1_dim+tile_m-1)/tile_m,1)
         threadsPerThreadgroup:MTLSizeMake(THREADS,1,1)];
}

static void encode_copy(fixture *f,id<MTLComputeCommandEncoder> e,uint32_t n) {
    uint32_t elems=n*in_dim;
    NSUInteger work=(elems+3u)/4u, nth=256u;
    if (nth>f->copy.maxTotalThreadsPerThreadgroup) nth=f->copy.maxTotalThreadsPerThreadgroup;
    [e setComputePipelineState:f->copy]; [e setBytes:&elems length:4 atIndex:0];
    [e setBuffer:f->x offset:f->x_off atIndex:1]; [e setBuffer:f->rhs offset:f->rhs_off atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake((work+nth-1)/nth,1,1)
         threadsPerThreadgroup:MTLSizeMake(nth,1,1)];
}

static bool finish(id<MTLCommandBuffer> cb,const char *label,double *secs) {
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status!=MTLCommandBufferStatusCompleted) {
        fprintf(stderr,"metal-q4-prefill-pair-bench: %s: %s\n",label,
                cb.error?[[cb.error localizedDescription] UTF8String]:"failed"); return false;
    }
    double t=cb.GPUEndTime-cb.GPUStartTime; if (!(t>0)) return false;
    if (secs) *secs=t; return true;
}

static bool workload(fixture *f,arm a,uint32_t n,uint32_t calls,uint32_t start,double *secs) {
    id<MTLCommandBuffer> cb=[f->queue commandBuffer];
    if ((a==ARM_BASELINE && (!f->mm_reference || f->reference_f32)) || f->rhs_f32) {
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        for(uint32_t i=0;i<calls;i++) { uint32_t s=(start+i)%f->sets;
            const bool base=a==ARM_BASELINE;
            encode_mm(f,e,false,base&&f->mm_reference!=nil,n,s,base?f->base0:f->pair0,base?f->base1:f->pair1); }
        [e endEncoding];
    } else {
        for(uint32_t i=0;i<calls;i++) { uint32_t s=(start+i)%f->sets;
            id<MTLComputeCommandEncoder> c=[cb computeCommandEncoder]; encode_copy(f,c,n); [c endEncoding];
            const bool ref=a==ARM_BASELINE;
            id<MTLComputeCommandEncoder> m=[cb computeCommandEncoder];
            encode_mm(f,m,true,ref,n,s,ref?f->base0:f->pair0,ref?f->base1:f->pair1); [m endEncoding];
        }
    }
    return finish(cb,a==ARM_BASELINE?"baseline":"pair-f16",secs);
}

static bool guards(fixture *f,id<MTLBuffer> b,NSUInteger stride,NSUInteger payload,const char *name) {
    uint32_t *p=b.contents;
    for(uint32_t s=0;s<f->sets;s++) {
        NSUInteger base=s*stride, begin=base+GUARD_BYTES, end=begin+payload;
        for(NSUInteger x=base;x<begin;x+=4) if(p[x/4]!=k_guard) goto bad;
        for(NSUInteger x=end;x<base+stride;x+=4) if(p[x/4]!=k_guard) goto bad;
    }
    return true;
bad: fprintf(stderr,"metal-q4-prefill-pair-bench: %s canary changed\n",name); return false;
}

static bool oracle(fixture *f,uint32_t n) {
    fill_guard(f->rhs); fill_guard(f->base0); fill_guard(f->base1); fill_guard(f->pair0); fill_guard(f->pair1);
    NSUInteger b0=(NSUInteger)n*out0_dim*4, b1=(NSUInteger)n*out1_dim*4;
    fill_payload(f->base0, f->out0_stride, f->sets, b0, k_baseline_poison);
    fill_payload(f->base1, f->out1_stride, f->sets, b1, k_baseline_poison);
    fill_payload(f->pair0, f->out0_stride, f->sets, b0, k_pair_poison);
    fill_payload(f->pair1, f->out1_stride, f->sets, b1, k_pair_poison);
    double ignored;
    if (!workload(f,ARM_BASELINE,n,f->sets,0,&ignored) ||
        !workload(f,ARM_PAIR_F16,n,f->sets,0,&ignored)) return false;
    for(uint32_t s=0;s<f->sets;s++) {
        uint8_t *a0=(uint8_t *)f->base0.contents+s*f->out0_stride+GUARD_BYTES;
        uint8_t *p0=(uint8_t *)f->pair0.contents+s*f->out0_stride+GUARD_BYTES;
        uint8_t *a1=(uint8_t *)f->base1.contents+s*f->out1_stride+GUARD_BYTES;
        uint8_t *p1=(uint8_t *)f->pair1.contents+s*f->out1_stride+GUARD_BYTES;
        if(memcmp(a0,p0,b0)||memcmp(a1,p1,b1)) { fprintf(stderr,"mismatch N=%u set=%u\n",n,s); return false; }
    }
    if(memcmp(f->x.contents,f->x_snapshot,f->x.length)) return false;
    uint32_t *r=f->rhs.contents;
    for(NSUInteger x=0;x<f->rhs_off;x+=4) if(r[x/4]!=k_guard) return false;
    for(NSUInteger x=f->rhs_off+(NSUInteger)n*in_dim*2;x<f->rhs.length;x+=4) if(r[x/4]!=k_guard) return false;
    if(!guards(f,f->base0,f->out0_stride,b0,"base0")||!guards(f,f->base1,f->out1_stride,b1,"base1")||
       !guards(f,f->pair0,f->out0_stride,b0,"pair0")||!guards(f,f->pair1,f->out1_stride,b1,"pair1")) return false;
    fprintf(stderr,"Metal Q4 prefill pair oracle N=%u: PASS (bit-exact, canaries intact)\n",n);
    return true;
}

static int cmp(const void *a,const void *b){double x=*(double*)a,y=*(double*)b;return(x>y)-(x<y);}
static double median(double *x,uint32_t n){qsort(x,n,sizeof(*x),cmp);return n&1?x[n/2]:(x[n/2-1]+x[n/2])*0.5;}
static double sample_sd(const double *x,uint32_t n) {
    double mean=0, squared=0;
    for(uint32_t i=0;i<n;i++) mean+=x[i];
    mean/=n;
    for(uint32_t i=0;i<n;i++) { const double d=x[i]-mean; squared+=d*d; }
    return sqrt(squared/(n-1));
}

static bool bench_case(fixture *f,const config *c,uint32_t n) {
    double tmp;
    for(int i=0;i<4&&c->warmup;i++) if(!workload(f,(i==0||i==3)?ARM_BASELINE:ARM_PAIR_F16,n,c->warmup,i*c->warmup,&tmp)) return false;
    double *b=calloc(c->samples,sizeof(*b)),*p=calloc(c->samples,sizeof(*p)); if(!b||!p) return false;
    uint32_t nb=0,np=0;
    for(uint32_t cyc=0;cyc<c->samples/2;cyc++) {
        arm abba[]={ARM_BASELINE,ARM_PAIR_F16,ARM_PAIR_F16,ARM_BASELINE};
        arm baab[]={ARM_PAIR_F16,ARM_BASELINE,ARM_BASELINE,ARM_PAIR_F16}; arm *order=cyc&1?baab:abba;
        for(int j=0;j<4;j++) { arm a=order[j]; double t;
            uint32_t idx=a==ARM_BASELINE?nb:np;
            if(!workload(f,a,n,c->dispatches,(uint64_t)idx*c->dispatches%f->sets,&t)){free(b);free(p);return false;}
            if(a==ARM_BASELINE)b[nb++]=t;else p[np++]=t;
        }
    }
    const double bsd=sample_sd(b,c->samples)*1e6/c->dispatches;
    const double psd=sample_sd(p,c->samples)*1e6/c->dispatches;
    double bm=median(b,c->samples)*1e6/c->dispatches, pm=median(p,c->samples)*1e6/c->dispatches;
    printf("  N=%-3u %s %.3f us (sd %.3f) | %s %.3f us (sd %.3f) | %+.2f%%\n",
           n,(c->rhs_f32||c->reference_f32)?"reference Q4/F32":(c->reference_root?"reference copy+Q4/F16":"baseline Q4/F32"),
           bm,bsd,c->rhs_f32?"candidate Q4/F32":"copy+Q4/F16",
           pm,psd,(bm-pm)*100.0/bm); free(b);free(p); return true;
}

int main(int argc,char **argv) { @autoreleasepool {
    config c=parse_options(argc,argv);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device) {
        fprintf(stderr,"metal-q4-prefill-pair-bench: SKIP: no Metal device\n");
        return 0;
    }
    const char *device_name=[[device name] UTF8String];
    fprintf(stderr,"metal-q4-prefill-pair-bench: device=%s\n",
            device_name?device_name:"unknown");
    if(!is_pre_m5_apple_silicon_name(device_name)) {
        fprintf(stderr,
                "metal-q4-prefill-pair-bench: SKIP: runtime candidate requires Apple M1-M4\n");
        return 0;
    }
    fixture f={0}; if(!init_fixture(&f,&c,device)) return 1;
    printf("Metal Q4 projections, resident GPU-event-only; candidate %s\n",
           c.rhs_f32 || c.legacy_tile ? "M64/N32/K32" : "M32/N32/K64");
    printf("  %u->%u + %u->%u, %.1f MiB weights, %u sets, %u calls/sample\n",
           in_dim,out0_dim,in_dim,out1_dim,(double)(f.w0.length+f.w1.length)/1048576.0,c.sets,c.dispatches);
    for(size_t i=0;i<token_cases;i++) {
        if(!oracle(&f,k_tokens[i])||!bench_case(&f,&c,k_tokens[i])) return 1;
    }
    printf("  correctness: bit-exact; input/RHS/output canaries intact\n");
    printf("  scope: resident buffers, GPU timestamps; no SSD/model runtime/CPU wall time\n");
    free(f.x_snapshot); return 0;
} }
