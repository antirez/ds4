/* Shared native Metal ABI, guarded buffers, and independent decode references. */
#ifndef DS4_METAL_DECODE_TEST_SUPPORT_H
#define DS4_METAL_DECODE_TEST_SUPPORT_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Buffer layouts mirror the shader ABI, including native 64-bit alignment. */
typedef struct {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1, nr0;
    int16_t r2, r3;
} MV;
typedef struct { uint32_t width, ratio, pos, ape_type; } Store;
typedef struct {
    int64_t n_embd, n_hc, n_tokens;
    uint64_t nb_block0, nb_block1, nb_add0, nb_add1;
    uint64_t nb_res0, nb_res1, nb_res2, nb_post0, nb_post1;
    uint64_t nb_comb0, nb_comb1, nb_comb2, nb0, nb1, nb2;
    int32_t has_add;
} HC;
typedef struct { _Float16 d; int8_t qs[32]; } Q8;
_Static_assert(sizeof(MV) == 112, "matvec ABI");
_Static_assert(sizeof(HC) == 152, "HC ABI");
_Static_assert(sizeof(Q8) == 34, "Q8_0 ABI");

enum { GUARD = 64 };
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLLibrary> library;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
static unsigned seed = 0x19ae548d;
static uint32_t random32(void) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed;
}
static float sample(void) {
    return ((int)(random32() % 2001) - 1000) / 1024.0f;
}
static void die(const char *message) { fprintf(stderr, "%s\n", message); exit(1); }
static void *contents(id<MTLBuffer> b) { return (char *)b.contents + GUARD; }
static size_t bytes(id<MTLBuffer> b) { return b.length - 2 * GUARD; }
static id<MTLBuffer> buffer(size_t size) {
    id<MTLBuffer> b = [device newBufferWithLength:size + 2 * GUARD
                                        options:MTLResourceStorageModeShared];
    if (!b) die("Metal allocation failed");
    memset(b.contents, 0xA5, b.length);
    return b;
}
static id<MTLBuffer> floats(size_t count) {
    id<MTLBuffer> b = buffer(count * sizeof(float));
    float *p = contents(b);
    for (size_t i = 0; i < count; i++) p[i] = sample();
    return b;
}
static id<MTLBuffer> halfs(size_t count) {
    id<MTLBuffer> b = buffer(count * sizeof(_Float16));
    _Float16 *p = contents(b);
    for (size_t i = 0; i < count; i++) p[i] = (_Float16)sample();
    return b;
}
static id<MTLBuffer> q8s(size_t count) {
    id<MTLBuffer> b = buffer(count * sizeof(Q8)); Q8 *p = contents(b);
    for (size_t i = 0; i < count; i++) {
        p[i].d = (_Float16)((random32() % 31 + 1) / 2048.0f);
        for (int j = 0; j < 32; j++) p[i].qs[j] = (int)(random32() % 255) - 127;
    }
    return b;
}
static id<MTLBuffer> copy(id<MTLBuffer> b) {
    id<MTLBuffer> c = buffer(bytes(b)); memcpy(c.contents, b.contents, b.length); return c;
}
static void equal(id<MTLBuffer> a, id<MTLBuffer> b, const char *label) {
    if (a.length != b.length) die("test buffer length mismatch");
    if (!memcmp(a.contents, b.contents, a.length)) return;
    const uint32_t *x = a.contents, *y = b.contents;
    for (size_t i = 0; i < a.length / 4; i++) if (x[i] != y[i]) {
        fprintf(stderr, "%s mismatch at word %zd: %08x != %08x\n",
                label, (ssize_t)i - GUARD/4, x[i], y[i]); break;
    }
    exit(1);
}
static void guards(id<MTLBuffer> b) {
    const unsigned char *p = b.contents;
    for (size_t i = 0; i < GUARD; i++)
        if (p[i] != 0xA5 || p[b.length - GUARD + i] != 0xA5) die("buffer guard modified");
}
static id<MTLComputePipelineState> pipeline(NSString *name, short nsg) {
    NSString *key = [NSString stringWithFormat:@"%@:%d", name, nsg];
    id<MTLComputePipelineState> result = pipelines[key];
    if (result) return result;
    NSError *error = nil;
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    [values setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    id<MTLFunction> fn = [library newFunctionWithName:name constantValues:values error:&error];
    if (fn) result = [device newComputePipelineStateWithFunction:fn error:&error];
    if (!result) { fprintf(stderr, "%s: %s\n", name.UTF8String, error.description.UTF8String); exit(1); }
    pipelines[key] = result; return result;
}
static void bind(id<MTLComputeCommandEncoder> e, id<MTLBuffer> b, unsigned i) {
    [e setBuffer:b offset:GUARD atIndex:i];
}
static void dispatch(id<MTLComputeCommandEncoder> e, unsigned groups, short nsg, unsigned smem) {
    if (smem) [e setThreadgroupMemoryLength:smem atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    [e endEncoding];
}
static void finish(id<MTLCommandBuffer> cb) {
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "GPU command failed: %s\n", cb.error.description.UTF8String); exit(1);
    }
}
static MV args(unsigned k, unsigned m, unsigned nr, unsigned row) {
    MV a = {0}; a.ne00 = a.ne10 = k; a.ne01 = a.ne0 = m;
    a.ne02 = a.ne11 = a.ne12 = a.ne1 = 1; a.r2 = a.r3 = 1;
    a.nb00 = 2; a.nb01 = row; a.nb02 = a.nb03 = (uint64_t)m * row;
    a.nb10 = 4; a.nb11 = a.nb12 = a.nb13 = (uint64_t)k * 4; a.nr0 = nr;
    return a;
}
static void plain_pair(id<MTLCommandBuffer> cb, MV a, short nsg,
                       id<MTLBuffer> w0, id<MTLBuffer> w1, id<MTLBuffer> x,
                       id<MTLBuffer> y0, id<MTLBuffer> y1) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pipeline(@"kernel_mul_mv_f16_f32_pair_4", nsg)];
    [e setBytes:&a length:sizeof(a) atIndex:0];
    bind(e,w0,1); bind(e,w1,2); bind(e,x,3); bind(e,y0,4); bind(e,y1,5);
    dispatch(e,(a.ne01+a.nr0-1)/a.nr0,nsg,32*a.nr0*4);
}
static void plain_q8(id<MTLCommandBuffer> cb, MV a, short nsg,
                     id<MTLBuffer> w, id<MTLBuffer> x, id<MTLBuffer> y) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pipeline(@"kernel_mul_mv_q8_0_f32", nsg)];
    [e setBytes:&a length:sizeof(a) atIndex:0]; bind(e,w,1); bind(e,x,2); bind(e,y,3);
    dispatch(e,(a.ne01+1)/2,nsg,32*2*4);
}
static void state_store(id<MTLCommandBuffer> cb, Store s,
                        id<MTLBuffer> y0, id<MTLBuffer> y1, id<MTLBuffer> ape,
                        id<MTLBuffer> s0, id<MTLBuffer> s1) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pipeline(@"reference_compressor_store", 1)];
    [e setBytes:&s length:sizeof(s) atIndex:0];
    bind(e,y0,1); bind(e,y1,2); bind(e,ape,3); bind(e,s0,4); bind(e,s1,5);
    dispatch(e,(s.width+31)/32,1,0);
}


#endif
