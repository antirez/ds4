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
static unsigned cases, seed = 0x19ae548d;
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

/* kind: paired compressor, quad compressor, Q8 QKV + quad compressor. */
static void compressor(int kind, unsigned k, unsigned m0, unsigned m1,
                       short nsg, unsigned nr, unsigned ratio, unsigned pos,
                       unsigned ape_type, unsigned pad) {
    @autoreleasepool {
        if (kind) nr = 2;
        unsigned row = (k + pad) * 2, p0 = (m0+nr-1)/nr*nr, p1=(m1+nr-1)/nr*nr;
        MV a = args(k,m0,nr,row);
        Store st[2] = {{m0,ratio,pos,ape_type}, {m1,ratio,pos,1-ape_type}};
        id<MTLBuffer> x = floats(k), weights[4] = {halfs(p0*(k+pad)),halfs(p0*(k+pad)),
                                                  halfs(p1*(k+pad)),halfs(p1*(k+pad))};
        id<MTLBuffer> ape[2] = {ape_type == 1 ? halfs(ratio*m0) : floats(ratio*m0),
                               ape_type == 0 ? halfs(ratio*m1) : floats(ratio*m1)};
        id<MTLBuffer> actual[4], reference[4], state[4], refstate[4];
        for (unsigned i=0;i<4;i++) {
            unsigned m=i<2?m0:m1, rows=ratio==4?8:ratio;
            actual[i]=buffer(m*4); reference[i]=copy(actual[i]);
            state[i]=floats(m*rows); refstate[i]=copy(state[i]);
        }
        const unsigned q0m=7, q1m=4, qrow=(k/32+1)*sizeof(Q8), vtgs=4;
        MV qa[2] = {args(k,q0m,2,qrow),args(k,q1m,2,qrow)};
        id<MTLBuffer> qw[2] = {q8s(8*(k/32+1)),q8s(4*(k/32+1))};
        id<MTLBuffer> qout[2]={buffer(q0m*4),buffer(q1m*4)};
        id<MTLBuffer> qref[2]={copy(qout[0]),copy(qout[1])};
        /* Ratio-128 production can omit the indexer compressor entirely and
         * bind its unused resources to the first compressor's buffers. */
        if (kind && !m1) {
            weights[2]=weights[0];weights[3]=weights[1];ape[1]=ape[0];
            actual[2]=actual[0];actual[3]=actual[1];
            reference[2]=reference[0];reference[3]=reference[1];
            state[2]=state[0];state[3]=state[1];
            refstate[2]=refstate[0];refstate[3]=refstate[1];
        }
        id<MTLCommandBuffer> cb=[queue commandBuffer];
        for (unsigned i=0;i<(kind?2u:1u);i++) {
            if (!st[i].width) continue;
            MV pa=a; pa.ne01=pa.ne0=st[i].width;
            plain_pair(cb,pa,nsg,weights[i*2],weights[i*2+1],x,reference[i*2],reference[i*2+1]);
            state_store(cb,st[i],reference[i*2],reference[i*2+1],ape[i],refstate[i*2],refstate[i*2+1]);
        }
        if (kind==2) for(unsigned i=0;i<2;i++) plain_q8(cb,qa[i],4,qw[i],x,qref[i]);
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        NSString *name=kind==0?@"kernel_mul_mv_f16_f32_pair_compressor_store_4":
                       kind==1?@"kernel_mul_mv_f16_f32_quad_compressor_store_4":
                               @"kernel_dsv4_qkv_pair_quad_compressor_store_q8_0";
        [e setComputePipelineState:pipeline(name,nsg)];
        if(kind==0) {
            [e setBytes:&a length:sizeof(a) atIndex:0];
            [e setBytes:&st[0] length:sizeof(Store) atIndex:1];
            bind(e,weights[0],2);bind(e,weights[1],3);bind(e,x,4);
            bind(e,actual[0],5);bind(e,actual[1],6);bind(e,ape[0],7);
            bind(e,state[0],8);bind(e,state[1],9);
        } else if(kind==1) {
            [e setBytes:&a length:sizeof(a) atIndex:0];
            [e setBytes:&st[0] length:sizeof(Store) atIndex:1];
            [e setBytes:&st[1] length:sizeof(Store) atIndex:2];
            for(unsigned i=0;i<4;i++)bind(e,weights[i],3+i);
            bind(e,x,7);for(unsigned i=0;i<4;i++)bind(e,actual[i],8+i);
            bind(e,ape[0],12);bind(e,ape[1],13);
            for(unsigned i=0;i<4;i++)bind(e,state[i],14+i);
        } else {
            [e setBytes:&qa[0] length:sizeof(MV) atIndex:0];
            [e setBytes:&qa[1] length:sizeof(MV) atIndex:1];
            [e setBytes:&a length:sizeof(a) atIndex:2];
            [e setBytes:&st[0] length:sizeof(Store) atIndex:3];
            [e setBytes:&st[1] length:sizeof(Store) atIndex:4];
            [e setBytes:&vtgs length:sizeof(vtgs) atIndex:5];
            bind(e,qw[0],6);bind(e,qw[1],7);
            for(unsigned i=0;i<4;i++)bind(e,weights[i],8+i);
            bind(e,x,12);bind(e,qout[0],13);bind(e,qout[1],14);
            for(unsigned i=0;i<4;i++)bind(e,actual[i],15+i);
            bind(e,ape[0],19);bind(e,ape[1],20);
            for(unsigned i=0;i<4;i++)bind(e,state[i],21+i);
        }
        unsigned groups=(m0+nr-1)/nr+(kind?(m1+nr-1)/nr:0)+(kind==2?(vtgs+1)/2:0);
        dispatch(e,groups,nsg,kind==2?1024:2*32*nr*4);finish(cb);
        for(unsigned i=0;i<(kind?4u:2u);i++) {
            equal(actual[i],reference[i],"F16 projection");equal(state[i],refstate[i],"compressor state");
            guards(actual[i]);guards(state[i]);
        }
        if(kind==2)for(unsigned i=0;i<2;i++){equal(qout[i],qref[i],"compound Q8");guards(qout[i]);}
        for(unsigned i=0;i<4;i++)guards(weights[i]);guards(x);guards(ape[0]);guards(ape[1]);
        cases++;
    }
}

static void q8_hc(int kind,unsigned k,unsigned m,short nsg,unsigned stride) {
    @autoreleasepool {
        unsigned row=(k/32+1)*sizeof(Q8), padded=(m+1)/2*2;
        MV a=args(k,m,2,row);
        HC h={0};h.n_embd=m;h.n_hc=4;h.n_tokens=1;h.has_add=kind==0;
        h.nb_block0=4*stride;h.nb_block1=m*h.nb_block0;
        h.nb_add0=4;h.nb_add1=m*4;
        h.nb_res0=4*stride;h.nb_res1=(m*stride+5)*4;h.nb_res2=h.nb_res1*4;
        h.nb_post0=kind==2?4:4*stride;h.nb_post1=h.nb_post0*4;
        h.nb_comb0=kind==2?4:4*stride;h.nb_comb1=h.nb_comb0*4+16;h.nb_comb2=h.nb_comb1*4;
        h.nb0=4*stride;h.nb1=(m*stride+7)*4;h.nb2=h.nb1*4;
        id<MTLBuffer>w=q8s(padded*(k/32+1)),x=floats(k),block=buffer(m*4),refblock=copy(block);
        id<MTLBuffer>routed=floats(m*stride),res=floats(h.nb_res2/4),post=floats(h.nb_post1/4);
        id<MTLBuffer>comb=floats(h.nb_comb2/4),out=buffer(h.nb2),refout=copy(out);
        id<MTLCommandBuffer>cb=[queue commandBuffer];
        plain_q8(cb,a,nsg,w,x,refblock);
        HC rh=h;if(kind!=0){rh.nb_block0=4;rh.nb_block1=m*4;}
        id<MTLComputeCommandEncoder>e=[cb computeCommandEncoder];
        [e setComputePipelineState:pipeline(@"kernel_dsv4_hc_expand4",1)];
        [e setBytes:&rh length:sizeof(rh) atIndex:0];
        bind(e,kind==0?routed:refblock,1);bind(e,res,2);bind(e,post,3);bind(e,comb,4);
        bind(e,refblock,5);bind(e,refout,6);dispatch(e,(m+31)/32,1,0);
        e=[cb computeCommandEncoder];
        NSString*name=kind==0?@"kernel_dsv4_shared_down_hc_expand4_q8_0":
                      kind==1?@"kernel_dsv4_q8_hc_expand4_q8_0":@"kernel_dsv4_q8_hc_expand4_q8_0_vec_hc";
        [e setComputePipelineState:pipeline(name,nsg)];
        [e setBytes:&a length:sizeof(a) atIndex:0];[e setBytes:&h length:sizeof(h) atIndex:1];
        bind(e,w,2);bind(e,x,3);bind(e,block,4);
        unsigned next=5;if(kind==0)bind(e,routed,next++);
        bind(e,res,next++);bind(e,post,next++);bind(e,comb,next++);bind(e,out,next);
        dispatch(e,(m+1)/2,nsg,32*2*4);finish(cb);
        equal(block,refblock,"Q8 materialized projection");equal(out,refout,"Q8 HC expansion");
        for(id<MTLBuffer>b in @[w,x,block,routed,res,post,comb,out])guards(b);
        cases++;
    }
}

int main(int argc,char**argv) {
    @autoreleasepool {
        if(argc!=2)die("usage: test_metal_decode_fusions SOURCE.metal");
        device=MTLCreateSystemDefaultDevice();if(!device)die("No Metal GPU available");
        queue=[device newCommandQueue];
        NSError*error=nil;
        NSString*source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                 encoding:NSUTF8StringEncoding error:&error];
        if(!source)die(error.description.UTF8String);
        for(int fast=0;fast<2;fast++) {
            MTLCompileOptions*options=[MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled=fast;
#pragma clang diagnostic pop
            library=[device newLibraryWithSource:source options:options error:&error];
            if(!library)die(error.description.UTF8String);
            pipelines=[NSMutableDictionary new];unsigned before=cases;
            for(unsigned ape=0;ape<=1;ape++)for(unsigned ri=0;ri<2;ri++) {
                unsigned ratio=ri?128:4,pos=ri?383:11;
                compressor(0,100,17,10,4,2,ratio,pos,ape,4);
                compressor(0,2048,18,10,8,2,ratio,pos,ape,8);
                compressor(0,4096,512,10,8,4,ratio,pos,ape,0);
                compressor(1,100,18,10,8,2,ratio,pos,ape,4);
                compressor(1,4096,512,256,8,2,ratio,pos,ape,0);
                compressor(2,2048,18,10,8,2,ratio,pos,ape,8);
                compressor(2,4096,512,256,8,2,ratio,pos,ape,0);
            }
            compressor(2,2048,512,0,8,2,128,0,1,0);
            compressor(2,4096,512,0,8,2,128,UINT32_MAX,0,0);
            compressor(0,100,18,10,4,2,4,0,1,4);
            compressor(0,100,18,10,8,2,4,1,0,4);
            for(int kind=0;kind<3;kind++)for(short nsg=1;nsg<=8;nsg*=2) {
                q8_hc(kind,96,17,nsg,1);q8_hc(kind,2048,18,nsg,3);
                q8_hc(kind,4096,512,nsg,1);
            }
            fprintf(stderr,"Metal decode fusions %s: %u cases bitwise PASS (%s)\n",
                    fast?"fast":"strict",cases-before,device.name.UTF8String);
        }
        return 0;
    }
}
