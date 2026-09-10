#include "metal_decode_test_support.h"

static unsigned cases;

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
