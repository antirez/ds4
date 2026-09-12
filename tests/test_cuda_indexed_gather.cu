#include "ds4_gpu.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); \
} } while (0)

static float value(size_t i, unsigned seed) {
    uint32_t x=(uint32_t)i*747796405u+seed*2891336453u;
    x=((x>>((x>>28u)+4u))^x)*277803737u;
    x=(x>>22u)^x;
    return ((int)(x%20001u)-10000)/40961.0f;
}
template<class T> static ds4_gpu_tensor *upload(const std::vector<T>& data) {
    ds4_gpu_tensor *tensor=ds4_gpu_tensor_alloc(data.size()*sizeof(T)); CHECK(tensor);
    CHECK(ds4_gpu_tensor_write(tensor,0,data.data(),data.size()*sizeof(T)));
    return tensor;
}
struct Shape {
    uint32_t nt=1,pos=32768,nr=256,cap=2304,start=2133,nc=8192,k=512;
    uint32_t window=256,ratio=4,nh=64,dim=512,pattern=0;
};
struct Case {
    Shape s;
    std::vector<float> q,raw,comp,output,reference;
    std::vector<int32_t> topk;
    ds4_gpu_tensor *dq,*dr,*dc,*dt,*storage,*dst;
    Case(Shape shape,unsigned seed):s(shape),
        q((size_t)s.nt*s.nh*s.dim),raw((size_t)s.cap*s.dim),
        comp((size_t)std::max(1u,s.nc)*s.dim),output(q.size()+32u,12345.25f),
        topk((size_t)s.nt*std::max(1u,s.k)) {
        for(size_t i=0;i<q.size();i++)q[i]=value(i,seed+1);
        for(size_t i=0;i<raw.size();i++)raw[i]=value(i,seed+2);
        for(size_t i=0;i<comp.size();i++)comp[i]=value(i,seed+3);
        if(s.pattern==5) {
            const uint32_t bits[]={0u,0x80000000u,1u,0x80000001u,
                0x00800000u,0x80800000u,0x3e000001u,0xbe000001u};
            for(auto *data:{&q,&raw,&comp})for(size_t i=0;i<data->size();i++)
                memcpy(&(*data)[i],&bits[(i+seed)%8u],sizeof(float));
        }
        for(size_t i=0;i<topk.size();i++) {
            int32_t index=(int32_t)((i*7919u+seed)%std::max(1u,s.nc));
            if(s.pattern==1 && i%3u==0)index=-1;
            if(s.pattern==2 && i%3u==0)index=(int32_t)s.nc+7;
            if(s.pattern==3)index=i%2u?-1:0;
            if(s.pattern==4)index=-1;
            topk[i]=index;
        }
        const uint32_t visible=s.ratio?std::min(s.nc,(s.pos+s.nt)/s.ratio):s.nc;
        for(size_t i=(size_t)visible*s.dim;i<comp.size();i++)comp[i]=NAN;
        dq=upload(q);dr=upload(raw);dc=upload(comp);dt=upload(topk);storage=upload(output);
        dst=ds4_gpu_tensor_view(storage,16u*sizeof(float),q.size()*sizeof(float));CHECK(dst);
    }
    ~Case() {
        for(auto *t:{dst,storage,dt,dc,dr,dq})ds4_gpu_tensor_free(t);
    }
    Case(const Case&)=delete;
    Case& operator=(const Case&)=delete;
    bool eligible(const char *mode="1") const {
        return !strcmp(mode,"1") && s.k<=512 &&
            s.nt==1 && s.nh==64 && s.dim==512 && s.nr<=256 &&
            s.pos<UINT32_MAX && s.nr<=s.pos+1u;
    }
    int infer(float *model) {
        return ds4_gpu_attention_indexed_mixed_batch_heads_tensor(dst,model,4096,0,
            dq,dr,dc,0,dt,s.nt,s.pos,s.nr,s.cap,s.start,s.nc,s.k,
            s.window,s.ratio,s.nh,s.dim);
    }
    void read() {
        CHECK(ds4_gpu_tensor_read(storage,0,output.data(),output.size()*sizeof(float)));
        for(size_t i=0;i<16u;i++) {
            CHECK(output[i]==12345.25f);
            CHECK(output[q.size()+16u+i]==12345.25f);
        }
        for(size_t i=0;i<q.size();i++)CHECK(std::isfinite(output[i+16u]));
    }
    void baseline(float *model) {
        CHECK(unsetenv("DS4_CUDA_INDEXED_DECODE_GATHER")==0);
        CHECK(infer(model));read();reference.assign(output.begin()+16,output.end()-16);
    }
    void exact() {
        read();CHECK(!memcmp(reference.data(),output.data()+16,q.size()*sizeof(float)));
    }
    // Independent double-precision checks across query/head/dimension ends.
    void scalar(float *model) {
        for(uint32_t t:{0u,s.nt-1u})for(uint32_t h:{0u,s.nh-1u}) {
            const uint32_t pos=s.pos+t,first=s.pos+s.nt-s.nr;
            const uint32_t visible=s.ratio?std::min(s.nc,(pos+1u)/s.ratio):s.nc;
            std::vector<const float*> rows;
            for(uint32_t r=0;r<s.nr && rows.size()<256u;r++) {
                const uint32_t p=first+r;
                if(p<=pos && (!s.window || pos+1u<=s.window || p>=pos+1u-s.window))
                    rows.push_back(raw.data()+(size_t)((s.start+r)%s.cap)*s.dim);
            }
            for(uint32_t k=0;k<s.k;k++) {
                int32_t index=topk[(size_t)t*s.k+k];
                if(index>=0 && (uint32_t)index<visible)
                    rows.push_back(comp.data()+(size_t)index*s.dim);
            }
            std::vector<double> scores(rows.size());double maximum=model[h];
            for(size_t r=0;r<rows.size();r++) {
                double sum=0;
                for(uint32_t d=0;d<s.dim;d++)sum+=(double)q[((size_t)t*s.nh+h)*s.dim+d]*rows[r][d];
                scores[r]=sum/sqrt((double)s.dim);maximum=std::max(maximum,scores[r]);
            }
            double denom=exp(model[h]-maximum);
            for(auto &score:scores)denom+=score=exp(score-maximum);
            for(uint32_t d:{0u,s.dim/2u,s.dim-1u}) {
                double expected=0;
                for(size_t r=0;r<rows.size();r++)expected+=rows[r][d]*scores[r]/denom;
                CHECK(fabs(expected-reference[((size_t)t*s.nh+h)*s.dim+d])<2e-5);
            }
        }
    }
};

static void churn_scratch() {
    const uint32_t nc=65536,nt=16,k=2048;
    std::vector<float> scores((size_t)nc*nt);
    for(size_t i=0;i<scores.size();i++)scores[i]=(float)(i%nc);
    std::vector<uint32_t> selected((size_t)nt*k);
    auto *ds=upload(scores),*di=upload(selected);
    CHECK(ds4_gpu_indexer_topk_tensor(di,ds,nc,nt,k));
    CHECK(ds4_gpu_tensor_read(di,0,selected.data(),selected.size()*sizeof(uint32_t)));
    for(uint32_t t=0;t<nt;t++) {
        std::vector<uint32_t> row(selected.begin()+t*k,selected.begin()+(t+1u)*k);
        std::sort(row.begin(),row.end());
        for(uint32_t j=0;j<k;j++)CHECK(row[j]==nc-k+j);
    }
    ds4_gpu_tensor_free(di);ds4_gpu_tensor_free(ds);
    puts("PASS existing top-k scratch growth/overwrite");
}

int main(int argc,char **argv) {
    const bool dispatch_probe=argc==4 && !strcmp(argv[1],"--dispatch-probe");
    CHECK(argc==1 || dispatch_probe);
    CHECK(ds4_gpu_init());
    float *model=NULL;CHECK(cudaMallocHost((void**)&model,4096)==cudaSuccess);
    for(unsigned i=0;i<1024;i++)model[i]=value(i,99);
    CHECK(setenv("DS4_CUDA_COPY_MODEL","1",1)==0);
    CHECK(ds4_gpu_set_model_map(model,4096));CHECK(unsetenv("DS4_CUDA_COPY_MODEL")==0);
    if(dispatch_probe) {
        char *end=NULL;errno=0;const unsigned long long pos=strtoull(argv[3],&end,10);
        CHECK(!errno && end!=argv[3] && !*end && pos<UINT32_MAX);
        Shape shape;shape.pos=(uint32_t)pos;shape.nr=std::min(256u,shape.pos+1u);
        shape.nc=std::max(1u,(shape.pos+1u)/4u);
        {
            Case c(shape,913);c.baseline(model);c.scalar(model);
            CHECK(setenv("DS4_CUDA_INDEXED_DECODE_GATHER",argv[2],1)==0);
            CHECK(c.infer(model));c.exact();
            printf("PASS dispatch probe mode=%s pos=%u expected_gather_kernel_calls=%u\n",
                argv[2],shape.pos,c.eligible(argv[2])?1u:0u);
        }
        CHECK(cudaDeviceSynchronize()==cudaSuccess);
        ds4_gpu_cleanup();CHECK(cudaFreeHost(model)==cudaSuccess);
        return 0;
    }
    unsigned cases=0,expected_gather_calls=0;
    auto run=[&](Shape shape) {
        Case c(shape,++cases);c.baseline(model);c.scalar(model);
        size_t before,total,after;CHECK(cudaMemGetInfo(&before,&total)==cudaSuccess);
        for(const char *mode:{"0","invalid","1"}) {
            CHECK(setenv("DS4_CUDA_INDEXED_DECODE_GATHER",mode,1)==0);
            for(int repeat=0;repeat<3;repeat++) {CHECK(c.infer(model));c.exact();}
            expected_gather_calls+=c.eligible(mode)?3u:0u;
            printf("dispatch,case,%u,mode,%s,expected_gathers,%u\n",
                cases,mode,c.eligible(mode)?3u:0u);
        }
        CHECK(cudaMemGetInfo(&after,&total)==cudaSuccess);
        printf("case,%u,eligible,%u,values,%zu,exact,1,scratch_growth_bytes,%zu\n",
            cases,c.eligible(),c.q.size(),before>after?before-after:0);fflush(stdout);
    };
    Shape first;first.nr=1;first.nc=3;first.k=1;run(first);
    run(Shape{}); // A real growth from ~1 MiB to ~1.5 MiB, not just reuse.
    Shape extremes;extremes.pattern=5;run(extremes);
    for(uint32_t pos:{2047u,8191u,16383u,32766u,32767u,32768u,32769u,131045u,262143u}) {
        Shape boundary;boundary.pos=pos;boundary.nc=(pos+1u)/4u;
        boundary.pattern=1;run(boundary);
    }
    churn_scratch();
    for(unsigned i=0;i<36;i++) {
        Shape s;s.nc=i%4u==0?3u:i%4u==1?257u:i%4u==2?8192u:65536u;
        s.nr=i%3u==0?1u:i%3u==1?255u:256u;
        s.k=i%4u==0?1u:i%4u==1?31u:i%4u==2?255u:512u;
        s.ratio=i%7u==0?0u:4u;s.pos=i%5u==0?17u:s.nc*4u-1u;
        s.nr=std::min(s.nr,s.pos+1u);s.start=i%2u?0u:s.cap-1u;
        s.window=i%5u==0?0u:i%5u==1?7u:256u;s.pattern=i%5u;
        if(i==31)s.nh=17;
        if(i==32)s.dim=128;
        // Valid multi-token inputs exercise the unchanged prefill fallback;
        // invalid and invisible selected rows are covered on decode above.
        if(i==33){s.nt=2;s.nr=2;s.pattern=0;}
        if(i==34)s.nr=257;
        run(s);
    }
    { // Pending A/B calls share scratch but never conversation outputs.
        Shape a,b;b.pos=131045;b.nc=32768;b.start=b.cap-1u;b.pattern=3;
        Case ca(a,701),cb(b,702);ca.baseline(model);cb.baseline(model);
        for(int repeat=0;repeat<8;repeat++) {
            CHECK(setenv("DS4_CUDA_INDEXED_DECODE_GATHER","1",1)==0);
            CHECK(ca.infer(model));CHECK(cb.infer(model));
            if(repeat==3)churn_scratch();
            ca.exact();cb.exact();
        }
        expected_gather_calls+=16;
        puts("PASS alternating live buffers and intervening scratch owner");
        // Existing rejection behavior remains rejection, with no new launch.
        ca.s.k=513;CHECK(!ca.infer(model));ca.s.k=512;
        ca.s.nr=0;CHECK(!ca.infer(model));ca.s.nr=256;
        ca.s.start=ca.s.cap;CHECK(!ca.infer(model));ca.s.start=0;
        ca.s.nc=0;CHECK(!ca.infer(model));ca.s.nc=8192;
    }
    CHECK(unsetenv("DS4_CUDA_INDEXED_DECODE_GATHER")==0);
    CHECK(cudaDeviceSynchronize()==cudaSuccess);
    ds4_gpu_cleanup();CHECK(cudaFreeHost(model)==cudaSuccess);
    printf("PASS %u API cases; expected_gather_kernel_calls=%u; exact floats, guards, scalar oracle, scratch transitions\n",
        cases,expected_gather_calls);
}
