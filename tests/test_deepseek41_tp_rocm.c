/* V4.1 Q8 production-shape harness with an independent sparse double oracle,
 * cast-sensitive fixtures and full-output parity against unchanged scalar
 * production calls. Explicit copies, guards and synchronization after stages. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 0; \
} } while (0)
#define RUN(x) do { fprintf(stderr, "stage: %s\n", #x); CHECK(x); CHECK(sync_guards()); } while (0)

enum { GUARD_BYTES = 64 };
typedef struct allocation {
    ds4_gpu_tensor *storage, *view;
    size_t bytes;
    struct allocation *next;
} allocation;
static allocation *allocations;

static int sync_guards(void) {
    CHECK(ds4_gpu_synchronize());
    for (allocation *a = allocations; a; a = a->next) {
        unsigned char before[GUARD_BYTES], after[GUARD_BYTES];
        CHECK(ds4_gpu_tensor_read(a->storage, 0, before, sizeof(before)));
        CHECK(ds4_gpu_tensor_read(a->storage, GUARD_BYTES + a->bytes, after, sizeof(after)));
        for (unsigned i = 0; i < GUARD_BYTES; i++) CHECK(before[i] == 0xa5 && after[i] == 0xa5);
    }
    return 1;
}

static ds4_gpu_tensor *upload(const void *data, size_t bytes) {
    allocation *a = calloc(1, sizeof(*a));
    if (!a || bytes > SIZE_MAX - 2 * GUARD_BYTES) { free(a); return NULL; }
    a->bytes = bytes;
    a->storage = ds4_gpu_tensor_alloc(bytes + 2 * GUARD_BYTES);
    if (!a->storage) { free(a); return NULL; }
    unsigned char guard[GUARD_BYTES];
    memset(guard, 0xa5, sizeof(guard));
    if (!ds4_gpu_tensor_write(a->storage, 0, guard, sizeof(guard)) ||
        !ds4_gpu_tensor_write(a->storage, GUARD_BYTES + bytes, guard, sizeof(guard))) goto fail;
    a->view = ds4_gpu_tensor_view(a->storage, GUARD_BYTES, bytes);
    if (!a->view || (data && !ds4_gpu_tensor_write(a->view, 0, data, bytes))) goto fail;
    a->next = allocations;
    allocations = a;
    return a->view;
fail:
    ds4_gpu_tensor_free(a->view);
    ds4_gpu_tensor_free(a->storage);
    free(a);
    return NULL;
}

static void guarded_free(ds4_gpu_tensor *t) {
    for (allocation **link = &allocations; *link; link = &(*link)->next) {
        allocation *a = *link;
        if (a->view == t) {
            *link = a->next;
            ds4_gpu_tensor_free(a->view);
            ds4_gpu_tensor_free(a->storage);
            free(a);
            return;
        }
    }
    ds4_gpu_tensor_free(t);
}
#define ds4_gpu_tensor_free guarded_free

/* Full production strides, sparse independent double reference, and exact BF16 boundary. */
typedef struct { uint16_t scale; int8_t q[32]; } q8_block;
static float bf16(float x) {
    uint32_t u; memcpy(&u,&x,4);u+=0x7fff+((u>>16)&1);u&=0xffff0000;memcpy(&x,&u,4);return x;
}
static unsigned column(unsigned row,unsigned term,unsigned k) { return (row*97+term*947)%k; }
static int coefficient(unsigned row,unsigned term) { return (int)((row+term*3)%7)-3; }
static void weights(q8_block *w,unsigned rows,unsigned k,unsigned base) {
    for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<8;++j) {
        unsigned c=column(r+base,j,k);q8_block *b=w+(size_t)r*(k/32)+c/32;
        b->scale=0x2000;b->q[c%32]=coefficient(r+base,j);
    }
}
static int run(unsigned n,unsigned rank,int cast_fixture) {
    enum {K=4096,R=1024,G=4,M=5120};
    size_t ab=(size_t)8*R*(K/32)*34,bb=(size_t)M*(8192/32)*34,wb=ab+bb;
    q8_block *model=NULL;CHECK(!posix_memalign((void**)&model,4096,wb));memset(model,0,wb);
    for(unsigned g=0;g<8;++g)weights(model+(size_t)g*R*(K/32),R,K,g*R);
    weights((q8_block*)((char*)model+ab),M,8192,0);
    size_t nx=(size_t)n*G*K,nl=(size_t)n*G*R,ny=(size_t)n*M;
    float *x=malloc(nx*4),*low=malloc(nl*4),*out=malloc(ny*4),*ref=malloc(4096*4);CHECK(x&&low&&out&&ref);
    for(size_t i=0;i<nx;++i)x[i]=cast_fixture?((i&1)?-1:1)*(1.00390625f+0.00001f):((int)((i*37+rank*19)%257)-128)/32.f;
    CHECK(ds4_gpu_set_model_map(model,wb));
    ds4_gpu_tensor *xt=upload(x,nx*4),*lt=upload(NULL,nl*4),*yt=upload(NULL,ny*4);CHECK(xt&&lt&&yt);
    /* Register the full baseline range before the owned subrange, avoiding
     * an overlapping expansion in the existing lazy model-map cache. */
    float *fx=calloc((size_t)n*32768,4),*fl=malloc((size_t)n*8192*4);CHECK(fx&&fl);
    for(unsigned t=0;t<n;++t)memcpy(fx+(size_t)t*32768+rank*16384,x+(size_t)t*16384,16384*4);
    ds4_gpu_tensor *fxt=upload(fx,(size_t)n*32768*4),*flt=upload(NULL,(size_t)n*8192*4),*fyt=upload(NULL,ny*4);CHECK(fxt&&flt&&fyt);
    RUN(ds4_gpu_dsv41_attention_output_batch(fyt,flt,model,wb,0,ab,fxt,n));
    CHECK(ds4_gpu_tensor_read(flt,0,fl,(size_t)n*8192*4));
    RUN(ds4_gpu_dsv41_attention_output_tp_batch(yt,lt,model,wb,0,ab,xt,n,rank));
    CHECK(ds4_gpu_tensor_read(lt,0,low,nl*4));CHECK(ds4_gpu_tensor_read(yt,0,out,ny*4));
    size_t bad_low=0,bad_out=0,outside_low_interval=0;double max_error=0;
    for(unsigned t=0;t<n;++t) {
        for(unsigned g=0;g<G;++g)for(unsigned r=0;r<R;++r) {
            unsigned wr=(rank*G+g)*R+r;double sum=0,l1=0;
            for(unsigned j=0;j<8;++j) {
                float xv=x[((size_t)t*G+g)*K+column(wr,j,K)];
                if(n>=32) xv=(float)(_Float16)xv;
                double term=(double)xv*coefficient(wr,j)/128.;sum+=term;l1+=fabs(term);
            }
            float expected=bf16((float)sum),got=low[((size_t)t*G+g)*R+r];
            /* Bound all K accumulation terms, then propagate through the
             * monotone BF16 rounding operation. Exact ties can round either
             * way after a legal accumulation error; retain their count. */
            double u=FLT_EPSILON/2.,radius=(4096*u/(1-4096*u))*fmax(l1,.001);
            if(!isfinite(got)||got<bf16((float)(sum-radius))||got>bf16((float)(sum+radius)))++outside_low_interval;
            ref[g*R+r]=got;
            if(memcmp(&expected,low+((size_t)t*G+g)*R+r,4)) {
                if(bad_low<1)fprintf(stderr,"low mismatch t=%u g=%u r=%u sum=%a expected=%a actual=%a\n",t,g,r,sum,(double)expected,(double)low[((size_t)t*G+g)*R+r]);
                ++bad_low;
            }
        }
        for(unsigned r=0;r<M;++r) {
            double sum=0,l1=0;
            for(unsigned j=0;j<8;++j) {
                unsigned c=column(r,j,8192);
                if(c<rank*4096 || c>=(rank+1)*4096)continue;
                double term=(double)ref[c-rank*4096]*coefficient(r,j)/128.;sum+=term;l1+=fabs(term);
            }
            double error=fabs(out[(size_t)t*M+r]-sum),u=FLT_EPSILON/2.;
            double tolerance=(4096*u/(1-4096*u))*fmax(l1,.001);
            max_error=fmax(max_error,error);
            if(!isfinite(out[(size_t)t*M+r])||error>tolerance)++bad_out;
        }
    }
    fprintf(stderr,"rows=%u rank=%u cast=%d low=%zu outputs=%zu bad_low=%zu bad_out=%zu max_error=%.17g\n",n,rank,cast_fixture,nl,ny,bad_low,bad_out,max_error);
    size_t differs=0;
    for(unsigned t=0;t<n;++t)for(unsigned j=0;j<4096;++j)
        if(memcmp(fl+(size_t)t*8192+rank*4096+j,low+(size_t)t*4096+j,4))++differs;
    fprintf(stderr,"existing eight-group versus TP low differences=%zu/%zu\n",differs,nl);
    CHECK(!differs);
    ds4_gpu_tensor_free(fyt);ds4_gpu_tensor_free(flt);ds4_gpu_tensor_free(fxt);free(fx);free(fl);
    fprintf(stderr,"independent low interval violations=%zu; strict BF16 oracle mismatches=%zu\n",outside_low_interval,bad_low);
    CHECK(!outside_low_interval&&!bad_out);
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(yt,lt,model,wb,0,ab,xt,n,2));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(yt,lt,model,wb-1,0,ab,xt,n,rank));
    CHECK(!ds4_gpu_dsv41_attention_output_tp_batch(yt,lt,model,wb,0,ab,xt,n+1,rank));
    CHECK(sync_guards());ds4_gpu_tensor_free(yt);ds4_gpu_tensor_free(lt);ds4_gpu_tensor_free(xt);
    ds4_gpu_cleanup();free(model);free(x);free(low);free(out);free(ref);return 1;
}
int main(int argc,char **argv) {
    if(argc!=4)return 2;
    unsigned n=strtoul(argv[1],NULL,10),rank=strtoul(argv[2],NULL,10);int cast=atoi(argv[3]);
    if(!n||n>2048||rank>1||cast<0||cast>1)return 2;
    fprintf(stderr,"PID=%ld rows=%u rank=%u cast=%d\n",(long)getpid(),n,rank,cast);
    if(!ds4_gpu_init()||!run(n,rank,cast)||allocations)return 1;
    puts("TP attention independent intervals, baseline low parity and canaries PASS");return 0;
}
