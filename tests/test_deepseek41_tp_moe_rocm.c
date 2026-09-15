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

#include <sys/mman.h>

/* IQ2 grid0/sign0 is eight copies of8; scale0 decodes exactly to d.
 * Q2_K uses constant two-bit lanes and explicit per-block scales. The
 * independent oracle needs no GPU quantization or dot-product helper. */
enum {E=384,U=6,K=5120,H=2304,M=5120};
typedef struct { uint16_t d,qs[32]; } iq2;
typedef struct { uint8_t scales[16],qs[64];uint16_t d,dmin; } q2;
static void route(int32_t *ids,unsigned rank,unsigned mode) {
    const int32_t balanced[6]={0,191,192,193,382,383},skewed[6]={0,1,2,3,191,192};
    for(unsigned j=0;j<6;++j) {
        if(mode<2)ids[j]=(int32_t)((mode?1-rank:rank)*192+(j<4?j:186+j));
        else if(mode==2)ids[j]=balanced[j];
        else ids[j]=rank?383-skewed[j]:skewed[j];
    }
}
static double gate_ref(unsigned e,unsigned row,unsigned tok,int up) {
    unsigned block=(e+row)%20;
    double x=(tok+block)&1?127.:-127.;
    unsigned sub=up?(e+row)%4:e%3;
    return x*(2*sub+1)/(up?256.:128.);
}
static double down_weight(unsigned e,unsigned row,unsigned block) {
    return (double)(1+(e+row+block)%3)*(1+(row+block)%3)/32.;
}
static int run(unsigned n,unsigned rank,unsigned mode) {
    const size_t ge=(size_t)H*(K/256)*sizeof(iq2),de=(size_t)M*(H/256)*sizeof(q2);
    const size_t table=E*ge,wb=2*table+E*de;
    unsigned char *model=mmap(NULL,wb,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);CHECK(model!=MAP_FAILED);
    int32_t ids_for[6];bool used[E]={0};
    for(unsigned r=0;r<2;++r)for(unsigned m=0;m<4;++m) {
        route(ids_for,r,m);for(unsigned j=0;j<U;++j)used[ids_for[j]]=true;
    }
    for(unsigned e=0;e<E;++e)if(used[e]) {
        for(unsigned r=0;r<H;r+=127) {
            unsigned b=(e+r)%20;
            iq2 *g=(iq2*)(model+e*ge)+(size_t)r*20+b;
            iq2 *u=(iq2*)(model+table+e*ge)+(size_t)r*20+b;
            g->d=0x2000;u->d=0x1c00;
            for(unsigned j=0;j<8;++j) {g->qs[j*4+3]=(e%3)<<12;u->qs[j*4+3]=((e+r)%4)<<12;}
        }
        q2 *d=(q2*)(model+2*table+e*de);
        for(unsigned r=0;r<M;++r)for(unsigned b=0;b<9;++b) {
            q2 *v=d+(size_t)r*9+b;v->d=0x2800;
            memset(v->scales,1+(e+r+b)%3,16);
            memset(v->qs,(1+(r+b)%3)*0x55,64);
        }
    }
    size_t pairs=(size_t)n*U,nmid=pairs*H,ny=(size_t)n*M;
    int32_t *ids=malloc(pairs*4);float *rw=malloc(pairs*4),*x=calloc((size_t)n*K,4),*scratch=malloc(nmid*4),*got=malloc(ny*4);CHECK(ids&&rw&&x&&scratch&&got);
    for(unsigned t=0;t<n;++t) {
        route(ids+(size_t)t*U,rank,(t+mode)%4);
        for(unsigned j=0;j<U;++j)rw[(size_t)t*U+j]=(j+1)/16.f;
        for(unsigned b=0;b<20;++b)x[(size_t)t*K+b*256]=(t+b)&1?127.f:-127.f;
    }
    ds4_gpu_set_deepseek41_model(true);CHECK(ds4_gpu_set_model_map(model,wb));
    /* A wrong ownership span must fail instead of quietly reading zeros. */
    unsigned other=1-rank;
    CHECK(!mprotect(model+other*192*ge,192*ge,PROT_NONE));
    CHECK(!mprotect(model+table+other*192*ge,192*ge,PROT_NONE));
    CHECK(!mprotect(model+2*table+other*192*de,192*de,PROT_NONE));
    for(size_t i=0;i<nmid;++i)scratch[i]=1234.f;
    ds4_gpu_tensor *it=upload(ids,pairs*4),*wt=upload(rw,pairs*4),*xt=upload(x,(size_t)n*K*4);
    ds4_gpu_tensor *gt=upload(scratch,nmid*4),*ut=upload(scratch,nmid*4),*mt=upload(scratch,nmid*4),*dt=upload(NULL,pairs*M*4),*ot=upload(NULL,ny*4);CHECK(it&&wt&&xt&&gt&&ut&&mt&&dt&&ot);
    if(mode==4 || mode==5) {
        ids[0]=mode==4?-1:384;CHECK(ds4_gpu_tensor_write(it,0,ids,pairs*4));
        CHECK(!ds4_gpu_dsv41_routed_moe_tp_tensor(ot,gt,ut,mt,dt,model,wb,0,table,2*table,it,wt,xt,n,rank));
        CHECK(sync_guards());puts("Invalid global expert ID rejected PASS");goto cleanup;
    }
    RUN(ds4_gpu_dsv41_routed_moe_tp_tensor(ot,gt,ut,mt,dt,model,wb,0,table,2*table,it,wt,xt,n,rank));
    CHECK(ds4_gpu_tensor_read(ot,0,got,ny*4));
    if(mode==6)got[0]+=1.f; /* Deliberate oracle corruption must fail. */
    size_t failures=0;double worst=0;
    if(n>=128) {
        /* Independently qualify weighted mid below. For down, use the observed
         * mid boundary, a double dot and the inherited F16 rounding intervals.
         * gamma2304 bounds FP32 accumulation, including rare half ties. */
        float *observed=malloc(nmid*4);uint16_t *slots=malloc(pairs*M*2);
        CHECK(observed&&slots&&ds4_gpu_tensor_read(mt,0,observed,nmid*4));
        CHECK(ds4_gpu_tensor_read(dt,0,slots,pairs*M*2));
        unsigned counts[E]={0};for(size_t p=0;p<pairs;++p)++counts[ids[p]];
        const double gamma=(H*(double)FLT_EPSILON/2)/(1-H*(double)FLT_EPSILON/2);
        for(unsigned t=0;t<n;++t)for(unsigned r=0;r<M;++r) {
            float lo_sum=0,hi_sum=0;
            for(unsigned j=0;j<U;++j) {
                unsigned e=ids[(size_t)t*U+j];double dot=0,l1=0;
                if(e/192==rank)for(unsigned k=0;k<H;k+=127) {
                    float v=observed[((size_t)t*U+j)*H+k];
                    if(counts[e]>=8)v=(float)(_Float16)v;
                    double term=v*down_weight(e,r,k/256);dot+=term;l1+=fabs(term);
                }
                float lo=(float)(_Float16)(dot-gamma*l1),hi=(float)(_Float16)(dot+gamma*l1);
                _Float16 actual;memcpy(&actual,slots+((size_t)t*U+j)*M+r,2);
                if(!isfinite((float)actual)||(float)actual<lo||(float)actual>hi) {
                    if(!failures)fprintf(stderr,"bulk slot t=%u r=%u j=%u value=%.9g interval=[%.9g,%.9g]\n",t,r,j,(float)actual,lo,hi);
                    ++failures;
                }
                lo_sum+=lo;hi_sum+=hi;
            }
            float v=got[(size_t)t*M+r];
            if(!isfinite(v)||v<lo_sum||v>hi_sum) {
                if(!failures)fprintf(stderr,"bulk sum t=%u r=%u value=%.9g interval=[%.9g,%.9g]\n",t,r,v,lo_sum,hi_sum);
                ++failures;
            }
        }
        free(slots);free(observed);
    }
    for(unsigned t=0;n<128&&t<n;++t) {
        double sums[U][9]={{0}};
        for(unsigned j=0;j<U;++j) {
            unsigned e=ids[(size_t)t*U+j];if(e/192!=rank)continue;
            for(unsigned r=0;r<H;r+=127) {
                double g=gate_ref(e,r,t,0),u=gate_ref(e,r,t,1);
                g=fmin(g,10.);u=fmin(fmax(u,-10.),10.);
                double mid=g/(1+exp(-g))*u*rw[(size_t)t*U+j];sums[j][r/256]+=mid;
            }
        }
        for(unsigned r=0;r<M;++r) {
            double expected=0,l1=0;
            for(unsigned j=0;j<U;++j)for(unsigned b=0;b<9;++b) {
                double term=sums[j][b]*down_weight(ids[(size_t)t*U+j],r,b);expected+=term;l1+=fabs(term);
            }
            double error=fabs(got[(size_t)t*M+r]-expected),tol=3e-5*fmax(1.,l1);worst=fmax(worst,error/tol);
            if(!isfinite(got[(size_t)t*M+r])||error>tol) {if(!failures)fprintf(stderr,"output t=%u r=%u got=%.9g expected=%.17g tol=%.9g\n",t,r,got[(size_t)t*M+r],expected,tol);++failures;}
        }
    }
    for(unsigned stage=0;stage<3;++stage) {
        CHECK(ds4_gpu_tensor_read(stage==0?gt:stage==1?ut:mt,0,scratch,nmid*4));
        for(unsigned t=0;t<n;++t)for(unsigned j=0;j<U;++j) {
            unsigned e=ids[(size_t)t*U+j];
            for(unsigned r=0;r<H;++r) {
                double expected=0;
                if(e/192==rank && r%127==0) {
                    double g=fmin(gate_ref(e,r,t,0),10.),u=fmin(fmax(gate_ref(e,r,t,1),-10.),10.);
                    expected=stage==0?(n>=128?gate_ref(e,r,t,0):g):stage==1?(n>=128?gate_ref(e,r,t,1):u):g/(1+exp(-g))*u*rw[(size_t)t*U+j];
                }
                float v=scratch[((size_t)t*U+j)*H+r];
                double tol=expected==0?0:3e-5*fmax(1.,fabs(expected));
                if(!isfinite(v)||fabs(v-expected)>tol)++failures;
            }
        }
    }
    fprintf(stderr,"MoE rows=%u rank=%u mode=%u full_outputs=%zu failures=%zu worst_bound_fraction=%.9g\n",n,rank,mode,ny,failures,worst);CHECK(!failures&&sync_guards());
cleanup:
    ds4_gpu_tensor_free(ot);ds4_gpu_tensor_free(dt);ds4_gpu_tensor_free(mt);ds4_gpu_tensor_free(ut);ds4_gpu_tensor_free(gt);ds4_gpu_tensor_free(xt);ds4_gpu_tensor_free(wt);ds4_gpu_tensor_free(it);
    ds4_gpu_cleanup();munmap(model,wb);free(ids);free(rw);free(x);free(scratch);free(got);return 1;
}
int main(int argc,char **argv) {
    if(argc!=4)return 2;
    unsigned n=strtoul(argv[1],NULL,10),rank=strtoul(argv[2],NULL,10),mode=strtoul(argv[3],NULL,10);
    if(!n||n>2048||rank>1||mode>6)return 2;
    fprintf(stderr,"PID=%ld rows=%u rank=%u route-mode=%u\n",(long)getpid(),n,rank,mode);
    if(!ds4_gpu_init()||!run(n,rank,mode))return 1;
    puts("TP MoE case PASS");return 0;
}
