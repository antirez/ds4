/* Independent dyadic Q2 oracle for the owned down operator.
 * Uses actual production strides, inaccessible unowned pages and canaries. */
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

enum { E=384,U=6,H=2304,M=5120 };
typedef struct { uint8_t scales[16],qs[64]; uint16_t d,dmin; } q2;
static void route(int32_t *ids,unsigned t,unsigned rank,unsigned mode) {
    for(unsigned j=0;j<U;++j) {
        if(mode==0)ids[j]=(j<3?rank:1-rank)*192+(j==2?191:j);
        else if(mode==1)ids[j]=((t*6+j)*7)%384;
        else ids[j]=j?((rank*192+1+(t*5+j-1)%383)%384):rank*192;
    }
}
static float input(unsigned e,unsigned r,unsigned t) {
    if(r%127)return 0.f;
    float x=(float)((e%7+1)*(r%3+1))/64.f;
    return (r+t)&1?-x:x;
}
static double weight(unsigned e,unsigned r,unsigned b) {
    return (double)(1+(e+r+b)%3)*(1+(r+b)%3)/32.;
}
static int run(unsigned n,unsigned rank,unsigned mode) {
    const size_t expert=(size_t)M*9*sizeof(q2), bytes=E*expert;
    unsigned char *model=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    CHECK(model!=MAP_FAILED);
    for(unsigned e=rank*192;e<(rank+1)*192;++e) {
        q2 *d=(q2*)(model+e*expert);
        for(unsigned r=0;r<M;++r)for(unsigned b=0;b<9;++b) {
            q2 *v=d+(size_t)r*9+b;v->d=0x2800;
            memset(v->scales,1+(e+r+b)%3,16);memset(v->qs,(1+(r+b)%3)*0x55,64);
        }
    }
    size_t pairs=(size_t)n*U,nmid=pairs*H,ny=(size_t)n*M,ns=pairs*M;
    int32_t *ids=malloc(pairs*4);float *mid=malloc(nmid*4),*got=malloc(ny*4),*poison=malloc(ns*4);
    uint16_t *slots=malloc(ns*2); CHECK(ids&&mid&&got&&poison&&slots);
    unsigned counts[E]={0};
    for(unsigned t=0;t<n;++t) {
        route(ids+(size_t)t*U,t,rank,mode%3);
        for(unsigned j=0;j<U;++j) {
            unsigned e=ids[(size_t)t*U+j];++counts[e];
            for(unsigned r=0;r<H;++r)mid[((size_t)t*U+j)*H+r]=input(e,r,t);
        }
    }
    for(size_t i=0;i<ns;++i)poison[i]=1234.f;
    ds4_gpu_set_deepseek41_model(true);CHECK(ds4_gpu_set_model_map(model,bytes));
    CHECK(!mprotect(model+(1-rank)*192*expert,192*expert,PROT_NONE));
    ds4_gpu_tensor *it=upload(ids,pairs*4),*mt=upload(mid,nmid*4),*dt=upload(poison,ns*4),*ot=upload(NULL,ny*4);
    CHECK(it&&mt&&dt&&ot);
    if(mode>=4) {
        if(mode==6)ids[1]=ids[0];else ids[0]=mode==4?-1:384;
        CHECK(ds4_gpu_tensor_write(it,0,ids,pairs*4));
        CHECK(!ds4_gpu_dsv41_moe_tp_down(ot,dt,mt,it,model,bytes,0,n,rank));
        CHECK(sync_guards());puts("Invalid or duplicate expert ID rejected PASS");goto cleanup;
    }
    RUN(ds4_gpu_dsv41_moe_tp_down(ot,dt,mt,it,model,bytes,0,n,rank));
    CHECK(ds4_gpu_tensor_read(ot,0,got,ny*4));CHECK(ds4_gpu_tensor_read(dt,0,slots,ns*2));
    if(mode==3)got[0]+=1.f;
    size_t failures=0;
    for(unsigned t=0;t<n;++t) {
        double sums[U][9]={{0}};
        for(unsigned j=0;j<U;++j) {
            unsigned e=ids[(size_t)t*U+j];if(e/192!=rank)continue;
            for(unsigned r=0;r<H;r+=127) {
                float x=input(e,r,t);
                if(n>=8 && counts[e]>=8)x=(float)(_Float16)x;
                sums[j][r/256]+=x;
            }
        }
        for(unsigned r=0;r<M;++r) {
            float expected=0;
            for(unsigned j=0;j<U;++j) {
                unsigned e=ids[(size_t)t*U+j];double v=0;
                for(unsigned b=0;b<9;++b)v+=sums[j][b]*weight(e,r,b);
                _Float16 h=(_Float16)v;uint16_t bits;memcpy(&bits,&h,2);
                size_t ix=((size_t)t*U+j)*M+r;
                if(slots[ix]!=bits) {
                    if(!failures)fprintf(stderr,"slot t=%u j=%u r=%u actual=%04x expected=%04x\n",t,j,r,slots[ix],bits);
                    ++failures;
                }
                expected+=(float)h;
            }
            if(got[(size_t)t*M+r]!=expected) {
                if(!failures)fprintf(stderr,"sum t=%u r=%u actual=%.9g expected=%.9g\n",t,r,got[(size_t)t*M+r],expected);
                ++failures;
            }
        }
    }
    fprintf(stderr,"Owned down rows=%u rank=%u mode=%u outputs=%zu expert_slots=%zu failures=%zu\n",n,rank,mode,ny,ns,failures);
    CHECK(!failures&&sync_guards());
cleanup:
    ds4_gpu_tensor_free(ot);ds4_gpu_tensor_free(dt);ds4_gpu_tensor_free(mt);ds4_gpu_tensor_free(it);
    ds4_gpu_cleanup();munmap(model,bytes);free(slots);free(poison);free(got);free(mid);free(ids);return 1;
}
int main(int argc,char **argv) {
    if(argc!=4)return 2;
    unsigned n=strtoul(argv[1],NULL,10),rank=strtoul(argv[2],NULL,10),mode=strtoul(argv[3],NULL,10);
    if(n<2||n>2048||rank>1||mode>6)return 2;
    fprintf(stderr,"PID=%ld rows=%u rank=%u mode=%u\n",(long)getpid(),n,rank,mode);
    if(!ds4_gpu_init()||!run(n,rank,mode))return 1;
    puts("TP owned down case PASS");return 0;
}

