#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
static void need(int ok,const char *why) { if(!ok){fprintf(stderr,"TOPK FAIL %s\n",why);exit(1);} }
static uint32_t seed=0xb719325e;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
static float bits(uint32_t b){float f;memcpy(&f,&b,4);return f;}
int main(void) {
    enum { MAX=131073, K=512 };
    const uint32_t widths[]={8192,12288,15543,37500,75000,131073};
    float *x=malloc(MAX*4); uint32_t a[K],b[K],stats[15]={0},before[15]={0};
    need(x!=NULL,"CPU allocation"); need(ds4_gpu_init(),"init");
    ds4_gpu_tensor *scores=ds4_gpu_tensor_alloc(MAX*4),*out=ds4_gpu_tensor_alloc(K*4);
    need(scores&&out,"GPU allocation");
    unsigned accepts=0,rejects=0,skips=0;
    for(unsigned w=0;w<sizeof(widths)/sizeof(widths[0]);w++)
    for(unsigned draw=0;draw<120;draw++) {
        const uint32_t n=widths[w], kind=draw%12;
        for(uint32_t i=0;i<n;i++) x[i]=(float)(i+1);
        if(kind==1) for(uint32_t i=0;i<n;i++) x[i]=-x[i];
        if(kind==2) for(uint32_t i=0;i<n;i++) x[i]-=(float)n/2.0f;
        if(kind==3) x[n-2]=x[n-1]; // tie among winners
        if(kind==4) x[n-K-1]=x[n-K]; // tie at cut
        if(kind==5) x[0]=bits(0x7fc01234); // nonfinite outside winners
        if(kind==6) x[0]=bits(0x7f800000);
        if(kind==7) x[0]=bits(1); // subnormal outside winners
        if(kind==8) for(uint32_t i=0;i<n;i++) x[i]=bits(0x3f800000+i); // overflow
        if(kind==9) for(uint32_t i=0;i<n;i++) x[i]=bits(i&1?0x80000000:0);
        if(kind==10) for(uint32_t i=0;i<n;i++) x[i]=bits(0x00800000+(i<<6));
        if(kind==11) for(uint32_t i=0;i<n;i++) x[i]=bits(0xff7fffff-(i<<6));
        for(uint32_t i=n-1;i;i--){uint32_t j=rnd()%(i+1);float t=x[i];x[i]=x[j];x[j]=t;}
        need(ds4_gpu_tensor_write(scores,0,x,n*4),"write");
        unsetenv("DS4_GLM_ENABLE_TOPK_FAST");
        need(ds4_gpu_indexer_topk_tensor(out,scores,n,1,K),"reference");
        need(ds4_gpu_tensor_read(out,0,a,sizeof(a)),"read reference");
        memcpy(before,stats,sizeof(stats));
        setenv("DS4_GLM_ENABLE_TOPK_FAST","1",1);
        memset(b,0xa5,sizeof(b));need(ds4_gpu_tensor_write(out,0,b,sizeof(b)),"poison");
        need(ds4_gpu_glm53_indexer_topk_tensor(out,scores,n,1,K),"candidate");
        need(ds4_gpu_tensor_read(out,0,b,sizeof(b)),"read candidate");
        if(memcmp(a,b,sizeof(a))){unsigned i=0;while(i<K&&a[i]==b[i])i++;fprintf(stderr,"n=%u draw=%u kind=%u rank=%u ref=%u got=%u\n",n,draw,kind,i,a[i],b[i]);return 1;}
        ds4_gpu_test_glm53_topk_stats(stats,15);
        if(stats[4]==before[4]){skips++;need(n==8192||n==131073,"unexpected skip");}
        else {need(stats[4]==before[4]+1,"call count");if(stats[5]>before[5]){accepts++;need(kind==0||kind==1||kind==2||kind==10||kind==11,"unsafe accept");}else rejects++;}
    }
    need(accepts>80&&rejects>160&&skips==240,"non-vacuous acceptance/fallback/host gates");
    printf("GLM DSA top-k exact PASS: 720 poisoned draws, accepted=%u fallback=%u host-skipped=%u\n",accepts,rejects,skips);
    ds4_gpu_tensor_free(scores);ds4_gpu_tensor_free(out);ds4_gpu_cleanup();free(x);return 0;
}
