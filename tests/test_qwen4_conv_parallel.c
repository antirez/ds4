#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static void require(int ok) { if (!ok) { fprintf(stderr,"conv test failed\n"); exit(1); } }
static uint32_t seed=123;
static float rnd(void) { seed ^= seed<<13; seed ^= seed>>17; seed ^= seed<<5; return ((int)(seed%513)-256)/256.f; }
static void check(void *map, uint64_t bytes, unsigned T, unsigned C, unsigned K, int silu) {
    uint64_t n=(uint64_t)T*C, h=(uint64_t)(K-1)*C;
    float *inputs[3]={malloc((n+16)*4),malloc((n+16)*4),malloc((h+16)*4)};
    float *refs[3]={malloc((n+16)*4),malloc((n+16)*4),malloc((h+16)*4)};
    float *actual=malloc((n+h+16)*4);
    ds4_gpu_tensor *buf[3]={ds4_gpu_tensor_alloc((n+16)*4),ds4_gpu_tensor_alloc((n+16)*4),ds4_gpu_tensor_alloc((h+16)*4)};
    for (unsigned b=0;b<3;b++) {
        require(inputs[b] && refs[b] && buf[b]);
        uint64_t size=b==2?h:n;
        for(uint64_t i=0;i<size+16;i++) inputs[b][i]=i<size?rnd():NAN;
    }
    for (int run=0;run<5;run++) {
        if (run == 4) require(unsetenv("DS4_QWEN4_PREFILL_REUSE")==0);
        else require(setenv("DS4_QWEN4_PREFILL_REUSE",run%2?"1":"0",1)==0);
        for(unsigned b=0;b<3;b++) require(ds4_gpu_tensor_write(buf[b],0,inputs[b],((b==2?h:n)+16)*4));
        require(ds4_gpu_begin_commands());
        for(unsigned b=0;b<2;b++) require(ds4_gpu_qwen4_conv_stream_tensor(buf[b],buf[2],map,bytes,0,T,C,K,silu));
        require(ds4_gpu_end_commands());
        for(unsigned b=0;b<3;b++) {
            uint64_t size=b==2?h:n;
            require(ds4_gpu_tensor_read(buf[b],0,actual,(size+16)*4));
            for(uint64_t i=0;i<size;i++) require(isfinite(actual[i]));
            for(uint64_t i=size;i<size+16;i++) require(isnan(actual[i]));
            if(!run) memcpy(refs[b],actual,(size+16)*4);
            else if(memcmp(refs[b],actual,(size+16)*4)) { fprintf(stderr,"mismatch T=%u C=%u K=%u silu=%d run=%d b=%u\n",T,C,K,silu,run,b); exit(1); }
        }
    }
    printf("PASS conv T=%u C=%u K=%u silu=%d paired append, state and guards exact\n",T,C,K,silu);
    for(unsigned b=0;b<3;b++) {free(inputs[b]);free(refs[b]);ds4_gpu_tensor_free(buf[b]);} free(actual);
}
int main(void) {
    require(ds4_gpu_init());
    uint64_t page=sysconf(_SC_PAGESIZE), bytes=(10240*4*4+page-1)/page*page;
    void *map=NULL; require(posix_memalign(&map,page,bytes)==0);
    for(uint64_t i=0;i<bytes/4;i++) ((float*)map)[i]=rnd()/8.f;
    require(ds4_gpu_set_model_map(map,bytes));
    const unsigned shapes[][2]={{1,31},{8,256},{9,1},{9,257},{31,63},{32,256},{33,257},{63,257},{64,257},{65,10240},{127,31},{128,31},{129,31},{1023,257},{1024,10240},{1025,257},{8191,31},{8192,10240},{8193,31}};
    for(unsigned i=0;i<sizeof(shapes)/sizeof(*shapes);i++) for(unsigned k=2;k<=4;k++) for(int silu=0;silu<2;silu++) check(map,bytes,shapes[i][0],shapes[i][1],k,silu);
    ds4_gpu_cleanup(); free(map); return 0;
}
