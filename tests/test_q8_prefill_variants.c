#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void require(int ok) { if (!ok) { fprintf(stderr, "Q8 prefill test failed\n"); exit(1); } }
static uint32_t seed=123;
static void *maps[32];
static unsigned map_count;
static uint32_t rnd(void) { seed^=seed<<13; seed^=seed>>17; seed^=seed<<5; return seed; }
static void check(uint32_t d, uint32_t o, uint32_t t) {
    const uint64_t bytes=(uint64_t)d/32*34*o, n=(uint64_t)t*o;
    const uint64_t page=sysconf(_SC_PAGESIZE), stride=(bytes+page-1)/page*page;
    void *map=NULL;
    require(posix_memalign(&map,page,2*stride)==0);
    maps[map_count++]=map;
    memset(map,0,2*stride);
    for (uint32_t matrix=0; matrix<2; matrix++) for (uint64_t b=0; b<bytes; b+=34) {
        uint8_t *p=(uint8_t *)map+matrix*stride+b;
        uint16_t scale=0x1800+(rnd()%16)*0x100;
        memcpy(p,&scale,2);
        for (int j=2; j<34; j++) p[j]=rnd();
    }
    require(ds4_gpu_set_model_map(map,2*stride));
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc((uint64_t)t*d*4);
    ds4_gpu_tensor *a=ds4_gpu_tensor_alloc((n+16)*4), *b=ds4_gpu_tensor_alloc((n+16)*4);
    require(x && a && b);
    float *input=malloc((uint64_t)t*d*4), *actual=malloc((n+16)*4);
    float *refs[2]={malloc(n*4),malloc(n*4)};
    require(input && actual && refs[0] && refs[1]);
    for (uint64_t i=0; i<(uint64_t)t*d; i++) input[i]=((int)(rnd()%257)-128)/256.f;
    require(ds4_gpu_tensor_write(x,0,input,(uint64_t)t*d*4));
    const char *variants[]={"0","1",NULL,"0",NULL};
    for (int run=0; run<5; run++) {
        if (variants[run]) require(setenv("DS4_QWEN4_Q8_PREFILL_UNPACK",variants[run],1)==0);
        else require(unsetenv("DS4_QWEN4_Q8_PREFILL_UNPACK")==0);
        require(ds4_gpu_tensor_fill_f32(a,NAN,n+16) && ds4_gpu_tensor_fill_f32(b,NAN,n+16));
        require(ds4_gpu_begin_commands());
        require(ds4_gpu_qwen4_matmul_q8_0_tensor(a,map,2*stride,0,d,o,x,t));
        require(ds4_gpu_qwen4_matmul_q8_0_tensor(b,map,2*stride,stride,d,o,x,t));
        require(ds4_gpu_end_commands());
        ds4_gpu_tensor *outputs[]={a,b};
        for (int m=0; m<2; m++) {
            require(ds4_gpu_tensor_read(outputs[m],0,actual,(n+16)*4));
            for (uint64_t i=0; i<n; i++) require(isfinite(actual[i]));
            for (uint64_t i=n; i<n+16; i++) require(isnan(actual[i]));
            if (!run) memcpy(refs[m],actual,n*4);
            else if (memcmp(refs[m],actual,n*4)) {
                fprintf(stderr,"Mismatch D=%u O=%u T=%u variant=%s matrix=%d\n",d,o,t,variants[run] ? variants[run] : "default",m);
                exit(1);
            }
        }
    }
    printf("PASS Q8 D=%u O=%u T=%u exact paired outputs, guards intact\n",d,o,t);
    ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(x);
    free(input); free(actual); free(refs[0]); free(refs[1]);
}
int main(void) {
    require(ds4_gpu_init());
    const uint32_t shapes[][3]={{32,1,32},{64,48,33},{96,63,63},{128,65,64},
        {2560,48,1},{2560,48,2},{2560,128,8},{2560,128,16},{2560,128,17},
        {2560,128,31},{2560,128,32},{2560,10240,128},{6144,2560,65},
        {2560,32768,32}}; /* decoded weights exceed the 128 MiB cap */
    for (unsigned i=0; i<sizeof(shapes)/sizeof(*shapes); i++) check(shapes[i][0],shapes[i][1],shapes[i][2]);
    unsetenv("DS4_QWEN4_Q8_PREFILL_UNPACK");
    ds4_gpu_cleanup();
    for (unsigned i=0; i<map_count; i++) free(maps[i]);
    return 0;
}
