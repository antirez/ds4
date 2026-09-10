#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static void require(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "router/shared FAIL: %s\n", what); exit(1); }
}
static uint32_t state = 0x514aac39u;
static uint32_t random_bits(void) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
static float sample(void) { return ((int)(random_bits() & 2047u)-1024) / 1024.0f; }

int main(void) {
    enum { WIDTH=4096, EXPERTS=288, MID=2048, USED=8,
           ROUTER=0, BIAS=WIDTH*EXPERTS*4,
           GATE=BIAS+16384, UP=GATE+128*34*MID,
           BYTES=UP+128*34*MID+16384 };
    uint8_t *model=mmap(NULL,BYTES,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    require(model!=MAP_FAILED,"model allocation");
    for (unsigned i=0;i<WIDTH*EXPERTS;i++) ((float *)model)[i]=sample()/32.0f;
    for (int arm=0;arm<2;arm++) {
        uint8_t *p=model+(arm?UP:GATE);
        for (unsigned b=0;b<128u*MID;b++,p+=34) {
            const uint16_t scale=(b%7==0 ? 0xa000u : 0x2000u);
            memcpy(p,&scale,2);
            for (unsigned q=0;q<32;q++) p[2+q]=(uint8_t)(random_bits()%255u-127u);
        }
    }
    require(ds4_gpu_init(),"GPU initialization");
    require(ds4_gpu_set_model_map(model,BYTES),"model registration");
    const uint64_t sizes[]={EXPERTS*4,USED*4,USED*4,EXPERTS*4,4,MID*4};
    ds4_gpu_tensor *out[6];
    uint8_t *expected[6], *observed[6];
    for (int i=0;i<6;i++) {
        out[i]=ds4_gpu_tensor_alloc(sizes[i]);
        expected[i]=malloc(sizes[i]); observed[i]=malloc(sizes[i]);
        require(out[i] && expected[i] && observed[i],"output allocation");
    }
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(WIDTH*4);
    require(x!=NULL,"input allocation");
    require(ds4_gpu_tensor_fill_f32(out[4],0.0f,1u),"counter reset");
    for (unsigned draw=0;draw<240;draw++) {
        float input[WIDTH];
        for (unsigned i=0;i<WIDTH;i++) input[i]=(draw%12==0)?0.0f:sample();
        for (unsigned i=0;i<EXPERTS;i++) ((float *)(model+BIAS))[i]=(draw%12==0)?0.0f:sample()/4.0f;
        if (draw%12==1) {
            uint32_t nan=0x7fc00129; memcpy(model+BIAS+4*179,&nan,4);
        }
        if (draw%12==2) {
            uint32_t inf=0x7f800000; memcpy(model+BIAS+4*219,&inf,4);
        }
        require(ds4_gpu_tensor_write(x,0,input,sizeof(input)),"input write");
        require(ds4_gpu_matmul_f32_tensor(out[0],model,BYTES,ROUTER,WIDTH,EXPERTS,x,1),"reference projection");
        require(ds4_gpu_glm_router_select_tensor(out[1],out[2],out[3],model,BYTES,BIAS,out[0],EXPERTS,USED,1.0f),"reference selection");
        require(ds4_gpu_shared_mid_swiglu_q8_0_tensor(out[5],model,BYTES,GATE,UP,WIDTH,MID,x,10.0f),"reference shared mid");
        for (int i=0;i<6;i++) {
            require(ds4_gpu_tensor_read(out[i],0,expected[i],sizes[i]),"reference read");
            if (i!=4) {
                memset(observed[i],0xa5,sizes[i]);
                require(ds4_gpu_tensor_write(out[i],0,observed[i],sizes[i]),"output poison");
            }
        }
        require(ds4_gpu_glm53_router_shared_exact(out[0],out[1],out[2],out[3],out[4],out[5],
            model,BYTES,ROUTER,BIAS,GATE,UP,x,1.0f,10.0f)==1,"fused dispatch");
        for (int i=0;i<6;i++) {
            require(ds4_gpu_tensor_read(out[i],0,observed[i],sizes[i]),"fused read");
            if (memcmp(expected[i],observed[i],sizes[i])) {
                unsigned bad=0;while(bad<sizes[i]/4 && ((uint32_t *)expected[i])[bad]==((uint32_t *)observed[i])[bad])bad++;
                fprintf(stderr,"draw=%u output=%d word=%u expected=%08x got=%08x\n",draw,i,bad,((uint32_t *)expected[i])[bad],((uint32_t *)observed[i])[bad]);
                return 1;
            }
        }
    }
    const char *disabled[] = {"DS4_METAL_DISABLE_GLM53_FLASH_TUNING",
        "DS4_METAL_DISABLE_M3_ULTRA_GLM53_DECODE", "DS4_METAL_DISABLE_GLM53_ROUTER_TOP8"};
    for (unsigned i=0;i<sizeof(disabled)/sizeof(disabled[0]);i++) {
        setenv(disabled[i], "1", 1);
        require(ds4_gpu_glm53_router_shared_exact(out[0],out[1],out[2],out[3],out[4],out[5],
            model,BYTES,ROUTER,BIAS,GATE,UP,x,1.0f,10.0f)==0,"aggregate rollback refusal");
        unsetenv(disabled[i]);
    }
    const uint32_t nonfinite_bits[]={0x7fc00123u,0x7f800000u,0xff800000u};
    for (unsigned i=0;i<sizeof(nonfinite_bits)/sizeof(nonfinite_bits[0]);i++) {
        float nonfinite;
        memcpy(&nonfinite,&nonfinite_bits[i],sizeof(nonfinite));
        require(ds4_gpu_glm53_router_shared_exact(out[0],out[1],out[2],out[3],out[4],out[5],
            model,BYTES,ROUTER,BIAS,GATE,UP,x,nonfinite,10.0f)==0,"nonfinite scale refusal");
        require(ds4_gpu_glm53_router_shared_exact(out[0],out[1],out[2],out[3],out[4],out[5],
            model,BYTES,ROUTER,BIAS,GATE,UP,x,1.0f,nonfinite)==0,"nonfinite clamp refusal");
    }
    require(ds4_gpu_glm53_router_shared_exact(out[0],out[1],out[2],out[3],out[4],out[5],
        model,BYTES,ROUTER,BIAS,GATE,BYTES-1,x,1.0f,10.0f)==0,"late model range refusal");
    ds4_gpu_set_ssd_streaming(true);
    require(ds4_gpu_glm53_router_shared_exact(out[0],out[1],out[2],out[3],out[4],out[5],
        model,BYTES,ROUTER,BIAS,GATE,UP,x,1.0f,10.0f)==0,"SSD refusal");
    ds4_gpu_set_ssd_streaming(false);
    require(setenv("DS4_METAL_DISABLE_GLM53_ROUTER_SHARED","1",1)==0,"set rollback");
    require(ds4_gpu_glm53_router_shared_exact(out[0],out[1],out[2],out[3],out[4],out[5],
        model,BYTES,ROUTER,BIAS,GATE,UP,x,1.0f,10.0f)==0,"rollback refusal");
    for (int i=0;i<6;i++) { ds4_gpu_tensor_free(out[i]);free(expected[i]);free(observed[i]); }
    ds4_gpu_tensor_free(x);ds4_gpu_cleanup();munmap(model,BYTES);
    puts("GLM router/shared exact: PASS (240 poisoned draws, ties, NaN/Inf bias, counter reuse)");
    return 0;
}
