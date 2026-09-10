#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
bool ds4_log_is_tty(FILE *fp){(void)fp;return false;}
static void need(int ok,const char *s){if(!ok){fprintf(stderr,"Q8 inputs FAIL %s\n",s);exit(1);}}
static uint32_t seed=0x6719bc41;
static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int main(void){
    const unsigned widths[6]={8192,8192,8192,128,128,64};
    uint64_t offsets[6],size=0;
    for(unsigned i=0;i<6;i++){offsets[i]=size;size+=(uint64_t)widths[i]*(i<3?128*34:8192);}
    size=(size+16383u)&~16383ull;
    uint8_t *model=mmap(NULL,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    need(model!=MAP_FAILED,"model");
    for(unsigned s=0;s<6;s++){
        if(s<3){uint8_t *b=model+offsets[s];for(unsigned k=0;k<widths[s]*128u;k++,b+=34){
            uint16_t scale=(k%3?0x1800:0x9800);memcpy(b,&scale,2);
            for(unsigned j=0;j<32;j++)b[2+j]=(uint8_t)(rnd()%255-127);
        }}else{uint16_t *w=(uint16_t *)(model+offsets[s]);for(unsigned k=0;k<widths[s]*4096u;k++)w[k]=(uint16_t)((rnd()&0x807fu)|0x3c00u);}
    }
    need(ds4_gpu_init(),"init");need(ds4_gpu_set_model_map(model,size),"register");
    ds4_gpu_tensor *out[6],*x=ds4_gpu_tensor_alloc(4096*4);
    float *expected[6],*observed[6];
    for(unsigned i=0;i<6;i++){out[i]=ds4_gpu_tensor_alloc(widths[i]*4);expected[i]=malloc(widths[i]*4);observed[i]=malloc(widths[i]*4);need(out[i]&&expected[i]&&observed[i],"outputs");}
    need(x!=NULL,"input");setenv("DS4_GLM_ENABLE_KDA_Q8_INPUTS","1",1);
    for(unsigned draw=0;draw<120;draw++){
        float input[4096];for(unsigned k=0;k<4096;k++)input[k]=draw%10?((int)(rnd()%4095)-2047)/1024.0f:0;
        need(ds4_gpu_tensor_write(x,0,input,sizeof(input)),"input write");
        for(unsigned i=0;i<6;i++){
            int ok=i<3?ds4_gpu_matmul_q8_0_tensor(out[i],model,size,offsets[i],4096,widths[i],x,1):ds4_gpu_glm53_matmul_bf16(out[i],model,size,offsets[i],4096,widths[i],x,1);
            need(ok,"reference");need(ds4_gpu_tensor_read(out[i],0,expected[i],widths[i]*4),"reference read");
            memset(observed[i],0xa5,widths[i]*4);need(ds4_gpu_tensor_write(out[i],0,observed[i],widths[i]*4),"poison");
        }
        need(ds4_gpu_glm53_kda_inputs_q8_bf16(out,offsets,model,size,x),"fused");
        for(unsigned i=0;i<6;i++){
            need(ds4_gpu_tensor_read(out[i],0,observed[i],widths[i]*4),"fused read");
            if(memcmp(expected[i],observed[i],widths[i]*4)){unsigned k=0;while(k<widths[i]&&!memcmp(expected[i]+k,observed[i]+k,4))k++;fprintf(stderr,"draw=%u slot=%u row=%u ref=%a observed=%a\n",draw,i,k,expected[i][k],observed[i][k]);return 1;}
        }
    }
    uint64_t saved=offsets[5];offsets[5]=size-1;
    need(!ds4_gpu_glm53_kda_inputs_q8_bf16(out,offsets,model,size,x),"invalid weight refusal");offsets[5]=saved;
    unsetenv("DS4_GLM_ENABLE_KDA_Q8_INPUTS");
    need(!ds4_gpu_glm53_kda_inputs_q8_bf16(out,offsets,model,size,x),"disabled refusal");
    for(unsigned i=0;i<6;i++){ds4_gpu_tensor_free(out[i]);free(expected[i]);free(observed[i]);}
    ds4_gpu_tensor_free(x);ds4_gpu_cleanup();munmap(model,size);
    puts("GLM mixed Q8/BF16 inputs exact PASS: 120 poisoned draws, six complete projections, refusal gates");return 0;
}
