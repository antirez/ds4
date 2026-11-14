/* Four distinct synthetic BF16 matrices, all row counts 1..8, no checkpoint.
 * Largest complete fixture, including CPU/GPU activations, stays <48 MiB. */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include "k3_metal.h"
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
static uint32_t mix(uint32_t v) {
    v ^= v>>16; v *= UINT32_C(0x7feb352d);
    v ^= v>>15; v *= UINT32_C(0x846ca68b); return v^(v>>16);
}
static int guards_ok(const uint8_t *p,uint64_t n,uint64_t offset,uint64_t bytes) {
    for (uint64_t i=0;i<n;i++) if ((i<offset||i>=offset+bytes)&&p[i]!=0xa5) return 0;
    return 1;
}
static int shape(uint32_t in_dim,uint32_t out_dim) {
    enum { ROWS=8, INPUT_OFFSET=64, OUTPUT_OFFSET=128, GUARD=128 };
    const uint64_t input_row=(uint64_t)in_dim*4,output_row=(uint64_t)out_dim*4;
    const uint64_t input_size=INPUT_OFFSET+ROWS*input_row+GUARD;
    const uint64_t output_size=OUTPUT_OFFSET+ROWS*output_row+GUARD;
    const uint64_t weight_bytes=(uint64_t)in_dim*out_dim*2;
    const long page=sysconf(_SC_PAGESIZE);
    uint64_t mapping_size=0,offsets[4]={0},lengths[4]={weight_bytes,weight_bytes,weight_bytes,weight_bytes};
    uint8_t *mapping=MAP_FAILED,*inputs=NULL,*check=NULL,*actual=NULL,*expected=NULL;
    ds4_gpu_tensor *input_base=NULL,*input=NULL,*input_rows[ROWS]={0};
    ds4_gpu_tensor *output_base[4]={0},*output[4]={0},*reference_base[4]={0};
    ds4_gpu_tensor *reference_rows[4][ROWS]={{0}};
    int ok=0,retired=0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"fused BF16 %ux%u line%d: %s\n",in_dim,out_dim,__LINE__,#c); goto cleanup; } } while(0)
    CHECK(page>0);
    uint64_t cursor=(uint64_t)page;
    for (uint32_t m=0;m<4;m++) {
        offsets[m]=cursor+128u+32u*m;
        cursor=(offsets[m]+weight_bytes+GUARD+(uint64_t)page-1)/(uint64_t)page*(uint64_t)page;
    }
    mapping_size=cursor;
    CHECK(mapping_size+3*input_size+16*output_size<UINT64_C(48)*1024*1024);
    mapping=mmap(NULL,(size_t)mapping_size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    CHECK(mapping!=MAP_FAILED); memset(mapping,0xa5,(size_t)mapping_size);
    for (uint32_t m=0;m<4;m++) {
        uint16_t *w=(uint16_t *)(mapping+offsets[m]);
        for (uint64_t i=0;i<weight_bytes/2;i++) {
            uint32_t h=mix((uint32_t)i+UINT32_C(0x9e3779b9)*(m+1));
            w[i]=(uint16_t)((h&0x8000u)|((119u+((h>>8)%5u))<<7)|(h&127u));
        }
    }
    inputs=malloc((size_t)input_size); check=malloc((size_t)input_size);
    actual=malloc((size_t)(4*output_size)); expected=malloc((size_t)(4*output_size));
    CHECK(inputs&&check&&actual&&expected); memset(inputs,0xa5,(size_t)input_size);
    for (uint64_t i=0;i<ROWS*input_row/4;i++) {
        uint32_t h=mix((uint32_t)i+UINT32_C(0x13579bdf));
        uint32_t u=(h&UINT32_C(0x80000000))|((120u+((h>>24)%6u))<<23)|(h&UINT32_C(0x7fffff));
        memcpy(inputs+INPUT_OFFSET+4*i,&u,4);
    }
    input_base=ds4_gpu_tensor_alloc_managed(input_size); CHECK(input_base);
    input=ds4_gpu_tensor_view(input_base,INPUT_OFFSET,ROWS*input_row); CHECK(input);
    CHECK(ds4_gpu_tensor_write(input_base,0,inputs,input_size));
    for (uint32_t row=0;row<ROWS;row++) {
        input_rows[row]=ds4_gpu_tensor_view(input,(uint64_t)row*input_row,input_row); CHECK(input_rows[row]);
    }
    for (uint32_t m=0;m<4;m++) {
        output_base[m]=ds4_gpu_tensor_alloc_managed(output_size);
        reference_base[m]=ds4_gpu_tensor_alloc_managed(output_size);
        CHECK(output_base[m]&&reference_base[m]);
        output[m]=ds4_gpu_tensor_view(output_base[m],OUTPUT_OFFSET,ROWS*output_row); CHECK(output[m]);
        for (uint32_t row=0;row<ROWS;row++) {
            reference_rows[m][row]=ds4_gpu_tensor_view(reference_base[m],OUTPUT_OFFSET+(uint64_t)row*output_row,output_row);
            CHECK(reference_rows[m][row]);
        }
    }
    CHECK(ds4_gpu_set_model_map_spans(mapping,mapping_size,offsets,lengths,4,4*weight_bytes));
    for (uint32_t rows=1;rows<=ROWS;rows++) {
        memset(actual,0xa5,(size_t)(4*output_size));
        for (uint32_t m=0;m<4;m++) {
            CHECK(ds4_gpu_tensor_write(output_base[m],0,actual,output_size));
            CHECK(ds4_gpu_tensor_write(reference_base[m],0,actual,output_size));
        }
        CHECK(ds4_gpu_begin_commands());
        CHECK(k3_gpu_matmul4_bf16(output,mapping,mapping_size,offsets,in_dim,out_dim,input,rows));
        for (uint32_t m=0;m<4;m++) for (uint32_t row=0;row<rows;row++)
            CHECK(k3_gpu_matmul_bf16(reference_rows[m][row],mapping,mapping_size,offsets[m],in_dim,out_dim,input_rows[row],1));
        CHECK(ds4_gpu_end_commands());
        for (uint32_t m=0;m<4;m++) {
            uint8_t *a=actual+m*output_size,*b=expected+m*output_size;
            CHECK(ds4_gpu_tensor_read(output_base[m],0,a,output_size));
            CHECK(ds4_gpu_tensor_read(reference_base[m],0,b,output_size));
            CHECK(guards_ok(a,output_size,OUTPUT_OFFSET,rows*output_row));
            CHECK(guards_ok(b,output_size,OUTPUT_OFFSET,rows*output_row));
            CHECK(memcmp(a,b,(size_t)output_size)==0);
            for (uint64_t i=0;i<(uint64_t)rows*out_dim;i++) {
                uint32_t u; memcpy(&u,a+OUTPUT_OFFSET+4*i,4);
                CHECK((u&UINT32_C(0x7f800000))!=UINT32_C(0x7f800000));
            }
        }
    }
    /* Independent double-precision samples catch shared addressing mistakes. */
    const uint32_t columns[3]={0,out_dim/2,out_dim-1};
    for (uint32_t m=0;m<4;m++) for (uint32_t row=0;row<ROWS;row++) for (uint32_t j=0;j<3;j++) {
        const uint16_t *w=(const uint16_t *)(mapping+offsets[m]);
        uint32_t col=columns[j]; double sum=0,abs_sum=0;
        for (uint32_t k=0;k<in_dim;k++) {
            uint32_t u=(uint32_t)w[(uint64_t)col*in_dim+k]<<16; float weight,x;
            memcpy(&weight,&u,4); memcpy(&x,inputs+INPUT_OFFSET+((uint64_t)row*in_dim+k)*4,4);
            double product=(double)weight*x; sum+=product; abs_sum+=fabs(product);
        }
        float value; memcpy(&value,actual+m*output_size+OUTPUT_OFFSET+((uint64_t)row*out_dim+col)*4,4);
        CHECK(fabs((double)value-sum)<=4e-6*abs_sum+1e-7);
    }
    CHECK(ds4_gpu_tensor_read(input_base,0,check,input_size));
    CHECK(memcmp(check,inputs,(size_t)input_size)==0);
    for (uint32_t m=0;m<4;m++) {
        for (uint32_t i=0;i<GUARD;i++) {
            CHECK(mapping[offsets[m]-GUARD+i]==0xa5);
            CHECK(mapping[offsets[m]+weight_bytes+i]==0xa5);
        }
    }
    ok=1;
cleanup:
    if (ds4_gpu_commands_active()&&!ds4_gpu_end_commands()) ok=0;
    retired=ds4_gpu_synchronize()&&ds4_gpu_release_model_views();
    if (!retired) ok=0;
    if (retired) {
        for (uint32_t row=0;row<ROWS;row++) ds4_gpu_tensor_free(input_rows[row]);
        for (uint32_t m=0;m<4;m++) {
            for (uint32_t row=0;row<ROWS;row++) ds4_gpu_tensor_free(reference_rows[m][row]);
            ds4_gpu_tensor_free(output[m]); ds4_gpu_tensor_free(output_base[m]); ds4_gpu_tensor_free(reference_base[m]);
        }
        ds4_gpu_tensor_free(input); ds4_gpu_tensor_free(input_base);
        if (mapping!=MAP_FAILED&&munmap(mapping,(size_t)mapping_size)!=0) ok=0;
    }
    free(expected); free(actual); free(check); free(inputs); return ok;
#undef CHECK
}
int main(int argc,char **argv) {
    const bool resident=argc==2&&!strcmp(argv[1],"--resident");
    if (argc>2||(argc==2&&!resident)) return 2;
    if (setenv("K3_BF16_MV_UNROLL","16",1)||setenv("K3_BF16_MV_NSG","4",1)) return 1;
    ds4_gpu_set_ssd_streaming(!resident);
    if (!ds4_gpu_init()||!k3_gpu_matmul4_bf16_enabled()) return 1;
    int ok=shape(1024,4096)&&shape(1025,4097)&&shape(1025,1031)&&shape(513,13);
    ds4_gpu_cleanup();
    if (ok) puts("Fused BF16 four-matrix rows1..8 match independent projections exactly");
    return ok?0:1;
}
