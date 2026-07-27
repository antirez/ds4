/* IQ2_XXS gate/up N-tile microbench -- forked from tests/d2r_ntile_bench.cu for the
 * format that is now the #1 prefill bucket.  Sweeps token tile NT in {8,10,12,16},
 * reports REAL occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor), times each
 * under BALANCED and IMBALANCED routing (the balanced harness overstates wide tiles),
 * and checks bit-exactness vs NT=8.
 *
 * The kernel body is copied verbatim from moe_gate_up_mid_expert_ntile_rowspan_kernel
 * in src/cuda/pulsar_cuda_moe.cu (dev_dot_iq2_xxs_q8_K_blockN + the ntile kernel).
 *
 *   nvcc -O3 --use_fast_math -arch=sm_120f -Isrc -o /tmp/iq2_ntile_bench iq2_ntile_bench.cu
 *   flock -w 3600 temp/gpu.lock /tmp/iq2_ntile_bench 2048 20
 *
 * v5mx shape: n_embd=4096 (xq_blocks=16), n_ff_exp=2048 (mid_dim), 192 experts, 6 used.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_QK_K 256
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)

typedef struct { uint16_t d; uint16_t qs[CUDA_QK_K/8]; } cuda_block_iq2_xxs;
typedef struct { float d; int8_t qs[CUDA_QK_K]; int16_t bsums[CUDA_QK_K/16]; } cuda_block_q8_K;

#include "cuda/pulsar_iq2_tables_cuda.inc"   /* cuda_iq2xxs_grid[256], cuda_ksigns_iq2xs[128] */

__device__ static float dev_f16_to_f32(uint16_t v){ return __half2float(*reinterpret_cast<const __half*>(&v)); }
__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v){ const uint32_t p=__popc(v)&1u; const uint32_t s=v^(p<<7u); return s*0x01010101u; }
__device__ __forceinline__ static void dev_iq2_i8x8_lut(const uint64_t*grid,const uint8_t*signs,uint8_t gi,uint32_t si,int32_t*w0,int32_t*w1){
    const uint32_t s=dev_unpack_iq2_signs(signs[si]);
    const int32_t sm0=__vcmpne4(s&0x08040201u,0); const int32_t sm1=__vcmpne4(s&0x80402010u,0);
    const uint64_t g=grid[gi]; *w0=__vsub4((int32_t)(uint32_t)g^sm0,sm0); *w1=__vsub4((int32_t)(uint32_t)(g>>32)^sm1,sm1);
}
__device__ static float quarter_warp_sum_f32(float v, uint32_t){ uint32_t mask=0xffu<<(threadIdx.x&24u); for(int o=4;o>0;o>>=1) v+=__shfl_down_sync(mask,v,o,8); return v; }

/* ---- verbatim from production ---- */
template <uint32_t NT>
__device__ static void dev_dot_iq2_xxs_q8_K_blockN(const cuda_block_iq2_xxs*x,const cuda_block_q8_K*const*ys,uint32_t n,float acc[NT],const uint64_t*grid,const uint8_t*signs){
    const float xd=dev_f16_to_f32(x->d); const uint16_t*q2=x->qs; int32_t bsum[NT];
    #pragma unroll
    for(uint32_t p=0;p<NT;p++) bsum[p]=0;
    for(int ib32=0; ib32<CUDA_QK_K/32; ib32++){
        const uint32_t aux0=(uint32_t)q2[0]|((uint32_t)q2[1]<<16); const uint32_t aux1=(uint32_t)q2[2]|((uint32_t)q2[3]<<16); q2+=4;
        const int32_t ls=(int32_t)(2u*(aux1>>28)+1u); int32_t w[8];
        dev_iq2_i8x8_lut(grid,signs,(uint8_t)(aux0&0xffu),(aux1>>0)&127u,&w[0],&w[1]);
        dev_iq2_i8x8_lut(grid,signs,(uint8_t)((aux0>>8)&0xffu),(aux1>>7)&127u,&w[2],&w[3]);
        dev_iq2_i8x8_lut(grid,signs,(uint8_t)((aux0>>16)&0xffu),(aux1>>14)&127u,&w[4],&w[5]);
        dev_iq2_i8x8_lut(grid,signs,(uint8_t)((aux0>>24)&0xffu),(aux1>>21)&127u,&w[6],&w[7]);
        #pragma unroll
        for(uint32_t p=0;p<NT;p++){ if(p>=n)break; const int8_t*q=ys[p]->qs+ib32*32; int32_t sumi=0;
            sumi=__dp4a(w[0],*(const int32_t*)(q+0),sumi); sumi=__dp4a(w[1],*(const int32_t*)(q+4),sumi);
            sumi=__dp4a(w[2],*(const int32_t*)(q+8),sumi); sumi=__dp4a(w[3],*(const int32_t*)(q+12),sumi);
            sumi=__dp4a(w[4],*(const int32_t*)(q+16),sumi); sumi=__dp4a(w[5],*(const int32_t*)(q+20),sumi);
            sumi=__dp4a(w[6],*(const int32_t*)(q+24),sumi); sumi=__dp4a(w[7],*(const int32_t*)(q+28),sumi);
            bsum[p]+=sumi*ls; }
    }
    #pragma unroll
    for(uint32_t p=0;p<NT;p++){ if(p>=n)break; acc[p]+=0.125f*xd*ys[p]->d*(float)bsum[p]; }
}

__host__ __device__ __forceinline__ static uint32_t iq2_smem_bytes(uint32_t nt,uint32_t xqb){ return (xqb<=16u)?(uint32_t)(nt*xqb*(uint32_t)sizeof(cuda_block_q8_K)):0u; }

template <uint32_t ROW_SPAN, uint32_t NT>
__global__ static void iq2_ntile_kernel(float*mid_out,const char*gate_base,const char*up_base,const cuda_block_q8_K*xq,
        const uint32_t*sorted_pairs,const uint32_t*offsets,const uint32_t*counts,const uint32_t*tile_total,
        const uint32_t*tile_experts,const uint32_t*tile_starts,const float*weights,uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,uint32_t xq_blocks,uint32_t expert_mid_dim,uint32_t n_expert,float clamp){
    uint32_t tile=blockIdx.y; if(tile>=*tile_total)return;
    uint32_t lane=threadIdx.x&7u; uint32_t row_lane=threadIdx.x>>3u;
    uint32_t expert=tile_experts[tile]; uint32_t local_start=tile_starts[tile];
    extern __shared__ cuda_block_q8_K sxq_iq2_dyn[];
    __shared__ uint64_t s_iq2_grid[256]; __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[NT],tok[NT],slot[NT]; const cuda_block_q8_K*xqb[NT];
    #pragma unroll
    for(uint32_t i=0;i<NT;i++){pair[i]=0;tok[i]=0;slot[i]=0;xqb[i]=NULL;}
    const uint32_t cnt=counts[expert]; const uint32_t avail=(local_start<cnt)?(cnt-local_start):0u; const uint32_t np=avail<NT?avail:NT;
    #pragma unroll
    for(uint32_t i=0;i<NT;i++){ if(i>=np)break; pair[i]=sorted_pairs[offsets[expert]+local_start+i]; tok[i]=pair[i]/n_expert; slot[i]=pair[i]-tok[i]*n_expert; xqb[i]=xq+(uint64_t)tok[i]*xq_blocks; }
    const bool stage=iq2_smem_bytes(NT,xq_blocks)!=0u;
    if(stage){ for(uint32_t i=threadIdx.x;i<np*xq_blocks;i+=blockDim.x){ uint32_t p=i/xq_blocks; uint32_t b=i-p*xq_blocks; sxq_iq2_dyn[p*xq_blocks+b]=xqb[p][b]; } }
    for(uint32_t i=threadIdx.x;i<256u;i+=blockDim.x) s_iq2_grid[i]=cuda_iq2xxs_grid[i];
    for(uint32_t i=threadIdx.x;i<128u;i+=blockDim.x) s_iq2_signs[i]=cuda_ksigns_iq2xs[i];
    __syncthreads();
    if(stage){
        #pragma unroll
        for(uint32_t p=0;p<NT;p++){ if(p>=np)break; xqb[p]=sxq_iq2_dyn+p*xq_blocks; }
    }
    for(uint32_t rr=0;rr<ROW_SPAN/32u;rr++){
        uint32_t row=blockIdx.x*ROW_SPAN+row_lane+rr*32u; if(row>=expert_mid_dim)continue;
        const cuda_block_iq2_xxs*gr=(const cuda_block_iq2_xxs*)(gate_base+(uint64_t)expert*gate_expert_bytes+(uint64_t)row*gate_row_bytes);
        const cuda_block_iq2_xxs*ur=(const cuda_block_iq2_xxs*)(up_base+(uint64_t)expert*gate_expert_bytes+(uint64_t)row*gate_row_bytes);
        float gate[NT],up[NT];
        #pragma unroll
        for(uint32_t p=0;p<NT;p++){gate[p]=0.0f;up[p]=0.0f;}
        for(uint32_t b=lane;b<xq_blocks;b+=8u){
            const cuda_block_q8_K*yb[NT];
            #pragma unroll
            for(uint32_t p=0;p<NT;p++){ yb[p]=(p<np)?xqb[p]+b:NULL; }
            dev_dot_iq2_xxs_q8_K_blockN<NT>(gr+b,yb,np,gate,s_iq2_grid,s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_blockN<NT>(ur+b,yb,np,up,s_iq2_grid,s_iq2_signs);
        }
        #pragma unroll
        for(uint32_t p=0;p<NT;p++){ if(p>=np)break;
            gate[p]=quarter_warp_sum_f32(gate[p],lane); up[p]=quarter_warp_sum_f32(up[p],lane);
            if(lane==0){ if(clamp>1.0e-6f){ if(gate[p]>clamp)gate[p]=clamp; if(up[p]>clamp)up[p]=clamp; if(up[p]<-clamp)up[p]=-clamp; }
                const uint64_t off=(uint64_t)pair[p]*expert_mid_dim+row;
                mid_out[off]=(gate[p]/(1.0f+expf(-gate[p])))*up[p]*weights[(uint64_t)tok[p]*n_expert+slot[p]]; }
        }
    }
}

static const uint32_t N_EXPERT_TOTAL=192, N_EXPERT_USED=6, IN_DIM=4096, MID_DIM=2048;

struct Routing { uint32_t n_tokens,pair_count; uint32_t*d_sorted_pairs,*d_offsets,*d_counts; };

static uint32_t rng=98765u; static uint32_t xrand(){ rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }

/* mode 0 = balanced (round-robin), mode 1 = imbalanced (random top-6 per token). */
static void build_routing(Routing*r,uint32_t n_tokens,int mode){
    r->n_tokens=n_tokens; r->pair_count=n_tokens*N_EXPERT_USED;
    uint32_t*h_pairs=(uint32_t*)malloc(r->pair_count*sizeof(uint32_t));
    uint32_t*h_counts=(uint32_t*)calloc(N_EXPERT_TOTAL,sizeof(uint32_t));
    uint32_t*h_off=(uint32_t*)calloc(N_EXPERT_TOTAL+1,sizeof(uint32_t));
    uint32_t*exp_of=(uint32_t*)malloc(r->pair_count*sizeof(uint32_t));
    for(uint32_t t=0;t<n_tokens;t++){
        if(mode==0){ for(uint32_t s=0;s<N_EXPERT_USED;s++){ uint32_t p=t*N_EXPERT_USED+s; uint32_t e=(t*N_EXPERT_USED+s)%N_EXPERT_TOTAL; exp_of[p]=e; h_counts[e]++; } }
        else { /* pick 6 distinct experts with a skewed distribution (square of uniform -> favors low ids) */
            uint32_t chosen[N_EXPERT_USED]; uint32_t nc=0;
            while(nc<N_EXPERT_USED){ double u=(double)(xrand()%100000)/100000.0; uint32_t e=(uint32_t)(u*u*N_EXPERT_TOTAL); if(e>=N_EXPERT_TOTAL)e=N_EXPERT_TOTAL-1; int dup=0; for(uint32_t k=0;k<nc;k++) if(chosen[k]==e)dup=1; if(!dup)chosen[nc++]=e; }
            for(uint32_t s=0;s<N_EXPERT_USED;s++){ uint32_t p=t*N_EXPERT_USED+s; exp_of[p]=chosen[s]; h_counts[chosen[s]]++; }
        }
    }
    for(uint32_t e=0;e<N_EXPERT_TOTAL;e++) h_off[e+1]=h_off[e]+h_counts[e];
    uint32_t*cur=(uint32_t*)calloc(N_EXPERT_TOTAL,sizeof(uint32_t));
    for(uint32_t t=0;t<n_tokens;t++) for(uint32_t s=0;s<N_EXPERT_USED;s++){ uint32_t p=t*N_EXPERT_USED+s; uint32_t e=exp_of[p]; h_pairs[h_off[e]+cur[e]++]=t*N_EXPERT_USED+s; }
    CHECK(cudaMalloc(&r->d_sorted_pairs,r->pair_count*sizeof(uint32_t)));
    CHECK(cudaMalloc(&r->d_offsets,(N_EXPERT_TOTAL+1)*sizeof(uint32_t)));
    CHECK(cudaMalloc(&r->d_counts,N_EXPERT_TOTAL*sizeof(uint32_t)));
    CHECK(cudaMemcpy(r->d_sorted_pairs,h_pairs,r->pair_count*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r->d_offsets,h_off,(N_EXPERT_TOTAL+1)*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r->d_counts,h_counts,N_EXPERT_TOTAL*sizeof(uint32_t),cudaMemcpyHostToDevice));
    /* report imbalance */
    uint32_t mx=0,nz=0; for(uint32_t e=0;e<N_EXPERT_TOTAL;e++){ if(h_counts[e]>mx)mx=h_counts[e]; if(h_counts[e])nz++; }
    printf("# routing mode=%d: %u active experts, max_count=%u, avg=%.1f\n",mode,nz,mx,(double)r->pair_count/nz);
    free(h_pairs);free(h_counts);free(h_off);free(cur);free(exp_of);
}

struct Tiles{ uint32_t*d_total,*d_experts,*d_starts; uint32_t n; };
static void build_tiles(Tiles*t,const Routing*r,uint32_t tw){
    uint32_t*hc=(uint32_t*)malloc(N_EXPERT_TOTAL*sizeof(uint32_t));
    CHECK(cudaMemcpy(hc,r->d_counts,N_EXPERT_TOTAL*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    uint32_t cap=0; for(uint32_t e=0;e<N_EXPERT_TOTAL;e++) cap+=(hc[e]+tw-1)/tw; if(cap==0)cap=1;
    uint32_t*he=(uint32_t*)malloc(cap*sizeof(uint32_t)); uint32_t*hs=(uint32_t*)malloc(cap*sizeof(uint32_t)); uint32_t n=0;
    for(uint32_t e=0;e<N_EXPERT_TOTAL;e++) for(uint32_t s=0;s<hc[e];s+=tw){ he[n]=e;hs[n]=s;n++; }
    t->n=n;
    CHECK(cudaMalloc(&t->d_total,sizeof(uint32_t))); CHECK(cudaMalloc(&t->d_experts,(n?n:1)*sizeof(uint32_t))); CHECK(cudaMalloc(&t->d_starts,(n?n:1)*sizeof(uint32_t)));
    CHECK(cudaMemcpy(t->d_total,&n,sizeof(uint32_t),cudaMemcpyHostToDevice));
    if(n){ CHECK(cudaMemcpy(t->d_experts,he,n*sizeof(uint32_t),cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(t->d_starts,hs,n*sizeof(uint32_t),cudaMemcpyHostToDevice)); }
    free(hc);free(he);free(hs);
}
static void free_tiles(Tiles*t){ cudaFree(t->d_total);cudaFree(t->d_experts);cudaFree(t->d_starts); }

template<typename K>
static int report_kernel(const char*name,K kern,uint32_t threads,size_t dyn){
    cudaFuncAttributes a; CHECK(cudaFuncGetAttributes(&a,(const void*)kern));
    int blocks=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,(const void*)kern,threads,dyn);
    if(e!=cudaSuccess){ (void)cudaGetLastError(); }
    int dev;CHECK(cudaGetDevice(&dev)); int mts;CHECK(cudaDeviceGetAttribute(&mts,cudaDevAttrMaxThreadsPerMultiProcessor,dev));
    printf("    %-16s regs=%-4d static_smem=%-6zu dyn_smem=%-7zu spill=%zu  blocks/SM=%d  occ=%.1f%%\n",
           name,a.numRegs,(size_t)a.sharedSizeBytes,dyn,(size_t)a.localSizeBytes,blocks,100.0*blocks*threads/mts);
    return blocks;
}

int main(int argc,char**argv){
    uint32_t n_tokens=argc>1?(uint32_t)atoi(argv[1]):2048;
    int iters=argc>2?atoi(argv[2]):20;
    const uint32_t xq_blocks=IN_DIM/CUDA_QK_K;                 /* 16 */
    const uint64_t gate_row_bytes=(uint64_t)IN_DIM/CUDA_QK_K*sizeof(cuda_block_iq2_xxs);
    const uint64_t gate_expert_bytes=gate_row_bytes*MID_DIM;
    const uint64_t gate_bytes=gate_expert_bytes*N_EXPERT_TOTAL;
    printf("# IQ2 gate/up N-tile bench: tokens=%u iters=%d xq_blocks=%u\n",n_tokens,iters,xq_blocks);
    printf("# gate_row=%lu B expert=%lu B total gate=%.2f GB (x2 gate+up)\n",gate_row_bytes,gate_expert_bytes,gate_bytes/1e9);

    char*d_gate,*d_up; CHECK(cudaMalloc(&d_gate,gate_bytes)); CHECK(cudaMalloc(&d_up,gate_bytes));
    /* fill weights with pseudo-random bytes */
    { size_t chunk=1<<26; uint8_t*hb=(uint8_t*)malloc(chunk); for(size_t off=0;off<gate_bytes;off+=chunk){ size_t nn=(gate_bytes-off<chunk)?gate_bytes-off:chunk; for(size_t i=0;i<nn;i++)hb[i]=(uint8_t)xrand(); CHECK(cudaMemcpy(d_gate+off,hb,nn,cudaMemcpyHostToDevice)); for(size_t i=0;i<nn;i++)hb[i]=(uint8_t)xrand(); CHECK(cudaMemcpy(d_up+off,hb,nn,cudaMemcpyHostToDevice)); } free(hb); }

    /* activations: n_tokens x xq_blocks q8_K blocks */
    size_t xq_count=(size_t)n_tokens*xq_blocks; cuda_block_q8_K*d_xq; CHECK(cudaMalloc(&d_xq,xq_count*sizeof(cuda_block_q8_K)));
    { cuda_block_q8_K*hx=(cuda_block_q8_K*)malloc(xq_count*sizeof(cuda_block_q8_K)); for(size_t i=0;i<xq_count;i++){ hx[i].d=0.02f+0.001f*(i%7); for(int j=0;j<CUDA_QK_K;j++)hx[i].qs[j]=(int8_t)((int)(xrand()&0x3f)-32); for(int j=0;j<CUDA_QK_K/16;j++)hx[i].bsums[j]=0; }
      CHECK(cudaMemcpy(d_xq,hx,xq_count*sizeof(cuda_block_q8_K),cudaMemcpyHostToDevice)); free(hx); }

    float*d_weights,*d_mid,*d_ref; size_t mid_elems=(size_t)n_tokens*N_EXPERT_USED*MID_DIM;
    CHECK(cudaMalloc(&d_weights,(size_t)n_tokens*N_EXPERT_USED*sizeof(float)));
    CHECK(cudaMalloc(&d_mid,mid_elems*sizeof(float))); CHECK(cudaMalloc(&d_ref,mid_elems*sizeof(float)));
    { float*hw=(float*)malloc((size_t)n_tokens*N_EXPERT_USED*sizeof(float)); for(size_t i=0;i<(size_t)n_tokens*N_EXPERT_USED;i++)hw[i]=0.5f+0.5f*((xrand()&255)/255.0f); CHECK(cudaMemcpy(d_weights,hw,(size_t)n_tokens*N_EXPERT_USED*sizeof(float),cudaMemcpyHostToDevice)); free(hw); }

    const float clamp=0.0f;
    cudaEvent_t e0,e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));

    /* opt-in for the wide tiles */
    #define OPTIN(NT) do{ size_t need=iq2_smem_bytes((NT),xq_blocks); cudaError_t a_=cudaFuncSetAttribute((const void*)iq2_ntile_kernel<1024u,(NT)>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)need); (void)cudaGetLastError(); if(a_!=cudaSuccess&&need>49152-2176)printf("  [optin NT=%d refused: %s]\n",(NT),cudaGetErrorString(a_)); }while(0)
    OPTIN(8); OPTIN(10); OPTIN(12); OPTIN(16);

    for(int mode=0;mode<2;mode++){
        printf("\n===== ROUTING %s =====\n", mode==0?"BALANCED":"IMBALANCED");
        Routing rt; build_routing(&rt,n_tokens,mode);
        Tiles t8,t10,t12,t16; build_tiles(&t8,&rt,8); build_tiles(&t10,&rt,10); build_tiles(&t12,&rt,12); build_tiles(&t16,&rt,16);
        printf("# tiles: n8=%u n10=%u n12=%u n16=%u\n",t8.n,t10.n,t12.n,t16.n);
        printf("# occupancy (real, cudaOccupancy):\n");
        report_kernel("NT=8", iq2_ntile_kernel<1024u,8u>,256,iq2_smem_bytes(8,xq_blocks));
        report_kernel("NT=10",iq2_ntile_kernel<1024u,10u>,256,iq2_smem_bytes(10,xq_blocks));
        report_kernel("NT=12",iq2_ntile_kernel<1024u,12u>,256,iq2_smem_bytes(12,xq_blocks));
        report_kernel("NT=16",iq2_ntile_kernel<1024u,16u>,256,iq2_smem_bytes(16,xq_blocks));

        #define RUN(NT,TL) do{ \
            size_t smem=iq2_smem_bytes((NT),xq_blocks); \
            dim3 g((MID_DIM+1023u)/1024u,(TL).n,1); \
            /* warmup + capture ref/out */ \
            CHECK(cudaMemset(d_mid,0,mid_elems*sizeof(float))); \
            iq2_ntile_kernel<1024u,(NT)><<<g,256,smem>>>(d_mid,d_gate,d_up,d_xq,rt.d_sorted_pairs,rt.d_offsets,rt.d_counts,(TL).d_total,(TL).d_experts,(TL).d_starts,d_weights,gate_expert_bytes,gate_row_bytes,xq_blocks,MID_DIM,N_EXPERT_USED,clamp); \
            CHECK(cudaDeviceSynchronize()); \
            float best=1e30f; \
            for(int it=0;it<iters;it++){ CHECK(cudaEventRecord(e0)); \
                iq2_ntile_kernel<1024u,(NT)><<<g,256,smem>>>(d_mid,d_gate,d_up,d_xq,rt.d_sorted_pairs,rt.d_offsets,rt.d_counts,(TL).d_total,(TL).d_experts,(TL).d_starts,d_weights,gate_expert_bytes,gate_row_bytes,xq_blocks,MID_DIM,N_EXPERT_USED,clamp); \
                CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1)); float ms; CHECK(cudaEventElapsedTime(&ms,e0,e1)); if(ms<best)best=ms; } \
            printf("    NT=%-2d  best=%.3f ms  tiles=%u\n",(NT),best,(TL).n); \
            iq2_last_ms=best; \
        }while(0)

        float iq2_last_ms=0;
        printf("# timings (best of %d):\n",iters);
        RUN(8,t8);  float ms8=iq2_last_ms; CHECK(cudaMemcpy(d_ref,d_mid,mid_elems*sizeof(float),cudaMemcpyDeviceToDevice));
        /* bit-exactness vs NT=8 for each wider tile */
        float*hr=(float*)malloc(mid_elems*sizeof(float)); CHECK(cudaMemcpy(hr,d_ref,mid_elems*sizeof(float),cudaMemcpyDeviceToHost));
        float*ho=(float*)malloc(mid_elems*sizeof(float));
        #define BITEXACT(TAG) do{ CHECK(cudaMemcpy(ho,d_mid,mid_elems*sizeof(float),cudaMemcpyDeviceToHost)); int bad=0; for(size_t i=0;i<mid_elems&&bad<5;i++) if(memcmp(&hr[i],&ho[i],4)!=0){bad++;} printf("      bitexact %s vs NT=8: %s\n",TAG,bad?"** DIFFERS **":"identical"); }while(0)
        RUN(10,t10); float ms10=iq2_last_ms; BITEXACT("NT=10");
        RUN(12,t12); float ms12=iq2_last_ms; BITEXACT("NT=12");
        RUN(16,t16); float ms16=iq2_last_ms; BITEXACT("NT=16");
        printf("  >> speedups vs NT=8: NT10=%.3fx  NT12=%.3fx  NT16=%.3fx\n", ms8/ms10, ms8/ms12, ms8/ms16);
        free(hr);free(ho);
        free_tiles(&t8);free_tiles(&t10);free_tiles(&t12);free_tiles(&t16);
        cudaFree(rt.d_sorted_pairs);cudaFree(rt.d_offsets);cudaFree(rt.d_counts);
    }
    return 0;
}
