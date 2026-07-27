/* plan-34 phase-2 increment 3 — K-row single-bank PREFILL through the mixed entry.
 *
 * Routes a K-row prefill chunk (one bank, positions p0..p0+K-1) through
 * ds4_session_decode_mixed and checks it is (1) COHERENT vs classic chunked
 * prefill, (2) correct across the K>ratio compressor read-after-write boundary,
 * and (3) TENSOR-CORE FAST (prefix=0 => no forced-custom kernel). NO decode
 * co-scheduling (that + the decode/prefill split + neutrality is inc 4).
 *
 * ORACLE = output-token coherence (NOT byte-identity): the mixed prefill resumes
 * with the accepted last-ulp warm-continuation KV delta, so greedy tokens are
 * compared, not logit bytes (same class as fork-gate P5).
 *
 * Shape: classic-prefill a small FIRST chunk [0,c0) to lift the bank frontier off
 * 0 (step_begin rejects pos-0), then feed [c0,c0+K) as ONE K-row mixed run; then
 * greedy-decode NGEN tokens (1-row decode_mixed steps) and compare to a fully
 * classic-prefilled reference decoded the same way.
 *
 * Run under DS4_MSEQ_BANKS>=1, pack on/off x idx-fp4 on/off. GPU discipline.
 *   usage: DS4_MSEQ_BANKS=2 ./tests/mixed_prefill_gate MODEL
 */
#include "ds4.h"
#include "ds4_engine_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static ds4_engine *g_e;
static ds4_tokens g_toks;
static int g_fail;
#define NGEN 24
#define C0   128            /* first (classic) chunk: lifts frontier off 0, ratio-aligned */

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static char *read_file(const char *p, size_t *n){ FILE *f=fopen(p,"rb"); if(!f)return NULL;
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET); char *b=(char *)malloc(s+1);
    if(!b||fread(b,1,s,f)!=(size_t)s){fclose(f);free(b);return NULL;} fclose(f); b[s]=0; if(n)*n=s; return b; }

/* greedy-decode NGEN tokens on bank 0, continuing from a frontier at F with the
 * prefill's first predicted token t0 already in out[0]; uses 1-row decode_mixed. */
static bool decode_cont(ds4_session *s, int F, int t0, int *out){
    out[0]=t0;
    const int vocab=(int)DS4_N_VOCAB;
    float *lg = (float *)malloc((size_t)vocab*sizeof(float));
    char e[256]; bool ok=true;
    for(int i=1; i<NGEN && ok; i++){
        ds4_multiseq_req r={.bank=0,.pos=F+i-1,.token=out[i-1]};
        uint32_t nr=0;
        if(ds4_session_decode_mixed(s,&r,1,lg,vocab,&nr,0u,e,sizeof e)!=0){ fprintf(stderr,"cont step %d: %s\n",i,e); ok=false; break; }
        out[i]=(int)argmax_f32(lg,(uint64_t)vocab);
    }
    free(lg); return ok;
}

/* classic RESUME reference: prefill [0,F) via ds4_session_sync (the second sync
 * resumes at c0>0 = same decode-attention kernel as the mixed path), copy the
 * last-position full-vocab logits, then decode NGEN. */
static bool classic_stream(int c0, int F, int *out, float *out_lg){
    ds4_session *s=NULL; if(ds4_session_create(&s,g_e,4096)!=0) return false;
    ds4_gpu_graph *g=&s->graph; char e[256]; bool ok=true;
    if(g->banks.n_banks && !gpu_graph_bank_repoint(g,0)){ ds4_session_free(s); return false; }
    ds4_session_invalidate(s);
    ds4_tokens p0={.v=g_toks.v,.len=c0,.cap=c0};
    if(ds4_session_sync(s,&p0,e,sizeof e)!=0){ fprintf(stderr,"classic first-chunk: %s\n",e); ok=false; }
    ds4_tokens p={.v=g_toks.v,.len=F,.cap=F};           /* RESUMES at c0 (pos0>0) */
    if(ok && ds4_session_sync(s,&p,e,sizeof e)!=0){ fprintf(stderr,"classic resume: %s\n",e); ok=false; }
    if(ok){ gpu_graph_bank_counters_capture(g,0);
            ds4_session_copy_logits(s,out_lg,(int)DS4_N_VOCAB);   /* last-position logits */
            ok=decode_cont(s,F,ds4_session_argmax(s),out); }
    ds4_session_free(s); return ok;
}

/* mixed: classic [0,c0), then K-row mixed run [c0,c0+K); decode NGEN. *secs = the
 * timed mixed-run seconds when non-NULL. */
static bool mixed_stream(int c0, int K, int *out, double *secs, float *out_lg){
    ds4_session *s=NULL; if(ds4_session_create(&s,g_e,4096)!=0) return false;
    ds4_gpu_graph *g=&s->graph; char e[256]; bool ok=true;
    if(g->banks.n_banks && !gpu_graph_bank_repoint(g,0)){ ds4_session_free(s); return false; }
    ds4_session_invalidate(s);
    ds4_tokens p={.v=g_toks.v,.len=c0,.cap=c0};
    if(ds4_session_sync(s,&p,e,sizeof e)!=0){ fprintf(stderr,"mixed first-chunk sync: %s\n",e); ok=false; }
    if(ok) gpu_graph_bank_counters_capture(g,0);
    const int vocab=(int)DS4_N_VOCAB;
    float *lg = (float *)malloc((size_t)vocab*sizeof(float));    /* 1 run => 1 logit row */
    ds4_multiseq_req *rq = (ds4_multiseq_req *)malloc((size_t)K*sizeof(*rq));
    for(int j=0;j<K;j++){ rq[j].bank=0; rq[j].pos=c0+j; rq[j].token=g_toks.v[c0+j]; }
    uint32_t nr=0;
    double t0=secs?now_s():0.0;
    if(ok && ds4_session_decode_mixed(s,rq,(uint32_t)K,lg,vocab,&nr,0u,e,sizeof e)!=0){ fprintf(stderr,"mixed K-run: %s\n",e); ok=false; }
    if(secs) *secs=now_s()-t0;
    if(ok && nr!=1){ fprintf(stderr,"mixed run n_rows=%u expected 1\n",nr); ok=false; }
    if(ok && out_lg) memcpy(out_lg,lg,(size_t)vocab*sizeof(float));   /* last-position logits */
    if(ok) ok=decode_cont(s,c0+K,(int)argmax_f32(lg,(uint64_t)vocab),out);
    free(lg); free(rq); ds4_session_free(s); return ok;
}

/* classic prefill of K tokens from a c0 frontier (for the speed baseline). */
static bool classic_prefill_time(int c0, int K, double *secs){
    ds4_session *s=NULL; if(ds4_session_create(&s,g_e,4096)!=0) return false;
    ds4_gpu_graph *g=&s->graph; char e[256]; bool ok=true;
    if(g->banks.n_banks && !gpu_graph_bank_repoint(g,0)){ ds4_session_free(s); return false; }
    ds4_session_invalidate(s);
    ds4_tokens p0={.v=g_toks.v,.len=c0,.cap=c0};
    if(ds4_session_sync(s,&p0,e,sizeof e)!=0) ok=false;              /* untimed first chunk */
    ds4_tokens p1={.v=g_toks.v,.len=c0+K,.cap=c0+K};
    double t0=now_s();
    if(ok && ds4_session_sync(s,&p1,e,sizeof e)!=0) ok=false;        /* resumes at c0, prefills K */
    *secs=now_s()-t0;
    ds4_session_free(s); return ok;
}

static int firstdiff(const int*a,const int*b,int n){ for(int i=0;i<n;i++) if(a[i]!=b[i]) return i; return -1; }

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s MODEL\n",argv[0]); return 2; }
    ds4_engine_options o; memset(&o,0,sizeof o); o.model_path=argv[1]; o.backend=DS4_BACKEND_CUDA;
    if(ds4_engine_open(&g_e,&o)!=0){ fprintf(stderr,"engine open failed\n"); return 1; }
    printf("CONFIG: DS4_ATTN_PACK=%s DS4_IDX_FP4=%s\n",
        getenv("DS4_ATTN_PACK")?getenv("DS4_ATTN_PACK"):"(unset,default 1)",
        getenv("DS4_IDX_FP4")?getenv("DS4_IDX_FP4"):"(unset,default 1)");
    size_t tl=0; char*txt=read_file("tests/long_context_story_prompt.txt",&tl);
    if(!txt){ fprintf(stderr,"prompt read failed\n"); return 1; }
    memset(&g_toks,0,sizeof g_toks); ds4_tokenize_text(g_e,txt,&g_toks); free(txt);

    const int K1=512;                 /* gates 1&2: spans many ratio-4 (and ratio-128) groups */
    if(C0+K1+NGEN > g_toks.len){ fprintf(stderr,"prompt too short (%d)\n",g_toks.len); return 1; }

    /* GATE 1 (rigorous oracle) + GATE 2 (K>ratio boundary). K1 spans >1 ratio-4/128
     * group; step_end self-check runs inside decode_mixed (rc=-1 on a frontier miss).
     * THREE assertions distinguish accepted last-ulp drift from KV corruption:
     *   (1) NEXT-TOKEN exact vs classic-resume (prefill boundary + committed KV right),
     *   (2) last-position full-vocab logit rel-RMS < 1e-2 (corruption => large error),
     *   (3) the NGEN continuation is coherent (valid, non-degenerate), not required to
     *       match classic past the drift point. */
    const int vocab=(int)DS4_N_VOCAB;
    int ref[NGEN], mix[NGEN];
    float *ref_lg = (float *)malloc((size_t)vocab*sizeof(float)), *mix_lg=(float *)malloc((size_t)vocab*sizeof(float));
    if(!classic_stream(C0, C0+K1, ref, ref_lg)){ fprintf(stderr,"GATE FAIL: classic-resume reference failed\n"); g_fail=1; free(ref_lg);free(mix_lg); goto done; }
    if(!mixed_stream(C0, K1, mix, NULL, mix_lg)){ fprintf(stderr,"GATE FAIL: mixed prefill failed (step_end/coherence)\n"); g_fail=1; free(ref_lg);free(mix_lg); goto done; }
    {
        /* (2) relative RMS of the last-position logit vectors. */
        double se=0, sr=0;
        for(int i=0;i<vocab;i++){ double d=(double)mix_lg[i]-ref_lg[i]; se+=d*d; sr+=(double)ref_lg[i]*ref_lg[i]; }
        double rel_rms = sr>0 ? sqrt(se/sr) : (se>0?1e9:0.0);
        /* (3) coherence: valid ids + not a single repeated token (non-degenerate). */
        int distinct=1; for(int i=1;i<NGEN;i++) if(mix[i]!=mix[0]){ distinct=2; break; }
        int allvalid=1; for(int i=0;i<NGEN;i++) if(mix[i]<0||mix[i]>=vocab){ allvalid=0; break; }
        int cd=firstdiff(mix,ref,NGEN);   /* informational: continuation divergence point */
        printf("GATE 1: next-token mixed=%d classic=%d %s | last-pos logit rel-RMS=%.3e (<1e-2: %s) | continuation coherent=%s (valid=%d,distinct=%d, matches-classic-for=%d tok)\n",
               mix[0],ref[0], mix[0]==ref[0]?"MATCH":"MISMATCH", rel_rms, rel_rms<1e-2?"YES":"NO",
               (allvalid&&distinct>=2)?"YES":"NO", allvalid, distinct, cd<0?NGEN:cd);
        printf("GATE 2: K=%d spans >1 ratio group; step_end frontier self-check PASSED (mixed run returned valid tokens, no rc=-1)\n",K1);
        if(mix[0]!=ref[0]){ fprintf(stderr,"GATE 1 FAIL: next-token mismatch (prefill boundary/KV wrong)\n"); g_fail=1; }
        if(rel_rms>=1e-2){ fprintf(stderr,"GATE 1 FAIL: last-pos logit rel-RMS %.3e >= 1e-2 (KV corruption, not last-ulp drift)\n",rel_rms); g_fail=1; }
        if(!(allvalid&&distinct>=2)){ fprintf(stderr,"GATE 1 FAIL: continuation degenerate/garbage\n"); g_fail=1; }
    }
    free(ref_lg); free(mix_lg);

    /* GATE 3: SPEED — mixed K-run vs classic prefill of K, at K in {512,2048}. */
    int Ks[2];
    Ks[0]=512; Ks[1]=2048;
    for(int ki=0; ki<2; ki++){
        int K=Ks[ki];
        if(C0+K+NGEN > g_toks.len){ printf("GATE 3: K=%d skipped (prompt too short)\n",K); continue; }
        double t_mix=0, t_cls=0; int tmp[NGEN];
        if(!mixed_stream(C0,K,tmp,&t_mix,NULL)){ fprintf(stderr,"GATE 3 FAIL: mixed K=%d\n",K); g_fail=1; continue; }
        if(!classic_prefill_time(C0,K,&t_cls)){ fprintf(stderr,"GATE 3 FAIL: classic K=%d\n",K); g_fail=1; continue; }
        double sp_mix=K/t_mix, sp_cls=K/t_cls, ratio=t_mix/t_cls;
        /* Enforce parity only at a REALISTIC chunk (K<=512) vs classic-RESUME. Large
         * K (2048) is INFORMATIONAL: the mseq banked per-KEY indirection scales with
         * attended keys (~K^2 for prefill), so it grows with K. That banked cost is
         * the honest fused-step price inc-4's MULTI-bank step pays (it CANNOT use the
         * single-bank descr==0 shortcut) and inc-5 measures against the jitter win —
         * so it is documented here, deliberately NOT optimized away in a single-bank
         * special case inc-4 discards. */
        const char *tag = (K<=512) ? "PARITY(enforced)" : "INFORMATIONAL(banked per-key cost; inc-4 pays it)";
        printf("GATE 3 SPEED K=%d [%s]: mixed %.0f tok/s (%.3fs) vs classic-RESUME %.0f tok/s (%.3fs) -> %.2fx classic time\n",
               K,tag,sp_mix,t_mix,sp_cls,t_cls,ratio);
        if(K<=512 && ratio>1.5){ fprintf(stderr,"GATE 3 FAIL: K=%d mixed %.2fx slower than classic-RESUME (>1.5x)\n",K,ratio); g_fail=1; }
    }

done:
    ds4_engine_close(g_e);
    if(g_fail){ fprintf(stderr,"MIXED-PREFILL GATE: FAIL\n"); return 1; }
    printf("MIXED-PREFILL GATE: PASS\n"); return 0;
}
