/* Public-API, two-host V4.1 state/cancellation test. Peer: ordinary worker. */
#include "ds4.h"
#include "ds4_tp.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/stat.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s (%s)\n",__LINE__,#x,err);goto done; } } while(0)
static char err[512];
static const char *output;
static unsigned captures;
static int vocab;
static float *left,*right;
static FILE *record;
static int save(const char *name,const void *p,size_t n) {
    char path[4096];if(snprintf(path,sizeof(path),"%s/%s",output,name)>=(int)sizeof(path))return 0;
    FILE *f=fopen(path,"wb");if(!f)return 0;
    int ok=fwrite(p,1,n,f)==n;if(fclose(f))ok=0;return ok;
}
static int capture(ds4_session *s,float *v,const char *name) {
    unsigned char *raw=(unsigned char *)v-64;memset(raw,0xa5,(size_t)vocab*4+128);
    for(int j=0;j<vocab;j++)v[j]=NAN;
    if(ds4_session_copy_logits(s,v,vocab)!=vocab)return 0;
    for(int j=0;j<64;j++)if(raw[j]!=0xa5||raw[64+(size_t)vocab*4+j]!=0xa5)return 0;
    char path[128];snprintf(path,sizeof(path),"%03u-%s.f32",captures,name);
    if(!save(path,v,(size_t)vocab*4))return 0;
    const ds4_tokens *t=ds4_session_tokens(s);
    if(!t||t->len!=ds4_session_pos(s))return 0;
    snprintf(path,sizeof(path),"%03u-%s.tokens",captures,name);
    if(!save(path,t->v,(size_t)t->len*sizeof(*t->v)))return 0;
    for(int j=0;j<vocab;j++)if(!isfinite(v[j]))return 0;
    return 1;
}
static int same(ds4_session *a,ds4_session *b,const char *phase) {
    ++captures;
    if(!capture(a,left,"control")||!capture(b,right,"subject"))return 0;
    const ds4_tokens *ta=ds4_session_tokens(a),*tb=ds4_session_tokens(b);
    int tokens=ta->len==tb->len&&!memcmp(ta->v,tb->v,(size_t)ta->len*4);
    size_t mismatches=0;double max=0;
    for(int j=0;j<vocab;j++) {if(left[j]!=right[j])mismatches++;double d=fabs((double)left[j]-right[j]);if(d>max)max=d;}
    fprintf(record,"%u\t%s\t%d\t%d\t%zu\t%.17g\n",captures,phase,ta->len,tokens,mismatches,max);fflush(record);
    fprintf(stderr,"state phase=%s pos=%d history=%s logit_mismatches=%zu max=%g\n",phase,ta->len,tokens?"PASS":"FAIL",mismatches,max);
    return tokens&&!mismatches;
}
typedef struct {int stop_at;unsigned events;bool stopped;} interruption;
static void progress(void *ud,const char *event,int current,int total) {
    (void)event;(void)total;interruption *p=ud;p->events++;if(current>=p->stop_at)p->stopped=true;
}
static bool cancelled(void *ud) {return ((interruption *)ud)->stopped;}
int main(int argc,char **argv) {
    if(argc!=8){fprintf(stderr,"usage: %s MODEL PROMPT OUT tcp|usb4stream|rdma DEVICE LISTEN PORT\n",argv[0]);return 2;}
    int rc=1;ds4_engine *engine=NULL;ds4_tp *tp=NULL;ds4_session *control=NULL,*subject=NULL;
    ds4_tokens prompt={0};ds4_session_snapshot snap={0},decoded={0};char *text=NULL;FILE *f=NULL;
    output=argv[3];
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_CUDA,.context_size=16384,
        .power_percent=100,.placement_session_count_hint=2};
    opt.tp=(ds4_tp_options){.requested=true,.role=DS4_TP_LEADER,.listen_host=argv[6],.listen_port=atoi(argv[7])};
    if(!strcmp(argv[4],"tcp"))opt.tp.transport=DS4_TP_TRANSPORT_TCP;
    else if(!strcmp(argv[4],"usb4stream")){opt.tp.transport=DS4_TP_TRANSPORT_USB4STREAM;opt.tp.usb4stream_device=argv[5];}
    else if(!strcmp(argv[4],"rdma")){opt.tp.transport=DS4_TP_TRANSPORT_RDMA;opt.tp.rdma_device=argv[5];opt.tp.rdma_port=1;opt.tp.rdma_gid_index=1;opt.tp.rdma_gid_index_set=true;}
    else return 2;
    CHECK(mkdir(output,0700)==0);
    char path[4096];snprintf(path,sizeof(path),"%s/comparisons.tsv",output);record=fopen(path,"w");CHECK(record);
    fprintf(record,"capture\tphase\tpos\thistory_equal\tlogit_mismatches\tmax_delta\n");
    f=fopen(argv[2],"rb");CHECK(f);CHECK(!fseek(f,0,SEEK_END));long n=ftell(f);CHECK(n>0&&n<100000000);rewind(f);
    text=malloc((size_t)n+1);CHECK(text);CHECK(fread(text,1,n,f)==(size_t)n);text[n]=0;fclose(f);f=NULL;
    CHECK(ds4_engine_open(&engine,&opt)==0);
    ds4_tp_identity id={.gguf_bytes=ds4_engine_model_bytes(engine),.model_id=(uint32_t)ds4_engine_model_id(engine),
        .n_layer=(uint32_t)ds4_engine_layer_count(engine),.n_embd=(uint32_t)ds4_engine_embd_dim(engine),
        .n_vocab=(uint32_t)ds4_engine_vocab_size(engine),.quant_bits=(uint32_t)ds4_engine_routed_quant_bits(engine),.ctx_size=16384};
    ds4_engine_tp_gate_schedule(engine,&id.gate_slot_start,&id.gate_slot_step,&id.gates_per_token,id.gate_slot_mask);
    CHECK(ds4_tp_create(&tp,&opt.tp,&id,err,sizeof(err)));CHECK(ds4_engine_tp_bind(engine,tp,err,sizeof(err)));
    ds4_tokenize_text(engine,text,&prompt);CHECK(prompt.len>8194);free(text);text=NULL;
    vocab=ds4_engine_vocab_size(engine);CHECK(vocab==129280);
    unsigned char *a=malloc((size_t)vocab*4+128),*b=malloc((size_t)vocab*4+128);
    left=a?(float *)(a+64):NULL;right=b?(float *)(b+64):NULL;CHECK(left&&right);
    CHECK(!ds4_session_create(&control,engine,16384));CHECK(!ds4_session_create(&subject,engine,16384));
    const int lengths[]={31,32,255,256,1025,4097};
    for(unsigned i=0;i<sizeof(lengths)/sizeof(*lengths);i++) {
        prompt.len=lengths[i];ds4_session_invalidate(control);ds4_session_invalidate(subject);
        CHECK(!ds4_session_sync(control,&prompt,err,sizeof(err)));CHECK(!ds4_session_sync(subject,&prompt,err,sizeof(err)));
        CHECK(same(control,subject,"fresh"));
        CHECK(!ds4_session_sync(subject,&prompt,err,sizeof(err)));CHECK(same(control,subject,"no-op"));
        CHECK(!ds4_session_save_snapshot(subject,&snap,err,sizeof(err)));
        snprintf(path,sizeof(path),"snapshot-%d.bin",prompt.len);CHECK(save(path,snap.ptr,snap.len));
        for(int k=0;k<2;k++){CHECK(!ds4_session_eval(control,prompt.v[prompt.len+k],err,sizeof(err)));CHECK(!ds4_session_eval(subject,prompt.v[prompt.len+k],err,sizeof(err)));CHECK(same(control,subject,"interleaved-decode"));}
        CHECK(!ds4_session_save_snapshot(subject,&decoded,err,sizeof(err)));
        CHECK(!ds4_session_eval(subject,prompt.v[prompt.len+2],err,sizeof(err)));
        CHECK(!ds4_session_load_snapshot(subject,&decoded,err,sizeof(err)));
        CHECK(same(control,subject,"decoded-snapshot-exact"));
        ds4_session_snapshot_free(&decoded);
        CHECK(!ds4_session_load_snapshot(subject,&snap,err,sizeof(err)));
        ds4_session_invalidate(control);CHECK(!ds4_session_sync(control,&prompt,err,sizeof(err)));CHECK(same(control,subject,"snapshot-rebuild"));
        CHECK(!ds4_session_eval(control,prompt.v[prompt.len],err,sizeof(err)));CHECK(!ds4_session_eval(subject,prompt.v[prompt.len],err,sizeof(err)));CHECK(same(control,subject,"restored-decode"));
        prompt.len=lengths[i]-1;ds4_session_rewind(subject,prompt.len);
        CHECK(!ds4_session_sync(subject,&prompt,err,sizeof(err)));ds4_session_invalidate(control);CHECK(!ds4_session_sync(control,&prompt,err,sizeof(err)));
        CHECK(same(control,subject,"nonzero-rewind-rebuild"));
        CHECK(!ds4_session_eval(control,prompt.v[prompt.len],err,sizeof(err)));CHECK(!ds4_session_eval(subject,prompt.v[prompt.len],err,sizeof(err)));CHECK(same(control,subject,"rewound-decode"));
        /* Payload header corruption must be rejected without poisoning TP. */
        ((unsigned char *)snap.ptr)[0]^=1;CHECK(ds4_session_load_snapshot(subject,&snap,err,sizeof(err))!=0);((unsigned char *)snap.ptr)[0]^=1;
        CHECK(!ds4_tp_failed(tp));CHECK(!ds4_session_load_snapshot(subject,&snap,err,sizeof(err)));err[0]=0;
        ds4_session_snapshot_free(&snap);
    }
    for(unsigned mode=0;mode<3;mode++) {
        ds4_session_invalidate(subject);
        int prefix=mode==2?1025:0;
        if(prefix){prompt.len=prefix;CHECK(!ds4_session_sync(subject,&prompt,err,sizeof(err)));}
        prompt.len=mode==0?31:8192;
        interruption p={.stop_at=prefix+(prompt.len-prefix)/2,.stopped=mode==0};
        ds4_session_set_progress(subject,progress,&p);ds4_session_set_display_progress(subject,progress,&p);ds4_session_set_cancel(subject,cancelled,&p);
        CHECK(ds4_session_sync(subject,&prompt,err,sizeof(err))==DS4_SESSION_SYNC_INTERRUPTED);
        CHECK(p.stopped&&(mode==0||p.events));CHECK(!ds4_tp_failed(tp));
        CHECK(ds4_session_save_snapshot(subject,&snap,err,sizeof(err))!=0);ds4_session_snapshot_free(&snap);
        ds4_session_set_progress(subject,NULL,NULL);ds4_session_set_display_progress(subject,NULL,NULL);ds4_session_set_cancel(subject,NULL,NULL);
        prompt.len=1025;ds4_session_invalidate(control);CHECK(!ds4_session_sync(control,&prompt,err,sizeof(err)));CHECK(!ds4_session_sync(subject,&prompt,err,sizeof(err)));
        CHECK(same(control,subject,"cancelled-rebuild"));
        CHECK(!ds4_session_eval(control,prompt.v[prompt.len],err,sizeof(err)));CHECK(!ds4_session_eval(subject,prompt.v[prompt.len],err,sizeof(err)));CHECK(same(control,subject,"cancelled-next-decode"));
        fprintf(stderr,"cancellation mode=%u callbacks=%u both-rank same-connection recovery PASS\n",mode,p.events);
    }
    CHECK(!ds4_tp_failed(tp));rc=0;
    fprintf(stderr,"TP state: snapshots, corruption, rewind, interleaved sessions, cancellation full-vectors/guards PASS captures=%u\n",captures);
done:
    if(subject){ds4_session_set_cancel(subject,NULL,NULL);ds4_session_set_progress(subject,NULL,NULL);ds4_session_set_display_progress(subject,NULL,NULL);}
    if(f)fclose(f);
    if(record)fclose(record);
    ds4_session_snapshot_free(&snap);ds4_session_snapshot_free(&decoded);ds4_session_free(subject);ds4_session_free(control);
    if(tp&&!ds4_tp_failed(tp))ds4_tp_send_stop(tp);
    ds4_engine_close(engine);ds4_tp_free(tp);ds4_tokens_free(&prompt);free(text);
    if(left)free((unsigned char *)left-64);
    if(right)free((unsigned char *)right-64);
    return rc;
}
