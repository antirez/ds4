/* Actual supplementary kvstore regression. Tiny mock session payloads
 * exercise file publication, eligibility and ownership, not model arithmetic. */
#include "ds4.h"
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
struct ds4_engine {
    int qwen;
    int model_id;
};
struct ds4_session {
    ds4_tokens tokens;
    uint32_t state[4];
};
static char fixture_dir[256];
static bool override_payload_size;
static uint64_t reported_payload_bytes;
static int fail_stage, fail_write, fail_rename, stage_calls;
static int fail_dir,fail_read_dir;
static const char *fail_unlink_path;
static const char *fail_update_path,*fail_read_path;
static int fixture_rename(const char *from, const char *to) {
    if (fail_rename) {
        errno=EIO;
        return -1;
    }
    return rename(from,to);
}
static DIR *fixture_opendir(const char *path){
    if(fail_dir){
        errno=EACCES;
        return NULL;
    }
    return opendir(path);
}
static struct dirent *fixture_readdir(DIR *dir){
    if(fail_read_dir){
        errno=EIO;
        return NULL;
    }
    return readdir(dir);
}
static int fixture_unlink(const char *path){
    if(fail_unlink_path&&!strcmp(path,fail_unlink_path)){
        errno=EIO;
        return -1;
    }
    return unlink(path);
}
static FILE *fixture_fopen(const char *path,const char *mode){
    if((fail_update_path&&!strcmp(path,fail_update_path)&&!strcmp(mode,"r+b"))||(fail_read_path&&!strcmp(path,fail_read_path)&&!strcmp(mode,"rb"))){
        errno=EACCES;
        return NULL;
    }
    return fopen(path,mode);
}
#define rename fixture_rename
#define opendir fixture_opendir
#define readdir fixture_readdir
#define unlink fixture_unlink
#define fopen fixture_fopen
#include "../ds4_kvstore.c"
#undef rename
#undef opendir
#undef readdir
#undef unlink
#undef fopen
void ds4_tokens_push(ds4_tokens *t,int x) {
    if(t->len==t->cap){
        t->cap=t->cap?t->cap*2:16;
        t->v=realloc(t->v,(size_t)t->cap*sizeof(int));
        assert(t->v);
    }
    t->v[t->len++]=x;
}
void ds4_tokens_free(ds4_tokens *t){
    free(t->v);
    memset(t,0,sizeof(*t));
}
void ds4_tokens_copy(ds4_tokens *to,const ds4_tokens *from){
    to->len=0;
    for(int i=0;i<from->len;i++)ds4_tokens_push(to,from->v[i]);
}
bool ds4_tokens_starts_with(const ds4_tokens *t,const ds4_tokens *p){
    return t->len>=p->len&&!memcmp(t->v,p->v,(size_t)p->len*sizeof(int));
}
int ds4_engine_routed_quant_bits(ds4_engine *e){
    (void)e;
    return 4;
}
int ds4_engine_model_id(ds4_engine *e){
    return e->model_id;
}
bool ds4_engine_is_qwen4(ds4_engine *e){
    return e->qwen!=0;
}
uint32_t ds4_engine_prefill_chunk(ds4_engine *e){
    (void)e;
    return 8192;
}
int ds4_token_eos(ds4_engine *e){
    (void)e;
    return -1;
}
char *ds4_token_text(ds4_engine *e,int id,size_t *n){
    (void)e;
    char *p=malloc(2);
    assert(p);
    p[0]=(char)('a'+id%26);
    p[1]=0;
    *n=1;
    return p;
}
void ds4_tokenize_rendered_chat(ds4_engine *e,const char *s,ds4_tokens *t){
    (void)e;
    for(;*s;s++)ds4_tokens_push(t,(unsigned char)*s);
}
const ds4_tokens *ds4_session_tokens(ds4_session *s){
    return &s->tokens;
}
int ds4_session_ctx(ds4_session *s){
    (void)s;
    return 262144;
}
void ds4_session_invalidate(ds4_session *s){
    s->tokens.len=0;
}
uint64_t ds4_session_payload_bytes(ds4_session *s) {
    return override_payload_size ? reported_payload_bytes : 4ull + 4ull * s->tokens.len + 16ull;
}
int ds4_session_stage_payload(ds4_session *s,ds4_session_payload_file *out,char *err,size_t cap){
    (void)err;
    (void)cap;
    stage_calls++;
    if(fail_stage)return 1;
    char path[300];
    snprintf(path,sizeof(path),"%s/staged-XXXXXX",fixture_dir);
    int fd=mkstemp(path);
    assert(fd>=0);
    FILE *f=fdopen(fd,"wb");
    assert(f);
    uint32_t n=(uint32_t)s->tokens.len;
    assert(fwrite(&n,4,1,f)==1);
    assert(fwrite(s->tokens.v,4,n,f)==n);
    assert(fwrite(s->state,4,4,f)==4);
    assert(!fclose(f));
    out->path=strdup(path);
    out->bytes=4ull+4ull*n+16ull;
    return 0;
}
int ds4_session_write_staged_payload(const ds4_session_payload_file *p,FILE *out,char *err,size_t cap){
    (void)err;
    (void)cap;
    FILE *f=fopen(p->path,"rb");
    assert(f);
    unsigned char buf[512];
    uint64_t left=p->bytes;
    if(fail_write){
        assert(fwrite("partial",1,7,out)==7);
        fclose(f);
        errno=EIO;
        return 1;
    }
    while(left){
        size_t n=left<sizeof(buf)?(size_t)left:sizeof(buf);
        assert(fread(buf,1,n,f)==n);
        assert(fwrite(buf,1,n,out)==n);
        left-=n;
    }fclose(f);
    return 0;
}
void ds4_session_payload_file_free(ds4_session_payload_file *p){
    if(p->path){
        assert(!unlink(p->path));
        free(p->path);
    }memset(p,0,sizeof(*p));
}
int ds4_session_load_payload(ds4_session *s,FILE *f,uint64_t bytes,char *err,size_t cap){
    (void)err;
    (void)cap;
    uint32_t n=0;
    if(fread(&n,4,1,f)!=1||bytes!=4ull+4ull*n+16ull)return 1;
    s->tokens.len=0;
    for(uint32_t i=0;i<n;i++){
        int x;
        if(fread(&x,4,1,f)!=1)return 1;
        ds4_tokens_push(&s->tokens,x);
    }
    return fread(s->state,4,4,f)==4?0:1;
}
static bool growing_trailer_size(void *ud, const char *text, uint64_t *bytes) {
    (void)ud;
    (void)text;
    *bytes=1;
    return true;
}

static bool growing_trailer_write(void *ud, FILE *fp, const char *text,
                                  uint64_t *bytes) {
    (void)ud;
    (void)text;
    const unsigned char zero[1024]={0};
    *bytes=0;
    for(int i=0;i<200;i++) {
        if(fwrite(zero,1,sizeof(zero),fp)!=sizeof(zero))return false;
        *bytes+=sizeof(zero);
    }
    return true;
}

static bool store(ds4_kvstore *k,ds4_engine *e,ds4_session *s,const char *text,const char *why){
    char err[160];
    return ds4_kvstore_store_live_prefix_text(k,e,s,&s->tokens,s->tokens.len,why,text,DS4_KVSTORE_EXT_RESPONSES_VISIBLE,"responses-prompt",NULL,err,sizeof(err));
}
static char *key_path(ds4_kvstore *k,const char *text){
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text,strlen(text),sha);
    return ds4_kvstore_path_for_sha(k,sha);
}
static void limits(ds4_kvstore *k){
    kv_cache_refresh(k);
    uint64_t n=0;
    int c=0;
    for(int i=0;i<k->len;i++)if(k->entry[i].reason==DS4_KVSTORE_REASON_NEAR_TAIL){
        c++;
        n+=k->entry[i].file_size;
    }assert(c<=2&&n<=kv_cache_tail_budget(k));
}
static unsigned char *file_bytes(const char *path,size_t *n){
    struct stat st;
    assert(!stat(path,&st));
    *n=(size_t)st.st_size;
    unsigned char *p=malloc(*n);
    assert(p);
    FILE *f=fopen(path,"rb");
    assert(f&&fread(p,1,*n,f)==*n);
    fclose(f);
    return p;
}
int main(void){
    strcpy(fixture_dir,"/tmp/ds4-kvstore-tail-XXXXXX");
    assert(mkdtemp(fixture_dir));
    ds4_engine e={
        .qwen=1
    };
    ds4_session s={
        0
    },loaded={
        0
    };
    for(int i=0;i<512;i++)ds4_tokens_push(&s.tokens,i%26);
    uint32_t state[4]={
        0x80000000,0x3f800001,0x7f7fffff,0x00000001
    };
    memcpy(s.state,state,sizeof(state));
    ds4_kvstore k={
        0
    };
    ds4_kvstore_options opt=ds4_kvstore_default_options();
    assert(ds4_kvstore_open(&k,fixture_dir,1,false,opt,"kvstore-test",NULL,NULL));
    assert(store(&k,&e,&s,"ordinary","cold"));
    char *ordinary=key_path(&k,"ordinary");
    struct stat old;
    assert(!stat(ordinary,&old));
    assert(store(&k,&e,&s,"prefix-A","near-tail"));
    char *a=key_path(&k,"prefix-A");
    assert(store(&k,&e,&s,"prefix-B","near-tail"));
    char *b=key_path(&k,"prefix-B");
    limits(&k);
    /* Failed staging/write/rename leaves both prior tails and ordinary anchor. */
    for(int i=0;i<3;i++){
        fail_stage=i==0;
        fail_write=i==1;
        fail_rename=i==2;
        assert(!store(&k,&e,&s,"failed","near-tail"));
        assert(!access(a,F_OK)&&!access(b,F_OK)&&!access(ordinary,F_OK));
        limits(&k);
    }fail_stage=fail_write=fail_rename=0;
    fail_dir=1;
    assert(!store(&k,&e,&s,"dir-failure","near-tail"));
    fail_dir=0;
    fail_read_dir=1;
    assert(!store(&k,&e,&s,"read-dir-failure","near-tail"));
    fail_read_dir=0;
    limits(&k);
    fail_read_path=ordinary;
    assert(!store(&k,&e,&s,"unreadable-inventory","near-tail"));
    fail_read_path=NULL;
    assert(!access(ordinary,F_OK));
    limits(&k);
    char badpath[300];
    snprintf(badpath,sizeof(badpath),"%s/0000000000000000000000000000000000000000.kv",fixture_dir);
    FILE *bad=fopen(badpath,"wb");
    assert(bad&&fwrite("bad",1,3,bad)==3&&!fclose(bad));
    assert(!store(&k,&e,&s,"bad-inventory","near-tail"));
    assert(!access(ordinary,F_OK));
    assert(!unlink(badpath));
    limits(&k);
    /* Fail the precise first prune victim: new path rolls back, old tails stay. */
    kv_cache_refresh(&k);
    const ds4_kvstore_entry *victim=NULL;
    for(int i=0;i<k.len;i++)if(k.entry[i].reason==DS4_KVSTORE_REASON_NEAR_TAIL&&(!victim||k.entry[i].last_used<victim->last_used||(k.entry[i].last_used==victim->last_used&&strcmp(k.entry[i].sha,victim->sha)<0)))victim=&k.entry[i];
    assert(victim);
    char *vpath=strdup(victim->path);
    fail_unlink_path=vpath;
    assert(!store(&k,&e,&s,"prune-failure","near-tail"));
    fail_unlink_path=NULL;
    free(vpath);
    assert(!access(a,F_OK)&&!access(b,F_OK)&&!access(ordinary,F_OK));
    char *failed=key_path(&k,"prune-failure");
    assert(access(failed,F_OK));
    free(failed);
    limits(&k);
    /* Compatible ordinary read-only duplicate is neither demoted nor rewritten. */
    assert(!chmod(ordinary,0400));
    int stages=stage_calls;
    assert(store(&k,&e,&s,"ordinary","near-tail"));
    assert(stage_calls==stages);
    struct stat now;
    assert(!stat(ordinary,&now)&&old.st_size==now.st_size);
    assert(!chmod(ordinary,0600));
    /* The same visible key from another model cannot replace an anchor. */
    size_t model_n1,model_n2;
    unsigned char *model_before=file_bytes(ordinary,&model_n1);
    e.model_id=1;
    stages=stage_calls;
    assert(!store(&k,&e,&s,"ordinary","near-tail")&&stage_calls==stages);
    unsigned char *model_after=file_bytes(ordinary,&model_n2);
    assert(model_n1==model_n2&&!memcmp(model_before,model_after,model_n1));
    free(model_before);
    free(model_after);
    e.model_id=0;
    /* Growing trailer can pass estimation but must fail final published cap. */
    ds4_kvstore_trailer_hooks growing={.serialized_size=growing_trailer_size,
                                      .write=growing_trailer_write};
    char trailer_err[160];
    stages=stage_calls;
    assert(!ds4_kvstore_store_live_prefix_text(&k,&e,&s,&s.tokens,s.tokens.len,
        "near-tail","growing-trailer",DS4_KVSTORE_EXT_RESPONSES_VISIBLE,
        "responses-prompt",&growing,trailer_err,sizeof(trailer_err)));
    char *grown_path=key_path(&k,"growing-trailer");
    assert(stage_calls==stages+1&&access(grown_path,F_OK)&&!access(ordinary,F_OK));
    free(grown_path);
    limits(&k);
    /* Incompatible published ordinary path is never unlinked or replaced. */
    FILE *f=fopen(ordinary,"r+b");
    assert(f);
    assert(!fseek(f,16,SEEK_SET));
    uint8_t ctx[4];
    ds4_kvstore_le_put32(ctx,524288);
    assert(fwrite(ctx,1,4,f)==4&&!fclose(f));
    size_t n1,n2;
    unsigned char *before=file_bytes(ordinary,&n1);
    stages=stage_calls;
    assert(!store(&k,&e,&s,"ordinary","near-tail"));
    unsigned char *after=file_bytes(ordinary,&n2);
    assert(n1==n2&&!memcmp(before,after,n1)&&stage_calls==stages);
    free(before);
    free(after);
    f=fopen(ordinary,"r+b");
    assert(f&&!fseek(f,16,SEEK_SET));
    ds4_kvstore_le_put32(ctx,262144);
    assert(fwrite(ctx,1,4,f)==4&&!fclose(f));
    assert(store(&k,&e,&s,"prefix-C","near-tail"));
    limits(&k);
    assert(!access(ordinary,F_OK));
    /* Prefix/hash and mock payload restoration, including exact f32 bit patterns. */
    ds4_tokens effective={
        0
    };
    ds4_kvstore_load_result result={
        0
    };
    assert(ds4_kvstore_try_load_text(&k,&e,&loaded,"prefix-C-suffix",&effective,&result,NULL,true)==512);
    assert(!memcmp(loaded.state,state,sizeof(state)));
    assert(ds4_tokens_starts_with(&effective,&s.tokens)&&effective.len==519);
    ds4_kvstore_load_result_free(&result);
    ds4_tokens_free(&effective);
    assert(ds4_kvstore_find_text_prefix(&k,"prefix-D",0,4,262144)<0);
    /* All ordinary entries retain positive score; tails remain expendable after hits. */
    kv_cache_refresh(&k);
    for(int i=0;i<k.len;i++){
        k.entry[i].hits=UINT32_MAX;
        double x=ds4_kvstore_entry_eviction_score(&k.entry[i],NULL,(uint64_t)time(NULL),NULL);
        assert(k.entry[i].reason==DS4_KVSTORE_REASON_NEAR_TAIL?x==0:x>0);
    }
    k.budget_bytes=1024;
    stages=stage_calls;
    assert(!store(&k,&e,&s,"too-large","near-tail"));
    assert(!access(ordinary,F_OK));
    assert(stage_calls==stages);
    k.budget_bytes=1024*1024;
    /* Reject known oversized/invalid supplementary payloads before readback.
    * Test metadata sizes only; never allocate the represented payload. */
    override_payload_size=true;
    uint64_t sizes[]={
        0, UINT64_MAX, 1024*1024
    };
    for(size_t i=0;i<sizeof(sizes)/sizeof(sizes[0]);i++){
        reported_payload_bytes=sizes[i];
        stages=stage_calls;
        assert(!store(&k,&e,&s,"preflight-oversized","near-tail"));
        assert(stage_calls==stages&&!access(ordinary,F_OK));
    }
    override_payload_size=false;
    /* Large-size metadata boundary, no payload allocation or inference. */
    k.budget_bytes=256000ull*1024*1024;
    uint64_t cap16=16ull*1024*1024*1024;
    assert(kv_cache_tail_admit(&k,cap16,cap16+cap16/100+1));
    assert(!kv_cache_tail_admit(&k,cap16+1,cap16+cap16/100+2));
    k.budget_bytes=1024*1024;
    /* Same-key ordinary periodic store promotes policy without changing payload. */
    char *c=key_path(&k,"prefix-C");
    size_t cb1,cb2;
    unsigned char *oldc=file_bytes(c,&cb1);
    fail_update_path=c;
    assert(!store(&k,&e,&s,"prefix-C","continued"));
    fail_update_path=NULL;
    unsigned char *failedc=file_bytes(c,&cb2);
    assert(cb1==cb2&&!memcmp(oldc,failedc,cb1));
    free(failedc);
    assert(store(&k,&e,&s,"prefix-C","continued"));
    unsigned char *newc=file_bytes(c,&cb2);
    assert(cb1==cb2&&oldc[5]==DS4_KVSTORE_REASON_NEAR_TAIL&&newc[5]==DS4_KVSTORE_REASON_CONTINUED);
    oldc[5]=newc[5];
    assert(!memcmp(oldc,newc,cb1));
    free(oldc);
    free(newc);
    free(c);
    limits(&k);
    /* Reopen retains reason/priority; old reason header values still parse. */
    ds4_kvstore_close(&k);
    assert(ds4_kvstore_open(&k,fixture_dir,1,false,opt,"kvstore-test",NULL,NULL));
    limits(&k);
    for(int reason=0;reason<=DS4_KVSTORE_REASON_AGENT_SESSION;reason++){
        FILE *f=tmpfile();
        assert(f);
        uint8_t h[48],tb[4];
        ds4_kvstore_fill_header(h,0,4,(uint8_t)reason,0,512,0,262144,1,1,0);
        ds4_kvstore_le_put32(tb,0);
        assert(fwrite(h,1,48,f)==48&&fwrite(tb,1,4,f)==4);
        rewind(f);
        ds4_kvstore_entry ent={
            0
        };
        uint32_t size;
        assert(ds4_kvstore_read_header(f,&ent,&size)&&ent.reason==reason);
        fclose(f);
    }
    free(a);
    free(b);
    free(ordinary);
    ds4_tokens_free(&s.tokens);
    ds4_tokens_free(&loaded.tokens);
    kv_cache_refresh(&k);
    for(int i=0;i<k.len;i++)assert(!unlink(k.entry[i].path));
    ds4_kvstore_close(&k);
    assert(!rmdir(fixture_dir));
    puts("KV supplementary checkpoint fixture PASS (mock payload; no model oracle)");
    return 0;
}
