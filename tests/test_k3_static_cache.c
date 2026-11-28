/* Includes private helpers for format/corruption tests; do NOT link module.o. */
#define K3_STATIC_CACHE_TEST 1
#include "../k3_static_cache.c"
#include <assert.h>
#include <string.h>

#ifndef K3_STATIC_CACHE_ZSTD
int main(void) {
    char e[256];k3_static_cache *c=(void *)1;
    assert(!k3_static_cache_open("unused",-1,NULL,0,4,&c,e,sizeof(e)) && !c);
    assert(strstr(e,"libzstd"));k3_static_cache_retain(NULL);k3_static_cache_release(NULL);
    puts("K3 static cache unavailable-build test passed");return 0;
}
#else
static uint32_t random32(uint32_t *s) {*s^=*s<<13;*s^=*s>>17;*s^=*s<<5;return *s;}
typedef struct {k3_static_cache *c;uint32_t layer;uint64_t off,n;uint8_t *out;int ok;} reader;
static void *read_thread(void *p) {
    reader *r=p;char err[256];uint64_t io=0;
    r->ok=k3_static_cache_read(r->c,r->layer,r->off,r->n,r->out,r->n,&io,err,sizeof(err));
    if(!r->ok)fprintf(stderr,"reader: %s\n",err);
    return NULL;
}
static void rewrite_header(int fd,k3sc_header *h) {
    h->header_crc32=header_crc(*h);assert(write_all(fd,h,sizeof(*h),0));
}

int main(void) {
    assert(crc32_bytes("123456789",9)==0xcbf43926u);
    char model_path[]="/tmp/k3-cache-model-XXXXXX";
    int fd=mkstemp(model_path);assert(fd>=0);
    char cache_path[sizeof(model_path)+16];snprintf(cache_path,sizeof(cache_path),"%s.cache",model_path);
    const size_t bytes=2u*K3SC_BLOCK_BYTES+K3SC_ALIGNMENT+14u;
    uint8_t *raw=malloc(bytes);assert(raw);
    uint32_t state=0x351abcdeu;
    for(size_t i=0;i<bytes;i+=2) {
        uint32_t r=random32(&state);raw[i]=(uint8_t)r;
        raw[i+1]=i<K3SC_BLOCK_BYTES+K3SC_ALIGNMENT
            ?(uint8_t)(0x3a+((r>>9)&3)+((r>>1)&0x80)):(uint8_t)(r>>8);
    }
    assert(write_all(fd,raw,bytes,0));
    k3_static_cache_span spans[]={
        {.layer=0,.offset=0,.bytes=K3SC_BLOCK_BYTES+K3SC_ALIGNMENT},
        {.layer=1,.offset=K3SC_BLOCK_BYTES+K3SC_ALIGNMENT,.bytes=K3SC_BLOCK_BYTES},
        {.layer=3,.offset=2u*K3SC_BLOCK_BYTES+K3SC_ALIGNMENT,.bytes=14},
    };
    char err[512]={0};
    assert(k3_static_cache_build(cache_path,fd,spans,3,err,sizeof(err)));
    assert(!k3_static_cache_build(cache_path,fd,spans,3,err,sizeof(err)));
    k3_static_cache *c=NULL;
    assert(k3_static_cache_open(cache_path,fd,&spans[1],1,4,&c,err,sizeof(err)));
    assert(c->header.block_count==4);
    assert(c->blocks[0].codec==K3SC_BF16_PLANES_ZSTD);
    assert(c->blocks[2].codec==K3SC_RAW);
    assert(k3_static_cache_workspace_bytes(c)<140ull*1024*1024);
    uint8_t *out=malloc(spans[0].bytes),*out2=malloc(spans[1].bytes);assert(out&&out2);
    uint64_t io=999;
    assert(!k3_static_cache_read(c,0,0,spans[0].bytes,out,spans[0].bytes,&io,err,sizeof(err)) && io==0);
    k3sc_test_fail_worker=2;
    assert(!k3_static_cache_prepare(c,err,sizeof(err)));
    assert(c->created==0 && !c->arena);
    k3sc_test_fail_worker=-1;
    assert(k3_static_cache_prepare(c,err,sizeof(err)));
    assert(k3_static_cache_prepare(c,err,sizeof(err)));
    uint64_t workspace=k3_static_cache_workspace_bytes(c);
    assert(k3_static_cache_read(c,0,0,spans[0].bytes,out,spans[0].bytes,&io,err,sizeof(err)));
    assert(!memcmp(out,raw,spans[0].bytes));assert(io==c->blocks[0].padded_bytes+c->blocks[1].padded_bytes);
    assert(!k3_static_cache_read(c,0,2,spans[0].bytes,out,spans[0].bytes,&io,err,sizeof(err)));
    assert(!k3_static_cache_read(c,0,0,spans[0].bytes,out,spans[0].bytes-1,&io,err,sizeof(err)));
    assert(!k3_static_cache_read(c,2,0,spans[0].bytes,out,spans[0].bytes,&io,err,sizeof(err)));
    reader r1={c,0,spans[0].offset,spans[0].bytes,out,0};
    reader r2={c,1,spans[1].offset,spans[1].bytes,out2,0};pthread_t t1,t2;
    k3_static_cache_retain(c);
    assert(!pthread_create(&t1,NULL,read_thread,&r1));assert(!pthread_create(&t2,NULL,read_thread,&r2));
    pthread_join(t1,NULL);pthread_join(t2,NULL);
    assert(r1.ok&&r2.ok&&!memcmp(out,raw,r1.n)&&!memcmp(out2,raw+r2.off,r2.n));
    assert(k3_static_cache_workspace_bytes(c)==workspace);
    k3_static_cache_release(c);
    assert(k3_static_cache_read(c,3,spans[2].offset,14,out,spans[0].bytes,&io,err,sizeof(err)));
    assert(!memcmp(out,raw+spans[2].offset,14));
    int cache_fd=open(cache_path,O_RDWR);assert(cache_fd>=0);
    k3sc_header good=c->header;uint64_t raw_payload=c->blocks[2].file_offset;
    uint64_t compressed_payload=c->blocks[0].file_offset;
    /* Cache stat changes fail closed even when already prepared. */
    uint8_t b;assert(read_all(cache_fd,&b,1,raw_payload));uint8_t corrupt=b^1;
    assert(write_all(cache_fd,&corrupt,1,raw_payload));
    assert(!k3_static_cache_read(c,1,spans[1].offset,spans[1].bytes,out2,spans[1].bytes,&io,err,sizeof(err)));
    k3_static_cache_release(c);c=NULL;
    /* Opening already-corrupted payload must fail on decoded CRC. */
    assert(k3_static_cache_open(cache_path,fd,spans,3,4,&c,err,sizeof(err)));
    assert(k3_static_cache_prepare(c,err,sizeof(err)));
    assert(!k3_static_cache_read(c,1,spans[1].offset,spans[1].bytes,out2,spans[1].bytes,&io,err,sizeof(err)));
    assert(strstr(err,"CRC"));k3_static_cache_release(c);c=NULL;
    assert(write_all(cache_fd,&b,1,raw_payload));
    /* Corrupted dominant-codec frame exercises error propagation and drains
     * other in-flight workers before returning the slot to its owner. */
    assert(read_all(cache_fd,&b,1,compressed_payload));corrupt=b^1;
    assert(write_all(cache_fd,&corrupt,1,compressed_payload));
    assert(k3_static_cache_open(cache_path,fd,spans,3,4,&c,err,sizeof(err)));
    assert(k3_static_cache_prepare(c,err,sizeof(err)));
    assert(!k3_static_cache_read(c,0,0,spans[0].bytes,out,spans[0].bytes,&io,err,sizeof(err)));
    assert(strstr(err,"zstd frame"));k3_static_cache_release(c);c=NULL;
    assert(write_all(cache_fd,&b,1,compressed_payload));
    k3sc_header bad=good;bad.block_count=UINT32_MAX;rewrite_header(cache_fd,&bad);
    assert(!k3_static_cache_open(cache_path,fd,spans,3,4,&c,err,sizeof(err)) && !c);
    assert(write_all(cache_fd,&good,sizeof(good),0));
    /* CRC-valid but structurally invalid block bounds. */
    uint8_t *table=malloc((size_t)good.table_bytes);assert(table);
    assert(read_all(cache_fd,table,(size_t)good.table_bytes,sizeof(good)));
    k3sc_block *bs=(k3sc_block *)(table+good.layer_count*sizeof(k3sc_layer));
    uint32_t saved=bs[0].decoded_bytes;bs[0].decoded_bytes=UINT32_MAX;
    bad=good;bad.table_crc32=crc32_bytes(table,(size_t)good.table_bytes);
    assert(write_all(cache_fd,table,(size_t)good.table_bytes,sizeof(good)));rewrite_header(cache_fd,&bad);
    assert(!k3_static_cache_open(cache_path,fd,spans,3,4,&c,err,sizeof(err)) && !c);
    bs[0].decoded_bytes=saved;
    assert(write_all(cache_fd,table,(size_t)good.table_bytes,sizeof(good)));assert(write_all(cache_fd,&good,sizeof(good),0));free(table);
    assert(ftruncate(cache_fd,(off_t)good.file_bytes-1)==0);
    assert(!k3_static_cache_open(cache_path,fd,spans,3,4,&c,err,sizeof(err)) && !c);
    assert(ftruncate(cache_fd,(off_t)good.file_bytes)==0);
    /* A modified source invalidates both existing and newly opened caches. */
    assert(k3_static_cache_open(cache_path,fd,spans,3,1,&c,err,sizeof(err)));
    assert(k3_static_cache_prepare(c,err,sizeof(err)));
    b=raw[0]^1;assert(write_all(fd,&b,1,0));
    assert(!k3_static_cache_read(c,0,0,spans[0].bytes,out,spans[0].bytes,&io,err,sizeof(err)));
    k3_static_cache_release(c);c=NULL;
    assert(!k3_static_cache_open(cache_path,fd,spans,3,4,&c,err,sizeof(err)) && !c);
    k3_static_cache_retain(NULL);k3_static_cache_release(NULL);
    free(out);free(out2);free(raw);close(fd);close(cache_fd);unlink(model_path);unlink(cache_path);
    printf("K3 static cache CPU fixture passed: CRC vector, compressed+RAW+EOF tail, "
           "4 workers, subset, exact read, serialized clones, corruption/bounds/stat, "
           "exclusive publish; workspace=%llu bytes\n",(unsigned long long)workspace);
    return 0;
}
#endif
