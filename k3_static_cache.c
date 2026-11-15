#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "k3_static_cache.h"
#include <stdarg.h>
#include <stdio.h>

static void fail(char *err, size_t cap, const char *fmt, ...) {
    if (!err || !cap) return;
    va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap);
}

#ifndef K3_STATIC_CACHE_ZSTD
int k3_static_cache_supported(void) { return 0; }
int k3_static_cache_open(const char *p, int fd, const k3_static_cache_span *s,
    uint32_t n, unsigned w, k3_static_cache **out, char *e, size_t cap) {
    (void)p; (void)fd; (void)s; (void)n; (void)w;
    if (out) *out = NULL;
    fail(e, cap, "K3 static cache unavailable: build with libzstd support"); return 0;
}
uint64_t k3_static_cache_workspace_bytes(const k3_static_cache *c) { (void)c; return 0; }
int k3_static_cache_prepare(k3_static_cache *c, char *e, size_t cap) {
    (void)c; fail(e, cap, "K3 static cache unavailable: build with libzstd support"); return 0;
}
int k3_static_cache_read(k3_static_cache *c, uint32_t l, uint64_t o, uint64_t n,
    void *d, uint64_t dc, uint64_t *io, char *e, size_t cap) {
    (void)c; (void)l; (void)o; (void)n; (void)d; (void)dc; if (io) *io=0;
    fail(e, cap, "K3 static cache unavailable: build with libzstd support"); return 0;
}
void k3_static_cache_retain(k3_static_cache *c) { (void)c; }
void k3_static_cache_release(k3_static_cache *c) { (void)c; }
int k3_static_cache_build(const char *p, int fd, const k3_static_cache_span *s,
    uint32_t n, char *e, size_t cap) {
    (void)p; (void)fd; (void)s; (void)n;
    fail(e, cap, "K3 static cache unavailable: build with libzstd support"); return 0;
}
#else
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>
#include "k3_static_cache_format.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

int k3_static_cache_supported(void) { return 1; }

#define K3SC_WORKERS 4u
#define K3SC_STACK_BYTES (512u * 1024u)
#ifdef K3_STATIC_CACHE_TEST
static int k3sc_test_fail_worker=-1;
#endif

typedef struct {
    struct k3_static_cache *owner;
    pthread_t thread;
    uint8_t *input, *planes, *stack;
    ZSTD_DCtx *dctx;
} k3sc_worker;

struct k3_static_cache {
    atomic_uint refs;
    int fd, model_fd;
    struct stat cache_stat;
    k3sc_header header;
    void *table;
    k3sc_layer *layers;
    k3sc_block *blocks;
    unsigned workers, created, prepared;
    size_t dctx_bytes, worker_bytes, arena_bytes;
    uint8_t *arena;
    pthread_mutex_t api_mutex, pool_mutex;
    pthread_cond_t work_cv, done_cv;
    unsigned generation, pending, stop, failed;
    uint32_t next_block, end_block;
    uint64_t destination_offset, physical_bytes;
    uint8_t *destination;
    char job_error[256];
    k3sc_worker worker[K3SC_WORKERS];
};

static int little_endian(void) { const uint16_t x=1; return *(const uint8_t *)&x; }
static uint64_t align_up(uint64_t n) {
    return (n + K3SC_ALIGNMENT - 1u) & ~(uint64_t)(K3SC_ALIGNMENT - 1u);
}
static int same_stat(const struct stat *a, const struct stat *b) {
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
        a->st_size == b->st_size &&
        a->st_mtimespec.tv_sec == b->st_mtimespec.tv_sec &&
        a->st_mtimespec.tv_nsec == b->st_mtimespec.tv_nsec &&
        a->st_ctimespec.tv_sec == b->st_ctimespec.tv_sec &&
        a->st_ctimespec.tv_nsec == b->st_ctimespec.tv_nsec;
}
static void save_model_stat(k3sc_header *h, const struct stat *s) {
    h->model_dev=(uint64_t)s->st_dev; h->model_ino=(uint64_t)s->st_ino;
    h->model_bytes=(uint64_t)s->st_size;
    h->model_mtime_sec=s->st_mtimespec.tv_sec; h->model_mtime_nsec=s->st_mtimespec.tv_nsec;
    h->model_ctime_sec=s->st_ctimespec.tv_sec; h->model_ctime_nsec=s->st_ctimespec.tv_nsec;
}
static int model_matches(const k3sc_header *h, int fd) {
    struct stat s;
    if (fstat(fd,&s) || !S_ISREG(s.st_mode) || s.st_size < 0) return 0;
    return h->model_dev==(uint64_t)s.st_dev && h->model_ino==(uint64_t)s.st_ino &&
        h->model_bytes==(uint64_t)s.st_size && h->model_mtime_sec==s.st_mtimespec.tv_sec &&
        h->model_mtime_nsec==(uint64_t)s.st_mtimespec.tv_nsec &&
        h->model_ctime_sec==s.st_ctimespec.tv_sec &&
        h->model_ctime_nsec==(uint64_t)s.st_ctimespec.tv_nsec;
}
static int read_all(int fd, void *p_, size_t n, uint64_t off) {
    uint8_t *p=p_;
    if (off > INT64_MAX || n > (uint64_t)INT64_MAX-off) return 0;
    while(n) {
        ssize_t z=pread(fd,p,n,(off_t)off);
        if(z<0 && errno==EINTR) continue;
        if(z<=0) return 0;
        p+=z; n-=(size_t)z; off+=(uint64_t)z;
    }
    return 1;
}
static int write_all(int fd, const void *p_, size_t n, uint64_t off) {
    const uint8_t *p=p_;
    if (off > INT64_MAX || n > (uint64_t)INT64_MAX-off) return 0;
    while(n) {
        ssize_t z=pwrite(fd,p,n,(off_t)off);
        if(z<0 && errno==EINTR) continue;
        if(z<=0) return 0;
        p+=z; n-=(size_t)z; off+=(uint64_t)z;
    }
    return 1;
}

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_CRC32)
static uint32_t crc_table[8][256];
static pthread_once_t crc_once=PTHREAD_ONCE_INIT;
static void crc_init(void) {
    for(unsigned i=0;i<256;i++) {
        uint32_t c=i;
        for(unsigned j=0;j<8;j++) c=(c>>1) ^ (0xedb88320u & (0u-(c&1)));
        crc_table[0][i]=c;
    }
    for(unsigned k=1;k<8;k++) for(unsigned i=0;i<256;i++) {
        uint32_t c=crc_table[k-1][i]; crc_table[k][i]=(c>>8)^crc_table[0][c&255];
    }
}
#endif
static uint32_t crc32_bytes(const void *p_, size_t n) {
    const uint8_t *p=p_; uint32_t c=~0u;
#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
    while(n>=8) { uint64_t v; memcpy(&v,p,8); c=__crc32d(c,v); p+=8;n-=8; }
    while(n--) c=__crc32b(c,*p++);
#else
    pthread_once(&crc_once,crc_init);
    while(n>=8) {
        uint32_t a,b; memcpy(&a,p,4);memcpy(&b,p+4,4);a^=c;
        c=crc_table[7][a&255]^crc_table[6][(a>>8)&255]^crc_table[5][(a>>16)&255]^crc_table[4][a>>24]^
          crc_table[3][b&255]^crc_table[2][(b>>8)&255]^crc_table[1][(b>>16)&255]^crc_table[0][b>>24];
        p+=8;n-=8;
    }
    while(n--) c=(c>>8)^crc_table[0][(c^*p++)&255];
#endif
    return ~c;
}
static void shuffle(uint8_t *restrict out,const uint8_t *restrict in,size_t n) {
    size_t half=n/2;
    for(size_t i=0;i<half;i++){out[i]=in[2*i];out[half+i]=in[2*i+1];}
}
static void unshuffle(uint8_t *restrict out,const uint8_t *restrict in,size_t n) {
    size_t half=n/2;
    for(size_t i=0;i<half;i++){out[2*i]=in[i];out[2*i+1]=in[half+i];}
}
static uint32_t header_crc(k3sc_header h) {h.header_crc32=0;return crc32_bytes(&h,sizeof(h));}

static int spans_valid(const k3_static_cache_span *s,uint32_t count,uint64_t filesize) {
    if(!s || !count || count>K3SC_MAX_LAYERS) return 0;
    for(uint32_t i=0;i<count;i++) {
        if(s[i].layer>=K3SC_MAX_LAYERS || (i && s[i].layer<=s[i-1].layer) ||
           !s[i].bytes || (s[i].bytes&1) || (s[i].offset&(K3SC_ALIGNMENT-1)) ||
           s[i].offset>filesize || s[i].bytes>filesize-s[i].offset ||
           (s[i].bytes%K3SC_ALIGNMENT && s[i].offset+s[i].bytes!=filesize)) return 0;
    }
    return 1;
}
static const k3sc_layer *find_layer(const k3_static_cache *c,uint32_t layer) {
    for(uint32_t i=0;i<c->header.layer_count;i++) if(c->layers[i].layer==layer)return &c->layers[i];
    return NULL;
}

int k3_static_cache_open(const char *path,int model_fd,const k3_static_cache_span *spans,
    uint32_t span_count,unsigned workers,k3_static_cache **out,char *err,size_t cap) {
    if(out)*out=NULL;
    if(!path || !out || !little_endian() || workers<1 || workers>K3SC_WORKERS) {
        fail(err,cap,"invalid K3 static cache open arguments");return 0;
    }
    /* Nonblocking open prevents an invalid/replaced FIFO from hanging before
     * fstat can reject it. O_NONBLOCK does not change regular-file reads. */
    int fd=open(path,O_RDONLY|O_CLOEXEC|O_NONBLOCK); k3sc_header h; struct stat st,after;
    if(fd<0 || fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size<(off_t)sizeof(h) ||
       !read_all(fd,&h,sizeof(h),0)) {
        if(fd>=0)close(fd); fail(err,cap,"cannot read K3 static cache header: %s",strerror(errno));return 0;
    }
    if(memcmp(h.magic,K3SC_MAGIC,8) || h.version!=K3SC_VERSION || h.header_bytes!=sizeof(h) ||
       h.reserved || h.header_crc32!=header_crc(h) || !h.layer_count || h.layer_count>K3SC_MAX_LAYERS ||
       !h.block_count || h.block_count>K3SC_MAX_BLOCKS || h.block_bytes!=K3SC_BLOCK_BYTES ||
       h.alignment!=K3SC_ALIGNMENT || h.table_bytes!=(uint64_t)h.layer_count*sizeof(k3sc_layer)+(uint64_t)h.block_count*sizeof(k3sc_block) ||
       h.data_offset!=align_up(sizeof(h)+h.table_bytes) || h.file_bytes!=(uint64_t)st.st_size ||
       h.file_bytes<h.data_offset || !model_matches(&h,model_fd) ||
       !spans_valid(spans,span_count,h.model_bytes)) {
        close(fd);fail(err,cap,"K3 static cache header/model identity mismatch");return 0;
    }
    k3_static_cache *c=calloc(1,sizeof(*c));
    if(!c){close(fd);fail(err,cap,"out of memory opening K3 static cache");return 0;}
    c->fd=fd;c->model_fd=-1;c->header=h;c->cache_stat=st;c->workers=workers;
    c->table=malloc((size_t)h.table_bytes);
    if(!c->table || !read_all(fd,c->table,(size_t)h.table_bytes,sizeof(h)) ||
       crc32_bytes(c->table,(size_t)h.table_bytes)!=h.table_crc32) goto invalid;
    c->layers=c->table;c->blocks=(k3sc_block *)((uint8_t *)c->table+h.layer_count*sizeof(k3sc_layer));
    uint32_t cursor=0; uint64_t disk_cursor=h.data_offset;
    for(uint32_t i=0;i<h.layer_count;i++) {
        const k3sc_layer *l=&c->layers[i];
        if(l->layer>=K3SC_MAX_LAYERS || (i && l->layer<=c->layers[i-1].layer) || l->reserved ||
           l->first_block!=cursor || !l->block_count || l->block_count>h.block_count-cursor ||
           !l->bytes || (l->bytes&1) || (l->offset&(K3SC_ALIGNMENT-1)) ||
           l->offset>h.model_bytes || l->bytes>h.model_bytes-l->offset ||
           (l->bytes%K3SC_ALIGNMENT && l->offset+l->bytes!=h.model_bytes) ||
           l->block_count!=(l->bytes+K3SC_BLOCK_BYTES-1)/K3SC_BLOCK_BYTES) goto invalid;
        uint64_t decoded=l->offset;
        for(uint32_t j=0;j<l->block_count;j++,cursor++) {
            const k3sc_block *b=&c->blocks[cursor];
            uint64_t remain=l->offset+l->bytes-decoded;
            uint32_t expected=remain<K3SC_BLOCK_BYTES?(uint32_t)remain:K3SC_BLOCK_BYTES;
            if(b->reserved || b->decoded_offset!=decoded || b->decoded_bytes!=expected ||
               !b->stored_bytes || b->stored_bytes>b->decoded_bytes ||
               b->padded_bytes!=align_up(b->stored_bytes) || b->padded_bytes>K3SC_BLOCK_BYTES ||
               b->file_offset!=disk_cursor || b->file_offset>h.file_bytes || b->padded_bytes>h.file_bytes-b->file_offset ||
               (b->codec!=K3SC_RAW && b->codec!=K3SC_BF16_PLANES_ZSTD) ||
               (b->codec==K3SC_RAW && b->stored_bytes!=b->decoded_bytes) ||
               (b->codec==K3SC_BF16_PLANES_ZSTD && b->stored_bytes>=b->decoded_bytes)) goto invalid;
            decoded+=b->decoded_bytes; disk_cursor+=b->padded_bytes;
        }
    }
    if(cursor!=h.block_count || disk_cursor!=h.file_bytes)goto invalid;
    for(uint32_t i=0;i<span_count;i++) {
        const k3sc_layer *l=find_layer(c,spans[i].layer);
        if(!l || l->offset!=spans[i].offset || l->bytes!=spans[i].bytes)goto invalid;
    }
    if(fstat(fd,&after) || !same_stat(&st,&after))goto invalid;
    c->model_fd=dup(model_fd);if(c->model_fd<0)goto invalid;
    c->dctx_bytes=(size_t)align_up(ZSTD_estimateDCtxSize());
    c->worker_bytes=2u*(size_t)K3SC_BLOCK_BYTES+c->dctx_bytes+K3SC_STACK_BYTES;
    c->arena_bytes=c->worker_bytes*workers;
    /* All mutexes/conditions are initialized before publishing ownership. */
    if(pthread_mutex_init(&c->api_mutex,NULL))goto invalid;
    if(pthread_mutex_init(&c->pool_mutex,NULL)){pthread_mutex_destroy(&c->api_mutex);goto invalid;}
    if(pthread_cond_init(&c->work_cv,NULL)){pthread_mutex_destroy(&c->pool_mutex);pthread_mutex_destroy(&c->api_mutex);goto invalid;}
    if(pthread_cond_init(&c->done_cv,NULL)){pthread_cond_destroy(&c->work_cv);pthread_mutex_destroy(&c->pool_mutex);pthread_mutex_destroy(&c->api_mutex);goto invalid;}
    atomic_init(&c->refs,1);
#ifdef F_NOCACHE
    (void)fcntl(fd,F_NOCACHE,1);
#endif
    *out=c;return 1;
invalid:
    if(c->model_fd>=0)close(c->model_fd);
    close(c->fd);free(c->table);free(c);
    fail(err,cap,"K3 static cache index invalid, changed, or allocation failed");return 0;
}

uint64_t k3_static_cache_workspace_bytes(const k3_static_cache *c) {
    return c?align_up(sizeof(*c))+align_up(c->header.table_bytes)+c->arena_bytes+
        c->workers*64ull*1024u:0;
}
static int decode_block(k3sc_worker *w,const k3sc_block *b,char *err,size_t cap) {
    k3_static_cache *c=w->owner;
    if(!read_all(c->fd,w->input,b->padded_bytes,b->file_offset)) {
        fail(err,cap,"K3 static cache short read at %llu",(unsigned long long)b->file_offset);return 0;
    }
    uint8_t *out=c->destination+(b->decoded_offset-c->destination_offset);
    if(b->codec==K3SC_RAW)memcpy(out,w->input,b->decoded_bytes);
    else {
        ZSTD_frameHeader frame;
        size_t z=ZSTD_getFrameHeader(&frame,w->input,b->stored_bytes);
        if(z || frame.frameContentSize!=b->decoded_bytes || frame.windowSize>K3SC_BLOCK_BYTES || frame.dictID ||
           ZSTD_findFrameCompressedSize(w->input,b->stored_bytes)!=b->stored_bytes) {
            fail(err,cap,"K3 static cache invalid bounded zstd frame");return 0;
        }
        z=ZSTD_decompressDCtx(w->dctx,w->planes,b->decoded_bytes,w->input,b->stored_bytes);
        if(ZSTD_isError(z) || z!=b->decoded_bytes) {
            fail(err,cap,"K3 static cache decode failed: %s",ZSTD_getErrorName(z));return 0;
        }
        unshuffle(out,w->planes,b->decoded_bytes);
    }
    if(crc32_bytes(out,b->decoded_bytes)!=b->crc32) {
        fail(err,cap,"K3 static cache decoded CRC mismatch");return 0;
    }
    return 1;
}
static void *worker_main(void *arg) {
    k3sc_worker *w=arg;k3_static_cache *c=w->owner;unsigned seen=0;
    pthread_mutex_lock(&c->pool_mutex);
    for(;;) {
        while(!c->stop && seen==c->generation)pthread_cond_wait(&c->work_cv,&c->pool_mutex);
        if(c->stop)break;
        seen=c->generation;
        while(!c->failed && c->next_block<c->end_block) {
            const k3sc_block *b=&c->blocks[c->next_block++];
            pthread_mutex_unlock(&c->pool_mutex);
            char err[256]={0};int ok=decode_block(w,b,err,sizeof(err));
            pthread_mutex_lock(&c->pool_mutex);
            c->physical_bytes+=b->padded_bytes;
            if(!ok && !c->failed){c->failed=1;snprintf(c->job_error,sizeof(c->job_error),"%s",err);}
        }
        if(--c->pending==0)pthread_cond_signal(&c->done_cv);
    }
    pthread_mutex_unlock(&c->pool_mutex);return NULL;
}
static void stop_workers(k3_static_cache *c) {
    pthread_mutex_lock(&c->pool_mutex);c->stop=1;pthread_cond_broadcast(&c->work_cv);pthread_mutex_unlock(&c->pool_mutex);
    for(unsigned i=0;i<c->created;i++)pthread_join(c->worker[i].thread,NULL);
    c->created=0;c->prepared=0;
    if(c->arena){munmap(c->arena,c->arena_bytes);c->arena=NULL;}
}
int k3_static_cache_prepare(k3_static_cache *c,char *err,size_t cap) {
    if(!c){fail(err,cap,"null K3 static cache");return 0;}
    pthread_mutex_lock(&c->api_mutex);
    if(c->prepared){pthread_mutex_unlock(&c->api_mutex);return 1;}
    c->stop=0;c->generation=0;
    c->arena=mmap(NULL,c->arena_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    if(c->arena==MAP_FAILED){c->arena=NULL;fail(err,cap,"K3 static cache workspace allocation failed");goto error;}
    for(unsigned i=0;i<c->workers;i++) {
        k3sc_worker *w=&c->worker[i];w->owner=c;
        w->input=c->arena+i*c->worker_bytes;w->planes=w->input+K3SC_BLOCK_BYTES;
        w->dctx=ZSTD_initStaticDCtx(w->planes+K3SC_BLOCK_BYTES,c->dctx_bytes);
        w->stack=w->planes+K3SC_BLOCK_BYTES+c->dctx_bytes;
        if(!w->dctx){fail(err,cap,"K3 static cache fixed zstd workspace rejected");goto error;}
        pthread_attr_t a;int rc=pthread_attr_init(&a);
        if(rc){fail(err,cap,"K3 static cache worker attributes failed");goto error;}
        rc=pthread_attr_setstack(&a,w->stack,K3SC_STACK_BYTES);
#ifdef K3_STATIC_CACHE_TEST
        if((int)i==k3sc_test_fail_worker)rc=EAGAIN;
#endif
        if(!rc)rc=pthread_create(&w->thread,&a,worker_main,w);
        pthread_attr_destroy(&a);
        if(rc){fail(err,cap,"K3 static cache worker creation failed: %s",strerror(rc));goto error;}
        c->created++;
    }
    c->prepared=1;pthread_mutex_unlock(&c->api_mutex);return 1;
error:
    stop_workers(c);pthread_mutex_unlock(&c->api_mutex);return 0;
}
int k3_static_cache_read(k3_static_cache *c,uint32_t layer,uint64_t offset,uint64_t bytes,
    void *dst,uint64_t dst_capacity,uint64_t *physical_bytes,char *err,size_t cap) {
    if(physical_bytes)*physical_bytes=0;
    if(!c || !dst){fail(err,cap,"invalid K3 static cache read");return 0;}
    pthread_mutex_lock(&c->api_mutex);
    const k3sc_layer *l=find_layer(c,layer);struct stat st;
    if(!c->prepared || !l || offset!=l->offset || bytes!=l->bytes || dst_capacity<bytes ||
       !model_matches(&c->header,c->model_fd) || fstat(c->fd,&st) || !same_stat(&c->cache_stat,&st)) {
        fail(err,cap,"K3 static cache read span/state/model identity mismatch");goto error;
    }
    pthread_mutex_lock(&c->pool_mutex);
    c->destination=dst;c->destination_offset=offset;c->next_block=l->first_block;
    c->end_block=l->first_block+l->block_count;c->pending=c->workers;c->failed=0;
    c->physical_bytes=0;c->job_error[0]=0;c->generation++;
    pthread_cond_broadcast(&c->work_cv);
    while(c->pending)pthread_cond_wait(&c->done_cv,&c->pool_mutex);
    if(physical_bytes)*physical_bytes=c->physical_bytes;
    int ok=!c->failed;
    if(!ok)fail(err,cap,"%s",c->job_error);
    c->destination=NULL;pthread_mutex_unlock(&c->pool_mutex);
    if(!ok)goto error;
    if(!model_matches(&c->header,c->model_fd) || fstat(c->fd,&st) || !same_stat(&c->cache_stat,&st)) {
        fail(err,cap,"K3 static cache source changed during layer read");goto error;
    }
    pthread_mutex_unlock(&c->api_mutex);return 1;
error:
    pthread_mutex_unlock(&c->api_mutex);return 0;
}
void k3_static_cache_retain(k3_static_cache *c) {if(c)atomic_fetch_add(&c->refs,1);}
void k3_static_cache_release(k3_static_cache *c) {
    if(!c || atomic_fetch_sub(&c->refs,1)!=1)return;
    stop_workers(c);close(c->fd);close(c->model_fd);free(c->table);
    pthread_cond_destroy(&c->work_cv);pthread_cond_destroy(&c->done_cv);
    pthread_mutex_destroy(&c->api_mutex);pthread_mutex_destroy(&c->pool_mutex);free(c);
}

int k3_static_cache_build(const char *path,int model_fd,const k3_static_cache_span *spans,
    uint32_t count,char *err,size_t cap) {
    struct stat source,end,existing;
    if(!path || strlen(path)>PATH_MAX-32u || !little_endian() || fstat(model_fd,&source) || !S_ISREG(source.st_mode) || source.st_size<=0 ||
       !spans_valid(spans,count,(uint64_t)source.st_size)) {
        fail(err,cap,"invalid K3 static cache build spans/model");return 0;
    }
    if(lstat(path,&existing)==0 || errno!=ENOENT) {
        fail(err,cap,"K3 static cache target already exists or is inaccessible");return 0;
    }
    k3sc_header h={0};memcpy(h.magic,K3SC_MAGIC,8);h.version=K3SC_VERSION;h.header_bytes=sizeof(h);
    save_model_stat(&h,&source);h.layer_count=count;h.block_bytes=K3SC_BLOCK_BYTES;h.alignment=K3SC_ALIGNMENT;
    uint64_t blocks=0;
    for(uint32_t i=0;i<count;i++)blocks+=(spans[i].bytes+K3SC_BLOCK_BYTES-1)/K3SC_BLOCK_BYTES;
    if(blocks>K3SC_MAX_BLOCKS){fail(err,cap,"too many K3 static cache blocks");return 0;}
    h.block_count=(uint32_t)blocks;h.table_bytes=count*sizeof(k3sc_layer)+blocks*sizeof(k3sc_block);
    h.data_offset=align_up(sizeof(h)+h.table_bytes);
    uint8_t *table=calloc(1,(size_t)h.table_bytes),*raw=NULL,*planes=NULL,*encoded=NULL,*decoded=NULL,*verify=NULL;
    void *ccmem=NULL,*dcmem=NULL;char *tmp=NULL;int fd=-1,ok=0;
    const size_t ccbytes=ZSTD_estimateCCtxSize(1),dcbytes=ZSTD_estimateDCtxSize();
    raw=malloc(K3SC_BLOCK_BYTES);planes=malloc(K3SC_BLOCK_BYTES);encoded=malloc(ZSTD_compressBound(K3SC_BLOCK_BYTES)+K3SC_ALIGNMENT);
    decoded=malloc(K3SC_BLOCK_BYTES);verify=malloc(K3SC_BLOCK_BYTES);ccmem=malloc(ccbytes);dcmem=malloc(dcbytes);
    tmp=malloc(strlen(path)+32);
    if(tmp)tmp[0]=0;
    if(!table||!raw||!planes||!encoded||!decoded||!verify||!ccmem||!dcmem||!tmp) {
        fail(err,cap,"K3 static cache builder allocation failed");goto done;
    }
    ZSTD_CCtx *cc=ZSTD_initStaticCCtx(ccmem,ccbytes);ZSTD_DCtx *dc=ZSTD_initStaticDCtx(dcmem,dcbytes);
    if(!cc||!dc){fail(err,cap,"K3 static cache builder fixed zstd workspace failed");goto done;}
    snprintf(tmp,strlen(path)+32,"%s.tmp.XXXXXX",path);fd=mkstemp(tmp);
    if(fd<0){fail(err,cap,"cannot create K3 static cache temporary file: %s",strerror(errno));goto done;}
    if(fchmod(fd,0600)){fail(err,cap,"cannot protect K3 static cache temp");goto done;}
    k3sc_layer *ls=(k3sc_layer *)table;k3sc_block *bs=(k3sc_block *)(table+count*sizeof(*ls));
    uint32_t cursor=0;uint64_t disk=h.data_offset,verified_bytes=0;
    for(uint32_t i=0;i<count;i++) {
        if(!model_matches(&h,model_fd)){fail(err,cap,"source changed during K3 static cache build");goto done;}
        ls[i]=(k3sc_layer){.layer=spans[i].layer,.first_block=cursor,
            .block_count=(uint32_t)((spans[i].bytes+K3SC_BLOCK_BYTES-1)/K3SC_BLOCK_BYTES),
            .offset=spans[i].offset,.bytes=spans[i].bytes};
        uint64_t off=spans[i].offset,remaining=spans[i].bytes;
        while(remaining) {
            uint32_t n=remaining<K3SC_BLOCK_BYTES?(uint32_t)remaining:K3SC_BLOCK_BYTES;
            if(!read_all(model_fd,raw,n,off)){fail(err,cap,"K3 static cache source short read");goto done;}
            shuffle(planes,raw,n);
            size_t z=ZSTD_compressCCtx(cc,encoded,ZSTD_compressBound(n),planes,n,1);
            if(ZSTD_isError(z)){fail(err,cap,"K3 static cache compression failed: %s",ZSTD_getErrorName(z));goto done;}
            size_t dz=ZSTD_decompressDCtx(dc,decoded,n,encoded,z);
            if(ZSTD_isError(dz)||dz!=n){fail(err,cap,"K3 static cache roundtrip decode failed");goto done;}
            unshuffle(verify,decoded,n);
            if(memcmp(verify,raw,n)){fail(err,cap,"K3 static cache roundtrip byte mismatch");goto done;}
            uint32_t codec=K3SC_BF16_PLANES_ZSTD;
            if(align_up(z)>=align_up(n)){codec=K3SC_RAW;z=n;memcpy(encoded,raw,n);}
            uint32_t padded=(uint32_t)align_up(z);
            memset(encoded+z,0,padded-z);
            bs[cursor]=(k3sc_block){.file_offset=disk,.decoded_offset=off,.stored_bytes=(uint32_t)z,
                .padded_bytes=padded,.decoded_bytes=n,.crc32=crc32_bytes(raw,n),.codec=codec};
            if(!write_all(fd,encoded,padded,disk)){fail(err,cap,"K3 static cache write failed: %s",strerror(errno));goto done;}
            cursor++;verified_bytes+=n;disk+=padded;off+=n;remaining-=n;
        }
        fprintf(stderr,"k3-static-cache: layer=%u verified_blocks=%u verified_bytes=%llu cache_bytes=%llu\n",
            spans[i].layer,cursor,(unsigned long long)verified_bytes,(unsigned long long)disk);
    }
    h.file_bytes=disk;h.table_crc32=crc32_bytes(table,(size_t)h.table_bytes);h.header_crc32=header_crc(h);
    if(fstat(model_fd,&end)||!same_stat(&source,&end)||!write_all(fd,table,(size_t)h.table_bytes,sizeof(h)) ||
       !write_all(fd,&h,sizeof(h),0)||ftruncate(fd,(off_t)disk)||fsync(fd)) {
        fail(err,cap,"K3 static cache final validation/sync failed");goto done;
    }
    if(!model_matches(&h,model_fd)) {
        fail(err,cap,"source changed before K3 static cache publication");goto done;
    }
    /* Atomic publication without replacing any existing file, including a race. */
#if defined(__APPLE__) && defined(RENAME_EXCL)
    if(renamex_np(tmp,path,RENAME_EXCL)) {
#else
    if(link(tmp,path)) {
#endif
        fail(err,cap,"K3 static cache exclusive publish failed: %s",strerror(errno));goto done;
    }
    (void)unlink(tmp);tmp[0]=0;ok=1;
    fprintf(stderr,"k3-static-cache: build verified_blocks=%u verified_bytes=%llu layers=%u cache_bytes=%llu roundtrip_exact=yes\n",
        cursor,(unsigned long long)verified_bytes,count,(unsigned long long)disk);
done:
    if(fd>=0)close(fd);if(tmp&&tmp[0])unlink(tmp);
    free(tmp);free(table);free(raw);free(planes);free(encoded);free(decoded);free(verify);free(ccmem);free(dcmem);
    return ok;
}
#endif
