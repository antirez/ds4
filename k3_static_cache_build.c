#define _DARWIN_C_SOURCE
#include "k3_static_cache.h"
#include "k3_static_cache_format.h"
#include "k3_model.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int same_source(const struct stat *a,const struct stat *b) {
    return a->st_dev==b->st_dev && a->st_ino==b->st_ino && a->st_size==b->st_size &&
        a->st_mtimespec.tv_sec==b->st_mtimespec.tv_sec && a->st_mtimespec.tv_nsec==b->st_mtimespec.tv_nsec &&
        a->st_ctimespec.tv_sec==b->st_ctimespec.tv_sec && a->st_ctimespec.tv_nsec==b->st_ctimespec.tv_nsec;
}

int main(int argc,char **argv) {
    if(argc!=3){fprintf(stderr,"usage: %s MODEL.gguf NEW_CACHE.k3sc\n",argv[0]);return 2;}
    char err[512]={0};k3_model *model=NULL;k3_global_weights global={0};
    k3_static_cache_span spans[K3_N_LAYER];int ok=0;
    if(!k3_model_open(&model,argv[1],err,sizeof(err)) || !model->gguf ||
       !k3_model_open_globals(model,&global,err,sizeof(err)))goto done;
    struct stat metadata_stat;
    if(fstat(k3_st_fd(global.shard),&metadata_stat)) {
        snprintf(err,sizeof(err),"cannot stat metadata source");goto done;
    }
    const uint64_t file_bytes=k3_st_file_size(global.shard);
    for(uint32_t i=0;i<K3_N_LAYER;i++) {
        k3_layer *layer=NULL;
        if(!k3_model_open_layer(model,i,&layer,err,sizeof(err)))goto done;
        uint64_t start=layer->static_offset,bytes=layer->static_size;
        if(!bytes || start>file_bytes || bytes>file_bytes-start) {
            k3_layer_close(layer);snprintf(err,sizeof(err),"invalid static span at layer %u",i);goto done;
        }
        uint64_t lo=start&~(uint64_t)(K3SC_ALIGNMENT-1u),hi=start+bytes;
        if(hi<=UINT64_MAX-(K3SC_ALIGNMENT-1u))hi=(hi+K3SC_ALIGNMENT-1u)&~(uint64_t)(K3SC_ALIGNMENT-1u);
        else hi=file_bytes;
        if(hi>file_bytes)hi=file_bytes;
        spans[i]=(k3_static_cache_span){.layer=i,.offset=lo,.bytes=hi-lo};
        k3_layer_close(layer);
    }
    /* A separately opened source descriptor avoids changing model API flags. */
    int fd=open(argv[1],O_RDONLY|O_CLOEXEC);
    if(fd<0){snprintf(err,sizeof(err),"cannot open source: %s",strerror(errno));goto done;}
    struct stat build_stat;
    if(fstat(fd,&build_stat) || !same_source(&metadata_stat,&build_stat)) {
        close(fd);snprintf(err,sizeof(err),"source identity changed while parsing static spans");goto done;
    }
#ifdef F_NOCACHE
    (void)fcntl(fd,F_NOCACHE,1);
#endif
    ok=k3_static_cache_build(argv[2],fd,spans,K3_N_LAYER,err,sizeof(err));
    close(fd);
done:
    k3_global_weights_close(&global);k3_model_close(model);
    if(!ok)fprintf(stderr,"k3-static-cache: %s\n",err[0]?err:"GGUF model required");
    return ok?0:1;
}
