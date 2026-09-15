/* Manual, model-free two-host transport qualification. No GPU or GGUF access.
 * rank address port tcp|usb4stream device-or-dash */
#define DS4_ROCM_BUILD 1
#include "../ds4_tp.c"

int main(int argc, char **argv) {
    if (argc != 6 && argc != 8) return 2;
    int rank=atoi(argv[1]), port=atoi(argv[3]);
    if ((rank!=0 && rank!=1) || port<1 || port>65535) return 2;
    ds4_tp_options opt={.role=rank?DS4_TP_WORKER:DS4_TP_LEADER,
        .listen_host=argv[2],.leader_host=argv[2],.listen_port=port,.leader_port=port};
    if (!strcmp(argv[4],"tcp")) opt.transport=DS4_TP_TRANSPORT_TCP;
    else if (!strcmp(argv[4],"usb4stream")) {
        opt.transport=DS4_TP_TRANSPORT_USB4STREAM;opt.usb4stream_device=argv[5];
    } else if (!strcmp(argv[4],"rdma") && argc==8) {
        opt.transport=DS4_TP_TRANSPORT_RDMA;opt.rdma_device=argv[5];
        opt.rdma_port=atoi(argv[6]);opt.rdma_gid_index=atoi(argv[7]);opt.rdma_gid_index_set=true;
    } else return 2;
    ds4_tp_identity id={.gguf_bytes=1,.model_id=41,.n_layer=40,.n_embd=5120,
        .n_vocab=129280,.quant_bits=2,.ctx_size=65536};
    ds4_tp *tp=NULL;char err[512];
    if (!ds4_tp_create(&tp,&opt,&id,err,sizeof(err))) {
        fprintf(stderr,"create failed: %s\n",err);return 1;
    }
    uint32_t rows[]={1,2,8,32,65,256,2033,2048};
    const size_t capacity=2048*20480;
    unsigned char *out=malloc(capacity),*storage=malloc(capacity+2),*in=storage?storage+1:NULL;
    int ok=out&&storage;
    for (unsigned n=0;ok && n<sizeof(rows)/sizeof(*rows);++n) {
        size_t bytes=(size_t)rows[n]*20480;
        memset(storage,0xa5,capacity+2);
        for (size_t j=0;j<bytes;++j) out[j]=(unsigned char)(j*31+rank*73+n*7);
        ok=ds4_tp_big_gate_exchange(tp,39,n+1,out,in,bytes);
        if (ok) for (size_t j=0;j<bytes;++j)
            if (in[j]!=(unsigned char)(j*31+(1-rank)*73+n*7)) {
                fprintf(stderr,"mismatch rows=%u byte=%zu\n",rows[n],j);ok=0;break;
            }
        if (ok) ok=storage[0]==0xa5 && in[bytes]==0xa5;
        printf("rank=%d rows=%u bytes=%zu full_payload_canaries=%s\n",rank,rows[n],bytes,ok?"PASS":"FAIL");
        fflush(stdout);
    }
    free(out);free(storage);ds4_tp_free(tp);
    return ok?0:1;
}
