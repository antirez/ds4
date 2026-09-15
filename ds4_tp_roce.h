/* Optional Linux RC SEND/RECV adapter. Setup symbols are loaded lazily;
 * posting/polling use the verbs provider's public inline dispatch. Buffers
 * are ordinary aligned host memory pinned once by ibv_reg_mr, not GPU MR. */
#ifdef DS4_TP_HAVE_ROCE
#define TP_RC_WINDOW 4u
#define TP_RC_CHUNK (2u * 1024u * 1024u)
struct ds4_tp_roce {
    void *lib;
    struct ibv_device **(*get_device_list)(int *);
    void (*free_device_list)(struct ibv_device **);
    const char *(*get_device_name)(struct ibv_device *);
    struct ibv_context *(*open_device)(struct ibv_device *);
    int (*close_device)(struct ibv_context *);
    int (*query_device)(struct ibv_context *, struct ibv_device_attr *);
    int (*query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
    int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *);
    int (*dealloc_pd)(struct ibv_pd *);
    struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int);
    int (*destroy_cq)(struct ibv_cq *);
    struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
    int (*destroy_qp)(struct ibv_qp *);
    int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
    int (*dereg_mr)(struct ibv_mr *);
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *tx_mr, *rx_mr;
    uint8_t *tx, *rx;
    union ibv_gid gid;
    uint32_t psn;
    enum ibv_mtu mtu;
    uint64_t window_id;
    uint32_t lengths[TP_RC_WINDOW];
    unsigned posted;
};
static void tp_roce_close(ds4_tp *tp) {
    struct ds4_tp_roce *r = tp->roce;
    if (!r) return;
    /* Destroying the QP ends DMA references before deregistration/free. */
    if (r->qp && r->destroy_qp(r->qp)) {
        /* A provider that cannot destroy a QP may still hold DMA references.
         * Quarantine its registered buffers instead of freeing live memory. */
        fprintf(stderr,"ds4-tp: RoCE QP destruction failed; retaining registered buffers\n");
        tp->roce = NULL; return;
    }
    if (r->tx_mr) r->dereg_mr(r->tx_mr);
    if (r->rx_mr) r->dereg_mr(r->rx_mr);
    free(r->tx); free(r->rx);
    if (r->cq) r->destroy_cq(r->cq);
    if (r->pd) r->dealloc_pd(r->pd);
    if (r->ctx) r->close_device(r->ctx);
    if (r->lib) dlclose(r->lib);
    free(r); tp->roce = NULL;
}
static int tp_roce_probe(ds4_tp *tp, char *err, size_t errlen) {
    if (!tp->opt.rdma_device || !*tp->opt.rdma_device || !tp->opt.rdma_gid_index_set) {
        tp_set_err(err,errlen,"Linux RoCE requires --rdma-device and --rdma-gid-index"); return 0;
    }
    struct ds4_tp_roce *r = calloc(1,sizeof(*r));
    if (!r) return 0;
    tp->roce = r;
    r->lib = dlopen("libibverbs.so.1",RTLD_NOW|RTLD_LOCAL);
    if (!r->lib) { tp_set_err(err,errlen,"RoCE: libibverbs.so.1 unavailable: %s",dlerror()); goto fail; }
#define RC_LOAD(name) do { *(void **)(&r->name)=dlsym(r->lib,"ibv_" #name); if(!r->name) { tp_set_err(err,errlen,"RoCE: missing ibv_%s",#name); goto fail; } } while(0)
    RC_LOAD(get_device_list); RC_LOAD(free_device_list); RC_LOAD(get_device_name);
    RC_LOAD(open_device); RC_LOAD(close_device); RC_LOAD(query_device);
    RC_LOAD(query_port); RC_LOAD(query_gid); RC_LOAD(alloc_pd); RC_LOAD(dealloc_pd);
    RC_LOAD(create_cq); RC_LOAD(destroy_cq); RC_LOAD(create_qp); RC_LOAD(destroy_qp);
    RC_LOAD(modify_qp); RC_LOAD(reg_mr); RC_LOAD(dereg_mr);
#undef RC_LOAD
    int count=0;
    struct ibv_device **devices=r->get_device_list(&count);
    if (!devices) goto syserr;
    for(int i=0;i<count;i++) if(!strcmp(r->get_device_name(devices[i]),tp->opt.rdma_device)) {
        r->ctx=r->open_device(devices[i]); break;
    }
    r->free_device_list(devices);
    if(!r->ctx) { tp_set_err(err,errlen,"RoCE: device %s unavailable",tp->opt.rdma_device); goto fail; }
    struct ibv_device_attr dev={0}; struct ibv_port_attr port={0};
    const unsigned portnum=tp->opt.rdma_port ? (unsigned)tp->opt.rdma_port : 1u;
    if(r->query_device(r->ctx,&dev) || portnum>dev.phys_port_cnt ||
       r->query_port(r->ctx,(uint8_t)portnum,&port)) goto syserr;
    if(port.state!=IBV_PORT_ACTIVE || port.link_layer!=IBV_LINK_LAYER_ETHERNET ||
       port.active_mtu<IBV_MTU_256 || port.active_mtu>IBV_MTU_4096 ||
       tp->opt.rdma_gid_index<0 || tp->opt.rdma_gid_index>255 || tp->opt.rdma_gid_index>=port.gid_tbl_len ||
       dev.max_qp_wr<(int)TP_RC_WINDOW || dev.max_cqe<(int)(2*TP_RC_WINDOW) || dev.max_sge<1) {
        tp_set_err(err,errlen,"RoCE: inactive Ethernet port, invalid GID or insufficient queue capacity"); goto fail;
    }
    if(r->query_gid(r->ctx,(uint8_t)portnum,tp->opt.rdma_gid_index,&r->gid)) goto syserr;
    union ibv_gid zero={0};
    if(!memcmp(&r->gid,&zero,sizeof(zero))) { tp_set_err(err,errlen,"RoCE: selected GID is zero"); goto fail; }
    r->mtu=port.active_mtu;
    r->pd=r->alloc_pd(r->ctx); if(!r->pd) goto syserr;
    r->cq=r->create_cq(r->ctx,2*TP_RC_WINDOW,NULL,NULL,0); if(!r->cq) goto syserr;
    struct ibv_qp_init_attr qi={.send_cq=r->cq,.recv_cq=r->cq,
        .cap={.max_send_wr=TP_RC_WINDOW,.max_recv_wr=TP_RC_WINDOW,.max_send_sge=1,.max_recv_sge=1},.qp_type=IBV_QPT_RC};
    r->qp=r->create_qp(r->pd,&qi); if(!r->qp) goto syserr;
    if(qi.cap.max_send_wr<TP_RC_WINDOW || qi.cap.max_recv_wr<TP_RC_WINDOW || r->cq->cqe<(int)(2*TP_RC_WINDOW)) goto syserr;
    if(posix_memalign((void **)&r->tx,4096,TP_RC_WINDOW*TP_RC_CHUNK) ||
       posix_memalign((void **)&r->rx,4096,TP_RC_WINDOW*TP_RC_CHUNK)) goto syserr;
    r->tx_mr=r->reg_mr(r->pd,r->tx,TP_RC_WINDOW*TP_RC_CHUNK,IBV_ACCESS_LOCAL_WRITE);
    r->rx_mr=r->reg_mr(r->pd,r->rx,TP_RC_WINDOW*TP_RC_CHUNK,IBV_ACCESS_LOCAL_WRITE);
    if(!r->tx_mr || !r->rx_mr) goto syserr;
    if(getrandom(&r->psn,sizeof(r->psn),0)!=(ssize_t)sizeof(r->psn)) goto syserr;
    r->psn &= 0xffffffu;
    struct ibv_qp_attr init={.qp_state=IBV_QPS_INIT,.pkey_index=0,.port_num=(uint8_t)portnum,.qp_access_flags=0};
    if(r->modify_qp(r->qp,&init,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)) goto syserr;
    return 1;
syserr:
    tp_set_err(err,errlen,"RoCE setup on %s: %s",tp->opt.rdma_device,strerror(errno ? errno : EINVAL));
fail:
    tp_roce_close(tp); return 0;
}
static int tp_roce_connect(ds4_tp *tp,char *err,size_t errlen) {
    struct ds4_tp_roce *r=tp->roce;
    struct { uint64_t epoch; uint32_t qpn,psn,mtu; uint8_t gid[16]; uint32_t reserved; } mine={0},peer={0};
    mine.epoch=tp->epoch;mine.qpn=r->qp->qp_num;mine.psn=r->psn;mine.mtu=r->mtu;memcpy(mine.gid,&r->gid,16);
    if(!ds4_tp_io_exchange(tp->data_fd,false,&mine,&peer,sizeof(mine),tp->gate_timeout_ms,&tp->failed)) goto fail;
    if(peer.epoch!=mine.epoch || !peer.qpn || peer.qpn>0xffffffu || peer.psn>0xffffffu ||
       peer.mtu<IBV_MTU_256 || peer.mtu>IBV_MTU_4096 || peer.reserved) { errno=EPROTO; goto fail; }
    const uint8_t port=tp->opt.rdma_port?tp->opt.rdma_port:1;
    struct ibv_qp_attr a={0};
    a.qp_state=IBV_QPS_RTR;a.path_mtu=peer.mtu<(uint32_t)r->mtu?(enum ibv_mtu)peer.mtu:r->mtu;
    a.dest_qp_num=peer.qpn;a.rq_psn=peer.psn;a.max_dest_rd_atomic=1;a.min_rnr_timer=12;
    a.ah_attr.is_global=1;a.ah_attr.port_num=port;memcpy(&a.ah_attr.grh.dgid,peer.gid,16);
    a.ah_attr.grh.sgid_index=tp->opt.rdma_gid_index;a.ah_attr.grh.hop_limit=64;
    if(r->modify_qp(r->qp,&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER)) goto fail;
    memset(&a,0,sizeof(a));a.qp_state=IBV_QPS_RTS;a.sq_psn=r->psn;a.timeout=14;a.retry_cnt=3;a.rnr_retry=3;a.max_rd_atomic=1;
    if(r->modify_qp(r->qp,&a,IBV_QP_STATE|IBV_QP_SQ_PSN|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|IBV_QP_RNR_RETRY|IBV_QP_MAX_QP_RD_ATOMIC)) goto fail;
    fprintf(stderr,"ds4-tp: RoCE RC device=%s port=%u gid-index=%d window=%u chunk=%u host-staging=%uMiB\n",tp->opt.rdma_device,port,tp->opt.rdma_gid_index,TP_RC_WINDOW,TP_RC_CHUNK,2*TP_RC_WINDOW*TP_RC_CHUNK/(1024*1024));
    return 1;
fail:
    tp_set_err(err,errlen,"RoCE connect: %s",strerror(errno));return 0;
}
/* Called before the shared TCP gate header: crossing that header guarantees
 * both ranks have posted their first bounded receive window. */
static int tp_roce_prepare(ds4_tp *tp,uint64_t bytes) {
    struct ds4_tp_roce *r=tp->roce;
    if(r->posted || !bytes || r->window_id>=UINT64_MAX/16u) { errno=EINVAL; return 0; }
    ++r->window_id;
    for(unsigned i=0;i<TP_RC_WINDOW && bytes;i++) {
        uint32_t n=bytes>TP_RC_CHUNK?TP_RC_CHUNK:(uint32_t)bytes;
        r->lengths[i]=n;
        struct ibv_sge sg={.addr=(uintptr_t)(r->rx+i*TP_RC_CHUNK),.length=n,.lkey=r->rx_mr->lkey};
        struct ibv_recv_wr wr={.wr_id=r->window_id*16u+i,.sg_list=&sg,.num_sge=1},*bad=NULL;
        if(ibv_post_recv(r->qp,&wr,&bad)) return 0;
        r->posted++;bytes-=n;
    }
    return 1;
}
static int tp_roce_exchange(ds4_tp *tp,const void *out,void *in,uint64_t bytes) {
    struct ds4_tp_roce *r=tp->roce;
    const double deadline=tp_now_sec()+(double)tp->gate_timeout_ms/1000.0;
    uint64_t offset=0;
    while(offset<bytes) {
        unsigned nr=r->posted;
        if(!nr) { errno=EPROTO; return 0; }
        uint64_t covered=0;
        for(unsigned i=0;i<nr;i++) {
            memcpy(r->tx+i*TP_RC_CHUNK,(const uint8_t*)out+offset+covered,r->lengths[i]);
            struct ibv_sge sg={.addr=(uintptr_t)(r->tx+i*TP_RC_CHUNK),.length=r->lengths[i],.lkey=r->tx_mr->lkey};
            struct ibv_send_wr wr={.wr_id=r->window_id*16u+8u+i,.sg_list=&sg,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad=NULL;
            if(ibv_post_send(r->qp,&wr,&bad)) return 0;
            covered+=r->lengths[i];
        }
        unsigned sent=0,received=0,mask=(1u<<nr)-1u;
        while(sent!=mask || received!=mask) {
            if(ds4_tp_failed(tp)) { errno=ECANCELED; return 0; }
            if(tp_now_sec()>deadline) { errno=ETIMEDOUT; return 0; }
            struct ibv_wc wc[2*TP_RC_WINDOW];int n=ibv_poll_cq(r->cq,2*TP_RC_WINDOW,wc);
            if(n<0) return 0;
            for(int i=0;i<n;i++) {
                uint64_t id=wc[i].wr_id;
                if(wc[i].status!=IBV_WC_SUCCESS) {
                    fprintf(stderr,"ds4-tp: RoCE completion status=%u vendor=%u id=%llu\n",wc[i].status,wc[i].vendor_err,(unsigned long long)id);errno=EIO;return 0;
                }
                if(id/16u!=r->window_id) { errno=EPROTO; return 0; }
                unsigned k=(unsigned)(id%16u),*done=NULL;
                if(k<nr && wc[i].opcode==IBV_WC_RECV && wc[i].byte_len==r->lengths[k]) done=&received;
                else if(k>=8u && k<8u+nr && wc[i].opcode==IBV_WC_SEND) { k-=8u;done=&sent; }
                if(!done || (*done&(1u<<k))) { errno=EPROTO;return 0; }
                *done|=1u<<k;
            }
        }
        atomic_thread_fence(memory_order_acquire);
        covered=0;
        for(unsigned i=0;i<nr;i++) { memcpy((uint8_t*)in+offset+covered,r->rx+i*TP_RC_CHUNK,r->lengths[i]);covered+=r->lengths[i]; }
        r->posted=0;offset+=covered;
        if(offset<bytes) {
            if(!tp_roce_prepare(tp,bytes-offset)) return 0;
            uint64_t mine[2]={r->window_id,offset},peer[2];
            double remaining=(deadline-tp_now_sec())*1000.0;
            if(remaining<1 || !ds4_tp_io_exchange(tp->data_fd,false,mine,peer,sizeof(mine),(uint64_t)remaining,&tp->failed) || memcmp(mine,peer,sizeof(mine))) return 0;
        }
    }
    return 1;
}
#else
static void tp_roce_close(ds4_tp *tp) { (void)tp; }
static int tp_roce_probe(ds4_tp *tp,char *err,size_t errlen) {
    (void)tp;tp_set_err(err,errlen,"RoCE unavailable: rebuild with libibverbs development headers");return 0;
}
static int tp_roce_connect(ds4_tp *tp,char *err,size_t errlen) { (void)tp;(void)err;(void)errlen;return 0; }
static int tp_roce_prepare(ds4_tp *tp,uint64_t bytes) { (void)tp;(void)bytes;return 0; }
static int tp_roce_exchange(ds4_tp *tp,const void *out,void *in,uint64_t bytes) { (void)tp;(void)out;(void)in;(void)bytes;return 0; }
#endif
