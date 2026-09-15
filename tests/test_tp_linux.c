/* Model-free protocol and bounded full-duplex regression tests. */
#define DS4_ROCM_BUILD 1
#include "../ds4_tp.c"
#include <assert.h>

/* The model-free command parser only needs the image allocation destructor. */
void ds4_vision_embedding_free(ds4_vision_embedding *embedding) {
    if (!embedding) return;
    free(embedding->data);
    memset(embedding, 0, sizeof(*embedding));
}

typedef struct {
    ds4_tp tp;
    unsigned char *out, *in;
    uint64_t bytes, seq;
    unsigned kind;
    int ok;
} peer;

static void *exchange(void *opaque) {
    peer *p = opaque;
    p->ok = tp_linux_gate(&p->tp, p->kind, 39, 1, p->seq,
                          p->out, p->in, p->bytes, 0);
    return NULL;
}

static void socket_pair(int fd[2]) {
    assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fd));
    for (int i = 0; i < 2; ++i) {
        int small = 1024;
        assert(!setsockopt(fd[i], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small)));
        assert(!fcntl(fd[i], F_SETFL, O_NONBLOCK));
    }
}

static void transfers(void) {
    const uint64_t sizes[] = {1, 3, 20480, 8*20480, 2*1024*1024-1,
                              2*1024*1024, 2*1024*1024+1, 7*1024*1024+3};
    for (int device = 0; device <= 1; ++device) {
        for (unsigned n = 0; n < sizeof(sizes)/sizeof(*sizes); ++n) {
            int fd[2], stream[2];
            socket_pair(fd); socket_pair(stream);
            peer p[2] = {0};
            for (unsigned rank = 0; rank < 2; ++rank) {
                p[rank].tp = (ds4_tp){.data_fd=fd[rank], .stream_fd=stream[rank],
                    .stream_active=device, .epoch=12345, .n_layer=40, .gate_timeout_ms=5000};
                atomic_init(&p[rank].tp.failed, false);
                p[rank].bytes=sizes[n]; p[rank].seq=7; p[rank].kind=3;
                p[rank].out=malloc(sizes[n]); p[rank].in=malloc(sizes[n]+2);
                assert(p[rank].out && p[rank].in);
                memset(p[rank].in, 0xa5, sizes[n]+2);
                ++p[rank].in;
                for (uint64_t j=0; j<sizes[n]; ++j)
                    p[rank].out[j]=(unsigned char)(j*31+rank*73);
            }
            /* Reuse the connection with a new sequence; no stale generation. */
            for (int round=0; round<2; ++round) {
                pthread_t worker;
                assert(!pthread_create(&worker, NULL, exchange, &p[1]));
                exchange(&p[0]); assert(!pthread_join(worker, NULL));
                assert(p[0].ok && p[1].ok);
                for (int rank=0;rank<2;++rank) {
                    assert(!memcmp(p[rank].in, p[1-rank].out, sizes[n]));
                    assert(p[rank].in[-1]==0xa5 && p[rank].in[sizes[n]]==0xa5);
                    ++p[rank].seq;
                }
            }
            for (int rank=0;rank<2;++rank) {
                free(p[rank].out); free(p[rank].in-1);
                close(fd[rank]);close(stream[rank]);
            }
        }
    }
}

static void failures(void) {
    for (int fault=0;fault<9;++fault) {
        int fd[2];socket_pair(fd);
        unsigned char out[8]={1}, in[8]; memset(in, 0xa5, sizeof(in));
        ds4_tp tp={.data_fd=fd[0], .n_layer=40, .epoch=123, .gate_timeout_ms=30};
        atomic_init(&tp.failed, false);
        ds4_tp_linux_gate_header h={DS4_TP_MAGIC,3,39,1,123,7,8};
        if (fault==0) h.bytes++;
        if (fault==1) h.seq++;
        if (fault==2) h.epoch++;
        if (fault==3) h.kind++;
        if (fault==4) h.layer++;
        if (fault==5) h.magic++;
        if (fault<6 || fault==8) assert(write(fd[1], &h, sizeof(h))==sizeof(h));
        if (fault==7) ds4_tp_mark_failed(&tp);
        if (fault==8) assert(!shutdown(fd[1], SHUT_WR));
        double start=tp_now_sec();
        assert(!tp_linux_gate(&tp,3,39,1,7,out,in,8,0));
        assert(ds4_tp_failed(&tp));
        assert(tp_now_sec()-start<.5);
        for (unsigned i=0;i<sizeof(in);++i) assert(in[i]==0xa5);
        assert(!tp_linux_gate(&tp,3,39,1,7,out,in,8,0));
        close(fd[0]);close(fd[1]);
    }
}

static void negotiation(void) {
    /* Exhaustive symmetry, explicit selection, and common-capability checks. */
    for (unsigned a=0;a<5;++a) for (unsigned b=0;b<5;++b)
        for (unsigned ac=0;ac<16;++ac) for (unsigned bc=0;bc<16;++bc) {
            int selected=tp_linux_select(a,b,ac,bc);
            assert(selected==tp_linux_select(b,a,bc,ac));
            int expected=-1;
            if (a<=3 && b<=3 && !(a && b && a!=b)) {
                unsigned requested=a?a:b, common=ac&bc;
                if (requested) expected=(common & (1u<<requested))?(int)requested:-1;
                else if (common&2) expected=1;
                else if (common&8) expected=3;
                else if (common&4) expected=2;
            }
            assert(selected==expected);
        }
    assert(tp_linux_open_stream("/nonexistent-ds4-tp-device")<0);
    assert(tp_linux_open_stream("tests/test_tp_linux.c")<0);
}

typedef struct { ds4_tp tp; ds4_tp_identity id; int ok; char err[256]; } hello_peer;
static void *hello(void *arg) {
    hello_peer *p=arg;
    p->ok=tp_hello_exchange(&p->tp,&p->id,0,p->err,sizeof(p->err));
    return NULL;
}
static void handshakes(void) {
    for (unsigned mode=0;mode<5;++mode) {
        int fd[2];socket_pair(fd);
        hello_peer p[2]={0};
        for (unsigned r=0;r<2;++r) {
            p[r].tp.control_fd=fd[r];p[r].tp.stream_fd=-1;
            p[r].tp.timeout_sec=1;
            p[r].tp.opt.role=r?DS4_TP_WORKER:DS4_TP_LEADER;
            p[r].id=(ds4_tp_identity){.gguf_bytes=999,.model_id=41,.n_layer=40,
                .n_embd=5120,.n_vocab=129280,.quant_bits=2,.ctx_size=8192};
        }
        if (mode==1) p[1].id.n_embd++;
        if (mode==2) p[1].tp.opt.transport=DS4_TP_TRANSPORT_USB4STREAM;
        if (mode>=3) {
            /* Reject legacy12 and CUDA/old-ROCm14 before reading a nonce or
             * backend-specific frames, even if the peer keeps the socket open. */
            ds4_tp_hello_fixed old={.magic=DS4_TP_MAGIC,.version=mode==3?12:14};
            assert(write(fd[1], &old, offsetof(ds4_tp_hello_fixed,nonce))==
                   offsetof(ds4_tp_hello_fixed,nonce));
            double start=tp_now_sec();hello(&p[0]);
            assert(!p[0].ok && strstr(p[0].err,"protocol version"));
            assert(tp_now_sec()-start<.5);
        } else {
            pthread_t t;assert(!pthread_create(&t,NULL,hello,&p[1]));
            hello(&p[0]);assert(!pthread_join(t,NULL));
            assert(p[0].ok==(mode==0) && p[1].ok==(mode==0));
            if (!mode) {
                assert(p[0].tp.epoch==p[1].tp.epoch);
                assert(p[0].tp.vec_bytes==20480 && p[1].tp.vec_bytes==20480);
            }
        }
        close(fd[0]);close(fd[1]);
    }
}
static void *cancel_exchange(void *opaque) {
    peer *p=opaque;
    usleep(20000);
    ds4_tp_mark_failed(&p->tp);
    return NULL;
}
static void cancellation(void) {
    int fd[2];socket_pair(fd);
    unsigned char out=1,in=0xa5;
    peer p={.tp={.data_fd=fd[0],.n_layer=40,.gate_timeout_ms=5000},
            .out=&out,.in=&in,.bytes=1,.seq=1,.kind=3};
    pthread_t t;assert(!pthread_create(&t,NULL,cancel_exchange,&p));
    double start=tp_now_sec();exchange(&p);
    assert(!pthread_join(t,NULL));
    assert(!p.ok && ds4_tp_failed(&p.tp) && in==0xa5);
    assert(tp_now_sec()-start<.5);
    close(fd[0]);close(fd[1]);
}

typedef struct {
    ds4_tp tp;
    const unsigned char *data;
    size_t bytes;
    int mode;
} restore_peer;

static void *restore_receive(void *opaque) {
    restore_peer *p = opaque;
    char err[256] = {0};
    ds4_tp_command command;
    assert(ds4_tp_recv_command(&p->tp, &command, err, sizeof(err)));
    assert(command.type == DS4_TP_FRAME_RESTORE_PAYLOAD &&
           command.session_id == 42 && command.payload_bytes == p->bytes);
    ds4_tp_command_free(&command);
    ds4_tp_payload_reader reader = {&p->tp, p->bytes, tp_now_sec() + 2};
    cookie_io_functions_t io = {.read = tp_payload_read};
    FILE *fp = fopencookie(&reader, "rb", io);
    assert(fp);
    unsigned char *storage = malloc(p->bytes + 2);
    assert(storage); memset(storage, 0xa5, p->bytes + 2);
    unsigned char *data = storage + 1;
    /* A small first read exercises stdio read-ahead; the cookie must never
     * consume bytes belonging to the following control frame. */
    assert(fread(data, 1, 17, fp) == 17);
    size_t got = fread(data + 17, 1, p->bytes - 17, fp);
    if (p->mode == 1) {
        assert(got < p->bytes - 17 && ferror(fp) && ds4_tp_failed(&p->tp));
    } else {
        assert(got == p->bytes - 17 && !memcmp(data, p->data, p->bytes));
        assert(fgetc(fp) == EOF && !ferror(fp) && !reader.remaining);
        assert(ds4_tp_send_command_ack(&p->tp, 42, p->mode == 2));
        assert(ds4_tp_recv_command(&p->tp, &command, err, sizeof(err)));
        assert(command.type == DS4_TP_FRAME_STOP);
        ds4_tp_command_free(&command);
    }
    assert(storage[0] == 0xa5 && storage[p->bytes + 1] == 0xa5);
    fclose(fp); free(storage);
    return NULL;
}

static void checkpoint_streams(void) {
    const size_t bytes = 3 * 65536 + 9;
    unsigned char *data = malloc(bytes); assert(data);
    for (size_t i = 0; i < bytes; ++i) data[i] = (unsigned char)(i * 37 + 11);
    for (int mode = 0; mode < 3; ++mode) {
        int fd[2]; socket_pair(fd);
        for (int i = 0; i < 2; ++i) {
            assert(!fcntl(fd[i], F_SETFL, 0));
            struct timeval limit = {.tv_sec = 2};
            assert(!setsockopt(fd[i], SOL_SOCKET, SO_RCVTIMEO, &limit, sizeof(limit)));
            assert(!setsockopt(fd[i], SOL_SOCKET, SO_SNDTIMEO, &limit, sizeof(limit)));
        }
        ds4_tp leader = {.control_fd = fd[0], .rank = 0, .timeout_sec = 2};
        restore_peer worker = {.tp = {.control_fd = fd[1], .rank = 1},
            .data = data, .bytes = bytes, .mode = mode};
        atomic_init(&leader.failed, false); atomic_init(&worker.tp.failed, false);
        pthread_t thread; assert(!pthread_create(&thread, NULL, restore_receive, &worker));
        FILE *fp = fmemopen(data, bytes - (mode == 1), "rb"); assert(fp);
        char err[256] = {0};
        assert(ds4_tp_send_restore_payload(&leader, 42, fp, bytes, err, sizeof(err)) == (mode == 0));
        assert(ds4_tp_failed(&leader) == (mode == 1));
        if (mode == 1) assert(!shutdown(fd[0], SHUT_WR));
        else assert(ds4_tp_send_stop(&leader));
        fclose(fp); assert(!pthread_join(thread, NULL));
        close(fd[0]); close(fd[1]);
    }
    free(data);
}

int main(void) {
    assert(DS4_TP_PROTOCOL_VERSION==15);
    negotiation(); handshakes(); transfers(); failures(); cancellation(); checkpoint_streams();
    puts("Linux TP: negotiation, full-duplex TCP/device I/O, tails, canaries, generations and failures PASS");
    return 0;
}
