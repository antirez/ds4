#include "../ds4_tp.c"
#include <assert.h>
#include <pthread.h>

typedef struct {
    ds4_tp tp;
    uint8_t *out, *in;
    uint64_t bytes;
    useconds_t delay;
    int ok;
} bulk_peer;

static void *exchange_bulk(void *arg) {
    bulk_peer *p = arg;
    if (p->delay) usleep(p->delay);
    p->ok = ds4_tp_big_gate_exchange(&p->tp, 39, 17, p->out, p->in, p->bytes);
    return NULL;
}

static void check_bulk_exchange(void) {
    const uint64_t sizes[] = {20480, 2 * 1024 * 1024 - 4,
        2 * 1024 * 1024, 2 * 1024 * 1024 + 4, 7 * 1024 * 1024 + 4};
    for (unsigned n = 0; n < sizeof(sizes) / sizeof(*sizes); n++) {
        int fd[2];
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        bulk_peer peer[2] = {0};
        for (unsigned rank = 0; rank < 2; rank++) {
            tp_socket_tune(fd[rank]);
            assert(tp_socket_set_gate_timeout(fd[rank], 3000));
            peer[rank].tp.data_fd = fd[rank];
            peer[rank].tp.gate_timeout_ms = 3000;
            peer[rank].bytes = sizes[n];
            peer[rank].out = malloc(sizes[n]);
            peer[rank].in = malloc(sizes[n]);
            assert(peer[rank].out && peer[rank].in);
            for (uint64_t i = 0; i < sizes[n]; i++)
                peer[rank].out[i] = (uint8_t)(i * 31u + rank * 17u);
        }
        if (n == 0) {
            peer[0].tp.gate_timeout_ms = peer[1].tp.gate_timeout_ms = 50;
            peer[1].delay = 150000;
        }
        pthread_t thread;
        assert(pthread_create(&thread, NULL, exchange_bulk, &peer[1]) == 0);
        exchange_bulk(&peer[0]);
        assert(pthread_join(thread, NULL) == 0);
        assert(peer[0].ok && peer[1].ok);
        assert(!memcmp(peer[0].in, peer[1].out, sizes[n]));
        assert(!memcmp(peer[1].in, peer[0].out, sizes[n]));
        for (unsigned rank = 0; rank < 2; rank++) {
            struct timeval timeout;
            socklen_t len = sizeof(timeout);
            assert(getsockopt(fd[rank], SOL_SOCKET, SO_RCVTIMEO, &timeout, &len) == 0);
            assert((uint64_t)timeout.tv_sec * 1000u + timeout.tv_usec / 1000u ==
                   peer[rank].tp.gate_timeout_ms);
            free(peer[rank].in); free(peer[rank].out); close(fd[rank]);
        }
    }
    for (unsigned failure = 0; failure < 3; failure++) {
        int fd[2];
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        assert(tp_socket_set_gate_timeout(fd[0], 50));
        uint32_t out = 42, in = 123;
        ds4_tp tp = {.data_fd = fd[0], .gate_timeout_ms = 50};
        if (failure == 0) {
            /* EOF after a complete header, before its promised payload. */
            ds4_tp_gate_header h = {DS4_TP_BATCH_MAGIC, 39, 0xB16u, 17};
            assert(tp_write_full(fd[1], &h, sizeof(h)));
            assert(shutdown(fd[1], SHUT_WR) == 0);
        } else if (failure == 1) {
            ds4_tp_gate_header h = {DS4_TP_BATCH_MAGIC, 38, 0xB16u, 17};
            assert(tp_write_full(fd[1], &h, sizeof(h)));
        }
        assert(!ds4_tp_big_gate_exchange(&tp, 39, 17, &out, &in, sizeof(in)));
        assert(in == 123);
        close(fd[0]); close(fd[1]);
    }
    puts("TP bulk TCP: exact payloads, chunk boundaries, delayed peer, EOF/desync/timeout: ok");
}

typedef struct {
    ds4_tp tp;
    uint32_t point;
    bool request, cancelled;
    int ok;
} cancel_peer;

static void *exchange_cancel(void *ud) {
    cancel_peer *p = ud;
    p->ok = ds4_tp_sync_checkpoint(&p->tp, p->point, 4096, 65536,
                                   p->request, &p->cancelled);
    return NULL;
}

static void check_sync_cancellation(void) {
    int fd[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    cancel_peer peer[2] = {{{.control_fd = fd[0]}, 2, false, false, 0},
                           {{.control_fd = fd[1]}, 2, false, false, 0}};
    for (unsigned i = 0; i < 6; i++) {
        peer[0].request = i == 1;
        peer[1].request = i == 3;
        pthread_t thread;
        assert(pthread_create(&thread, NULL, exchange_cancel, &peer[1]) == 0);
        exchange_cancel(&peer[0]);
        assert(pthread_join(thread, NULL) == 0);
        assert(peer[0].ok && peer[1].ok);
        assert(peer[0].cancelled == (i == 1 || i == 3));
        assert(peer[0].cancelled == peer[1].cancelled);
        assert(!ds4_tp_failed(&peer[0].tp) && !ds4_tp_failed(&peer[1].tp));
    }
    char err[256];
    int status = -1;
    assert(ds4_tp_send_command_ack(&peer[1].tp, 42, DS4_SESSION_SYNC_INTERRUPTED));
    assert(ds4_tp_wait_command_status(&peer[0].tp, 42, &status, "cancel", err, sizeof(err)));
    assert(status == DS4_SESSION_SYNC_INTERRUPTED && !ds4_tp_failed(&peer[0].tp));
    peer[1].point = 3;
    pthread_t thread;
    assert(pthread_create(&thread, NULL, exchange_cancel, &peer[1]) == 0);
    exchange_cancel(&peer[0]);
    assert(pthread_join(thread, NULL) == 0);
    assert(!peer[0].ok && !peer[1].ok);
    assert(ds4_tp_failed(&peer[0].tp) && ds4_tp_failed(&peer[1].tp));
    close(fd[0]); close(fd[1]);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    ds4_tp tp = {.control_fd = fd[0]};
    assert(tp_socket_set_gate_timeout(fd[0], 50));
    bool cancelled = false;
    assert(!ds4_tp_sync_checkpoint(&tp, 0, 0, 1, false, &cancelled));
    assert(ds4_tp_failed(&tp));
    close(fd[0]); close(fd[1]);
    puts("TP cancellation agreement, reuse, interrupted acknowledgment, mismatch and timeout: ok");
}

static void check_backend_options(void) {
    char err[256];
    ds4_engine_options opt = {.backend = DS4_BACKEND_METAL,
        .tp = {.role = DS4_TP_LEADER, .requested = true}};
    assert(ds4_tp_validate_engine_options(&opt, err, sizeof(err)));
    opt.backend = DS4_BACKEND_CPU;
    assert(!ds4_tp_validate_engine_options(&opt, err, sizeof(err)));
    opt.backend = DS4_BACKEND_CUDA;
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
    assert(ds4_tp_validate_engine_options(&opt, err, sizeof(err)));
    opt.ssd_streaming = true;
    assert(!ds4_tp_validate_engine_options(&opt, err, sizeof(err)));
    opt.ssd_streaming = false;
    opt.cuda_tensor_parallel = true;
    assert(!ds4_tp_validate_engine_options(&opt, err, sizeof(err)));
#else
    assert(!ds4_tp_validate_engine_options(&opt, err, sizeof(err)));
#endif
    puts("TP backend and mutually exclusive placement options: ok");
}

static void check_logits_halves(void) {
    const float expected[] = {1.25f, -2.5f, 0, -0.0f, 9, -17, 0.125f, 65536};
    float received[10];
    const uint32_t canary = UINT32_C(0xa5a5a5a5);
    for (unsigned mode = 0; mode < 7; mode++) {
        int fd[2];
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        assert(tp_socket_set_gate_timeout(fd[0], 50));
        ds4_tp leader = {.control_fd = fd[0]}, worker = {.control_fd = fd[1]};
        memset(received, 0xa5, sizeof(received));
        if (mode == 0) {
            for (unsigned repeat = 0; repeat < 3; repeat++) {
                assert(ds4_tp_send_logits_half(&worker, expected, 8));
                assert(ds4_tp_recv_logits_half(&leader, received + 1, 8));
                assert(!memcmp(received + 1, expected, sizeof(expected)));
            }
        } else {
            if (mode != 1 && mode != 5) {
                ds4_tp_frame_header h = {DS4_TP_MAGIC,
                    mode == 3 ? DS4_TP_FRAME_STOP : DS4_TP_FRAME_LOGITS,
                    mode == 4 ? sizeof(expected) - sizeof(float) : sizeof(expected)};
                assert(tp_write_full(fd[1], &h, sizeof(h)));
                if (mode == 2 || mode == 6)
                    assert(tp_write_full(fd[1], expected, 2 * sizeof(float)));
            }
            /* Missing header, partial payload, wrong frame, wrong size,
             * then stalled header and stalled payload. */
            if (mode <= 4) assert(shutdown(fd[1], SHUT_WR) == 0);
            assert(!ds4_tp_recv_logits_half(&leader, received + 1, 8));
        }
        assert(!memcmp(received, &canary, sizeof(canary)));
        assert(!memcmp(received + 9, &canary, sizeof(canary)));
        close(fd[0]); close(fd[1]);
    }
    puts("TP logits halves: exact frames, reuse, canaries, EOF, invalid frames and timeout: ok");
}

int main(void) {
    check_backend_options();
    check_bulk_exchange();
    check_sync_cancellation();
    check_logits_halves();
    int fd[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    ds4_tp leader = { .control_fd = fd[0] };
    ds4_tp worker = { .control_fd = fd[1] };
    char err[256] = "";
    ds4_tp_command cmd;
    assert(DS4_TP_PROTOCOL_VERSION == 14);
    for (int i = 0; i < 4; i++) {
        assert(ds4_tp_send_eval(&leader, 42, 2*i, 100+i));
        assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
        assert(cmd.type == DS4_TP_FRAME_EVAL && cmd.value == 100+i);
        assert(cmd.limit == 0 && cmd.session_id == 42 && cmd.seq == (uint64_t)2*i);
        ds4_tp_command_free(&cmd);
        const int limit = 1 + i%2;
        assert(ds4_tp_send_glm_mtp(&leader, 42, 2*i+1, 200+i, limit));
        assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
        assert(cmd.type == DS4_TP_FRAME_GLM_MTP && cmd.value == 200+i);
        assert(cmd.limit == limit && cmd.session_id == 42 && cmd.seq == (uint64_t)2*i+1);
        ds4_tp_command_free(&cmd);
    }
    assert(!ds4_tp_send_glm_mtp(&leader, 42, 9, 1, 0));
    assert(!ds4_tp_send_glm_mtp(&leader, 42, 9, 1, 3));
    for (uint32_t bad = 0; bad <= 3; bad += 3) {
        ds4_tp_eval_command msg = {42, 9, 1, bad};
        assert(tp_send_frame(fd[0], DS4_TP_FRAME_GLM_MTP, &msg, sizeof(msg)));
        assert(!ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
        ds4_tp_command_free(&cmd);
    }
    ds4_tp_eval_command bad_eval = {42, 9, 1, 2};
    assert(tp_send_frame(fd[0], DS4_TP_FRAME_EVAL, &bad_eval, sizeof(bad_eval)));
    assert(!ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    ds4_tp_command_free(&cmd);
    assert(ds4_tp_send_rewind(&leader, 42, 123));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_REWIND && cmd.value == 123);
    ds4_tp_command_free(&cmd);
    assert(ds4_tp_send_invalidate(&leader, 42));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_INVALIDATE && cmd.session_id == 42);
    ds4_tp_command_free(&cmd);
    const int tokens[] = {7, 19, 5};
    assert(ds4_tp_send_sync(&leader, 42, tokens, 3));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_SYNC && cmd.n_tokens == 3);
    assert(memcmp(cmd.tokens, tokens, sizeof(tokens)) == 0);
    ds4_tp_command_free(&cmd);
    /* Multimodal sync carries f32 image rows; the worker sizes it against
     * n_embd and the negotiated context (unset here, so the absolute cap). */
    leader.n_embd = 8;
    worker.n_embd = 8;
    float rows[2 * 8];
    for (int i = 0; i < 16; i++) rows[i] = 0.5f * i;
    ds4_vision_span image;
    memset(&image, 0, sizeof(image));
    image.token_start = 1;
    image.embedding.data = rows;
    image.embedding.token_count = 2;
    image.embedding.width = 16;
    image.embedding.height = 16;
    assert(ds4_tp_send_sync_multimodal(&leader, 42, tokens, 3, &image, 1));
    assert(ds4_tp_recv_command(&worker, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_SYNC_MULTIMODAL && cmd.n_tokens == 3);
    assert(memcmp(cmd.tokens, tokens, sizeof(tokens)) == 0);
    assert(cmd.n_images == 1 && cmd.images[0].token_start == 1);
    assert(cmd.images[0].embedding.token_count == 2);
    assert(memcmp(cmd.images[0].embedding.data, rows, sizeof(rows)) == 0);
    ds4_tp_command_free(&cmd);
    assert(ds4_tp_send_command_ack(&worker, 42, 0));
    assert(ds4_tp_wait_command_ack(&leader, 42, "rebuild", err, sizeof(err)));
    assert(ds4_tp_send_command_ack(&worker, 42, 1));
    assert(!ds4_tp_wait_command_ack(&leader, 42, "GLM MTP", err, sizeof(err)));
    close(fd[0]);
    close(fd[1]);
    puts("TP command tests: ok");
    return 0;
}
