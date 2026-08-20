#include "ds4_tp.h"

#include <arpa/inet.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int reserve_loopback_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = 0,
        .sin_addr.s_addr = htonl(INADDR_LOOPBACK),
    };
    socklen_t len = sizeof(addr);
    if (bind(fd, (const struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        getsockname(fd, (struct sockaddr *)&addr, &len) != 0) {
        close(fd);
        return -1;
    }
    const int port = ntohs(addr.sin_port);
    close(fd);
    return port;
}

static ds4_tp_identity test_identity(uint32_t ctx_size) {
    return (ds4_tp_identity) {
        .gguf_bytes = UINT64_C(211075856448),
        .layout_fingerprint = UINT64_C(0x9d7e6c5b4a392817),
        .model_id = 52,
        .n_layer = 78,
        .n_embd = 6144,
        .n_vocab = 128000,
        .quant_bits = 2,
        .quant_profile = DS4_QUANT_PROFILE_PRO_IQ2_Q2,
        .ctx_size = ctx_size,
        .gate_slot_start = 7,
        .gate_slot_step = 2,
        .gates_per_token = 75,
    };
}

static void child_error(uint32_t rank, const char *what, const char *detail,
                        int code) {
    fprintf(stderr, "test_tp_protocol: rank %u %s%s%s\n",
            rank, what, detail && detail[0] ? ": " : "",
            detail && detail[0] ? detail : "");
    _exit(code);
}

static void run_success_worker(uint32_t rank, uint32_t world, int port,
                               bool expert_only) {
    alarm(30);
    /* Make the highest rank connect first and rank 1 last. The leader must
     * index peers by their advertised rank rather than accept order. */
    if (world > 2u) usleep((useconds_t)(world - 1u - rank) * 20000u);
    ds4_tp_options opt = {
        .role = DS4_TP_WORKER,
        .requested = true,
        .expert_only = expert_only,
        .leader_host = "127.0.0.1",
        .leader_port = port,
        .world_size = world,
        .rank = rank,
        .rank_set = true,
        .transport = DS4_TP_TRANSPORT_NCCL,
    };
    const ds4_tp_identity id = test_identity(4096u);
    char err[256] = "";
    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        child_error(rank, "create failed", err, 10);
    }
    if (!ds4_tp_collective_preflight(tp, 1, err, sizeof(err))) {
        child_error(rank, "preflight failed", err, 11);
    }
    uint8_t blob[32] = {0};
    if (!ds4_tp_broadcast_blob(tp, blob, sizeof(blob), err, sizeof(err))) {
        child_error(rank, "bootstrap receive failed", err, 12);
    }
    for (uint32_t i = 0; i < sizeof(blob); i++) {
        const uint8_t expected = (uint8_t)(i ^ (world << 4u));
        if (blob[i] != expected) {
            child_error(rank, "bootstrap payload mismatch", NULL, 13);
        }
    }
    ds4_tp_command command;
    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err))) {
        child_error(rank, "command receive failed", err, 14);
    }
    const uint64_t session_id = UINT64_C(0x12340000) + world;
    const uint64_t session_id_2 = UINT64_C(0x56780000) + world;
    if (command.type != DS4_TP_FRAME_SESSION_CREATE ||
        command.session_id != session_id || command.value != 1024) {
        ds4_tp_command_free(&command);
        child_error(rank, "session-create payload mismatch", NULL, 15);
    }
    ds4_tp_command_free(&command);
    if (!ds4_tp_send_command_ack(tp, session_id, 0)) {
        child_error(rank, "ACK send failed", NULL, 16);
    }

    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_SESSION_CREATE ||
        command.session_id != session_id_2 || command.value != 768) {
        ds4_tp_command_free(&command);
        child_error(rank, "second session-create mismatch", err, 17);
    }
    ds4_tp_command_free(&command);
    if (!ds4_tp_send_command_ack(tp, session_id_2, 0)) {
        child_error(rank, "second create ACK failed", NULL, 18);
    }

    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_SYNC ||
        command.session_id != session_id || command.n_tokens != 3 ||
        command.tokens[0] != 11 || command.tokens[1] != 22 ||
        command.tokens[2] != 33) {
        ds4_tp_command_free(&command);
        child_error(rank, "sync payload mismatch", err, 19);
    }
    ds4_tp_command_free(&command);
    if (!ds4_tp_send_command_ack(tp, session_id, 0)) {
        child_error(rank, "sync ACK failed", NULL, 20);
    }

    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_EVAL ||
        command.session_id != session_id || command.seq != 77 ||
        command.value != 44) {
        ds4_tp_command_free(&command);
        child_error(rank, "eval payload mismatch", err, 21);
    }
    ds4_tp_command_free(&command);

    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_REWIND ||
        command.session_id != session_id || command.value != 2) {
        ds4_tp_command_free(&command);
        child_error(rank, "rewind payload mismatch", err, 22);
    }
    ds4_tp_command_free(&command);

    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_INVALIDATE ||
        command.session_id != session_id_2) {
        ds4_tp_command_free(&command);
        child_error(rank, "invalidate payload mismatch", err, 23);
    }
    ds4_tp_command_free(&command);

    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_EVAL_BATCH || command.n_items != 2 ||
        command.items[0].session_id != session_id ||
        command.items[0].token != 55 ||
        command.items[1].session_id != session_id_2 ||
        command.items[1].token != 66) {
        ds4_tp_command_free(&command);
        child_error(rank, "eval-batch payload mismatch", err, 24);
    }
    ds4_tp_command_free(&command);
    if (!ds4_tp_send_command_ack(tp, 0, 0)) {
        child_error(rank, "eval-batch ACK failed", NULL, 25);
    }

    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_MIXED_BATCH ||
        command.session_id != session_id_2 || command.n_tokens != 2 ||
        command.tokens[0] != 7 || command.tokens[1] != 8 ||
        command.n_items != 1 ||
        command.items[0].session_id != session_id ||
        command.items[0].token != 9) {
        ds4_tp_command_free(&command);
        child_error(rank, "mixed-batch payload mismatch", err, 26);
    }
    ds4_tp_command_free(&command);
    if (!ds4_tp_send_command_ack(tp, session_id_2, 0)) {
        child_error(rank, "mixed-batch ACK failed", NULL, 27);
    }

    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_SESSION_DESTROY ||
        command.session_id != session_id) {
        ds4_tp_command_free(&command);
        child_error(rank, "first session-destroy mismatch", err, 28);
    }
    ds4_tp_command_free(&command);
    if (!ds4_tp_send_command_ack(tp, session_id, 0)) {
        child_error(rank, "first destroy ACK failed", NULL, 29);
    }

    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_SESSION_DESTROY ||
        command.session_id != session_id_2) {
        ds4_tp_command_free(&command);
        child_error(rank, "second session-destroy mismatch", err, 30);
    }
    ds4_tp_command_free(&command);
    if (!ds4_tp_send_command_ack(tp, session_id_2, 0)) {
        child_error(rank, "second destroy ACK failed", NULL, 31);
    }

    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err))) {
        child_error(rank, "stop receive failed", err, 32);
    }
    if (command.type != DS4_TP_FRAME_STOP) {
        ds4_tp_command_free(&command);
        child_error(rank, "expected stop", NULL, 33);
    }
    ds4_tp_command_free(&command);
    ds4_tp_free(tp);
    _exit(0);
}

static int wait_workers(pid_t *pids, uint32_t count) {
    int ok = 1;
    for (uint32_t i = 0; i < count; i++) {
        int status = 0;
        if (waitpid(pids[i], &status, 0) != pids[i] ||
            !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr,
                    "test_tp_protocol: worker pid %ld failed (status=0x%x)\n",
                    (long)pids[i], status);
            ok = 0;
        }
    }
    return ok;
}

static void terminate_workers(pid_t *pids, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        if (pids[i] > 0) (void)kill(pids[i], SIGTERM);
    }
    (void)wait_workers(pids, count);
}

static int run_success_case(uint32_t world, bool expert_only) {
    const int port = reserve_loopback_port();
    if (port <= 0) return 0;
    pid_t pids[DS4_TP_MAX_WORLD - 1u] = {0};
    for (uint32_t rank = 1; rank < world; rank++) {
        const pid_t pid = fork();
        if (pid == 0) run_success_worker(rank, world, port, expert_only);
        if (pid < 0) {
            terminate_workers(pids, rank - 1u);
            return 0;
        }
        pids[rank - 1u] = pid;
    }

    alarm(30);
    ds4_tp_options opt = {
        .role = DS4_TP_LEADER,
        .requested = true,
        .expert_only = expert_only,
        .listen_host = "127.0.0.1",
        .listen_port = port,
        .world_size = world,
        .rank = 0,
        .rank_set = true,
        .transport = DS4_TP_TRANSPORT_NCCL,
    };
    const ds4_tp_identity id = test_identity(4096u);
    char err[256] = "";
    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        fprintf(stderr, "test_tp_protocol: leader create: %s\n", err);
        terminate_workers(pids, world - 1u);
        return 0;
    }
    int ok = ds4_tp_rank(tp) == 0 && ds4_tp_world(tp) == world &&
             ds4_tp_is_collective(tp) &&
             ds4_tp_is_expert_only(tp) == expert_only &&
             ds4_tp_peer_ctx(tp) == 4096u;
    if (!ok) fprintf(stderr, "test_tp_protocol: leader topology mismatch\n");
    if (ok && !ds4_tp_collective_preflight(tp, 1, err, sizeof(err))) {
        fprintf(stderr, "test_tp_protocol: leader preflight: %s\n", err);
        ok = 0;
    }
    uint8_t blob[32];
    for (uint32_t i = 0; i < sizeof(blob); i++) {
        blob[i] = (uint8_t)(i ^ (world << 4u));
    }
    if (ok && !ds4_tp_broadcast_blob(tp, blob, sizeof(blob), err,
                                     sizeof(err))) {
        fprintf(stderr, "test_tp_protocol: leader bootstrap: %s\n", err);
        ok = 0;
    }
    const uint64_t session_id = UINT64_C(0x12340000) + world;
    const uint64_t session_id_2 = UINT64_C(0x56780000) + world;
    if (ok && (!ds4_tp_send_session_create(tp, session_id, 1024) ||
               !ds4_tp_wait_command_ack(tp, session_id, "test create",
                                        err, sizeof(err)))) {
        fprintf(stderr, "test_tp_protocol: command fan-out: %s\n", err);
        ok = 0;
    }
    if (ok && (!ds4_tp_send_session_create(tp, session_id_2, 768) ||
               !ds4_tp_wait_command_ack(tp, session_id_2, "test create 2",
                                        err, sizeof(err)))) {
        fprintf(stderr, "test_tp_protocol: second create fan-out: %s\n", err);
        ok = 0;
    }
    const int sync_tokens[] = {11, 22, 33};
    if (ok && (!ds4_tp_send_sync(tp, session_id, sync_tokens, 3) ||
               !ds4_tp_wait_command_ack(tp, session_id, "test sync",
                                        err, sizeof(err)))) {
        fprintf(stderr, "test_tp_protocol: sync fan-out: %s\n", err);
        ok = 0;
    }
    if (ok && (!ds4_tp_send_eval(tp, session_id, 77, 44) ||
               !ds4_tp_send_rewind(tp, session_id, 2) ||
               !ds4_tp_send_invalidate(tp, session_id_2))) {
        fprintf(stderr, "test_tp_protocol: eval/rewind/invalidate fan-out failed\n");
        ok = 0;
    }
    const ds4_tp_batch_item batch[] = {
        {.session_id = session_id, .token = 55},
        {.session_id = session_id_2, .token = 66},
    };
    if (ok && (!ds4_tp_send_eval_batch(tp, batch, 2) ||
               !ds4_tp_wait_command_ack(tp, 0, "test eval batch",
                                        err, sizeof(err)))) {
        fprintf(stderr, "test_tp_protocol: eval-batch fan-out: %s\n", err);
        ok = 0;
    }
    const int mixed_prompt[] = {7, 8};
    const ds4_tp_batch_item mixed[] = {
        {.session_id = session_id, .token = 9},
    };
    if (ok && (!ds4_tp_send_mixed_batch(tp, session_id_2,
                                        mixed_prompt, 2, mixed, 1) ||
               !ds4_tp_wait_command_ack(tp, session_id_2,
                                        "test mixed batch",
                                        err, sizeof(err)))) {
        fprintf(stderr, "test_tp_protocol: mixed-batch fan-out: %s\n", err);
        ok = 0;
    }
    if (ok && (!ds4_tp_send_session_destroy(tp, session_id) ||
               !ds4_tp_wait_command_ack(tp, session_id, "test destroy",
                                        err, sizeof(err)) ||
               !ds4_tp_send_session_destroy(tp, session_id_2) ||
               !ds4_tp_wait_command_ack(tp, session_id_2, "test destroy 2",
                                        err, sizeof(err)))) {
        fprintf(stderr, "test_tp_protocol: destroy fan-out: %s\n", err);
        ok = 0;
    }
    if (!ds4_tp_send_stop(tp)) ok = 0;
    ds4_tp_free(tp);
    if (!wait_workers(pids, world - 1u)) ok = 0;
    alarm(0);
    if (ok) {
        fprintf(stderr, "test_tp_protocol: %u-rank %s path OK\n",
                world, expert_only ? "expert" : "tensor");
    }
    return ok;
}

static void run_failed_preflight_worker(int port) {
    alarm(30);
    ds4_tp_options opt = {
        .role = DS4_TP_WORKER,
        .requested = true,
        .expert_only = true,
        .leader_host = "127.0.0.1",
        .leader_port = port,
        .world_size = 2,
        .rank = 1,
        .rank_set = true,
        .transport = DS4_TP_TRANSPORT_NCCL,
    };
    const ds4_tp_identity id = test_identity(4096u);
    char err[256] = "";
    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        child_error(1, "failure-case create failed", err, 30);
    }
    if (ds4_tp_collective_preflight(tp, 0, err, sizeof(err)) ||
        strstr(err, "unavailable") == NULL) {
        child_error(1, "failure-case preflight was not rejected", err, 31);
    }
    ds4_tp_command command;
    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_STOP) {
        child_error(1, "failure-case stop receive failed", err, 32);
    }
    ds4_tp_command_free(&command);
    ds4_tp_free(tp);
    _exit(0);
}

static int run_failed_preflight_case(void) {
    const int port = reserve_loopback_port();
    if (port <= 0) return 0;
    const pid_t pid = fork();
    if (pid == 0) run_failed_preflight_worker(port);
    if (pid < 0) return 0;
    pid_t pids[1] = {pid};

    alarm(30);
    ds4_tp_options opt = {
        .role = DS4_TP_LEADER,
        .requested = true,
        .expert_only = true,
        .listen_host = "127.0.0.1",
        .listen_port = port,
        .world_size = 2,
        .rank = 0,
        .rank_set = true,
        .transport = DS4_TP_TRANSPORT_NCCL,
    };
    const ds4_tp_identity id = test_identity(4096u);
    char err[256] = "";
    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        fprintf(stderr, "test_tp_protocol: failure-case leader create: %s\n",
                err);
        terminate_workers(pids, 1);
        return 0;
    }
    int ok = !ds4_tp_collective_preflight(tp, 1, err, sizeof(err)) &&
             strstr(err, "unavailable") != NULL;
    if (!ok) {
        fprintf(stderr,
                "test_tp_protocol: unavailable-rank preflight was not rejected: %s\n",
                err);
    }
    if (!ds4_tp_send_stop(tp)) ok = 0;
    ds4_tp_free(tp);
    if (!wait_workers(pids, 1)) ok = 0;
    alarm(0);
    if (ok) fprintf(stderr, "test_tp_protocol: failed preflight path OK\n");
    return ok;
}

static void run_failed_batch_worker(int port) {
    alarm(30);
    const ds4_tp_options opt = {
        .role = DS4_TP_WORKER,
        .requested = true,
        .expert_only = true,
        .leader_host = "127.0.0.1",
        .leader_port = port,
        .world_size = 2u,
        .rank = 1u,
        .rank_set = true,
        .transport = DS4_TP_TRANSPORT_NCCL,
    };
    const ds4_tp_identity id = test_identity(4096u);
    char err[256] = "";
    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        child_error(1u, "failed-batch create failed", err, 34);
    }
    ds4_tp_command command;
    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_EVAL_BATCH ||
        command.n_items != 2u) {
        ds4_tp_command_free(&command);
        child_error(1u, "failed-batch command mismatch", err, 35);
    }
    ds4_tp_command_free(&command);
    if (!ds4_tp_send_command_ack(tp, 0u, 7)) {
        child_error(1u, "failed-batch ACK send failed", NULL, 36);
    }
    if (!ds4_tp_recv_command(tp, &command, err, sizeof(err)) ||
        command.type != DS4_TP_FRAME_STOP) {
        ds4_tp_command_free(&command);
        child_error(1u, "failed-batch stop receive failed", err, 37);
    }
    ds4_tp_command_free(&command);
    ds4_tp_free(tp);
    _exit(0);
}

static int run_failed_batch_case(void) {
    const int port = reserve_loopback_port();
    if (port <= 0) return 0;
    const pid_t pid = fork();
    if (pid == 0) run_failed_batch_worker(port);
    if (pid < 0) return 0;
    pid_t pids[1] = {pid};

    alarm(30);
    const ds4_tp_options opt = {
        .role = DS4_TP_LEADER,
        .requested = true,
        .expert_only = true,
        .listen_host = "127.0.0.1",
        .listen_port = port,
        .world_size = 2u,
        .rank = 0u,
        .rank_set = true,
        .transport = DS4_TP_TRANSPORT_NCCL,
    };
    const ds4_tp_identity id = test_identity(4096u);
    char err[256] = "";
    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        fprintf(stderr, "test_tp_protocol: failed-batch leader create: %s\n",
                err);
        terminate_workers(pids, 1u);
        return 0;
    }
    const ds4_tp_batch_item batch[] = {
        {.session_id = 11u, .token = 101},
        {.session_id = 22u, .token = 202},
    };
    int ok = ds4_tp_send_eval_batch(tp, batch, 2u) &&
             !ds4_tp_wait_command_ack(tp, 0u, "injected batch",
                                      err, sizeof(err)) &&
             ds4_tp_failed(tp) && strstr(err, "status 7") != NULL;
    if (!ok) {
        fprintf(stderr,
                "test_tp_protocol: negative batch ACK did not fail context: %s\n",
                err);
    }
    if (!ds4_tp_send_stop(tp)) ok = 0;
    ds4_tp_free(tp);
    if (!wait_workers(pids, 1u)) ok = 0;
    alarm(0);
    if (ok) fprintf(stderr, "test_tp_protocol: failed batch ACK path OK\n");
    return ok;
}

typedef enum {
    TEST_MISMATCH_CONTEXT,
    TEST_MISMATCH_QUANT_PROFILE,
    TEST_MISMATCH_LAYOUT,
    TEST_MISMATCH_WORLD,
    TEST_MISMATCH_MODE,
} test_mismatch_kind;

static void mutate_mismatch(test_mismatch_kind kind,
                            ds4_tp_options *opt,
                            ds4_tp_identity *id) {
    switch (kind) {
    case TEST_MISMATCH_CONTEXT:
        id->ctx_size /= 2u;
        break;
    case TEST_MISMATCH_QUANT_PROFILE:
        id->quant_profile = DS4_QUANT_PROFILE_NONE;
        break;
    case TEST_MISMATCH_LAYOUT:
        id->layout_fingerprint ^= UINT64_C(0x100);
        break;
    case TEST_MISMATCH_WORLD:
        opt->world_size = 4u;
        break;
    case TEST_MISMATCH_MODE:
        opt->expert_only = false;
        break;
    }
}

static void run_mismatch_worker(int port, test_mismatch_kind kind,
                                const char *expected) {
    alarm(30);
    ds4_tp_options opt = {
        .role = DS4_TP_WORKER,
        .requested = true,
        .expert_only = true,
        .leader_host = "127.0.0.1",
        .leader_port = port,
        .world_size = 2u,
        .rank = 1u,
        .rank_set = true,
        .transport = DS4_TP_TRANSPORT_NCCL,
    };
    ds4_tp_identity id = test_identity(4096u);
    mutate_mismatch(kind, &opt, &id);
    char err[256] = "";
    ds4_tp *tp = NULL;
    if (ds4_tp_create(&tp, &opt, &id, err, sizeof(err))) {
        ds4_tp_free(tp);
        child_error(1u, "mismatched hello was accepted", expected, 40);
    }
    if (!strstr(err, expected)) {
        child_error(1u, "mismatch error was not actionable", err, 41);
    }
    _exit(0);
}

static int run_mismatch_case(test_mismatch_kind kind,
                             const char *name,
                             const char *expected) {
    const int port = reserve_loopback_port();
    if (port <= 0) return 0;
    const pid_t pid = fork();
    if (pid == 0) run_mismatch_worker(port, kind, expected);
    if (pid < 0) return 0;
    pid_t pids[1] = {pid};

    alarm(30);
    ds4_tp_options opt = {
        .role = DS4_TP_LEADER,
        .requested = true,
        .expert_only = true,
        .listen_host = "127.0.0.1",
        .listen_port = port,
        .world_size = 2u,
        .rank = 0u,
        .rank_set = true,
        .transport = DS4_TP_TRANSPORT_NCCL,
    };
    const ds4_tp_identity id = test_identity(4096u);
    char err[256] = "";
    ds4_tp *tp = NULL;
    int ok = !ds4_tp_create(&tp, &opt, &id, err, sizeof(err)) &&
             strstr(err, expected) != NULL;
    if (tp) ds4_tp_free(tp);
    if (!ok) {
        fprintf(stderr, "test_tp_protocol: %s mismatch: %s\n", name, err);
        terminate_workers(pids, 1u);
        alarm(0);
        return 0;
    }
    if (!wait_workers(pids, 1u)) ok = 0;
    alarm(0);
    if (ok) fprintf(stderr, "test_tp_protocol: %s mismatch rejected\n", name);
    return ok;
}

/* The actual v10 wire layout. The v11 reader consumes magic/version first so
 * it can reject this shorter hello without waiting for v11-only fields. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t role;
    uint32_t rank;
    uint32_t world;
    uint32_t collective;
    uint32_t expert_only;
    uint32_t rdma_ok;
    uint64_t gguf_bytes;
    uint32_t model_id;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t quant_bits;
    uint32_t ctx_size;
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
} test_hello_v10;

static int test_read_full(int fd, void *buf, size_t len) {
    uint8_t *p = buf;
    while (len != 0u) {
        const ssize_t n = recv(fd, p, len, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

static int test_write_full(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    while (len != 0u) {
        const ssize_t n = send(fd, p, len, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

static void run_old_version_worker(int port) {
    alarm(30);
    int fd = -1;
    for (uint32_t attempt = 0u; attempt < 1000u; attempt++) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) child_error(1u, "old-version socket failed", NULL, 50);
        struct sockaddr_in addr = {
            .sin_family = AF_INET,
            .sin_port = htons((uint16_t)port),
            .sin_addr.s_addr = htonl(INADDR_LOOPBACK),
        };
        if (connect(fd, (const struct sockaddr *)&addr, sizeof(addr)) == 0) {
            break;
        }
        close(fd);
        fd = -1;
        usleep(10000u);
    }
    if (fd < 0) child_error(1u, "old-version connect failed", NULL, 51);
    uint32_t leader_prefix[2] = {0};
    if (!test_read_full(fd, leader_prefix, sizeof(leader_prefix))) {
        close(fd);
        child_error(1u, "old-version hello read failed", NULL, 52);
    }
    test_hello_v10 hello = {
        .magic = leader_prefix[0],
        .version = DS4_TP_PROTOCOL_VERSION - 1u,
        .role = DS4_TP_WORKER,
        .rank = 1u,
        .world = 2u,
        .collective = 1u,
        .expert_only = 1u,
        .gguf_bytes = UINT64_C(211075856448),
        .model_id = 52u,
        .n_layer = 78u,
        .n_embd = 6144u,
        .n_vocab = 128000u,
        .quant_bits = 2u,
        .ctx_size = 4096u,
        .gate_slot_start = 7u,
        .gate_slot_step = 2u,
        .gates_per_token = 75u,
    };
    if (!test_write_full(fd, &hello, sizeof(hello))) {
        close(fd);
        child_error(1u, "old-version hello write failed", NULL, 53);
    }
    close(fd);
    _exit(0);
}

static int run_old_version_case(void) {
    const int port = reserve_loopback_port();
    if (port <= 0) return 0;
    const pid_t pid = fork();
    if (pid == 0) run_old_version_worker(port);
    if (pid < 0) return 0;
    pid_t pids[1] = {pid};

    alarm(30);
    const ds4_tp_options opt = {
        .role = DS4_TP_LEADER,
        .requested = true,
        .expert_only = true,
        .listen_host = "127.0.0.1",
        .listen_port = port,
        .world_size = 2u,
        .rank = 0u,
        .rank_set = true,
        .transport = DS4_TP_TRANSPORT_NCCL,
    };
    const ds4_tp_identity id = test_identity(4096u);
    char err[256] = "";
    ds4_tp *tp = NULL;
    int ok = !ds4_tp_create(&tp, &opt, &id, err, sizeof(err)) &&
             strstr(err, "protocol version") != NULL;
    if (tp) ds4_tp_free(tp);
    if (!ok) {
        fprintf(stderr, "test_tp_protocol: version mismatch: %s\n", err);
        terminate_workers(pids, 1u);
        alarm(0);
        return 0;
    }
    if (!wait_workers(pids, 1u)) ok = 0;
    alarm(0);
    if (ok) fprintf(stderr, "test_tp_protocol: version mismatch rejected\n");
    return ok;
}

static int run_partition_cases(void) {
    static const uint32_t worlds[] = {2u, 4u, 8u};
    for (uint32_t wi = 0u; wi < sizeof(worlds) / sizeof(worlds[0]); wi++) {
        const uint32_t world = worlds[wi];
        uint32_t next = 0u;
        uint32_t sum = 0u;
        for (uint32_t rank = 0u; rank < world; rank++) {
            uint32_t base = UINT32_MAX;
            uint32_t count = 0u;
            if (!ds4_tp_partition(257u, rank, world, &base, &count) ||
                base != next || count == 0u) {
                fprintf(stderr,
                        "test_tp_protocol: %u-rank partition failed at rank %u\n",
                        world, rank);
                return 0;
            }
            next = base + count;
            sum += count;
        }
        if (next != 257u || sum != 257u) {
            fprintf(stderr,
                    "test_tp_protocol: %u-rank partition has a gap/overlap\n",
                    world);
            return 0;
        }
    }
    uint32_t base = 0u;
    uint32_t count = 0u;
    if (ds4_tp_partition(256u, 4u, 4u, &base, &count) ||
        ds4_tp_partition(0u, 0u, 2u, &base, &count) ||
        ds4_tp_partition(1u, 0u, 4u, &base, &count)) {
        fprintf(stderr, "test_tp_protocol: invalid partition accepted\n");
        return 0;
    }
    for (uint32_t rank = 0u; rank < 8u; rank++) {
        if (!ds4_tp_partition(384u, rank, 8u, &base, &count) ||
            base != rank * 48u || count != 48u) {
            fprintf(stderr,
                    "test_tp_protocol: Pro EP8 ownership failed at rank %u\n",
                    rank);
            return 0;
        }
    }
    fprintf(stderr, "test_tp_protocol: 2/4/8-rank partitions OK\n");
    return 1;
}

static int run_memory_plan_cases(void) {
    const uint64_t gib = UINT64_C(1024) * 1024u * 1024u;
    const ds4_memory_plan base = {
        .model_span_bytes = 60u * gib,
        .mandatory_artifact_bytes = 10u * gib,
        .shared_workspace_bytes = 2u * gib,
        .ordered_reduce_bytes = 1u * gib,
        .per_session_bytes = 1u * gib,
        .working_set_bytes = 128u * gib,
        .reserve_bytes = 32u * gib,
        .session_count = 1u,
        .ordered_reduce_requested = true,
    };
    const uint32_t sessions[] = {1u, 8u, 16u};
    for (uint32_t i = 0u; i < sizeof(sessions) / sizeof(sessions[0]); i++) {
        ds4_memory_plan plan = base;
        plan.session_count = sessions[i];
        const uint64_t expected = (73u + sessions[i]) * gib;
        if (!ds4_memory_plan_evaluate(&plan) ||
            !plan.admitted || plan.required_bytes != expected ||
            plan.budget_bytes != 96u * gib) {
            fprintf(stderr,
                    "test_tp_protocol: %u-session memory plan mismatch\n",
                    sessions[i]);
            return 0;
        }
    }

    ds4_memory_plan fast = base;
    fast.optional_fast_artifact_bytes = 5u * gib;
    fast.fast_aligned_requested = true;
    if (!ds4_memory_plan_evaluate(&fast) ||
        fast.required_bytes != 79u * gib) {
        fprintf(stderr, "test_tp_protocol: fast-artifact memory plan mismatch\n");
        return 0;
    }

    ds4_memory_plan insufficient = base;
    insufficient.working_set_bytes = 96u * gib;
    if (ds4_memory_plan_evaluate(&insufficient) ||
        insufficient.admitted || insufficient.budget_bytes != 64u * gib ||
        insufficient.required_bytes != 74u * gib) {
        fprintf(stderr,
                "test_tp_protocol: insufficient-budget memory plan mismatch\n");
        return 0;
    }

    ds4_memory_plan overflow = base;
    overflow.model_span_bytes = UINT64_MAX;
    overflow.working_set_bytes = UINT64_MAX;
    overflow.reserve_bytes = 0u;
    if (ds4_memory_plan_evaluate(&overflow) || overflow.admitted ||
        overflow.required_bytes != UINT64_MAX) {
        fprintf(stderr, "test_tp_protocol: overflowed memory plan admitted\n");
        return 0;
    }
    ds4_memory_plan no_sessions = base;
    no_sessions.session_count = 0u;
    no_sessions.admitted = true;
    if (ds4_memory_plan_evaluate(&no_sessions) || no_sessions.admitted) {
        fprintf(stderr, "test_tp_protocol: zero-session memory plan admitted\n");
        return 0;
    }
    fprintf(stderr,
            "test_tp_protocol: exact/fast 1/8/16-session memory plans OK\n");
    return 1;
}

static int parse_world(uint32_t world, ds4_tp_options *tp,
                       char *err, size_t errlen) {
    char world_buf[16];
    snprintf(world_buf, sizeof(world_buf), "%u", world);
    char *argv[] = {"test", "--tensor-parallel-world", world_buf};
    int index = 1;
    memset(tp, 0, sizeof(*tp));
    return ds4_tp_parse_cli_arg(argv[index], &index, 3, argv, tp,
                                err, errlen) == DS4_TP_CLI_MATCHED &&
           index == 2 && tp->world_size == world;
}

static int run_world_option_cases(void) {
    static const uint32_t worlds[] = {2u, 4u, 8u};
    char err[256];
    for (uint32_t wi = 0u; wi < sizeof(worlds) / sizeof(worlds[0]); wi++) {
        ds4_tp_options tp;
        err[0] = '\0';
        if (!parse_world(worlds[wi], &tp, err, sizeof(err))) {
            fprintf(stderr, "test_tp_protocol: world %u parse failed: %s\n",
                    worlds[wi], err);
            return 0;
        }
    }
    ds4_tp_options tp;
    err[0] = '\0';
    if (parse_world(3u, &tp, err, sizeof(err)) ||
        strstr(err, "must be 2, 4, or 8") == NULL) {
        fprintf(stderr, "test_tp_protocol: invalid world parse mismatch: %s\n",
                err);
        return 0;
    }
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    static const struct {
        uint32_t world;
        bool expert_only;
        bool accepted;
    } cases[] = {
        {2u, true, true}, {4u, true, true}, {8u, true, true},
        {2u, false, true}, {4u, false, true}, {8u, false, false},
    };
    for (uint32_t i = 0u; i < sizeof(cases) / sizeof(cases[0]); i++) {
        ds4_engine_options opt;
        memset(&opt, 0, sizeof(opt));
        opt.backend = DS4_BACKEND_CUDA;
        opt.tp.role = DS4_TP_LEADER;
        opt.tp.requested = true;
        opt.tp.expert_only = cases[i].expert_only;
        opt.tp.world_size = cases[i].world;
        opt.tp.transport = DS4_TP_TRANSPORT_NCCL;
        err[0] = '\0';
        const int accepted = ds4_tp_validate_engine_options(
                &opt, err, sizeof(err));
        if (!!accepted != cases[i].accepted ||
            (!cases[i].accepted && strstr(err, "TP8 unsupported; use EP8") == NULL)) {
            fprintf(stderr,
                    "test_tp_protocol: %s%u validation mismatch: %s\n",
                    cases[i].expert_only ? "EP" : "TP", cases[i].world, err);
            return 0;
        }
    }
#endif
    const ds4_tp_options direct_tp8 = {
        .role = DS4_TP_LEADER,
        .requested = true,
        .expert_only = false,
        .listen_host = "127.0.0.1",
        .listen_port = 1,
        .world_size = 8u,
        .rank = 0u,
        .rank_set = true,
        .transport = DS4_TP_TRANSPORT_NCCL,
    };
    const ds4_tp_identity direct_id = test_identity(4096u);
    ds4_tp *direct_tp = NULL;
    err[0] = '\0';
    if (ds4_tp_create(&direct_tp, &direct_tp8, &direct_id,
                      err, sizeof(err)) || direct_tp != NULL ||
        strstr(err, "TP8 unsupported; use EP8") == NULL) {
        if (direct_tp) ds4_tp_free(direct_tp);
        fprintf(stderr,
                "test_tp_protocol: direct TP8 create was not rejected: %s\n",
                err);
        return 0;
    }
    fprintf(stderr, "test_tp_protocol: EP/TP world option cases OK\n");
    return 1;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    setenv("DS4_TP_TIMEOUT_SEC", "10", 1);
    if (!run_partition_cases() ||
        !run_memory_plan_cases() ||
        !run_world_option_cases() ||
        !run_success_case(2, true) ||
        !run_success_case(2, false) ||
        !run_success_case(4, true) ||
        !run_success_case(4, false) ||
        !run_success_case(8, true) ||
        !run_mismatch_case(TEST_MISMATCH_CONTEXT, "context",
                           "context capacity mismatch") ||
        !run_mismatch_case(TEST_MISMATCH_QUANT_PROFILE, "quant profile",
                           "quant profile mismatch") ||
        !run_mismatch_case(TEST_MISMATCH_LAYOUT, "layout fingerprint",
                           "layout fingerprint mismatch") ||
        !run_mismatch_case(TEST_MISMATCH_WORLD, "world", "invalid peer topology") ||
        !run_mismatch_case(TEST_MISMATCH_MODE, "parallel mode",
                           "parallel mode mismatch") ||
        !run_old_version_case() ||
        !run_failed_preflight_case() ||
        !run_failed_batch_case()) {
        fprintf(stderr, "test_tp_protocol: FAIL\n");
        return 1;
    }
    fprintf(stderr, "test_tp_protocol: PASS\n");
    return 0;
}
