/* Physical two-host transport test; no model or GPU is needed.
 *
 * The command phase mirrors leader -> worker command frames in order: 300
 * back-to-back EVALs (the RDMA mailbox ring wraps several times and the
 * worker naps to force credit waits), 300 VERIFY/commit/DRAFT sequences,
 * SYNC frames larger than a mailbox slot
 * (marker + TCP) each followed by short commands, REWIND, EVAL_BATCH and
 * SESSION_CREATE with their ACKs, and STOP.
 *   DS4_TEST_TP_EXPECT_MAILBOX=0|1  require the agreed mailbox state
 *                                   (set DS4_TP_DISABLE_RDMA_MAILBOX on one
 *                                   rank and expect 0 on both).
 *   DS4_TEST_TP_PEER_EXIT=leader|worker  that rank exits without STOP; the
 *                                   other must detect it and pass (leader: it
 *                                   exits after the worker acknowledges the
 *                                   300 EVALs, so the worker is idle-waiting).
 *   DS4_TEST_TP_STOP_FREE=1          the leader frees its context (queue
 *                                   pair) right after STOP returns, as the
 *                                   frontends do; the worker must still get
 *                                   STOP.  The link phases are skipped.
 *
 * After every verify block the link phase also runs the verify-head split
 * exchanges (RDMA mailbox only; without it both must refuse): the per-row
 * top-1 pairs both ways (ds4_tp_small_exchange), one round after a 12 ms
 * idle gap on the leader to time it after the drafter's idle period, then
 * the worker's committed half-vocabulary row (ds4_tp_head_row_send/recv).
 * In phase 1 the leader leaves the row unread, as a rolled-back block does;
 * phase 2 must receive its own row, keyed by the verify block. */
#include "ds4_tp.h"
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s (%s)\n", \
    __FILE__, __LINE__, #x, error); goto done; } } while (0)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

#define CMD_SESSION 7u
#define CMD_EVALS 300u
#define CMD_SYNC_TOKENS 3000u

static int sync_token(uint32_t i, uint32_t round) { return (int)((i * 7u + round) % 1000u); }

static int32_t commit_mode(uint32_t i) {
    const int32_t modes[] = {DS4_TP_VERIFY_COMMIT_FULL,
        DS4_TP_VERIFY_COMMIT_PREFIX, DS4_TP_VERIFY_ROLLBACK_REPLAY};
    return modes[i % 3u];
}

static int recv_one(ds4_tp *tp, ds4_tp_command *c, ds4_tp_frame_type type,
                    char *error, size_t errlen) {
    if (!ds4_tp_recv_command(tp, c, error, errlen)) return 0;
    if (c->type != type) {
        snprintf(error, errlen, "command type %d, expected %d", (int)c->type, (int)type);
        ds4_tp_command_free(c);
        return 0;
    }
    return 1;
}

/* Returns 1 to continue with the link phases, 2 when a peer-exit scenario
 * passed, 3 after STOP in DS4_TEST_TP_STOP_FREE mode (the test ends there for
 * both), 0 on failure. */
static int command_phase(ds4_tp *tp, int rank, char *error, size_t errlen) {
    const bool active = ds4_tp_command_mailbox_active(tp);
    fprintf(stderr, "rank=%d command mailbox %s\n", rank, active ? "active" : "inactive");
    const char *expect = getenv("DS4_TEST_TP_EXPECT_MAILBOX");
    if (expect && (atoi(expect) != 0) != active) {
        snprintf(error, errlen, "command mailbox %s, expected %s",
                 active ? "active" : "inactive", expect);
        return 0;
    }
    const char *exit_rank = getenv("DS4_TEST_TP_PEER_EXIT");
    const bool leader_exits = exit_rank && !strcmp(exit_rank, "leader");
    const bool worker_exits = exit_rank && !strcmp(exit_rank, "worker");
    int *tokens = malloc(CMD_SYNC_TOKENS * sizeof(int));
    if (!tokens) return 0;
    int ok = 0;
    const double t0 = now();
    if (rank == 0) {
        if (worker_exits) {
            for (uint32_t i = 0; i < 1000000u; i++) {
                if (!ds4_tp_send_eval(tp, CMD_SESSION, i, (int)i)) {
                    fprintf(stderr, "rank=0 exited worker detected after %u commands, %.0f ms: PASS\n",
                            i, (now() - t0) * 1e3);
                    ok = 2;
                    goto out;
                }
            }
            snprintf(error, errlen, "sends kept succeeding after the worker exited");
            goto out;
        }
        for (uint32_t i = 0; i < CMD_EVALS; i++) {
            if (!ds4_tp_send_eval(tp, CMD_SESSION, i, (int)(i * 3u))) {
                snprintf(error, errlen, "eval %u send failed", i);
                goto out;
            }
        }
        if (leader_exits) {
            /* Exit only once the worker confirms it consumed every command,
             * so the test is about detecting the closed peer while idle. */
            if (!ds4_tp_wait_command_ack(tp, CMD_SESSION, "eval stream", error, errlen)) goto out;
            fprintf(stderr, "rank=0 worker consumed %u commands; exiting without STOP\n", CMD_EVALS);
            fflush(stderr);
            _exit(0);
        }
        for (uint32_t i = 0; i < CMD_EVALS; i++) {
            const int draft[] = {7, 19, 5};
            if (!ds4_tp_send_verify(tp, CMD_SESSION, draft, 3) ||
                !ds4_tp_send_verify_commit(tp, commit_mode(i), i % 3u ? 2 : 0) ||
                !ds4_tp_send_dspark_draft(tp, CMD_SESSION, (int)i, (int)i + 1,
                                           DS4_TP_DSPARK_DRAFT)) {
                snprintf(error, errlen, "verify/commit/draft %u send failed", i);
                goto out;
            }
        }
        for (uint32_t round = 0; round < 3; round++) {
            for (uint32_t i = 0; i < CMD_SYNC_TOKENS; i++) tokens[i] = sync_token(i, round);
            if (!ds4_tp_send_sync(tp, CMD_SESSION, tokens, CMD_SYNC_TOKENS) ||
                !ds4_tp_send_rewind(tp, CMD_SESSION, 42 + (int)round) ||
                !ds4_tp_send_eval(tp, CMD_SESSION, 1000u + round, 5)) {
                snprintf(error, errlen, "sync round %u send failed", round);
                goto out;
            }
        }
        const ds4_tp_batch_item items[3] = {
            {CMD_SESSION, 11, 0}, {CMD_SESSION + 1u, 12, 0}, {CMD_SESSION + 2u, 13, 0},
        };
        if (!ds4_tp_send_eval_batch(tp, items, 3) ||
            !ds4_tp_wait_command_ack(tp, CMD_SESSION, "eval batch", error, errlen) ||
            !ds4_tp_send_session_create(tp, 9, 4096) ||
            !ds4_tp_wait_command_ack(tp, 9, "session create", error, errlen) ||
            !ds4_tp_send_stop(tp)) {
            if (!error[0]) snprintf(error, errlen, "batch/ack/stop failed");
            goto out;
        }
        if (getenv("DS4_TEST_TP_STOP_FREE")) {
            ok = 3;
            goto out;
        }
    } else {
        if (worker_exits) {
            fprintf(stderr, "rank=1 exiting without reading commands\n");
            fflush(stderr);
            _exit(0);
        }
        ds4_tp_command c;
        for (uint32_t i = 0; i < CMD_EVALS; i++) {
            if (!recv_one(tp, &c, DS4_TP_FRAME_EVAL, error, errlen)) goto out;
            const bool good = c.session_id == CMD_SESSION && c.seq == i && c.value == (int)(i * 3u);
            ds4_tp_command_free(&c);
            if (!good) { snprintf(error, errlen, "eval %u out of order or corrupt", i); goto out; }
            if (i % 50u == 49u) usleep(2000);   /* let the leader run out of credits */
        }
        if (leader_exits) {
            if (!ds4_tp_send_command_ack(tp, CMD_SESSION, 0)) {
                snprintf(error, errlen, "eval stream ack failed");
                goto out;
            }
            if (ds4_tp_recv_command(tp, &c, error, errlen)) {
                ds4_tp_command_free(&c);
                snprintf(error, errlen, "command received after the leader exited");
                goto out;
            }
            fprintf(stderr, "rank=1 exited leader detected in %.0f ms: PASS\n", (now() - t0) * 1e3);
            error[0] = 0;
            ok = 2;
            goto out;
        }
        for (uint32_t i = 0; i < CMD_EVALS; i++) {
            if (!recv_one(tp, &c, DS4_TP_FRAME_VERIFY, error, errlen)) goto out;
            bool good = c.session_id == CMD_SESSION && c.n_tokens == 3 &&
                        c.tokens[0] == 7 && c.tokens[1] == 19 && c.tokens[2] == 5;
            ds4_tp_command_free(&c);
            int32_t mode = -1, count = -1;
            if (!good || !ds4_tp_recv_verify_commit(tp, &mode, &count) ||
                mode != commit_mode(i) || count != (i % 3u ? 2 : 0)) {
                snprintf(error, errlen, "verify commit %u corrupt or out of order", i);
                goto out;
            }
            if (!recv_one(tp, &c, DS4_TP_FRAME_DSPARK_DRAFT, error, errlen)) goto out;
            good = c.session_id == CMD_SESSION && c.value == (int)i &&
                   c.limit == (int)i + 1 && c.seq == DS4_TP_DSPARK_DRAFT;
            ds4_tp_command_free(&c);
            if (!good) { snprintf(error, errlen, "draft %u out of order", i); goto out; }
            if (i % 50u == 49u) usleep(2000);
        }
        for (uint32_t round = 0; round < 3; round++) {
            if (!recv_one(tp, &c, DS4_TP_FRAME_SYNC, error, errlen)) goto out;
            bool good = c.session_id == CMD_SESSION && c.n_tokens == CMD_SYNC_TOKENS;
            for (uint32_t i = 0; good && i < CMD_SYNC_TOKENS; i++) good = c.tokens[i] == sync_token(i, round);
            ds4_tp_command_free(&c);
            if (!good) { snprintf(error, errlen, "sync round %u corrupt", round); goto out; }
            if (!recv_one(tp, &c, DS4_TP_FRAME_REWIND, error, errlen)) goto out;
            good = c.value == 42 + (int)round;
            ds4_tp_command_free(&c);
            if (!recv_one(tp, &c, DS4_TP_FRAME_EVAL, error, errlen)) goto out;
            good = good && c.seq == 1000u + round && c.value == 5;
            ds4_tp_command_free(&c);
            if (!good) { snprintf(error, errlen, "short commands after sync round %u wrong", round); goto out; }
        }
        if (!recv_one(tp, &c, DS4_TP_FRAME_EVAL_BATCH, error, errlen)) goto out;
        bool good = c.n_items == 3 && c.items[0].session_id == CMD_SESSION &&
                    c.items[1].token == 12 && c.items[2].session_id == CMD_SESSION + 2u;
        ds4_tp_command_free(&c);
        if (!good || !ds4_tp_send_command_ack(tp, CMD_SESSION, 0)) {
            snprintf(error, errlen, "eval batch wrong or ack failed");
            goto out;
        }
        if (!recv_one(tp, &c, DS4_TP_FRAME_SESSION_CREATE, error, errlen)) goto out;
        good = c.session_id == 9 && c.value == 4096;
        ds4_tp_command_free(&c);
        if (!good || !ds4_tp_send_command_ack(tp, 9, 0)) {
            snprintf(error, errlen, "session create wrong or ack failed");
            goto out;
        }
        if (!recv_one(tp, &c, DS4_TP_FRAME_STOP, error, errlen)) goto out;
        ds4_tp_command_free(&c);
        if (getenv("DS4_TEST_TP_STOP_FREE")) {
            fprintf(stderr, "rank=1 STOP received from a leader that frees right after: PASS\n");
            ok = 3;
            goto out;
        }
    }
    fprintf(stderr, "rank=%d %u evals + 3 x (%u-token sync, rewind, eval) + batch/ack/create/stop in %.2f ms: PASS\n",
            rank, CMD_EVALS, CMD_SYNC_TOKENS, (now() - t0) * 1e3);
    ok = 1;
out:
    free(tokens);
    return ok;
}

static void fill(float *out, uint32_t n, unsigned rank, uint32_t epoch) {
    for (uint32_t i = 0; i < n; i++) out[i] = epoch + i % 127u + rank * 1000u;
}

static int equal(const float *in, uint32_t n, unsigned peer, uint32_t epoch) {
    for (uint32_t i = 0; i < n; i++) {
        if (in[i] != epoch + i % 127u + peer * 1000u) {
            fprintf(stderr, "payload mismatch epoch=%u index=%u value=%g\n", epoch, i, in[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 7) {
        fprintf(stderr, "usage: %s RANK COORDINATOR PORT tcp|rdma [DEVICE GID]\n", argv[0]);
        return 2;
    }
    const int rank = atoi(argv[1]), port = atoi(argv[3]);
    if ((rank != 0 && rank != 1) || port <= 0 || port > 65535 ||
        (strcmp(argv[4], "rdma") && strcmp(argv[4], "tcp"))) return 2;
    ds4_tp_options opt = {.role = rank ? DS4_TP_WORKER : DS4_TP_LEADER,
        .listen_host = argv[2], .leader_host = argv[2],
        .listen_port = port, .leader_port = port,
        .transport = !strcmp(argv[4], "rdma") ? DS4_TP_TRANSPORT_RDMA : DS4_TP_TRANSPORT_TCP};
    if (argc == 7) {
        opt.rdma_device = argv[5];
        opt.rdma_gid_index = atoi(argv[6]);
        opt.rdma_gid_index_set = true;
    }
    ds4_tp_identity id = {.gguf_bytes = 1, .n_layer = 40, .n_embd = 5120,
        .n_vocab = 16, .ctx_size = 8192, .gate_slot_step = 1, .gates_per_token = 80};
    char error[256] = "";
    ds4_tp *tp = NULL;
    void *slab = NULL;
    float *out = NULL, *in = NULL;
    const uint32_t width = id.n_embd, capacity = 8192u * width;
    const uint64_t vec = (uint64_t)width * sizeof(float);
    int rc = 1;
    CHECK(ds4_tp_create(&tp, &opt, &id, error, sizeof(error)));
    CHECK(ds4_tp_is_rdma(tp) == (opt.transport == DS4_TP_TRANSPORT_RDMA));
    CHECK(ds4_tp_agree_execution_mode(tp, 0, error, sizeof(error)));
    CHECK(ds4_tp_agree_execution_mode(tp, DS4_TP_MODE_EXPERT_INTERMEDIATE_SPLIT,
                                     error, sizeof(error)));
    CHECK(ds4_tp_agree_execution_mode(tp, DS4_TP_MODE_V41_SHARED_OWNER,
                                     error, sizeof(error)));
    CHECK(ds4_tp_agree_execution_mode(tp, DS4_TP_MODE_V41_SHARED_OWNER |
                                     DS4_TP_MODE_V41_SHARED_BALANCE, error, sizeof(error)));
    CHECK(ds4_tp_agree_execution_mode(tp, DS4_TP_MODE_V41_DSPARK |
                                     DS4_TP_MODE_V41_SHARED_OWNER, error, sizeof(error)));
    if (getenv("DS4_TEST_TP_MODE_MISMATCH")) {
        const int test = atoi(getenv("DS4_TEST_TP_MODE_MISMATCH"));
        const uint32_t mode = test == 3 ? DS4_TP_MODE_V41_DSPARK :
            test == 2 ? DS4_TP_MODE_V41_SHARED_BALANCE : DS4_TP_MODE_EXPERT_INTERMEDIATE_SPLIT;
        CHECK(!ds4_tp_agree_execution_mode(tp, rank ?
            mode : 0, error, sizeof(error)));
        CHECK(ds4_tp_failed(tp));
        CHECK(strstr(error, "execution mode mismatch") != NULL);
        fprintf(stderr, "rank=%d mismatched execution modes refused: PASS\n", rank);
        rc = 0;
        goto done;
    }
    slab = calloc(1, ds4_tp_slab_bytes(id.n_layer, width));
    out = malloc((size_t)capacity * sizeof(float));
    in = malloc((size_t)capacity * sizeof(float));
    CHECK(slab && out && in && ds4_tp_attach_slab(tp, slab, error, sizeof(error)));
    {
        const int commands = command_phase(tp, rank, error, sizeof(error));
        CHECK(commands != 0);
        if (commands == 2) {
            rc = 0;
            goto done;
        }
        if (commands == 3) {
            if (rank == 0) {
                ds4_tp_free(tp);   /* destroys the queue pair right after STOP */
                tp = NULL;
                fprintf(stderr, "rank=0 context freed immediately after STOP: PASS\n");
            }
            rc = 0;
            goto done;
        }
    }
    uint64_t seq = 0;
    const uint32_t rows[] = {7, 512, 2049, 8192};
    for (unsigned phase = 0; phase < sizeof(rows) / sizeof(*rows); phase++) {
        if (rank == (int)(phase % 2u)) usleep(100000);
        double start = now();
        for (unsigned step = 0; step < 160; step++) {
            const unsigned slot = step % 80u, layer = slot / 2u, gate = slot % 2u;
            const uint32_t epoch = phase * 10000u + step;
            float *a = (float *)((char *)slab + ds4_tp_slab_out_offset(tp, layer, gate));
            const float *b = (const float *)((char *)slab + ds4_tp_slab_in_offset(tp, layer, gate));
            fill(a, width, (unsigned)rank, epoch);
            CHECK(ds4_tp_gate_exchange(tp, layer, gate, ++seq));
            CHECK(equal(b, width, 1u - (unsigned)rank, epoch));
            if (!phase && !step && getenv("DS4_TEST_TP_LINK_STALL") && rank == 1) {
                fprintf(stderr, "TP_LINK_STALL_READY\n");
                fflush(stderr);
                raise(SIGSTOP);
            }
        }
        fprintf(stderr, "rank=%d phase=%u 20KiB decode exchange %.2f us: PASS\n",
            rank, phase, (now() - start) * 1e6 / 160);
        const unsigned small_rows = phase * 2u + 1u;
        CHECK(ds4_tp_batch_block_begin(tp, small_rows, 40));
        for (unsigned layer = 0; layer < 40; layer++) {
            const uint32_t epoch = phase * 10000u + layer + 200u;
            float *a = (float *)((char *)slab + ds4_tp_slab_batch_out_offset(tp, layer));
            const float *b = (const float *)((char *)slab + ds4_tp_slab_batch_in_offset(tp, layer));
            fill(a, small_rows * width, (unsigned)rank, epoch);
            CHECK(ds4_tp_batch_gate_exchange(tp, layer, small_rows, phase * 40u + layer));
            CHECK(equal(b, small_rows * width, 1u - (unsigned)rank, epoch));
        }
        CHECK(ds4_tp_batch_block_end(tp));
        if (!ds4_tp_command_mailbox_active(tp)) {
            uint32_t mine[2] = {1, 2}, peer[2];
            CHECK(!ds4_tp_small_exchange(tp, mine, peer, sizeof(mine)));
            CHECK(rank ? !ds4_tp_head_row_send(tp, 0, 64640u) : !ds4_tp_head_row_recv(tp, 0, in, 64640u));
            CHECK(!ds4_tp_failed(tp));
        } else {
            for (unsigned round = 0; round < 4; round++) {
                const uint32_t pairs = small_rows, tag = phase << 16 | round << 8;
                uint32_t mine[32], peer[32];
                for (uint32_t i = 0; i < 2u * pairs; i++) mine[i] = (uint32_t)rank << 24 | tag | i;
                if (round == 3 && rank == 0) usleep(12000);
                const double x0 = now();
                CHECK(ds4_tp_small_exchange(tp, mine, peer, pairs * 8u));
                const double x1 = now();
                for (uint32_t i = 0; i < 2u * pairs; i++) CHECK(peer[i] == ((1u - (uint32_t)rank) << 24 | tag | i));
                if (phase == 0 || round == 3)
                    fprintf(stderr, "rank=%d phase=%u round=%u%s: top-1 exchange %u B %.1f us: PASS\n",
                            rank, phase, round, round == 3 ? " after 12 ms idle" : "",
                            pairs * 8u, (x1 - x0) * 1e6);
            }
            /* One head row per verify block.  Phase 1 is a block the leader
             * rolls back without reading the row; phase 2 must then get its
             * own row, never phase 1's. */
            const uint32_t floats = 64640u, row = phase + 2u, epoch = phase * 10000u + 500u;
            const bool bad_row = phase == 0 && getenv("DS4_TEST_TP_HEAD_ROW_FAULT");
            const double h0 = now();
            if (rank == 1) {
                float *buf = ds4_tp_head_row_out(tp);
                CHECK(buf != NULL);
                fill(buf, floats, 1u, epoch);
                CHECK(ds4_tp_head_row_send(tp, row + (bad_row ? 1u : 0u), floats));
            } else if (phase != 1) {
                if (phase == 2) usleep(20000);   /* phase 1's row is long in place */
                memset(in, 0, floats * sizeof(float));
                const int received = ds4_tp_head_row_recv(tp, row, in, floats);
                if (bad_row) {
                    CHECK(!received && ds4_tp_failed(tp));
                    CHECK(!ds4_tp_send_verify_commit(tp, DS4_TP_VERIFY_ROLLBACK_REPLAY, 1));
                    CHECK(!ds4_tp_send_eval(tp, 1, seq, 7));
                    CHECK(!ds4_tp_gate_exchange(tp, 0, 0, ++seq));
                    CHECK(!ds4_tp_batch_block_begin(tp, small_rows, 40));
                } else {
                    CHECK(received && equal(in, floats, 1u, epoch));
                }
            }
            if (bad_row) {
                if (rank == 1) {
                    ds4_tp_command command;
                    CHECK(!ds4_tp_recv_command(tp, &command, error, sizeof(error)));
                    ds4_tp_command_free(&command);
                }
                fprintf(stderr, "rank=%d one-sided head failure refuses replay/decode and closes peer: PASS\n", rank);
                rc = 0;
                goto done;
            }
            fprintf(stderr, "rank=%d phase=%u head row %u x %u floats %s %.1f us: PASS\n",
                    rank, phase, row, floats,
                    rank ? "sent" : phase == 1 ? "left unread (rolled back)" : "received",
                    (now() - h0) * 1e6);
        }
        const uint32_t count = rows[phase] * width;
        fill(out, count, (unsigned)rank, phase);
        start = now();
        for (unsigned round = 0; round < 8; round++) {
            memset(in, 0, count * sizeof(float));
            CHECK(ds4_tp_big_gate_exchange(tp, round, phase * 8u + round, out, in, rows[phase] * vec));
            CHECK(equal(in, count, 1u - (unsigned)rank, phase));
        }
        fprintf(stderr, "rank=%d phase=%u rows=%u bidirectional bulk %.2f GiB/s: PASS\n",
            rank, phase, rows[phase], 16.0 * rows[phase] * vec / ((now() - start) * 1073741824.0));
    }
    rc = 0;
done:
    ds4_tp_free(tp);
    free(in); free(out); free(slab);
    return rc;
}
