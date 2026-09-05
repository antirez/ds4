#include "../ds4_tp.c"
#include <assert.h>

int main(void) {
    int fd[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    ds4_tp leader = { .control_fd = fd[0] };
    ds4_tp worker = { .control_fd = fd[1] };
    char err[256] = "";
    ds4_tp_command cmd;
    assert(DS4_TP_PROTOCOL_VERSION == 10);
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
