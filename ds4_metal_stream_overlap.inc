/* Early commit of streamed Qwen decode layers.
 *
 * A streamed decode layer starts with a full GPU drain: the CPU needs the
 * router's expert ids before it can load experts. Everything encoded after
 * that drain (the layer's experts, the reduce, the next layer's attention
 * and router) used to sit in one command buffer that was committed only by
 * the next layer's drain, so the GPU idled while the CPU encoded it. Now the
 * experts' command buffer is committed as soon as they are encoded, and the
 * GPU runs them while the CPU encodes the
 * next layer. The next drain still waits for both buffers, in queue order.
 * On by default; DS4_QWEN4_STREAM_EARLY_COMMIT=0 turns it off.
 *
 * Included by ds4_metal.m before ds4_gpu_qwen4_moe_stream_tensor. */

static int qwen4_stream_early_commit_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("DS4_QWEN4_STREAM_EARLY_COMMIT");
        enabled = !(env && strcmp(env, "0") == 0);
    }
    return enabled;
}

/* Called at the end of a streamed MoE; had_batch is whether the caller owns
 * an open batch that continues after this layer. */
static int qwen4_stream_early_commit(int ok, int had_batch, uint32_t n_tokens) {
    if (!ok || !had_batch || !g_batch_cb || !g_batch_has_work || n_tokens > 3u ||
        !qwen4_stream_early_commit_enabled()) return ok;
    return ds4_gpu_flush_commands();
}
