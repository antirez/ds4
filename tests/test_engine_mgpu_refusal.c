/* CUDA-build regression test for the multi-tier CPU-spill refusal path.
 *
 * Half-B (B7) lifted the GPU-only refusal: GPU-only multi-tier placements
 * now run normally. CPU-spill placements (where any layer lands on the
 * CPU tier because GPU VRAM is too tight) continue to refuse with stderr
 * naming the wave-3b follow-up `mgpu-graph-session-cpu-spill`.
 *
 * This test forces CPU spill by configuring two GPUs with deliberately
 * tiny VRAM budgets (1 GiB each, not enough to hold the model), then
 * asserts that engine creation:
 *   - returns nonzero,
 *   - leaves the engine pointer NULL,
 *   - emits stderr containing "multi-GPU layout" (proves the layout
 *     printer ran — init_multi succeeded and placement was computed),
 *   - emits stderr containing "mgpu-graph-session-cpu-spill" (proves
 *     the CPU-spill branch fired with the new wording).
 *
 * Requires DS4_TEST_MODEL to be set to a valid GGUF path. */

#include "ds4.h"
#include "ds4_gpu_mgpu.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);       \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static int read_file_to_buf(const char *path, char **out_buf, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 1; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return 1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 1; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return 1; }
    size_t r = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[r] = '\0';
    *out_buf = buf;
    *out_len = (long)r;
    return 0;
}

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    fprintf(stderr, "test_engine_mgpu_refusal: %d CUDA devices visible\n", dev_count);
    if (dev_count < 2) {
        fprintf(stderr, "  skipping (need >= 2 devices)\n");
        return 0;
    }

    const char *model_path = getenv("DS4_TEST_MODEL");
    if (!model_path || !model_path[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL not set\n");
        return 1;
    }

    /* Capture stderr to a temp file. We use a known path so failures are
     * easy to inspect; the file is removed at the end on success. */
    const char *cap_path = "/tmp/ds4_mgpu_refusal_stderr.log";
    (void)unlink(cap_path);
    fflush(stderr);
    int saved_stderr = dup(fileno(stderr));
    CHECK(saved_stderr >= 0, "dup stderr");
    FILE *redir = freopen(cap_path, "w+", stderr);
    CHECK(redir != NULL, "freopen stderr");

    /* Build a 2-GPU config. */
    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    /* Pick small-but-not-too-small budgets:
     *   - each must exceed the per-tier graph overhead (~4 GiB at default
     *     ctx) so the pre-subtract refusal at engine_classify_multi_tier
     *     doesn't fire (that path returns early without printing the
     *     "multi-GPU layout:" header this test asserts on);
     *   - but the combined budget must be far below the model's tensor
     *     bytes so the packer is forced to spill some entries to CPU,
     *     triggering the CPU-spill refusal path this test exercises.
     * 8 GiB per GPU = 16 GiB total: leaves ~4 GiB usable per tier after
     * the pre-subtract — enough for some layer entries — while the
     * ~36 GiB model forces spill. */
    cfg.vram_bytes[0] = (size_t)8ull * 1024u * 1024u * 1024u;   /* 8 GiB */
    cfg.vram_bytes[1] = (size_t)8ull * 1024u * 1024u * 1024u;   /* 8 GiB */
    cfg.safety_margin_bytes = 0;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 1;
    opt.warm_weights = false;
    opt.quality = false;

    ds4_engine *engine = NULL;
    int rc = ds4_engine_create_with_gpu_config(&engine, &opt, &cfg);

    /* Restore stderr so subsequent prints reach the terminal. */
    fflush(stderr);
    FILE *sink = freopen("/dev/null", "w", stderr); /* close redir's FILE* cleanly */
    CHECK(sink != NULL, "freopen stderr sink");
    int err_fd = fileno(stderr);
    if (err_fd >= 0) {
        (void)dup2(saved_stderr, err_fd);
        (void)close(saved_stderr);
    }

    fprintf(stderr, "  engine_create_with_gpu_config -> rc=%d, engine=%p\n",
            rc, (void *)engine);

    /* Read captured stderr. */
    char *cap = NULL; long cap_len = 0;
    int read_rc = read_file_to_buf(cap_path, &cap, &cap_len);
    if (read_rc != 0) {
        fprintf(stderr, "FAIL: could not read captured stderr %s\n", cap_path);
        return 1;
    }
    fprintf(stderr, "  captured stderr (%ld bytes):\n----\n%s\n----\n",
            cap_len, cap);

    CHECK(rc != 0, "engine_create should refuse multi-tier and return nonzero");
    CHECK(engine == NULL, "engine pointer should be NULL on refusal");
    CHECK(strstr(cap, "multi-GPU layout") != NULL,
          "stderr must contain 'multi-GPU layout' (init_multi must have succeeded "
          "and layout must have been printed before refusal)");
    CHECK(strstr(cap, "mgpu-graph-session-cpu-spill") != NULL,
          "stderr must name the wave-3b CPU-spill follow-up task");
    CHECK(strstr(cap, "CPU-spill placement detected") != NULL,
          "stderr must contain the CPU-spill diagnostic line");

    free(cap);
    (void)unlink(cap_path);
    fprintf(stderr, "test_engine_mgpu_refusal PASS\n");
    return 0;
}
