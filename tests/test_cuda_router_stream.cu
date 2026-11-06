/* The batch router selection (ds4_gpu_router_select_batch_tensor) must run
 * on the active stream with every launch override (warp, parallel and scalar
 * kernel): captured in a decode graph and replayed with new logits, and in a
 * side-stream region behind earlier side work, each equal to the same
 * override's eager output. */
#include "ds4_gpu.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static const uint32_t ROWS = 6, EXPERTS = 256;
static const uint64_t LOGIT_BYTES = (uint64_t)ROWS * EXPERTS * sizeof(float);
static const uint64_t SEL_BYTES = (uint64_t)ROWS * 6 * sizeof(int32_t);
static uint32_t rng = 1234567;
static float uniform(float lo, float hi) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return lo + (hi - lo) * (float)(rng >> 8) / 16777216.0f;
}

struct router {
    std::vector<float> bias;
    ds4_gpu_tensor *logits, *tokens, *sel, *w, *prob;
};
struct outputs {
    std::vector<uint8_t> sel, w, prob;
    bool operator==(const outputs &o) const { return sel == o.sel && w == o.w && prob == o.prob; }
};

static void write_logits(router &r) {
    std::vector<float> l(ROWS * EXPERTS);
    for (float &x : l) x = uniform(-8.0f, 8.0f);
    CHECK(ds4_gpu_tensor_write(r.logits, 0, l.data(), LOGIT_BYTES));
}
/* Outputs a run must overwrite. */
static void sentinel(router &r) {
    std::vector<uint8_t> s(LOGIT_BYTES, 0xa5);
    CHECK(ds4_gpu_tensor_write(r.sel, 0, s.data(), SEL_BYTES));
    CHECK(ds4_gpu_tensor_write(r.w, 0, s.data(), SEL_BYTES));
    CHECK(ds4_gpu_tensor_write(r.prob, 0, s.data(), LOGIT_BYTES));
}
static int run(router &r) {
    return ds4_gpu_router_select_batch_tensor(r.sel, r.w, r.prob, r.bias.data(), r.bias.size() * sizeof(float), 0, 0, 0,
                                              1, 0, true, false, r.logits, r.tokens, EXPERTS, 6, 1.5f, ROWS);
}
static outputs read_outputs(router &r) {
    CHECK(ds4_gpu_synchronize());
    outputs o;
    o.sel.resize(SEL_BYTES); o.w.resize(SEL_BYTES); o.prob.resize(LOGIT_BYTES);
    CHECK(ds4_gpu_tensor_read(r.sel, 0, o.sel.data(), SEL_BYTES));
    CHECK(ds4_gpu_tensor_read(r.w, 0, o.w.data(), SEL_BYTES));
    CHECK(ds4_gpu_tensor_read(r.prob, 0, o.prob.data(), LOGIT_BYTES));
    return o;
}
static outputs eager(router &r) {
    sentinel(r);
    CHECK(run(r) == 1);
    return read_outputs(r);
}

int main(int argc, char **argv) {
    (void)argv;
    CHECK(argc == 1);
    CHECK(setenv("DS4_CUDA_DECODE_GRAPHS", "1", 1) == 0);
    CHECK(ds4_gpu_init());
    router r;
    r.bias.resize(EXPERTS);
    for (float &x : r.bias) x = uniform(-0.5f, 0.5f);
    /* The bias reaches the device from a model file, as in the engine. */
    const uint64_t span_off = 0, span_size = r.bias.size() * sizeof(float);
    CHECK(ds4_gpu_set_model_map_spans(r.bias.data(), span_size, &span_off, &span_size, 1, 0));
    char path[] = "/tmp/ds4-router-stream-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    FILE *file = fdopen(dup(fd), "wb");
    CHECK(file && fwrite(r.bias.data(), 1, span_size, file) == span_size);
    CHECK(fclose(file) == 0 && unlink(path) == 0);
    CHECK(ds4_gpu_set_model_fd_for_map(fd, r.bias.data()));

    CHECK((r.logits = ds4_gpu_tensor_alloc(LOGIT_BYTES)) && (r.tokens = ds4_gpu_tensor_alloc(ROWS * sizeof(int32_t))) &&
          (r.sel = ds4_gpu_tensor_alloc(SEL_BYTES)) && (r.w = ds4_gpu_tensor_alloc(SEL_BYTES)) &&
          (r.prob = ds4_gpu_tensor_alloc(LOGIT_BYTES)));
    const std::vector<int32_t> tok(ROWS, 0);
    CHECK(ds4_gpu_tensor_write(r.tokens, 0, tok.data(), tok.size() * sizeof(int32_t)));
    const uint64_t big_bytes = 256ull << 20;
    ds4_gpu_tensor *big_src = ds4_gpu_tensor_alloc(big_bytes), *big_dst = ds4_gpu_tensor_alloc(big_bytes),
                   *stage = ds4_gpu_tensor_alloc(LOGIT_BYTES);
    CHECK(big_src && big_dst && stage);
    {
        const std::vector<uint8_t> fill(big_bytes, 0x3c);
        CHECK(ds4_gpu_tensor_write(big_src, 0, fill.data(), big_bytes));
    }
    const std::vector<float> old(ROWS * EXPERTS, 1.0f);

    struct launch_mode { const char *name, *env; };
    const launch_mode modes[] = {{"warp", NULL}, {"parallel", "DS4_CUDA_NO_WARP_ROUTER_SELECT"},
                                 {"scalar", "DS4_CUDA_NO_PARALLEL_ROUTER_SELECT"}};
    for (const launch_mode &m : modes) {
        if (m.env) CHECK(setenv(m.env, "1", 1) == 0);

        /* Decode-graph capture and replay: new logits every round. */
        if (ds4_gpu_decode_graphs_supported()) {
            ds4_decode_graph_key key = {};
            key.il = 63; key.island = 2; key.variant = 129; key.cur_hc = r.logits;   /* valid, unused island */
            unsigned captures = 0, replays = 0;
            for (unsigned round = 0; round < 6; round++) {
                write_logits(r);
                const outputs want = eager(r);
                sentinel(r);
                const int graph = ds4_gpu_decode_graph_begin(&key);
                CHECK(graph == (round == 0 ? -1 : round == 1 ? 0 : 1));
                if (graph == 1) {
                    replays++;
                } else {
                    CHECK(run(r) == 1);
                    if (graph == 0) { CHECK(ds4_gpu_decode_graph_end(&key) == 0); captures++; }
                }
                if (!(read_outputs(r) == want)) {
                    fprintf(stderr, "router stream: %s: decode graph round %u differs from eager\n", m.name, round);
                    return 1;
                }
            }
            CHECK(captures == 1 && replays == 4);
            ds4_gpu_decode_graphs_invalidate();
            printf("router stream: %s: captured in a decode graph, 4 replays with new logits equal to eager\n", m.name);
        } else {
            printf("router stream: %s: SKIP decode graphs: not supported here\n", m.name);
        }

        /* Side-stream region: the selection must read the logits a side copy
         * wrote behind a long side copy. */
        unsigned regions = 0;
        for (unsigned iter = 0; iter < 4; iter++) {
            write_logits(r);
            const outputs want = eager(r);
            CHECK(ds4_gpu_tensor_copy(stage, 0, r.logits, 0, LOGIT_BYTES));
            CHECK(ds4_gpu_synchronize());
            CHECK(ds4_gpu_tensor_write(r.logits, 0, old.data(), LOGIT_BYTES));
            sentinel(r);
            const int side = ds4_gpu_side_begin();
            CHECK(side >= 0);
            if (side == 0) break;
            CHECK(ds4_gpu_tensor_copy(big_dst, 0, big_src, 0, big_bytes));
            CHECK(ds4_gpu_tensor_copy(r.logits, 0, stage, 0, LOGIT_BYTES));
            CHECK(run(r) == 1);
            CHECK(ds4_gpu_side_end() == 1);
            CHECK(ds4_gpu_side_join() == 1);
            if (!(read_outputs(r) == want)) {
                fprintf(stderr, "router stream: %s: side region %u read logits out of stream order\n", m.name, iter);
                return 1;
            }
            regions++;
        }
        if (regions) printf("router stream: %s: %u side-stream regions in order behind side copies\n", m.name, regions);
        else printf("router stream: %s: SKIP side regions: side stream not available here\n", m.name);
        if (m.env) CHECK(unsetenv(m.env) == 0);
    }

    ds4_gpu_tensor_free(stage); ds4_gpu_tensor_free(big_dst); ds4_gpu_tensor_free(big_src);
    ds4_gpu_tensor_free(r.logits); ds4_gpu_tensor_free(r.tokens); ds4_gpu_tensor_free(r.sel);
    ds4_gpu_tensor_free(r.w); ds4_gpu_tensor_free(r.prob);
    ds4_gpu_cleanup();
    close(fd);
    printf("router stream: PASS\n");
    return 0;
}
