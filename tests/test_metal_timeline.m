/* Native timeline regression: include the real backend so its private selected
 * readback restart can be tested without adding a production test hook. */
#include "../ds4_metal.m"

enum { TEST_K = 4096, TEST_W0 = 512, TEST_W1 = 256, TEST_GUARD = 16 };
typedef struct { ds4_gpu_tensor *base, *view; size_t n; } test_tensor;
#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "timeline: failed line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

static test_tensor test_alloc(size_t n) {
    test_tensor t = { .n = n };
    t.base = ds4_gpu_tensor_alloc((n + 2 * TEST_GUARD) * sizeof(float));
    t.view = t.base ? ds4_gpu_tensor_view(t.base, TEST_GUARD * sizeof(float), n * sizeof(float)) : NULL;
    if (!t.view || !ds4_gpu_tensor_fill_f32(t.base, -123.125f, n + 2 * TEST_GUARD)) exit(2);
    return t;
}

static int test_equal(test_tensor a, test_tensor b) {
    size_t n = a.n + 2 * TEST_GUARD, bytes = n * sizeof(float);
    uint32_t *av = malloc(bytes), *bv = malloc(bytes);
    int ok = av && bv && ds4_gpu_tensor_read(a.base, 0, av, bytes) && ds4_gpu_tensor_read(b.base, 0, bv, bytes);
    if (ok) ok = memcmp(av, bv, bytes) == 0;
    float sentinel = -123.125f;
    uint32_t bits; memcpy(&bits, &sentinel, sizeof(bits));
    for (unsigned i = 0; ok && i < TEST_GUARD; i++)
        ok = av[i] == bits && av[n - 1 - i] == bits;
    free(av); free(bv);
    return ok;
}

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

int main(void) {
    CHECK(ds4_gpu_init());
    if (!ds4_gpu_timeline_set_phase("setup", 0)) {
        fprintf(stderr, "timeline: timestamp counters unsupported\n");
        ds4_gpu_cleanup();
        return 77;
    }
    CHECK(ds4_gpu_timeline_set_phase("bad phase", 0) == -1);
    CHECK(ds4_gpu_timeline_set_phase("phase_name_too_long", 0) == -1);
    CHECK(ds4_gpu_timeline_set_phase(NULL, 0) == -1);
    const uint32_t widths[] = { TEST_W0, TEST_W0, TEST_W1, TEST_W1 };
    const size_t page = (size_t)getpagesize();
    uint64_t woff[4], ape[2], cursor = page + 128;
    for (int i = 0; i < 4; i++) { woff[i] = cursor; cursor += (uint64_t)widths[i] * TEST_K * sizeof(_Float16); }
    for (int i = 0; i < 2; i++) { ape[i] = cursor; cursor += 4ull * (i ? TEST_W1 : TEST_W0) * sizeof(float); }
    uint64_t model_size = (cursor + page - 1) / page * page;
    void *model = NULL;
    CHECK(posix_memalign(&model, page, model_size) == 0);
    memset(model, 0, model_size);
    for (int j = 0; j < 4; j++) {
        _Float16 *w = (_Float16 *)((char *)model + woff[j]);
        for (size_t i = 0; i < (size_t)widths[j] * TEST_K; i++)
            w[i] = (_Float16)(((int)((i * 13 + i / 23 + j * 37) % 255) - 127) / 1024.0f);
    }
    for (int j = 0; j < 2; j++) for (size_t i = 0; i < 4u * (j ? TEST_W1 : TEST_W0); i++) {
        float v = ((int)((i * 7 + j * 19) % 63) - 31) / 32.0f;
        if (j) ((_Float16 *)((char *)model + ape[j]))[i] = (_Float16)v;
        else ((float *)((char *)model + ape[j]))[i] = v;
    }
    CHECK(ds4_gpu_set_model_map(model, model_size));
    test_tensor x = test_alloc(TEST_K), got[8], ref[8];
    for (int i = 0; i < 8; i++) { size_t n = widths[i % 4] * (i >= 4 ? 8u : 1u); got[i] = test_alloc(n); ref[i] = test_alloc(n); }
    float xv[TEST_K];
    for (int i = 0; i < TEST_K; i++) xv[i] = ((int)((i * 17 + i / 5) % 511) - 255) / 512.0f;
    CHECK(ds4_gpu_tensor_write(x.view, 0, xv, sizeof(xv)));
    for (int j = 0; j < 2; j++) {
        CHECK(ds4_gpu_matmul_f16_pair_tensor(ref[j*2].view, ref[j*2+1].view, model, model_size,
            woff[j*2], woff[j*2+1], TEST_K, widths[j*2], x.view, 1));
        CHECK(ds4_gpu_compressor_store_batch_tensor(ref[j*2].view, ref[j*2+1].view, ref[4+j*2].view,
            ref[5+j*2].view, model, model_size, ape[j], j, widths[j*2]/2, 4, 5, 1));
    }
#define QUAD() ds4_gpu_matmul_f16_quad_compressor_store_tensor( \
        got[0].view, got[1].view, got[2].view, got[3].view, got[4].view, got[5].view, got[6].view, got[7].view, \
        model, model_size, woff[0], woff[1], woff[2], woff[3], ape[0], 0, ape[1], 1, TEST_K, TEST_W0, TEST_W1, x.view, 4, 5)
    CHECK(ds4_gpu_timeline_set_phase("prefill", 11) == 1);
    CHECK(ds4_gpu_begin_commands());
    DS4TimelineBatch *prefill = g_timeline_batch;
    CHECK(QUAD() == 1);
    CHECK(ds4_gpu_timeline_set_phase("invalid_open", 77) == -1);
    CHECK(ds4_gpu_end_commands());

    CHECK(ds4_gpu_timeline_set_phase("decode", 12) == 1);
    CHECK(strcmp(prefill->phase, "prefill") == 0 && prefill->position == 11);
    CHECK(ds4_gpu_begin_commands());
    CHECK(QUAD() == 1);
    DS4TimelineBatch *before_flush = g_timeline_batch;
    CHECK(ds4_gpu_flush_commands());
    CHECK(g_timeline_batch != before_flush);
    /* A pending earlier batch cannot be relabelled through the open restart. */
    CHECK(ds4_gpu_timeline_set_phase("pending", 88) == -1);
    CHECK(strcmp(before_flush->phase, "decode") == 0 && before_flush->position == 12);
    CHECK(QUAD() == 1);
    uint64_t event = 0;
    CHECK(ds4_gpu_signal_selected_readback_ready(&event));
    DS4TimelineBatch *before_public = g_timeline_batch;
    CHECK(ds4_gpu_commit_and_wait_selected_readback(event, "timeline public restart"));
    CHECK(g_timeline_batch != before_public);
    CHECK(QUAD() == 1);
    DS4TimelineBatch *before_private = g_timeline_batch;
    CHECK(ds4_gpu_signal_batch_and_wait_event("timeline private restart"));
    CHECK(g_timeline_batch != before_private);
    CHECK(QUAD() == 1);
    CHECK(ds4_gpu_end_commands());

    CHECK(ds4_gpu_timeline_set_phase("after_owned", UINT32_MAX) == 1);
    DS4TimelineBatch *completed = g_timeline_batch;
    CHECK(completed->count == 1 && completed->recs[0].n_dispatch == 1);
    /* This owned, untraced dispatch must not append to the completed record,
     * even when a driver's completion callback has already printed it. */
    CHECK(ds4_gpu_matmul_f16_pair_tensor(ref[0].view, ref[1].view, model, model_size,
        woff[0], woff[1], TEST_K, TEST_W0, x.view, 1));
    CHECK(completed->count == 1 && completed->recs[0].n_dispatch == 1);
    CHECK(strcmp(completed->recs[0].kernel, "kernel_mul_mv_f16_f32_quad_compressor_store_4") == 0);
    CHECK(strcmp(completed->phase, "decode") == 0 && completed->position == 12);
    CHECK(ds4_gpu_begin_commands());
    CHECK(QUAD() == 1);
    CHECK(ds4_gpu_end_commands());
    for (int i = 0; i < 8; i++) CHECK(test_equal(got[i], ref[i]));
    CHECK(ds4_gpu_synchronize());
    for (int i = 0; i < 8; i++) {
        ds4_gpu_tensor_free(got[i].view); ds4_gpu_tensor_free(got[i].base);
        ds4_gpu_tensor_free(ref[i].view); ds4_gpu_tensor_free(ref[i].base);
    }
    ds4_gpu_tensor_free(x.view); ds4_gpu_tensor_free(x.base);
    ds4_gpu_cleanup();
    free(model);
    puts("PASS Metal timeline: 6 traced quad passes, 3 restart paths, immutable contexts, untraced owned dispatch, 8 guarded tensor comparisons");
    return 0;
}
