#include "../ds4.c"

static void require(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "HC pre repeat FAIL: %s\n", what);
        exit(1);
    }
}

int main(int argc, char **argv) {
    require(argc == 3, "expected repeat and reference flags");
    const bool repeat = atoi(argv[1]) != 0;
    const bool reference = atoi(argv[2]) != 0;
    require(setenv("DS4_METAL_DISABLE_GLM53_HC_PRODUCER_FUSE", "1", 1) == 0,
            "disable compound producer");
    require(setenv("DS4_METAL_DISABLE_HC_FUSION", reference ? "1" : "0", 1) == 0,
            "select split/collapse variant");
    require(setenv("DS4_GLM_DECODE_REPEAT", repeat ? "hc_pre" : "", 1) == 0,
            "select repeat mode");
    g_ds4_shape = DS4_SHAPE_GLM53;
    const uint32_t dim = DS4_N_HC * DS4_N_EMBD;
    const uint32_t mix = DS4_N_HC * (DS4_N_HC + 2u);
    const ds4_tensor fn = {.type = DS4_TENSOR_BF16, .abs_offset = 0};
    const ds4_tensor scale = {.abs_offset = (uint64_t)dim * mix * sizeof(uint16_t)};
    const ds4_tensor base = {.abs_offset = scale.abs_offset + 3u * sizeof(float)};
    const ds4_tensor norm = {.abs_offset = base.abs_offset + mix * sizeof(float)};
    ds4_model model = {0};
    model.size = (norm.abs_offset + DS4_N_EMBD * sizeof(float) + 16383u) & ~16383ull;
    model.map = mmap(NULL, model.size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    require(model.map != MAP_FAILED, "allocate model fixture");
    uint16_t *weights = (uint16_t *)model.map;
    for (uint32_t i = 0; i < dim * mix; i++) {
        const float value = (float)((int)((i * 1664525u + 1013904223u) >> 24) - 128) / 4096.0f;
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        weights[i] = (uint16_t)(bits >> 16);
    }
    for (uint32_t i = 0; i < 3; i++)
        ((float *)(model.map + scale.abs_offset))[i] = 0.5f + 0.25f * i;
    for (uint32_t i = 0; i < mix; i++)
        ((float *)(model.map + base.abs_offset))[i] = 0.125f * (float)((int)(i % 5) - 2);
    for (uint32_t i = 0; i < DS4_N_EMBD; i++)
        ((float *)(model.map + norm.abs_offset))[i] = 1.0f + 0.001f * (i % 7);
    require(ds4_gpu_init(), "initialize Metal");
    require(ds4_gpu_set_model_map(model.map, model.size), "register model fixture");
    ds4_glm_gpu_graph graph = {0};
    const uint32_t counts[] = {dim, mix, mix, DS4_N_EMBD, DS4_N_EMBD};
    ds4_gpu_tensor *expected[5], *actual[5];
    for (unsigned i = 0; i < 5; i++) {
        expected[i] = ds4_gpu_tensor_alloc(counts[i] * sizeof(float));
        actual[i] = ds4_gpu_tensor_alloc(counts[i] * sizeof(float));
        require(expected[i] && actual[i], "allocate producer outputs");
    }
    graph.hc_flat = actual[0];
    graph.hc_mix = actual[1];
    graph.hc_split = actual[2];
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(dim * sizeof(float));
    float *input = malloc(dim * sizeof(float));
    float *a = malloc(dim * sizeof(float)), *b = malloc(dim * sizeof(float));
    require(residual && input && a && b, "allocate residual and readback");
    for (unsigned draw = 0; draw < 3; draw++) {
        for (uint32_t i = 0; i < dim; i++) {
            input[i] = draw == 0 ? 0.0f :
                (0.01f * (float)((int)(i % 23) - 11) + 0.002f * (i % 5)) *
                (draw == 2 && i % 7 == 0 ? 32.0f : 1.0f);
        }
        require(ds4_gpu_tensor_write(residual, 0, input, dim * sizeof(float)), "write residual");
        require(ds4_gpu_begin_commands(), "begin reference");
        const uint64_t before_reference = ds4_gpu_encoder_count();
        require(ds4_gpu_rms_norm_plain_tensor(expected[0], residual, dim, DS4_RMS_EPS), "reference RMS");
        require(ds4_gpu_glm53_matmul_bf16(expected[1], model.map, model.size,
                    fn.abs_offset, dim, mix, expected[0], 1), "reference projection");
        if (reference) {
            require(ds4_gpu_hc_split_sinkhorn_tensor(expected[2], expected[1],
                        model.map, model.size, scale.abs_offset, base.abs_offset,
                        DS4_N_HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS), "reference split");
            require(ds4_gpu_hc_weighted_sum_tensor(expected[3], residual, expected[2],
                        DS4_N_EMBD, DS4_N_HC), "reference collapse");
        } else {
            require(ds4_gpu_hc_split_weighted_sum_tensor(expected[3], expected[2],
                        expected[1], residual, model.map, model.size,
                        scale.abs_offset, base.abs_offset, DS4_N_EMBD, DS4_N_HC,
                        DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS), "reference split/collapse");
        }
        require(ds4_gpu_rms_norm_weight_tensor(expected[4], expected[3], model.map,
                    model.size, norm.abs_offset, DS4_N_EMBD, DS4_RMS_EPS), "reference weighted RMS");
        const uint64_t single_dispatches = ds4_gpu_encoder_count() - before_reference;
        require(ds4_gpu_end_commands(), "finish reference");
        require(single_dispatches == (reference ? 5u : 4u), "single producer dispatch count");
        for (unsigned i = 0; i < 5; i++)
            require(ds4_gpu_tensor_fill_f32(actual[i], -17.0f, counts[i]), "poison outputs");
        require(ds4_gpu_begin_commands(), "begin graph producer");
        const uint64_t before = ds4_gpu_encoder_count();
        require(glm53_graph_hc_pre(&graph, &model, &fn, &scale, &base, &norm,
                    residual, actual[3], actual[4]), "graph producer");
        const uint64_t dispatches = ds4_gpu_encoder_count() - before;
        require(ds4_gpu_end_commands(), "finish graph producer");
        const uint64_t wanted = single_dispatches * (repeat ? 2u : 1u);
        if (dispatches != wanted) {
            fprintf(stderr, "HC pre dispatches: got %llu, expected %llu\n",
                    (unsigned long long)dispatches, (unsigned long long)wanted);
            return 1;
        }
        for (unsigned i = 0; i < 5; i++) {
            const size_t bytes = counts[i] * sizeof(float);
            require(ds4_gpu_tensor_read(expected[i], 0, a, bytes) &&
                    ds4_gpu_tensor_read(actual[i], 0, b, bytes), "read producer outputs");
            require(memcmp(a, b, bytes) == 0, "producer output bytes match single execution");
        }
        require(ds4_gpu_tensor_read(residual, 0, a, dim * sizeof(float)), "read residual");
        require(memcmp(a, input, dim * sizeof(float)) == 0, "residual remains unchanged");
    }
    for (unsigned i = 0; i < 5; i++) {
        ds4_gpu_tensor_free(expected[i]);
        ds4_gpu_tensor_free(actual[i]);
    }
    ds4_gpu_tensor_free(residual);
    free(input); free(a); free(b);
    ds4_gpu_cleanup();
    munmap((void *)model.map, model.size);
    printf("HC pre repeat PASS: repeat=%u reference=%u, three draws, exact bytes and dispatch count\n",
           repeat, reference);
    return 0;
}
