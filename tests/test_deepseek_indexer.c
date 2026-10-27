/* Check the actual decode selection, not just similarity of generated text.
 * Run on a dedicated GPU host with a resident DeepSeek V4 Flash GGUF. */
#include "../ds4.c"

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s FLASH_GGUF\n", argv[0]);
        return 2;
    }
    ds4_engine_options opt = {
        .model_path = argv[1], .n_threads = 1, .context_size = 8192,
        .prefill_chunk = 2048,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
    };
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0}, corpus = {0};
    char err[256] = "";
    int rc = 1;
    uint32_t *ids = NULL;
    float *scores = NULL;
    unsigned char *seen = NULL;
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK4 &&
            DS4_MODEL_VARIANT == DS4_VARIANT_FLASH && DS4_N_INDEXER_TOP_K == 512);
    REQUIRE(ds4_session_create(&session, engine, 8192) == 0);
    ds4_tokenize_text(engine,
        "A library keeps numbered records for every book. The catalogue lists "
        "authors, titles, years and shelf locations. Readers can search for a "
        "title or reserve an available copy. Returned books are checked and "
        "placed back on their shelves. Accurate records prevent lost loans.\n",
        &corpus);
    REQUIRE(corpus.len > 0);
    int last_indexer = -1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++)
        if (ds4_layer_compress_ratio(il) == 4) last_indexer = (int)il;
    REQUIRE(last_indexer >= 0);
    const uint32_t top_k = DS4_N_INDEXER_TOP_K;
    ids = malloc(top_k * sizeof(*ids));
    scores = malloc(1025 * sizeof(*scores));
    seen = malloc(1025);
    REQUIRE(ids && scores && seen);
    const int ends[] = {2044, 2048, 2052, 3072, 4096, 4100};
    for (unsigned c = 0; c < sizeof(ends) / sizeof(*ends); c++) {
        const int end = ends[c];
        while (prompt.len < end - 1)
            ds4_tokens_push(&prompt, corpus.v[prompt.len % corpus.len]);
        REQUIRE(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
        ds4_gpu_graph *g = &session->graph;
        REQUIRE(g->active_tier == 0 && g->tp_world <= 1);
        ds4_gpu_tensor *selected = metal_graph_comp_selected(g);
        memset(ids, 0xff, top_k * sizeof(*ids));
        REQUIRE(ds4_gpu_tensor_write(selected, 0, ids, top_k * sizeof(*ids)));
        const int token = corpus.v[prompt.len % corpus.len];
        REQUIRE(ds4_session_eval(session, token, err, sizeof(err)) == 0);
        ds4_tokens_push(&prompt, token);
        const uint32_t n_comp = g->layer_n_index_comp[last_indexer];
        REQUIRE(n_comp == (uint32_t)end / 4);
        REQUIRE(ds4_gpu_tensor_read(selected, 0, ids, top_k * sizeof(*ids)));
        if (n_comp <= top_k) {
            for (uint32_t i = 0; i < top_k; i++) REQUIRE(ids[i] == UINT32_MAX);
        } else {
            REQUIRE(ds4_gpu_tensor_read(metal_graph_indexer_scores(g), 0,
                                        scores, n_comp * sizeof(*scores)));
            memset(seen, 0, n_comp);
            float min_selected = INFINITY, max_excluded = -INFINITY;
            for (uint32_t i = 0; i < n_comp; i++) REQUIRE(isfinite(scores[i]));
            for (uint32_t i = 0; i < top_k; i++) {
                REQUIRE(ids[i] < n_comp && !seen[ids[i]]);
                seen[ids[i]] = 1;
                min_selected = fminf(min_selected, scores[ids[i]]);
            }
            for (uint32_t i = 0; i < n_comp; i++)
                if (!seen[i]) max_excluded = fmaxf(max_excluded, scores[i]);
            REQUIRE(min_selected >= max_excluded);
        }
        fprintf(stderr, "decode indexer end=%d compressed=%u selected=%u: PASS\n",
                end, n_comp, n_comp > top_k ? top_k : n_comp);
    }
    rc = 0;
done:
    if (rc) fprintf(stderr, "decode indexer: FAIL %s\n", err);
    free(seen);
    free(scores);
    free(ids);
    ds4_tokens_free(&corpus);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return rc;
}
