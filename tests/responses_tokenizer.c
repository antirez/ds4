/* Load the real tokenizer without binding weights or running inference. */
#include "../ds4.c"

ds4_engine *responses_bench_engine_open(const char *path) {
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->backend = DS4_BACKEND_CPU;
    model_open(&e->model, path, false, false);
    config_validate_model(&e->model);
    vocab_load(&e->vocab, &e->model);
    return e;
}

void responses_bench_engine_close(ds4_engine *e) {
    if (!e) return;
    vocab_free(&e->vocab);
    model_close(&e->model);
    free(e);
}
