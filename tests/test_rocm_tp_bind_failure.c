/* Exercise the actual engine bind's first allocation failure without a model. */
#include "../ds4.c"
static unsigned calls;
ds4_gpu_tensor *__wrap_ds4_gpu_tensor_alloc_coherent(uint64_t bytes) {
    (void)bytes; ++calls; return NULL;
}
int main(void) {
    ds4_engine e = {0}; char err[256] = {0};
    /* The forced first-allocation failure must return before touching this
     * opaque transport token; no connection or GPU initialization is needed. */
    unsigned char token;
    struct ds4_tp *tp = (struct ds4_tp *)(void *)&token;
    e.backend = DS4_BACKEND_CUDA; g_ds4_shape = DS4_SHAPE_FLASH41;
    if (ds4_engine_tp_bind(&e, NULL, err, sizeof(err)) || calls) return 1;
    int rc = ds4_engine_tp_bind(&e, tp, err, sizeof(err));
    int clean = !e.tp.ctx && !e.tp.slab && !e.tp.zero_vec && !e.tp.out_views && !e.tp.in_views &&
        !e.tp.batch_out_views && !e.tp.batch_in_views && !e.tp.active;
    printf("rc=%d coherent_calls=%u clean=%d error=%s\n", rc, calls, clean, err);
    /* The pre-fix control leaves these tables allocated. Release evidence-owned memory. */
    free(e.tp.out_views); free(e.tp.in_views);
    free(e.tp.batch_out_views); free(e.tp.batch_in_views);
    return rc != 0 || calls != 1 || !clean || !strstr(err, "slab allocation failed");
}
