/* Compare complete scalar and batched decoder KV publication, without a GGUF. */
#include "../ds4.c"

#define CHECK(v) do { if (!(v)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #v); exit(1); } } while (0)
enum { K = 5120, D = 512, I = 128, CAP = 2048, CTX = 19000 };
static uint32_t rng = 91341;
static uint32_t bits(void) { rng = rng * 1664525u + 1013904223u; return rng; }
static ds4_gpu_tensor *allocf(uint64_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * 4u);
    CHECK(t && ds4_gpu_tensor_contents(t)); return t;
}
static ds4_gpu_tensor *input, *views[CAP];
static ds41_gpu_graph scalar, batch;
static ds4_model model;
static ds4_layer_weights weights;
static ds4_tensor tensor[4];
static uint64_t input_hash(void) {
    const uint32_t *p = ds4_gpu_tensor_contents(input);
    uint64_t h = UINT64_C(1469598103934665603);
    for (uint64_t i = 0; i < (uint64_t)CAP * K; i++)
        h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static double run(bool bulk, uint32_t start, uint32_t count) {
    const double begin = now_sec();
    CHECK(ds4_gpu_begin_commands());
    if (bulk) CHECK(ds41_attention_publish_batch(&batch, &batch.batch, &model, &weights, 20, start, count));
    else for (uint32_t t = 0; t < count; t++) {
        scalar.pos = start + t;
        scalar.norm = views[t];
        CHECK(ds41_attention_publish(&scalar, &model, &weights, 20));
    }
    CHECK(ds4_gpu_end_commands());
    return (now_sec() - begin) * 1000;
}
static void guard_and_check(uint32_t start, uint32_t count) {
    ds4_gpu_tensor *a[2] = {scalar.compressed[3], scalar.index_cache[3]};
    ds4_gpu_tensor *b[2] = {batch.compressed[3], batch.index_cache[3]};
    for (unsigned j = 0; j < 2; j++) {
        memset(ds4_gpu_tensor_contents(a[j]), 0x3f, ds4_gpu_tensor_bytes(a[j]));
        memset(ds4_gpu_tensor_contents(b[j]), 0x3f, ds4_gpu_tensor_bytes(b[j]));
    }
    const uint64_t hash = input_hash();
    run(false, start, count); run(true, start, count);
    CHECK(input_hash() == hash);
    for (unsigned j = 0; j < 2; j++) {
        uint32_t width = j ? I : D;
        float *x = ds4_gpu_tensor_contents(a[j]), *y = ds4_gpu_tensor_contents(b[j]);
        uint64_t n = (uint64_t)CTX * width, changed = 0;
        for (uint64_t i = 0; i < n; i++) {
            CHECK(isfinite(x[i]) && isfinite(y[i]));
            changed += x[i] != y[i];
            if (i < (uint64_t)start * width || i >= (uint64_t)(start + count) * width)
                CHECK(((uint32_t *)x)[i] == 0x3f3f3f3f && ((uint32_t *)y)[i] == 0x3f3f3f3f);
        }
        fprintf(stderr, "publication start=%u rows=%u cache=%u changed=%llu\n", start, count, j, (unsigned long long)changed);
        CHECK(!memcmp(x, y, n * 4u));
    }
}
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
int main(int argc, char **argv) {
    const bool bench = argc == 2 && !strcmp(argv[1], "--bench");
    if (argc > 1 && !bench) return 2;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    CHECK(ds4_gpu_init());
    const uint32_t widths[] = {K, D, D, I}, outputs[] = {D, I, 1, 1};
    uint64_t end = 16384;
    for (unsigned j = 0; j < 4; j++) {
        tensor[j].abs_offset = end;
        tensor[j].type = j < 2 ? DS4_TENSOR_F16 : DS4_TENSOR_F32;
        tensor[j].dim[0] = widths[j]; tensor[j].dim[1] = outputs[j];
        end += (uint64_t)widths[j] * outputs[j] * (j < 2 ? 2u : 4u);
        end = (end + 16383u) & ~UINT64_C(16383);
    }
    model.map = mmap(NULL, end, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    model.size = end; CHECK(model.map != MAP_FAILED);
    for (unsigned j = 0; j < 4; j++) {
        void *p = (char *)model.map + tensor[j].abs_offset;
        for (uint64_t i = 0; i < (uint64_t)widths[j] * outputs[j]; i++) {
            if (j < 2) ((uint16_t *)p)[i] = 0x2800u | (bits() & 0x83ffu);
            else ((float *)p)[i] = 0.5f + (bits() >> 16u) / 65536.0f;
        }
    }
    weights.attn_compressor_kv = &tensor[0]; weights.indexer_attn_k = &tensor[1];
    weights.attn_compressor_norm = &tensor[2]; weights.indexer_k_norm = &tensor[3];
    CHECK(ds4_gpu_set_model_map(model.map, model.size));
    input = allocf((uint64_t)CAP * K);
    float *x = ds4_gpu_tensor_contents(input);
    for (uint64_t i = 0; i < (uint64_t)CAP * K; i++) x[i] = ((int)(bits() >> 16) - 32768) / 16384.0f;
    for (unsigned i = 0; i < CAP; i++) CHECK((views[i] = ds4_gpu_tensor_view(input, (uint64_t)i * K * 4, K * 4)));
    scalar.pool_kv = allocf(D); scalar.latent = allocf(D); scalar.index_k = allocf(I);
    batch.batch.norm = input; batch.batch.latent = allocf((uint64_t)CAP * D); batch.batch.index_k = allocf((uint64_t)CAP * I);
    scalar.compressed[3] = allocf((uint64_t)CTX * D); batch.compressed[3] = allocf((uint64_t)CTX * D);
    scalar.index_cache[3] = allocf((uint64_t)CTX * I); batch.index_cache[3] = allocf((uint64_t)CTX * I);
    const uint32_t starts[] = {0, 0, 127, 128, 0, 0, 1, 129, 8191, 16383};
    const uint32_t counts[] = {1, 2, 31, 129, 437, 511, 512, 513, 1024, 2048};
    for (unsigned i = 0; i < sizeof(counts) / sizeof(*counts); i++) guard_and_check(starts[i], counts[i]);
    ds4_gpu_set_quality(true);
    guard_and_check(511, 437);
    ds4_gpu_set_quality(false);
    if (bench) {
        double t[2][9];
        for (unsigned i = 0; i < 9; i++) for (unsigned j = 0; j < 2; j++) {
            unsigned arm = j ^ (i & 1u); t[arm][i] = run(arm, 0, 437);
            fprintf(stderr, "sample=%u bulk=%u ms=%.3f\n", i, arm, t[arm][i]);
        }
        qsort(t[0], 9, sizeof(double), cmp); qsort(t[1], 9, sizeof(double), cmp);
        printf("layer20 publication fixture: byte-identical caches/guards PASS; median scalar=%.3f ms bulk=%.3f ms; time saved %.2f%%\n", t[0][4], t[1][4], 100 * (1 - t[1][4] / t[0][4]));
    }
    for (unsigned i = 0; i < CAP; i++) ds4_gpu_tensor_free(views[i]);
    ds4_gpu_tensor_free(input);
    ds4_gpu_tensor_free(scalar.pool_kv);
    ds4_gpu_tensor_free(scalar.latent);
    ds4_gpu_tensor_free(scalar.index_k);
    ds4_gpu_tensor_free(batch.batch.latent);
    ds4_gpu_tensor_free(batch.batch.index_k);
    ds4_gpu_tensor_free(scalar.compressed[3]);
    ds4_gpu_tensor_free(batch.compressed[3]);
    ds4_gpu_tensor_free(scalar.index_cache[3]);
    ds4_gpu_tensor_free(batch.index_cache[3]);
    ds4_gpu_cleanup();
    munmap((void *)model.map, model.size);
    puts("V4.1 decoder publication: byte-identical caches and intact guards PASS");
    return 0;
}
