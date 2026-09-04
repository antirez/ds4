/* Unit tests for the GLP (GGUF Layer Projection) steering-vector reader.
 *
 * Pure C99: no CUDA, no Metal, no model. Every test builds a real GGUF v3 file
 * on disk and reads it back, so the container layout is exercised rather than
 * mocked.
 *
 * The refusal paths are the point of this file. A GLP reader that accepts a
 * bad vector does not crash and does not print anything wrong -- it produces
 * plausible, degraded output. Each of the cases below was chosen because it is
 * a failure that cannot be caught downstream:
 *
 *   wrong operation   an additive vector applied projectively (or the reverse)
 *   wrong hook        the same direction 9x weaker
 *   wrong layer map   adjacent-layer cosine 0.55-0.98, so a shift still works
 *   wrong base model  undefined, and shapes can still match
 */

#include "../ds4_glp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_failed = 0;
static int g_total  = 0;

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__);            \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

#define RUN(fn) do {                                                           \
    fprintf(stderr, "RUN: %s\n", #fn);                                         \
    int _before = g_failed;                                                    \
    (fn)();                                                                    \
    fprintf(stderr, "  %s\n", (_before == g_failed) ? "ok" : "FAIL");          \
} while (0)

/* ------------------------------------------------------------------ */
/* Minimal GGUF v3 writer                                             */
/* ------------------------------------------------------------------ */

#define GGUF_TYPE_UINT32  4
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TYPE_STRING  8
#define GGML_TYPE_F32     0
#define GGML_TYPE_F16     1
#define GGUF_ALIGNMENT    32

#define MAX_KV   32
#define MAX_DIRS 64

typedef struct {
    char     key[64];
    uint32_t type;
    char     s[192];
    uint32_t u32;
    float    f32;
} kv_entry;

typedef struct {
    uint32_t layer;      /* the N in direction.<N> */
    uint32_t n;          /* element count */
    uint32_t ggml_type;  /* GGML_TYPE_F32 unless a test wants otherwise */
    uint32_t ndim;       /* 1 unless a test wants otherwise */
    float    scale;      /* norm to write; 1.0 for a unit direction */
    float    fill;       /* value pattern seed; 0 writes a zero direction */
} dir_entry;

typedef struct {
    kv_entry  kv[MAX_KV];
    int       n_kv;
    dir_entry dirs[MAX_DIRS];
    int       n_dirs;
} gguf_spec;

static void kv_str(gguf_spec *g, const char *key, const char *val) {
    kv_entry *e = &g->kv[g->n_kv++];
    memset(e, 0, sizeof(*e));
    snprintf(e->key, sizeof(e->key), "%s", key);
    snprintf(e->s, sizeof(e->s), "%s", val);
    e->type = GGUF_TYPE_STRING;
}

static void kv_u32(gguf_spec *g, const char *key, uint32_t val) {
    kv_entry *e = &g->kv[g->n_kv++];
    memset(e, 0, sizeof(*e));
    snprintf(e->key, sizeof(e->key), "%s", key);
    e->u32  = val;
    e->type = GGUF_TYPE_UINT32;
}

static void kv_f32(gguf_spec *g, const char *key, float val) {
    kv_entry *e = &g->kv[g->n_kv++];
    memset(e, 0, sizeof(*e));
    snprintf(e->key, sizeof(e->key), "%s", key);
    e->f32  = val;
    e->type = GGUF_TYPE_FLOAT32;
}

static void add_dir(gguf_spec *g, uint32_t layer, uint32_t n) {
    dir_entry *d = &g->dirs[g->n_dirs++];
    memset(d, 0, sizeof(*d));
    d->layer     = layer;
    d->n         = n;
    d->ggml_type = GGML_TYPE_F32;
    d->ndim      = 1;
    d->scale     = 1.0f;
    d->fill      = 1.0f;
}

static void w_u32(FILE *fp, uint32_t v) { fwrite(&v, sizeof(v), 1, fp); }
static void w_u64(FILE *fp, uint64_t v) { fwrite(&v, sizeof(v), 1, fp); }
static void w_f32(FILE *fp, float v)    { fwrite(&v, sizeof(v), 1, fp); }

static void w_str(FILE *fp, const char *s) {
    const uint64_t len = strlen(s);
    w_u64(fp, len);
    fwrite(s, 1, (size_t)len, fp);
}

static void w_pad(FILE *fp) {
    const long pos = ftell(fp);
    long pad = (GGUF_ALIGNMENT - (pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT;
    while (pad-- > 0) fputc(0, fp);
}

/* Build the direction values: a deterministic pattern normalised to
 * d->scale, so a test can ask for a unit direction or a deliberately
 * off-unit one. fill == 0 writes an all-zero direction. */
static void fill_direction(const dir_entry *d, float *buf) {
    if (d->fill == 0.0f) {
        memset(buf, 0, (size_t)d->n * sizeof(buf[0]));
        return;
    }
    double sumsq = 0.0;
    for (uint32_t i = 0; i < d->n; i++) {
        buf[i] = d->fill * (float)((i % 7) + 1) * ((i % 2) ? -1.0f : 1.0f);
        sumsq += (double)buf[i] * (double)buf[i];
    }
    const float norm = (float)sqrt(sumsq);
    const float k = d->scale / norm;
    for (uint32_t i = 0; i < d->n; i++) buf[i] *= k;
}

/* Write the spec to path. Tensor data offsets are relative to the aligned data
 * start, which is what GGUF specifies and what the reader must resolve. */
static int write_gguf(const char *path, const gguf_spec *g) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;

    w_u32(fp, 0x46554747u); /* "GGUF" */
    w_u32(fp, 3);
    w_u64(fp, (uint64_t)g->n_dirs);
    w_u64(fp, (uint64_t)g->n_kv);

    for (int i = 0; i < g->n_kv; i++) {
        const kv_entry *e = &g->kv[i];
        w_str(fp, e->key);
        w_u32(fp, e->type);
        switch (e->type) {
        case GGUF_TYPE_STRING:  w_str(fp, e->s);  break;
        case GGUF_TYPE_UINT32:  w_u32(fp, e->u32); break;
        case GGUF_TYPE_FLOAT32: w_f32(fp, e->f32); break;
        default: break;
        }
    }

    /* Every direction is padded to the alignment so each tensor starts on an
     * aligned offset, matching what real writers emit. */
    uint64_t offset = 0;
    for (int i = 0; i < g->n_dirs; i++) {
        const dir_entry *d = &g->dirs[i];
        char name[64];
        snprintf(name, sizeof(name), "direction.%u", d->layer);
        w_str(fp, name);
        w_u32(fp, d->ndim);
        for (uint32_t k = 0; k < d->ndim; k++) w_u64(fp, k == 0 ? d->n : 1);
        w_u32(fp, d->ggml_type);
        w_u64(fp, offset);

        const uint64_t elem = (d->ggml_type == GGML_TYPE_F16) ? 2 : 4;
        uint64_t bytes = (uint64_t)d->n * elem;
        for (uint32_t k = 1; k < d->ndim; k++) bytes *= 1;
        bytes = (bytes + GGUF_ALIGNMENT - 1) / GGUF_ALIGNMENT * GGUF_ALIGNMENT;
        offset += bytes;
    }

    w_pad(fp);
    for (int i = 0; i < g->n_dirs; i++) {
        const dir_entry *d = &g->dirs[i];
        if (d->ggml_type == GGML_TYPE_F32) {
            float *buf = malloc((size_t)d->n * sizeof(float));
            fill_direction(d, buf);
            fwrite(buf, sizeof(float), d->n, fp);
            free(buf);
        } else {
            for (uint32_t k = 0; k < d->n; k++) fputc(0, fp), fputc(0, fp);
        }
        w_pad(fp);
    }
    fclose(fp);
    return 1;
}

/* A conforming vector: project mode, ds4's FFN hook, layers 2..5 of an 8-layer
 * 64-wide model. */
static void spec_valid(gguf_spec *g, uint32_t n_embd) {
    memset(g, 0, sizeof(*g));
    kv_str(g, "general.architecture", "controlvector");
    kv_str(g, "glp.mode", "project");
    kv_u32(g, "glp.spec_version", 1);
    kv_f32(g, "glp.alpha_default", 3.0f);
    kv_u32(g, "glp.rank", 1);
    kv_str(g, "glp.hook_point", "ffn_out_pre_residual");
    kv_str(g, "glp.layer_ids_zero_based", "2,3,4,5");
    kv_str(g, "glp.method", "paired_difference_of_means");
    kv_str(g, "general.base_model.0.name", "DeepSeek-V4-Flash-0731");
    kv_str(g, "general.base_model.0.organization", "deepseek-ai");
    for (uint32_t l = 2; l <= 5; l++) add_dir(g, l, n_embd);
}

static const char *tmp_path(const char *leaf) {
    static char buf[512];
    const char *dir = getenv("TMPDIR");
    if (!dir || !dir[0]) dir = "/tmp";
    size_t n = strlen(dir);
    snprintf(buf, sizeof(buf), "%s%sds4-glp-%d-%s.gguf",
             dir, (n && dir[n - 1] == '/') ? "" : "/", (int)getpid(), leaf);
    return buf;
}

/* ------------------------------------------------------------------ */
/* Tests                                                              */
/* ------------------------------------------------------------------ */

#define N_EMBD   64
#define N_LAYERS 8

static void test_probe(void) {
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    const char *path = tmp_path("probe");
    CHECK(write_gguf(path, &g), "fixture written");
    CHECK(ds4_glp_is_gguf(path) == 1, "GGUF magic detected");

    /* A raw .f32 blob must route to the legacy path, not to the GLP reader:
     * that is what keeps existing --dir-steering-file usage working. */
    const char *raw = tmp_path("raw");
    FILE *fp = fopen(raw, "wb");
    float zero[4] = {0, 0, 0, 0};
    fwrite(zero, sizeof(float), 4, fp);
    fclose(fp);
    CHECK(ds4_glp_is_gguf(raw) == 0, "raw f32 is not GGUF");
    CHECK(ds4_glp_is_gguf("/nonexistent/ds4-glp") == -1, "missing file reports -1");

    unlink(path);
    unlink(raw);
}

static void test_load_valid(void) {
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    const char *path = tmp_path("valid");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    ds4_glp_info info;
    char err[512];
    const int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                                dirs, N_LAYERS, N_EMBD, &info, err, sizeof(err));
    CHECK(rc == 0, "conforming vector loads");
    if (rc != 0) fprintf(stderr, "    err: %s\n", err);

    CHECK(info.n_dirs == 4, "four directions");
    CHECK(info.n_embd == N_EMBD, "n_embd resolved");
    CHECK(info.layer_min == 2 && info.layer_max == 5, "layer range resolved");
    CHECK(info.has_alpha_default && fabsf(info.alpha_default - 3.0f) < 1e-6f,
          "alpha_default read");
    CHECK(info.hook == DS4_GLP_HOOK_FFN_OUT, "hook resolved");
    CHECK(strcmp(info.base_model, "DeepSeek-V4-Flash-0731") == 0, "base model read");

    /* THE layer-map assertion. direction.N lands at row N with no offset.
     * This is the check that would have caught our exporter off-by-one, and
     * the reason it is here rather than in a smoke test is that a one-layer
     * shift produces coherent output: adjacent layers' refusal directions have
     * cosine similarity 0.555-0.979, so a shifted stack still ablates. */
    for (uint32_t l = 0; l < N_LAYERS; l++) {
        double sumsq = 0.0;
        for (uint32_t i = 0; i < N_EMBD; i++) {
            const float v = dirs[(size_t)l * N_EMBD + i];
            sumsq += (double)v * (double)v;
        }
        const int covered = (l >= 2 && l <= 5);
        if (covered) {
            CHECK(fabs(sumsq - 1.0) < 1e-5, "covered layer holds a unit direction");
        } else {
            CHECK(sumsq == 0.0, "uncovered layer is zero (a no-op projection)");
        }
    }
    unlink(path);
}

static void test_refuse_missing_mode(void) {
    /* A plain llama.cpp control vector: right tensors, no glp.mode. Absent
     * means additive, and ds4 has no additive path -- but the message has to
     * say that rather than "malformed", because the file is not malformed. */
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    /* Drop glp.mode by rewriting the KV list without it. */
    gguf_spec h;
    memset(&h, 0, sizeof(h));
    for (int i = 0; i < g.n_kv; i++) {
        if (strcmp(g.kv[i].key, "glp.mode") == 0) continue;
        h.kv[h.n_kv++] = g.kv[i];
    }
    memcpy(h.dirs, g.dirs, sizeof(g.dirs));
    h.n_dirs = g.n_dirs;

    const char *path = tmp_path("nomode");
    CHECK(write_gguf(path, &h), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    char err[512];
    const int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                                dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "no glp.mode is refused");
    CHECK(strstr(err, "glp.mode") != NULL, "error names glp.mode");
    unlink(path);
}

static void test_refuse_add_mode(void) {
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    for (int i = 0; i < g.n_kv; i++) {
        if (strcmp(g.kv[i].key, "glp.mode") == 0) {
            snprintf(g.kv[i].s, sizeof(g.kv[i].s), "add");
        }
    }
    const char *path = tmp_path("addmode");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    char err[512];
    const int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                                dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "glp.mode=add is refused, never coerced to project");
    unlink(path);
}

static void test_refuse_hook_mismatch(void) {
    /* Every GLP vector we have published so far declares
     * residual_stream_post_layer, because that is where llama.cpp's
     * build_cvec() and the vLLM overlay apply. ds4 applies at the block
     * writers. Loading one into the other measured 9x weaker on the
     * attention writer, so the default is a refusal. */
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    for (int i = 0; i < g.n_kv; i++) {
        if (strcmp(g.kv[i].key, "glp.hook_point") == 0) {
            snprintf(g.kv[i].s, sizeof(g.kv[i].s), "residual_stream_post_layer");
        }
    }
    const char *path = tmp_path("hook");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    char err[512];
    int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                          dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "hook mismatch is refused by default");
    CHECK(strstr(err, "hook_point") != NULL, "error names hook_point");
    CHECK(strstr(err, "--dir-steering-allow-hook-mismatch") != NULL,
          "error names the override flag");

    /* ...and loads with the explicit override, which is the whole point of
     * having one: the operation is right, only the calibration moves. */
    ds4_glp_info info;
    rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 1,
                      dirs, N_LAYERS, N_EMBD, &info, err, sizeof(err));
    CHECK(rc == 0, "hook mismatch loads under the override");
    CHECK(info.hook == DS4_GLP_HOOK_RESID_POST_LAYER, "declared hook preserved");
    unlink(path);
}

static void test_refuse_unknown_hook(void) {
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    for (int i = 0; i < g.n_kv; i++) {
        if (strcmp(g.kv[i].key, "glp.hook_point") == 0) {
            snprintf(g.kv[i].s, sizeof(g.kv[i].s), "attn.wo_b_output");
        }
    }
    const char *path = tmp_path("unkhook");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    char err[512];
    const int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                                dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "a hook name this build does not implement is refused");
    unlink(path);
}

static void test_refuse_layer_id_mismatch(void) {
    /* The exporter bug that actually happened: tensor names one layer ahead of
     * the informational layer list. The names are what execute. */
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    for (int i = 0; i < g.n_kv; i++) {
        if (strcmp(g.kv[i].key, "glp.layer_ids_zero_based") == 0) {
            snprintf(g.kv[i].s, sizeof(g.kv[i].s), "1,2,3,4");
        }
    }
    const char *path = tmp_path("layerids");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    char err[512];
    const int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                                dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "declared layer ids disagreeing with tensor names is refused");
    CHECK(strstr(err, "layer_ids_zero_based") != NULL, "error names the field");
    unlink(path);
}

static void test_refuse_direction_zero(void) {
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    g.n_dirs = 0;
    add_dir(&g, 0, N_EMBD);
    for (int i = 0; i < g.n_kv; i++) {
        if (strcmp(g.kv[i].key, "glp.layer_ids_zero_based") == 0) g.kv[i].s[0] = '\0';
    }
    const char *path = tmp_path("dir0");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    char err[512];
    const int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                                dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "direction.0 is refused");
    CHECK(strstr(err, "direction.0") != NULL, "error names direction.0");
    unlink(path);
}

static void test_refuse_wrong_dtype_and_rank(void) {
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    g.dirs[0].ggml_type = GGML_TYPE_F16;
    const char *path = tmp_path("f16");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    char err[512];
    int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                          dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "a non-F32 direction is refused");
    unlink(path);

    spec_valid(&g, N_EMBD);
    for (int i = 0; i < g.n_kv; i++) {
        if (strcmp(g.kv[i].key, "glp.rank") == 0) g.kv[i].u32 = 4;
    }
    path = tmp_path("rank4");
    CHECK(write_gguf(path, &g), "fixture written");
    rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                      dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "rank>1 is refused rather than silently truncated");
    unlink(path);
}

static void test_refuse_shape_mismatch(void) {
    /* Wrong n_embd: a direction from a different checkpoint. */
    gguf_spec g;
    spec_valid(&g, 32);
    const char *path = tmp_path("width");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    char err[512];
    int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                          dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "n_embd mismatch is refused");
    unlink(path);

    /* A layer beyond what this model can steer. */
    memset(&g, 0, sizeof(g));
    kv_str(&g, "glp.mode", "project");
    kv_str(&g, "glp.hook_point", "ffn_out_pre_residual");
    add_dir(&g, 9, N_EMBD);
    path = tmp_path("layerhigh");
    CHECK(write_gguf(path, &g), "fixture written");
    rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                      dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "a direction past the model's layer count is refused");
    unlink(path);
}

static void test_renormalise(void) {
    /* Projection is quadratic in the direction's norm, so an off-unit
     * direction silently scales the removal by ||v||^2. The projector is
     * invariant to ||v||, so rescaling is the correct fix rather than a
     * refusal -- but it has to be reported. */
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    for (int i = 0; i < g.n_dirs; i++) g.dirs[i].scale = 2.5f;
    const char *path = tmp_path("norm");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    ds4_glp_info info;
    char err[512];
    const int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                                dirs, N_LAYERS, N_EMBD, &info, err, sizeof(err));
    CHECK(rc == 0, "off-unit directions load");
    CHECK(info.n_renormalized == 4, "all four are reported as rescaled");
    double sumsq = 0.0;
    for (uint32_t i = 0; i < N_EMBD; i++) {
        const float v = dirs[2 * N_EMBD + i];
        sumsq += (double)v * (double)v;
    }
    CHECK(fabs(sumsq - 1.0) < 1e-5, "stored direction is unit after rescaling");
    unlink(path);

    /* A zero direction cannot be normalised and steers nothing: malformed, not
     * "partial coverage" -- partial coverage is expressed by omitting the
     * tensor. */
    spec_valid(&g, N_EMBD);
    g.dirs[0].fill = 0.0f;
    path = tmp_path("zerodir");
    CHECK(write_gguf(path, &g), "fixture written");
    const int rc2 = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                                 dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc2 != 0, "an all-zero direction is refused");
    unlink(path);
}

static void test_refuse_legacy_namespace(void) {
    /* Pre-2026-08-22 internal files used dspark.*, renamed with no alias.
     * Such a file has the right tensors and no glp.mode, so without an
     * explicit check it gets diagnosed as "an additive control vector" and
     * refused for the wrong reason. */
    gguf_spec g;
    memset(&g, 0, sizeof(g));
    kv_str(&g, "dspark.mode", "project");
    kv_str(&g, "dspark.hook_point", "residual_stream_post_layer");
    for (uint32_t l = 2; l <= 5; l++) add_dir(&g, l, N_EMBD);
    const char *path = tmp_path("legacy");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    char err[512];
    const int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                                dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "a dspark.*-only file is refused");
    CHECK(strstr(err, "dspark") != NULL, "error names the old namespace");
    CHECK(strstr(err, "re-export") != NULL, "error says what to do about it");
    unlink(path);
}

static void test_refuse_no_directions(void) {
    gguf_spec g;
    memset(&g, 0, sizeof(g));
    kv_str(&g, "glp.mode", "project");
    kv_str(&g, "glp.hook_point", "ffn_out_pre_residual");
    const char *path = tmp_path("nodirs");
    CHECK(write_gguf(path, &g), "fixture written");

    float dirs[N_LAYERS * N_EMBD];
    char err[512];
    const int rc = ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                                dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err));
    CHECK(rc != 0, "a GGUF with no direction.<N> tensors is refused");
    unlink(path);
}

static void test_read_info_needs_no_model(void) {
    /* The inspect path: what is actually in this file, without the several-GB
     * checkpoint it belongs to. Our exporter off-by-one surfaced exactly here,
     * as a dump reporting layers 11-39 for a file derived from 10-38. */
    gguf_spec g;
    spec_valid(&g, N_EMBD);
    const char *path = tmp_path("info");
    CHECK(write_gguf(path, &g), "fixture written");

    ds4_glp_info info;
    char err[512];
    const int rc = ds4_glp_read_info(path, &info, err, sizeof(err));
    CHECK(rc == 0, "read_info succeeds with no model shape");
    if (rc != 0) fprintf(stderr, "    err: %s\n", err);
    CHECK(info.n_dirs == 4 && info.layer_min == 2 && info.layer_max == 5,
          "read_info resolves coverage");
    CHECK(info.norm_min > 0.99f && info.norm_max < 1.01f, "read_info reports norms");

    /* read_info deliberately does not check the hook: it must be able to
     * describe a file this build would refuse to apply. That is what makes it
     * the first tool to reach for. */
    for (int i = 0; i < g.n_kv; i++) {
        if (strcmp(g.kv[i].key, "glp.hook_point") == 0) {
            snprintf(g.kv[i].s, sizeof(g.kv[i].s), "residual_stream_post_layer");
        }
    }
    CHECK(write_gguf(path, &g), "fixture rewritten");
    CHECK(ds4_glp_read_info(path, &info, err, sizeof(err)) == 0,
          "read_info describes a file this build cannot apply");
    CHECK(info.hook == DS4_GLP_HOOK_RESID_POST_LAYER, "and reports its hook");
    unlink(path);
}

static void test_malformed_input(void) {
    /* Not a crash test for its own sake: --dir-steering-file is a path a user
     * types, so truncation and "you passed the model" are ordinary inputs. */
    float dirs[N_LAYERS * N_EMBD];
    char err[512];

    CHECK(ds4_glp_load("/nonexistent/ds4-glp.gguf", DS4_GLP_HOOK_FFN_OUT, 0,
                       dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err)) != 0,
          "missing file is refused");

    const char *path = tmp_path("trunc");
    FILE *fp = fopen(path, "wb");
    w_u32(fp, 0x46554747u);
    w_u32(fp, 3);
    w_u64(fp, 4);       /* claims 4 tensors */
    w_u64(fp, 1000000); /* ...and a million KV entries in a 24-byte file */
    fclose(fp);
    CHECK(ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                       dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err)) != 0,
          "header counts exceeding the file are refused before allocating");
    unlink(path);

    /* GGUF v2 is out of spec for GLP. */
    path = tmp_path("v2");
    fp = fopen(path, "wb");
    w_u32(fp, 0x46554747u);
    w_u32(fp, 2);
    w_u64(fp, 0);
    w_u64(fp, 0);
    fclose(fp);
    CHECK(ds4_glp_load(path, DS4_GLP_HOOK_FFN_OUT, 0,
                       dirs, N_LAYERS, N_EMBD, NULL, err, sizeof(err)) != 0,
          "GGUF v2 is refused");
    unlink(path);
}

static void test_hook_names_round_trip(void) {
    CHECK(strcmp(ds4_glp_hook_name(DS4_GLP_HOOK_RESID_POST_LAYER),
                 "residual_stream_post_layer") == 0, "resid hook name");
    CHECK(strcmp(ds4_glp_hook_name(DS4_GLP_HOOK_FFN_OUT),
                 "ffn_out_pre_residual") == 0, "ffn hook name");
    CHECK(strcmp(ds4_glp_hook_name(DS4_GLP_HOOK_ATTN_OUT),
                 "attn_out_pre_residual") == 0, "attn hook name");
}

int main(void) {
    RUN(test_probe);
    RUN(test_load_valid);
    RUN(test_refuse_missing_mode);
    RUN(test_refuse_add_mode);
    RUN(test_refuse_hook_mismatch);
    RUN(test_refuse_unknown_hook);
    RUN(test_refuse_layer_id_mismatch);
    RUN(test_refuse_direction_zero);
    RUN(test_refuse_wrong_dtype_and_rank);
    RUN(test_refuse_shape_mismatch);
    RUN(test_renormalise);
    RUN(test_refuse_legacy_namespace);
    RUN(test_refuse_no_directions);
    RUN(test_read_info_needs_no_model);
    RUN(test_malformed_input);
    RUN(test_hook_names_round_trip);

    fprintf(stderr, "\n%d checks, %d failed\n", g_total, g_failed);
    return g_failed == 0 ? 0 : 1;
}
