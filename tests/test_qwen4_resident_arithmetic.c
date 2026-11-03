/* Resident arithmetic regressions use main 0aaea5a's frozen dispatch;
 * SSD retains its legacy half/unsplit prefill. Synthetic weights have real
 * Qwen dimensions without loading a model. Fixtures and split-K scratch
 * together stay below 100 MiB; cases release their activations sequentially. */
#include "../ds4.c"

#if defined(DS4_HAS_QWEN4_METAL) && defined(__APPLE__)

/* Copied verbatim from 0aaea5a:ds4.c, except the function name. Backend
 * kernels are shared; this oracle freezes dispatch, not historical shaders. */
static bool resident_base_projection(ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w,
                            const ds4_gpu_tensor *x, uint32_t n_tok, uint64_t rows) {
    const uint64_t in_dim = w->dim[0], full_dim = w->ndim >= 2 ? w->dim[1] : 1u;
    const uint64_t out_dim = rows && rows < full_dim ? rows : full_dim;
    int rc = 0;
#if !defined(__APPLE__)
    /* Qwen's recurrent graph keeps activations in FP32. The generic CUDA
     * projections can round them to half or quantize them for other models. */
    if (in_dim <= UINT32_MAX && out_dim <= UINT32_MAX)
        rc = ds4_gpu_qwen4_dense_mm_tensor(out, x, m->map, m->size, w->abs_offset,
                                           w->type, n_tok, (uint32_t)in_dim, (uint32_t)out_dim);
#else

    /* Small F16 batches use float operands. Two/three-row verification and
     * large prefills retain their existing kernels. The legacy switch is an
     * arithmetic/performance control, not a user tuning option. */
    const bool legacy = getenv("DS4_QWEN4_DENSE_MM_LEGACY") != NULL;
    const bool f16_batch = !legacy && n_tok > 3u && w->type == DS4_TENSOR_F16 &&
        n_tok <= 64u && (out_dim <= 512u || n_tok > 8u);
    if (((n_tok > 8u && w->type == DS4_TENSOR_F32) || f16_batch) &&
        (in_dim % 32) == 0) {
        rc = ds4_gpu_qwen4_dense_mm_tensor(out, x, m->map, m->size, w->abs_offset, w->type, n_tok,
                                           (uint32_t)in_dim, (uint32_t)out_dim);
        if (rc) return true;
    }
    /* Q8 decode batches of 8 or 16 rows on fp32 simdgroup matrices: the
     * per-token kernel reads the matrix once per four rows and streams x
     * once per four weight rows; this reads the weights once and x once per
     * 32 rows.  DS4_QWEN4_NO_BATCH_MM=1 keeps the per-token kernel for A/B. */
    if (!legacy && w->type == DS4_TENSOR_Q8_0 && n_tok >= 5u && n_tok <= 32u &&
        (in_dim % 32) == 0 && (out_dim % 32) == 0 && getenv("DS4_QWEN4_NO_BATCH_MM") == NULL &&
        ds4_gpu_tensor_bytes(x) >= 32u * in_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(out) >= 32u * out_dim * sizeof(float)) {
        /* Other widths round up to the tile: rows past n_tok are scratch
         * rows both buffers hold, computed and ignored.  Past sixteen rows
         * the second half is a second product on views of both. */
        const uint32_t first = n_tok <= 8u ? 8u : 16u;
        rc = ds4_gpu_qwen4_batch_mm_q8_tensor(out, x, m->map, m->size, w->abs_offset, first,
                                              (uint32_t)in_dim, (uint32_t)out_dim);
        if (rc && n_tok > 16u) {
            ds4_gpu_tensor *x2 = ds4_gpu_tensor_view(x, 16u * in_dim * sizeof(float), 16u * in_dim * sizeof(float));
            ds4_gpu_tensor *o2 = ds4_gpu_tensor_view(out, 16u * out_dim * sizeof(float), 16u * out_dim * sizeof(float));
            rc = x2 && o2 && ds4_gpu_qwen4_batch_mm_q8_tensor(o2, x2, m->map, m->size, w->abs_offset, 16u,
                                                              (uint32_t)in_dim, (uint32_t)out_dim);
            ds4_gpu_tensor_free(o2);
            ds4_gpu_tensor_free(x2);
        }
        if (rc) return true;
    }
    switch (w->type) {
    case DS4_TENSOR_Q8_0: rc = ds4_gpu_qwen4_matmul_q8_0_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n_tok); break;
    case DS4_TENSOR_F16:  rc = ds4_gpu_matmul_f16_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n_tok); break;
    case DS4_TENSOR_F32:  rc = ds4_gpu_matmul_f32_tensor(out, m->map, m->size, w->abs_offset, in_dim, out_dim, x, n_tok); break;
    case DS4_TENSOR_Q4_0: rc = ds4_gpu_matmul_quant_tensor(out, m->map, m->size, w->abs_offset, w->type, in_dim, out_dim, x, n_tok); break;
    case DS4_TENSOR_BF16: {
        ds4_gpu_tensor *outs[1] = { out };
        const uint64_t offs[1] = { w->abs_offset };
        const uint32_t types[1] = { w->type };
        const uint32_t rows[1] = { (uint32_t)out_dim };
        rc = ds4_gpu_qwen4_multi_gemv_tensor(x, n_tok, (uint32_t)in_dim, 1, outs, m->map, m->size, offs, types, rows);
        break;
    }
    default: break;
    }
#endif
    if (!rc) {
        fprintf(stderr, "ds4: Qwen3.8 matmul failed for %.*s (type %u, %" PRIu64 "x%" PRIu64 ", %u tokens)\n",
                (int)w->name.len, w->name.ptr, w->type, in_dim, out_dim, n_tok);
    }
    return rc != 0;
}

static void resident_need(bool ok, const char *what) {
    if (ok) return;
    fprintf(stderr, "Qwen resident arithmetic: %s\n", what);
    exit(1);
}

static uint32_t resident_mix(uint64_t index, uint32_t seed) {
    uint32_t v = (uint32_t)index ^ (uint32_t)(index >> 32) ^ seed;
    v ^= v >> 16; v *= 0x7feb352du;
    v ^= v >> 15; v *= 0x846ca68bu;
    return v ^ (v >> 16);
}

static float resident_value(uint64_t index, uint32_t seed) {
    return ((int32_t)(resident_mix(index, seed) % 4097u) - 2048) * 0.00053117f;
}

static uint64_t resident_weight_hash(const ds4_model *m) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint64_t i = 0; i < m->size; i++) {
        hash ^= m->map[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

typedef struct {
    ds4_gpu_tensor *storage, *view;
    uint64_t count;
} resident_guarded;

enum { RESIDENT_GUARD = 32, RESIDENT_CHUNK = 16384 };

static resident_guarded resident_alloc(uint64_t count) {
    resident_guarded t = { .count = count };
    t.storage = ds4_gpu_tensor_alloc((count + 2u * RESIDENT_GUARD) * sizeof(float));
    resident_need(t.storage != NULL, "guarded tensor allocation");
    t.view = ds4_gpu_tensor_view(t.storage, RESIDENT_GUARD * sizeof(float), count * sizeof(float));
    resident_need(t.view != NULL, "guarded tensor view");
    resident_need(ds4_gpu_tensor_fill_f32(t.storage, 17.25f, count + 2u * RESIDENT_GUARD), "tensor guard initialization");
    return t;
}

static void resident_guards(const resident_guarded *t) {
    float left[RESIDENT_GUARD], right[RESIDENT_GUARD];
    resident_need(ds4_gpu_tensor_read(t->storage, 0, left, sizeof(left)) &&
                  ds4_gpu_tensor_read(t->storage, (RESIDENT_GUARD + t->count) * sizeof(float), right, sizeof(right)), "tensor guard read");
    for (uint32_t i = 0; i < RESIDENT_GUARD; i++)
        resident_need(left[i] == 17.25f && right[i] == 17.25f, "tensor guard overwritten");
}

static void resident_free(resident_guarded *t) {
    ds4_gpu_tensor_free(t->view);
    ds4_gpu_tensor_free(t->storage);
}

static void resident_input(resident_guarded *t, uint32_t seed, bool verify) {
    float chunk[RESIDENT_CHUNK];
    for (uint64_t off = 0; off < t->count; off += RESIDENT_CHUNK) {
        const uint64_t n = t->count - off < RESIDENT_CHUNK ? t->count - off : RESIDENT_CHUNK;
        if (verify) {
            resident_need(ds4_gpu_tensor_read(t->view, off * sizeof(float), chunk, n * sizeof(float)), "input read");
            for (uint64_t i = 0; i < n; i++)
                resident_need(chunk[i] == resident_value(off + i, seed), "projection input mutated");
        } else {
            for (uint64_t i = 0; i < n; i++) chunk[i] = resident_value(off + i, seed);
            resident_need(ds4_gpu_tensor_write(t->view, off * sizeof(float), chunk, n * sizeof(float)), "input upload");
        }
    }
    resident_guards(t);
}

static float *resident_read(ds4_gpu_tensor *t, uint64_t count) {
    float *values = malloc(count * sizeof(float));
    resident_need(values != NULL && ds4_gpu_tensor_read(t, 0, values, count * sizeof(float)), "output read");
    for (uint64_t i = 0; i < count; i++) resident_need(isfinite(values[i]), "nonfinite projection/attention output");
    return values;
}

static void resident_equal(const float *expected, const float *actual, uint64_t count, const char *name, const char *mode, uint32_t rows) {
    if (memcmp(expected, actual, count * sizeof(float)) == 0) return;
    uint64_t first = 0, changed = 0;
    double worst = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(expected + i, actual + i, sizeof(float))) {
            if (!changed) first = i;
            changed++;
            worst = fmax(worst, fabs((double)expected[i] - actual[i]));
        }
    }
    fprintf(stderr, "Qwen resident arithmetic: %s %s T=%u: %" PRIu64 "/%" PRIu64
            " bits differ, max_error=%.9g, first=%" PRIu64 " expected=%.9g actual=%.9g\n",
            name, mode, rows, changed, count, worst, first, expected[first], actual[first]);
    exit(1);
}

static void *resident_map;
static uint64_t resident_map_bytes;

static ds4_model resident_weights(ds4_tensor *w, const char *name, uint32_t type, uint32_t k, uint32_t n) {
    const uint64_t bytes = type == DS4_TENSOR_Q8_0 ? (uint64_t)n * (k / 32u) * 34u :
                           (uint64_t)k * n * (type == DS4_TENSOR_F16 ? 2u : 4u);
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE), size = (bytes + page - 1u) / page * page;
    uint8_t *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    resident_need(map != MAP_FAILED, "synthetic weight mapping");
    if (type == DS4_TENSOR_Q8_0) {
        for (uint64_t b = 0; b < (uint64_t)n * (k / 32u); b++) {
            _Float16 scale = (_Float16)(0.001731f + (resident_mix(b, 37u) % 11u) * 0.000113f);
            memcpy(map + b * 34u, &scale, sizeof(scale));
            for (uint32_t j = 0; j < 32u; j++)
                ((int8_t *)(map + b * 34u + 2u))[j] = (int8_t)((int)(resident_mix(b * 32u + j, 91u) % 255u) - 127);
        }
    } else if (type == DS4_TENSOR_F16) {
        for (uint64_t i = 0; i < (uint64_t)k * n; i++) ((_Float16 *)map)[i] = (_Float16)(resident_value(i, 73u) * 0.0625f);
    } else {
        for (uint64_t i = 0; i < (uint64_t)k * n; i++) ((float *)map)[i] = resident_value(i, 113u) * 0.0625f;
    }
    resident_need(ds4_gpu_synchronize() && ds4_gpu_set_model_map(map, size), "synthetic model map registration");
    if (resident_map) resident_need(munmap(resident_map, resident_map_bytes) == 0, "previous synthetic map release");
    resident_map = map;
    resident_map_bytes = size;
    *w = (ds4_tensor){ .name = { name, strlen(name) }, .ndim = 2, .dim = { k, n },
                      .type = type, .elements = (uint64_t)k * n, .bytes = bytes };
    return (ds4_model){ .map = map, .size = size };
}

static void resident_projection_case(const ds4_model *m, const ds4_tensor *w, uint32_t rows, bool require_difference) {
    const uint64_t k = w->dim[0], n = w->dim[1], capacity = rows < 32u ? 32u : rows;
    const uint64_t values = (uint64_t)rows * n;
    const uint64_t weight_hash = resident_weight_hash(m);
    resident_guarded x = resident_alloc(capacity * k), out = resident_alloc(capacity * n);
    resident_input(&x, 197u, false);

    resident_need(resident_base_projection(out.view, m, w, x.view, rows, 0), "frozen main projection");
    float *base = resident_read(out.view, values);
    resident_guards(&out);
    /* The old SSD reference disables float batch operands and split-K.
     * Only the independent reference call sees these switches. */
    resident_need(setenv("DS4_QWEN4_DENSE_MM_LEGACY", "1", 1) == 0 &&
                  setenv("DS4_QWEN4_NO_DENSE_MM_KSPLIT", "1", 1) == 0, "SSD reference switches");
    resident_need(resident_base_projection(out.view, m, w, x.view, rows, 0), "legacy unsplit SSD reference");
    float *ssd = resident_read(out.view, values);
    resident_guards(&out);
    unsetenv("DS4_QWEN4_DENSE_MM_LEGACY");
    unsetenv("DS4_QWEN4_NO_DENSE_MM_KSPLIT");
    if (require_difference && memcmp(base, ssd, values * sizeof(float)) == 0) {
        fprintf(stderr, "Qwen resident arithmetic: %s T=%u has no base/SSD difference\n", w->name.ptr, rows);
        resident_need(false, "fixture must detect the old resident prefill policy");
    }

    const char *modes[] = { "resident prefill", "SSD prefill", "resident decode", "SSD decode" };
    for (unsigned mode = 0; mode < 4; mode++) {
        ds4_qwen4_gpu_graph g = { .ssd_streaming = (mode & 1u) != 0,
            .projection_phase = mode < 2 ? QWEN4_PROJECTION_PREFILL : QWEN4_PROJECTION_DECODE };
        resident_need(ds4_gpu_tensor_fill_f32(out.storage, 17.25f, out.count + 2u * RESIDENT_GUARD), "projection output reset");
        resident_need(qwen4_gemv_rows(&g, out.view, m, w, x.view, rows, 0), modes[mode]);
        float *actual = resident_read(out.view, values);
        resident_equal(mode == 1 ? ssd : base, actual, values, w->name.ptr, modes[mode], rows);
        resident_guards(&out);
        free(actual);
    }
    resident_input(&x, 197u, true);
    resident_need(resident_weight_hash(m) == weight_hash, "projection weights mutated");
    fprintf(stderr, "Qwen resident arithmetic: %s T=%u matches resident/SSD prefill and decode references\n", w->name.ptr, rows);
    free(ssd);
    free(base);
    resident_free(&out);
    resident_free(&x);
}

/* The attention calls in main 0aaea5a used the original public API for
 * each internal range; the optional partial buffer depended on parent T. */
static bool resident_base_attention(ds4_qwen4_gpu_graph *g, ds4_gpu_tensor *out,
        ds4_gpu_tensor *q, ds4_gpu_tensor *gate, uint32_t rows, uint32_t pos0, uint32_t parent_rows) {
    return ds4_gpu_qwen4_attn_decode_tensor(out, q, gate, g->layer_k_cache[0], g->layer_v_cache[0],
            g->sel_tokens, g->n_sel, parent_rows <= 2u ? g->attn_part : NULL, rows,
            DS4_N_HEAD, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, pos0, false, g->sel_stride,
            1.0f / sqrtf((float)DS4_N_HEAD_DIM)) != 0;
}

static void resident_attention_partition(void) {
    const uint32_t rows = 9u, pos0 = 2042u, tail = 3u;
    const uint64_t row = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv = (uint64_t)(pos0 + rows) * DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    resident_guarded q = resident_alloc(rows * row), gate = resident_alloc(rows * row);
    resident_guarded whole = resident_alloc(rows * row), out = resident_alloc(tail * row);
    resident_input(&q, 271u, false);
    resident_input(&gate, 313u, false);
    ds4_gpu_tensor *k = ds4_gpu_tensor_alloc(kv * sizeof(_Float16));
    ds4_gpu_tensor *v = ds4_gpu_tensor_alloc(kv * sizeof(_Float16));
    resident_need(k && v, "attention KV allocation");
    _Float16 chunk[RESIDENT_CHUNK];
    for (uint64_t off = 0; off < kv; off += RESIDENT_CHUNK) {
        const uint64_t count = kv - off < RESIDENT_CHUNK ? kv - off : RESIDENT_CHUNK;
        for (uint64_t i = 0; i < count; i++) chunk[i] = (_Float16)resident_value(off + i, 337u);
        resident_need(ds4_gpu_tensor_write(k, off * sizeof(*chunk), chunk, count * sizeof(*chunk)), "attention K upload");
        for (uint64_t i = 0; i < count; i++) chunk[i] = (_Float16)resident_value(off + i, 353u);
        resident_need(ds4_gpu_tensor_write(v, off * sizeof(*chunk), chunk, count * sizeof(*chunk)), "attention V upload");
    }
    ds4_qwen4_gpu_graph g = { .layer_k_cache = { k }, .layer_v_cache = { v }, .sel_stride = 2052u };
    ds4_gpu_tensor *qt = ds4_gpu_tensor_view(q.view, 6u * row * sizeof(float), tail * row * sizeof(float));
    ds4_gpu_tensor *gt = ds4_gpu_tensor_view(gate.view, 6u * row * sizeof(float), tail * row * sizeof(float));
    resident_need(qt && gt, "attention tail views");
    resident_need(resident_base_attention(&g, out.view, qt, gt, tail, 2048u, 2048u), "original three-row attention dispatch");
    float *base = resident_read(out.view, tail * row);
    /* Nine rows select the original matrix path independently of the new
     * prefill API. Its final three queries see exactly the same KV prefix. */
    resident_need(resident_base_attention(&g, whole.view, q.view, gate.view, rows, pos0, rows), "original nine-row matrix attention");
    float *matrix = resident_read(whole.view, rows * row);
    resident_need(memcmp(base, matrix + 6u * row, tail * row * sizeof(float)) != 0,
                  "attention fixture must detect scalar/matrix policy drift");
    const char *modes[] = { "resident prefill", "SSD prefill", "resident decode", "SSD decode" };
    for (unsigned mode = 0; mode < 4; mode++) {
        g.ssd_streaming = (mode & 1u) != 0;
        g.projection_phase = mode < 2 ? QWEN4_PROJECTION_PREFILL : QWEN4_PROJECTION_DECODE;
        resident_need(ds4_gpu_tensor_fill_f32(out.storage, 17.25f, out.count + 2u * RESIDENT_GUARD), "attention output reset");
        resident_need(qwen4_graph_attention_dispatch(&g, 0, out.view, qt, gt, tail, 2048u, false, 2048u), modes[mode]);
        float *actual = resident_read(out.view, tail * row);
        resident_equal(mode == 1 ? matrix + 6u * row : base, actual, tail * row, "dense attention partition", modes[mode], tail);
        resident_guards(&out);
        free(actual);
        /* A genuinely short parent keeps the original scalar policy even
         * in SSD mode, unlike a short partition of a large parent. */
        resident_need(qwen4_graph_attention_dispatch(&g, 0, out.view, qt, gt, tail, 2048u, false, tail), "short-parent attention");
        actual = resident_read(out.view, tail * row);
        resident_equal(base, actual, tail * row, "short parent attention", modes[mode], tail);
        resident_guards(&out);
        free(actual);
    }
    resident_input(&q, 271u, true);
    resident_input(&gate, 313u, true);
    resident_guards(&whole);
    free(matrix);
    free(base);
    ds4_gpu_tensor_free(gt);
    ds4_gpu_tensor_free(qt);
    ds4_gpu_tensor_free(v);
    ds4_gpu_tensor_free(k);
    resident_free(&out);
    resident_free(&whole);
    resident_free(&gate);
    resident_free(&q);
    fprintf(stderr, "Qwen resident arithmetic: three-row attention at pos=2048 matches independent range/parent references\n");
}

int main(void) {
    /* These arithmetic overrides would make both sides bypass the policy
     * under test. Leave shader math controls intact for fast/safe test runs. */
    const char *overrides[] = { "DS4_QWEN4_DENSE_MM_LEGACY", "DS4_QWEN4_NO_DENSE_MM_KSPLIT",
        "DS4_QWEN4_NO_BATCH_MM", "DS4_QWEN4_NO_ATTN_MM", "DS4_DIAG_QWEN4_PREFILL_OPT",
        "DS4_DIAG_QWEN4_PREFILL_BUNDLE" };
    for (unsigned i = 0; i < sizeof(overrides) / sizeof(overrides[0]); i++) unsetenv(overrides[i]);
    /* Keep optional weight-unpack scratch bounded also on M3 Ultra. This
     * control applies equally to the independent oracle and candidate. */
    resident_need(setenv("DS4_QWEN4_Q8_PREFILL_UNPACK", "0", 1) == 0, "bounded Q8 reference scratch");
    g_ds4_shape = DS4_SHAPE_QWEN4_EXP;
    resident_need(ds4_gpu_init(), "Metal initialization");
    ds4_tensor w;
    ds4_model m = resident_weights(&w, "HC-down F16", DS4_TENSOR_F16, 10240u, 320u);
    const uint32_t short_rows[] = { 1u, 3u, 8u, 9u, 29u, 64u, 65u };
    for (unsigned i = 0; i < sizeof(short_rows) / sizeof(short_rows[0]); i++) resident_projection_case(&m, &w, short_rows[i], short_rows[i] == 29u);
    m = resident_weights(&w, "HC-up F16", DS4_TENSOR_F16, 320u, 10240u);
    for (unsigned i = 0; i < sizeof(short_rows) / sizeof(short_rows[0]); i++) resident_projection_case(&m, &w, short_rows[i], short_rows[i] == 29u);
    m = resident_weights(&w, "attention Q8", DS4_TENSOR_Q8_0, 2560u, 6144u);
    const uint32_t q8_rows[] = { 1u, 3u, 4u, 5u, 8u, 9u, 16u, 17u, 29u, 32u, 33u };
    for (unsigned i = 0; i < sizeof(q8_rows) / sizeof(q8_rows[0]); i++) resident_projection_case(&m, &w, q8_rows[i], q8_rows[i] == 29u);
    m = resident_weights(&w, "GDN alpha F32", DS4_TENSOR_F32, 2560u, 48u);
    const uint32_t alpha_rows[] = { 1u, 3u, 8u, 9u, 29u, 575u, 1846u, 2048u };
    for (unsigned i = 0; i < sizeof(alpha_rows) / sizeof(alpha_rows[0]); i++) resident_projection_case(&m, &w, alpha_rows[i], alpha_rows[i] == 29u);
    m = resident_weights(&w, "router F32", DS4_TENSOR_F32, 2560u, 512u);
    resident_projection_case(&m, &w, 29u, true);
    resident_projection_case(&m, &w, 575u, false);
    resident_attention_partition();
    ds4_gpu_cleanup();
    resident_need(munmap(resident_map, resident_map_bytes) == 0, "last synthetic map release");
    puts("Qwen resident and SSD arithmetic policy tests passed");
    return 0;
}

#else
int main(void) {
    puts("Qwen resident arithmetic policy tests require Metal");
    return 0;
}
#endif
