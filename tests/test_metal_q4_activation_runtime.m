// SPDX-License-Identifier: MIT
// Exercise production Q-B activation reuse and its scratch lifetime without a GGUF.
// Including the backend permits dispatch assertions without a public test API.
#include "../ds4_metal.m"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
int ds4_deepseek4_attention_bounds(const int *a, uint32_t b, uint32_t c,
        uint32_t d, uint32_t e, uint32_t f, uint32_t *g) {
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
    return 0;
}

enum { QBA_K = 1024, QBA_M = 32768, QBA_GUARD = 256 };
static const uint32_t qba_poison = 0x7fc12345u;
typedef struct { uint16_t d, dmin; uint8_t scales[12], qs[128]; } qba_block;
_Static_assert(sizeof(qba_block) == 144, "Q4_K block ABI");
typedef struct {
    ds4_gpu_tensor *base, *view;
    NSUInteger bytes;
} qba_tensor;
typedef struct {
    void *model;
    size_t model_bytes, weight_offset, weight_bytes;
    uint64_t weight_hash;
    unsigned cases;
} qba_fixture;

static void qba_require(bool ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "Metal Q4 activation runtime FAIL: %s\n", message);
        exit(1);
    }
}
static uint32_t qba_random(uint32_t *state) {
    return *state = *state * 1664525u + 1013904223u;
}
static uint64_t qba_hash(const void *data, size_t bytes) {
    const uint8_t *p = data;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; ++i)
        h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}
static void qba_reset(qba_tensor *t) {
    uint32_t *p = ds4_gpu_tensor_contents(t->base);
    for (NSUInteger i = 0; i < (t->bytes + 2u*QBA_GUARD)/4u; ++i)
        p[i] = qba_poison;
}
static qba_tensor qba_alloc(NSUInteger bytes) {
    qba_tensor t = {.bytes=bytes};
    t.base = ds4_gpu_tensor_alloc(bytes + 2u*QBA_GUARD);
    t.view = t.base ? ds4_gpu_tensor_view(t.base, QBA_GUARD, bytes) : NULL;
    qba_require(t.base && t.view, "guarded tensor allocation");
    qba_reset(&t);
    return t;
}
static void qba_free(qba_tensor *t) {
    ds4_gpu_tensor_free(t->view);
    ds4_gpu_tensor_free(t->base);
    *t = (qba_tensor){0};
}
static void qba_fill(qba_tensor *x, uint32_t seed) {
    float *p = ds4_gpu_tensor_contents(x->view);
    for (NSUInteger i = 0; i < x->bytes/4u; ++i)
        p[i] = i % 17u ? ((int)(qba_random(&seed) % 8193u) - 4096)/4096.f : -0.f;
}
static void qba_guards(id<MTLBuffer> buffer, NSUInteger bytes) {
    const uint32_t *p = buffer.contents;
    for (NSUInteger i = 0; i < QBA_GUARD/4u; ++i)
        qba_require(p[i] == qba_poison &&
                    p[(QBA_GUARD + bytes)/4u + i] == qba_poison,
                    "tensor guard overwritten");
}
static void qba_equal(id<MTLBuffer> actual, id<MTLBuffer> expected,
                       NSUInteger bytes, const char *label) {
    qba_guards(actual, bytes);
    qba_guards(expected, bytes);
    const uint32_t *a = (const uint32_t *)((const char *)actual.contents + QBA_GUARD);
    const uint32_t *b = (const uint32_t *)((const char *)expected.contents + QBA_GUARD);
    for (NSUInteger i = 0; i < bytes/4u; ++i) {
        if (a[i] != b[i] || (a[i] & 0x7f800000u) == 0x7f800000u) {
            fprintf(stderr, "%s: index=%lu actual=%08x expected=%08x\n",
                    label, (unsigned long)i, a[i], b[i]);
            qba_require(false, "output must be finite and bitwise equal");
        }
    }
}
static int qba_reference(qba_fixture *f, qba_tensor *out,
                         qba_tensor *x, uint32_t tokens) {
    return ds4_gpu_test_q4_attn_q_b_mm_variant_tensor(
        out->view, NULL, f->model, f->model_bytes, f->weight_offset,
        QBA_K, QBA_M, x->view, tokens, DS4_GPU_TEST_Q4_QB_MM_Q4_F32, false);
}
static int qba_automatic(qba_fixture *f, qba_tensor *out,
                         qba_tensor *x, uint32_t tokens) {
    return ds4_gpu_matmul_quant_tensor(out->view, f->model,
        f->model_bytes, f->weight_offset, DS4_METAL_TENSOR_Q4_K,
        QBA_K, QBA_M, x->view, tokens);
}
static void qba_init(qba_fixture *f) {
    *f = (qba_fixture){0};
    const size_t page = (size_t)getpagesize();
    f->weight_offset = page;
    f->weight_bytes = (size_t)QBA_M*(QBA_K/256u)*sizeof(qba_block);
    f->model_bytes = f->weight_bytes + 2u*page;
    qba_require(posix_memalign(&f->model, page, f->model_bytes) == 0,
                "synthetic model allocation");
    memset(f->model, 0xa5, f->model_bytes);
    qba_block *w = (qba_block *)((char *)f->model + f->weight_offset);
    uint32_t seed = 83129u;
    for (size_t i = 0; i < f->weight_bytes/sizeof(*w); ++i) {
        w[i].d = 0x2000u | (qba_random(&seed) & 0x3ffu);
        w[i].dmin = i % 7u ? 0x1800u | (qba_random(&seed) & 0x3ffu) : 0u;
        for (unsigned j = 0; j < 12u; ++j) w[i].scales[j] = qba_random(&seed) >> 24;
        for (unsigned j = 0; j < 128u; ++j) w[i].qs[j] = qba_random(&seed) >> 24;
    }
    f->weight_hash = qba_hash(f->model, f->model_bytes);
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    const uint64_t offset = f->weight_offset, bytes = f->weight_bytes;
    qba_require(ds4_gpu_set_model_map_spans(f->model, f->model_bytes,
                    &offset, &bytes, 1u, bytes), "model range registration");
}

// Runtime selection/lifecycle cases follow below; the reference always uses
// the existing explicit Q4/F32 arm, independent of production eligibility.
static void qba_case(qba_fixture *f, uint32_t tokens, bool batch,
                      bool expect_reuse, const char *label) {
    @autoreleasepool {
        qba_tensor x = qba_alloc((NSUInteger)tokens*QBA_K*4u);
        qba_tensor expected = qba_alloc((NSUInteger)tokens*QBA_M*4u);
        qba_tensor actual = qba_alloc(expected.bytes);
        qba_fill(&x, 173u + tokens + f->cases);
        const uint64_t xhash = qba_hash(ds4_gpu_tensor_contents(x.base),
                                       x.bytes + 2u*QBA_GUARD);
        qba_require(qba_reference(f, &expected, &x, tokens), "reference dispatch");
        const uint64_t before = g_q4_qb_rhs_f16_launches;
        id<MTLBuffer> previous = g_q4_qb_rhs_f16_buffer;
        if (batch) qba_require(ds4_gpu_begin_commands(), "begin automatic batch");
        qba_require(qba_automatic(f, &actual, &x, tokens), "automatic dispatch");
        if (batch) qba_require(ds4_gpu_end_commands(), "finish automatic batch");
        qba_require(g_q4_qb_rhs_f16_launches == before + (expect_reuse ? 1u : 0u),
                    "unexpected production branch");
        qba_equal(ds4_gpu_tensor_buffer(actual.base),
                    ds4_gpu_tensor_buffer(expected.base), actual.bytes, label);
        qba_require(xhash == qba_hash(ds4_gpu_tensor_contents(x.base),
                        x.bytes + 2u*QBA_GUARD), "input or input guards modified");
        if (expect_reuse) {
            qba_require(g_q4_qb_rhs_f16_bytes == DS4_METAL_Q4_QB_RHS_BYTES &&
                        g_q4_qb_rhs_f16_buffer.length == DS4_METAL_Q4_QB_RHS_BYTES,
                        "scratch must stay at the fixed 4 MiB budget");
            qba_require(!previous || previous == g_q4_qb_rhs_f16_buffer,
                        "scratch changed while growing token count");
        }
        ++f->cases;
        qba_free(&actual); qba_free(&expected); qba_free(&x);
    }
}

static void qba_policy(qba_fixture *f) {
    const uint64_t rejected[][4] = {
        {DS4_METAL_TENSOR_Q4_0,QBA_K,QBA_M,128u},
        {DS4_METAL_TENSOR_Q8_0,QBA_K,QBA_M,128u},
        {DS4_METAL_TENSOR_Q4_K,512u,QBA_M,128u},
        {DS4_METAL_TENSOR_Q4_K,QBA_K,16384u,128u},
        {DS4_METAL_TENSOR_Q4_K,QBA_K,QBA_M,0u},
        {DS4_METAL_TENSOR_Q4_K,QBA_K,QBA_M,127u},
        {DS4_METAL_TENSOR_Q4_K,QBA_K,QBA_M,2049u},
        {DS4_METAL_TENSOR_Q4_K,QBA_K,QBA_M,UINT64_MAX},
    };
    for (unsigned i = 0; i < sizeof(rejected)/sizeof(rejected[0]); ++i) {
        qba_require(!ds4_gpu_q4_qb_activation_reuse_eligible(
            (uint32_t)rejected[i][0], rejected[i][1], rejected[i][2], rejected[i][3]),
            "ineligible shape admitted");
        ++f->cases;
    }
    g_batch_encoder_concurrent = YES;
    qba_require(!ds4_gpu_q4_qb_activation_reuse_eligible(12u,QBA_K,QBA_M,128u),
                "concurrent encoder admitted");
    g_batch_encoder_concurrent = NO;
    g_tp_split_world = 2;
    qba_require(!ds4_gpu_q4_qb_activation_reuse_eligible(12u,QBA_K,QBA_M,128u),
                "TP admitted");
    g_tp_split_world = 1;
    f->cases += 2u;
    ds4_gpu_set_quality(true);
    qba_case(f,128u,true,false,"quality fallback");
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(true);
    qba_case(f,128u,true,false,"SSD fallback");
    ds4_gpu_set_ssd_streaming(false);
    qba_require(setenv("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY","1",1) == 0,
                "copy fallback setup");
    qba_case(f,128u,true,false,"copy fallback");
    qba_require(unsetenv("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") == 0,
                "copy fallback restore");
    id<MTLComputePipelineState> saved = g_cpy_contig_f32_f16_pipeline;
    g_cpy_contig_f32_f16_pipeline = nil;
    qba_case(f,128u,false,false,"unavailable copy pipeline fallback");
    g_cpy_contig_f32_f16_pipeline = saved;
    qba_case(f,96u,true,false,"small batch fallback");
    qba_case(f,2049u,false,false,"large batch fallback");
}

static void qba_execution_phases(qba_fixture *f) {
    const ds4_gpu_execution_phase phases[] = {
        DS4_GPU_PHASE_AUTO, DS4_GPU_PHASE_PREFILL, DS4_GPU_PHASE_DECODE,
        DS4_GPU_PHASE_VERIFY, DS4_GPU_PHASE_BATCH_DECODE, DS4_GPU_PHASE_MIXED,
    };
    qba_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_AUTO,
                "phase cases must start in legacy AUTO");
    for (unsigned i = 0; i < sizeof(phases)/sizeof(phases[0]); ++i) {
        DS4_GPU_PHASE_SCOPE(scope, phases[i]);
        const bool prefill = phases[i] == DS4_GPU_PHASE_AUTO || phases[i] == DS4_GPU_PHASE_PREFILL;
        // The token count and weights are identical in every phase. Selection
        // must follow the explicit phase rather than inferring prefill from N.
        qba_case(f,128u,(i&1u)!=0u,prefill,"same-N explicit phase");
        qba_require(ds4_gpu_iq2_activation_reuse_eligible(true,
                        DS4_METAL_TENSOR_IQ2_XXS, DS4_METAL_TENSOR_Q2_K,
                        4096u,2048u,256u,6u,512u) == prefill,
                    "IQ2 same-shape phase policy");
        ++f->cases;
        qba_require(ds4_gpu_get_execution_phase() == phases[i],
                    "dispatch must preserve its caller phase");
    }
    qba_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_AUTO,
                "phase loop must restore AUTO");

    // Encode every phase in one command buffer and restore the CPU phase
    // before submission. This exercises dispatch selection at encoding time,
    // including adjacent half-RHS and ordinary F32-RHS matmuls.
    enum { TOKENS = 129, ARMS = 6 };
    qba_tensor x = qba_alloc((NSUInteger)TOKENS*QBA_K*4u);
    qba_tensor expected = qba_alloc((NSUInteger)TOKENS*QBA_M*4u);
    qba_tensor actual[ARMS];
    qba_fill(&x,831u);
    const uint64_t xhash = qba_hash(ds4_gpu_tensor_contents(x.base), x.bytes + 2u*QBA_GUARD);
    qba_require(qba_reference(f,&expected,&x,TOKENS),"phase batch reference");
    for (unsigned i = 0; i < ARMS; ++i) actual[i] = qba_alloc(expected.bytes);
    const uint64_t before = g_q4_qb_rhs_f16_launches;
    qba_require(ds4_gpu_begin_commands(),"begin interleaved phase batch");
    for (unsigned i = 0; i < ARMS; ++i) {
        DS4_GPU_PHASE_SCOPE(scope, phases[i]);
        qba_require(qba_automatic(f,&actual[i],&x,TOKENS),"phase batch dispatch");
        const uint64_t allowed = i < 2u ? i+1u : 2u;
        qba_require(g_q4_qb_rhs_f16_launches == before+allowed,
                    "phase batch selected wrong pipeline");
    }
    qba_require(ds4_gpu_get_execution_phase() == DS4_GPU_PHASE_AUTO,
                "phase must restore before deferred GPU execution");
    qba_require(ds4_gpu_end_commands(),"finish interleaved phase batch");
    for (unsigned i = 0; i < ARMS; ++i) {
        qba_equal(ds4_gpu_tensor_buffer(actual[i].base),
                    ds4_gpu_tensor_buffer(expected.base), actual[i].bytes,
                    "interleaved phase batch");
        qba_free(&actual[i]);
        ++f->cases;
    }
    qba_require(xhash == qba_hash(ds4_gpu_tensor_contents(x.base), x.bytes + 2u*QBA_GUARD),
                "phase batch modified input");
    qba_free(&expected); qba_free(&x);
}

static void qba_rejections(qba_fixture *f) {
    qba_tensor x = qba_alloc(128u*QBA_K*4u);
    qba_tensor out = qba_alloc(128u*QBA_M*4u);
    qba_fill(&x,93u);
    uint64_t inner = 0u;
    id<MTLBuffer> weights = ds4_gpu_wrap_model_range(f->model,f->model_bytes,
        f->weight_offset,f->weight_bytes,&inner);
    qba_require(weights != nil,"weight view for rejection tests");
    const uint64_t before = g_q4_qb_rhs_f16_launches;
    const uint64_t out_hash = qba_hash(ds4_gpu_tensor_contents(out.base),
                                       out.bytes + 2u*QBA_GUARD);
    qba_require(!ds4_gpu_commands_active(),"rejection tests require idle queue");
    ds4_gpu_tensor *alias = ds4_gpu_tensor_view(out.view,0u,x.bytes);
    ds4_gpu_tensor *short_x = ds4_gpu_tensor_view(x.view,0u,x.bytes-4u);
    ds4_gpu_tensor *short_out = ds4_gpu_tensor_view(out.view,0u,out.bytes-4u);
    qba_require(alias && short_x && short_out,"rejection views");
    qba_require(ds4_gpu_try_q4_qb_activation_reuse(weights,inner,alias,out.view,128u)==0,
                "input/output alias must decline");
    qba_require(ds4_gpu_try_q4_qb_activation_reuse(weights,inner,short_x,out.view,128u)==0,
                "short input must decline");
    qba_require(ds4_gpu_try_q4_qb_activation_reuse(weights,inner,x.view,short_out,128u)==0,
                "short output must decline");
    qba_require(ds4_gpu_try_q4_qb_activation_reuse(weights,weights.length,x.view,out.view,128u)==0,
                "short weight view must decline");
    qba_require(ds4_gpu_try_q4_qb_activation_reuse(nil,0u,x.view,out.view,128u)==0,
                "nil weights must decline");
    // Inject already allocated arena aliases. Capacity stays fixed, so ensure
    // cannot replace the arena and conceal an alias before the dispatch check.
    id<MTLBuffer> saved = g_q4_qb_rhs_f16_buffer;
    const NSUInteger saved_bytes = g_q4_qb_rhs_f16_bytes;
    id<MTLBuffer> alias_buffers[] = {ds4_gpu_tensor_buffer(x.base),
                                    ds4_gpu_tensor_buffer(out.base),weights};
    for (unsigned i = 0; i < 3u; ++i) {
        g_q4_qb_rhs_f16_buffer = alias_buffers[i];
        g_q4_qb_rhs_f16_bytes = DS4_METAL_Q4_QB_RHS_BYTES;
        qba_require(ds4_gpu_try_q4_qb_activation_reuse(weights,inner,x.view,out.view,128u)==0,
                    "scratch alias must decline before encoding");
        qba_require(!g_batch_cb && !g_batch_enc && g_q4_qb_rhs_f16_launches == before,
                    "declined call encoded work");
    }
    g_q4_qb_rhs_f16_buffer = saved;
    g_q4_qb_rhs_f16_bytes = saved_bytes;
    qba_require(out_hash == qba_hash(ds4_gpu_tensor_contents(out.base),
                    out.bytes + 2u*QBA_GUARD),"declined call touched output");
    ds4_gpu_tensor_free(short_out); ds4_gpu_tensor_free(short_x);
    ds4_gpu_tensor_free(alias);
    qba_free(&out); qba_free(&x);
    f->cases += 8u;
}

static void qba_batch(qba_fixture *f, bool cleanup) {
    enum { TOKENS = 129, REPEATS = 3 };
    qba_tensor input[REPEATS], expected[REPEATS], actual[REPEATS];
    qba_tensor x = qba_alloc((NSUInteger)TOKENS*QBA_K*4u);
    // Keep every resource alive independently of tensor handles, including in
    // unretained command-buffer mode while cleanup submits the pending batch.
    id<MTLBuffer> input_buffers[REPEATS], expected_buffers[REPEATS], actual_buffers[REPEATS];
    id<MTLBuffer> x_buffer = ds4_gpu_tensor_buffer(x.base);
    uint64_t input_hash[REPEATS];
    for (unsigned i = 0; i < REPEATS; ++i) {
        input[i] = qba_alloc(x.bytes);
        expected[i] = qba_alloc((NSUInteger)TOKENS*QBA_M*4u);
        actual[i] = qba_alloc(expected[i].bytes);
        qba_fill(&input[i], (cleanup ? 795u : 973u) + i*119u);
        qba_require(qba_reference(f,&expected[i],&input[i],TOKENS),"batch reference");
        input_buffers[i] = ds4_gpu_tensor_buffer(input[i].base);
        expected_buffers[i] = ds4_gpu_tensor_buffer(expected[i].base);
        actual_buffers[i] = ds4_gpu_tensor_buffer(actual[i].base);
        input_hash[i] = qba_hash(input_buffers[i].contents,input_buffers[i].length);
    }
    const uint64_t before = g_q4_qb_rhs_f16_launches;
    // Do not extend the arena's lifetime in the cleanup oracle itself.
    __weak id<MTLBuffer> scratch = g_q4_qb_rhs_f16_buffer;
    qba_require(ds4_gpu_begin_commands(),"begin repeated reuse batch");
    for (unsigned i = 0; i < REPEATS; ++i) {
        qba_require(ds4_gpu_tensor_copy(x.view,0u,input[i].view,0u,x.bytes),
                    "GPU activation producer");
        qba_require(qba_automatic(f,&actual[i],&x,TOKENS),"batch automatic dispatch");
        qba_require(g_q4_qb_rhs_f16_buffer == scratch,"scratch replaced during batch");
    }
    qba_require(g_q4_qb_rhs_f16_launches == before + REPEATS,
                "repeated batch selected wrong path");
    const NSUInteger output_bytes = actual[0].bytes;
    const NSUInteger x_bytes = x.bytes;
    if (cleanup) {
        qba_require(ds4_gpu_commands_active() && g_batch_has_work,
                    "cleanup must drain an unsubmitted batch");
        for (unsigned i = 0; i < REPEATS; ++i) {
            qba_free(&input[i]); qba_free(&expected[i]); qba_free(&actual[i]);
        }
        qba_free(&x);
        ds4_gpu_cleanup();
        qba_require(!g_q4_qb_rhs_f16_buffer && !g_q4_qb_rhs_f16_bytes &&
                    !g_batch_cb,"cleanup did not release scratch after draining");
    } else qba_require(ds4_gpu_end_commands(),"finish repeated reuse batch");
    for (unsigned i = 0; i < REPEATS; ++i) {
        qba_equal(actual_buffers[i],expected_buffers[i],output_bytes,
                    cleanup ? "cleanup drain" : "scratch reuse batch");
        qba_require(input_hash[i] == qba_hash(input_buffers[i].contents,input_buffers[i].length),
                    "immutable batch source modified");
        ++f->cases;
    }
    qba_guards(x_buffer,x_bytes);
    if (!cleanup) {
        for (unsigned i = 0; i < REPEATS; ++i) {
            qba_free(&input[i]); qba_free(&expected[i]); qba_free(&actual[i]);
        }
        qba_free(&x);
    }
}

int main(void) {
    @autoreleasepool {
        qba_require(unsetenv("DS4_METAL_DISABLE_CONTIG_F32_F16_COPY") == 0,
                    "default copy setup");
        qba_require(ds4_gpu_init(),"Metal initialization; run with GPU access");
        if (!ds4_gpu_device_is_pre_m5_apple_silicon()) {
            fprintf(stderr,"SKIP: runtime activation reuse requires Apple M1-M4\n");
            ds4_gpu_cleanup();
            return 0;
        }
        qba_fixture f;
        qba_init(&f);
        const uint32_t tokens[] = {128u,129u,256u,257u,512u,2048u};
        for (unsigned i = 0; i < sizeof(tokens)/sizeof(tokens[0]); ++i)
            qba_case(&f,tokens[i],(i&1u)!=0u,true,"automatic Q-B");
        qba_policy(&f);
        qba_execution_phases(&f);
        qba_rejections(&f);
        qba_batch(&f,false);
        qba_require(f.weight_hash == qba_hash(f.model,f.model_bytes),
                    "weight mapping or its guards modified");
        qba_batch(&f,true);
        qba_require(f.weight_hash == qba_hash(f.model,f.model_bytes),
                    "cleanup batch modified model mapping");
        free(f.model);
        printf("PASS: %u Metal Q4 activation runtime cases; automatic selection, "
               "bitwise outputs, guards, fixed 4 MiB scratch, fallbacks, alias "
               "rejection, all six execution phases, mixed-phase batch, "
               "batch reuse and cleanup drain; unretained=%s\n",
               f.cases,getenv("DS4_METAL_UNRETAINED_COMMAND_BUFFERS") ? "on" : "off");
    }
    return 0;
}
