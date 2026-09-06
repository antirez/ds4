/* HIP-only hipBLASLt state and helpers.
 * Included from ds4_cuda.cu under __HIP_PLATFORM_AMD__ to keep ROCm
 * planning/cache code out of the CUDA host runtime body. */

#include <hipblaslt/hipblaslt-ext.hpp>

static hipblasLtHandle_t g_hipblaslt;
static int g_hipblaslt_ready;
/* Zero is unchecked, one version-matched, minus one disabled until cleanup. */
static int g_hipblaslt_prefill_state;
struct cuda_hipblaslt_gemm_plan {
    uint32_t out_dim;
    uint32_t n_tok;
    uint32_t in_dim;
    hipDataType output_type;
    int solution_index;
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t a_desc;
    hipblasLtMatrixLayout_t b_desc;
    hipblasLtMatrixLayout_t c_desc;
    hipblasLtMatrixLayout_t d_desc;
    hipblasLtMatmulAlgo_t algo;
};
static std::vector<cuda_hipblaslt_gemm_plan> g_hipblaslt_gemm_plans;

static void hipblaslt_gemm_plan_clear(void) {
    for (size_t i = 0; i < g_hipblaslt_gemm_plans.size(); i++) {
        cuda_hipblaslt_gemm_plan &p = g_hipblaslt_gemm_plans[i];
        if (p.d_desc) (void)hipblasLtMatrixLayoutDestroy(p.d_desc);
        if (p.c_desc) (void)hipblasLtMatrixLayoutDestroy(p.c_desc);
        if (p.b_desc) (void)hipblasLtMatrixLayoutDestroy(p.b_desc);
        if (p.a_desc) (void)hipblasLtMatrixLayoutDestroy(p.a_desc);
        if (p.desc) (void)hipblasLtMatmulDescDestroy(p.desc);
    }
    g_hipblaslt_gemm_plans.clear();
    __atomic_store_n(&g_hipblaslt_prefill_state, 0, __ATOMIC_RELAXED);
}

static int hipblaslt_ok(hipblasStatus_t st, const char *what) {
    if (st == HIPBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: hipBLASLt %s failed: status %d\n", what, (int)st);
    return 0;
}

static cuda_hipblaslt_gemm_plan *hipblaslt_gemm_plan_get(
        uint32_t out_dim,
        uint32_t n_tok,
        uint32_t in_dim,
        const char *label,
        hipDataType output_type = HIP_R_16F,
        int solution_index = -1) {
    for (size_t i = 0; i < g_hipblaslt_gemm_plans.size(); i++) {
        cuda_hipblaslt_gemm_plan &p = g_hipblaslt_gemm_plans[i];
        if (p.out_dim == out_dim && p.n_tok == n_tok && p.in_dim == in_dim &&
            p.output_type == output_type && p.solution_index == solution_index) return &p;
    }

    hipblasLtMatmulDesc_t desc = NULL;
    hipblasLtMatrixLayout_t a_desc = NULL, b_desc = NULL, c_desc = NULL, d_desc = NULL;
    hipblasLtMatmulPreference_t pref = NULL;
    hipblasLtMatmulHeuristicResult_t heur[8];
    int returned = 0;
    int ok = 0;
    do {
        if (!hipblaslt_ok(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                          "matmul desc create")) break;
        hipblasOperation_t op_a = HIPBLAS_OP_T;
        hipblasOperation_t op_b = HIPBLAS_OP_N;
        if (!hipblaslt_ok(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                          &op_a, sizeof(op_a)),
                          "set transA")) break;
        if (!hipblaslt_ok(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                          &op_b, sizeof(op_b)),
                          "set transB")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&a_desc, HIP_R_16F, in_dim, out_dim, in_dim),
                          "A layout create")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&b_desc, HIP_R_16F, in_dim, n_tok, in_dim),
                          "B layout create")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&c_desc, output_type, out_dim, n_tok, out_dim),
                          "C layout create")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&d_desc, output_type, out_dim, n_tok, out_dim),
                          "D layout create")) break;
        if (solution_index >= 0) {
            std::vector<int> indices{solution_index};
            std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
            if (!hipblaslt_ok(hipblaslt_ext::getAlgosFromIndex(g_hipblaslt, indices, algorithms),
                              "prefill fixed index")) break;
            if (algorithms.size() != 1 || algorithms[0].state != HIPBLAS_STATUS_SUCCESS ||
                hipblaslt_ext::getIndexFromAlgo(algorithms[0].algo) != solution_index) break;
            const float alpha = 1.0f, beta = 0.0f;
            size_t workspace = 0;
            if (!hipblaslt_ok(hipblaslt_ext::matmulIsAlgoSupported(
                    g_hipblaslt, desc, &alpha, a_desc, b_desc, &beta, c_desc, d_desc,
                    algorithms[0].algo, workspace), "prefill fixed index support")) break;
            if (workspace != 0) break;
            heur[0] = algorithms[0];
            ok = 1;
            break;
        }
        if (!hipblaslt_ok(hipblasLtMatmulPreferenceCreate(&pref), "preference create")) break;
        const size_t max_workspace = 0;
        if (!hipblaslt_ok(hipblasLtMatmulPreferenceSetAttribute(
                                  pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                  &max_workspace, sizeof(max_workspace)),
                          "set max workspace")) break;
        if (!hipblaslt_ok(hipblasLtMatmulAlgoGetHeuristic(g_hipblaslt, desc,
                                                          a_desc, b_desc, c_desc, d_desc,
                                                          pref, 8, heur, &returned),
                          "algo heuristic")) break;
        if (returned <= 0 || heur[0].state != HIPBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "ds4: hipBLASLt no algo for %s m=%u n=%u k=%u\n",
                    label ? label : "gemm", out_dim, n_tok, in_dim);
            break;
        }
        ok = 1;
    } while (0);
    if (pref) (void)hipblasLtMatmulPreferenceDestroy(pref);
    if (!ok) {
        if (d_desc) (void)hipblasLtMatrixLayoutDestroy(d_desc);
        if (c_desc) (void)hipblasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) (void)hipblasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) (void)hipblasLtMatrixLayoutDestroy(a_desc);
        if (desc) (void)hipblasLtMatmulDescDestroy(desc);
        return NULL;
    }

    cuda_hipblaslt_gemm_plan p;
    p.out_dim = out_dim;
    p.n_tok = n_tok;
    p.in_dim = in_dim;
    p.output_type = output_type;
    p.solution_index = solution_index;
    p.desc = desc;
    p.a_desc = a_desc;
    p.b_desc = b_desc;
    p.c_desc = c_desc;
    p.d_desc = d_desc;
    p.algo = heur[0].algo;
    g_hipblaslt_gemm_plans.push_back(p);
    return &g_hipblaslt_gemm_plans.back();
}

static int hipblaslt_gemm_tn_f16_out_f16(
        __half *out,
        const __half *w_rowmajor_out_in,
        const __half *x_rowmajor_tok_in,
        uint32_t out_dim,
        uint32_t n_tok,
        uint32_t in_dim,
        const char *label) {
    if (!g_hipblaslt_ready || !out || !w_rowmajor_out_in || !x_rowmajor_tok_in ||
        out_dim == 0 || n_tok == 0 || in_dim == 0) return 0;
    cuda_hipblaslt_gemm_plan *p = hipblaslt_gemm_plan_get(out_dim, n_tok, in_dim, label);
    if (!p) return 0;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return hipblaslt_ok(hipblasLtMatmul(g_hipblaslt, p->desc, &alpha,
                                        w_rowmajor_out_in, p->a_desc,
                                        x_rowmajor_tok_in, p->b_desc,
                                        &beta,
                                        out, p->c_desc,
                                        out, p->d_desc,
                                        &p->algo,
                                        NULL, 0, 0),
                        label ? label : "gemm");
}

/* gfx1151 prefill selections for hipBLASLt 100401 / 8d1ae90e.
 * These are Lt algorithm indices; other library revisions retain the fallback. */
static int hipblaslt_prefill_solution_index(
        uint32_t out_dim, uint32_t n_tok, uint32_t in_dim) {
    if (n_tok != 2048u && n_tok != 4096u) return -1;
    if (out_dim == 24u && in_dim == 16384u) return 2537;
    if (out_dim == 8192u && in_dim == 1024u) return 2539;
    if (in_dim == 4096u &&
        (out_dim == 64u || out_dim == 256u ||
         out_dim == 512u || out_dim == 1024u)) {
        return out_dim == 64u && n_tok == 2048u ? 2537 : 2539;
    }
    return -1;
}

/* The caller restricts this path to gfx1151 DS4 with quality mode off;
 * validate the dimensions here as well before choosing any fixed-index plan. */
static int hipblaslt_gemm_tn_f16_out_f32_prefill(
        float *out, const __half *w, const __half *x,
        uint32_t out_dim, uint32_t n_tok, uint32_t in_dim) {
    const int solution_index = hipblaslt_prefill_solution_index(out_dim, n_tok, in_dim);
    if (solution_index < 0 || !g_hipblaslt_ready || !out || !w || !x ||
        __atomic_load_n(&g_hipblaslt_prefill_state, __ATOMIC_RELAXED) < 0) return 0;
    if (__atomic_load_n(&g_hipblaslt_prefill_state, __ATOMIC_RELAXED) == 0) {
        int version = 0;
        char revision[128] = {0};
        if (!hipblaslt_ok(hipblasLtGetVersion(g_hipblaslt, &version), "prefill version") ||
            !hipblaslt_ok(hipblasLtGetGitRevision(g_hipblaslt, revision), "prefill revision") ||
            version != 100401 || strcmp(revision, "8d1ae90e") != 0) {
            __atomic_store_n(&g_hipblaslt_prefill_state, -1, __ATOMIC_RELAXED);
            return 0;
        }
        __atomic_store_n(&g_hipblaslt_prefill_state, 1, __ATOMIC_RELAXED);
    }
    cuda_hipblaslt_gemm_plan *p = hipblaslt_gemm_plan_get(
            out_dim, n_tok, in_dim, "prefill F16/F32", HIP_R_32F, solution_index);
    if (p) {
        const float alpha = 1.0f, beta = 0.0f;
        if (hipblaslt_ok(hipblasLtMatmul(g_hipblaslt, p->desc, &alpha,
                w, p->a_desc, x, p->b_desc, &beta, out, p->c_desc, out, p->d_desc,
                &p->algo, NULL, 0, 0), "prefill F16/F32")) return 1;
    }
    /* Original same-stream fallback overwrites the whole output. This
     * handles library rejection; asynchronous GPU faults remain fatal. */
    __atomic_store_n(&g_hipblaslt_prefill_state, -1, __ATOMIC_RELAXED);
    return 0;
}

/* Experimental DSpark attention A: five fixed-width plans, independent of
 * the prefill cache and its failure state. C/D contain eight interleaved groups. */
static int g_hipblaslt_dspark_a_state;
static cuda_hipblaslt_gemm_plan g_hipblaslt_dspark_a_plans[5];

static void hipblaslt_dspark_a_plan_destroy(cuda_hipblaslt_gemm_plan &p) {
    if (p.d_desc) (void)hipblasLtMatrixLayoutDestroy(p.d_desc);
    if (p.c_desc) (void)hipblasLtMatrixLayoutDestroy(p.c_desc);
    if (p.b_desc) (void)hipblasLtMatrixLayoutDestroy(p.b_desc);
    if (p.a_desc) (void)hipblasLtMatrixLayoutDestroy(p.a_desc);
    if (p.desc) (void)hipblasLtMatmulDescDestroy(p.desc);
    p = {};
}

static void hipblaslt_dspark_a_clear(void) {
    for (auto &p : g_hipblaslt_dspark_a_plans) hipblaslt_dspark_a_plan_destroy(p);
    __atomic_store_n(&g_hipblaslt_dspark_a_state, 0, __ATOMIC_RELAXED);
}

/* The caller additionally gates model family, verifier mode and gfx1151. */
static int hipblaslt_dspark_attention_a(
        __half *out, const __half *w, const __half *x, uint32_t n_tok) {
    if (n_tok < 2u || n_tok > 6u || !out || !w || !x ||
        !g_hipblaslt_ready || !g_cublas_ready ||
        g_rocblas_f16_solution_set != DS4_ROCBLAS_F16_SOLUTIONS_5_6_8D1AE90E ||
        __atomic_load_n(&g_hipblaslt_dspark_a_state, __ATOMIC_RELAXED) < 0) return 0;
    if (__atomic_load_n(&g_hipblaslt_dspark_a_state, __ATOMIC_RELAXED) == 0) {
        int version = 0;
        char revision[128] = {0};
        if (!hipblaslt_ok(hipblasLtGetVersion(g_hipblaslt, &version), "DSpark A version") ||
            !hipblaslt_ok(hipblasLtGetGitRevision(g_hipblaslt, revision), "DSpark A revision") ||
            version != 100401 || strcmp(revision, "8d1ae90e") != 0) {
            __atomic_store_n(&g_hipblaslt_dspark_a_state, -1, __ATOMIC_RELAXED);
            return 0;
        }
        __atomic_store_n(&g_hipblaslt_dspark_a_state, 1, __ATOMIC_RELAXED);
    }
    cuda_hipblaslt_gemm_plan &p = g_hipblaslt_dspark_a_plans[n_tok - 2u];
    const float alpha = 1.0f, beta = 0.0f;
    if (!p.desc) {
        int ok = 0;
        do {
            if (!hipblaslt_ok(hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                              "DSpark A matmul desc")) break;
            const hipblasOperation_t op_a = HIPBLAS_OP_T, op_b = HIPBLAS_OP_N;
            if (!hipblaslt_ok(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                              &op_a, sizeof(op_a)), "DSpark A transA") ||
                !hipblaslt_ok(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                              &op_b, sizeof(op_b)), "DSpark A transB")) break;
            if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&p.a_desc, HIP_R_16F, 4096, 1024, 4096),
                              "DSpark A weights") ||
                !hipblaslt_ok(hipblasLtMatrixLayoutCreate(&p.b_desc, HIP_R_16F, 4096, n_tok, 4096),
                              "DSpark A inputs") ||
                !hipblaslt_ok(hipblasLtMatrixLayoutCreate(&p.c_desc, HIP_R_16F, 1024, n_tok, 8192),
                              "DSpark A C") ||
                !hipblaslt_ok(hipblasLtMatrixLayoutCreate(&p.d_desc, HIP_R_16F, 1024, n_tok, 8192),
                              "DSpark A D")) break;
            const int32_t groups = 8;
            const int64_t strides[] = {INT64_C(4194304), INT64_C(4096) * n_tok, 1024, 1024};
            const hipblasLtMatrixLayout_t layouts[] = {p.a_desc, p.b_desc, p.c_desc, p.d_desc};
            bool layouts_ok = true;
            for (unsigned i = 0; i < 4u && layouts_ok; i++) {
                layouts_ok = hipblaslt_ok(hipblasLtMatrixLayoutSetAttribute(layouts[i],
                        HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &groups, sizeof(groups)), "DSpark A batches") &&
                    hipblaslt_ok(hipblasLtMatrixLayoutSetAttribute(layouts[i],
                        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i], sizeof(strides[i])),
                        "DSpark A stride");
            }
            if (!layouts_ok) break;
            std::vector<int> indices{2379};
            std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
            if (!hipblaslt_ok(hipblaslt_ext::getAlgosFromIndex(g_hipblaslt, indices, algorithms),
                              "DSpark A fixed index")) break;
            if (algorithms.size() != 1 || algorithms[0].state != HIPBLAS_STATUS_SUCCESS ||
                hipblaslt_ext::getIndexFromAlgo(algorithms[0].algo) != 2379) break;
            size_t workspace = 0;
            if (!hipblaslt_ok(hipblaslt_ext::matmulIsAlgoSupported(g_hipblaslt, p.desc,
                    &alpha, p.a_desc, p.b_desc, &beta, p.c_desc, p.d_desc,
                    algorithms[0].algo, workspace), "DSpark A support") || workspace != 0) break;
            p.algo = algorithms[0].algo;
            ok = 1;
        } while (0);
        if (!ok) {
            hipblaslt_dspark_a_plan_destroy(p);
            __atomic_store_n(&g_hipblaslt_dspark_a_state, -1, __ATOMIC_RELAXED);
            return 0;
        }
    }
    hipStream_t stream = nullptr;
    if (hipblaslt_ok(hipblasGetStream(g_cublas, &stream), "DSpark A stream") &&
        hipblaslt_ok(hipblasLtMatmul(g_hipblaslt, p.desc, &alpha,
                w, p.a_desc, x, p.b_desc, &beta, out, p.c_desc, out, p.d_desc,
                &p.algo, nullptr, 0, stream), "DSpark A matmul")) return 1;
    /* A synchronous library rejection falls back on the same stream and
     * overwrites every output. An asynchronous GPU fault is not recoverable. */
    __atomic_store_n(&g_hipblaslt_dspark_a_state, -1, __ATOMIC_RELAXED);
    return 0;
}
