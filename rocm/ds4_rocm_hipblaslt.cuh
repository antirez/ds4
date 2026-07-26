/* HIP-only hipBLASLt state and helpers.
 * Included from ds4_cuda.cu under __HIP_PLATFORM_AMD__ to keep ROCm
 * planning/cache code out of the CUDA host runtime body. */

static hipblasLtHandle_t g_hipblaslt;
static int g_hipblaslt_ready;
struct cuda_hipblaslt_gemm_plan {
    uint32_t out_dim;
    uint32_t n_tok;
    uint32_t in_dim;
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
        const char *label) {
    for (size_t i = 0; i < g_hipblaslt_gemm_plans.size(); i++) {
        cuda_hipblaslt_gemm_plan &p = g_hipblaslt_gemm_plans[i];
        if (p.out_dim == out_dim && p.n_tok == n_tok && p.in_dim == in_dim) return &p;
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
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&c_desc, HIP_R_16F, out_dim, n_tok, out_dim),
                          "C layout create")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&d_desc, HIP_R_16F, out_dim, n_tok, out_dim),
                          "D layout create")) break;
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

// ---------------------------------------------------------------------------
// FP32 strided-batched GEMM via hipBLASLt  (for attention prefill SGEMMs).
// Keyed on (m, n, k, batch_count, stride_a, stride_b, stride_c, trans).
// ---------------------------------------------------------------------------

struct cuda_hipblaslt_gemm_plan_sb_f32 {
    uint32_t m, n, k;
    uint32_t batch_count;
    int64_t stride_a, stride_b, stride_c;
    hipblasOperation_t trans_a, trans_b;
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t a_desc, b_desc, c_desc, d_desc;
    hipblasLtMatmulAlgo_t algo;
};

static std::vector<cuda_hipblaslt_gemm_plan_sb_f32> g_hipblaslt_sb_plans;

static void hipblaslt_sb_plan_clear(void) {
    for (size_t i = 0; i < g_hipblaslt_sb_plans.size(); i++) {
        auto &p = g_hipblaslt_sb_plans[i];
        if (p.d_desc) (void)hipblasLtMatrixLayoutDestroy(p.d_desc);
        if (p.c_desc) (void)hipblasLtMatrixLayoutDestroy(p.c_desc);
        if (p.b_desc) (void)hipblasLtMatrixLayoutDestroy(p.b_desc);
        if (p.a_desc) (void)hipblasLtMatrixLayoutDestroy(p.a_desc);
        if (p.desc) (void)hipblasLtMatmulDescDestroy(p.desc);
    }
    g_hipblaslt_sb_plans.clear();
}

static cuda_hipblaslt_gemm_plan_sb_f32 *hipblaslt_gemm_plan_get_sb_f32(
        hipblasOperation_t trans_a, hipblasOperation_t trans_b,
        uint32_t m, uint32_t n, uint32_t k,
        int64_t stride_a, int64_t stride_b, int64_t stride_c,
        uint32_t batch_count, const char *label) {
    for (size_t i = 0; i < g_hipblaslt_sb_plans.size(); i++) {
        auto &p = g_hipblaslt_sb_plans[i];
        if (p.m == m && p.n == n && p.k == k &&
            p.batch_count == batch_count &&
            p.stride_a == stride_a && p.stride_b == stride_b && p.stride_c == stride_c &&
            p.trans_a == trans_a && p.trans_b == trans_b)
            return &p;
    }

    hipblasLtMatmulDesc_t desc = NULL;
    hipblasLtMatrixLayout_t a_desc = NULL, b_desc = NULL, c_desc = NULL, d_desc = NULL;
    hipblasLtMatmulPreference_t pref = NULL;
    hipblasLtMatmulHeuristicResult_t heur[8];
    int returned = 0, ok = 0;
    do {
        if (!hipblaslt_ok(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                          "sb desc create")) break;
        if (!hipblaslt_ok(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                          &trans_a, sizeof(trans_a)),
                          "sb set transA")) break;
        if (!hipblaslt_ok(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                          &trans_b, sizeof(trans_b)),
                          "sb set transB")) break;

        // Dimensions: for op(A) × op(B), A is (m × k), B is (k × n), C/D is (m × n)
        uint32_t a_rows = (trans_a == HIPBLAS_OP_N) ? m : k;
        uint32_t a_cols = (trans_a == HIPBLAS_OP_N) ? k : m;
        uint32_t a_ld   = a_rows;

        uint32_t b_rows = (trans_b == HIPBLAS_OP_N) ? k : n;
        uint32_t b_cols = (trans_b == HIPBLAS_OP_N) ? n : k;
        uint32_t b_ld   = b_rows;

        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&a_desc, HIP_R_32F, a_rows, a_cols, a_ld),
                          "sb A layout")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&b_desc, HIP_R_32F, b_rows, b_cols, b_ld),
                          "sb B layout")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&c_desc, HIP_R_32F, m, n, m),
                          "sb C layout")) break;
        if (!hipblaslt_ok(hipblasLtMatrixLayoutCreate(&d_desc, HIP_R_32F, m, n, m),
                          "sb D layout")) break;

        // Set batch count and strides per-layout (hipBLASLt 7.2 API)
        (void)hipblasLtMatrixLayoutSetAttribute(a_desc, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batch_count, sizeof(batch_count));
        (void)hipblasLtMatrixLayoutSetAttribute(b_desc, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batch_count, sizeof(batch_count));
        (void)hipblasLtMatrixLayoutSetAttribute(c_desc, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batch_count, sizeof(batch_count));
        (void)hipblasLtMatrixLayoutSetAttribute(d_desc, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                 &batch_count, sizeof(batch_count));

        if (stride_a != 0)
            (void)hipblasLtMatrixLayoutSetAttribute(a_desc, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                     &stride_a, sizeof(stride_a));
        if (stride_b != 0)
            (void)hipblasLtMatrixLayoutSetAttribute(b_desc, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                     &stride_b, sizeof(stride_b));
        if (stride_c != 0)
            (void)hipblasLtMatrixLayoutSetAttribute(c_desc, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                     &stride_c, sizeof(stride_c));
        (void)hipblasLtMatrixLayoutSetAttribute(d_desc, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                 &stride_c, sizeof(stride_c));

        if (!hipblaslt_ok(hipblasLtMatmulPreferenceCreate(&pref), "sb pref create")) break;
        const size_t max_workspace = 0;
        if (!hipblaslt_ok(hipblasLtMatmulPreferenceSetAttribute(
                pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &max_workspace, sizeof(max_workspace)),
                "sb set max workspace")) break;
        if (!hipblaslt_ok(hipblasLtMatmulAlgoGetHeuristic(g_hipblaslt, desc,
                                                          a_desc, b_desc, c_desc, d_desc,
                                                          pref, 8, heur, &returned),
                          "sb algo heuristic")) break;
        if (returned <= 0 || heur[0].state != HIPBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "ds4: hipBLASLt no sb algo for %s m=%u n=%u k=%u b=%u\n",
                    label ? label : "sb_gemm", m, n, k, batch_count);
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

    cuda_hipblaslt_gemm_plan_sb_f32 p;
    p.m = m; p.n = n; p.k = k;
    p.batch_count = batch_count;
    p.stride_a = stride_a; p.stride_b = stride_b; p.stride_c = stride_c;
    p.trans_a = trans_a; p.trans_b = trans_b;
    p.desc = desc;
    p.a_desc = a_desc; p.b_desc = b_desc; p.c_desc = c_desc; p.d_desc = d_desc;
    p.algo = heur[0].algo;
    g_hipblaslt_sb_plans.push_back(p);
    return &g_hipblaslt_sb_plans.back();
}

static int hipblaslt_gemm_strided_batched_f32(
        float *out,
        const float *a,
        const float *b,
        hipblasOperation_t trans_a, hipblasOperation_t trans_b,
        uint32_t m, uint32_t n, uint32_t k,
        int64_t stride_a, int64_t stride_b, int64_t stride_c,
        uint32_t batch_count,
        float alpha, float beta,
        const char *label) {
    if (!g_hipblaslt_ready || !out || !a || !b ||
        m == 0 || n == 0 || k == 0 || batch_count == 0) return 0;
    cuda_hipblaslt_gemm_plan_sb_f32 *p = hipblaslt_gemm_plan_get_sb_f32(
            trans_a, trans_b, m, n, k, stride_a, stride_b, stride_c, batch_count, label);
    if (!p) return 0;
    return hipblaslt_ok(hipblasLtMatmul(g_hipblaslt, p->desc, &alpha,
                                        a, p->a_desc,
                                        b, p->b_desc,
                                        &beta,
                                        out, p->c_desc,
                                        out, p->d_desc,
                                        &p->algo,
                                        NULL, 0, 0),
                        label ? label : "sb_gemm");
}
