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
