/* HIP-only hipBLASLt state and helpers.
 * Included from ds4_cuda.cu under __HIP_PLATFORM_AMD__ to keep ROCm
 * planning/cache code out of the CUDA host runtime body. */

#include <hipblaslt/hipblaslt-ext.hpp>

static hipblasLtHandle_t g_hipblaslt;
static int g_hipblaslt_ready;

/* Below this batch the native Q8/F16 row kernels stay in use: they stream
 * the narrower weight format once per token, which is what bounds decode and
 * DSpark verification. */
#define DS4_ROCM_F16_GEMM_MIN_TOKENS 32u

/* Prefill GEMMs run through hipBLASLt.  All of them share one shape family:
 *
 *   out[n_tok][out_dim] = x[n_tok][in_dim] . w[out_dim][in_dim]^T
 *
 * which in hipBLASLt's column-major view is C(out_dim x n_tok) = A^T . B with
 * A = w (lda in_dim) and B = x (ldb in_dim).  An optional strided batch covers
 * the grouped attention output projection, whose eight groups write
 * interleaved column blocks of one row-major [n_tok][out_ld] matrix.
 *
 * The heuristic order hipBLASLt returns is not reliable on gfx1151: for F32
 * outputs its first choice runs four times slower than the best of the first
 * eight.  Each distinct shape therefore times the candidates once, on the
 * first call, and remembers the winner for the shape class.  n_tok is
 * bucketed by power of two so ragged chunk tails reuse the measured choice
 * instead of paying a new sweep per prompt length. */
struct cuda_hipblaslt_gemm_shape {
    uint32_t out_dim, n_tok, in_dim, batch, out_ld;
    int64_t w_stride, x_stride, out_stride;
    hipDataType out_type;
    bool operator==(const cuda_hipblaslt_gemm_shape &o) const {
        return out_dim == o.out_dim && n_tok == o.n_tok && in_dim == o.in_dim &&
               batch == o.batch && out_ld == o.out_ld && w_stride == o.w_stride &&
               x_stride == o.x_stride && out_stride == o.out_stride &&
               out_type == o.out_type;
    }
};

struct cuda_hipblaslt_gemm_plan {
    cuda_hipblaslt_gemm_shape shape;
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t a_desc, b_desc, c_desc;
    hipblasLtMatmulAlgo_t algo;
};

struct cuda_hipblaslt_algo_choice {
    cuda_hipblaslt_gemm_shape shape; /* n_tok holds the bucket */
    hipblasLtMatmulAlgo_t algo;
};

static std::vector<cuda_hipblaslt_gemm_plan> g_hipblaslt_gemm_plans;
static std::vector<cuda_hipblaslt_algo_choice> g_hipblaslt_algo_choices;

static void hipblaslt_gemm_plan_destroy(cuda_hipblaslt_gemm_plan &p) {
    if (p.c_desc) (void)hipblasLtMatrixLayoutDestroy(p.c_desc);
    if (p.b_desc) (void)hipblasLtMatrixLayoutDestroy(p.b_desc);
    if (p.a_desc) (void)hipblasLtMatrixLayoutDestroy(p.a_desc);
    if (p.desc) (void)hipblasLtMatmulDescDestroy(p.desc);
    p.desc = NULL;
    p.a_desc = p.b_desc = p.c_desc = NULL;
}

static void hipblaslt_gemm_plan_clear(void) {
    for (cuda_hipblaslt_gemm_plan &p : g_hipblaslt_gemm_plans) hipblaslt_gemm_plan_destroy(p);
    g_hipblaslt_gemm_plans.clear();
    g_hipblaslt_algo_choices.clear();
}

static int hipblaslt_ok(hipblasStatus_t st, const char *what) {
    if (st == HIPBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: hipBLASLt %s failed: status %d\n", what, (int)st);
    return 0;
}

static uint32_t hipblaslt_n_tok_bucket(uint32_t n_tok) {
    uint32_t b = 32u;
    while (b < n_tok) b <<= 1u;
    return b;
}

static int hipblaslt_layout_batched(hipblasLtMatrixLayout_t layout, uint32_t batch, int64_t stride) {
    if (batch <= 1u) return 1;
    const int32_t count = (int32_t)batch;
    return hipblaslt_ok(hipblasLtMatrixLayoutSetAttribute(layout,
                            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)),
                        "batch count") &&
           hipblaslt_ok(hipblasLtMatrixLayoutSetAttribute(layout,
                            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)),
                        "batch stride");
}

static int hipblaslt_gemm_plan_create(cuda_hipblaslt_gemm_plan &p, const cuda_hipblaslt_gemm_shape &s) {
    p.shape = s;
    p.desc = NULL;
    p.a_desc = p.b_desc = p.c_desc = NULL;
    const hipblasOperation_t op_a = HIPBLAS_OP_T, op_b = HIPBLAS_OP_N;
    const int ok =
        hipblaslt_ok(hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), "matmul desc") &&
        hipblaslt_ok(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                     &op_a, sizeof(op_a)), "transA") &&
        hipblaslt_ok(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                     &op_b, sizeof(op_b)), "transB") &&
        hipblaslt_ok(hipblasLtMatrixLayoutCreate(&p.a_desc, HIP_R_16F, s.in_dim, s.out_dim, s.in_dim),
                     "A layout") &&
        hipblaslt_ok(hipblasLtMatrixLayoutCreate(&p.b_desc, HIP_R_16F, s.in_dim, s.n_tok, s.in_dim),
                     "B layout") &&
        hipblaslt_ok(hipblasLtMatrixLayoutCreate(&p.c_desc, s.out_type, s.out_dim, s.n_tok, s.out_ld),
                     "C layout") &&
        hipblaslt_layout_batched(p.a_desc, s.batch, s.w_stride) &&
        hipblaslt_layout_batched(p.b_desc, s.batch, s.x_stride) &&
        hipblaslt_layout_batched(p.c_desc, s.batch, s.out_stride);
    if (!ok) hipblaslt_gemm_plan_destroy(p);
    return ok;
}

static hipblasStatus_t hipblaslt_gemm_plan_run(const cuda_hipblaslt_gemm_plan &p,
                                               const hipblasLtMatmulAlgo_t *algo,
                                               void *out, const __half *w, const __half *x) {
    const float alpha = 1.0f, beta = 0.0f;
    return hipblasLtMatmul(g_hipblaslt, p.desc, &alpha, w, p.a_desc, x, p.b_desc,
                           &beta, out, p.c_desc, out, p.c_desc, algo, NULL, 0, 0);
}

/* Times every heuristic candidate on the caller's buffers and keeps the
 * fastest.  Each candidate is timed three times and judged by its best run,
 * and a later candidate replaces the leader only when it is clearly faster,
 * so first-launch code loading and clock jitter rarely change which
 * algorithm (and therefore which summation order) a process ends up with.
 * The output is rewritten by every run, so the caller still has to run the
 * winner. */
static int hipblaslt_gemm_choose_algo(const cuda_hipblaslt_gemm_plan &p,
                                      void *out, const __half *w, const __half *x,
                                      hipblasLtMatmulAlgo_t *best, const char *label) {
    hipblasLtMatmulPreference_t pref = NULL;
    const size_t max_workspace = 0;
    hipblasLtMatmulHeuristicResult_t heur[8];
    int returned = 0;
    if (!hipblaslt_ok(hipblasLtMatmulPreferenceCreate(&pref), "preference create")) return 0;
    const int heur_ok =
        hipblaslt_ok(hipblasLtMatmulPreferenceSetAttribute(pref,
                         HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                         &max_workspace, sizeof(max_workspace)), "max workspace") &&
        hipblaslt_ok(hipblasLtMatmulAlgoGetHeuristic(g_hipblaslt, p.desc, p.a_desc, p.b_desc,
                                                     p.c_desc, p.c_desc, pref,
                                                     8, heur, &returned), "algo heuristic");
    (void)hipblasLtMatmulPreferenceDestroy(pref);
    if (!heur_ok) return 0;
    if (returned <= 0) {
        fprintf(stderr, "ds4: hipBLASLt no algo for %s m=%u n=%u k=%u\n",
                label, p.shape.out_dim, p.shape.n_tok, p.shape.in_dim);
        return 0;
    }
    hipEvent_t e0, e1;
    if (hipEventCreate(&e0) != hipSuccess) return 0;
    if (hipEventCreate(&e1) != hipSuccess) {
        (void)hipEventDestroy(e0);
        return 0;
    }
    float best_ms = 0.0f;
    int found = 0;
    for (int i = 0; i < returned; i++) {
        if (heur[i].state != HIPBLAS_STATUS_SUCCESS) continue;
        float ms = 0.0f;
        int timed = 0;
        for (int run = 0; run < 3; run++) {
            float run_ms = 0.0f;
            if (hipEventRecord(e0, 0) != hipSuccess ||
                hipblaslt_gemm_plan_run(p, &heur[i].algo, out, w, x) != HIPBLAS_STATUS_SUCCESS ||
                hipEventRecord(e1, 0) != hipSuccess ||
                hipEventSynchronize(e1) != hipSuccess ||
                hipEventElapsedTime(&run_ms, e0, e1) != hipSuccess) {
                (void)hipGetLastError();
                break;
            }
            if (!timed || run_ms < ms) ms = run_ms;
            timed = 1;
        }
        if (timed && (!found || ms < best_ms * 0.95f)) {
            best_ms = ms;
            *best = heur[i].algo;
            found = 1;
        }
    }
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
    return found;
}

static cuda_hipblaslt_gemm_plan *hipblaslt_gemm_plan_get(const cuda_hipblaslt_gemm_shape &s,
                                                         void *out, const __half *w, const __half *x,
                                                         const char *label) {
    for (cuda_hipblaslt_gemm_plan &p : g_hipblaslt_gemm_plans) {
        if (p.shape == s) return &p;
    }
    /* Plans are per exact token count, so prompt tails make the set grow
     * with every new length in a long-running server.  Past a bound, drop
     * the plans: the measured algorithm choices below are kept, so a plan
     * is rebuilt without a new timing sweep. */
    if (g_hipblaslt_gemm_plans.size() >= 256u) {
        for (cuda_hipblaslt_gemm_plan &old : g_hipblaslt_gemm_plans) hipblaslt_gemm_plan_destroy(old);
        g_hipblaslt_gemm_plans.clear();
    }
    cuda_hipblaslt_gemm_plan p;
    if (!hipblaslt_gemm_plan_create(p, s)) return NULL;

    /* The batched B operand is packed [batch][n_tok][in_dim], so its stride
     * scales with the token count and follows the bucket too. */
    cuda_hipblaslt_gemm_shape cls = s;
    cls.n_tok = hipblaslt_n_tok_bucket(s.n_tok);
    cls.x_stride = s.x_stride / s.n_tok * cls.n_tok;
    cuda_hipblaslt_algo_choice *choice = NULL;
    for (cuda_hipblaslt_algo_choice &c : g_hipblaslt_algo_choices) {
        if (c.shape == cls) choice = &c;
    }
    size_t workspace = 0;
    const float alpha = 1.0f, beta = 0.0f;
    if (choice &&
        hipblaslt_ext::matmulIsAlgoSupported(g_hipblaslt, p.desc, &alpha, p.a_desc, p.b_desc,
                                             &beta, p.c_desc, p.c_desc, choice->algo,
                                             workspace) == HIPBLAS_STATUS_SUCCESS &&
        workspace == 0) {
        p.algo = choice->algo;
    } else if (hipblaslt_gemm_choose_algo(p, out, w, x, &p.algo, label)) {
        if (choice) {
            choice->algo = p.algo;
        } else {
            g_hipblaslt_algo_choices.push_back({cls, p.algo});
        }
    } else {
        hipblaslt_gemm_plan_destroy(p);
        return NULL;
    }
    g_hipblaslt_gemm_plans.push_back(p);
    return &g_hipblaslt_gemm_plans.back();
}

static int hipblaslt_gemm_batched(void *out, hipDataType out_type, uint32_t out_ld,
                                  const __half *w, const __half *x,
                                  uint32_t out_dim, uint32_t n_tok, uint32_t in_dim,
                                  uint32_t batch, int64_t w_stride, int64_t x_stride,
                                  int64_t out_stride, const char *label) {
    if (!g_hipblaslt_ready || !out || !w || !x ||
        out_dim == 0 || n_tok == 0 || in_dim == 0 || batch == 0) return 0;
    const cuda_hipblaslt_gemm_shape s = {out_dim, n_tok, in_dim, batch, out_ld,
                                         w_stride, x_stride, out_stride, out_type};
    cuda_hipblaslt_gemm_plan *p = hipblaslt_gemm_plan_get(s, out, w, x, label);
    if (!p) return 0;
    return hipblaslt_ok(hipblaslt_gemm_plan_run(*p, &p->algo, out, w, x), label);
}

static int hipblaslt_gemm_f32(float *out, const __half *w, const __half *x,
                              uint32_t out_dim, uint32_t n_tok, uint32_t in_dim, const char *label) {
    return hipblaslt_gemm_batched(out, HIP_R_32F, out_dim, w, x, out_dim, n_tok, in_dim,
                                  1u, 0, 0, 0, label);
}

static int hipblaslt_gemm_f16(__half *out, const __half *w, const __half *x,
                              uint32_t out_dim, uint32_t n_tok, uint32_t in_dim, const char *label) {
    return hipblaslt_gemm_batched(out, HIP_R_16F, out_dim, w, x, out_dim, n_tok, in_dim,
                                  1u, 0, 0, 0, label);
}

/* Experimental DSpark attention A: five fixed-width plans, independent of
 * the prefill cache and its failure state. C/D contain eight interleaved groups. */
static int g_hipblaslt_dspark_a_state;
static cuda_hipblaslt_gemm_plan g_hipblaslt_dspark_a_plans[5];

static void hipblaslt_dspark_a_clear(void) {
    for (auto &p : g_hipblaslt_dspark_a_plans) hipblaslt_gemm_plan_destroy(p);
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
                              "DSpark A C")) break;
            const int32_t groups = 8;
            const int64_t strides[] = {INT64_C(4194304), INT64_C(4096) * n_tok, 1024};
            const hipblasLtMatrixLayout_t layouts[] = {p.a_desc, p.b_desc, p.c_desc};
            bool layouts_ok = true;
            for (unsigned i = 0; i < 3u && layouts_ok; i++) {
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
                    &alpha, p.a_desc, p.b_desc, &beta, p.c_desc, p.c_desc,
                    algorithms[0].algo, workspace), "DSpark A support") || workspace != 0) break;
            p.algo = algorithms[0].algo;
            ok = 1;
        } while (0);
        if (!ok) {
            hipblaslt_gemm_plan_destroy(p);
            __atomic_store_n(&g_hipblaslt_dspark_a_state, -1, __ATOMIC_RELAXED);
            return 0;
        }
    }
    hipStream_t stream = nullptr;
    if (hipblaslt_ok(hipblasGetStream(g_cublas, &stream), "DSpark A stream") &&
        hipblaslt_ok(hipblasLtMatmul(g_hipblaslt, p.desc, &alpha,
                w, p.a_desc, x, p.b_desc, &beta, out, p.c_desc, out, p.c_desc,
                &p.algo, nullptr, 0, stream), "DSpark A matmul")) return 1;
    /* A synchronous library rejection falls back on the same stream and
     * overwrites every output. An asynchronous GPU fault is not recoverable. */
    __atomic_store_n(&g_hipblaslt_dspark_a_state, -1, __ATOMIC_RELAXED);
    return 0;
}
