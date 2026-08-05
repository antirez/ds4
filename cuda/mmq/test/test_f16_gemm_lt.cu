/* P3.3 proto (2026-07-07, CLOSED "cuBLAS knows best"): are the cutlass_80_*
 * (Ampere-tile) f16 GEMMs on sm_121 leaving throughput on the table?  NO:
 * - The 6.25 ms cutlass_80_64x256 launch (129x/12k admission) is the
 *   ATTENTION OUTPUT projection (M=8192=8 groups x 1024 rank, K=4096,
 *   N=4096 -> 47 TFLOPS, near roof).  The "11 TFLOPS" that chartered this
 *   recon assumed it was the indexer q_b (K=1024) -- wrong dims.  The real
 *   indexer q_b GEMM is producer-gated (n_comp > top-k; last chunk only at
 *   12k) and runs cutlass_80_128x256 at 1.3 ms / 53 TFLOPS.
 * - cublasLt heuristics beat GemmEx by only ~6-10% on these shapes (not
 *   worth the plumbing); TF32 math mode, operand alignment (A/B/C +16/+32B),
 *   L2 warmth (8-set rotation) and host-vs-device A were all bracketed:
 *   only host-mapped A costs real time (2x), and production weights already
 *   resolve to device-resident copies (HBM cache / WS import) -- verified
 *   in-server via DS4_F16_PTR_PROBE (recon-only, not in tree).
 * - hc mix (m=24) is latency-floor at any tile; comp_proj m=256 is fine.
 *
 * Shapes measured (production call form: A=weights f16 OP_T lda=k, B=acts
 * f16 OP_N ldb=k, C f32 ldc=m, computeType F32, 32 MiB handle workspace):
 *
 * Build (on a GB10 box):
 *   /usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -std=c++17 \
 *     -o /home/ent/dspark_work/test_f16_gemm_lt cuda/mmq/test/test_f16_gemm_lt.cu \
 *     -lcublasLt -lcublas
 * Run:  ./test_f16_gemm_lt            (all shapes, 100 iters each)
 * Kernel names: nsys profile -o /tmp/lt ./test_f16_gemm_lt && inspect sqlite.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#define CK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(_e)); exit(1); } } while (0)
#define CB(x) do { auto _e = (x); if (_e != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "CUBLAS ERR %s @%d: %d\n", #x, __LINE__, (int)_e); exit(1); } } while (0)

static void fill_half(__half *p, size_t nelem, unsigned seed) {
    __half *h = (__half *)malloc(nelem * sizeof(__half));
    unsigned s = seed;
    for (size_t i = 0; i < nelem; i++) {
        s = s * 1664525u + 1013904223u;
        h[i] = __float2half(((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 0.25f);
    }
    CK(cudaMemcpy(p, h, nelem * sizeof(__half), cudaMemcpyHostToDevice));
    free(h);
}

static float time_iters(cudaStream_t st, int iters, void (*fn)(void *), void *ud) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int i = 0; i < 10; i++) fn(ud);       /* warmup */
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(e0, st));
    for (int i = 0; i < iters; i++) fn(ud);
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / iters;
}

struct GemmCase {
    int m, n, k;
    const __half *A, *B;
    float *C;
    cublasHandle_t bl;
    cublasLtHandle_t lt;
    void *ws;
    size_t ws_bytes;
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulAlgo_t algo;
    int has_algo;
};

static void run_gemmex(void *ud) {
    GemmCase *g = (GemmCase *)ud;
    const float alpha = 1.f, beta = 0.f;
    CB(cublasGemmEx(g->bl, CUBLAS_OP_T, CUBLAS_OP_N, g->m, g->n, g->k,
                    &alpha, g->A, CUDA_R_16F, g->k, g->B, CUDA_R_16F, g->k,
                    &beta, g->C, CUDA_R_32F, g->m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

static void run_lt(void *ud) {
    GemmCase *g = (GemmCase *)ud;
    const float alpha = 1.f, beta = 0.f;
    CB(cublasLtMatmul(g->lt, g->op, &alpha, g->A, g->la, g->B, g->lb,
                      &beta, g->C, g->lc, g->C, g->lc,
                      g->has_algo ? &g->algo : NULL, g->ws, g->ws_bytes, 0));
}

int main(void) {
    struct { const char *name; int m, n, k; } shapes[] = {
        { "indexer_q_b m=8192 n=4096 k=1024", 8192, 4096, 1024 },
        { "hc_mix      m=24   n=4096 k=16384", 24, 4096, 16384 },
        { "comp_proj   m=256  n=4096 k=4096", 256, 4096, 4096 },
    };
    cublasHandle_t bl;
    CB(cublasCreate(&bl));
    void *hws = NULL;
    const size_t hws_bytes = 32u * 1024u * 1024u;
    CK(cudaMalloc(&hws, hws_bytes));
    CB(cublasSetWorkspace(bl, hws, hws_bytes));
    cublasLtHandle_t lt;
    CB(cublasLtCreate(&lt));
    void *ltws = NULL;
    const size_t ltws_bytes = 32u * 1024u * 1024u;
    CK(cudaMalloc(&ltws, ltws_bytes));

    for (auto &s : shapes) {
        __half *A, *B;
        float *C;
        CK(cudaMalloc(&A, (size_t)s.m * s.k * sizeof(__half)));
        CK(cudaMalloc(&B, (size_t)s.n * s.k * sizeof(__half)));
        CK(cudaMalloc(&C, (size_t)s.m * s.n * sizeof(float)));
        fill_half(A, (size_t)s.m * s.k, 1234);
        fill_half(B, (size_t)s.n * s.k, 5678);

        GemmCase g = {};
        g.m = s.m; g.n = s.n; g.k = s.k;
        g.A = A; g.B = B; g.C = C;
        g.bl = bl; g.lt = lt; g.ws = ltws; g.ws_bytes = ltws_bytes;

        const double flops = 2.0 * s.m * s.n * s.k;
        const float ms_ex = time_iters(0, 100, run_gemmex, &g);
        printf("%-36s GemmEx DEFAULT      : %8.3f ms  %6.1f TFLOPS\n",
               s.name, ms_ex, flops / (ms_ex * 1e9));
        /* Production handle state (ds4_gpu_init): deprecated TF32 math mode.
         * On CUDA 13 this may shunt the heuristic onto a legacy kernel table
         * -- the suspected source of the 64x256 cutlass_80 pick. */
        CB(cublasSetMathMode(bl, CUBLAS_TF32_TENSOR_OP_MATH));
        const float ms_tf = time_iters(0, 100, run_gemmex, &g);
        printf("%-36s GemmEx TF32-mode    : %8.3f ms  %6.1f TFLOPS\n",
               s.name, ms_tf, flops / (ms_tf * 1e9));
        CB(cublasSetMathMode(bl, CUBLAS_DEFAULT_MATH));
        /* Production pointer classes: A = gguf tensor offset inside a mapped
         * model (32B alignment class, not cudaMalloc's 256B); B = cuda_tmp
         * suballocation.  Heuristics gate kernels on operand alignment. */
        {
            __half *A32, *B32;
            CK(cudaMalloc(&A32, (size_t)s.m * s.k * sizeof(__half) + 256));
            CK(cudaMalloc(&B32, (size_t)s.n * s.k * sizeof(__half) + 256));
            CK(cudaMemcpy((char *)A32 + 32, A, (size_t)s.m * s.k * sizeof(__half), cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy((char *)B32 + 32, B, (size_t)s.n * s.k * sizeof(__half), cudaMemcpyDeviceToDevice));
            GemmCase g2 = g;
            g2.A = (const __half *)((const char *)A32 + 32);
            const float ms_a = time_iters(0, 100, run_gemmex, &g2);
            printf("%-36s GemmEx A+32B        : %8.3f ms  %6.1f TFLOPS\n",
                   s.name, ms_a, flops / (ms_a * 1e9));
            GemmCase g3 = g;
            g3.B = (const __half *)((const char *)B32 + 32);
            const float ms_b = time_iters(0, 100, run_gemmex, &g3);
            printf("%-36s GemmEx B+32B        : %8.3f ms  %6.1f TFLOPS\n",
                   s.name, ms_b, flops / (ms_b * 1e9));
            GemmCase g4 = g2;
            g4.B = g3.B;
            const float ms_ab = time_iters(0, 100, run_gemmex, &g4);
            printf("%-36s GemmEx A+32B B+32B  : %8.3f ms  %6.1f TFLOPS\n",
                   s.name, ms_ab, flops / (ms_ab * 1e9));
            float *C32;
            CK(cudaMalloc(&C32, (size_t)s.m * s.n * sizeof(float) + 256));
            GemmCase g5 = g;
            g5.C = (float *)((char *)C32 + 32);
            const float ms_c = time_iters(0, 100, run_gemmex, &g5);
            printf("%-36s GemmEx C+32B        : %8.3f ms  %6.1f TFLOPS\n",
                   s.name, ms_c, flops / (ms_c * 1e9));
            GemmCase g6 = g;
            g6.C = (float *)((char *)C32 + 16);
            const float ms_c16 = time_iters(0, 100, run_gemmex, &g6);
            printf("%-36s GemmEx C+16B        : %8.3f ms  %6.1f TFLOPS\n",
                   s.name, ms_c16, flops / (ms_c16 * 1e9));
            CK(cudaFree(C32));
            CK(cudaFree(A32));
            CK(cudaFree(B32));
        }

        /* Lt setup: same TN f16/f16/f32-compute layout */
        CB(cublasLtMatmulDescCreate(&g.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CB(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CB(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        CB(cublasLtMatrixLayoutCreate(&g.la, CUDA_R_16F, s.k, s.m, s.k)); /* A: k x m, lda=k (OP_T) */
        CB(cublasLtMatrixLayoutCreate(&g.lb, CUDA_R_16F, s.k, s.n, s.k));
        CB(cublasLtMatrixLayoutCreate(&g.lc, CUDA_R_32F, s.m, s.n, s.m));

        /* Substrate x L2-warmth matrix.  Production weight pointers are
         * host-mapped (client mmap-registered / WS IPC import -- cmtp3d
         * measured both at the same speed); the proto default is cudaMalloc
         * A reused every iteration (L2-warm).  8-set rotation defeats L2
         * reuse; host-alloc A reproduces the mapped-weight class. */
        {
            enum { NSETS = 8 };
            static __half *As[NSETS], *Bs[NSETS];
            static float *Cs[NSETS];
            static GemmCase gsets;
            static int set_idx;
            struct Rot {
                static void run(void *ud) {
                    GemmCase *g = (GemmCase *)ud;
                    (void)g;
                    GemmCase tmp = gsets;
                    tmp.A = As[set_idx % NSETS];
                    tmp.B = Bs[set_idx % NSETS];
                    tmp.C = Cs[set_idx % NSETS];
                    set_idx++;
                    run_gemmex(&tmp);
                }
            };
            gsets = g;
            /* dev-cold */
            for (int i = 0; i < NSETS; i++) {
                CK(cudaMalloc(&As[i], (size_t)s.m * s.k * sizeof(__half)));
                CK(cudaMalloc(&Bs[i], (size_t)s.n * s.k * sizeof(__half)));
                CK(cudaMalloc(&Cs[i], (size_t)s.m * s.n * sizeof(float)));
                fill_half(As[i], (size_t)s.m * s.k, 100 + i);
                fill_half(Bs[i], (size_t)s.n * s.k, 200 + i);
            }
            set_idx = 0;
            const float ms_dc = time_iters(0, 100, Rot::run, &g);
            printf("%-36s GemmEx devA cold    : %8.3f ms  %6.1f TFLOPS\n",
                   s.name, ms_dc, flops / (ms_dc * 1e9));
            for (int i = 0; i < NSETS; i++) { CK(cudaFree(As[i])); CK(cudaFree(Bs[i])); CK(cudaFree(Cs[i])); }
            /* hostA warm */
            __half *Ah;
            CK(cudaHostAlloc(&Ah, (size_t)s.m * s.k * sizeof(__half), cudaHostAllocMapped));
            {
                unsigned sd = 42;
                for (size_t i = 0; i < (size_t)s.m * s.k; i++) {
                    sd = sd * 1664525u + 1013904223u;
                    Ah[i] = __float2half(((float)(sd >> 8) / (float)(1u << 24) - 0.5f) * 0.25f);
                }
            }
            GemmCase gh = g;
            gh.A = Ah;
            const float ms_hw = time_iters(0, 100, run_gemmex, &gh);
            printf("%-36s GemmEx hostA warm   : %8.3f ms  %6.1f TFLOPS\n",
                   s.name, ms_hw, flops / (ms_hw * 1e9));
            /* hostA cold: rotate device B/C, host A stays (weights are the
             * suspect; hostA likely bypasses L2 anyway) */
            for (int i = 0; i < NSETS; i++) {
                CK(cudaMalloc(&Bs[i], (size_t)s.n * s.k * sizeof(__half)));
                CK(cudaMalloc(&Cs[i], (size_t)s.m * s.n * sizeof(float)));
                fill_half(Bs[i], (size_t)s.n * s.k, 300 + i);
                As[i] = Ah;
            }
            gsets = gh;
            set_idx = 0;
            const float ms_hc = time_iters(0, 100, Rot::run, &g);
            printf("%-36s GemmEx hostA cold   : %8.3f ms  %6.1f TFLOPS\n",
                   s.name, ms_hc, flops / (ms_hc * 1e9));
            for (int i = 0; i < NSETS; i++) { CK(cudaFree(Bs[i])); CK(cudaFree(Cs[i])); }
            CK(cudaFreeHost(Ah));
        }
        cublasLtMatmulPreference_t pref;
        CB(cublasLtMatmulPreferenceCreate(&pref));
        CB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &ltws_bytes, sizeof(ltws_bytes)));
        cublasLtMatmulHeuristicResult_t res[8];
        int nres = 0;
        CB(cublasLtMatmulAlgoGetHeuristic(lt, g.op, g.la, g.lb, g.lc, g.lc, pref, 8, res, &nres));
        printf("%-36s Lt heuristics: %d candidates\n", s.name, nres);
        float best_ms = 1e30f;
        int best_i = -1;
        for (int i = 0; i < nres; i++) {
            g.algo = res[i].algo;
            g.has_algo = 1;
            const float ms = time_iters(0, 100, run_lt, &g);
            int algo_id = -1, tile = -1, splitk = -1;
            size_t sz;
            cublasLtMatmulAlgoConfigGetAttribute(&g.algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&g.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &sz);
            cublasLtMatmulAlgoConfigGetAttribute(&g.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk), &sz);
            printf("%-36s   Lt[%d] algo=%d tile=%d splitk=%d ws=%zu: %8.3f ms  %6.1f TFLOPS\n",
                   s.name, i, algo_id, tile, splitk, res[i].workspaceSize,
                   ms, flops / (ms * 1e9));
            if (ms < best_ms) { best_ms = ms; best_i = i; }
        }
        if (best_i >= 0)
            printf("%-36s Lt BEST[%d]          : %8.3f ms  %6.1f TFLOPS  (GemmEx/Lt = %.2fx)\n",
                   s.name, best_i, best_ms, flops / (best_ms * 1e9), ms_ex / best_ms);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(g.la);
        cublasLtMatrixLayoutDestroy(g.lb);
        cublasLtMatrixLayoutDestroy(g.lc);
        cublasLtMatmulDescDestroy(g.op);
        CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C));
    }
    printf("LT-GEMM DONE\n");
    return 0;
}
