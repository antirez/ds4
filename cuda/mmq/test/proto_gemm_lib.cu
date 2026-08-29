// SPDX-License-Identifier: MIT
// proto_gemm_lib.cu - NVIDIA-library dense GEMM ceiling at the exact MoE-leg
// shapes (cuBLASLt heuristics, up to 16 algos, best-of; TN col-major):
//   down-shape:   m=4096 n=24576 k=2048  (== one W4096 down launch's total FLOP)
//   gateup-shape: m=2048 n=24576 k=4096  (== one gate or up GEMM of the pair)
// Kinds: int8->int32 (COMPUTE_32I) | f16->f16 @ f32-acc (COMPUTE_32F) |
//        f16->f16 @ f16-acc (COMPUTE_16F).
//
// This is "what NVIDIA's best dense kernel does on this silicon at our shape" —
// the practical ceiling any MoE grouped-GEMM rewrite chases (grouping + quant
// decode overhead comes on top of it).
//
// Build: /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_121 \
//        cuda/mmq/test/proto_gemm_lib.cu -o proto_gemm_lib -lcublasLt
// argv: [reps=10]

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%d: %s\n",#call,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)
#define LT(call) do { cublasStatus_t s_=(call); if(s_!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cublasLt %s @%d: status %d\n",#call,__LINE__,(int)s_); goto done; } } while(0)

namespace {

__global__ void fillk(unsigned *p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = ((unsigned)i * 2654435761u + seed) & 0x3BFF3BFFu;  // finite small f16 pairs; valid int8 bytes
}

void bench_case(cublasLtHandle_t lt, void *dA, void *dB, void *dC, void *ws,
                size_t wsSize, int m, int n, int k, int kind, int reps,
                const char *label) {
    const cudaDataType ab = (kind == 0) ? CUDA_R_8I : CUDA_R_16F;
    const cudaDataType cd = (kind == 0) ? CUDA_R_32I : CUDA_R_16F;
    const cublasComputeType_t ct = (kind == 0) ? CUBLAS_COMPUTE_32I
                                : (kind == 1) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F;
    const cudaDataType st = (kind == 0) ? CUDA_R_32I
                          : (kind == 1) ? CUDA_R_32F : CUDA_R_16F;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t La = nullptr, Lb = nullptr, Lc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t res[16];
    int nres = 0;
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    const int32_t ai = 1, bi = 0; const float af = 1.f, bf = 0.f;
    const __half ah = __float2half(1.f), bh = __float2half(0.f);
    const void *alpha = (kind == 0) ? (const void *)&ai : (kind == 1) ? (const void *)&af : (const void *)&ah;
    const void *beta  = (kind == 0) ? (const void *)&bi : (kind == 1) ? (const void *)&bf : (const void *)&bh;
    float best = 1e30f; int besti = -1;

    LT(cublasLtMatmulDescCreate(&op, ct, st));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    LT(cublasLtMatrixLayoutCreate(&La, ab, k, m, k));  // stored kxm; opA=T -> mxk
    LT(cublasLtMatrixLayoutCreate(&Lb, ab, k, n, k));
    LT(cublasLtMatrixLayoutCreate(&Lc, cd, m, n, m));
    LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &wsSize, sizeof(wsSize)));
    LT(cublasLtMatmulAlgoGetHeuristic(lt, op, La, Lb, Lc, Lc, pref, 16, res, &nres));
    if (!nres) { printf("  %-14s m=%5d n=%5d k=%5d: NOALGO\n", label, m, n, k); goto done; }

    for (int i = 0; i < nres; ++i) {
        if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
        if (cublasLtMatmul(lt, op, alpha, dA, La, dB, Lb, beta, dC, Lc, dC, Lc,
                           &res[i].algo, ws, wsSize, 0) != CUBLAS_STATUS_SUCCESS) continue;
        CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
        CHECK(cudaEventRecord(e0));
        for (int r = 0; r < reps; ++r)
            cublasLtMatmul(lt, op, alpha, dA, La, dB, Lb, beta, dC, Lc, dC, Lc,
                           &res[i].algo, ws, wsSize, 0);
        CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
        float ms = 0; CHECK(cudaEventElapsedTime(&ms, e0, e1)); ms /= (float)reps;
        if (ms < best) { best = ms; besti = i; }
        CHECK(cudaEventDestroy(e0)); CHECK(cudaEventDestroy(e1));
    }
    if (besti < 0) { printf("  %-14s m=%5d n=%5d k=%5d: ALL-ALGOS-FAILED (n=%d)\n", label, m, n, k, nres); goto done; }
    printf("  %-14s m=%5d n=%5d k=%5d: best %7.3f ms  %6.1f TFLOPS  (algo %d of %d)\n",
           label, m, n, k, best, 2.0 * m * (double)n * k / (best * 1e-3) / 1e12, besti, nres);
done:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (Lc) cublasLtMatrixLayoutDestroy(Lc);
    if (Lb) cublasLtMatrixLayoutDestroy(Lb);
    if (La) cublasLtMatrixLayoutDestroy(La);
    if (op) cublasLtMatmulDescDestroy(op);
}

} // namespace

int main(int argc, char **argv) {
    const int reps = (argc > 1) ? atoi(argv[1]) : 10;
    printf("### proto_gemm_lib  cublasLt %zu  reps=%d ###\n", cublasLtGetVersion(), reps);
    cublasLtHandle_t lt; if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "lt create fail\n"); return 1; }
    const size_t maxA = (size_t)32768 * 8192 * 2;  // covers kxm all shapes, f16
    const size_t maxB = (size_t)8192 * 24576 * 2;
    const size_t maxC = (size_t)32768 * 24576 * 4;
    const size_t wsSize = 256u << 20;
    void *dA, *dB, *dC, *ws;
    CHECK(cudaMalloc(&dA, maxA)); CHECK(cudaMalloc(&dB, maxB));
    CHECK(cudaMalloc(&dC, maxC)); CHECK(cudaMalloc(&ws, wsSize));
    fillk<<<512, 256>>>((unsigned *)dA, maxA / 4, 0xA5A5u);
    fillk<<<512, 256>>>((unsigned *)dB, maxB / 4, 0x5A5Au);
    CHECK(cudaDeviceSynchronize());

    /* MoE-equiv shapes + the dense-q8 census shapes (cmdq8shape15 2026-07-09):
     * q_b [32768x1024], o_proj [4096x8192], sexp gate/up [2048x4096],
     * sexp down [4096x2048], all at N=4096 chunk tokens. */
    const int shapes[][3] = { {4096, 24576, 2048}, {2048, 24576, 4096},
                              {32768, 4096, 1024}, {4096, 4096, 8192},
                              {2048, 4096, 4096},  {4096, 4096, 2048} };
    const char *kn[] = { "int8.s32acc", "f16.f32acc", "f16.f16acc" };
    for (const auto &s : shapes) {
        printf("--- shape m=%d n=%d k=%d ---\n", s[0], s[1], s[2]);
        for (int kind = 0; kind < 3; ++kind)
            bench_case(lt, dA, dB, dC, ws, wsSize, s[0], s[1], s[2], kind, reps, kn[kind]);
    }
    cublasLtDestroy(lt);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC)); CHECK(cudaFree(ws));
    return 0;
}
