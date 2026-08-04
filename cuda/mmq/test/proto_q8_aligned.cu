// proto_q8_aligned.cu — M1-Inc3 prototype (megakernel program).
//
// HYPOTHESIS: the dense Q8_0 decode GEMVs run below the ~200-202 GB/s kernel
// ceiling (attrib15b: attn_q_b 1024x32768 @164, 2048x4096 @157, head
// 4096x129280 @146) because block_q8_0 is 34 bytes, so the int8 code stream
// is only 2-byte aligned (same misalignment class as IQ2_XXS, proven +12%
// by proto_iq2_aligned).  Repacking into an aligned SoA layout (d[] halves
// separate, qs[] 64B-aligned 32B per block) allows int4 (16B) aligned loads
// at identical byte count.
//
// A/B at the three production decode shapes (K = in_dim, M = out rows):
//   attn_q_b : K=1024,  M=32768   (34.0 MiB/call, 164 GB/s baseline)
//   mid      : K=2048,  M=4096    ( 8.5 MiB/call, 157 GB/s baseline)
//   head     : K=4096,  M=129280  (536.6 MiB/call, 146 GB/s baseline)
//   baseline = ds4_mmq_q8_0_dense_vec (production decode entry, quantizes X
//              internally; bare-kernel aligned numbers exclude that ~small
//              cost — Inc1 lesson: judge the KERNEL here, entry design later)
//
// Build (box):
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 -I.. \
//        proto_q8_aligned.cu ../*.o -lcudart -lcuda -o proto_q8_aligned

#include "ds4_mmq.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

namespace {

constexpr int QK = 32;               // values per Q8_0 block
constexpr int BLOCK_BYTES = 34;      // sizeof(block_q8_0) = 2 + 32

// ---------------------------------------------------------------- host q8_1
struct q8_1_block { __half2 ds; int8_t qs[32]; };   // mirrors block_q8_1

static void quantize_q8_1_host(const float *x, q8_1_block *out, int k) {
    for (int b = 0; b < k / 32; b++) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        int sum = 0;
        for (int i = 0; i < 32; i++) {
            int q = (int)roundf(x[b * 32 + i] * id);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            out[b].qs[i] = (int8_t)q;
            sum += q;
        }
        out[b].ds = __floats2half2_rn(d, d * (float)sum);
    }
}

// ------------------------------------------------------------ device kernel
// Aligned SoA weights: dq[blk] __half scale; qs stream 64B-aligned, 32B per
// block (int4 x2).  Warp-per-row GEMV: lane covers one block per pass.
__global__ void q8_aligned_dense_vec_kernel(
        float             *out,        // [M]
        const int4        *qs,         // aligned codes, 2 int4 per block
        const __half      *dq,         // block scales
        const q8_1_block  *x8,         // [K/32] quantized activation
        int                M,
        int                nb)         // blocks per row = K/32
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const long long rbase = (long long)row * nb;

    float acc = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        const int b = b0 + lane;
        const int4 w0 = qs[(rbase + b) * 2 + 0];   // aligned 16B
        const int4 w1 = qs[(rbase + b) * 2 + 1];
        const int *u = (const int *)x8[b].qs;
        int sumi = 0;
        sumi = __dp4a(w0.x, u[0], sumi);
        sumi = __dp4a(w0.y, u[1], sumi);
        sumi = __dp4a(w0.z, u[2], sumi);
        sumi = __dp4a(w0.w, u[3], sumi);
        sumi = __dp4a(w1.x, u[4], sumi);
        sumi = __dp4a(w1.y, u[5], sumi);
        sumi = __dp4a(w1.z, u[6], sumi);
        sumi = __dp4a(w1.w, u[7], sumi);
        acc += __half2float(dq[rbase + b]) * __low2float(x8[b].ds) * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

// Variant R2: two rows per 64-thread block (one warp each) — probes whether
// higher occupancy per SM changes the DRAM rate at skinny K.
__global__ void q8_aligned_dense_vec_r2_kernel(
        float             *out,
        const int4        *qs,
        const __half      *dq,
        const q8_1_block  *x8,
        int                M,
        int                nb)
{
    const int row  = blockIdx.x * 2 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= M) return;
    const long long rbase = (long long)row * nb;
    float acc = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        const int b = b0 + lane;
        const int4 w0 = qs[(rbase + b) * 2 + 0];
        const int4 w1 = qs[(rbase + b) * 2 + 1];
        const int *u = (const int *)x8[b].qs;
        int sumi = 0;
        sumi = __dp4a(w0.x, u[0], sumi);
        sumi = __dp4a(w0.y, u[1], sumi);
        sumi = __dp4a(w0.z, u[2], sumi);
        sumi = __dp4a(w0.w, u[3], sumi);
        sumi = __dp4a(w1.x, u[4], sumi);
        sumi = __dp4a(w1.y, u[5], sumi);
        sumi = __dp4a(w1.z, u[6], sumi);
        sumi = __dp4a(w1.w, u[7], sumi);
        acc += __half2float(dq[rbase + b]) * __low2float(x8[b].ds) * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

static void run_shape(const char *label, int M, int K, int iters, uint32_t seed) {
    const int nb = K / QK;
    const long long row_bytes = (long long)nb * BLOCK_BYTES;
    const double call_gb = (double)M * row_bytes / 1e9;

    std::mt19937 rng(seed);

    // Random bytes are valid Q8_0 content; keep block scales sane.
    std::vector<uint8_t> W((size_t)M * row_bytes);
    for (auto &b : W) b = (uint8_t)(rng() & 0xff);
    const long long nblk = (long long)M * nb;
    for (long long blk = 0; blk < nblk; blk++) {
        uint16_t h = (uint16_t)(0x2c00 | (rng() & 0xff));
        memcpy(&W[blk * BLOCK_BYTES], &h, 2);
    }

    std::vector<float> X(K);
    for (auto &v : X) v = ((float)(rng() % 2000) - 1000.0f) / 500.0f;
    std::vector<q8_1_block> X8(K / 32);
    quantize_q8_1_host(X.data(), X8.data(), K);

    // Aligned SoA repack: [half dq[nblk]][pad to 64B][int8 qs[nblk*32]]
    const uint64_t dq_bytes = ((uint64_t)nblk * 2u + 63u) & ~63ull;
    std::vector<uint8_t> ART(dq_bytes + (uint64_t)nblk * 32u);
    for (long long blk = 0; blk < nblk; blk++) {
        memcpy(&ART[blk * 2], &W[blk * BLOCK_BYTES], 2);
        memcpy(&ART[dq_bytes + (uint64_t)blk * 32u], &W[blk * BLOCK_BYTES + 2], 32);
    }

    // L2 trap (proven on the iq2 pair test AND the first run of this proto:
    // "mid" printed 931 GB/s, 4x the DRAM ceiling): a weight buffer smaller
    // than ~L2 stays resident across timing iters.  Rotate through enough
    // copies that consecutive iters never re-hit cache (~256 MB footprint).
    const int n_copies = (int)((256ull * 1024 * 1024) / (W.size() + 1)) + 1;
    uint8_t *dW, *dArt; float *dX, *dOutBase, *dOutAl; q8_1_block *dX8;
    CK(cudaMalloc(&dW, W.size() * n_copies));
    CK(cudaMalloc(&dArt, ART.size() * n_copies));
    CK(cudaMalloc(&dX, sizeof(float) * K));
    CK(cudaMalloc(&dOutBase, sizeof(float) * M));
    CK(cudaMalloc(&dOutAl, sizeof(float) * M));
    CK(cudaMalloc(&dX8, X8.size() * sizeof(q8_1_block)));
    for (int c = 0; c < n_copies; c++) {
        CK(cudaMemcpy(dW + (size_t)c * W.size(), W.data(), W.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dArt + (size_t)c * ART.size(), ART.data(), ART.size(), cudaMemcpyHostToDevice));
    }
    CK(cudaMemcpy(dX, X.data(), sizeof(float) * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX8, X8.data(), X8.size() * sizeof(q8_1_block), cudaMemcpyHostToDevice));

    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // ---- baseline: production dense vec entry ------------------------------
    int rc = ds4_mmq_q8_0_dense_vec(dW, dX, dOutBase, M, 1, K, stream);
    if (rc != 0) { printf("%s: baseline dense_vec rc=%d\n", label, rc); exit(1); }
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
        (void)ds4_mmq_q8_0_dense_vec(dW + (size_t)(i % n_copies) * W.size(),
                                     dX, dOutBase, M, 1, K, stream);
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms_base = 0.0f; CK(cudaEventElapsedTime(&ms_base, e0, e1));
    ms_base /= iters;

    // ---- aligned prototype (warp per row) ----------------------------------
    const int4 *qs = (const int4 *)(dArt + dq_bytes);
    const __half *dq = (const __half *)dArt;
    q8_aligned_dense_vec_kernel<<<M, 32, 0, stream>>>(dOutAl, qs, dq, dX8, M, nb);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++) {
        const uint8_t *art_i = dArt + (size_t)(i % n_copies) * ART.size();
        q8_aligned_dense_vec_kernel<<<M, 32, 0, stream>>>(
            dOutAl, (const int4 *)(art_i + dq_bytes), (const __half *)art_i, dX8, M, nb);
    }
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms_al = 0.0f; CK(cudaEventElapsedTime(&ms_al, e0, e1));
    ms_al /= iters;

    // ---- variant R2 ---------------------------------------------------------
    q8_aligned_dense_vec_r2_kernel<<<(M + 1) / 2, 64, 0, stream>>>(dOutAl, qs, dq, dX8, M, nb);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++) {
        const uint8_t *art_i = dArt + (size_t)(i % n_copies) * ART.size();
        q8_aligned_dense_vec_r2_kernel<<<(M + 1) / 2, 64, 0, stream>>>(
            dOutAl, (const int4 *)(art_i + dq_bytes), (const __half *)art_i, dX8, M, nb);
    }
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms_r2 = 0.0f; CK(cudaEventElapsedTime(&ms_r2, e0, e1));
    ms_r2 /= iters;

    // ---- correctness: both kernels vs a DOUBLE host reference --------------
    // The per-block integer dot is exact on both paths; only the float
    // accumulation order differs (256 block terms at K=8192), so
    // kernel-vs-kernel tolerance false-fails on cancellation-heavy rows.
    // Judge each kernel by its distance from the double reference instead.
    (void)ds4_mmq_q8_0_dense_vec(dW, dX, dOutBase, M, 1, K, stream);
    q8_aligned_dense_vec_kernel<<<M, 32, 0, stream>>>(dOutAl, qs, dq, dX8, M, nb);
    CK(cudaStreamSynchronize(stream));
    std::vector<float> ob(M), oa(M);
    CK(cudaMemcpy(ob.data(), dOutBase, sizeof(float) * M, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(oa.data(), dOutAl, sizeof(float) * M, cudaMemcpyDeviceToHost));
    double err_b = 0.0, err_a = 0.0; int bad = 0;
    for (int i = 0; i < M; i++) {
        double ref = 0.0;
        const uint8_t *rw = &W[(size_t)i * row_bytes];
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = rw + (size_t)b * BLOCK_BYTES;
            uint16_t h; memcpy(&h, blk, 2);
            const float dw = __half2float(__ushort_as_half(h));
            const int8_t *q = (const int8_t *)(blk + 2);
            const int8_t *u = X8[b].qs;
            int s = 0;
            for (int j = 0; j < 32; j++) s += (int)q[j] * (int)u[j];
            ref += (double)dw * (double)__half2float(__low2half(X8[b].ds)) * (double)s;
        }
        const double eb = fabs((double)ob[i] - ref);
        const double ea = fabs((double)oa[i] - ref);
        if (eb > err_b) err_b = eb;
        if (ea > err_a) err_a = ea;
        // aligned must not be meaningfully further from ref than baseline
        if (ea > eb * 4.0 + 1e-2) bad++;
    }

    printf("%-9s K=%-5d M=%-7d (%.1f MB/call, iters=%d)\n", label, K, M, call_gb * 1000.0, iters);
    printf("  baseline dense_vec : %.4f ms -> %6.1f GB/s\n", ms_base, call_gb / (ms_base / 1e3));
    printf("  aligned  warp/row  : %.4f ms -> %6.1f GB/s  (%+.1f%%)\n",
           ms_al, call_gb / (ms_al / 1e3), 100.0 * (ms_base / ms_al - 1.0));
    printf("  aligned  2row/blk  : %.4f ms -> %6.1f GB/s  (%+.1f%%)\n",
           ms_r2, call_gb / (ms_r2 / 1e3), 100.0 * (ms_base / ms_r2 - 1.0));
    printf("  parity vs double ref: max_abs_err base=%.3e aligned=%.3e bad=%d -> %s\n",
           err_b, err_a, bad, bad == 0 ? "PASS" : "FAIL");
    if (bad != 0) exit(2);

    CK(cudaFree(dW)); CK(cudaFree(dArt)); CK(cudaFree(dX));
    CK(cudaFree(dOutBase)); CK(cudaFree(dOutAl)); CK(cudaFree(dX8));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    CK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char **argv) {
    const uint32_t seed = argc > 1 ? (uint32_t)atoi(argv[1]) : 1234u;
    if (ds4_mmq_init(0) != 0) { printf("ds4_mmq_init failed\n"); return 1; }
    printf("PROTO_Q8_ALIGNED (block_q8_0 34B -> aligned SoA [dq][pad64][qs 32B])\n");
    run_shape("attn_q_b", 32768, 1024, 2000, seed);
    run_shape("mid",      4096,  2048, 2000, seed + 1);
    run_shape("out_a",    4096,  8192, 2000, seed + 3);
    run_shape("head",     129280, 4096, 200, seed + 2);
    return 0;
}
