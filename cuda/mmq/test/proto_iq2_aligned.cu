// proto_iq2_aligned.cu — M1-MoE Inc1 prototype (megakernel program).
//
// HYPOTHESIS: the IQ2_XXS decode matvec runs at ~142 GB/s (vs the ~200
// engine ceiling) because block_iq2_xxs is 66 bytes, so the code stream is
// only 2-byte aligned and every 32-bit weight word costs two 16-bit loads
// (get_int_b2).  Repacking into an aligned SoA layout (d[] halves separate,
// qs[] 64B-aligned per block) allows full-width aligned loads at identical
// byte count.
//
// A/B at the exact production decode shape (per layer, per token):
//   gate/up experts: M=2048 rows, K=4096, top-6 of n_experts, n_tokens=1
//   baseline = ds4_mmq_iq2_xxs_moe_vec (the production vec-tier entry)
//   aligned  = warp-per-row kernel, same dp4a/grid/sign math, SoA weights
//
// Correctness: aligned vs baseline output, tolerance (float accum order
// differs across blocks; per-pair integer math is identical).
//
// Build (box):
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 -I.. \
//        proto_iq2_aligned.cu ../ds4_mmq.o -lcudart -lcuda -o proto_iq2_aligned

#include "ds4_mmq.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "../ggml-common.h"

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

constexpr int QKK = 256;              // values per IQ2_XXS block
constexpr int BLOCK_BYTES = 66;       // sizeof(block_iq2_xxs) = 2 + 32*2
constexpr int PAIRS_PER_BLOCK = 8;    // 8 (q2,aux32) uint32 pairs, 32 values each

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
__device__ __forceinline__ uint32_t proto_unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}

__device__ __forceinline__ int proto_dp4a(const int a, const int b, int c) {
    return __dp4a(a, b, c);
}

// Aligned SoA weights: for (expert e, row r) with nb blocks per row:
//   dq[(e*M + r)*nb + b]                      : __half block scale
//   qs[((e*M + r)*nb + b) * 8 + p]            : uint2 (q2, aux32) pair p
// qs base is 16B-aligned; each block's 8 uint2 = 64B, 64B-aligned stride.
__global__ void iq2_aligned_moe_vec_kernel(
        float             *out,        // [n_slots, M]
        const uint2       *qs,         // aligned code pairs
        const __half      *dq,         // block scales
        const q8_1_block  *x8,         // [K/32] quantized activation (shared by slots)
        const int32_t     *ids,        // [n_slots] expert ids
        const float       *slot_w,     // [n_slots] router weights (1.0 in proto)
        int                M,
        int                nb)         // blocks per row = K/256
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;
    const int lane = threadIdx.x;      // 32 lanes: lane covers (block b, pair p)
    const long long rbase = ((long long)ids[slot] * M + row) * nb;

    float acc = 0.0f;
    // 32 lanes cover 4 blocks x 8 pairs per pass.
    for (int b0 = 0; b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const uint2 cw   = qs[(rbase + b) * PAIRS_PER_BLOCK + p];   // aligned 8B
        const uint32_t q2 = cw.x, aux32 = cw.y;
        const uint8_t *aux8 = (const uint8_t *)&q2;

        int sumi = 0;
        const int q8i = (b * QKK + p * 32) / 32;   // q8_1 block covering these 32 values
        const int *u = (const int *)x8[q8i].qs;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = proto_unpack_ksigns(aux32 >> (7 * k0 / 2));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
            sumi = proto_dp4a(grid0, u[k0 + 0], sumi);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = proto_dp4a(grid1, u[k0 + 1], sumi);
        }
        const int ls = aux32 >> 27 | 1;
        sumi = sumi * ls / 8;
        const float d = __half2float(dq[rbase + b]) * __low2float(x8[q8i].ds);
        acc += d * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[(long long)slot * M + row] = acc * slot_w[slot];
}

} // namespace

int main(int argc, char **argv) {
    const int M = 2048, K = 4096;
    const int n_experts = 256;
    const int n_slots = 6;
    const int n_idsets = 32;
    const int iters = 2000;
    const int nb = K / QKK;            // 16 blocks/row
    const long long row_bytes = (long long)nb * BLOCK_BYTES;   // 1056
    const double slot_gb = (double)n_slots * M * row_bytes / 1e9;

    std::mt19937 rng(argc > 1 ? (uint32_t)atoi(argv[1]) : 1234u);

    // Random IQ2_XXS blocks (same trick as test_mmq_parity: random bytes are
    // valid IQ2_XXS content; both kernels see identical bits).
    std::vector<uint8_t> W((size_t)n_experts * M * row_bytes);
    for (auto &b : W) b = (uint8_t)(rng() & 0xff);
    // keep block d halves sane (avoid inf/nan): overwrite each block's d
    for (long long blk = 0; blk < (long long)n_experts * M * nb; blk++) {
        uint16_t h = (uint16_t)(0x2c00 | (rng() & 0xff));   // ~0.0625..0.1 range fp16
        memcpy(&W[blk * BLOCK_BYTES], &h, 2);
    }

    std::vector<float> X(K);
    for (auto &v : X) v = ((float)(rng() % 2000) - 1000.0f) / 500.0f;
    std::vector<q8_1_block> X8(K / 32);
    quantize_q8_1_host(X.data(), X8.data(), K);

    std::vector<int32_t> ids(n_idsets * n_slots);
    for (int s = 0; s < n_idsets; s++)
        for (int j = 0; j < n_slots; j++)
            ids[s * n_slots + j] = (s * n_slots + j) % n_experts;

    // Aligned SoA repack
    const long long nblocks = (long long)n_experts * M * nb;
    std::vector<uint2>  QS(nblocks * PAIRS_PER_BLOCK);
    std::vector<__half> DQ(nblocks);
    for (long long blk = 0; blk < nblocks; blk++) {
        const uint8_t *src = &W[blk * BLOCK_BYTES];
        uint16_t h; memcpy(&h, src, 2);
        DQ[blk] = __ushort_as_half(h);
        memcpy(&QS[blk * PAIRS_PER_BLOCK], src + 2, 64);
    }

    // Device buffers
    uint8_t *dW; float *dX, *dOutBase, *dOutAl; int32_t *dIds; uint2 *dQS; __half *dDQ;
    q8_1_block *dX8; float *dSlotW;
    CK(cudaMalloc(&dW, W.size()));
    CK(cudaMalloc(&dX, sizeof(float) * K * n_slots));
    CK(cudaMalloc(&dOutBase, sizeof(float) * n_slots * M));
    CK(cudaMalloc(&dOutAl, sizeof(float) * n_slots * M));
    CK(cudaMalloc(&dIds, sizeof(int32_t) * ids.size()));
    CK(cudaMalloc(&dQS, QS.size() * sizeof(uint2)));
    CK(cudaMalloc(&dDQ, DQ.size() * sizeof(__half)));
    CK(cudaMalloc(&dX8, X8.size() * sizeof(q8_1_block)));
    CK(cudaMalloc(&dSlotW, sizeof(float) * n_slots));
    CK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice));
    // moe_vec's X layout: [n_tokens*n_expert_used rows? or one row]; the vec
    // entry treats X as [n_tokens, K] with router weights applied later; we
    // pass the same activation replicated per slot to keep both paths equal.
    for (int s = 0; s < n_slots; s++)
        CK(cudaMemcpy(dX + (size_t)s * K, X.data(), sizeof(float) * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dIds, ids.data(), sizeof(int32_t) * ids.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dQS, QS.data(), QS.size() * sizeof(uint2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dDQ, DQ.data(), DQ.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX8, X8.data(), X8.size() * sizeof(q8_1_block), cudaMemcpyHostToDevice));
    std::vector<float> slotw(n_slots, 1.0f);
    CK(cudaMemcpy(dSlotW, slotw.data(), sizeof(float) * n_slots, cudaMemcpyHostToDevice));

    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // ---- baseline: production vec entry -----------------------------------
    int rc = ds4_mmq_iq2_xxs_moe_vec(dW, dX, dIds, dOutBase, M, K, 1, n_experts, n_slots, stream);
    if (rc != 0) { printf("baseline moe_vec rc=%d\n", rc); return 1; }
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
        (void)ds4_mmq_iq2_xxs_moe_vec(dW, dX, dIds + (i % n_idsets) * n_slots, dOutBase,
                                      M, K, 1, n_experts, n_slots, stream);
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms_base = 0.0f; CK(cudaEventElapsedTime(&ms_base, e0, e1));
    ms_base /= iters;

    // ---- aligned prototype -------------------------------------------------
    dim3 grid(M, n_slots, 1);
    iq2_aligned_moe_vec_kernel<<<grid, 32, 0, stream>>>(dOutAl, dQS, dDQ, dX8, dIds, dSlotW, M, nb);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
        iq2_aligned_moe_vec_kernel<<<grid, 32, 0, stream>>>(dOutAl, dQS, dDQ, dX8,
                                                            dIds + (i % n_idsets) * n_slots,
                                                            dSlotW, M, nb);
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms_al = 0.0f; CK(cudaEventElapsedTime(&ms_al, e0, e1));
    ms_al /= iters;

    // ---- correctness (vs baseline, last idset used by both loops) ----------
    // rerun both once on idset 0 for a clean compare
    (void)ds4_mmq_iq2_xxs_moe_vec(dW, dX, dIds, dOutBase, M, K, 1, n_experts, n_slots, stream);
    iq2_aligned_moe_vec_kernel<<<grid, 32, 0, stream>>>(dOutAl, dQS, dDQ, dX8, dIds, dSlotW, M, nb);
    CK(cudaStreamSynchronize(stream));
    std::vector<float> ob(n_slots * M), oa(n_slots * M);
    CK(cudaMemcpy(ob.data(), dOutBase, sizeof(float) * ob.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(oa.data(), dOutAl, sizeof(float) * oa.size(), cudaMemcpyDeviceToHost));
    double max_rel = 0.0, max_abs = 0.0; int bad = 0;
    for (size_t i = 0; i < ob.size(); i++) {
        const double a = ob[i], b = oa[i];
        const double ad = fabs(a - b);
        const double rd = ad / (fabs(a) > 1e-3 ? fabs(a) : 1e-3);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        if (rd > 2e-2 && ad > 1e-2) bad++;
    }

    printf("PROTO_IQ2_ALIGNED  M=%d K=%d slots=%d experts=%d iters=%d (weights/iter %.1f MB)\n",
           M, K, n_slots, n_experts, iters, slot_gb * 1000.0);
    printf("  baseline moe_vec : %.4f ms  -> %6.1f GB/s\n", ms_base, slot_gb / (ms_base / 1e3));
    printf("  aligned  proto   : %.4f ms  -> %6.1f GB/s   (%+.1f%%)\n",
           ms_al, slot_gb / (ms_al / 1e3), 100.0 * (ms_base / ms_al - 1.0));
    printf("  parity: max_rel=%.3e max_abs=%.3e bad=%d -> %s\n",
           max_rel, max_abs, bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 2;
}
