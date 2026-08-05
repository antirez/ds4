// test_iq2_aligned_entry.cu — M1-Inc1 integration parity gate.
//
// A/B of the PRODUCTION aligned entry (ds4_mmq_iq2_xxs_aligned_moe_vec, fed
// the artifact layout the weight server builds under --repack-iq2-aligned)
// against the production raw-layout vec entry (ds4_mmq_iq2_xxs_moe_vec) at
// the decode shape.  Both entries quantize the activation through
// quantize_row_q8_1_cuda, so only the float accumulation order differs.
//
// The kernel-level A/B with hand-rolled quantize lives in
// proto_iq2_aligned.cu (the original +12% proof); this test locks the
// integrated entry + artifact layout contract.
//
// Build (box):
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 -I.. \
//        test_iq2_aligned_entry.cu ../*.o -lcudart -lcuda -o test_iq2_aligned_entry

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

int main(int argc, char **argv) {
    const int M = 2048, K = 4096;
    const int n_experts = 256;
    const int n_slots = 6;
    const int n_idsets = 32;
    const int iters = 2000;
    const int nb = K / 256;
    const long long row_bytes = (long long)nb * 66;
    const double slot_gb = (double)n_slots * M * row_bytes / 1e9;

    if (ds4_mmq_init(0) != 0) { printf("ds4_mmq_init failed\n"); return 1; }

    std::mt19937 rng(argc > 1 ? (uint32_t)atoi(argv[1]) : 1234u);

    // Random bytes are valid IQ2_XXS content; keep block scales sane.
    std::vector<uint8_t> W((size_t)n_experts * M * row_bytes);
    for (auto &b : W) b = (uint8_t)(rng() & 0xff);
    const long long nblk = (long long)n_experts * M * nb;
    for (long long blk = 0; blk < nblk; blk++) {
        uint16_t h = (uint16_t)(0x2c00 | (rng() & 0xff));
        memcpy(&W[blk * 66], &h, 2);
    }

    std::vector<float> X(K);
    for (auto &v : X) v = ((float)(rng() % 2000) - 1000.0f) / 500.0f;

    std::vector<int32_t> ids(n_idsets * n_slots);
    for (int s = 0; s < n_idsets; s++)
        for (int j = 0; j < n_slots; j++)
            ids[s * n_slots + j] = (s * n_slots + j) % n_experts;

    // Artifact layout, exactly as the weight server repack kernel builds it:
    //   [__half dq[nblk]][pad to 64B][uint2 qs[nblk*8]]
    const uint64_t expect_bytes = ds4_mmq_iq2_xxs_aligned_bytes(M, K, n_experts);
    const uint64_t dq_bytes = ((uint64_t)nblk * 2u + 63u) & ~63ull;
    const uint64_t art_bytes = dq_bytes + (uint64_t)nblk * 64u;
    if (expect_bytes != art_bytes) {
        printf("aligned_bytes mismatch: helper=%llu local=%llu\n",
               (unsigned long long)expect_bytes, (unsigned long long)art_bytes);
        return 1;
    }
    std::vector<uint8_t> ART(art_bytes);
    for (long long blk = 0; blk < nblk; blk++) {
        memcpy(&ART[blk * 2], &W[blk * 66], 2);
        memcpy(&ART[dq_bytes + (uint64_t)blk * 64u], &W[blk * 66 + 2], 64);
    }

    // dArt2: identical bytes at a DIFFERENT address, used as the "up" stream
    // in the pair/fused entries.  Passing dArt twice lets the second stream's
    // reads ride the L2 lines the first stream just fetched, inflating the
    // effective rate past the DRAM ceiling; production gate/up are distinct
    // 553 MB artifacts.
    uint8_t *dW, *dArt, *dArt2; float *dX, *dOutBase, *dOutAl; int32_t *dIds;
    CK(cudaMalloc(&dW, W.size()));
    CK(cudaMalloc(&dArt, ART.size()));
    CK(cudaMalloc(&dArt2, ART.size()));
    CK(cudaMalloc(&dX, sizeof(float) * K));
    CK(cudaMalloc(&dOutBase, sizeof(float) * n_slots * M));
    CK(cudaMalloc(&dOutAl, sizeof(float) * n_slots * M));
    CK(cudaMalloc(&dIds, sizeof(int32_t) * ids.size()));
    CK(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dArt, ART.data(), ART.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dArt2, dArt, ART.size(), cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(dX, X.data(), sizeof(float) * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dIds, ids.data(), sizeof(int32_t) * ids.size(), cudaMemcpyHostToDevice));

    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // ---- de-repack round-trip (M1-Inc2b): artifact -> raw must be byte-exact
    {
        uint8_t *dRaw2; CK(cudaMalloc(&dRaw2, W.size()));
        CK(cudaMemset(dRaw2, 0xAB, W.size()));
        const int drc = ds4_mmq_iq2_xxs_aligned_derepack(dArt, dRaw2, M, K, n_experts, stream);
        if (drc != 0) { printf("derepack rc=%d\n", drc); return 1; }
        CK(cudaStreamSynchronize(stream));
        std::vector<uint8_t> RT(W.size());
        CK(cudaMemcpy(RT.data(), dRaw2, W.size(), cudaMemcpyDeviceToHost));
        if (memcmp(RT.data(), W.data(), W.size()) != 0) {
            size_t off = 0; while (off < W.size() && RT[off] == W[off]) off++;
            printf("derepack round-trip MISMATCH at byte %zu\n", off);
            return 1;
        }
        CK(cudaEventRecord(e0, stream));
        const int dr_iters = 50;
        for (int i = 0; i < dr_iters; i++)
            (void)ds4_mmq_iq2_xxs_aligned_derepack(dArt, dRaw2, M, K, n_experts, stream);
        CK(cudaEventRecord(e1, stream));
        CK(cudaStreamSynchronize(stream));
        float ms_dr = 0.0f; CK(cudaEventElapsedTime(&ms_dr, e0, e1));
        ms_dr /= dr_iters;
        printf("  derepack round-trip: byte-exact PASS, %.3f ms/fill -> %.1f GB/s (%.1f MB tensor)\n",
               ms_dr, (double)(ART.size() + W.size()) / ms_dr / 1e6,
               (double)W.size() / 1e6);
        CK(cudaFree(dRaw2));
    }

    // ---- baseline: raw-layout production vec entry -------------------------
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

    // ---- aligned production entry ------------------------------------------
    rc = ds4_mmq_iq2_xxs_aligned_moe_vec(dArt, dX, dIds, dOutAl, M, K, 1, n_experts, n_slots, stream);
    if (rc != 0) { printf("aligned entry rc=%d\n", rc); return 1; }
    // width > 16 (beyond the vec-tier envelope) must be rejected so the
    // caller can fall back; 1..16 are accepted since the multi-token fix.
    if (ds4_mmq_iq2_xxs_aligned_moe_vec(dArt, dX, dIds, dOutAl, M, K, 17, n_experts, n_slots, stream) == 0) {
        printf("aligned entry accepted n_tokens=17 (must reject)\n");
        return 1;
    }
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
        (void)ds4_mmq_iq2_xxs_aligned_moe_vec(dArt, dX, dIds + (i % n_idsets) * n_slots, dOutAl,
                                              M, K, 1, n_experts, n_slots, stream);
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms_al = 0.0f; CK(cudaEventElapsedTime(&ms_al, e0, e1));
    ms_al /= iters;

    // ---- Inc2 variant P: pair (one quantize, one z=2 launch) ----------------
    float *dOutUpP; CK(cudaMalloc(&dOutUpP, sizeof(float) * n_slots * M));
    rc = ds4_mmq_iq2_xxs_aligned_moe_pair_vec(dArt, dArt2, dX, dIds, dOutAl, dOutUpP,
                                              M, K, 1, n_experts, n_slots, stream);
    if (rc != 0) { printf("pair entry rc=%d\n", rc); return 1; }
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
        (void)ds4_mmq_iq2_xxs_aligned_moe_pair_vec(dArt, dArt2, dX,
                                                   dIds + (i % n_idsets) * n_slots,
                                                   dOutAl, dOutUpP, M, K, 1, n_experts, n_slots, stream);
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms_pair = 0.0f; CK(cudaEventElapsedTime(&ms_pair, e0, e1));
    ms_pair /= iters;

    // ---- Inc2 variant F: fused gate+up+swiglu -> mid ------------------------
    float *dMid, *dSlotW; CK(cudaMalloc(&dMid, sizeof(float) * n_slots * M));
    CK(cudaMalloc(&dSlotW, sizeof(float) * n_slots));
    std::vector<float> slotw(n_slots);
    for (int j = 0; j < n_slots; j++) slotw[j] = 0.5f + 0.1f * j;
    CK(cudaMemcpy(dSlotW, slotw.data(), sizeof(float) * n_slots, cudaMemcpyHostToDevice));
    const float clampv = 10.0f;
    rc = ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec(dArt, dArt2, dX, dIds, dSlotW, dMid,
                                                     M, K, 1, n_experts, n_slots, clampv, stream);
    if (rc != 0) { printf("fused entry rc=%d\n", rc); return 1; }
    CK(cudaStreamSynchronize(stream));
    CK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++)
        (void)ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec(dArt, dArt2, dX,
                                                          dIds + (i % n_idsets) * n_slots,
                                                          dSlotW, dMid, M, K, 1, n_experts, n_slots,
                                                          clampv, stream);
    CK(cudaEventRecord(e1, stream));
    CK(cudaStreamSynchronize(stream));
    float ms_fused = 0.0f; CK(cudaEventElapsedTime(&ms_fused, e0, e1));
    ms_fused /= iters;

    // NOTE: pair/fused per-iter weight bytes are 2x slot_gb (gate + up); the
    // baseline-equivalent cost of one iter is 2x the single-entry time.

    // ---- parity on idset 0 --------------------------------------------------
    (void)ds4_mmq_iq2_xxs_moe_vec(dW, dX, dIds, dOutBase, M, K, 1, n_experts, n_slots, stream);
    (void)ds4_mmq_iq2_xxs_aligned_moe_vec(dArt, dX, dIds, dOutAl, M, K, 1, n_experts, n_slots, stream);
    float *dPairG; CK(cudaMalloc(&dPairG, sizeof(float) * n_slots * M));
    (void)ds4_mmq_iq2_xxs_aligned_moe_pair_vec(dArt, dArt2, dX, dIds, dPairG, dOutUpP,
                                               M, K, 1, n_experts, n_slots, stream);
    (void)ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec(dArt, dArt2, dX, dIds, dSlotW, dMid,
                                                      M, K, 1, n_experts, n_slots, clampv, stream);
    CK(cudaStreamSynchronize(stream));
    std::vector<float> ob(n_slots * M), oa(n_slots * M), opg(n_slots * M), opu(n_slots * M), om(n_slots * M);
    CK(cudaMemcpy(ob.data(), dOutBase, sizeof(float) * ob.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(oa.data(), dOutAl, sizeof(float) * oa.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(opg.data(), dPairG, sizeof(float) * opg.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(opu.data(), dOutUpP, sizeof(float) * opu.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(om.data(), dMid, sizeof(float) * om.size(), cudaMemcpyDeviceToHost));

    auto count_bad = [](const std::vector<float> &ref, const std::vector<float> &got,
                        double *mr, double *ma) {
        double max_rel = 0.0, max_abs = 0.0; int bad = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            const double a = ref[i], b = got[i];
            const double ad = fabs(a - b);
            const double rd = ad / (fabs(a) > 1e-3 ? fabs(a) : 1e-3);
            if (ad > max_abs) max_abs = ad;
            if (rd > max_rel) max_rel = rd;
            if (rd > 2e-2 && ad > 1e-2) bad++;
        }
        if (mr) *mr = max_rel;
        if (ma) *ma = max_abs;
        return bad;
    };
    double max_rel = 0.0, max_abs = 0.0;
    int bad = count_bad(ob, oa, &max_rel, &max_abs);
    // pair: gate and up both computed from dArt, so both must match the
    // aligned single-entry output exactly (same kernel math, same inputs).
    int bad_pair = count_bad(ob, opg, nullptr, nullptr) + count_bad(ob, opu, nullptr, nullptr);
    // fused: expected mid from the baseline outputs + the exact epilogue.
    std::vector<float> em(n_slots * M);
    for (int s = 0; s < n_slots; s++) {
        for (int r = 0; r < M; r++) {
            float g = ob[(size_t)s * M + r], u = g;
            if (!std::isfinite(g)) g = 0.0f;
            if (!std::isfinite(u)) u = 0.0f;
            if (clampv > 1.0e-6f) {
                if (g > clampv) g = clampv;
                if (u > clampv) u = clampv;
                if (u < -clampv) u = -clampv;
            }
            const float silu = g / (1.0f + expf(-g));
            em[(size_t)s * M + r] = silu * u * slotw[s];
        }
    }
    int bad_mid = count_bad(em, om, nullptr, nullptr);

    // ---- multi-token parity (batch_eval-hang fix, 2026-07-03): aligned vs
    // raw production entries at n_tokens=4, flat assignment layout
    // [tok*n_slots+slot], including one -1 router id (NaN-path guard).
    int bad4 = 0, bad4_mid = 0;
    {
        const int n_tok = 4, n_asg = n_tok * n_slots;
        std::vector<float> X4((size_t)n_tok * K);
        for (auto &v : X4) v = ((float)(rng() % 2000) - 1000.0f) / 500.0f;
        std::vector<int32_t> ids4(n_asg);
        for (int a = 0; a < n_asg; a++) ids4[a] = (a * 7 + 3) % n_experts;
        ids4[13] = -1;  // router NaN path: both entries must write a clean 0 row
        std::vector<float> w4(n_asg);
        for (int a = 0; a < n_asg; a++) w4[a] = 0.25f + 0.05f * a;
        float *dX4, *dOut4B, *dOut4A, *dMid4, *dW4; int32_t *dIds4;
        CK(cudaMalloc(&dX4, sizeof(float) * X4.size()));
        CK(cudaMalloc(&dOut4B, sizeof(float) * n_asg * M));
        CK(cudaMalloc(&dOut4A, sizeof(float) * n_asg * M));
        CK(cudaMalloc(&dMid4, sizeof(float) * n_asg * M));
        CK(cudaMalloc(&dW4, sizeof(float) * n_asg));
        CK(cudaMalloc(&dIds4, sizeof(int32_t) * n_asg));
        CK(cudaMemcpy(dX4, X4.data(), sizeof(float) * X4.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW4, w4.data(), sizeof(float) * n_asg, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dIds4, ids4.data(), sizeof(int32_t) * n_asg, cudaMemcpyHostToDevice));
        int r1 = ds4_mmq_iq2_xxs_moe_vec(dW, dX4, dIds4, dOut4B, M, K, n_tok, n_experts, n_slots, stream);
        int r2 = ds4_mmq_iq2_xxs_aligned_moe_vec(dArt, dX4, dIds4, dOut4A, M, K, n_tok, n_experts, n_slots, stream);
        int r3 = ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec(dArt, dArt2, dX4, dIds4, dW4, dMid4,
                                                             M, K, n_tok, n_experts, n_slots, clampv, stream);
        if (r1 != 0 || r2 != 0 || r3 != 0) {
            printf("multi-token entries rc=%d/%d/%d\n", r1, r2, r3);
            return 1;
        }
        CK(cudaStreamSynchronize(stream));
        std::vector<float> ob4((size_t)n_asg * M), oa4((size_t)n_asg * M), om4((size_t)n_asg * M);
        CK(cudaMemcpy(ob4.data(), dOut4B, sizeof(float) * ob4.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(oa4.data(), dOut4A, sizeof(float) * oa4.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(om4.data(), dMid4, sizeof(float) * om4.size(), cudaMemcpyDeviceToHost));
        bad4 = count_bad(ob4, oa4, nullptr, nullptr);
        std::vector<float> em4((size_t)n_asg * M);
        for (int a = 0; a < n_asg; a++) {
            for (int r = 0; r < M; r++) {
                float g = ob4[(size_t)a * M + r], u = g;
                if (!std::isfinite(g)) g = 0.0f;
                if (!std::isfinite(u)) u = 0.0f;
                if (clampv > 1.0e-6f) {
                    if (g > clampv) g = clampv;
                    if (u > clampv) u = clampv;
                    if (u < -clampv) u = -clampv;
                }
                const float silu = g / (1.0f + expf(-g));
                em4[(size_t)a * M + r] = silu * u * w4[a];
            }
        }
        bad4_mid = count_bad(em4, om4, nullptr, nullptr);
        CK(cudaFree(dX4)); CK(cudaFree(dOut4B)); CK(cudaFree(dOut4A));
        CK(cudaFree(dMid4)); CK(cudaFree(dW4)); CK(cudaFree(dIds4));
    }

    printf("TEST_IQ2_ALIGNED_ENTRY  M=%d K=%d slots=%d experts=%d iters=%d (weights/iter %.1f MB)\n",
           M, K, n_slots, n_experts, iters, slot_gb * 1000.0);
    printf("  baseline moe_vec : %.4f ms  -> %6.1f GB/s   (x2 for gate+up: %.4f ms)\n",
           ms_base, slot_gb / (ms_base / 1e3), 2.0f * ms_base);
    printf("  aligned  entry   : %.4f ms  -> %6.1f GB/s   (%+.1f%%; x2: %.4f ms)\n",
           ms_al, slot_gb / (ms_al / 1e3), 100.0 * (ms_base / ms_al - 1.0), 2.0f * ms_al);
    printf("  pair     entry   : %.4f ms  -> %6.1f GB/s   (vs 2x aligned %+.1f%%)\n",
           ms_pair, 2.0 * slot_gb / (ms_pair / 1e3), 100.0 * (2.0f * ms_al / ms_pair - 1.0));
    printf("  fused    entry   : %.4f ms  -> %6.1f GB/s   (vs 2x aligned %+.1f%%)\n",
           ms_fused, 2.0 * slot_gb / (ms_fused / 1e3), 100.0 * (2.0f * ms_al / ms_fused - 1.0));
    printf("  parity: aligned max_rel=%.3e max_abs=%.3e bad=%d | pair bad=%d | fused-mid bad=%d"
           " | ntok4 bad=%d ntok4-mid bad=%d -> %s\n",
           max_rel, max_abs, bad, bad_pair, bad_mid, bad4, bad4_mid,
           (bad == 0 && bad_pair == 0 && bad_mid == 0 && bad4 == 0 && bad4_mid == 0) ? "PASS" : "FAIL");
    return (bad == 0 && bad_pair == 0 && bad_mid == 0 && bad4 == 0 && bad4_mid == 0) ? 0 : 2;
}
