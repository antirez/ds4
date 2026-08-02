// Validates ds4_rocm_wmma_gfx906.cuh on real gfx906 hardware.
// Mimics the ds4 MoE tile kernel pattern: blockDim=256, wave=tid>>5 (8 software
// wave32 groups on wave64 hardware), rocwmma fragment load/mma/store.
// Verifies: launch safety (no HSA exception) + numerics vs CPU reference.
#include "ds4_rocm.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS4_ROCM_UNUSED __attribute__((unused))
#include "rocm/ds4_rocm_common.cuh"
#include "rocm/ds4_rocm_q8.cuh"

enum {
    DS4_ROCM_N_EXPERT = 256u,
    DS4_ROCM_MAX_N_EXPERT = 384u,
    DS4_ROCM_N_EXPERT_USED = 6u
};
#define DS4_ROCM_ROUTER_KERNEL_ONLY
#include "rocm/ds4_rocm_router.cuh"

#define MTILES 8
#define NB 4  // 4 B fragments per wave like the MoE gate_up_mid kernel (bg0,bu0,bg1,bu1)

static bool hip_ok(hipError_t err, const char *what) {
    if (err == hipSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
    return false;
}

__global__ void shim_mma_kernel(const half *A, const half *B, float *C, int n_waves) {
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    if (wave >= (uint32_t)n_waves) return;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::row_major> a;
    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half, rocwmma::row_major> b0, b1, b2, b3;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float, rocwmma::row_major> c0, c1, c2, c3;

    rocwmma::fill_fragment(c0, 0.0f);
    rocwmma::fill_fragment(c1, 0.0f);
    rocwmma::fill_fragment(c2, 0.0f);
    rocwmma::fill_fragment(c3, 0.0f);

    // 2 K-step iterations to exercise accumulation (like k0 loop in ds4).
    for (uint32_t k0 = 0; k0 < 2u; k0++) {
        rocwmma::load_matrix_sync(a, A + (size_t)wave * 512 + k0 * 256, 16);
        rocwmma::load_matrix_sync(b0, B + ((size_t)wave * NB + 0) * 512 + k0 * 256, 16);
        rocwmma::load_matrix_sync(b1, B + ((size_t)wave * NB + 1) * 512 + k0 * 256, 16);
        rocwmma::load_matrix_sync(b2, B + ((size_t)wave * NB + 2) * 512 + k0 * 256, 16);
        rocwmma::load_matrix_sync(b3, B + ((size_t)wave * NB + 3) * 512 + k0 * 256, 16);
        rocwmma::mma_sync(c0, a, b0, c0);
        rocwmma::mma_sync(c1, a, b1, c1);
        rocwmma::mma_sync(c2, a, b2, c2);
        rocwmma::mma_sync(c3, a, b3, c3);
    }

    rocwmma::store_matrix_sync(C + ((size_t)wave * NB + 0) * 256, c0, 16, rocwmma::mem_row_major);
    rocwmma::store_matrix_sync(C + ((size_t)wave * NB + 1) * 256, c1, 16, rocwmma::mem_row_major);
    rocwmma::store_matrix_sync(C + ((size_t)wave * NB + 2) * 256, c2, 16, rocwmma::mem_row_major);
    rocwmma::store_matrix_sync(C + ((size_t)wave * NB + 3) * 256, c3, 16, rocwmma::mem_row_major);
}

// CPU reference: C[i][j] = sum over 2 K-tiles of sum_k A[i][k]*B[k][j]
static void ref_gemm(const std::vector<float> &A, const std::vector<float> &B,
                     std::vector<float> &C, int wave, int nb) {
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            float acc = 0.0f;
            for (int kt = 0; kt < 2; kt++)
                for (int k = 0; k < 16; k++)
                    acc += A[(size_t)wave * 512 + kt * 256 + i * 16 + k] *
                           B[((size_t)wave * NB + nb) * 512 + kt * 256 + k * 16 + j];
            C[((size_t)wave * NB + nb) * 256 + i * 16 + j] = acc;
        }
}

static bool test_q8_wave64_row_packing() {
    const uint64_t in_dim = 4096;
    const uint64_t out_dim = 4097;
    const uint64_t blocks = in_dim / 32;
    const size_t weight_bytes = (size_t)out_dim * blocks * 34u;

    std::vector<unsigned char> hw(weight_bytes);
    std::vector<int8_t> hxq(blocks * 32u);
    std::vector<float> hxscale(blocks);
    std::vector<float> h1(out_dim), h2(out_dim);
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            const half scale = __float2half(0.0025f * (float)(1u + ((row + block) % 13u)));
            unsigned char *dst = hw.data() + ((size_t)row * blocks + block) * 34u;
            std::memcpy(dst, &scale, sizeof(scale));
            for (uint32_t i = 0; i < 32u; i++)
                dst[2u + i] = (unsigned char)(int8_t)(((row * 3u + block * 5u + i * 7u) % 255u) - 127);
        }
    }
    for (size_t i = 0; i < hxq.size(); i++) hxq[i] = (int8_t)(((i * 11u) % 255u) - 127);
    for (size_t i = 0; i < hxscale.size(); i++) hxscale[i] = 0.001f * (float)(1u + (i % 17u));

    unsigned char *dw = nullptr;
    int8_t *dxq = nullptr;
    float *dxscale = nullptr, *d1 = nullptr, *d2 = nullptr;
    const bool allocated =
        hip_ok(hipMalloc(&dw, hw.size()), "q8 malloc weights") &&
        hip_ok(hipMalloc(&dxq, hxq.size()), "q8 malloc activation") &&
        hip_ok(hipMalloc(&dxscale, hxscale.size() * sizeof(float)), "q8 malloc scales") &&
        hip_ok(hipMalloc(&d1, h1.size() * sizeof(float)), "q8 malloc rpb1") &&
        hip_ok(hipMalloc(&d2, h2.size() * sizeof(float)), "q8 malloc rpb2");
    if (!allocated) {
        (void)hipFree(d2); (void)hipFree(d1); (void)hipFree(dxscale);
        (void)hipFree(dxq); (void)hipFree(dw);
        return false;
    }
    bool ok =
        hip_ok(hipMemcpy(dw, hw.data(), hw.size(), hipMemcpyHostToDevice), "q8 copy weights") &&
        hip_ok(hipMemcpy(dxq, hxq.data(), hxq.size(), hipMemcpyHostToDevice), "q8 copy activation") &&
        hip_ok(hipMemcpy(dxscale, hxscale.data(), hxscale.size() * sizeof(float), hipMemcpyHostToDevice),
               "q8 copy scales");
    if (ok) {
        matmul_q8_0_preq_rows_w32_kernel<<<(unsigned)out_dim, 32>>>(
            d1, dw, dxq, dxscale, in_dim, out_dim, blocks, 1u, 1);
        matmul_q8_0_preq_rows_w32_kernel<<<((unsigned)out_dim + 1u) / 2u, 64>>>(
            d2, dw, dxq, dxscale, in_dim, out_dim, blocks, 2u, 1);
        ok = hip_ok(hipDeviceSynchronize(), "q8 packed rows launch") &&
             hip_ok(hipMemcpy(h1.data(), d1, h1.size() * sizeof(float), hipMemcpyDeviceToHost),
                    "q8 copy rpb1") &&
             hip_ok(hipMemcpy(h2.data(), d2, h2.size() * sizeof(float), hipMemcpyDeviceToHost),
                    "q8 copy rpb2");
    }
    size_t mismatches = 0;
    if (ok) {
        for (size_t i = 0; i < h1.size(); i++)
            if (std::memcmp(&h1[i], &h2[i], sizeof(float)) != 0) mismatches++;
    }
    std::printf("Q8 WAVE64 ROW PACKING %s (%zu rows, %zu bit mismatches)\n",
                ok && mismatches == 0 ? "PASS" : "FAIL", h1.size(), mismatches);
    (void)hipFree(d2); (void)hipFree(d1); (void)hipFree(dxscale);
    (void)hipFree(dxq); (void)hipFree(dw);
    return ok && mismatches == 0;
}

static bool test_f16_pair_workgroup_sizing() {
    const uint32_t in_dim = 1024;
    const uint32_t out_dim = 513;
    std::vector<half> hw0((size_t)in_dim * out_dim), hw1(hw0.size());
    std::vector<float> hx(in_dim), href0(out_dim), href1(out_dim), h0(out_dim), h1(out_dim);
    for (size_t i = 0; i < hw0.size(); i++) {
        hw0[i] = __float2half((float)((int)(i % 31u) - 15) / 32.0f);
        hw1[i] = __float2half((float)((int)(i % 29u) - 14) / 32.0f);
    }
    for (uint32_t i = 0; i < in_dim; i++) hx[i] = (float)((int)(i % 23u) - 11) / 16.0f;

    half *dw0 = nullptr, *dw1 = nullptr;
    float *dx = nullptr, *do0 = nullptr, *do1 = nullptr;
    const bool allocated =
        hip_ok(hipMalloc(&dw0, hw0.size() * sizeof(half)), "f16 pair malloc w0") &&
        hip_ok(hipMalloc(&dw1, hw1.size() * sizeof(half)), "f16 pair malloc w1") &&
        hip_ok(hipMalloc(&dx, hx.size() * sizeof(float)), "f16 pair malloc x") &&
        hip_ok(hipMalloc(&do0, href0.size() * sizeof(float)), "f16 pair malloc out0") &&
        hip_ok(hipMalloc(&do1, href1.size() * sizeof(float)), "f16 pair malloc out1");
    if (!allocated) {
        (void)hipFree(do1); (void)hipFree(do0); (void)hipFree(dx);
        (void)hipFree(dw1); (void)hipFree(dw0);
        return false;
    }
    bool ok =
        hip_ok(hipMemcpy(dw0, hw0.data(), hw0.size() * sizeof(half), hipMemcpyHostToDevice),
               "f16 pair copy w0") &&
        hip_ok(hipMemcpy(dw1, hw1.data(), hw1.size() * sizeof(half), hipMemcpyHostToDevice),
               "f16 pair copy w1") &&
        hip_ok(hipMemcpy(dx, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice),
               "f16 pair copy x");
    if (ok) {
        matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel<<<(out_dim + 31u) / 32u,
            1024u, (size_t)in_dim * sizeof(float)>>>(do0, do1, dw0, dw1, dx, in_dim, out_dim);
        ok = hip_ok(hipDeviceSynchronize(), "f16 pair rpb32 launch") &&
             hip_ok(hipMemcpy(href0.data(), do0, href0.size() * sizeof(float), hipMemcpyDeviceToHost),
                    "f16 pair copy reference 0") &&
             hip_ok(hipMemcpy(href1.data(), do1, href1.size() * sizeof(float), hipMemcpyDeviceToHost),
                    "f16 pair copy reference 1");
    }
    if (ok) {
        matmul_f16_pair_f32_sharedx_warp_rows_w32_kernel<<<(out_dim + 7u) / 8u,
            256u, (size_t)in_dim * sizeof(float)>>>(do0, do1, dw0, dw1, dx, in_dim, out_dim);
        ok = hip_ok(hipDeviceSynchronize(), "f16 pair rpb8 launch") &&
             hip_ok(hipMemcpy(h0.data(), do0, h0.size() * sizeof(float), hipMemcpyDeviceToHost),
                    "f16 pair copy rpb8 0") &&
             hip_ok(hipMemcpy(h1.data(), do1, h1.size() * sizeof(float), hipMemcpyDeviceToHost),
                    "f16 pair copy rpb8 1");
    }
    size_t mismatches = 0;
    if (ok) {
        for (size_t i = 0; i < h0.size(); i++)
            if (std::memcmp(&h0[i], &href0[i], sizeof(float)) != 0 ||
                std::memcmp(&h1[i], &href1[i], sizeof(float)) != 0) mismatches++;
    }
    std::printf("F16 PAIR WORKGROUP SIZING %s (%zu rows, %zu bit mismatches)\n",
                ok && mismatches == 0 ? "PASS" : "FAIL", h0.size(), mismatches);
    (void)hipFree(do1); (void)hipFree(do0); (void)hipFree(dx);
    (void)hipFree(dw1); (void)hipFree(dw0);
    return ok && mismatches == 0;
}

template <uint32_t N_EXPERT>
static bool test_router_wave64() {
    std::vector<float> hlogits(N_EXPERT), hbias(N_EXPERT);
    std::vector<int32_t> href_sel(DS4_ROCM_N_EXPERT_USED), h_sel(DS4_ROCM_N_EXPERT_USED);
    std::vector<float> href_weights(DS4_ROCM_N_EXPERT_USED), h_weights(DS4_ROCM_N_EXPERT_USED);
    std::vector<float> href_probs(N_EXPERT), h_probs(N_EXPERT);
    for (uint32_t i = 0; i < N_EXPERT; i++) {
        hlogits[i] = (float)((int)((i * 17u) % 101u) - 50) / 9.0f;
        hbias[i] = (float)((int)((i * 7u) % 31u) - 15) / 100.0f;
    }
    /* Deliberate equal scores exercise the expert-index tie break. */
    hlogits[3] = hlogits[131] = 4.0f;
    hbias[3] = hbias[131] = 0.25f;

    float *dlogits = nullptr, *dbias = nullptr, *dweights = nullptr, *dprobs = nullptr;
    int32_t *dselected = nullptr;
    const bool allocated =
        hip_ok(hipMalloc(&dlogits, hlogits.size() * sizeof(float)), "router malloc logits") &&
        hip_ok(hipMalloc(&dbias, hbias.size() * sizeof(float)), "router malloc bias") &&
        hip_ok(hipMalloc(&dselected, h_sel.size() * sizeof(int32_t)), "router malloc selected") &&
        hip_ok(hipMalloc(&dweights, h_weights.size() * sizeof(float)), "router malloc weights") &&
        hip_ok(hipMalloc(&dprobs, h_probs.size() * sizeof(float)), "router malloc probs");
    if (!allocated) {
        (void)hipFree(dprobs); (void)hipFree(dweights); (void)hipFree(dselected);
        (void)hipFree(dbias); (void)hipFree(dlogits);
        return false;
    }
    bool ok =
        hip_ok(hipMemcpy(dlogits, hlogits.data(), hlogits.size() * sizeof(float), hipMemcpyHostToDevice),
               "router copy logits") &&
        hip_ok(hipMemcpy(dbias, hbias.data(), hbias.size() * sizeof(float), hipMemcpyHostToDevice),
               "router copy bias");
    if (ok) {
        router_select_warp_topk_kernel<N_EXPERT, 32u><<<1, dim3(32, 4, 1)>>>(
            dselected, dweights, dprobs, dbias, nullptr, dlogits, nullptr, 0,
            0, 1, DS4_ROCM_N_EXPERT_USED, 1.5f, 1, 0);
        ok = hip_ok(hipDeviceSynchronize(), "router wave32 launch") &&
             hip_ok(hipMemcpy(href_sel.data(), dselected, href_sel.size() * sizeof(int32_t),
                              hipMemcpyDeviceToHost), "router copy selected reference") &&
             hip_ok(hipMemcpy(href_weights.data(), dweights, href_weights.size() * sizeof(float),
                              hipMemcpyDeviceToHost), "router copy weights reference") &&
             hip_ok(hipMemcpy(href_probs.data(), dprobs, href_probs.size() * sizeof(float),
                              hipMemcpyDeviceToHost), "router copy probs reference");
    }
    if (ok) {
        router_select_warp_topk_kernel<N_EXPERT, 64u><<<1, dim3(64, 1, 1)>>>(
            dselected, dweights, dprobs, dbias, nullptr, dlogits, nullptr, 0,
            0, 1, DS4_ROCM_N_EXPERT_USED, 1.5f, 1, 0);
        ok = hip_ok(hipDeviceSynchronize(), "router wave64 launch") &&
             hip_ok(hipMemcpy(h_sel.data(), dselected, h_sel.size() * sizeof(int32_t),
                              hipMemcpyDeviceToHost), "router copy selected wave64") &&
             hip_ok(hipMemcpy(h_weights.data(), dweights, h_weights.size() * sizeof(float),
                              hipMemcpyDeviceToHost), "router copy weights wave64") &&
             hip_ok(hipMemcpy(h_probs.data(), dprobs, h_probs.size() * sizeof(float),
                              hipMemcpyDeviceToHost), "router copy probs wave64");
    }
    size_t mismatches = 0;
    if (ok) {
        for (size_t i = 0; i < h_sel.size(); i++) {
            mismatches += h_sel[i] != href_sel[i];
            mismatches += std::memcmp(&h_weights[i], &href_weights[i], sizeof(float)) != 0;
        }
        for (size_t i = 0; i < h_probs.size(); i++)
            mismatches += std::memcmp(&h_probs[i], &href_probs[i], sizeof(float)) != 0;
    }
    std::printf("ROUTER WAVE64 %s (%u experts, %zu bit mismatches)\n",
                ok && mismatches == 0 ? "PASS" : "FAIL", N_EXPERT, mismatches);
    (void)hipFree(dprobs); (void)hipFree(dweights); (void)hipFree(dselected);
    (void)hipFree(dbias); (void)hipFree(dlogits);
    return ok && mismatches == 0;
}

int main() {
    const int waves = MTILES;
    std::vector<float> hA(waves * 512), hB(waves * NB * 512);
    std::vector<float> hC_ref(waves * NB * 256, 0.0f);
    std::vector<float> hC(waves * NB * 256, -1.0f);

    srand(42);
    for (auto &v : hA) v = ((rand() % 17) - 8) / 8.0f;
    for (auto &v : hB) v = ((rand() % 13) - 6) / 8.0f;

    for (int w = 0; w < waves; w++)
        for (int nb = 0; nb < NB; nb++) ref_gemm(hA, hB, hC_ref, w, nb);

    half *dA = nullptr, *dB = nullptr;
    float *dC = nullptr;
    std::vector<half> hAh(hA.size()), hBh(hB.size());
    for (size_t i = 0; i < hA.size(); i++) hAh[i] = __float2half(hA[i]);
    for (size_t i = 0; i < hB.size(); i++) hBh[i] = __float2half(hB[i]);

    if (!hip_ok(hipMalloc(&dA, hAh.size() * sizeof(half)), "malloc A") ||
        !hip_ok(hipMalloc(&dB, hBh.size() * sizeof(half)), "malloc B") ||
        !hip_ok(hipMalloc(&dC, hC.size() * sizeof(float)), "malloc C") ||
        !hip_ok(hipMemcpy(dA, hAh.data(), hAh.size() * sizeof(half), hipMemcpyHostToDevice), "copy A") ||
        !hip_ok(hipMemcpy(dB, hBh.data(), hBh.size() * sizeof(half), hipMemcpyHostToDevice), "copy B") ||
        !hip_ok(hipMemset(dC, 0, hC.size() * sizeof(float)), "clear C")) {
        (void)hipFree(dC);
        (void)hipFree(dB);
        (void)hipFree(dA);
        return 2;
    }

    shim_mma_kernel<<<1, 256>>>(dA, dB, dC, waves);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("SHIM TEST LAUNCH FAILURE: %s\n", hipGetErrorString(err));
        return 3;
    }
    if (!hip_ok(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost), "copy C")) {
        (void)hipFree(dC);
        (void)hipFree(dB);
        (void)hipFree(dA);
        return 3;
    }

    double max_err = 0.0;
    for (size_t i = 0; i < hC.size(); i++) {
        double e = fabs((double)hC[i] - (double)hC_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("max_err=%g (256 tiles checked, %zu values)\n", max_err, hC.size());
    (void)hipFree(dC);
    (void)hipFree(dB);
    (void)hipFree(dA);
    if (max_err > 1e-3) { printf("SHIM TEST NUMERIC MISMATCH\n"); return 4; }
    if (!test_q8_wave64_row_packing()) return 5;
    if (!test_f16_pair_workgroup_sizing()) return 6;
    if (!test_router_wave64<DS4_ROCM_N_EXPERT>()) return 7;
    if (!test_router_wave64<DS4_ROCM_MAX_N_EXPERT>()) return 8;
    printf("SHIM TEST PASS\n");
    return 0;
}
