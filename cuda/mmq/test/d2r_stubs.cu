#include "ds4_mmq_d2r.cuh"

extern "C" int ds4_cuda_q8_fold_take_q81(
        const void *src, uint64_t in_dim, const void **q81) {
    (void)src;
    (void)in_dim;
    (void)q81;
    return 0;
}

// The D2R kernels use NVIDIA cp.async and MMA assembly. Disable them in ROCm
// builds and use the portable raw/SoA paths instead.
bool ds4_mmq_q2_K_moe_d2r_available(int) { return false; }
bool ds4_mmq_iq2_xxs_moe_d2r_available(int) { return false; }
bool ds4_mmq_q8_0_dense_d2r_available(int) { return false; }

size_t ds4_mmq_q2_K_moe_d2r_scratch_bytes(int64_t, int) { return 0; }
size_t ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(int64_t, int) { return 0; }
size_t ds4_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(int64_t, int) { return 0; }

int ds4_mmq_q2_K_moe_d2r_launch(
    const void *, int64_t, const void *, const int32_t *, const int32_t *,
    float *, int, int, int64_t, int, void *, size_t, cudaStream_t) { return 1; }

int ds4_mmq_q8_0_dense_d2r_launch(
    const void *, const void *, float *, int, int, int, cudaStream_t) { return 1; }

int ds4_mmq_iq2_xxs_moe_d2r_pair_launch(
    const void *, const void *, int64_t, const void *, const int32_t *,
    const int32_t *, float *, float *, int, int, int64_t, int, void *, size_t,
    cudaStream_t) { return 1; }

int ds4_mmq_iq2_xxs_moe_d2r_fused_launch(
    const void *, const void *, int64_t, const void *, const int32_t *, int,
    const int32_t *, const int32_t *, const float *, void *, int, int, int64_t,
    int, float, void *, size_t, cudaStream_t) { return 1; }
