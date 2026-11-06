#include "ds4_mmq.h"
#include "ds4_repack.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

static void check(uint32_t kind, int rows, int cols) {
    const int32_t selected[] = {6, 1, 4, 0, 6};
    constexpr int source_count = 7, count = 5;
    const bool iq2 = kind == DS4_REPACK_IQ2_XXS_ALIGNED_MOE;
    const size_t expert_bytes = (size_t)rows * (cols / 256) * (iq2 ? 66u : 84u);
    const size_t packed_bytes = iq2 ? ds4_mmq_iq2_xxs_aligned_bytes(rows, cols, count) :
        ds4_mmq_q2_k_aligned_bytes(rows, cols, count);
    std::vector<unsigned char> source(expert_bytes * source_count);
    std::vector<unsigned char> result(expert_bytes * count);
    uint32_t state = 1729;
    for (auto &byte : source) {
        state = state * 1664525u + 1013904223u;
        byte = state >> 24;
    }
    void *raw = nullptr, *packed = nullptr, *decoded = nullptr;
    int32_t *slots = nullptr;
    CHECK(cudaMalloc(&raw, source.size()) == cudaSuccess);
    CHECK(cudaMalloc(&packed, packed_bytes + 256) == cudaSuccess);
    CHECK(cudaMalloc(&decoded, result.size()) == cudaSuccess);
    CHECK(cudaMalloc((void **)&slots, sizeof(selected)) == cudaSuccess);
    CHECK(cudaMemcpy(raw, source.data(), source.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(slots, selected, sizeof(selected), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemset(packed, 0xa5, packed_bytes + 256) == cudaSuccess);
    CHECK(ds4_repack_selected_experts(packed, raw, slots, kind, rows, cols, count));
    CHECK((iq2 ? ds4_mmq_iq2_xxs_aligned_derepack(packed, decoded, rows, cols, count, 0) :
                 ds4_mmq_q2_K_aligned_derepack(packed, decoded, rows, cols, count, 0)) == 0);
    CHECK(cudaMemcpy(result.data(), decoded, result.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < count; i++)
        CHECK(!std::memcmp(result.data() + i * expert_bytes,
                           source.data() + selected[i] * expert_bytes, expert_bytes));
    unsigned char guard[256];
    CHECK(cudaMemcpy(guard, (char *)packed + packed_bytes, sizeof(guard), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (auto byte : guard) CHECK(byte == 0xa5);
    CHECK(!ds4_repack_selected_experts(packed, raw, slots, kind, rows, cols - 1, count));
    CHECK(!ds4_repack_selected_experts(packed, raw, slots, kind, rows, cols, 0));
    CHECK(!ds4_repack_selected_experts(packed, raw, slots, 0, rows, cols, count));
    CHECK(!ds4_repack_selected_experts(packed, raw, nullptr, kind, rows, cols, count));
    if (!iq2) CHECK(!ds4_repack_selected_experts(packed, raw, slots, kind, rows + 1, cols, count));
    CHECK(cudaFree(slots) == cudaSuccess);
    CHECK(cudaFree(decoded) == cudaSuccess);
    CHECK(cudaFree(packed) == cudaSuccess);
    CHECK(cudaFree(raw) == cudaSuccess);
    std::printf("SSD aligned repack kind=%u rows=%d cols=%d repeated/reordered experts: byte-exact PASS\n",
                kind, rows, cols);
}

int main(void) {
    check(DS4_REPACK_IQ2_XXS_ALIGNED_MOE, 3, 256);
    check(DS4_REPACK_IQ2_XXS_ALIGNED_MOE, 2304, 5120);
    check(DS4_REPACK_Q2_K_ALIGNED_MOE, 6, 768);
    check(DS4_REPACK_Q2_K_ALIGNED_MOE, 5120, 2304);
    return 0;
}
