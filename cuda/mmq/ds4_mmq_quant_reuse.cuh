// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>

namespace ds4_mmq_quant_reuse {

constexpr uint32_t values_per_record = 128u;
constexpr uint32_t bytes_per_record = 144u;
constexpr uint32_t vectors_per_record = 9u;
constexpr uint32_t gather_threads = 256u;

// The gfx1151 IQ2 gate/up pair with complete expert tables is submitted in
// chunks of at most 2048 tokens. This includes a resident model or a complete
// layer bank loaded by SSD streaming. Other shapes keep their original path.
static inline bool select(bool iq2, bool gfx1151, bool direct_gateup_q8,
                          int k, int tokens, int experts, int top_k) {
    return iq2 && gfx1151 && !direct_gateup_q8 && k == 4096 &&
           tokens >= 128 && tokens <= 2048 && experts == 256 && top_k == 6;
}

static inline size_t compact_bytes(int64_t padded_k, int tokens) {
    return size_t(padded_k / values_per_record) * size_t(tokens) * bytes_per_record;
}

} // namespace ds4_mmq_quant_reuse

// Both buffers keep the original MMQ transposed-record layout. Only the row
// count differs: compact[kseg][token] -> gathered[kseg][assignment]. Each
// record is exactly nine aligned uint4 vectors, including all four F32 D4
// scales. No quantization, floating-point operation or permutation changes.
__global__ static void ds4_mmq_gather_q8_1_records_kernel(
        const void * __restrict__ compact,
        const int32_t * __restrict__ ids_src1,
        void * __restrict__ gathered,
        uint32_t tokens,
        uint32_t assignments) {
    const uint64_t vector = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t row_vectors = uint64_t(assignments) * ds4_mmq_quant_reuse::vectors_per_record;
    if (vector >= row_vectors) return;
    const uint32_t assignment = uint32_t(vector / ds4_mmq_quant_reuse::vectors_per_record);
    const uint32_t within = uint32_t(vector % ds4_mmq_quant_reuse::vectors_per_record);
    // mm_ids_helper emits token IDs in [0,tokens). Its unwritten map tail is
    // pre-zeroed by the pair wrapper and must gather token zero, not zeros.
    const uint32_t token = uint32_t(ids_src1[assignment]);
    const uint64_t src = (uint64_t(blockIdx.y) * tokens + token) *
                         ds4_mmq_quant_reuse::vectors_per_record + within;
    const uint64_t dst = uint64_t(blockIdx.y) * row_vectors + vector;
    reinterpret_cast<uint4 *>(gathered)[dst] = reinterpret_cast<const uint4 *>(compact)[src];
}
