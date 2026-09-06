// SPDX-License-Identifier: MIT
// Execute the production LDS copy helper on the host against the old mapping.
// This does not compile HIP kernels or emulate GPU barrier/occupancy behavior.
#include "rocm/ds4_rocm_q4_lds.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char *label) {
    if (!ok) { std::fprintf(stderr, "Q4 LDS host FAIL: %s\n", label); std::exit(1); }
}

template<uint32_t BLOCKS>
static uint32_t test_copies() {
    const uint32_t guard = 17u;
    uint32_t cases = 0u;
    for (uint32_t nt = 1u; nt <= 8u; ++nt)
    for (uint32_t nb = 1u; nb <= BLOCKS; ++nb)
    for (uint32_t stride_blocks : {BLOCKS, 2u * BLOCKS + 3u, 64u})
    for (uint32_t threads : {32u, 64u, 256u})
    for (uint32_t offset : {0u, 1u, 73u, 333u}) {
        const uint64_t stride = (uint64_t)stride_blocks * 73u;
        const size_t src_words = offset + (nt - 1u) * stride + nb * 73u;
        // End the allocation at the last valid word (plus a checked guard).
        std::vector<uint32_t> src(src_words + guard);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = (uint32_t)i * 2654435761u + cases * 7919u;
        const auto source_before = src;
        std::vector<uint32_t> expected(2u * guard + 8u * BLOCKS * 73u);
        for (size_t i = 0; i < expected.size(); ++i)
            expected[i] = 0xdead0000u + (uint32_t)i;
        auto got = expected;
        const uint32_t words = nt * BLOCKS * 73u;
        // Original block/word decomposition, deliberately not the helper's
        // token-pitch formula. Also validates the exact per-thread write set.
        for (uint32_t tid = 0u; tid < threads; ++tid) {
            ds4_rocm_q4_lds::copy_thread<BLOCKS>(
                got.data() + guard, src.data() + offset, tid, threads,
                nt, nb, stride);
            for (uint32_t i = tid; i < words; i += threads) {
                const uint32_t slot = i / 73u;
                const uint32_t word = i - slot * 73u;
                const uint32_t p = slot / BLOCKS;
                const uint32_t bb = slot % BLOCKS;
                if (bb < nb) {
                    const uint64_t address = offset +
                        ((uint64_t)p * stride_blocks + bb) * 73u + word;
                    require(address < src_words, "reference source bounds");
                    expected[guard + i] = src[address];
                }
            }
            require(got == expected, "copy bytes, guards, tails or thread ownership");
        }
        require(src == source_before, "source modified");
        ++cases;
    }
    return cases;
}

template<uint32_t BLOCKS>
static uint32_t test_vector_copies() {
    constexpr uint32_t pitch = BLOCKS * 73u;
    uint32_t cases = 0u, vector_cases = 0u, fallback_cases = 0u;
    for (uint32_t nt = 1u; nt <= 8u; ++nt)
    for (uint32_t nb = 1u; nb <= BLOCKS; ++nb)
    for (uint32_t stride_blocks : {BLOCKS, BLOCKS + 1u, 2u * BLOCKS + 4u})
    for (uint32_t threads : {32u, 64u, 256u})
    for (uint32_t src_offset = 0u; src_offset < 4u; ++src_offset)
    for (uint32_t dst_offset = 0u; dst_offset < 4u; ++dst_offset) {
        const uint32_t src_start = 16u + src_offset;
        const uint32_t dst_start = 16u + dst_offset;
        const uint64_t stride = (uint64_t)stride_blocks * 73u;
        // No readable suffix after the last valid source block: ASan also
        // checks that a four-word copy cannot round the final K tail up.
        std::vector<uint32_t> src(src_start + (nt - 1u) * stride + nb * 73u);
        for (size_t i = 0u; i < src.size(); ++i)
            src[i] = (uint32_t)i * 2654435761u + cases * 7919u;
        const auto before = src;
        std::vector<uint32_t> got(dst_start + 8u * pitch + 17u, 0xdeadbeefu);
        auto expected = got;
        const bool vector = ((uintptr_t)(got.data() + dst_start) % 16u == 0u) &&
                            ((uintptr_t)(src.data() + src_start) % 16u == 0u) &&
                            stride % 4u == 0u;
        require(ds4_rocm_q4_lds::vector_copy_aligned(
                    got.data() + dst_start, src.data() + src_start, stride) == vector,
                "vector alignment predicate");
        vector_cases += vector;
        fallback_cases += !vector;
        // Independent original block/word map. Derive the owner for each
        // valid word, including words straddling adjacent 73-word blocks.
        std::vector<std::vector<uint32_t>> owned(threads);
        for (uint32_t p = 0u; p < nt; ++p)
        for (uint32_t b = 0u; b < nb; ++b)
        for (uint32_t w = 0u; w < 73u; ++w) {
            const uint32_t index = p * pitch + b * 73u + w;
            const uint32_t owner = (vector ? index / 4u : index) % threads;
            owned[owner].push_back(index);
        }
        for (uint32_t tid = 0u; tid < threads; ++tid) {
            // Isolate each writer so duplicate stores by a different thread
            // cannot hide behind the value left by the correct owner.
            std::fill(got.begin(), got.end(), 0xdeadbeefu);
            std::fill(expected.begin(), expected.end(), 0xdeadbeefu);
            ds4_rocm_q4_lds::copy_thread_selected<BLOCKS, true>(
                got.data() + dst_start, src.data() + src_start, tid, threads,
                nt, nb, stride);
            for (uint32_t index : owned[tid]) {
                const uint32_t p = index / pitch;
                const uint32_t within = index % pitch;
                expected[dst_start + index] = src[src_start + p * stride + within];
            }
            require(got == expected, "vector/fallback bytes, per-thread ownership, guards or tails");
        }
        require(src == before, "vector source modified");
        ++cases;
    }
    require(vector_cases != 0u && fallback_cases != 0u, "both loader paths exercised");
    return cases;
}

int main() {
    const uint32_t copies = test_copies<4u>() + test_copies<8u>();
    const uint32_t vectors = test_vector_copies<4u>() + test_vector_copies<8u>();
    for (uint32_t blocks = 1u; blocks <= 4096u; ++blocks) {
        uint32_t load_barriers = 0u, reuse_barriers = 0u;
        for (uint32_t b0 = 0u; b0 < blocks; b0 += 8u) {
            ++load_barriers;
            const bool next_tile = b0 / 8u + 1u < (blocks + 7u) / 8u;
            require(ds4_rocm_q4_lds::needs_reuse_barrier(b0, blocks) == next_tile,
                    "reuse fence must cover every non-final tile");
            reuse_barriers += ds4_rocm_q4_lds::needs_reuse_barrier(b0, blocks);
        }
        require(load_barriers + reuse_barriers == 2u * load_barriers - 1u,
                "exactly one redundant fence removed");
    }
    std::printf("PASS: %u Q4 LDS copy cases (per-thread mapping, offsets, strides, tails, guards); "
                "4096 K-loop barrier schedules.\n", copies);
    std::printf("PASS: %u vector/scalar LDS cases (all word alignments, short K/N tiles, exact source ends).\n",
                vectors);
    std::puts("Host-only: HIP compilation, GPU bitwise parity and timing remain required.");
    return 0;
}
