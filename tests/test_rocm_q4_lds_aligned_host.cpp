// SPDX-License-Identifier: MIT
// Native byte/ownership oracle for packed Q8_K -> aligned LDS staging.
// This does not compile HIP kernels or predict GPU bank conflicts/occupancy.
#include "rocm/ds4_rocm_q4_lds.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lds = ds4_rocm_q4_lds;

struct packed_q8_K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

static_assert(sizeof(packed_q8_K) == 292, "independent packed Q8_K layout");
static_assert(offsetof(packed_q8_K, qs) == 4, "packed quant offset");
static_assert(offsetof(packed_q8_K, bsums) == 260, "packed sum offset");
static_assert(sizeof(lds::aligned_q8_K) == 304, "aligned Q8_K pitch");
static_assert(alignof(lds::aligned_q8_K) == 16, "aligned Q8_K alignment");
static_assert(offsetof(lds::aligned_q8_K, d) == 0, "aligned scale offset");
static_assert(offsetof(lds::aligned_q8_K, padding) == 4, "aligned padding offset");
static_assert(offsetof(lds::aligned_q8_K, qs) == 16, "aligned quant offset");
static_assert(offsetof(lds::aligned_q8_K, bsums) == 272, "aligned sum offset");

static void require(bool ok, const char *what) {
    if (!ok) {
        std::fprintf(stderr, "Q4 aligned LDS host: FAIL %s\n", what);
        std::exit(1);
    }
}

template<uint32_t BLOCKS>
static void test_tile(uint32_t nt, uint32_t nb, uint32_t threads,
                      uint64_t src_start, uint64_t src_stride,
                      uint32_t salt, bool check_owners) {
    // No readable suffix after the final source block: ASan catches rounded
    // loads over the source end, including a one-block final K tile.
    std::vector<uint32_t> src(src_start + (nt - 1u) * src_stride + nb * 73u);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = (uint32_t)i * 2654435761u + salt * 7919u;
    const auto source_before = src;
    std::vector<uint32_t> got(16u + 8u * BLOCKS * 76u + 17u);
    const uint32_t dst_start = 8u +
        (uint32_t)((16u - ((uintptr_t)(got.data() + 8u) & 15u)) & 15u) / 4u;
    require((uintptr_t)(got.data() + dst_start) % 16u == 0u, "destination alignment");
    for (size_t i = 0; i < got.size(); ++i)
        got[i] = 0xdeadbeefu ^ ((uint32_t)i * 2246822519u);
    const auto untouched = got;
    auto expected = got;
    std::vector<int> owner(got.size(), -1);

    // The oracle enumerates fields/blocks rather than decomposing a helper
    // loop index. Each wave owns complete blocks; lanes stripe their words.
    for (uint32_t p = 0; p < nt; ++p)
    for (uint32_t b = 0; b < nb; ++b)
    for (uint32_t w = 0; w < 73u; ++w) {
        const uint64_t source = src_start + (uint64_t)p * src_stride + b * 73u + w;
        const uint32_t dest = dst_start + (p * BLOCKS + b) * 76u +
                              (w == 0u ? 0u : w + 3u);
        require(source < src.size() && dest < got.size(), "reference bounds");
        expected[dest] = src[source];
        owner[dest] = (int)(((p * BLOCKS + b) % (threads / 32u)) * 32u + w % 32u);
        require(expected[dest] != untouched[dest], "distinct copy and canary values");
    }

    if (check_owners) {
        for (uint32_t tid = 0; tid < threads; ++tid) {
            // Isolate each writer: a duplicate store must not hide behind
            // the identical value previously written by its correct owner.
            got = untouched;
            lds::copy_thread_aligned<BLOCKS>(got.data() + dst_start,
                src.data() + src_start, tid, threads, nt, nb, src_stride);
            for (size_t i = 0; i < got.size(); ++i)
                require(got[i] == (owner[i] == (int)tid ? expected[i] : untouched[i]),
                        "per-thread ownership, guard, padding or K/N tail");
        }
    }

    got = untouched;
    // Reverse execution also checks that cooperating writers are disjoint.
    for (uint32_t tid = threads; tid != 0u; --tid)
        lds::copy_thread_aligned<BLOCKS>(got.data() + dst_start,
            src.data() + src_start, tid - 1u, threads, nt, nb, src_stride);
    require(got == expected, "complete staging, guards, padding or K/N tails");
    for (uint32_t p = 0; p < nt; ++p)
    for (uint32_t b = 0; b < nb; ++b) {
        packed_q8_K packed;
        lds::aligned_q8_K aligned;
        std::memcpy(&packed, src.data() + src_start + p * src_stride + b * 73u,
                    sizeof(packed));
        std::memcpy(&aligned, got.data() + dst_start + (p * BLOCKS + b) * 76u,
                    sizeof(aligned));
        require(std::memcmp(&packed.d, &aligned.d, sizeof(packed.d)) == 0,
                "scale bits");
        require(std::memcmp(packed.qs, aligned.qs, sizeof(packed.qs)) == 0,
                "all signed quant bytes");
        require(std::memcmp(packed.bsums, aligned.bsums, sizeof(packed.bsums)) == 0,
                "all signed block sums");
    }
    require(src == source_before, "source immutable");
}

template<uint32_t BLOCKS>
static uint32_t test_single_tiles() {
    uint32_t cases = 0;
    for (uint32_t nt = 1; nt <= 8u; ++nt)
    for (uint32_t nb = 1; nb <= BLOCKS; ++nb)
    for (uint32_t threads : {32u, 64u, 128u, 256u})
    for (uint32_t layout = 0; layout < 4u; ++layout) {
        const uint32_t stride_blocks[] = {BLOCKS, BLOCKS + 1u, 3u * BLOCKS, 64u};
        const uint32_t source_offsets[] = {0u, 1u, BLOCKS * 73u, 333u};
        test_tile<BLOCKS>(nt, nb, threads, source_offsets[layout],
                         (uint64_t)stride_blocks[layout] * 73u, cases, true);
        ++cases;
    }
    return cases;
}

template<uint32_t BLOCKS>
static uint32_t test_k_loops() {
    uint32_t cases = 0;
    for (uint32_t nt = 1; nt <= 8u; ++nt)
    for (uint32_t tail = 0; tail <= BLOCKS; ++tail)
    for (uint32_t groups : {1u, 3u}) {
        // K8192 output-B has 32 Q8_K blocks. Extend it by every possible
        // short K tile, and place the selected group after other groups.
        const uint32_t blocks = 32u + tail;
        const uint64_t stride = (uint64_t)groups * blocks * 73u;
        const uint64_t start = 1u + 2u * stride + (groups - 1u) * blocks * 73u;
        for (uint32_t b0 = 0; b0 < blocks; b0 += BLOCKS) {
            test_tile<BLOCKS>(nt, std::min(BLOCKS, blocks - b0), 256u,
                             start + b0 * 73u, stride, cases, false);
            ++cases;
        }
    }
    return cases;
}

static uint32_t test_scope() {
    uint32_t cases = 0;
    for (uint32_t blocks : {4u, 8u, 31u, 32u, 33u, 64u})
    for (uint32_t out_dim : {0u, 512u, 4095u, 4096u, 4097u})
    for (uint32_t nt : {0u, 1u, 8u, 255u, 256u, 257u, 4095u, 4096u, 4097u, 8192u})
    for (uint32_t groups : {0u, 1u, 2u, 8u})
    for (uint32_t flags = 0; flags < 16u; ++flags) {
        const bool ssd = flags & 1u, quality = flags & 2u;
        const bool gfx1151 = flags & 4u, disabled = flags & 8u;
        const bool expected = blocks == 32u && out_dim == 4096u &&
            nt >= 256u && nt <= 4096u && groups == 1u &&
            !ssd && !quality && gfx1151 && !disabled;
        require(lds::aligned_scope(blocks, out_dim, nt, groups,
                                  ssd, quality, gfx1151, disabled) == expected,
                "aligned output-B policy boundaries and exclusions");
        ++cases;
    }
    return cases;
}

int main() {
    const uint32_t policies = test_scope();
    const uint32_t tiles = test_single_tiles<4u>() + test_single_tiles<8u>();
    const uint32_t k_tiles = test_k_loops<4u>() + test_k_loops<8u>();
    std::printf("Q4 aligned LDS host: PASS %u policies, %u per-thread tiles, %u K-loop tiles "
                "(all N/K tails, group strides, exact source ends, field bits, "
                "padding/canaries and immutable source).\n", policies, tiles, k_tiles);
    std::puts("Host-only: HIP compilation, GPU bitwise parity and timing remain required.");
    return 0;
}
