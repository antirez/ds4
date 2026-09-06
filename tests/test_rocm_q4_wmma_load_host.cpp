// SPDX-License-Identifier: MIT
// Native address/policy oracle, not an emulator of WMMA or HIP FP16 rounding.
#include "rocm/ds4_rocm_q4_wmma_load.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace load = ds4_rocm_q4_wmma_load;
static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "Q4 K64 LOAD4 host: FAIL %s\n", what); std::exit(1); }
}

int main() {
    size_t barrier_cases = 0;
    // Every stage publishes before WMMA reads it; a reuse fence is needed
    // precisely when another K stage can overwrite those same LDS slots.
    for (uint32_t blocks = 1u; blocks <= 128u; ++blocks)
    for (uint32_t stages : {2u, 4u}) {
        uint32_t barriers = 0;
        for (uint32_t b = 0u; b < blocks; ++b)
        for (uint32_t s = 0u; s < stages; ++s) {
            ++barriers; // unchanged producer-to-consumer barrier
            const bool another_stage = b * stages + s + 1u < blocks * stages;
            check(load::needs_reuse_barrier(b, blocks, s, stages) == another_stage,
                  "uniform K64/K128 reuse barrier across K-block boundaries");
            barriers += load::needs_reuse_barrier(b, blocks, s, stages);
            ++barrier_cases;
        }
        check(barriers == 2u * blocks * stages - 1u, "exactly one fence removed");
    }
    alignas(16) unsigned char base[32] = {};
    size_t policies = 0, maps = 0;
    for (unsigned offset = 0; offset < 17; ++offset)
    for (uint64_t ts : {0ull, 4096ull, 4097ull, 4098ull, 4099ull, 1ull << 33})
    for (uint64_t gs : {0ull, 256ull, 257ull, 258ull, 259ull, 1ull << 34})
    for (uint32_t rows : {0u, 64u, 128u, 256u, 512u})
    for (unsigned flags = 0; flags < 8; ++flags) {
        const void *x = offset == 16 ? nullptr : base + offset;
        const bool disabled = flags & 1, k64 = flags & 2, k128 = flags & 4;
        const bool expected = !disabled && k64 && !k128 && x && offset == 0 &&
            ts % 4 == 0 && gs % 4 == 0 && (rows == 128 || rows == 256);
        check(load::select(disabled, k64, k128, rows, x, ts, gs) == expected,
              "alignment/staging/geometry policy");
        ++policies;
    }

    // Simulate the production float4 address schedule, retaining raw 32-bit
    // input patterns instead of converting to half. The independent reference
    // is token/column based, not the cooperative thread loop.
    constexpr uint32_t guard = 16, poison = 0xdeadbeefu;
    for (uint32_t threads : {256u, 512u})
    for (uint32_t nt = 1; nt <= 64; ++nt)
    for (uint32_t tok0 : {0u, 64u})
    for (uint32_t k0 : {0u, 192u})
    for (uint32_t groups : {1u, 3u})
    for (uint32_t padding : {0u, 4u}) {
        const uint32_t gs = 256 + padding, ts = groups * gs + padding;
        const uint32_t group = groups - 1;
        // End the allocation at the last valid element; tail tokens must
        // never issue reads, even though their LDS rows are initialized.
        const size_t count = (size_t)(tok0 + nt - 1) * ts + group * gs + k0 + 64;
        std::vector<uint32_t> input(count);
        for (size_t i = 0; i < count; ++i) input[i] = (uint32_t)i * 2654435761u;
        std::vector<uint32_t> actual(guard + 64 * 80 + guard, poison);
        std::vector<uint32_t> expected = actual;
        std::vector<unsigned> writers(64 * 80, 0);
        for (uint32_t t = 0; t < 64; ++t)
        for (uint32_t k = 0; k < 64; ++k)
            expected[guard + t * 80 + k] = t < nt
                ? input[(size_t)(tok0 + t) * ts + group * gs + k0 + k] : 0;
        for (uint32_t tid = 0; tid < threads; ++tid)
        for (uint32_t j = tid * 4; j < 64 * 64; j += threads * 4) {
            const uint32_t dst = load::destination(j);
            check(dst % 4 == 0 && dst + 3 < 64 * 80, "LDS alignment/bounds");
            uint64_t src = 0;
            const bool valid = tok0 + load::token(j) < tok0 + nt;
            if (valid) {
                src = load::source(j, tok0, group, k0, ts, gs);
                check(src % 4 == 0 && src + 3 < input.size(), "global alignment/bounds");
            }
            for (uint32_t k = 0; k < 4; ++k) {
                check(++writers[dst + k] == 1, "overlapping LDS writers");
                actual[guard + dst + k] = valid ? input[src + k] : 0;
            }
        }
        check(actual == expected, "token/K/group/tail map and canaries");
        ++maps;
    }
    // Source strides are uint64_t; no narrowing at large token/group offsets.
    check(load::source(4092u, 4096u, 7u, 192u, 1ull << 33, 1ull << 32) ==
          4159ull * (1ull << 33) + 7ull * (1ull << 32) + 252ull,
          "64-bit source addressing");
    std::printf("Q4 K64 LOAD4 host: PASS %zu policies, %zu staging maps, %zu K64/K128 barrier cases\n",
                policies, maps, barrier_cases);
}
