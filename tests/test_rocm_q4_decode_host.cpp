// SPDX-License-Identifier: MIT
// Geometry/policy/reduction model only; this is not a HIP kernel emulator.
#include "rocm/ds4_rocm_q4_decode.cuh"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace q4 = ds4_rocm_q4_decode;
static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "Q4 decode host FAIL: %s\n", what); std::exit(1); }
}
static float from_bits(uint32_t u) {
    float f; std::memcpy(&f, &u, sizeof(f)); return f;
}
static uint32_t bits(float f) {
    uint32_t u; std::memcpy(&u, &f, sizeof(u)); return u;
}
// Use one non-inlined scalar addition for both simulated shuffle widths.
// Otherwise the host may commute operands differently after vectorization,
// choosing different NaN payloads even without fast-math. This models the
// tree; it does not assert the HIP compiler's instruction/NaN behavior.
__attribute__((noinline)) static float modeled_add(float a, float b) {
    return a + b;
}
template<size_t W>
static void shuffle_add(std::array<float, W> &v, uint32_t offset) {
    const auto before = v;
    for (uint32_t i = 0u; i < W; ++i) {
        // HIP shuffle-down keeps this lane when the source exceeds width.
        v[i] = modeled_add(v[i], before[i + offset < W ? i + offset : i]);
    }
}
int main() {
    uint32_t policy_cases = 0u;
    for (uint64_t k : {0ull, 256ull, 768ull, 1024ull, 1280ull, 4096ull,
                       (1ull << 32) + 1024ull})
    for (uint64_t m : {0ull, 65ull, 32767ull, 32768ull, 32769ull,
                       (1ull << 32) + 32768ull})
    for (uint64_t n = 0u; n <= 10u; ++n)
    for (uint32_t wave : {0u, 16u, 32u, 64u, 128u})
    for (int quality = 0; quality < 2; ++quality)
    for (int e = 0; e < 2; ++e)
    for (int d = 0; d < 2; ++d)
    for (int r = 0; r < 2; ++r) {
        const bool eligible = k == 1024u && m == 32768u && n >= 1u &&
            n <= 8u && (wave == 32u || wave == 64u) && !d && !quality;
        const int expected = (e || r) && eligible ? 1 : r ? -1 : 0;
        check(q4::select(k, m, n, wave, quality, e, d, r) == expected,
              "dispatch truth table");
        ++policy_cases;
    }
    uint32_t geometry_cases = 0u;
    for (uint32_t wave : {32u, 64u})
    for (uint32_t n = 1u; n <= 8u; ++n) {
        std::vector<uint8_t> visits((size_t)n * 32768u * 4u, 0u);
        std::vector<uint8_t> writes((size_t)n * 32768u, 0u);
        for (uint32_t tok = 0u; tok < n; ++tok)
        for (uint32_t tile = 0u; tile < 512u; ++tile)
        for (uint32_t tid = 0u; tid < 256u; ++tid) {
            const uint32_t row = q4::row(tile, tid), lane = q4::lane(tid);
            const uint64_t mask = q4::mask(tid, wave);
            check(row < 32768u && lane < 4u, "row/block bounds");
            check(row == tile * 64u + tid / 4u, "row ownership");
            uint64_t expected_mask = 0u;
            for (uint32_t l = 0u; l < wave; ++l)
                if (l / 4u == (tid % wave) / 4u) expected_mask |= uint64_t(1u) << l;
            check(mask == expected_mask, "physical wave mask, including upper wave64");
            for (uint32_t offset : {2u, 1u}) {
                const uint32_t src = lane + offset < 4u ? tid + offset : tid;
                check(q4::row(tile, src) == row, "shuffle crossed row");
                check((mask & (uint64_t(1u) << (src % wave))) != 0u, "source outside mask");
            }
            ++visits[((size_t)tok * 32768u + row) * 4u + lane];
            if (lane == 0u) ++writes[(size_t)tok * 32768u + row];
        }
        for (uint8_t v : visits) check(v == 1u, "each dot visited exactly once");
        for (uint8_t v : writes) check(v == 1u, "each output written exactly once");
        ++geometry_cases;
    }
    // Inputs are per-lane accumulators after the dot, not activations sent
    // through the float-to-integer quantizer. Cover both NaN signs/payloads,
    // infinities, cancellation, subnormals and all combinations of signed zero.
    const uint32_t values[] = {0u, 0x80000000u, 0x3f800000u, 0xbf800000u,
        0x00800000u, 1u, 0x80000001u, 0x7f800000u, 0xff800000u,
        0x7fc01234u, 0xffc05678u, 0x7f801234u, 0x7f7fffffu};
    uint32_t reduction_cases = 0u;
    for (uint32_t a : values) for (uint32_t b : values)
    for (uint32_t c : values) for (uint32_t d : values) {
        std::array<float, 8> canonical = {from_bits(a), from_bits(b),
                                         from_bits(c), from_bits(d), 0, 0, 0, 0};
        std::array<float, 4> candidate;
        for (uint32_t l = 0u; l < 4u; ++l)
            candidate[l] = q4::canonical_offset4(canonical[l]);
        shuffle_add(canonical, 4u);
        shuffle_add(canonical, 2u); shuffle_add(candidate, 2u);
        shuffle_add(canonical, 1u); shuffle_add(candidate, 1u);
        if (bits(candidate[0]) != bits(canonical[0])) {
            std::fprintf(stderr, "input=%08x,%08x,%08x,%08x canonical=%08x candidate=%08x\n",
                         a, b, c, d, bits(canonical[0]), bits(candidate[0]));
            check(false, "bitwise lane0 reduction");
        }
        ++reduction_cases;
    }
    check(bits(q4::canonical_offset4(from_bits(0x80000000u))) == 0u,
          "canonical -0 + +0 must be +0 (round-to-nearest)");
    std::printf("PASS: %u policies, %u full N/wave geometries, %u reductions.\n",
                policy_cases, geometry_cases, reduction_cases);
    std::puts("Host-only: no HIP compilation, GPU parity or speedup attested.");
}
