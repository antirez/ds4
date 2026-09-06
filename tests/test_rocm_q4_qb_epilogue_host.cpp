// SPDX-License-Identifier: MIT
// CPU mapping/reduction oracle, not evidence of HIP compilation or GPU parity.
#include "rocm/ds4_rocm_q4_qb_epilogue_layout.cuh"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

namespace ep = ds4_rocm_q4_qb_epilogue;
static void check(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "FAIL Q4 epilogue host: %s\n", what); std::exit(1); }
}
static uint32_t bits(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u)); return u;
}
static float legacy(const std::array<float, 512> &x) {
#if defined(__clang__)
#pragma clang fp reassociate(off) contract(on)
#endif
    float p[256];
    for (uint32_t i = 0; i < 256; ++i) {
        p[i] = 0.0f;
        p[i] += x[i] * x[i];
        p[i] += x[i + 256] * x[i + 256];
    }
    for (uint32_t stride = 128; stride; stride >>= 1)
        for (uint32_t i = 0; i < stride; ++i) p[i] += p[i + stride];
    return p[0];
}
static float candidate(const std::array<float, 512> &x) {
#if defined(__clang__)
#pragma clang fp reassociate(off) contract(on)
#endif
    float p[32];
    for (uint32_t lane = 0; lane < 32; ++lane) {
        float v[16];
        for (uint32_t j = 0; j < 16; ++j) v[j] = x[ep::column(lane, j)];
        p[lane] = ep::fold_columns(v);
    }
    for (uint32_t stride = 16; stride; stride >>= 1)
        for (uint32_t lane = 0; lane < stride; ++lane) p[lane] += p[lane + stride];
    return p[0];
}
int main() {
    float adversarial[16];
    for (float &v : adversarial) v = 1.0f;
    adversarial[0] = 4096.0f;
    check(bits(ep::fold_columns(adversarial)) == bits(16777230.0f),
          "fast-math must not reassociate the register tree");
    size_t policies = 0;
    for (uint32_t n : {0u,1u,8u,9u,255u,256u,257u,2048u,4096u,4097u,UINT32_MAX})
    for (uint32_t h : {0u,1u,32u,64u,128u,UINT32_MAX})
    for (uint32_t d : {0u,256u,511u,512u,576u})
    for (uint32_t r : {0u,63u,64u,128u,513u})
    for (uint32_t flags = 0; flags < 16; ++flags) {
        const bool device = flags & 1, quality = flags & 2;
        const bool ssd = flags & 4, disabled = flags & 8;
        const bool expected = n >= 256 && n <= 4096 && h == 64 &&
            d == 512 && r == 64 && device && !quality && !ssd && !disabled;
        check(ep::select(n,h,d,r,device,quality,ssd,disabled) == expected, "admission");
        ++policies;
    }
    std::array<unsigned, 512> readers{}, writers{};
    for (uint32_t lane = 0; lane < 32; ++lane) {
        for (uint32_t j = 0; j < 16; ++j) ++readers[ep::column(lane,j)];
        for (uint32_t j = 0; j < 14; ++j) ++writers[ep::column(lane,j)];
        for (uint32_t component = 0; component < 2; ++component) {
            const uint32_t source = (lane & 15u) * 2u + component;
            const uint32_t slot = lane < 16 ? 14 : 15;
            check(ep::column(source,slot) == 448 + lane * 2 + component, "rotary shuffle");
            ++writers[448 + lane * 2 + component];
        }
    }
    for (uint32_t i = 0; i < 512; ++i)
        check(readers[i] == 1 && writers[i] == 1, "unique read/write ownership");

    // A symbolic tree checks ordering, not only equality on friendly numbers.
    std::string old[256], next[32];
    for (unsigned i = 0; i < 256; ++i) old[i] = "p" + std::to_string(i);
    for (unsigned lane = 0; lane < 32; ++lane) {
        std::string p[8];
        for (unsigned j = 0; j < 8; ++j) p[j] = old[ep::column(lane,j)];
        for (unsigned s = 4; s; s >>= 1)
            for (unsigned j = 0; j < s; ++j) p[j] = "(" + p[j] + "+" + p[j+s] + ")";
        next[lane] = p[0];
    }
    for (unsigned s = 128; s; s >>= 1)
        for (unsigned j = 0; j < s; ++j) old[j] = "(" + old[j] + "+" + old[j+s] + ")";
    for (unsigned s = 16; s; s >>= 1)
        for (unsigned j = 0; j < s; ++j) next[j] = "(" + next[j] + "+" + next[j+s] + ")";
    check(old[0] == next[0], "symbolic reduction tree");

    uint32_t rng = 0x95eb12a3;
    std::array<float, 512> x{};
    for (unsigned sample = 0; sample < 10000; ++sample) {
        for (unsigned i = 0; i < 512; ++i) {
            rng = rng * 1664525u + 1013904223u;
            x[i] = std::ldexp((float)((int)(rng >> 16) - 32768),
                              (int)(rng & 31u) - 30);
            if (sample < 512) x[i] = i == sample ? 1.0f : -0.0f;
            if (sample == 512) x[i] = std::ldexp(1.0f, -140);
        }
        check(bits(legacy(x)) == bits(candidate(x)), "F32 reduction bits");
    }
    std::printf("PASS Q4 epilogue host: %zu policies, unique ownership, "
                "rotary mapping, exact tree, 10000 F32 reductions. GPU unverified.\n", policies);
}
