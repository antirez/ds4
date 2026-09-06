// SPDX-License-Identifier: MIT
// Run the real Q8_K fold and shuffle body against the former shared tree.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DS4_TEST_MASK_BITS
#define DS4_TEST_MASK_BITS 64
#endif
#if DS4_TEST_MASK_BITS == 64
typedef uint64_t test_mask;
#define MASK_T uint64_t
#else
typedef uint32_t test_mask;
#ifndef DS4_TEST_UNDEFINED_MASK
#define MASK_T uint32_t
#endif
#endif

static float stage_magnitude[6][32];
static uint32_t stage_index[6][32];
static uint32_t active_lane, calls, active_wave, cases, seed = 0x81794u;

static uint32_t bits(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof(u));
    return u;
}

static float from_bits(uint32_t u) {
    float v;
    memcpy(&v, &u, sizeof(v));
    return v;
}

static void require(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "Q8_K max FAIL: %s, case=%u wave=%u lane=%u call=%u\n",
                what, cases, active_wave, active_lane, calls);
        exit(1);
    }
}

static uint32_t source_lane(test_mask mask, uint32_t offset, uint32_t width) {
    const uint32_t step = calls / 2u;
    test_mask expected = 0;
    for (uint32_t lane = 0; lane < active_wave; ++lane)
        if (lane < 32u) expected |= (test_mask)1u << lane;
    require(mask == expected, "only the first 32 physical lanes participate");
    require(width == 32u && step < 5u && offset == (16u >> step), "shuffle tree");
    return active_lane + offset < 32u ? active_lane + offset : active_lane;
}

static float shuffle_float(test_mask mask, float v, uint32_t offset, uint32_t width) {
    const uint32_t source = source_lane(mask, offset, width);
    const uint32_t step = calls / 2u;
    require((calls & 1u) == 0u, "magnitude before index");
    require(bits(v) == bits(stage_magnitude[step][active_lane]), "magnitude operand");
    ++calls;
    return stage_magnitude[step][source];
}

static uint32_t shuffle_index(test_mask mask, uint32_t v, uint32_t offset, uint32_t width) {
    const uint32_t source = source_lane(mask, offset, width);
    const uint32_t step = calls / 2u;
    require((calls & 1u) == 1u, "index after magnitude");
    require(v == stage_index[step][active_lane], "index operand");
    ++calls;
    return stage_index[step][source];
}

#define __CUDACC__
#define __host__
#define __device__
#define __forceinline__ inline
#define __shfl_down_sync(mask, v, offset, width) \
    _Generic((v), float: shuffle_float, uint32_t: shuffle_index)(mask, v, offset, width)
#include "cuda/ds4_q8_k_reduce.h"

static void check(const float *values) {
    float magnitudes[256], legacy[256];
    uint32_t indices[256];
    for (uint32_t i = 0; i < 256u; ++i) {
        // Bitwise fabs also preserves NaN payloads for the reduction-only
        // oracle; no non-finite input is passed to integer quantization.
        magnitudes[i] = legacy[i] = from_bits(bits(values[i]) & 0x7fffffffu);
        indices[i] = i;
    }
    // Independent implementation of the exact former shared-memory loop.
    for (uint32_t stride = 128u; stride != 0u; stride >>= 1u) {
        for (uint32_t i = 0; i < stride; ++i) {
            if (legacy[i + stride] > legacy[i]) {
                legacy[i] = legacy[i + stride];
                indices[i] = indices[i + stride];
            }
        }
        const uint32_t step = stride == 32u ? 0u : stride == 16u ? 1u :
            stride == 8u ? 2u : stride == 4u ? 3u : stride == 2u ? 4u : 5u;
        if (stride <= 32u) {
            memcpy(stage_magnitude[step], legacy, sizeof(stage_magnitude[step]));
            memcpy(stage_index[step], indices, sizeof(stage_index[step]));
        }
    }
    for (active_lane = 0; active_lane < 32u; ++active_lane) {
        const ds4_q8_K_maximum fold = ds4_q8_K_max_fold(magnitudes, active_lane);
        require(bits(fold.magnitude) == bits(stage_magnitude[0][active_lane]) &&
                fold.index == stage_index[0][active_lane], "stride-128/64/32 fold");
        calls = 0u;
        const ds4_q8_K_maximum got = ds4_q8_K_max_first_warp(magnitudes, active_lane);
        require(calls == 10u, "five paired shuffles");
        require(got.index == stage_index[5][active_lane] &&
                bits(got.magnitude) == bits(stage_magnitude[5][active_lane]),
                "all-lane result matches shared tree");
        if (active_lane == 0u)
            require(bits(values[got.index]) == bits(values[indices[0]]),
                    "signed maximum and payload preserved");
    }
    ++cases;
}

static uint32_t next(void) {
    seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u;
    return seed;
}

static void run(void) {
    float values[256] = {0};
    // Opposite-sign maxima at every pair of source positions. The original
    // tree prefers bit-reversed traversal order, not the lowest source index.
    for (uint32_t a = 0; a < 256u; ++a)
        for (uint32_t b = a + 1u; b < 256u; ++b)
            for (uint32_t sign = 0; sign < 2u; ++sign) {
                values[a] = sign ? -1.0f : 1.0f;
                values[b] = -values[a];
                check(values);
                values[a] = values[b] = 0.0f;
            }
    const uint32_t special[] = {
        0, 0x80000000u, 1, 0x807fffffu, 0x00800000u, 0x7f7fffffu,
        0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc54321u,
    };
    for (uint32_t kind = 0; kind < sizeof(special)/sizeof(special[0]); ++kind) {
        for (uint32_t i = 0; i < 256u; ++i) values[i] = from_bits(special[kind]);
        check(values);
        for (uint32_t at = 0; at < 256u; ++at) {
            for (uint32_t i = 0; i < 256u; ++i) values[i] = (i & 1u) ? -0.5f : 0.5f;
            values[at] = from_bits(special[kind]);
            check(values);
        }
    }
    for (uint32_t trial = 0; trial < 4096u; ++trial) {
        for (uint32_t i = 0; i < 256u; ++i) values[i] = from_bits(next());
        check(values);
    }
}

int main(void) {
    for (active_wave = 32u; active_wave <= DS4_TEST_MASK_BITS; active_wave *= 2u)
        run();
    printf("GPU Q8_K maximum host PASS: %u blocks, %u-bit masks; exhaustive "
           "opposite-sign ties, signed zero, subnormals, infinities, NaNs, random bits.\n",
           cases, DS4_TEST_MASK_BITS);
    puts("Reduction-only host oracle; native GPU compilation, parity and timing remain required.");
    return 0;
}
