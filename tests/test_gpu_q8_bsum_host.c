// SPDX-License-Identifier: MIT
// Execute the production integer reduction with a synchronous shuffle oracle.
// This checks lane masks and arithmetic; it does not compile or time GPU code.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DS4_TEST_MASK_BITS
#define DS4_TEST_MASK_BITS 64
#endif
#if DS4_TEST_MASK_BITS == 32
#define MASK_T uint32_t
#else
#define MASK_T uint64_t
#endif
#define __device__
#define __forceinline__ inline

static int32_t stages[5][256];
static uint32_t active_tid, active_wave, shuffle_call;
static uint32_t state = 0x1879abcdu;

static void require(int valid, const char *label) {
    if (!valid) {
        fprintf(stderr, "GPU Q8_K bsum host FAIL: %s (wave=%u tid=%u step=%u)\n",
                label, active_wave, active_tid, shuffle_call);
        exit(1);
    }
}

static int32_t __shfl_down_sync(MASK_T mask, int32_t value,
                                uint32_t offset, uint32_t width) {
    const uint32_t lane = active_tid % active_wave;
    const uint32_t subgroup = lane / 16u;
    MASK_T expected_mask = 0;
    // Enumerate participants independently of the production bit shift.
    for (uint32_t member = 0; member < active_wave; ++member)
        if (member / 16u == subgroup) expected_mask |= (MASK_T)1 << member;
    require(mask == expected_mask, "exact physical-wave participant mask");
    require(width == 16u, "subgroup isolation");
    require(shuffle_call < 4u && offset == (8u >> shuffle_call), "shuffle tree");
    require(value == stages[shuffle_call][active_tid], "previous integer stage");
    const uint32_t source = lane % 16u + offset < 16u
        ? active_tid + offset : active_tid;
    return stages[shuffle_call++][source];
}

#ifdef DS4_TEST_UNDEFINED_MASK
#undef MASK_T
#endif
#include "cuda/ds4_q8_k_bsum.h"

static uint32_t next(void) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void check_block(uint32_t trial) {
    for (uint32_t tid = 0; tid < 256u; ++tid) {
        int32_t value;
        if (trial < 256u) value = (int32_t)trial - 128;
        else if (trial % 7u == 0u) value = tid % 2u ? -128 : 127;
        else if (trial % 7u == 1u) value = (int32_t)(tid / 16u) * 16 - 128;
        else if (trial % 7u == 2u) value = tid % 16u == trial % 16u ? -128 : 0;
        else value = (int32_t)(next() & 255u) - 128;
        stages[0][tid] = value;
    }
    // Model only the documented shuffle behavior. A subgroup leader's
    // result is separately checked against the old sequential byte sum.
    for (uint32_t step = 0; step < 4u; ++step) {
        const uint32_t offset = 8u >> step;
        for (uint32_t tid = 0; tid < 256u; ++tid) {
            const uint32_t source = tid % 16u + offset < 16u ? tid + offset : tid;
            stages[step + 1u][tid] = stages[step][tid] + stages[step][source];
        }
    }
    uint32_t written[16] = {0};
    for (active_tid = 0; active_tid < 256u; ++active_tid) {
        shuffle_call = 0;
        const int32_t got = ds4_q8_K_bsum16(
            stages[0][active_tid], active_tid, active_wave);
        require(shuffle_call == 4u, "four integer shuffles");
        require(got == stages[4][active_tid], "all-lane result");
        if (active_tid % 16u == 0u) {
            int32_t expected = 0;
            for (uint32_t i = 0; i < 16u; ++i)
                expected += stages[0][active_tid + i];
            require(got == expected, "canonical 16-byte sum");
            require(got >= -2048 && got <= 2032, "valid Q8_K sum bounds");
            ++written[active_tid / 16u];
        }
    }
    for (uint32_t group = 0; group < 16u; ++group)
        require(written[group] == 1u, "one writer per bsum");
}

int main(void) {
    uint32_t blocks = 0;
    for (active_wave = 32u; active_wave <= DS4_TEST_MASK_BITS; active_wave *= 2u)
        for (uint32_t trial = 0; trial < 4096u; ++trial) {
            check_block(trial);
            ++blocks;
        }
    printf("GPU Q8_K bsum host PASS: %u blocks, %u exact sums, %u-bit masks "
           "(all int8 extrema, alternating signs, subgroups and random inputs).\n",
           blocks, blocks * 16u, DS4_TEST_MASK_BITS);
    puts("Host shuffle oracle only: native CUDA/HIP parity and timing remain required.");
    return 0;
}
