/* Exhaustive test for the ROCm IQ2_XXS sign expansion (rocm/ds4_rocm_moe.cuh
 * dev_iq2_negmask / dev_iq2_apply_negmask, and iq2_grid_signed4 in
 * cuda/mmq/vecdotq.cuh).
 *
 * The oracle follows the IQ2_XXS format directly: a 7-bit sign index s
 * selects the sign byte s | parity(s) << 7 (even parity), and grid byte j of
 * an 8-weight group is negated when bit j of that byte is set.  For every one
 * of the 256 x 128 (grid index, sign index) pairs it checks both 32-bit
 * signed weight words of the ROCm and MMQ expansions and the dev_iq2_dp4a_8
 * dot product against random int8 activations.  It also checks the property
 * the expansions rely on: every grid byte is in 1..0x7f.
 */

#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define N_GRID 256u
#define N_SIGN 128u

static uint32_t rng_state = 0x2545f491u;

static uint32_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static int signed_byte(uint64_t grid, uint32_t sign_idx, uint32_t j) {
    uint32_t parity = 0;
    for (uint32_t b = 0; b < 7u; b++) parity ^= (sign_idx >> b) & 1u;
    const uint32_t sign_byte = sign_idx | (parity << 7);
    const int g = (int)((grid >> (8u * j)) & 0xffu);
    return ((sign_byte >> j) & 1u) ? -g : g;
}

static int32_t pack_word(uint64_t grid, uint32_t sign_idx, uint32_t half) {
    uint32_t w = 0;
    for (uint32_t k = 0; k < 4u; k++) {
        w |= (uint32_t)(uint8_t)(int8_t)signed_byte(grid, sign_idx, 4u * half + k) << (8u * k);
    }
    return (int32_t)w;
}

int main(void) {
    const size_t n = (size_t)N_GRID * N_SIGN;
    int8_t q8[N_SIGN * 8u];
    uint64_t grid[N_GRID];
    int32_t *words = malloc(2u * n * sizeof(int32_t));
    int32_t *dots = malloc(n * sizeof(int32_t));
    int32_t *mmq_words = malloc(2u * n * sizeof(int32_t));
    if (!words || !dots || !mmq_words) {
        fprintf(stderr, "iq2 signs test: out of memory\n");
        return 1;
    }
    for (size_t i = 0; i < sizeof(q8); i++) q8[i] = (int8_t)(int)(rng_next() % 255u) - 127;
    if (!ds4_gpu_test_iq2_xxs_signs(q8, grid, words, dots, mmq_words)) {
        fprintf(stderr, "iq2 signs test: ds4_gpu_test_iq2_xxs_signs failed\n");
        return 1;
    }

    unsigned grid_bad = 0, word_bad = 0, mmq_bad = 0, dot_bad = 0;
    for (uint32_t g = 0; g < N_GRID; g++) {
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t b = (uint32_t)((grid[g] >> (8u * j)) & 0xffu);
            if (b == 0u || b >= 0x80u) grid_bad++;
        }
        for (uint32_t s = 0; s < N_SIGN; s++) {
            const size_t i = (size_t)g * N_SIGN + s;
            int32_t dot = 0;
            for (uint32_t j = 0; j < 8u; j++) dot += signed_byte(grid[g], s, j) * q8[8u * s + j];
            for (uint32_t h = 0; h < 2u; h++) {
                const int32_t want = pack_word(grid[g], s, h);
                if (words[2u * i + h] != want) {
                    if (word_bad++ < 4u)
                        fprintf(stderr, "rocm word mismatch: grid %u sign %u half %u: got %08x want %08x\n",
                                g, s, h, (unsigned)words[2u * i + h], (unsigned)want);
                }
                if (mmq_words[2u * i + h] != want) {
                    if (mmq_bad++ < 4u)
                        fprintf(stderr, "mmq word mismatch: grid %u sign %u half %u: got %08x want %08x\n",
                                g, s, h, (unsigned)mmq_words[2u * i + h], (unsigned)want);
                }
            }
            if (dots[i] != dot) {
                if (dot_bad++ < 4u)
                    fprintf(stderr, "dp4a mismatch: grid %u sign %u: got %d want %d\n", g, s, dots[i], dot);
            }
        }
    }
    printf("IQ2_XXS sign expansion: %zu (grid, sign) pairs, grid bytes outside 1..0x7f %u, "
           "ROCm word mismatches %u, MMQ word mismatches %u, dp4a mismatches %u\n",
           n, grid_bad, word_bad, mmq_bad, dot_bad);
    const int ok = grid_bad == 0 && word_bad == 0 && mmq_bad == 0 && dot_bad == 0;
    printf("IQ2_XXS sign expansion ROCm: %s\n", ok ? "PASS" : "FAIL");
    free(mmq_words);
    free(dots);
    free(words);
    return ok ? 0 : 1;
}
