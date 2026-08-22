/* DFlash metadata safety contract.
 *
 * The binder must reject shapes that would overflow the fixed CPU/Metal SDPA
 * accumulator, divide by zero while mapping GQA heads, index beyond KV rows,
 * break rotate-half RoPE pairs, or overflow the runtime's uint32 dimensions.
 */
#define DS4_TEST_HOOKS
#include "../ds4.h"

#include <stdint.h>
#include <stdio.h>

static int checks;
static int failures;

#define CHECK(expr, label) do {                                           \
    checks++;                                                             \
    if (!(expr)) {                                                        \
        failures++;                                                       \
        fprintf(stderr, "FAIL: %s\n", label);                            \
    }                                                                     \
} while (0)

int main(void) {
    CHECK(ds4_test_dflash_attention_shape_valid(32, 8, 128),
          "standard GQA shape accepted");
    CHECK(ds4_test_dflash_attention_shape_valid(1, 1, 2),
          "small paired shape accepted");

    CHECK(!ds4_test_dflash_attention_shape_valid(32, 8, 130),
          "head_dim above 128 rejected");
    CHECK(!ds4_test_dflash_attention_shape_valid(32, 8, 127),
          "odd head_dim rejected");
    CHECK(!ds4_test_dflash_attention_shape_valid(8, 16, 128),
          "more KV than query heads rejected");
    CHECK(!ds4_test_dflash_attention_shape_valid(10, 3, 128),
          "non-divisible GQA rejected");
    CHECK(!ds4_test_dflash_attention_shape_valid(0, 0, 0),
          "zero dimensions rejected");
    CHECK(!ds4_test_dflash_attention_shape_valid(UINT32_MAX, 1, 128),
          "q_dim overflow rejected");

    fprintf(stderr, "test_dflash_shape: %d/%d checks passed\n",
            checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
