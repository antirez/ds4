#define DS4_TEST_HOOKS
#include "../ds4.h"

#include <assert.h>
#include <stdio.h>

static ds4_test_dspark_readiness valid_fixture(void) {
    return (ds4_test_dspark_readiness){
        .n_stages = 3,
        .block_size = 4,
        .markov_rank = 64,
        .target_layer_count = 4,
        .has_block_size = true,
        .has_markov_rank = true,
        .has_noise_token_id = true,
        .has_target_layers = true,
    };
}

int main(void) {
    assert(ds4_test_dspark_distributed_role_allowed(
            false, DS4_DISTRIBUTED_NONE));
    assert(ds4_test_dspark_distributed_role_allowed(
            false, DS4_DISTRIBUTED_COORDINATOR));
    assert(ds4_test_dspark_distributed_role_allowed(
            true, DS4_DISTRIBUTED_NONE));
    assert(!ds4_test_dspark_distributed_role_allowed(
            true, DS4_DISTRIBUTED_COORDINATOR));
    assert(!ds4_test_dspark_distributed_role_allowed(
            true, DS4_DISTRIBUTED_WORKER));
    assert(!ds4_test_dspark_distributed_role_allowed(true, 99));

    ds4_test_dspark_readiness fixture = valid_fixture();
    assert(ds4_test_dspark_support_ready(&fixture));

    /* The parser stores this bit when the declared array exceeds its fixed
     * destination. A full destination is valid; a truncated declaration is
     * not, even when the retained prefix itself looks valid. */
    fixture.target_layer_count = 8;
    assert(ds4_test_dspark_support_ready(&fixture));
    fixture.target_layers_overflow = true;
    assert(!ds4_test_dspark_support_ready(&fixture));

    puts("test_dspark_startup_guards: PASS");
    return 0;
}
