/* Synthetic no-model regression for the explicit --dspark readiness gate.
 * It exercises the exact predicate used by engine startup, before any support
 * map can be registered. */

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
    /* A filename-matched candidate is not trusted until production content
     * detection classifies it. Explicit DSpark rejects legacy content both
     * with and without SSD streaming; ordinary legacy MTP remains valid. */
    assert(!ds4_test_support_kind_allowed_for_startup(
            true, false, DS4_TEST_SUPPORT_MTP_LEGACY));
    assert(!ds4_test_support_kind_allowed_for_startup(
            true, true, DS4_TEST_SUPPORT_MTP_LEGACY));
    assert(ds4_test_support_kind_allowed_for_startup(
            false, false, DS4_TEST_SUPPORT_MTP_LEGACY));
    assert(!ds4_test_support_kind_allowed_for_startup(
            false, true, DS4_TEST_SUPPORT_MTP_LEGACY));
    assert(ds4_test_support_kind_allowed_for_startup(
            true, false, DS4_TEST_SUPPORT_DSPARK));
    assert(ds4_test_support_kind_allowed_for_startup(
            true, true, DS4_TEST_SUPPORT_DSPARK));

    ds4_test_dspark_readiness fixture = valid_fixture();
    assert(ds4_test_dspark_support_ready(&fixture));

    fixture = valid_fixture();
    fixture.missing_tensors = 1;
    assert(!ds4_test_dspark_support_ready(&fixture));

    fixture = valid_fixture();
    fixture.invalid_tensors = 1;
    assert(!ds4_test_dspark_support_ready(&fixture));

    fixture = valid_fixture();
    fixture.metadata_errors = 1;
    assert(!ds4_test_dspark_support_ready(&fixture));

    fixture = valid_fixture();
    fixture.n_stages = 2;
    assert(!ds4_test_dspark_support_ready(&fixture));

    fixture = valid_fixture();
    fixture.block_size = 0;
    assert(!ds4_test_dspark_support_ready(&fixture));

    fixture = valid_fixture();
    fixture.has_target_layers = false;
    assert(!ds4_test_dspark_support_ready(&fixture));

    puts("test_dspark_readiness: PASS");
    return 0;
}
