#define DS4_TEST_HOOKS
#include "../ds4.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static int allowed(uint32_t active,
                   uint32_t limit,
                   int requested_ctx,
                   int admitted_ctx,
                   uint64_t context_scratch,
                   uint64_t admitted_context_scratch,
                   uint64_t graph,
                   uint64_t admitted_graph,
                   uint64_t speculative,
                   uint64_t admitted_speculative,
                   uint64_t workspace,
                   uint64_t admitted_workspace) {
    return ds4_test_dspark_session_request_within_admitted_ceilings(
            active,
            limit,
            requested_ctx,
            admitted_ctx,
            context_scratch,
            admitted_context_scratch,
            graph,
            admitted_graph,
            speculative,
            admitted_speculative,
            workspace,
            admitted_workspace);
}

int main(void) {
    /* Exact ceilings are admitted. */
    assert(allowed(2, 3, 4096, 4096, 400, 400, 500, 500, 700, 700, 900, 900));

    /* A fourth independently allocated session is not. */
    assert(!allowed(3, 3, 4096, 4096, 400, 400, 500, 500, 700, 700, 900, 900));
    assert(!allowed(0, 0, 4096, 4096, 400, 400, 500, 500, 700, 700, 900, 900));

    /* Context admission is bounded and never means unlimited. */
    assert(!allowed(0, 3, 4097, 4096, 400, 400, 500, 500, 700, 700, 900, 900));
    assert(!allowed(0, 3, 0, 4096, 400, 400, 500, 500, 700, 700, 900, 900));
    assert(!allowed(0, 3, 4096, 0, 400, 400, 500, 500, 700, 700, 900, 900));

    /* Graph, speculative state, and prefill workspace have independent ceilings. */
    assert(!allowed(0, 3, 4096, 4096, 401, 400, 500, 500, 700, 700, 900, 900));
    assert(!allowed(0, 3, 4096, 4096, 400, 400, 501, 500, 700, 700, 900, 900));
    assert(!allowed(0, 3, 4096, 4096, 400, 400, 500, 500, 701, 700, 900, 900));
    assert(!allowed(0, 3, 4096, 4096, 400, 400, 500, 500, 700, 700, 901, 900));

    /* Concurrent contenders get one creator; every later session observes
     * only the published workspace and its admitted maximum context. */
    for (unsigned int round = 0; round < 32; round++) {
        unsigned int owners = 0;
        unsigned int ready = 0;
        unsigned int published_cap = 0;
        assert(ds4_test_shared_prefill_workspace_claims(
                16, 2048, 4096, &owners, &ready, &published_cap));
        assert(owners == 1);
        assert(ready == 15);
        assert(published_cap == 4096);
    }
    assert(ds4_test_shared_prefill_workspace_abort_reopens());
    assert(ds4_test_dspark_small_context_admission_exact());

    /* Close starts only after a creator has reserved the engine lifecycle
     * gate; it must reject a later create and wait for the parked creator.
     * The hook uses condition-variable handshakes rather than timing. */
    for (unsigned int round = 0; round < 32; round++) {
        assert(ds4_test_session_creation_close_race());
    }

    puts("test_dspark_session_admission: PASS");
    return 0;
}
