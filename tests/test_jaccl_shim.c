#include <stdio.h>
#include <stdlib.h>
#include "../jaccl_shim.h"

int main(void) {
    printf("test_jaccl_shim: checking JACCL availability...\n");

    bool avail = jaccl_is_available();
    printf("  jaccl_is_available() = %s\n", avail ? "true" : "false");

    if (avail) {
        printf("  JACCL RDMA is available on this system.\n");

        /* Try non-strict init (returns NULL if env vars not set). */
        jaccl_group_t g = jaccl_init_from_env(false);
        if (g) {
            int rank = jaccl_group_rank(g);
            int size = jaccl_group_size(g);
            printf("  Initialized group: rank=%d, size=%d\n", rank, size);
            jaccl_group_free(g);
            printf("  Group freed successfully.\n");
        } else {
            printf("  Non-strict init returned NULL (env vars not set -- expected in unit test).\n");
        }
    } else {
        printf("  JACCL RDMA not available (no TB5 RDMA hardware or SDK < 26.2).\n");
    }

    printf("test_jaccl_shim: PASS (linkage verified)\n");
    return 0;
}
