#include "ds4_distributed.h"

#include <stdio.h>

int main(void) {
    char err[256] = "";
    int rc = ds4_dist_test_tcp_bulk_split(err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "test_distributed_transport: FAIL: %s\n",
                err[0] ? err : "unknown error");
        return 1;
    }
    fprintf(stderr,
            "test_distributed_transport: WORK/RESULT framing, mapped rejection, and v3 terminal checks passed\n");
    return 0;
}
