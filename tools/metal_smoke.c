/* Compile-only Metal smoke check: surfaces MSL errors without a model. */
#include <stdio.h>
#include "ds4_gpu.h"

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "metal_smoke: ds4_gpu_init failed\n");
        return 1;
    }
    fprintf(stdout, "metal_smoke: OK\n");
    return 0;
}
