// SPDX-License-Identifier: MIT
// Verify the real Metal backend TLS without initializing or requiring a GPU.
#include "../ds4_metal.m"
#define DS4_GPU_EXECUTION_PHASE_TEST_NO_MAIN
#include "test_gpu_execution_phase.c"

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
int ds4_deepseek4_attention_bounds(const int *a, uint32_t b, uint32_t c,
        uint32_t d, uint32_t e, uint32_t f, uint32_t *g) {
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
    return 0;
}

int main(void) {
    @autoreleasepool {
        return ds4_test_gpu_execution_phase();
    }
}
