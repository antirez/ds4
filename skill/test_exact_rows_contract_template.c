/*
 * Exact-row kernel contract skeleton.
 *
 * This is intentionally NOT wired to a hypothetical DSpark kernel yet.
 * HY should replace the two adapter functions below after locating the
 * canonical single-row projection API and implementing the candidate
 * exact-row API on current main.
 *
 * Contract:
 *   candidate(n_rows=N) must equal N repeated canonical single-row executions
 *   bit-for-bit for N=1..5.
 */

#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 5u

static int run_reference_serial(
        ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src,
        uint32_t rows,
        uint32_t in_dim,
        uint32_t out_dim) {
    (void)dst;
    (void)src;
    (void)rows;
    (void)in_dim;
    (void)out_dim;

    /*
     * TODO:
     * for row in [0, rows):
     *   create row views
     *   call the SAME canonical projection/store path used by ordinary
     *   single-token decode.
     *
     * Do not substitute a generic GEMM reference.
     */
    return 0;
}

static int run_candidate_exact_rows(
        ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src,
        uint32_t rows,
        uint32_t in_dim,
        uint32_t out_dim) {
    (void)dst;
    (void)src;
    (void)rows;
    (void)in_dim;
    (void)out_dim;

    /*
     * TODO:
     * one dispatch may process several independent rows, but arithmetic
     * WITHIN EACH ROW must preserve the canonical single-token reduction
     * order.
     */
    return 0;
}

static int compare_bit_exact_f32(
        const float *a,
        const float *b,
        uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            uint32_t ab = 0, bb = 0;
            memcpy(&ab, &a[i], sizeof(ab));
            memcpy(&bb, &b[i], sizeof(bb));
            fprintf(stderr,
                    "exact-row mismatch index=%llu actual=0x%08x expected=0x%08x\n",
                    (unsigned long long)i, ab, bb);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    /*
     * TODO:
     * 1. initialize GPU
     * 2. build deterministic synthetic input/weights
     * 3. allocate reference/candidate outputs
     * 4. for rows=1..5:
     *      clear outputs
     *      run_reference_serial(...)
     *      read result
     *      clear outputs
     *      run_candidate_exact_rows(...)
     *      read result
     *      require bit equality
     *
     * Prefer synthetic fixtures so this test remains independent of a
     * 90+ GiB GGUF and can run in CI.
     */
    fprintf(stderr,
            "test_exact_rows_contract: scaffold only; wire current-main APIs first\n");
    return 77; /* deliberate SKIP-style value until adapters are implemented */
}
