#include "ds4_gpu.h"

#include <inttypes.h>
#include <stdio.h>

#ifdef __APPLE__

/* ds4_metal.m references the CLI logger hook; the standalone tensor oracle
 * does not need terminal detection. */
bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

int main(void) {
    ds4_gpu_iq2_mid_only_oracle_report report;
    if (!ds4_gpu_test_iq2_addr_mid_only_oracle(&report)) {
        fprintf(stderr, "test_metal_iq2_midonly: setup/execution failed\n");
        ds4_gpu_cleanup();
        return 1;
    }

    const int pass =
        report.mid_words == 6u * 2048u &&
        report.mid_mismatches == 0 &&
        report.canonical_gate_unwritten == 0 &&
        report.canonical_up_unwritten == 0 &&
        report.candidate_gate_writes == 0 &&
        report.candidate_up_writes == 0 &&
        report.masked_mid_mismatches == 0 &&
        report.masked_inactive_writes == 0 &&
        report.masked_canonical_gate_unwritten == 0 &&
        report.masked_canonical_up_unwritten == 0 &&
        report.masked_gate_writes == 0 &&
        report.masked_up_writes == 0 &&
        report.guard_byte_mismatches == 0;
    fprintf(stderr,
            "test_metal_iq2_midonly: %s mid_words=%" PRIu64
            " mid_mismatches=%" PRIu64
            " canonical_gate_unwritten=%" PRIu64
            " canonical_up_unwritten=%" PRIu64
            " candidate_gate_writes=%" PRIu64
            " candidate_up_writes=%" PRIu64
            " masked_mid_mismatches=%" PRIu64
            " masked_inactive_writes=%" PRIu64
            " masked_canonical_gate_unwritten=%" PRIu64
            " masked_canonical_up_unwritten=%" PRIu64
            " masked_gate_writes=%" PRIu64
            " masked_up_writes=%" PRIu64
            " guard_byte_mismatches=%" PRIu64 "\n",
            pass ? "PASS" : "FAIL",
            report.mid_words,
            report.mid_mismatches,
            report.canonical_gate_unwritten,
            report.canonical_up_unwritten,
            report.candidate_gate_writes,
            report.candidate_up_writes,
            report.masked_mid_mismatches,
            report.masked_inactive_writes,
            report.masked_canonical_gate_unwritten,
            report.masked_canonical_up_unwritten,
            report.masked_gate_writes,
            report.masked_up_writes,
            report.guard_byte_mismatches);
    ds4_gpu_cleanup();
    return pass ? 0 : 1;
}

#else

int main(void) {
    fprintf(stderr, "test_metal_iq2_midonly: skipped (Metal requires macOS)\n");
    return 0;
}

#endif
