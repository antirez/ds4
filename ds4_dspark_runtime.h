#ifndef DS4_DSPARK_RUNTIME_H
#define DS4_DSPARK_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>

#include "ds4.h"


typedef enum {
    DS4_DSPARK_SPEC_DISABLED = 0,
    DS4_DSPARK_SPEC_LEGACY_MTP,
    DS4_DSPARK_SPEC_DSPARK_NOT_READY,
} ds4_dspark_spec_gate;




ds4_dspark_spec_gate ds4_dspark_speculative_gate(ds4_mtp_draft_kind kind,
                                                 bool mtp_ready,
                                                 int mtp_draft_tokens);

const char *ds4_dspark_spec_gate_reason(ds4_dspark_spec_gate gate);

#endif