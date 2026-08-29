#ifndef DS4_DSPARK_RUNTIME_H
#define DS4_DSPARK_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>

#include "ds4.h"


typedef enum {
    DS4_DSPARK_SPEC_DISABLED = 0,
    DS4_DSPARK_SPEC_LEGACY_MTP,
    DS4_DSPARK_SPEC_DSPARK_ENABLED,
    DS4_DSPARK_SPEC_DSPARK_NOT_READY,
    DS4_DSPARK_SPEC_DSPARK_NONSEQ_NOT_READY,
} ds4_dspark_spec_gate;



float ds4_dspark_bf16_to_f32(uint16_t h);
int ds4_dspark_draft_len_until_eos(const int *drafts, int draft_n, int eos_token);
int ds4_dspark_prefix_slot_for_accept(int accepted, int draft_n);
int ds4_dspark_prefix_slot_count(ds4_mtp_draft_kind kind, int block_size, int max_slots);




ds4_dspark_spec_gate ds4_dspark_speculative_gate(ds4_mtp_draft_kind kind,
                                                 bool mtp_ready,
                                                 int mtp_draft_tokens);

const char *ds4_dspark_spec_gate_reason(ds4_dspark_spec_gate gate);

#endif