#include "ds4_dspark_runtime.h"

#include <string.h>


float ds4_dspark_bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}
int ds4_dspark_draft_len_until_eos(const int *drafts, int draft_n, int eos_token) {
    if (!drafts || draft_n <= 0) return 0;
    for (int i = 0; i < draft_n; i++) {
        if (drafts[i] == eos_token) return i + 1;
    }
    return draft_n;
}
int ds4_dspark_prefix_slot_for_accept(int accepted, int draft_n) {
    if (accepted <= 0 || draft_n <= 1 || accepted >= draft_n) return -1;
    return accepted - 1;
}

int ds4_dspark_prefix_slot_count(ds4_mtp_draft_kind kind, int block_size, int max_slots) {
    if (max_slots <= 0) return 0;
    if (kind != DS4_MTP_DRAFT_LEGACY &&
        kind != DS4_MTP_DRAFT_DSPARK &&
        kind != DS4_MTP_DRAFT_DSPARK_NONSEQ) {
        return 0;
    }
    int slots = 1;
    if (kind == DS4_MTP_DRAFT_DSPARK || kind == DS4_MTP_DRAFT_DSPARK_NONSEQ) {
        slots = block_size > 1 ? block_size - 1 : 1;
    }
    if (slots > max_slots) slots = max_slots;
    return slots;
}






ds4_dspark_spec_gate ds4_dspark_speculative_gate(ds4_mtp_draft_kind kind,
                                                 bool mtp_ready,
                                                 int mtp_draft_tokens) {
    if (!mtp_ready || mtp_draft_tokens <= 1) return DS4_DSPARK_SPEC_DISABLED;
    if (kind == DS4_MTP_DRAFT_LEGACY) return DS4_DSPARK_SPEC_LEGACY_MTP;
    if (kind == DS4_MTP_DRAFT_DSPARK ||
        kind == DS4_MTP_DRAFT_DSPARK_NONSEQ) return DS4_DSPARK_SPEC_DSPARK_ENABLED;
    return DS4_DSPARK_SPEC_DISABLED;
}

const char *ds4_dspark_spec_gate_reason(ds4_dspark_spec_gate gate) {
    switch (gate) {
    case DS4_DSPARK_SPEC_LEGACY_MTP:
        return "legacy MTP draft path (DSpark block draft not engaged)";
    case DS4_DSPARK_SPEC_DSPARK_ENABLED:
        return "DSpark block speculative decode enabled";
    case DS4_DSPARK_SPEC_DSPARK_NOT_READY:
        return "DSpark draft graph has not been validated on real DSpark GGUF weights; "
               "speculative decode stays off (no fake draft tokens)";
    case DS4_DSPARK_SPEC_DSPARK_NONSEQ_NOT_READY:
        return "DSpark nonseq draft head has not been validated on real trained DSpark GGUF weights; "
               "speculative decode stays off (no fake draft tokens)";
    case DS4_DSPARK_SPEC_DISABLED:
    default:
        return "speculative draft disabled";
    }
}