#include "first_divergence_capture.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool copy_text(char *dst, size_t dst_size, const char *src) {
    int written;

    if (!dst || dst_size == 0) return false;
    if (!src) src = "";
    written = snprintf(dst, dst_size, "%s", src);
    return written >= 0 && (size_t)written < dst_size;
}

static bool same_key(const ds4_first_divergence_snapshot *snapshot,
                     uint32_t row,
                     uint32_t layer,
                     ds4_first_divergence_checkpoint checkpoint,
                     const char *subobject) {
    return snapshot->row == row &&
           snapshot->layer == layer &&
           snapshot->checkpoint == checkpoint &&
           strcmp(snapshot->subobject, subobject ? subobject : "") == 0;
}

static bool reserve_one(ds4_first_divergence_capture *capture) {
    ds4_first_divergence_snapshot *grown;
    size_t capacity;

    if (capture->count < capture->capacity) return true;
    capacity = capture->capacity == 0 ? 16 : capture->capacity * 2;
    if (capacity < capture->capacity ||
        capacity > SIZE_MAX / sizeof(*capture->snapshots)) {
        return false;
    }
    grown = realloc(capture->snapshots,
                    capacity * sizeof(*capture->snapshots));
    if (!grown) return false;
    capture->snapshots = grown;
    capture->capacity = capacity;
    return true;
}

static bool capture_payload(ds4_first_divergence_capture *capture,
                            uint32_t row,
                            uint32_t layer,
                            ds4_first_divergence_checkpoint checkpoint,
                            const char *subobject,
                            ds4_first_divergence_payload_kind kind,
                            const void *values,
                            size_t element_count,
                            size_t element_size) {
    ds4_first_divergence_snapshot *snapshot;
    size_t bytes;
    size_t i;

    if (!capture || checkpoint >= DS4_FIRST_DIVERGENCE_CHECKPOINT_COUNT ||
        element_size == 0 ||
        (element_count != 0 && !values) ||
        element_count > SIZE_MAX / element_size) {
        return false;
    }
    for (i = 0; i < capture->count; ++i) {
        if (same_key(&capture->snapshots[i], row, layer, checkpoint,
                     subobject)) {
            return false;
        }
    }
    if (!reserve_one(capture)) return false;

    snapshot = &capture->snapshots[capture->count];
    memset(snapshot, 0, sizeof(*snapshot));
    if (!copy_text(snapshot->subobject, sizeof(snapshot->subobject),
                   subobject)) {
        return false;
    }
    bytes = element_count * element_size;
    if (bytes != 0) {
        snapshot->data = malloc(bytes);
        if (!snapshot->data) return false;
        memcpy(snapshot->data, values, bytes);
    }
    snapshot->row = row;
    snapshot->layer = layer;
    snapshot->checkpoint = checkpoint;
    snapshot->kind = kind;
    snapshot->element_count = element_count;
    snapshot->element_size = element_size;
    capture->count++;
    return true;
}

bool ds4_first_divergence_capture_init(ds4_first_divergence_capture *capture,
                                       const char *label) {
    if (!capture) return false;
    memset(capture, 0, sizeof(*capture));
    return copy_text(capture->label, sizeof(capture->label), label);
}

void ds4_first_divergence_capture_free(ds4_first_divergence_capture *capture) {
    size_t i;

    if (!capture) return;
    for (i = 0; i < capture->count; ++i) {
        free(capture->snapshots[i].data);
    }
    free(capture->snapshots);
    memset(capture, 0, sizeof(*capture));
}

bool ds4_first_divergence_capture_f32(
        ds4_first_divergence_capture *capture,
        uint32_t row,
        uint32_t layer,
        ds4_first_divergence_checkpoint checkpoint,
        const char *subobject,
        const float *values,
        size_t element_count) {
    return capture_payload(capture, row, layer, checkpoint, subobject,
                           DS4_FIRST_DIVERGENCE_PAYLOAD_F32,
                           values, element_count, sizeof(*values));
}

bool ds4_first_divergence_capture_u32(
        ds4_first_divergence_capture *capture,
        uint32_t row,
        uint32_t layer,
        ds4_first_divergence_checkpoint checkpoint,
        const char *subobject,
        const uint32_t *values,
        size_t element_count) {
    return capture_payload(capture, row, layer, checkpoint, subobject,
                           DS4_FIRST_DIVERGENCE_PAYLOAD_U32,
                           values, element_count, sizeof(*values));
}

bool ds4_first_divergence_capture_bytes(
        ds4_first_divergence_capture *capture,
        uint32_t row,
        uint32_t layer,
        ds4_first_divergence_checkpoint checkpoint,
        const char *subobject,
        const void *values,
        size_t element_count,
        size_t element_size) {
    return capture_payload(capture, row, layer, checkpoint, subobject,
                           DS4_FIRST_DIVERGENCE_PAYLOAD_BYTES,
                           values, element_count, element_size);
}

const char *ds4_first_divergence_checkpoint_name(
        ds4_first_divergence_checkpoint checkpoint) {
    static const char *const names[] = {
        "CP1", "CP2-Q", "CP2-KV-P", "CP2-KV-R",
        "CP3-P", "CP3-F", "CP4", "CP5"
    };

    if (checkpoint >= DS4_FIRST_DIVERGENCE_CHECKPOINT_COUNT) {
        return "UNKNOWN";
    }
    return names[checkpoint];
}

bool ds4_first_divergence_run_forced_pair(
        const int *forced_tokens,
        size_t token_count,
        const ds4_first_divergence_pair_ops *ops,
        ds4_first_divergence_capture *pass_a,
        ds4_first_divergence_capture *pass_b) {
    int *immutable_tokens;
    bool pass_a_ok;
    bool restore_ok;
    size_t i;

    if (!ops || !ops->run_pass_a || !ops->restore_s0 ||
        !ops->run_pass_b_token || !pass_a || !pass_b ||
        token_count == 0 || !forced_tokens ||
        token_count > SIZE_MAX / sizeof(*immutable_tokens)) {
        return false;
    }
    immutable_tokens = malloc(token_count * sizeof(*immutable_tokens));
    if (!immutable_tokens) return false;
    memcpy(immutable_tokens, forced_tokens,
           token_count * sizeof(*immutable_tokens));

    pass_a_ok = ops->run_pass_a(ops->context, immutable_tokens,
                                token_count, pass_a);
    restore_ok = ops->restore_s0(ops->context);
    if (!pass_a_ok || !restore_ok) {
        free(immutable_tokens);
        return false;
    }
    for (i = 0; i < token_count; ++i) {
        if (!ops->run_pass_b_token(ops->context, immutable_tokens[i],
                                   (uint32_t)i, pass_b)) {
            free(immutable_tokens);
            return false;
        }
    }
    free(immutable_tokens);
    return true;
}
