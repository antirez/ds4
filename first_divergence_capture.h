#ifndef DS4_FIRST_DIVERGENCE_CAPTURE_H
#define DS4_FIRST_DIVERGENCE_CAPTURE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define DS4_FIRST_DIVERGENCE_LABEL_MAX 16
#define DS4_FIRST_DIVERGENCE_SUBOBJECT_MAX 48

typedef enum {
    DS4_FIRST_DIVERGENCE_CP1 = 0,
    DS4_FIRST_DIVERGENCE_CP2_Q,
    DS4_FIRST_DIVERGENCE_CP2_KV_P,
    DS4_FIRST_DIVERGENCE_CP2_KV_R,
    DS4_FIRST_DIVERGENCE_CP3_P,
    DS4_FIRST_DIVERGENCE_CP3_F,
    DS4_FIRST_DIVERGENCE_CP4,
    DS4_FIRST_DIVERGENCE_CP5,
    DS4_FIRST_DIVERGENCE_CHECKPOINT_COUNT
} ds4_first_divergence_checkpoint;

typedef enum {
    DS4_FIRST_DIVERGENCE_PAYLOAD_F32 = 0,
    DS4_FIRST_DIVERGENCE_PAYLOAD_U32,
    DS4_FIRST_DIVERGENCE_PAYLOAD_BYTES
} ds4_first_divergence_payload_kind;

typedef struct {
    uint32_t row;
    uint32_t layer;
    ds4_first_divergence_checkpoint checkpoint;
    char subobject[DS4_FIRST_DIVERGENCE_SUBOBJECT_MAX];
    ds4_first_divergence_payload_kind kind;
    size_t element_count;
    size_t element_size;
    unsigned char *data;
} ds4_first_divergence_snapshot;

typedef struct {
    char label[DS4_FIRST_DIVERGENCE_LABEL_MAX];
    ds4_first_divergence_snapshot *snapshots;
    size_t count;
    size_t capacity;
} ds4_first_divergence_capture;

typedef bool (*ds4_first_divergence_pass_a_fn)(
        void *context,
        const int *forced_tokens,
        size_t token_count,
        ds4_first_divergence_capture *capture);

typedef bool (*ds4_first_divergence_restore_fn)(void *context);

typedef bool (*ds4_first_divergence_pass_b_token_fn)(
        void *context,
        int forced_token,
        uint32_t row,
        ds4_first_divergence_capture *capture);

typedef struct {
    void *context;
    ds4_first_divergence_pass_a_fn run_pass_a;
    ds4_first_divergence_restore_fn restore_s0;
    ds4_first_divergence_pass_b_token_fn run_pass_b_token;
} ds4_first_divergence_pair_ops;

typedef struct {
    bool bit_exact;
    size_t compared_objects;
    bool first_divergence_found;
    uint32_t row;
    uint32_t layer;
    ds4_first_divergence_checkpoint checkpoint;
    char subobject[DS4_FIRST_DIVERGENCE_SUBOBJECT_MAX];
} ds4_first_divergence_report;

bool ds4_first_divergence_capture_init(ds4_first_divergence_capture *capture,
                                       const char *label);
void ds4_first_divergence_capture_free(ds4_first_divergence_capture *capture);

bool ds4_first_divergence_capture_f32(
        ds4_first_divergence_capture *capture,
        uint32_t row,
        uint32_t layer,
        ds4_first_divergence_checkpoint checkpoint,
        const char *subobject,
        const float *values,
        size_t element_count);

bool ds4_first_divergence_capture_u32(
        ds4_first_divergence_capture *capture,
        uint32_t row,
        uint32_t layer,
        ds4_first_divergence_checkpoint checkpoint,
        const char *subobject,
        const uint32_t *values,
        size_t element_count);

bool ds4_first_divergence_capture_bytes(
        ds4_first_divergence_capture *capture,
        uint32_t row,
        uint32_t layer,
        ds4_first_divergence_checkpoint checkpoint,
        const char *subobject,
        const void *values,
        size_t element_count,
        size_t element_size);

const char *ds4_first_divergence_checkpoint_name(
        ds4_first_divergence_checkpoint checkpoint);

bool ds4_first_divergence_run_forced_pair(
        const int *forced_tokens,
        size_t token_count,
        const ds4_first_divergence_pair_ops *ops,
        ds4_first_divergence_capture *pass_a,
        ds4_first_divergence_capture *pass_b);

bool ds4_first_divergence_emit_report(
        const ds4_first_divergence_capture *pass_a,
        const ds4_first_divergence_capture *pass_b,
        FILE *stream,
        ds4_first_divergence_report *report);

bool ds4_first_divergence_emit_q_trace(
        const ds4_first_divergence_capture *pass_a,
        const ds4_first_divergence_capture *pass_b,
        FILE *stream,
        bool *q_projection_exact);

bool ds4_first_divergence_emit_qa_canonical_summary(
        const ds4_first_divergence_report *report,
        FILE *stream);

#endif
