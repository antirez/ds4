#include "first_divergence_capture.h"
#include "ds4_float_compare.h"

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

static int cp3_subobject_rank(const char *name) {
    static const char *const ordered[] = {
        "attn_state_kv",
        "attn_state_score",
        "layer_n_comp",
        "attn_cache",
        "index_state_kv",
        "index_state_score",
        "layer_n_index_comp",
        "index_cache"
    };
    size_t i;

    for (i = 0; i < sizeof(ordered) / sizeof(ordered[0]); ++i) {
        if (strcmp(name, ordered[i]) == 0) return (int)i;
    }
    return (int)(sizeof(ordered) / sizeof(ordered[0]));
}

static int snapshot_order(const ds4_first_divergence_snapshot *a,
                          const ds4_first_divergence_snapshot *b) {
    int a_rank;
    int b_rank;

    if (a->row != b->row) return a->row < b->row ? -1 : 1;
    if (a->layer != b->layer) return a->layer < b->layer ? -1 : 1;
    if (a->checkpoint != b->checkpoint) {
        return a->checkpoint < b->checkpoint ? -1 : 1;
    }
    if (a->checkpoint == DS4_FIRST_DIVERGENCE_CP3_F) {
        a_rank = cp3_subobject_rank(a->subobject);
        b_rank = cp3_subobject_rank(b->subobject);
        if (a_rank != b_rank) return a_rank < b_rank ? -1 : 1;
    }
    return strcmp(a->subobject, b->subobject);
}

static int snapshot_pointer_order(const void *lhs, const void *rhs) {
    const ds4_first_divergence_snapshot *const *a = lhs;
    const ds4_first_divergence_snapshot *const *b = rhs;
    return snapshot_order(*a, *b);
}

static const ds4_first_divergence_snapshot *find_snapshot(
        const ds4_first_divergence_capture *capture,
        const ds4_first_divergence_snapshot *key) {
    size_t i;

    for (i = 0; i < capture->count; ++i) {
        if (snapshot_order(&capture->snapshots[i], key) == 0) {
            return &capture->snapshots[i];
        }
    }
    return NULL;
}

static const ds4_first_divergence_snapshot *find_snapshot_fields(
        const ds4_first_divergence_capture *capture,
        uint32_t row,
        uint32_t layer,
        ds4_first_divergence_checkpoint checkpoint,
        const char *subobject) {
    size_t i;

    if (!capture) return NULL;
    for (i = 0; i < capture->count; ++i) {
        if (same_key(&capture->snapshots[i], row, layer, checkpoint,
                     subobject)) {
            return &capture->snapshots[i];
        }
    }
    return NULL;
}

static bool same_layout(const ds4_first_divergence_snapshot *a,
                        const ds4_first_divergence_snapshot *b) {
    return a->kind == b->kind &&
           a->element_count == b->element_count &&
           a->element_size == b->element_size;
}

static void compare_raw_elements(const ds4_first_divergence_snapshot *a,
                                 const ds4_first_divergence_snapshot *b,
                                 size_t *mismatch_count,
                                 size_t *first_mismatch) {
    size_t i;

    *mismatch_count = 0;
    *first_mismatch = SIZE_MAX;
    for (i = 0; i < a->element_count; ++i) {
        const size_t offset = i * a->element_size;
        if (memcmp(a->data + offset, b->data + offset,
                   a->element_size) != 0) {
            if (*first_mismatch == SIZE_MAX) *first_mismatch = i;
            (*mismatch_count)++;
        }
    }
}

static void print_raw(FILE *stream, const unsigned char *data, size_t bytes) {
    size_t i;

    fputs("0x", stream);
    for (i = bytes; i > 0; --i) {
        fprintf(stream, "%02x", data[i - 1]);
    }
}

static void remember_first(ds4_first_divergence_report *report,
                           const ds4_first_divergence_snapshot *key) {
    if (report->first_divergence_found) return;
    report->first_divergence_found = true;
    report->row = key->row;
    report->layer = key->layer;
    report->checkpoint = key->checkpoint;
    (void)copy_text(report->subobject, sizeof(report->subobject),
                    key->subobject);
}

static void print_object_prefix(FILE *stream,
                                const ds4_first_divergence_snapshot *key) {
    fprintf(stream,
            "FIRST_DIVERGENCE_OBJECT row=%u layer=%u checkpoint=%s subobject=%s",
            key->row,
            key->layer,
            ds4_first_divergence_checkpoint_name(key->checkpoint),
            key->subobject[0] ? key->subobject : "-");
}

static void print_float_metrics(FILE *stream,
                                const ds4_float_compare_result *comparison) {
    fprintf(stream,
            " elements=%zu mismatch_count=%zu first_index=%zu actual_bits=0x%08x expected_bits=0x%08x",
            comparison->length, comparison->mismatch_count,
            comparison->first_mismatch_index,
            comparison->first_actual_bits,
            comparison->first_expected_bits);
    if (comparison->max_abs_diff_defined) {
        fprintf(stream, " max_abs=%.17g", comparison->max_abs_diff);
    } else {
        fputs(" max_abs=undefined", stream);
    }
    if (comparison->max_rel_diff_defined) {
        fprintf(stream, " max_rel=%.17g", comparison->max_rel_diff);
    } else {
        fputs(" max_rel=undefined", stream);
    }
    if (comparison->max_ulp_distance_defined) {
        fprintf(stream, " max_ulp=%u", comparison->max_ulp_distance);
    } else {
        fputs(" max_ulp=undefined", stream);
    }
}

bool ds4_first_divergence_emit_q_trace(
        const ds4_first_divergence_capture *pass_a,
        const ds4_first_divergence_capture *pass_b,
        FILE *stream) {
    const ds4_first_divergence_snapshot *cp1_a;
    const ds4_first_divergence_snapshot *cp1_b;
    const ds4_first_divergence_snapshot *qr_a;
    const ds4_first_divergence_snapshot *qr_b;
    ds4_float_compare_result cp1_comparison;
    ds4_float_compare_result qr_comparison;

    if (!pass_a || !pass_b || !stream) return false;
    cp1_a = find_snapshot_fields(pass_a, 0, 0, DS4_FIRST_DIVERGENCE_CP1,
                                 "attn_norm");
    cp1_b = find_snapshot_fields(pass_b, 0, 0, DS4_FIRST_DIVERGENCE_CP1,
                                 "attn_norm");
    qr_a = find_snapshot_fields(pass_a, 0, 0, DS4_FIRST_DIVERGENCE_CP2_Q,
                                "qr");
    qr_b = find_snapshot_fields(pass_b, 0, 0, DS4_FIRST_DIVERGENCE_CP2_Q,
                                "qr");

    fputs("Q_DIVERGENCE_TRACE row=0 layer=0\n", stream);
    if (!cp1_a || !cp1_b || !qr_a || !qr_b) {
        fputs("Q_DIVERGENCE_SANITY FAIL reason=missing_required_snapshot\n",
              stream);
        return false;
    }
    if (!same_layout(cp1_a, cp1_b) ||
        cp1_a->kind != DS4_FIRST_DIVERGENCE_PAYLOAD_F32 ||
        !same_layout(qr_a, qr_b) ||
        qr_a->kind != DS4_FIRST_DIVERGENCE_PAYLOAD_F32) {
        fputs("Q_DIVERGENCE_SANITY FAIL reason=non_comparable_layout\n",
              stream);
        return false;
    }
    if (!ds4_float_compare_exact((const float *)cp1_a->data,
                                 (const float *)cp1_b->data,
                                 cp1_a->element_count, &cp1_comparison) ||
        !ds4_float_compare_exact((const float *)qr_a->data,
                                 (const float *)qr_b->data,
                                 qr_a->element_count, &qr_comparison)) {
        fputs("Q_DIVERGENCE_SANITY FAIL reason=compare_error\n", stream);
        return false;
    }

    fprintf(stream,
            "Q_DIVERGENCE_STAGE stage=CP1 semantic=normalized_attention_input subobject=attn_norm result=%s elements=%zu\n",
            cp1_comparison.bit_exact ? "EXACT" : "MISMATCH",
            cp1_comparison.length);
    if (!cp1_comparison.bit_exact) {
        fputs("Q_DIVERGENCE_SANITY FAIL reason=cp1_not_exact\n", stream);
        return false;
    }
    fputs("Q_DIVERGENCE_STAGE stage=CP2-Q semantic=q_a_projection_output subobject=qr result=",
          stream);
    if (qr_comparison.bit_exact) {
        fprintf(stream, "EXACT elements=%zu\n", qr_comparison.length);
        fputs("Q_DIVERGENCE_SANITY FAIL reason=q_a_projection_exact\n", stream);
        return false;
    }
    fputs("MISMATCH", stream);
    print_float_metrics(stream, &qr_comparison);
    fputc('\n', stream);
    fputs("Q_FIRST_DIVERGENCE stage=q_a_projection_output producer_generic=metal_graph_matmul_q8_0_named_tensor_attn_q_a producer_sequential=metal_graph_matmul_dense_quant_tensor_attn_q_a subobject=qr",
          stream);
    print_float_metrics(stream, &qr_comparison);
    fputc('\n', stream);
    fputs("Q_LOCALIZATION_RESULT FIRST_RUNTIME_DIVERGENCE_WITHIN_Q_PATH\n",
          stream);
    return ferror(stream) == 0;
}

bool ds4_first_divergence_emit_report(
        const ds4_first_divergence_capture *pass_a,
        const ds4_first_divergence_capture *pass_b,
        FILE *stream,
        ds4_first_divergence_report *report) {
    const ds4_first_divergence_snapshot **ordered;
    size_t ordered_count;
    size_t i;

    if (!pass_a || !pass_b || !stream || !report ||
        pass_a->count > SIZE_MAX - pass_b->count) {
        return false;
    }
    memset(report, 0, sizeof(*report));
    report->bit_exact = true;
    ordered_count = pass_a->count + pass_b->count;
    if (ordered_count == 0) {
        fputs("FIRST_DIVERGENCE NONE compared_objects=0\n", stream);
        return ferror(stream) == 0;
    }
    if (ordered_count > SIZE_MAX / sizeof(*ordered)) return false;
    ordered = malloc(ordered_count * sizeof(*ordered));
    if (!ordered) return false;
    for (i = 0; i < pass_a->count; ++i) ordered[i] = &pass_a->snapshots[i];
    for (i = 0; i < pass_b->count; ++i) {
        ordered[pass_a->count + i] = &pass_b->snapshots[i];
    }
    qsort(ordered, ordered_count, sizeof(*ordered), snapshot_pointer_order);

    for (i = 0; i < ordered_count; ++i) {
        const ds4_first_divergence_snapshot *key = ordered[i];
        const ds4_first_divergence_snapshot *a;
        const ds4_first_divergence_snapshot *b;

        if (i != 0 && snapshot_order(ordered[i - 1], key) == 0) continue;
        a = find_snapshot(pass_a, key);
        b = find_snapshot(pass_b, key);
        report->compared_objects++;
        print_object_prefix(stream, key);

        if (!a || !b) {
            report->bit_exact = false;
            remember_first(report, key);
            fprintf(stream, " result=MISMATCH reason=missing_%s\n",
                    a ? "PASS_B" : "PASS_A");
            continue;
        }
        if (!same_layout(a, b)) {
            report->bit_exact = false;
            remember_first(report, key);
            fprintf(stream,
                    " result=MISMATCH reason=layout pass_a_kind=%u pass_b_kind=%u pass_a_elements=%zu pass_b_elements=%zu pass_a_element_size=%zu pass_b_element_size=%zu\n",
                    (unsigned)a->kind, (unsigned)b->kind,
                    a->element_count, b->element_count,
                    a->element_size, b->element_size);
            continue;
        }

        if (a->kind == DS4_FIRST_DIVERGENCE_PAYLOAD_F32) {
            ds4_float_compare_result comparison;

            if (!ds4_float_compare_exact((const float *)a->data,
                                         (const float *)b->data,
                                         a->element_count, &comparison)) {
                free(ordered);
                return false;
            }
            if (comparison.bit_exact) {
                fprintf(stream, " result=EXACT elements=%zu\n",
                        a->element_count);
                continue;
            }
            report->bit_exact = false;
            remember_first(report, key);
            fprintf(stream,
                    " result=MISMATCH elements=%zu mismatch_count=%zu first_index=%zu actual_bits=0x%08x expected_bits=0x%08x",
                    comparison.length, comparison.mismatch_count,
                    comparison.first_mismatch_index,
                    comparison.first_actual_bits,
                    comparison.first_expected_bits);
            if (comparison.max_abs_diff_defined) {
                fprintf(stream, " max_abs=%.17g", comparison.max_abs_diff);
            } else {
                fputs(" max_abs=undefined", stream);
            }
            if (comparison.max_rel_diff_defined) {
                fprintf(stream, " max_rel=%.17g", comparison.max_rel_diff);
            } else {
                fputs(" max_rel=undefined", stream);
            }
            if (comparison.max_ulp_distance_defined) {
                fprintf(stream, " max_ulp=%u", comparison.max_ulp_distance);
            } else {
                fputs(" max_ulp=undefined", stream);
            }
            fputc('\n', stream);
        } else {
            size_t mismatch_count;
            size_t first_mismatch;

            compare_raw_elements(a, b, &mismatch_count, &first_mismatch);
            if (mismatch_count == 0) {
                fprintf(stream, " result=EXACT elements=%zu\n",
                        a->element_count);
                continue;
            }
            report->bit_exact = false;
            remember_first(report, key);
            fprintf(stream,
                    " result=MISMATCH elements=%zu mismatch_count=%zu first_index=%zu actual_raw=",
                    a->element_count, mismatch_count, first_mismatch);
            print_raw(stream, a->data + first_mismatch * a->element_size,
                      a->element_size);
            fputs(" expected_raw=", stream);
            print_raw(stream, b->data + first_mismatch * b->element_size,
                      b->element_size);
            fputs(" max_abs=undefined max_rel=undefined max_ulp=undefined\n",
                  stream);
        }
    }
    free(ordered);

    if (report->first_divergence_found) {
        fprintf(stream,
                "FIRST_DIVERGENCE row=%u layer=%u checkpoint=%s subobject=%s compared_objects=%zu\n",
                report->row,
                report->layer,
                ds4_first_divergence_checkpoint_name(report->checkpoint),
                report->subobject[0] ? report->subobject : "-",
                report->compared_objects);
    } else {
        fprintf(stream, "FIRST_DIVERGENCE NONE compared_objects=%zu\n",
                report->compared_objects);
    }
    if (report->first_divergence_found &&
        report->row == 0 && report->layer == 0 &&
        report->checkpoint == DS4_FIRST_DIVERGENCE_CP2_Q &&
        strcmp(report->subobject, "qr") == 0 &&
        !ds4_first_divergence_emit_q_trace(pass_a, pass_b, stream)) {
        return false;
    }
    return ferror(stream) == 0;
}
