#include "first_divergence_capture.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

typedef struct {
    const int *expected_tokens;
    size_t token_count;
    size_t pass_b_calls;
    size_t restore_calls;
    int mutable_state;
} pair_fixture;

static bool run_pass_a(void *context,
                       const int *forced_tokens,
                       size_t token_count,
                       ds4_first_divergence_capture *capture) {
    pair_fixture *fixture = context;
    size_t row;

    if (token_count != fixture->token_count ||
        memcmp(forced_tokens, fixture->expected_tokens,
               token_count * sizeof(*forced_tokens)) != 0) {
        return false;
    }
    fixture->mutable_state = 1;
    for (row = 0; row < token_count; ++row) {
        const float value = (float)(forced_tokens[row] + (int)row);
        if (!ds4_first_divergence_capture_f32(
                capture, (uint32_t)row, 0,
                DS4_FIRST_DIVERGENCE_CP1, "attn_norm", &value, 1)) {
            return false;
        }
    }
    return true;
}

static bool restore_s0(void *context) {
    pair_fixture *fixture = context;

    fixture->restore_calls++;
    fixture->mutable_state = 0;
    return true;
}

static bool run_pass_b_token(void *context,
                             int forced_token,
                             uint32_t row,
                             ds4_first_divergence_capture *capture) {
    pair_fixture *fixture = context;
    float value;

    if (fixture->mutable_state != 0 || row >= fixture->token_count ||
        forced_token != fixture->expected_tokens[row]) {
        return false;
    }
    fixture->pass_b_calls++;
    value = (float)(forced_token + (int)row);
    if (row == 1) value += 0.25f;
    return ds4_first_divergence_capture_f32(
        capture, row, 0, DS4_FIRST_DIVERGENCE_CP1, "attn_norm", &value, 1);
}

int main(void) {
    ds4_first_divergence_capture capture;
    ds4_first_divergence_capture pass_a;
    ds4_first_divergence_capture pass_b;
    ds4_first_divergence_report report;
    ds4_first_divergence_capture q_pass_a;
    ds4_first_divergence_capture q_pass_b;
    ds4_first_divergence_capture interval_pass_a;
    ds4_first_divergence_capture interval_pass_b;
    FILE *q_log;
    bool q_projection_exact;
    bool kv_projection_exact;
    ds4_first_divergence_float_signature signature;
    char q_log_text[4096];
    size_t q_log_bytes;
    const float cp1[] = {1.0f, -0.0f, 3.5f};
    const float qr_a[] = {0.125f, -0.5f, 1.0f, 2.0f};
    const float qr_b[] = {0.12500001f, -0.5f, 1.0000001f, 2.0f};
    const float kv_raw[] = {0.25f, -1.5f, 4.0f};
    const float interval_exact[] = {0.25f, -0.75f};
    const float interval_mismatch[] = {0.25f, -0.5f};
    const float signature_actual[] = {1.0f, 3.0f, 5.0f, 7.0f};
    const float signature_expected[] = {1.0f, 2.0f, 4.0f, 8.0f};
    const uint32_t n_comp = 17;
    const int forced_tokens[] = {101, 202, 303};
    pair_fixture fixture = {
        forced_tokens,
        sizeof(forced_tokens) / sizeof(forced_tokens[0]),
        0,
        0,
        0
    };
    const ds4_first_divergence_pair_ops ops = {
        &fixture,
        run_pass_a,
        restore_s0,
        run_pass_b_token
    };

    REQUIRE(ds4_first_divergence_capture_init(&capture, "PASS_A"));
    REQUIRE(ds4_first_divergence_capture_f32(
        &capture, 0, 0, DS4_FIRST_DIVERGENCE_CP1, "attn_norm",
        cp1, sizeof(cp1) / sizeof(cp1[0])));
    REQUIRE(ds4_first_divergence_capture_u32(
        &capture, 0, 0, DS4_FIRST_DIVERGENCE_CP3_F, "layer_n_comp",
        &n_comp, 1));
    REQUIRE(capture.count == 2);
    REQUIRE(capture.snapshots[0].element_count == 3);
    REQUIRE(memcmp(capture.snapshots[0].data, cp1, sizeof(cp1)) == 0);
    REQUIRE(strcmp(ds4_first_divergence_checkpoint_name(
                       DS4_FIRST_DIVERGENCE_CP2_KV_R),
                   "CP2-KV-R") == 0);
    REQUIRE(strcmp(ds4_first_divergence_checkpoint_name(
                       DS4_FIRST_DIVERGENCE_CP2_Q_NORM),
                   "CP2-Q-NORM") == 0);
    REQUIRE(strcmp(ds4_first_divergence_checkpoint_name(
                       DS4_FIRST_DIVERGENCE_CP4_HEADS),
                   "CP4-HEADS") == 0);
    REQUIRE(ds4_first_divergence_float_signature_compute(
        signature_actual, signature_expected, 4, &signature));
    REQUIRE(signature.mismatch_count == 3);
    REQUIRE(signature.mismatch_fraction == 0.75);
    REQUIRE(signature.finite_metrics_defined);
    REQUIRE(signature.mean_abs == 0.75);
    REQUIRE(fabs(signature.rms_abs - sqrt(0.75)) < 1e-15);
    REQUIRE(signature.p50_abs == 1.0);
    REQUIRE(signature.p95_abs == 1.0);
    REQUIRE(signature.p99_abs == 1.0);
    REQUIRE(signature.relative_l2_defined);
    REQUIRE(fabs(signature.relative_l2 - sqrt(3.0 / 85.0)) < 1e-15);
    REQUIRE(signature.cosine_similarity_defined);
    REQUIRE(signature.positive_delta_count == 2);
    REQUIRE(signature.negative_delta_count == 1);
    ds4_first_divergence_capture_free(&capture);
    REQUIRE(capture.count == 0);

    REQUIRE(ds4_first_divergence_capture_init(&pass_a, "PASS_A"));
    REQUIRE(ds4_first_divergence_capture_init(&pass_b, "PASS_B"));
    REQUIRE(ds4_first_divergence_run_forced_pair(
        forced_tokens, fixture.token_count, &ops, &pass_a, &pass_b));
    REQUIRE(fixture.restore_calls == 1);
    REQUIRE(fixture.pass_b_calls == fixture.token_count);
    REQUIRE(pass_a.count == fixture.token_count);
    REQUIRE(pass_b.count == fixture.token_count);
    REQUIRE(memcmp(forced_tokens, fixture.expected_tokens,
                   sizeof(forced_tokens)) == 0);
    REQUIRE(ds4_first_divergence_emit_report(
        &pass_a, &pass_b, stdout, &report));
    REQUIRE(!report.bit_exact);
    REQUIRE(report.first_divergence_found);
    REQUIRE(report.row == 1);
    REQUIRE(report.layer == 0);
    REQUIRE(report.checkpoint == DS4_FIRST_DIVERGENCE_CP1);
    REQUIRE(strcmp(report.subobject, "attn_norm") == 0);
    ds4_first_divergence_capture_free(&pass_a);
    ds4_first_divergence_capture_free(&pass_b);

    REQUIRE(ds4_first_divergence_capture_init(&q_pass_a, "PASS_A"));
    REQUIRE(ds4_first_divergence_capture_init(&q_pass_b, "PASS_B"));
    REQUIRE(ds4_first_divergence_capture_f32(
        &q_pass_a, 0, 0, DS4_FIRST_DIVERGENCE_CP1, "attn_norm",
        cp1, sizeof(cp1) / sizeof(cp1[0])));
    REQUIRE(ds4_first_divergence_capture_f32(
        &q_pass_b, 0, 0, DS4_FIRST_DIVERGENCE_CP1, "attn_norm",
        cp1, sizeof(cp1) / sizeof(cp1[0])));
    REQUIRE(ds4_first_divergence_capture_f32(
        &q_pass_a, 0, 0, DS4_FIRST_DIVERGENCE_CP2_Q, "qr",
        qr_a, sizeof(qr_a) / sizeof(qr_a[0])));
    REQUIRE(ds4_first_divergence_capture_f32(
        &q_pass_b, 0, 0, DS4_FIRST_DIVERGENCE_CP2_Q, "qr",
        qr_b, sizeof(qr_b) / sizeof(qr_b[0])));
    REQUIRE(ds4_first_divergence_capture_f32(
        &q_pass_a, 0, 0, DS4_FIRST_DIVERGENCE_CP2_KV_P, "kv_raw",
        kv_raw, sizeof(kv_raw) / sizeof(kv_raw[0])));
    REQUIRE(ds4_first_divergence_capture_f32(
        &q_pass_b, 0, 0, DS4_FIRST_DIVERGENCE_CP2_KV_P, "kv_raw",
        kv_raw, sizeof(kv_raw) / sizeof(kv_raw[0])));
    q_log = tmpfile();
    REQUIRE(q_log != NULL);
    REQUIRE(ds4_first_divergence_emit_report(
        &q_pass_a, &q_pass_b, q_log, &report));
    REQUIRE(report.first_divergence_found);
    REQUIRE(report.row == 0);
    REQUIRE(report.layer == 0);
    REQUIRE(report.checkpoint == DS4_FIRST_DIVERGENCE_CP2_Q);
    REQUIRE(strcmp(report.subobject, "qr") == 0);
    REQUIRE(fflush(q_log) == 0);
    REQUIRE(fseek(q_log, 0, SEEK_SET) == 0);
    q_log_bytes = fread(q_log_text, 1, sizeof(q_log_text) - 1, q_log);
    q_log_text[q_log_bytes] = '\0';
    REQUIRE(strstr(q_log_text, "Q_DIVERGENCE_TRACE row=0 layer=0") != NULL);
    REQUIRE(strstr(q_log_text,
                   "stage=CP1 semantic=normalized_attention_input subobject=attn_norm result=EXACT") != NULL);
    REQUIRE(strstr(q_log_text,
                   "stage=CP2-Q semantic=q_a_projection_output subobject=qr result=MISMATCH") != NULL);
    REQUIRE(strstr(q_log_text,
                   "Q_FIRST_DIVERGENCE stage=q_a_projection_output") != NULL);
    REQUIRE(strstr(q_log_text,
                   "Q_LOCALIZATION_RESULT FIRST_RUNTIME_DIVERGENCE_WITHIN_Q_PATH") != NULL);
    REQUIRE(fclose(q_log) == 0);

    memcpy(q_pass_b.snapshots[1].data, qr_a, sizeof(qr_a));
    q_log = tmpfile();
    REQUIRE(q_log != NULL);
    REQUIRE(ds4_first_divergence_emit_report(
        &q_pass_a, &q_pass_b, q_log, &report));
    REQUIRE(report.bit_exact);
    REQUIRE(!report.first_divergence_found);
    REQUIRE(ds4_first_divergence_emit_q_trace(
        &q_pass_a, &q_pass_b, q_log, &q_projection_exact));
    REQUIRE(q_projection_exact);
    REQUIRE(ds4_first_divergence_emit_kv_trace(
        &q_pass_a, &q_pass_b, q_log, &kv_projection_exact));
    REQUIRE(kv_projection_exact);
    REQUIRE(ds4_first_divergence_emit_qa_canonical_summary(
        &report, q_log));
    report.first_divergence_found = true;
    report.row = 0;
    report.layer = 0;
    report.checkpoint = DS4_FIRST_DIVERGENCE_CP2_KV_P;
    REQUIRE(snprintf(report.subobject, sizeof(report.subobject), "%s",
                     "kv_raw") > 0);
    REQUIRE(ds4_first_divergence_emit_qa_canonical_summary(
        &report, q_log));
    REQUIRE(fflush(q_log) == 0);
    REQUIRE(fseek(q_log, 0, SEEK_SET) == 0);
    q_log_bytes = fread(q_log_text, 1, sizeof(q_log_text) - 1, q_log);
    q_log_text[q_log_bytes] = '\0';
    REQUIRE(strstr(q_log_text,
                   "stage=CP2-Q semantic=q_a_projection_output subobject=qr result=EXACT") != NULL);
    REQUIRE(strstr(q_log_text,
                   "Q_LOCALIZATION_RESULT QA_PROJECTION_EXACT") != NULL);
    REQUIRE(strstr(q_log_text,
                   "stage=CP2-KV-P semantic=kv_projection_output_before_persistent_store subobject=kv_raw result=EXACT") != NULL);
    REQUIRE(strstr(q_log_text,
                   "KV_LOCALIZATION_RESULT KV_PROJECTION_EXACT") != NULL);
    REQUIRE(strstr(q_log_text,
                   "QA_CANONICALIZATION_RESULT baseline_first_divergence=row=0,layer=0,checkpoint=CP2-Q,q_a_projection_output qa_after_patch=EXACT new_first_divergence=NONE") != NULL);
    REQUIRE(strstr(q_log_text,
                   "NEXT_INDEPENDENT_DRIFT_SOURCE row=0 layer=0 checkpoint=CP2-KV-P subobject=kv_raw") != NULL);
    REQUIRE(fclose(q_log) == 0);
    ds4_first_divergence_capture_free(&q_pass_a);
    ds4_first_divergence_capture_free(&q_pass_b);

    REQUIRE(ds4_first_divergence_capture_init(&interval_pass_a, "PASS_A"));
    REQUIRE(ds4_first_divergence_capture_init(&interval_pass_b, "PASS_B"));
#define CAPTURE_INTERVAL_PAIR(checkpoint_, subobject_, a_, b_) do { \
        REQUIRE(ds4_first_divergence_capture_f32( \
            &interval_pass_a, 0, 0, (checkpoint_), (subobject_), \
            (a_), 2)); \
        REQUIRE(ds4_first_divergence_capture_f32( \
            &interval_pass_b, 0, 0, (checkpoint_), (subobject_), \
            (b_), 2)); \
    } while (0)
    /* Capture deliberately out of order; reporting must follow the causal
     * checkpoint enum, not insertion order or subobject spelling. */
    CAPTURE_INTERVAL_PAIR(DS4_FIRST_DIVERGENCE_CP4,
                          "after_attn_hc", interval_exact, interval_exact);
    CAPTURE_INTERVAL_PAIR(DS4_FIRST_DIVERGENCE_CP4_HEADS,
                          "attn_heads", interval_exact, interval_exact);
    CAPTURE_INTERVAL_PAIR(DS4_FIRST_DIVERGENCE_CP2_Q_CUR,
                          "q_cur", interval_exact, interval_mismatch);
    CAPTURE_INTERVAL_PAIR(DS4_FIRST_DIVERGENCE_CP2_Q_NORM,
                          "qr_norm", interval_exact, interval_exact);
#undef CAPTURE_INTERVAL_PAIR
    q_log = tmpfile();
    REQUIRE(q_log != NULL);
    REQUIRE(ds4_first_divergence_emit_report(
        &interval_pass_a, &interval_pass_b, q_log, &report));
    REQUIRE(report.first_divergence_found);
    REQUIRE(report.checkpoint == DS4_FIRST_DIVERGENCE_CP2_Q_CUR);
    REQUIRE(strcmp(report.subobject, "q_cur") == 0);
    REQUIRE(ds4_first_divergence_emit_attention_interval_trace(
        &interval_pass_a, &interval_pass_b, q_log));
    REQUIRE(fflush(q_log) == 0);
    REQUIRE(fseek(q_log, 0, SEEK_SET) == 0);
    q_log_bytes = fread(q_log_text, 1, sizeof(q_log_text) - 1, q_log);
    q_log_text[q_log_bytes] = '\0';
    REQUIRE(strstr(q_log_text,
                   "FIRST_DIVERGENCE row=0 layer=0 checkpoint=CP2-Q-CUR subobject=q_cur") != NULL);
    REQUIRE(strstr(q_log_text,
                   "stage=CP2-Q-NORM semantic=normalized_q_a_projection_output subobject=qr_norm result=EXACT") != NULL);
    REQUIRE(strstr(q_log_text,
                   "stage=CP2-Q-CUR semantic=q_after_q_b_head_norm_and_rope subobject=q_cur result=MISMATCH") != NULL);
    REQUIRE(strstr(q_log_text,
                   "ATTENTION_INTERVAL_RESULT FIRST_RUNTIME_DIVERGENCE stage=CP2-Q-CUR") != NULL);
    REQUIRE(fclose(q_log) == 0);
    ds4_first_divergence_capture_free(&interval_pass_a);
    ds4_first_divergence_capture_free(&interval_pass_b);

    puts("first-divergence forced pair: OK");
    return 0;
}
