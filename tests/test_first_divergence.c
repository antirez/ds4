#include "first_divergence_capture.h"

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
    const float cp1[] = {1.0f, -0.0f, 3.5f};
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

    puts("first-divergence forced pair: OK");
    return 0;
}
