#include "ds4_qwen4.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(expr) do {                                                       \
    if (!(expr)) {                                                             \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,     \
                #expr);                                                        \
        failures++;                                                            \
    }                                                                          \
} while (0)

static void check_plan(const char *label,
                       const ds4_qwen4_mtp_accept_plan *plan,
                       uint32_t proposed,
                       uint32_t accepted,
                       uint32_t rejected_at,
                       bool first_draft_matches,
                       bool full_accept) {
    if (plan->proposed != proposed || plan->accepted != accepted ||
        plan->rejected_at != rejected_at ||
        plan->first_draft_matches != first_draft_matches ||
        plan->full_accept != full_accept) {
        fprintf(stderr,
                "%s: got proposed=%u accepted=%u rejected_at=%u "
                "first_match=%d full_accept=%d; expected %u/%u/%u/%d/%d\n",
                label, plan->proposed, plan->accepted, plan->rejected_at,
                plan->first_draft_matches, plan->full_accept,
                proposed, accepted, rejected_at,
                first_draft_matches, full_accept);
        failures++;
    }
}

static void check_zero_plan(const ds4_qwen4_mtp_accept_plan *plan) {
    check_plan("invalid input reset", plan, 0u, 0u, 0u, false, false);
}

static void test_invalid_inputs(void) {
    int32_t drafts[DS4_QWEN4_MTP_MAX_DRAFTS + 1u] = {0};
    int32_t verifier_tops[DS4_QWEN4_MTP_MAX_DRAFTS + 1u] = {0};
    ds4_qwen4_mtp_accept_plan plan;

    CHECK(!ds4_qwen4_mtp_plan_greedy(
        drafts, 1u, drafts[0], verifier_tops, 1u, NULL));

#define CHECK_INVALID(drafts_, proposed_, tops_, rows_) do {                  \
        memset(&plan, 0xa5, sizeof(plan));                                    \
        CHECK(!ds4_qwen4_mtp_plan_greedy(                                    \
            (drafts_), (proposed_), 0, (tops_), (rows_), &plan));             \
        check_zero_plan(&plan);                                                \
    } while (0)

    CHECK_INVALID(NULL, 1u, verifier_tops, 1u);
    CHECK_INVALID(drafts, 0u, verifier_tops, 0u);
    CHECK_INVALID(drafts, DS4_QWEN4_MTP_MAX_DRAFTS + 1u,
                  verifier_tops, DS4_QWEN4_MTP_MAX_DRAFTS + 1u);
    CHECK_INVALID(drafts, 2u, verifier_tops, 1u);
    CHECK_INVALID(drafts, 2u, verifier_tops, 3u);
    CHECK_INVALID(drafts, 2u, NULL, 2u);
    CHECK_INVALID(drafts, 1u, NULL, 0u);
    CHECK_INVALID(drafts, 1u, NULL, 2u);

#undef CHECK_INVALID
}

static void test_first_miss(void) {
    const int32_t drafts[] = {10, 11, 12, 13};
    const int32_t verifier_tops[] = {11, 12, 13, 99};
    ds4_qwen4_mtp_accept_plan plan;
    CHECK(ds4_qwen4_mtp_plan_greedy(
        drafts, 4u, 9, verifier_tops, 4u, &plan));
    check_plan("first miss", &plan, 4u, 0u, 0u, false, false);
}

static void test_full_accept(void) {
    int32_t drafts[DS4_QWEN4_MTP_MAX_DRAFTS];
    int32_t verifier_tops[DS4_QWEN4_MTP_MAX_DRAFTS];
    for (uint32_t i = 0; i < DS4_QWEN4_MTP_MAX_DRAFTS; i++)
        drafts[i] = (int32_t)(100u + i);
    for (uint32_t i = 0; i < DS4_QWEN4_MTP_MAX_DRAFTS; i++) {
        verifier_tops[i] = i + 1u < DS4_QWEN4_MTP_MAX_DRAFTS
            ? drafts[i + 1u]
            : INT32_MIN;
    }

    ds4_qwen4_mtp_accept_plan plan;
    CHECK(ds4_qwen4_mtp_plan_greedy(
        drafts, DS4_QWEN4_MTP_MAX_DRAFTS, drafts[0], verifier_tops,
        DS4_QWEN4_MTP_MAX_DRAFTS, &plan));
    check_plan("full accept", &plan, DS4_QWEN4_MTP_MAX_DRAFTS,
               DS4_QWEN4_MTP_MAX_DRAFTS,
               DS4_QWEN4_MTP_MAX_DRAFTS, true, true);

    /* The final verifier row predicts the continuation after the last draft;
     * it must not participate in accepting the proposed block. */
    verifier_tops[DS4_QWEN4_MTP_MAX_DRAFTS - 1u] = INT32_MAX;
    CHECK(ds4_qwen4_mtp_plan_greedy(
        drafts, DS4_QWEN4_MTP_MAX_DRAFTS, drafts[0], verifier_tops,
        DS4_QWEN4_MTP_MAX_DRAFTS, &plan));
    check_plan("full accept ignores continuation", &plan,
               DS4_QWEN4_MTP_MAX_DRAFTS,
               DS4_QWEN4_MTP_MAX_DRAFTS,
               DS4_QWEN4_MTP_MAX_DRAFTS, true, true);
}

static void test_rejection_at_every_later_depth(void) {
    enum { DEPTH = 5 };
    const int32_t drafts[DEPTH] = {20, 21, 22, 23, 24};
    const int32_t continuation = 25;

    for (uint32_t rejected_at = 1u; rejected_at < DEPTH; rejected_at++) {
        int32_t verifier_tops[DEPTH];
        for (uint32_t row = 0; row + 1u < DEPTH; row++)
            verifier_tops[row] = drafts[row + 1u];
        verifier_tops[DEPTH - 1u] = continuation;
        verifier_tops[rejected_at - 1u] = 1000 + (int32_t)rejected_at;

        ds4_qwen4_mtp_accept_plan plan;
        CHECK(ds4_qwen4_mtp_plan_greedy(
            drafts, DEPTH, drafts[0], verifier_tops, DEPTH, &plan));
        char label[64];
        snprintf(label, sizeof(label), "reject at depth %u", rejected_at);
        check_plan(label, &plan, DEPTH, rejected_at, rejected_at, true, false);
    }
}

static void test_depth_one(void) {
    const int32_t draft = 42;
    ds4_qwen4_mtp_accept_plan plan;

    CHECK(ds4_qwen4_mtp_plan_greedy(
        &draft, 1u, draft, NULL, 1u, &plan));
    check_plan("depth one accept", &plan, 1u, 1u, 1u, true, true);

    CHECK(ds4_qwen4_mtp_plan_greedy(
        &draft, 1u, draft + 1, NULL, 1u, &plan));
    check_plan("depth one miss", &plan, 1u, 0u, 0u, false, false);
}

int main(void) {
    test_invalid_inputs();
    test_first_miss();
    test_full_accept();
    test_rejection_at_every_later_depth();
    test_depth_one();
    if (failures != 0) {
        fprintf(stderr, "qwen4 speculative planner tests: %d failure(s)\n",
                failures);
        return 1;
    }
    puts("qwen4 speculative planner tests: ok");
    return 0;
}
