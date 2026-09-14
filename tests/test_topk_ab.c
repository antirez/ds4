/* Standalone exactness regression for the Metal indexer top-k paths.
 *
 * The final #832 defaults are canonical argsort and, for top_k=512 with at
 * least 32 tokens, stream512.  Compare those actual final paths directly and
 * require both to match a CPU (score descending, index ascending) reference.
 * Cases cover dispatch gates, odd row widths, visibility below top_k, dense
 * finite ties, and a 64K-context-sized compressed frontier.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu.h"

typedef struct {
    const char *name;
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t top_k;
    uint32_t visible_base;
    uint32_t visible_step;
    uint32_t score_modulus;
} topk_case;

static const float *g_row;

static int cmp_desc_idx(const void *x, const void *y) {
    const uint32_t ix = *(const uint32_t *)x;
    const uint32_t iy = *(const uint32_t *)y;
    if (g_row[ix] > g_row[iy]) return -1;
    if (g_row[ix] < g_row[iy]) return 1;
    if (ix < iy) return -1;
    if (ix > iy) return 1;
    return 0;
}

static uint32_t next_random(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static float negative_infinity(void) {
    const uint32_t bits = UINT32_C(0xff800000);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int select_canonical(void) {
    return unsetenv("DS4_METAL_DISABLE_ARGSORT_CANON") == 0 &&
           setenv("DS4_METAL_DISABLE_TOPK_STREAM512", "1", 1) == 0;
}

static int select_default(void) {
    return unsetenv("DS4_METAL_DISABLE_ARGSORT_CANON") == 0 &&
           unsetenv("DS4_METAL_DISABLE_TOPK_STREAM512") == 0;
}

static int first_difference(
        const int32_t *expected,
        const int32_t *observed,
        size_t count,
        size_t *first,
        size_t *different) {
    *first = 0;
    *different = 0;
    for (size_t i = 0; i < count; i++) {
        if (expected[i] != observed[i]) {
            if ((*different)++ == 0) *first = i;
        }
    }
    return *different != 0;
}

static int run_case(const topk_case *tc) {
    const size_t score_count = (size_t)tc->n_comp * tc->n_tokens;
    const size_t selected_count = (size_t)tc->top_k * tc->n_tokens;
    const float ninf = negative_infinity();
    float *host_scores = NULL;
    int32_t *canonical = NULL, *actual = NULL, *reference = NULL;
    uint32_t *order = NULL;
    ds4_gpu_tensor *scores = NULL, *selected = NULL;
    int rc = 1;

    host_scores = malloc(score_count * sizeof(*host_scores));
    canonical = malloc(selected_count * sizeof(*canonical));
    actual = malloc(selected_count * sizeof(*actual));
    reference = malloc(selected_count * sizeof(*reference));
    order = malloc((size_t)tc->n_comp * sizeof(*order));
    if (!host_scores || !canonical || !actual || !reference || !order) {
        fprintf(stderr, "FAIL %s: host allocation\n", tc->name);
        goto done;
    }

    uint32_t random_state = UINT32_C(0x9e3779b9) ^ tc->n_comp ^
                            tc->n_tokens ^ tc->top_k;
    for (uint32_t token = 0; token < tc->n_tokens; token++) {
        uint64_t visible_wide = (uint64_t)tc->visible_base +
                                (uint64_t)token * tc->visible_step;
        const uint32_t visible = visible_wide < tc->n_comp
            ? (uint32_t)visible_wide : tc->n_comp;
        for (uint32_t comp = 0; comp < tc->n_comp; comp++) {
            host_scores[(size_t)token * tc->n_comp + comp] = comp < visible
                ? (float)(next_random(&random_state) % tc->score_modulus) / 8.0f
                : ninf;
        }
    }

    scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    selected = ds4_gpu_tensor_alloc(selected_count * sizeof(int32_t));
    if (!scores || !selected ||
        !ds4_gpu_tensor_write(scores, 0, host_scores,
                              score_count * sizeof(float))) {
        fprintf(stderr, "FAIL %s: GPU allocation/write\n", tc->name);
        goto done;
    }

    memset(canonical, 0x31, selected_count * sizeof(*canonical));
    if (!select_canonical() ||
        !ds4_gpu_tensor_write(selected, 0, canonical,
                              selected_count * sizeof(*canonical)) ||
        !ds4_gpu_indexer_topk_tensor(selected, scores, tc->n_comp,
                                     tc->n_tokens, tc->top_k) ||
        !ds4_gpu_tensor_read(selected, 0, canonical,
                             selected_count * sizeof(*canonical))) {
        fprintf(stderr, "FAIL %s: canonical dispatch/read\n", tc->name);
        goto done;
    }

    memset(actual, 0x73, selected_count * sizeof(*actual));
    if (!select_default() ||
        !ds4_gpu_tensor_write(selected, 0, actual,
                              selected_count * sizeof(*actual)) ||
        !ds4_gpu_indexer_topk_tensor(selected, scores, tc->n_comp,
                                     tc->n_tokens, tc->top_k) ||
        !ds4_gpu_tensor_read(selected, 0, actual,
                             selected_count * sizeof(*actual))) {
        fprintf(stderr, "FAIL %s: default dispatch/read\n", tc->name);
        goto done;
    }

    for (uint32_t token = 0; token < tc->n_tokens; token++) {
        g_row = host_scores + (size_t)token * tc->n_comp;
        for (uint32_t comp = 0; comp < tc->n_comp; comp++) order[comp] = comp;
        qsort(order, tc->n_comp, sizeof(*order), cmp_desc_idx);
        for (uint32_t rank = 0; rank < tc->top_k; rank++) {
            reference[(size_t)token * tc->top_k + rank] =
                (int32_t)order[rank];
        }
    }

    size_t first = 0, different = 0;
    if (first_difference(reference, canonical, selected_count,
                         &first, &different)) {
        fprintf(stderr,
                "FAIL %s: canonical differs from CPU at %zu/%zu entries; "
                "first token=%zu rank=%zu expected=%d observed=%d\n",
                tc->name, different, selected_count, first / tc->top_k,
                first % tc->top_k, reference[first], canonical[first]);
        goto done;
    }
    if (first_difference(reference, actual, selected_count,
                         &first, &different)) {
        fprintf(stderr,
                "FAIL %s: default differs from CPU at %zu/%zu entries; "
                "first token=%zu rank=%zu expected=%d observed=%d\n",
                tc->name, different, selected_count, first / tc->top_k,
                first % tc->top_k, reference[first], actual[first]);
        goto done;
    }

    fprintf(stderr, "PASS %s: canonical/default/CPU identical (%zu entries)\n",
            tc->name, selected_count);
    rc = 0;

done:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(order);
    free(reference);
    free(actual);
    free(canonical);
    free(host_scores);
    return rc;
}

int main(void) {
    static const topk_case cases[] = {
        {"below-token-gate", 1024u, 31u, 512u, 300u, 17u, 97u},
        {"token-gate", 1025u, 32u, 512u, 511u, 1u, 97u},
        {"minimum-stream-row", 512u, 32u, 512u, 512u, 0u, 17u},
        {"odd-row-width", 2053u, 33u, 512u, 400u, 37u, 97u},
        {"dense-finite-ties", 4099u, 40u, 512u, 4099u, 0u, 7u},
        {"64k-compressed-frontier", 16387u, 64u, 512u, 300u, 251u, 97u},
        {"non-512-fallback", 1025u, 32u, 511u, 700u, 0u, 17u},
    };

    if (!ds4_gpu_init()) {
        fprintf(stderr, "FAIL: GPU init\n");
        return 1;
    }
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (run_case(&cases[i]) != 0) {
            ds4_gpu_cleanup();
            return 1;
        }
    }
    ds4_gpu_cleanup();
    unsetenv("DS4_METAL_DISABLE_ARGSORT_CANON");
    unsetenv("DS4_METAL_DISABLE_TOPK_STREAM512");
    fprintf(stderr, "PASS: all indexer top-k regressions\n");
    return 0;
}
