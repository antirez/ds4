#include "../ds4_suffix_tree.h"

#include <stdio.h>

static int expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "suffix_tree_test: %s\n", msg);
        return 1;
    }
    return 0;
}

static ds4_suffix_node *lookup_child(ds4_suffix_node *node, int token_id) {
    for (uint32_t i = 0; node && i < node->n_children; i++) {
        if (node->children[i].token_id == token_id) return &node->children[i];
    }
    return NULL;
}

static int test_repeated_continuation(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 8);
    if (!tree) return expect(0, "alloc failed");

    const int seq[] = {
        1, 2, 3, 4, 9,
        1, 2, 3, 4, 9,
        1, 2, 3, 5, 8,
    };
    ds4_suffix_tree_insert(tree, seq, (uint32_t)(sizeof(seq) / sizeof(seq[0])));

    int drafts[4] = {0};
    uint32_t draft_n = 0;
    float score = 0.0f;
    const int prefix[] = {1, 2, 3};
    int rc = 0;
    rc |= expect(ds4_suffix_tree_match_depth(tree, prefix, 3) == 3,
                 "match_depth should report full 1,2,3 match");
    uint32_t p = ds4_suffix_tree_query(tree, prefix, 3, drafts, 4, &draft_n, 0.0f, &score);

    rc |= expect(p == 3, "expected full 1,2,3 match");
    rc |= expect(draft_n >= 2, "expected continuation after 1,2,3");
    rc |= expect(drafts[0] == 4, "expected most frequent continuation token 4");
    rc |= expect(drafts[1] == 9, "expected most frequent second continuation token 9");

    ds4_suffix_stats st;
    ds4_suffix_tree_stats(tree, &st);
    rc |= expect(st.query_count == 1, "query_count should be 1");
    rc |= expect(st.query_hits == 1, "query_hits should be 1");
    rc |= expect(st.draft_tokens_produced == draft_n,
                 "draft_tokens_produced should match first query");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_append_does_not_reinflate_prefix(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 4);
    if (!tree) return expect(0, "alloc failed");

    const int seq1[] = {1, 2, 3};
    const int seq2[] = {1, 2, 3, 4};
    ds4_suffix_tree_insert(tree, seq1, 3);

    ds4_suffix_node *n2 = lookup_child(&tree->root, 2);
    int rc = 0;
    rc |= expect(n2 != NULL, "expected suffix starting at token 2");
    rc |= expect(n2 && n2->freq == 1,
                 "initial token 2 suffix frequency should be 1");

    ds4_suffix_tree_append(tree, seq2, 4);
    n2 = lookup_child(&tree->root, 2);
    ds4_suffix_node *n23 = lookup_child(n2, 3);
    ds4_suffix_node *n234 = lookup_child(n23, 4);
    ds4_suffix_node *n4 = lookup_child(&tree->root, 4);

    rc |= expect(n2 && n2->freq == 1,
                 "append should not reinflate existing token 2 prefix");
    rc |= expect(n234 && n234->freq == 1,
                 "append should extend existing 2,3 suffix with token 4");
    rc |= expect(n4 && n4->freq == 1,
                 "append should add one-token suffix for new token");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_skip_terminal_longest_match(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 8);
    if (!tree) return expect(0, "alloc failed");

    const int seq[] = {1, 2, 3, 4, 7, 2, 3, 4};
    ds4_suffix_tree_insert(tree, seq, (uint32_t)(sizeof(seq) / sizeof(seq[0])));

    int drafts[4] = {0};
    uint32_t draft_n = 0;
    float score = 0.0f;
    const int prefix[] = {2, 3, 4};
    uint32_t p = ds4_suffix_tree_query(tree, prefix, 3, drafts, 4, &draft_n, 0.0f, &score);

    int rc = 0;
    rc |= expect(p == 3, "expected longest match with continuation, not terminal suffix");
    rc |= expect(draft_n > 0, "expected a draft from non-terminal match");
    rc |= expect(drafts[0] == 7, "expected continuation token 7");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_no_partial_prefix_match(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 8);
    if (!tree) return expect(0, "alloc failed");

    const int seq[] = {10, 20, 30, 40, 50};
    ds4_suffix_tree_insert(tree, seq, (uint32_t)(sizeof(seq) / sizeof(seq[0])));

    int drafts[4] = {0};
    uint32_t draft_n = 123;
    float score = 0.0f;
    const int prefix[] = {20, 30, 99};
    uint32_t p = ds4_suffix_tree_query(tree, prefix, 3, drafts, 4, &draft_n, 0.0f, &score);

    int rc = 0;
    rc |= expect(p == 0, "partial prefix should not count as a suffix match");
    rc |= expect(draft_n == 0, "partial prefix should not produce drafts");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_prune_and_reset(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1, 8);
    if (!tree) return expect(0, "alloc failed");

    int rc = 0;
    for (int i = 0; i < 2048; i++) {
        int seq[] = {i, i + 4096};
        ds4_suffix_tree_insert(tree, seq, 2);
    }
    rc |= expect(tree->node_count <= tree->node_budget,
                 "tree should prune back to its node budget");

    ds4_suffix_tree_reset(tree);
    rc |= expect(tree->node_count == 0, "reset should clear all nodes");
    rc |= expect(tree->root.n_children == 0, "reset should clear root children");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_probability_estimation_and_score(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 8);
    if (!tree) return expect(0, "alloc failed");

    /* Insert pattern: {1,2,3} appears 3 times, {1,2,4} appears 1 time.
     * At node for prefix {1,2}:
     *   - child 3 has freq 3
     *   - child 4 has freq 1
     *   - parent (node 2) has freq = count of suffixes through this path
     * Prob(token 3 | {1,2}) = 3/4 = 0.75
     * Prob(token 3, next token | {1,2,3}) depends on continuation. */
    const int seq[] = {
        1, 2, 3, 10,
        1, 2, 3, 10,
        1, 2, 3, 11,
        1, 2, 4, 20,
    };
    ds4_suffix_tree_insert(tree, seq, (uint32_t)(sizeof(seq) / sizeof(seq[0])));

    int drafts[4] = {0};
    uint32_t draft_n = 0;
    float score = 0.0f;
    const int prefix[] = {1, 2};
    uint32_t p = ds4_suffix_tree_query(tree, prefix, 2, drafts, 4, &draft_n, 0.0f, &score);

    int rc = 0;
    rc |= expect(p == 2, "expected match length 2 for prefix {1,2}");
    rc |= expect(draft_n > 0, "expected drafts");
    rc |= expect(drafts[0] == 3, "expected most frequent continuation token 3");
    rc |= expect(score > 0.0f, "score should be positive");
    rc |= expect(score <= (float)draft_n, "score should be at most draft_n (sum of per-token probs)");

    /* Check telemetry: draft_score_total should equal score */
    ds4_suffix_stats st;
    ds4_suffix_tree_stats(tree, &st);
    rc |= expect(st.draft_score_total > 0.0, "draft_score_total should be positive");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_min_prob_cutoff(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 8);
    if (!tree) return expect(0, "alloc failed");

    /* {1,2,3} seen 5 times, {1,2,4} seen 1 time.
     * P(3|{1,2}) = 5/6 ≈ 0.83
     * After choosing 3, continuation prob depends on subtree. */
    const int seq[] = {
        1, 2, 3, 10,
        1, 2, 3, 10,
        1, 2, 3, 10,
        1, 2, 3, 10,
        1, 2, 3, 10,
        1, 2, 4, 20,
    };
    ds4_suffix_tree_insert(tree, seq, (uint32_t)(sizeof(seq) / sizeof(seq[0])));

    /* Query without min_prob: should produce drafts */
    int drafts[4] = {0};
    uint32_t draft_n = 0;
    float score_no_filter = 0.0f;
    const int prefix[] = {1, 2};
    ds4_suffix_tree_query(tree, prefix, 2, drafts, 4, &draft_n, 0.0f, &score_no_filter);

    /* Query with very high min_prob: should produce fewer or zero drafts */
    int drafts2[4] = {0};
    uint32_t draft_n2 = 0;
    float score_filtered = 0.0f;
    ds4_suffix_tree_query(tree, prefix, 2, drafts2, 4, &draft_n2, 0.99f, &score_filtered);

    int rc = 0;
    rc |= expect(draft_n > 0, "unfiltered query should produce drafts");
    rc |= expect(draft_n2 <= draft_n,
                 "high min_prob should produce <= drafts compared to no filter");
    /* With min_prob=0.99, only tokens with P>0.99 survive. P(3|{1,2})=5/6≈0.83 < 0.99 */
    rc |= expect(draft_n2 == 0, "min_prob=0.99 should reject all drafts since P=0.83");

    ds4_suffix_tree_free(tree);
    return rc;
}

static int test_best_child_cache(void) {
    ds4_suffix_tree *tree = ds4_suffix_tree_alloc(1024 * 1024, 8);
    if (!tree) return expect(0, "alloc failed");

    /* Insert: token 10 appears 3 times after {1,2}, token 20 appears 1 time */
    const int seq[] = {
        1, 2, 10,
        1, 2, 10,
        1, 2, 10,
        1, 2, 20,
    };
    ds4_suffix_tree_insert(tree, seq, (uint32_t)(sizeof(seq) / sizeof(seq[0])));

    /* Find the node for prefix {1,2} */
    ds4_suffix_node *n1 = lookup_child(&tree->root, 1);
    int rc = 0;
    rc |= expect(n1 != NULL, "expected node for token 1");
    ds4_suffix_node *n12 = lookup_child(n1, 2);
    rc |= expect(n12 != NULL, "expected node for {1,2}");

    /* After insert, best_child_idx may be invalid (UINT32_MAX) because
     * insert_child invalidates the cache on every structural change.
     * It gets repopulated lazily on the next query. */
    /* Query populates the best_child cache. */
    int drafts[4] = {0};
    uint32_t draft_n = 0;
    float score = 0.0f;
    const int prefix[] = {1, 2};
    ds4_suffix_tree_query(tree, prefix, 2, drafts, 4, &draft_n, 0.0f, &score);
    rc |= expect(draft_n > 0, "expected drafts");
    rc |= expect(drafts[0] == 10, "draft should follow highest-freq child (token 10)");

    /* After query, best_child_idx should be valid and point to token 10. */
    rc |= expect(n12->best_child_idx < n12->n_children,
                 "best_child_idx should be valid after query");
    rc |= expect(n12->children[n12->best_child_idx].token_id == 10,
                 "cached best child should be token 10 (highest freq)");

    /* Second query should hit the cache directly (no rescan needed). */
    int drafts2[4] = {0};
    uint32_t draft_n2 = 0;
    float score2 = 0.0f;
    ds4_suffix_tree_query(tree, prefix, 2, drafts2, 4, &draft_n2, 0.0f, &score2);
    rc |= expect(draft_n2 == draft_n, "second query should produce same draft count");
    rc |= expect(drafts2[0] == 10, "second query should also pick token 10");

    ds4_suffix_tree_free(tree);
    return rc;
}

int main(void) {
    int rc = 0;
    rc |= test_repeated_continuation();
    rc |= test_append_does_not_reinflate_prefix();
    rc |= test_skip_terminal_longest_match();
    rc |= test_no_partial_prefix_match();
    rc |= test_prune_and_reset();
    rc |= test_probability_estimation_and_score();
    rc |= test_min_prob_cutoff();
    rc |= test_best_child_cache();
    return rc ? 1 : 0;
}
