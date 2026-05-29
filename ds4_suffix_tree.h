#ifndef DS4_SUFFIX_TREE_H
#define DS4_SUFFIX_TREE_H

#include <stdint.h>
#include <stddef.h>

/*
 * Suffix tree for model-free speculative decoding.
 *
 * Each node represents a token ID.  Paths from the root represent token
 * sequences observed in prompts and prior outputs.  The tree is used to
 * propose draft tokens during speculative decoding by looking up the current
 * context in the tree and returning the most frequent continuation.
 *
 * Reference: SuffixDecoding (arxiv:2411.04975).
 */

/* One node in the suffix trie. */
typedef struct ds4_suffix_node {
    int token_id;              /* token at this position (-1 for root) */
    uint32_t freq;             /* how many times this continuation appeared */
    uint32_t n_children;
    uint32_t cap_children;
    uint32_t best_child_idx;   /* cached index of highest-freq child (UINT32_MAX = invalid) */
    struct ds4_suffix_node *children;  /* sorted array by token_id */
} ds4_suffix_node;

#ifndef DS4_SUFFIX_STATS_DEFINED
#define DS4_SUFFIX_STATS_DEFINED
/* Statistics snapshot for telemetry. */
typedef struct ds4_suffix_stats {
    uint64_t node_count;
    uint64_t total_bytes;
    uint64_t query_count;
    uint64_t query_hits;
    uint64_t draft_tokens_produced;
    uint64_t draft_tokens_accepted;
    double   draft_score_total;
} ds4_suffix_stats;
#endif

/* The suffix tree handle. */
typedef struct ds4_suffix_tree {
    ds4_suffix_node root;
    uint64_t node_count;       /* current live node count (excl. root) */
    uint64_t node_budget;      /* max nodes allowed */
    uint64_t total_bytes;      /* estimated memory used */
    uint32_t max_depth;        /* max sequence length to store */
    /* telemetry counters */
    uint64_t query_count;
    uint64_t query_hits;
    uint64_t draft_tokens_produced;
    uint64_t draft_tokens_accepted;
    double   draft_score_total;
} ds4_suffix_tree;

/* Create a suffix tree with the given byte memory budget and max depth.
 * The byte budget is internally converted to a node budget based on
 * estimated per-node memory cost (~1.5 * sizeof(ds4_suffix_node)).
 * Returns NULL on allocation failure. */
ds4_suffix_tree *ds4_suffix_tree_alloc(uint64_t byte_budget,
                                        uint32_t max_depth);

/* Free a suffix tree and all its nodes. */
void ds4_suffix_tree_free(ds4_suffix_tree *tree);

/* Insert a token sequence into the tree.
 * Inserts suffixes starting from offsets [0, len-1] up to max_depth long.
 * Returns the number of new nodes created (0 if all paths existed). */
uint32_t ds4_suffix_tree_insert(ds4_suffix_tree *tree,
                                 const int *tokens, uint32_t len);

/* Append the final token of `tokens[0..len-1]` to a tree that already learned
 * tokens[0..len-2].  This extends only the suffix paths affected by the new
 * tail token, avoiding repeated frequency inflation for older prefixes. */
uint32_t ds4_suffix_tree_append(ds4_suffix_tree *tree,
                                 const int *tokens, uint32_t len);

/* Return the longest suffix of prefix[0..prefix_len-1] that exists in the
 * tree and has at least one continuation.  Returns 0 if no draftable match
 * exists. */
uint32_t ds4_suffix_tree_match_depth(ds4_suffix_tree *tree,
                                      const int *prefix, uint32_t prefix_len);

/* Query the tree for draft candidates given a prefix.
 * Traverses the tree from the root using the last `prefix_len` tokens.
 * Returns the match length `p` and fills `drafts[0..*draft_n-1]` with the
 * most frequent continuation path.
 * Sets *draft_n to the number of draft tokens proposed (may be 0).
 * Returns the match length p (0 if no match).
 *
 * Probability estimation: at each continuation step, prob *= child_freq / parent_freq.
 * Stops early if prob drops below min_prob (0.0 = disabled).
 * Accumulated score (sum of per-token probs) is written to *out_score. */
uint32_t ds4_suffix_tree_query(ds4_suffix_tree *tree,
                                const int *prefix, uint32_t prefix_len,
                                int *drafts, uint32_t max_drafts,
                                uint32_t *draft_n,
                                float min_prob, float *out_score);

/* Prune low-frequency nodes when the tree exceeds its budget.
 * Decrement all frequencies by 1, then remove leaf nodes with freq == 0.
 * Returns the number of nodes removed. */
uint32_t ds4_suffix_tree_prune(ds4_suffix_tree *tree);

/* Reset the tree (clear all nodes, keep the handle and settings). */
void ds4_suffix_tree_reset(ds4_suffix_tree *tree);

/* Get telemetry statistics. */
void ds4_suffix_tree_stats(const ds4_suffix_tree *tree, ds4_suffix_stats *out);

#endif /* DS4_SUFFIX_TREE_H */
