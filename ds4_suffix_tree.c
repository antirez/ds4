/* ds4_suffix_tree.c -- Suffix trie for model-free speculative decoding.
 *
 * Reference: SuffixDecoding (arxiv:2411.04975).
 *
 * This is a compact trie (not a full suffix tree) where:
 * - Insert adds all suffixes of the token sequence up to max_depth long.
 * - Query traverses the trie from the root using the last N context tokens,
 *   then follows the most frequent continuation path to propose drafts.
 * - Prune removes low-frequency leaf nodes to stay within memory budget.
 *
 * The trie uses sorted arrays for children (binary search lookup),
 * which is cache-friendly and allocation-efficient for the small fanout
 * typical of LLM token sequences.
 */

#include "ds4_suffix_tree.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---------- internal helpers ---------- */

#define ST_MALLOC(sz) malloc(sz)
#define ST_CALLOC(n, sz) calloc((n), (sz))
#define ST_REALLOC(ptr, sz) realloc((ptr), (sz))
#define ST_FREE(ptr) free(ptr)

/* Binary-search a sorted children array for token_id.
 * Returns the index if found, or -1. */
static int find_child(const ds4_suffix_node *node, int token_id) {
    uint32_t lo = 0, hi = node->n_children;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (node->children[mid].token_id == token_id) return (int)mid;
        if (node->children[mid].token_id < token_id) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}

/* Insert a child into the sorted array, growing if needed.
 * Returns a pointer to the new child node (inside the parent's array). */
static ds4_suffix_node *insert_child(ds4_suffix_node *parent, int token_id) {
    if (parent->n_children >= parent->cap_children) {
        uint32_t new_cap = parent->cap_children == 0 ? 4 :
                           parent->cap_children * 2;
        ds4_suffix_node *new_arr = (ds4_suffix_node *)ST_REALLOC(
            parent->children, (size_t)new_cap * sizeof(ds4_suffix_node));
        if (!new_arr) return NULL;
        parent->children = new_arr;
        parent->cap_children = new_cap;
    }

    /* Find insertion point to keep sorted order. */
    uint32_t pos = parent->n_children;
    for (uint32_t i = 0; i < parent->n_children; i++) {
        if (parent->children[i].token_id > token_id) {
            pos = i;
            break;
        }
    }

    /* Shift elements right. */
    memmove(parent->children + pos + 1,
            parent->children + pos,
            (size_t)(parent->n_children - pos) * sizeof(ds4_suffix_node));

    /* Initialize new child. */
    ds4_suffix_node *child = &parent->children[pos];
    memset(child, 0, sizeof(*child));
    child->token_id = token_id;
    child->freq = 0;
    child->n_children = 0;
    child->cap_children = 0;
    child->best_child_idx = UINT32_MAX;
    child->children = NULL;
    parent->n_children++;
    /* Invalidate cached best child: the memmove may have shifted the
     * old best_child_idx to a different position.  The next query will
     * rescan and repopulate it. */
    parent->best_child_idx = UINT32_MAX;
    return child;
}

/* Find or create a child node for token_id in parent. */
static ds4_suffix_node *ensure_child(ds4_suffix_node *parent, int token_id,
                                      int *created) {
    int idx = find_child(parent, token_id);
    if (idx >= 0) {
        *created = 0;
        return &parent->children[idx];
    }
    *created = 1;
    return insert_child(parent, token_id);
}

/* After bumping child->freq, check if it should become the cached best.
 * child_idx is the index of the child in parent->children. */
static void update_best_child(ds4_suffix_node *parent, uint32_t child_idx) {
    if (parent->best_child_idx >= parent->n_children ||
        parent->children[child_idx].freq >
            parent->children[parent->best_child_idx].freq) {
        parent->best_child_idx = child_idx;
    }
}

/* Recursively free all children. */
static void free_node_recursive(ds4_suffix_node *node) {
    if (!node) return;
    for (uint32_t i = 0; i < node->n_children; i++) {
        free_node_recursive(&node->children[i]);
    }
    ST_FREE(node->children);
    node->children = NULL;
    node->n_children = 0;
    node->cap_children = 0;
}

/* Decrement all node frequencies by 1 (clamp at 0) via iterative DFS.
 * Uses a dynamically-allocated stack to handle wide trees without
 * silently dropping nodes. */
static void decrement_all_freqs(ds4_suffix_node *root) {
    /* Start with a reasonable on-stack buffer; malloc a larger one if needed. */
    uint32_t stack_cap = 1024;
    ds4_suffix_node **stack = (ds4_suffix_node **)ST_MALLOC(
        (size_t)stack_cap * sizeof(ds4_suffix_node *));
    if (!stack) return;  /* OOM: skip aging (not fatal) */

    int sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
        sp--;
        ds4_suffix_node *node = stack[sp];
        /* Decrement freq, clamp at 0. */
        if (node->freq > 0) node->freq--;

        /* Grow stack if needed to fit all children. */
        if (node->n_children > 0 &&
            (uint32_t)(sp + (int)node->n_children) > stack_cap) {
            uint32_t new_cap = stack_cap * 2;
            if (new_cap < stack_cap + node->n_children)
                new_cap = stack_cap + node->n_children;
            ds4_suffix_node **new_stack = (ds4_suffix_node **)ST_REALLOC(
                stack, (size_t)new_cap * sizeof(ds4_suffix_node *));
            if (!new_stack) break;  /* OOM: stop early */
            stack = new_stack;
            stack_cap = new_cap;
        }

        /* Push children onto stack. */
        for (uint32_t i = 0; i < node->n_children; i++) {
            if (sp < (int)stack_cap) {
                stack[sp++] = &node->children[i];
            }
        }
    }

    ST_FREE(stack);
}

/* Recursively remove leaf nodes with freq==0.
 * Returns the number of nodes removed. */
static uint32_t remove_zero_leaves(ds4_suffix_node *node) {
    if (!node) return 0;
    uint32_t removed = 0;

    /* Process children back-to-front. */
    uint32_t i = node->n_children;
    while (i > 0) {
        i--;
        ds4_suffix_node *child = &node->children[i];

        /* Recurse first. */
        removed += remove_zero_leaves(child);

        /* After recursing, if this child is now a leaf with freq==0, remove. */
        if (child->n_children == 0 && child->freq == 0) {
            ST_FREE(child->children);
            child->children = NULL;
            /* Shift remaining children left. */
            if (i < node->n_children - 1) {
                memmove(node->children + i,
                        node->children + i + 1,
                        (size_t)(node->n_children - i - 1) *
                        sizeof(ds4_suffix_node));
            }
            node->n_children--;
            /* Invalidate cached best child after structural change. */
            node->best_child_idx = UINT32_MAX;
            removed++;
        }
    }
    return removed;
}

static void prune_toward_budget(ds4_suffix_tree *tree) {
    if (!tree) return;
    for (int pass = 0; tree->node_count > tree->node_budget && pass < 16; pass++) {
        uint32_t removed = ds4_suffix_tree_prune(tree);
        if (removed == 0) break;
    }
}

/* ---------- public API ---------- */

ds4_suffix_tree *ds4_suffix_tree_alloc(uint64_t byte_budget,
                                         uint32_t max_depth) {
    ds4_suffix_tree *tree = (ds4_suffix_tree *)ST_CALLOC(1, sizeof(*tree));
    if (!tree) return NULL;
    memset(&tree->root, 0, sizeof(tree->root));
    tree->root.token_id = -1;
    tree->root.freq = 0;
    tree->node_count = 0;
    /* Convert byte budget to node budget.  Each node uses ~sizeof(node) plus
     * average ~50% children-array overhead.  We use a conservative 1.5x. */
    uint64_t node_size_est = (uint64_t)(sizeof(ds4_suffix_node) * 3 / 2);
    tree->node_budget = byte_budget / (node_size_est > 0 ? node_size_est : 1);
    if (tree->node_budget < 1024) tree->node_budget = 1024;  /* floor */
    tree->total_bytes = sizeof(*tree);
    tree->max_depth = max_depth ? max_depth : 1;
    tree->query_count = 0;
    tree->query_hits = 0;
    tree->draft_tokens_produced = 0;
    tree->draft_tokens_accepted = 0;
    return tree;
}

void ds4_suffix_tree_free(ds4_suffix_tree *tree) {
    if (!tree) return;
    free_node_recursive(&tree->root);
    ST_FREE(tree);
}

uint32_t ds4_suffix_tree_insert(ds4_suffix_tree *tree,
                                 const int *tokens, uint32_t len) {
    if (!tree || !tokens || len == 0) return 0;
    uint32_t created = 0;
    uint32_t max_d = tree->max_depth;
    uint64_t prune_slack = tree->node_budget / 16;
    uint64_t min_slack = (uint64_t)max_d * 4;
    if (min_slack < 64) min_slack = 64;
    if (prune_slack < min_slack) prune_slack = min_slack;
    uint64_t prune_threshold = tree->node_budget + prune_slack;

    /* Insert all suffixes starting at each position.
     * Each suffix is truncated to max_depth tokens. */
    for (uint32_t start = 0; start < len; start++) {
        uint32_t suffix_len = len - start;
        if (suffix_len > max_d) suffix_len = max_d;

        ds4_suffix_node *cur = &tree->root;
        for (uint32_t j = 0; j < suffix_len; j++) {
            int tok = tokens[start + j];
            int did_create = 0;
            ds4_suffix_node *child = ensure_child(cur, tok, &did_create);
            if (!child) break;  /* allocation failure, skip rest */
            child->freq++;
            if (did_create) {
                tree->node_count++;
                tree->total_bytes += sizeof(ds4_suffix_node);
                created++;
            } else {
                /* Existing child: update cached best child if needed. */
                int cidx = find_child(cur, tok);
                if (cidx >= 0) update_best_child(cur, (uint32_t)cidx);
            }
            cur = child;
        }
        if (tree->node_count > prune_threshold) {
            prune_toward_budget(tree);
        }
    }

    /* Auto-prune if we exceeded budget. */
    if (tree->node_count > tree->node_budget) {
        prune_toward_budget(tree);
    }

    return created;
}

uint32_t ds4_suffix_tree_append(ds4_suffix_tree *tree,
                                 const int *tokens, uint32_t len) {
    if (!tree || !tokens || len == 0) return 0;
    if (len == 1) return ds4_suffix_tree_insert(tree, tokens, len);

    uint32_t created = 0;
    const uint32_t max_d = tree->max_depth;
    const uint32_t new_idx = len - 1;
    uint32_t start_min = 0;
    if (max_d > 0 && len > max_d) start_min = len - max_d;

    for (uint32_t start = start_min; start <= new_idx; start++) {
        ds4_suffix_node *cur = &tree->root;
        uint32_t j = start;

        /* Existing suffix prefixes are already counted.  Walk them without
         * bumping frequencies, then count only the newly appended tail edge. */
        for (; j < new_idx; j++) {
            int idx = find_child(cur, tokens[j]);
            if (idx < 0) break;
            cur = &cur->children[idx];
        }

        if (j < new_idx) {
            for (; j <= new_idx; j++) {
                int did_create = 0;
                ds4_suffix_node *child =
                    ensure_child(cur, tokens[j], &did_create);
                if (!child) break;
                child->freq++;
                if (did_create) {
                    tree->node_count++;
                    tree->total_bytes += sizeof(ds4_suffix_node);
                    created++;
                } else {
                    int cidx = find_child(cur, tokens[j]);
                    if (cidx >= 0) update_best_child(cur, (uint32_t)cidx);
                }
                cur = child;
            }
        } else {
            int did_create = 0;
            ds4_suffix_node *child =
                ensure_child(cur, tokens[new_idx], &did_create);
            if (!child) continue;
            child->freq++;
            if (did_create) {
                tree->node_count++;
                tree->total_bytes += sizeof(ds4_suffix_node);
                created++;
            } else {
                int cidx = find_child(cur, tokens[new_idx]);
                if (cidx >= 0) update_best_child(cur, (uint32_t)cidx);
            }
        }

        if (tree->node_count > tree->node_budget) {
            prune_toward_budget(tree);
        }
    }

    if (tree->node_count > tree->node_budget) {
        prune_toward_budget(tree);
    }

    return created;
}

static uint32_t suffix_tree_find_match(ds4_suffix_tree *tree,
                                        const int *prefix, uint32_t prefix_len,
                                        ds4_suffix_node **match_out) {
    if (match_out) *match_out = NULL;
    if (!tree || !prefix || prefix_len == 0) return 0;

    ds4_suffix_node *match = NULL;
    uint32_t match_depth = 0;

    for (uint32_t start = prefix_len > tree->max_depth ?
                          prefix_len - tree->max_depth : 0;
         start < prefix_len; start++) {
        ds4_suffix_node *cur = &tree->root;
        uint32_t depth = 0;
        uint32_t j = start;
        for (; j < prefix_len; j++) {
            int idx = find_child(cur, prefix[j]);
            if (idx < 0) break;
            cur = &cur->children[idx];
            depth++;
        }
        if (j == prefix_len && depth > match_depth && cur->n_children > 0) {
            match = cur;
            match_depth = depth;
        }
    }

    if (match_out) *match_out = match;
    return match ? match_depth : 0;
}

uint32_t ds4_suffix_tree_match_depth(ds4_suffix_tree *tree,
                                      const int *prefix, uint32_t prefix_len) {
    return suffix_tree_find_match(tree, prefix, prefix_len, NULL);
}

uint32_t ds4_suffix_tree_query(ds4_suffix_tree *tree,
                                const int *prefix, uint32_t prefix_len,
                                int *drafts, uint32_t max_drafts,
                                uint32_t *draft_n,
                                float min_prob, float *out_score) {
    if (!tree || !prefix || !drafts || !draft_n) {
        if (draft_n) *draft_n = 0;
        if (out_score) *out_score = 0.0f;
        return 0;
    }
    *draft_n = 0;
    if (out_score) *out_score = 0.0f;
    tree->query_count++;

    if (prefix_len == 0 || max_drafts == 0) return 0;

    ds4_suffix_node *match = NULL;
    uint32_t match_depth =
        suffix_tree_find_match(tree, prefix, prefix_len, &match);
    if (!match || match_depth == 0) return 0;

    /* Follow the most frequent continuation from the match node.
     * Estimate probability at each step: prob *= child_freq / parent_freq.
     * Stop early if prob drops below min_prob (ArcticInference-style filtering). */
    uint32_t d = 0;
    float prob = 1.0f;
    float score = 0.0f;
    ds4_suffix_node *cur = match;
    while (d < max_drafts && cur->n_children > 0) {
        /* Use cached best child if valid, otherwise scan. */
        uint32_t best = cur->best_child_idx;
        if (best >= cur->n_children) {
            best = 0;
            uint32_t best_freq = cur->children[0].freq;
            for (uint32_t i = 1; i < cur->n_children; i++) {
                if (cur->children[i].freq > best_freq) {
                    best_freq = cur->children[i].freq;
                    best = i;
                }
            }
            /* Cache the result for future queries. */
            cur->best_child_idx = best;
        }

        /* Probability estimation: P(next_token | context) ≈ child_freq / parent_freq */
        if (cur->freq > 0) {
            prob *= (float)cur->children[best].freq / (float)cur->freq;
        }

        /* Stop if confidence too low (min_prob = 0 disables filtering). */
        if (min_prob > 0.0f && prob < min_prob) break;

        drafts[d++] = cur->children[best].token_id;
        score += prob;
        cur = &cur->children[best];
    }

    if (d == 0) return match_depth;
    *draft_n = d;
    if (out_score) *out_score = score;
    tree->query_hits++;
    tree->draft_tokens_produced += d;
    tree->draft_score_total += (double)score;
    return match_depth;
}

uint32_t ds4_suffix_tree_prune(ds4_suffix_tree *tree) {
    if (!tree) return 0;

    /* Phase 1: decrement all frequencies by 1 (aging). */
    decrement_all_freqs(&tree->root);

    /* Phase 2: remove leaf nodes with freq==0. */
    uint32_t removed = remove_zero_leaves(&tree->root);

    tree->node_count -= removed;
    tree->total_bytes -= (uint64_t)removed * sizeof(ds4_suffix_node);
    return removed;
}

void ds4_suffix_tree_reset(ds4_suffix_tree *tree) {
    if (!tree) return;
    free_node_recursive(&tree->root);
    memset(&tree->root, 0, sizeof(tree->root));
    tree->root.token_id = -1;
    tree->node_count = 0;
    tree->total_bytes = sizeof(*tree);
    /* Keep telemetry counters intact for reporting. */
}

void ds4_suffix_tree_stats(const ds4_suffix_tree *tree,
                             ds4_suffix_stats *out) {
    if (!out) return;
    if (!tree) { memset(out, 0, sizeof(*out)); return; }
    out->node_count = tree->node_count;
    out->total_bytes = tree->total_bytes;
    out->query_count = tree->query_count;
    out->query_hits = tree->query_hits;
    out->draft_tokens_produced = tree->draft_tokens_produced;
    out->draft_tokens_accepted = tree->draft_tokens_accepted;
    out->draft_score_total = tree->draft_score_total;
}
