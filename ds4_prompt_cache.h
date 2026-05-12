#ifndef DS4_PROMPT_CACHE_H
#define DS4_PROMPT_CACHE_H

/* Prompt tokenization cache.
 *
 * Clients often send identical tool schemas on every request (~30 tools,
 * ~15-20K tokens once rendered to DSML).  The server re-renders and
 * re-tokenizes this on every request even though it rarely changes.
 * This cache eliminates that cost.
 *
 * Design:
 *   - Key: FNV-1a hash of the raw tools JSON string
 *   - Value: pre-computed ds4_tokens for the full rendered prompt prefix
 *   - The cache holds the rendered prompt text and tokens for reuse
 *
 * The cache is intentionally small (8-16 entries) since in practice there are
 * only 1-2 distinct tool schema sets across all sessions.
 *
 * Thread safety: The cache is accessed from multiple client threads (in
 * parse_chat_request) and protected by an internal pthread_mutex.  The lock is
 * only held for the duration of a hash lookup + strcmp, so contention is
 * negligible with 1-2 distinct tool sets.
 */

#include "ds4.h"
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>

#define DS4_PROMPT_CACHE_MAX_ENTRIES 16

typedef struct {
    /* Key: hash of the tools JSON string. */
    uint64_t tools_hash;
    /* Original tools JSON for collision verification. */
    char *tools_json;
    /* Cached rendered tool prompt prefix (the "## Tools\n\n..." section). */
    char *rendered_tools;
    /* Pre-tokenized tool section. */
    ds4_tokens tokens;
    /* Token count of JUST the tools portion (for prefix split). */
    int tools_token_count;
    /* How many times this entry was reused. */
    uint64_t hits;
    bool valid;
} ds4_prompt_cache_entry;

typedef struct {
    ds4_prompt_cache_entry entries[DS4_PROMPT_CACHE_MAX_ENTRIES];
    int count;
    uint64_t total_hits;
    uint64_t total_misses;
    pthread_mutex_t mu;  /* protects all fields — accessed from client threads */
} ds4_prompt_cache;

/* Initialize the cache. */
void ds4_prompt_cache_init(ds4_prompt_cache *cache);

/* Free all cache entries. */
void ds4_prompt_cache_free(ds4_prompt_cache *cache);

/* Look up cached tokenization for a tools string.
 * Returns the entry if found, NULL if miss. */
ds4_prompt_cache_entry *ds4_prompt_cache_find(ds4_prompt_cache *cache,
                                              const char *tools_json);

/* Insert a new entry.  The cache takes ownership of rendered_tools. */
ds4_prompt_cache_entry *ds4_prompt_cache_insert(ds4_prompt_cache *cache,
                                                const char *tools_json,
                                                char *rendered_tools,
                                                const ds4_tokens *tokens,
                                                int tools_token_count);

/* FNV-1a hash of a string — fast, good distribution for cache keys. */
uint64_t ds4_prompt_cache_hash(const char *s);

#endif
