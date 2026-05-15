#include "ds4_prompt_cache.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

/* =========================================================================
 * Prompt Tokenization Cache.
 *
 * Avoids re-tokenizing the same tool schemas on every request.  When the tool
 * set is constant across concurrent sessions, rendering and tokenizing 30+
 * tools into DSML format takes measurable time on the hot path.
 *
 * For a typical system prompt + tools (~20K tokens), tokenization alone
 * takes 5-15ms.  With multiple agents making requests, that's significant waste
 * per round of requests.  More importantly, it means the rendered prompt text
 * is always identical, so KV prefix matching works perfectly.
 *
 * The cache is append-only (no eviction) since there are at most 1-2 distinct
 * tool sets in practice.  If it fills up, new entries just don't get cached.
 * ========================================================================= */

void ds4_prompt_cache_init(ds4_prompt_cache *cache) {
    memset(cache, 0, sizeof(*cache));
    pthread_mutex_init(&cache->mu, NULL);
}

void ds4_prompt_cache_free(ds4_prompt_cache *cache) {
    pthread_mutex_lock(&cache->mu);
    for (int i = 0; i < cache->count; i++) {
        free(cache->entries[i].tools_json);
        free(cache->entries[i].rendered_tools);
        ds4_tokens_free(&cache->entries[i].tokens);
    }
    pthread_mutex_unlock(&cache->mu);
    pthread_mutex_destroy(&cache->mu);
    memset(cache, 0, sizeof(*cache));
}

uint64_t ds4_prompt_cache_hash(const char *s) {
    /* FNV-1a 64-bit. */
    uint64_t h = 14695981039346656037ULL;
    if (!s) return h;
    for (; *s; s++) {
        h ^= (uint64_t)(unsigned char)*s;
        h *= 1099511628211ULL;
    }
    return h;
}

ds4_prompt_cache_entry *ds4_prompt_cache_find(ds4_prompt_cache *cache,
                                              const char *tools_json) {
    if (!tools_json || !tools_json[0]) return NULL;
    uint64_t hash = ds4_prompt_cache_hash(tools_json);

    pthread_mutex_lock(&cache->mu);
    for (int i = 0; i < cache->count; i++) {
        if (cache->entries[i].valid && cache->entries[i].tools_hash == hash &&
            strcmp(cache->entries[i].tools_json, tools_json) == 0) {
            cache->entries[i].hits++;
            cache->total_hits++;
            pthread_mutex_unlock(&cache->mu);
            return &cache->entries[i];
        }
    }
    cache->total_misses++;
    pthread_mutex_unlock(&cache->mu);
    return NULL;
}

ds4_prompt_cache_entry *ds4_prompt_cache_insert(ds4_prompt_cache *cache,
                                                const char *tools_json,
                                                char *rendered_tools,
                                                const ds4_tokens *tokens,
                                                int tools_token_count) {
    pthread_mutex_lock(&cache->mu);
    if (cache->count >= DS4_PROMPT_CACHE_MAX_ENTRIES) {
        /* Cache full — just don't cache.  In practice this never happens
         * because most deployments have 1-2 distinct tool sets. */
        pthread_mutex_unlock(&cache->mu);
        free(rendered_tools);
        return NULL;
    }

    ds4_prompt_cache_entry *entry = &cache->entries[cache->count++];
    entry->tools_hash = ds4_prompt_cache_hash(tools_json);
    entry->tools_json = strdup(tools_json);
    entry->rendered_tools = rendered_tools;
    ds4_tokens_copy(&entry->tokens, tokens);
    entry->tools_token_count = tools_token_count;
    entry->hits = 0;
    entry->valid = true;
    pthread_mutex_unlock(&cache->mu);

    fprintf(stderr, "ds4_pool: cached tool tokenization (%d tokens, hash=%016llx)\n",
            tools_token_count, (unsigned long long)entry->tools_hash);
    return entry;
}
