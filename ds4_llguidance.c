#include "ds4_llguidance.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DS4_USE_LLGUIDANCE
#include <pthread.h>
#include "llguidance.h"
#endif

struct ds4_llguidance {
#ifdef DS4_USE_LLGUIDANCE
    LlgTokenizer *tokenizer;
    LlgMatcher *matcher;
    ds4_llguidance_cache *cache;
    size_t mask_words;
    int eos_token;
    bool deny_leading_ws;
    bool started;
#else
    int unused;
#endif
};

bool ds4_llguidance_available(void) {
#ifdef DS4_USE_LLGUIDANCE
    return true;
#else
    return false;
#endif
}

#ifdef DS4_USE_LLGUIDANCE

struct ds4_llguidance_cache {
    ds4_engine *engine;
    pthread_mutex_t mu;
    size_t refs;
    LlgTokenizer *tokenizer;
    uint32_t *leading_ws_mask;
    size_t mask_words;
    int n_vocab;
};

ds4_llguidance_cache *ds4_llguidance_cache_create(ds4_engine *e) {
    if (!e) return NULL;
    ds4_llguidance_cache *cache = calloc(1, sizeof(*cache));
    if (!cache) return NULL;
    if (pthread_mutex_init(&cache->mu, NULL) != 0) {
        free(cache);
        return NULL;
    }
    cache->engine = e;
    cache->refs = 1;
    return cache;
}

void ds4_llguidance_cache_free(ds4_llguidance_cache *cache) {
    if (!cache) return;
    pthread_mutex_lock(&cache->mu);
    const bool last = --cache->refs == 0;
    pthread_mutex_unlock(&cache->mu);
    if (!last) return;
    if (cache->tokenizer) llg_free_tokenizer(cache->tokenizer);
    free(cache->leading_ws_mask);
    pthread_mutex_destroy(&cache->mu);
    free(cache);
}

static void set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static bool json_ws_byte(unsigned char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static bool bytes_all_json_ws(const char *p, size_t len) {
    if (!p || len == 0) return false;
    for (size_t i = 0; i < len; i++) {
        if (!json_ws_byte((unsigned char)p[i])) return false;
    }
    return true;
}

static bool bytes_have_non_json_ws(const char *p, size_t len) {
    if (!p) return false;
    for (size_t i = 0; i < len; i++) {
        if (!json_ws_byte((unsigned char)p[i])) return true;
    }
    return false;
}

static bool token_text_is_special(const char *p, size_t len) {
    static const char *specials[] = {
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
        "｜DSML｜",
    };
    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        size_t n = strlen(specials[i]);
        if (len == n && memcmp(p, specials[i], n) == 0) return true;
    }

    const unsigned char bar[] = {0xef, 0xbd, 0x9c};
    for (size_t i = 0; i + sizeof(bar) <= len; i++) {
        if (!memcmp(p + i, bar, sizeof(bar))) return true;
    }
    return false;
}

static bool constraint_uses_json_leading_ws_rule(const char *constraint_type) {
    return constraint_type &&
        (!strcmp(constraint_type, "json") ||
         !strcmp(constraint_type, "json_schema") ||
         !strcmp(constraint_type, "json_object"));
}

static void bitset_set(uint32_t *mask, int token) {
    mask[(uint32_t)token / 32u] |= UINT32_C(1) << ((uint32_t)token & 31u);
}

static bool bitset_get(const uint32_t *mask, size_t words, uint32_t token) {
    const size_t word = token / 32u;
    if (!mask || word >= words) return false;
    return (mask[word] & (UINT32_C(1) << (token & 31u))) != 0;
}

static bool mask_has_non_denied_token(const uint32_t *allow,
                                      size_t allow_words,
                                      const uint32_t *deny,
                                      size_t deny_words,
                                      int n_vocab) {
    if (!allow) return false;
    for (int i = 0; i < n_vocab; i++) {
        if (bitset_get(allow, allow_words, (uint32_t)i) &&
            !bitset_get(deny, deny_words, (uint32_t)i))
        {
            return true;
        }
    }
    return false;
}

static size_t ds4_llg_tokenize_fn(const void *user_data,
                                  const uint8_t *bytes,
                                  size_t bytes_len,
                                  uint32_t *output_tokens,
                                  size_t output_tokens_len) {
    ds4_engine *e = (ds4_engine *)user_data;
    char *text = malloc(bytes_len + 1);
    if (!text) return 0;
    memcpy(text, bytes, bytes_len);
    text[bytes_len] = '\0';

    ds4_tokens toks = {0};
    ds4_tokenize_text(e, text, &toks);
    free(text);

    const size_t n = toks.len < 0 ? 0 : (size_t)toks.len;
    const size_t copy = n < output_tokens_len ? n : output_tokens_len;
    for (size_t i = 0; i < copy; i++) output_tokens[i] = (uint32_t)toks.v[i];
    ds4_tokens_free(&toks);
    return n;
}

static LlgTokenizer *build_tokenizer(ds4_engine *e,
                                     uint32_t **leading_ws_mask_out,
                                     size_t *leading_ws_words_out,
                                     int *n_vocab_out,
                                     char *err,
                                     size_t errlen) {
    const int n_vocab = ds4_engine_vocab_size(e);
    if (n_vocab <= 0) {
        set_err(err, errlen, "llguidance tokenizer cannot use an empty vocabulary");
        return NULL;
    }

    size_t total = 0;
    uint32_t *token_lens = calloc((size_t)n_vocab, sizeof(token_lens[0]));
    if (!token_lens) {
        set_err(err, errlen, "out of memory");
        return NULL;
    }

    const size_t mask_words = ((size_t)n_vocab + 31u) / 32u;
    uint32_t *leading_ws = calloc(mask_words, sizeof(leading_ws[0]));
    if (!leading_ws) {
        free(token_lens);
        set_err(err, errlen, "out of memory");
        return NULL;
    }

    for (int i = 0; i < n_vocab; i++) {
        size_t len = 0;
        char *piece = ds4_token_text(e, i, &len);
        const bool special = token_text_is_special(piece, len);
        token_lens[i] = (uint32_t)(len + (special ? 1u : 0u));
        total += token_lens[i];
        if (!special && bytes_all_json_ws(piece, len)) bitset_set(leading_ws, i);
        free(piece);
    }

    uint8_t *token_bytes = malloc(total ? total : 1);
    if (!token_bytes) {
        free(leading_ws);
        free(token_lens);
        set_err(err, errlen, "out of memory");
        return NULL;
    }

    size_t off = 0;
    for (int i = 0; i < n_vocab; i++) {
        size_t len = 0;
        char *piece = ds4_token_text(e, i, &len);
        if (token_text_is_special(piece, len)) token_bytes[off++] = 0xffu;
        memcpy(token_bytes + off, piece, len);
        off += len;
        free(piece);
    }

    LlgTokenizerInit init = {0};
    init.vocab_size = (uint32_t)n_vocab;
    init.tok_eos = (uint32_t)ds4_token_eos(e);
    init.token_lens = token_lens;
    init.token_bytes = token_bytes;
    init.tokenize_assumes_string = true;
    init.tokenize_fn = ds4_llg_tokenize_fn;
    init.use_approximate_greedy_tokenize_fn = false;
    init.tokenize_user_data = e;
    init.slices = NULL;

    char llg_err[1024] = {0};
    LlgTokenizer *tok = llg_new_tokenizer(&init, llg_err, sizeof(llg_err));
    free(token_bytes);
    free(token_lens);
    if (!tok) {
        free(leading_ws);
        set_err(err, errlen, "llguidance tokenizer error: %s", llg_err);
        return NULL;
    }

    *leading_ws_mask_out = leading_ws;
    *leading_ws_words_out = mask_words;
    *n_vocab_out = n_vocab;
    return tok;
}

/* Each clone retains the immutable vocabulary data as well as the tokenizer. */
static LlgTokenizer *cached_tokenizer_clone(ds4_llguidance_cache *cache,
                                            char *err, size_t errlen) {
    pthread_mutex_lock(&cache->mu);
    if (!cache->tokenizer) {
        cache->tokenizer = build_tokenizer(cache->engine, &cache->leading_ws_mask,
                                           &cache->mask_words, &cache->n_vocab,
                                           err, errlen);
        if (!cache->tokenizer) {
            pthread_mutex_unlock(&cache->mu);
            return NULL;
        }
    }
    LlgTokenizer *clone = llg_clone_tokenizer(cache->tokenizer);
    if (clone) cache->refs++;
    pthread_mutex_unlock(&cache->mu);
    if (!clone) set_err(err, errlen, "llguidance tokenizer clone failed");
    return clone;
}

ds4_llguidance *ds4_llguidance_create(ds4_llguidance_cache *cache,
                                      const char *constraint_type,
                                      const char *constraint_data,
                                      char *err, size_t errlen) {
    if (!cache || !constraint_type || !constraint_type[0]) {
        set_err(err, errlen, "invalid structured output constraint");
        return NULL;
    }
    ds4_llguidance *g = calloc(1, sizeof(*g));
    if (!g) {
        set_err(err, errlen, "out of memory");
        return NULL;
    }
    g->tokenizer = cached_tokenizer_clone(cache, err, errlen);
    if (!g->tokenizer) goto bad;
    g->cache = cache;

    LlgConstraintInit init;
    llg_constraint_init_set_defaults(&init, g->tokenizer);
    const char *log_level = getenv("LLGUIDANCE_LOG_LEVEL");
    if (!log_level || !log_level[0]) log_level = getenv("DS4_LLGUIDANCE_LOG_LEVEL");
    if (log_level && log_level[0]) init.log_stderr_level = (uint32_t)atoi(log_level);

    g->matcher = llg_new_matcher(&init, constraint_type,
                                 constraint_data ? constraint_data : "");
    const char *llg_err = g->matcher ? llg_matcher_get_error(g->matcher) : "allocation failed";
    if (llg_err) {
        set_err(err, errlen, "llguidance grammar error: %s", llg_err);
        goto bad;
    }
    const size_t mask_bytes = llg_matcher_get_mask_byte_size(g->matcher);
    if (mask_bytes != cache->mask_words * sizeof(uint32_t)) {
        set_err(err, errlen, "llguidance mask size mismatch");
        goto bad;
    }
    g->mask_words = cache->mask_words;
    g->eos_token = ds4_token_eos(cache->engine);
    g->deny_leading_ws = constraint_uses_json_leading_ws_rule(constraint_type);
    return g;
bad:
    ds4_llguidance_free(g);
    return NULL;
}

void ds4_llguidance_free(ds4_llguidance *g) {
    if (!g) return;
    if (g->matcher) llg_free_matcher(g->matcher);
    if (g->tokenizer) llg_free_tokenizer(g->tokenizer);
    ds4_llguidance_cache_free(g->cache);
    free(g);
}

int ds4_llguidance_sample(ds4_llguidance *g,
                          ds4_session *s,
                          float temperature,
                          int top_k,
                          float top_p,
                          float min_p,
                          uint64_t *rng,
                          char *err,
                          size_t errlen) {
    if (!g || !g->matcher || !s) {
        set_err(err, errlen, "structured output decoder is not active");
        return -1;
    }
    if (llg_matcher_is_stopped(g->matcher)) return g->eos_token;
    if (llg_matcher_compute_mask(g->matcher) != 0) {
        set_err(err, errlen, "llguidance mask error: %s",
                llg_matcher_get_error(g->matcher));
        return -1;
    }
    const uint32_t *allow = llg_matcher_get_mask(g->matcher);
    if (!allow) {
        set_err(err, errlen, "llguidance did not return a token mask");
        return -1;
    }

    const uint32_t *deny = NULL;
    size_t deny_words = 0;
    if (g->deny_leading_ws &&
        !g->started &&
        mask_has_non_denied_token(allow, g->mask_words, g->cache->leading_ws_mask,
                                  g->cache->mask_words, g->cache->n_vocab))
    {
        deny = g->cache->leading_ws_mask;
        deny_words = g->cache->mask_words;
    }

    int token = ds4_session_sample_masked(s, temperature, top_k, top_p, min_p,
                                          allow, g->mask_words, deny,
                                          deny_words, rng);
    if (token < 0) set_err(err, errlen, "llguidance mask allowed no sampleable token");
    return token;
}

bool ds4_llguidance_accept(ds4_llguidance *g,
                           ds4_engine *e,
                           int token,
                           char *err,
                           size_t errlen) {
    if (!g || !g->matcher) return true;
    if (token < 0) return true;
    if (llg_matcher_consume_token(g->matcher, (uint32_t)token) != 0) {
        set_err(err, errlen, "llguidance consume error: %s",
                llg_matcher_get_error(g->matcher));
        return false;
    }
    if (g->deny_leading_ws && !g->started && e) {
        size_t len = 0;
        char *piece = ds4_token_text(e, token, &len);
        if (bytes_have_non_json_ws(piece, len)) g->started = true;
        free(piece);
    }
    return true;
}

#else

ds4_llguidance_cache *ds4_llguidance_cache_create(ds4_engine *e) {
    (void)e;
    return NULL;
}

void ds4_llguidance_cache_free(ds4_llguidance_cache *cache) {
    (void)cache;
}

ds4_llguidance *ds4_llguidance_create(ds4_llguidance_cache *cache,
                                      const char *constraint_type,
                                      const char *constraint_data,
                                      char *err,
                                      size_t errlen) {
    (void)cache;
    (void)constraint_type;
    (void)constraint_data;
    if (err && errlen) {
        snprintf(err, errlen,
                 "structured outputs require building ds4 with LLGUIDANCE=1");
    }
    return NULL;
}

void ds4_llguidance_free(ds4_llguidance *g) {
    (void)g;
}

int ds4_llguidance_sample(ds4_llguidance *g,
                          ds4_session *s,
                          float temperature,
                          int top_k,
                          float top_p,
                          float min_p,
                          uint64_t *rng,
                          char *err,
                          size_t errlen) {
    (void)g;
    (void)s;
    (void)temperature;
    (void)top_k;
    (void)top_p;
    (void)min_p;
    (void)rng;
    if (err && errlen) {
        snprintf(err, errlen,
                 "structured outputs require building ds4 with LLGUIDANCE=1");
    }
    return -1;
}

bool ds4_llguidance_accept(ds4_llguidance *g,
                           ds4_engine *e,
                           int token,
                           char *err,
                           size_t errlen) {
    (void)g;
    (void)e;
    (void)token;
    (void)err;
    (void)errlen;
    return true;
}

#endif
