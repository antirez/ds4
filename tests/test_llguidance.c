/* Exercise the real llguidance adapter with tiny, differently numbered byte
 * vocabularies. No model, inference backend, or tokenizer cache globals. */
#include "../ds4_llguidance.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef DS4_USE_LLGUIDANCE
#include <pthread.h>

struct ds4_engine { unsigned permutation; int token_reads; };
struct ds4_session { int next_token; };

int ds4_engine_vocab_size(ds4_engine *e) { (void)e; return 257; }
int ds4_token_eos(ds4_engine *e) { (void)e; return 256; }

char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    const char *eos = "<｜end▁of▁sentence｜>";
    *len = token == 256 ? strlen(eos) : 1;
    char *text = malloc(*len + 1);
    assert(text);
    if (token == 256) memcpy(text, eos, *len);
    else text[0] = (char)((unsigned)token ^ e->permutation);
    text[*len] = '\0';
    e->token_reads++;
    return text;
}

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    out->len = (int)strlen(text);
    out->v = malloc((size_t)out->len * sizeof(*out->v));
    assert(out->v || !out->len);
    for (int i = 0; i < out->len; i++)
        out->v[i] = (unsigned char)text[i] ^ e->permutation;
}

void ds4_tokens_free(ds4_tokens *tokens) {
    free(tokens->v);
    memset(tokens, 0, sizeof(*tokens));
}

static bool bit(const uint32_t *mask, size_t words, int token) {
    return mask && (size_t)token / 32 < words &&
        (mask[token / 32] & (UINT32_C(1) << (token % 32)));
}

/* Probe the adapter's mask using a requested token. Sampler correctness is
 * covered separately in test_sampling.c. */
int ds4_session_sample_masked(ds4_session *s, float temperature, int top_k,
                              float top_p, float min_p,
                              const uint32_t *allow, size_t allow_words,
                              const uint32_t *deny, size_t deny_words,
                              uint64_t *rng) {
    (void)temperature; (void)top_k; (void)top_p; (void)min_p; (void)rng;
    int token = s->next_token;
    return bit(allow, allow_words, token) && !bit(deny, deny_words, token) ? token : -1;
}

static ds4_llguidance *new_matcher(ds4_llguidance_cache *cache,
                                   const char *type, const char *data) {
    char err[256] = {0};
    ds4_llguidance *g = ds4_llguidance_create(cache, type, data, err, sizeof(err));
    if (!g) fprintf(stderr, "%s\n", err);
    assert(g);
    return g;
}

static int probe(ds4_llguidance *g, ds4_engine *e, unsigned char ch) {
    ds4_session session = {(int)(ch ^ e->permutation)};
    char err[256] = {0};
    uint64_t rng = 1;
    return ds4_llguidance_sample(g, &session, 0, 0, 1, 0, &rng, err, sizeof(err));
}

static void consume(ds4_llguidance *g, ds4_engine *e, const char *text) {
    char err[256] = {0};
    for (; *text; text++) {
        int token = probe(g, e, (unsigned char)*text);
        assert(token >= 0);
        assert(ds4_llguidance_accept(g, e, token, err, sizeof(err)));
    }
}

static void *create_on_thread(void *cache) {
    return new_matcher(cache, "regex", "OK");
}

static void test_cache_lifetime(void) {
    ds4_engine a = {0}, b = {127, 0};
    ds4_llguidance_cache *cache_a = ds4_llguidance_cache_create(&a);
    ds4_llguidance_cache *cache_b = ds4_llguidance_cache_create(&b);
    assert(cache_a && cache_b);
    pthread_t threads[4];
    ds4_llguidance *matchers[4];
    for (int i = 0; i < 4; i++)
        assert(pthread_create(&threads[i], NULL, create_on_thread, cache_a) == 0);
    for (int i = 0; i < 4; i++) {
        void *result;
        assert(pthread_join(threads[i], &result) == 0);
        matchers[i] = result;
    }
    assert(a.token_reads == 2 * 257); /* One lazy vocabulary build. */
    char err[256] = {0};
    assert(!ds4_llguidance_create(cache_a, "regex", "[", err, sizeof(err)));
    ds4_llguidance *json_a = new_matcher(cache_a, "json_object", "");
    ds4_llguidance_cache_free(cache_a); /* Matchers retain the vocabulary. */
    ds4_llguidance *json_b = new_matcher(cache_b, "json_object", "");
    ds4_llguidance_cache_free(cache_b);
    for (int i = 0; i < 4; i++) {
        assert(probe(matchers[i], &a, 'X') == -1);
        consume(matchers[i], &a, "OK");
        ds4_llguidance_free(matchers[i]);
    }
    assert(probe(json_a, &a, ' ') == -1);
    assert(probe(json_b, &b, ' ') == -1);
    consume(json_b, &b, "{ }");
    ds4_llguidance_free(json_b);
    consume(json_a, &a, "{ }"); /* Other engine destruction changes nothing. */
    ds4_llguidance_free(json_a);
}
#endif

int main(void) {
#ifdef DS4_USE_LLGUIDANCE
    assert(ds4_llguidance_available());
    test_cache_lifetime();
    puts("llguidance tests: enabled, cache lifetime and concurrent creation OK");
#else
    assert(!ds4_llguidance_available());
    assert(!ds4_llguidance_cache_create(NULL));
    char err[128] = {0};
    assert(!ds4_llguidance_create(NULL, "regex", "OK", err, sizeof(err)));
    assert(strstr(err, "LLGUIDANCE=1"));
    puts("llguidance tests: disabled, unavailable-feature errors OK");
#endif
    ds4_llguidance_free(NULL);
    ds4_llguidance_cache_free(NULL);
    return 0;
}
