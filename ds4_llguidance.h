#ifndef DS4_LLGUIDANCE_H
#define DS4_LLGUIDANCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4.h"

typedef struct ds4_llguidance ds4_llguidance;
typedef struct ds4_llguidance_cache ds4_llguidance_cache;

/* The server owns one cache per engine; matchers retain it independently.
 * The engine must outlive both the cache and all matchers created from it.
 * Tokenizer construction is lazy and serialized; separate matchers may be
 * created and used on different threads. */
ds4_llguidance_cache *ds4_llguidance_cache_create(ds4_engine *e);
void ds4_llguidance_cache_free(ds4_llguidance_cache *cache);

bool ds4_llguidance_available(void);

ds4_llguidance *ds4_llguidance_create(ds4_llguidance_cache *cache,
                                      const char *constraint_type,
                                      const char *constraint_data,
                                      char *err,
                                      size_t errlen);
void ds4_llguidance_free(ds4_llguidance *g);

int ds4_llguidance_sample(ds4_llguidance *g,
                          ds4_session *s,
                          float temperature,
                          int top_k,
                          float top_p,
                          float min_p,
                          uint64_t *rng,
                          char *err,
                          size_t errlen);
bool ds4_llguidance_accept(ds4_llguidance *g,
                           ds4_engine *e,
                           int token,
                           char *err,
                           size_t errlen);

#endif
