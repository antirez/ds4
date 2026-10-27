#ifndef DS4_PROMPT_PREFIX_H
#define DS4_PROMPT_PREFIX_H

#include "ds4.h"

#include <stddef.h>

typedef enum {
    DS4_PROMPT_PREFIX_USER,
    DS4_PROMPT_PREFIX_ASSISTANT,
} ds4_prompt_prefix_role;

typedef struct {
    ds4_prompt_prefix_role role;
    char *content;
} ds4_prompt_prefix_turn;

typedef struct {
    ds4_prompt_prefix_turn *turns;
    size_t count;
} ds4_prompt_prefix;

int ds4_prompt_prefix_parse(ds4_prompt_prefix *out,
                            const char *data,
                            size_t len,
                            char *error,
                            size_t error_cap);
int ds4_prompt_prefix_load(ds4_prompt_prefix *out,
                           const char *path,
                           char *error,
                           size_t error_cap);
void ds4_prompt_prefix_free(ds4_prompt_prefix *prefix);

static inline void ds4_prompt_prefix_append(ds4_engine *engine,
                                             ds4_tokens *tokens,
                                             const ds4_prompt_prefix *prefix) {
    if (!prefix) return;
    for (size_t i = 0; i < prefix->count; i++) {
        const ds4_prompt_prefix_turn *turn = &prefix->turns[i];
        const bool assistant = turn->role == DS4_PROMPT_PREFIX_ASSISTANT;
        ds4_chat_append_message(engine, tokens,
                                assistant ? "assistant" : "user",
                                turn->content);
        if (assistant && !ds4_engine_is_glm_dsa(engine))
            ds4_tokens_push(tokens, ds4_token_eos(engine));
    }
}

#endif
