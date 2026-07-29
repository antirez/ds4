#ifndef K3_TOKENIZER_H
#define K3_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

enum {
    K3_TOKEN_BOS = 163584,
    K3_TOKEN_EOS = 163585,        /* Literal [EOS] control token. */
    K3_TOKEN_END_OF_MSG = 163586, /* config.json generation EOS. */
    K3_TOKEN_OPEN = 163587,
    K3_TOKEN_CLOSE = 163588,
    K3_TOKEN_SEP = 163589,
    K3_TOKEN_EOT = 163593,
};

typedef struct k3_tokenizer k3_tokenizer;

typedef struct {
    int32_t *data;
    uint32_t count;
    uint32_t capacity;
} k3_tokens;

int k3_tokenizer_open(k3_tokenizer **out,
                      const char *model_directory,
                      char *err,
                      size_t errcap);
void k3_tokenizer_close(k3_tokenizer *tokenizer);

void k3_tokens_free(k3_tokens *tokens);

/* Ordinary text never interprets checkpoint control-token spellings. */
int k3_tokenizer_encode(k3_tokenizer *tokenizer,
                        const char *text,
                        k3_tokens *out,
                        char *err,
                        size_t errcap);
int k3_tokenizer_encode_rendered(k3_tokenizer *tokenizer,
                                 const char *text,
                                 k3_tokens *out,
                                 char *err,
                                 size_t errcap);

/* Append XTML segments without clearing out, for the shared DwarfStar chat API. */
int k3_tokenizer_append_text(k3_tokenizer *tokenizer,
                             const char *text,
                             k3_tokens *out,
                             char *err,
                             size_t errcap);
int k3_tokenizer_append_thinking_effort(k3_tokenizer *tokenizer,
                                        const char *effort,
                                        k3_tokens *out,
                                        char *err,
                                        size_t errcap);
int k3_tokenizer_append_message(k3_tokenizer *tokenizer,
                                const char *role,
                                const char *content,
                                k3_tokens *out,
                                char *err,
                                size_t errcap);
int k3_tokenizer_append_tool_declaration(k3_tokenizer *tokenizer,
                                         const char *tools_json,
                                         k3_tokens *out,
                                         char *err,
                                         size_t errcap);
int k3_tokenizer_append_tool_result(k3_tokenizer *tokenizer,
                                    const char *tool,
                                    uint32_t index,
                                    const char *content,
                                    k3_tokens *out,
                                    char *err,
                                    size_t errcap);
int k3_tokenizer_append_assistant_close(k3_tokenizer *tokenizer,
                                        k3_tokens *out,
                                        char *err,
                                        size_t errcap);
int k3_tokenizer_append_assistant_prefix(k3_tokenizer *tokenizer,
                                         int thinking,
                                         k3_tokens *out,
                                         char *err,
                                         size_t errcap);

/*
 * Encode one user turn with the official XTML segmentation. thinking=true
 * also emits the checkpoint's default thinking_effort=max system message.
 */
int k3_tokenizer_encode_chat(k3_tokenizer *tokenizer,
                             const char *prompt,
                             int thinking,
                             k3_tokens *out,
                             char *err,
                             size_t errcap);

/*
 * Return the byte sequence represented by one token. The pointer remains
 * owned by the tokenizer. Unknown reserved tokens return an empty sequence.
 */
const uint8_t *k3_tokenizer_token_bytes(const k3_tokenizer *tokenizer,
                                        int32_t token,
                                        uint32_t *length);

#endif
