#ifndef PULSAR_KVSTORE_H
#define PULSAR_KVSTORE_H

#include "pulsar.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>


#define PULSAR_KVSTORE_FIXED_HEADER 48u
#define PULSAR_KVSTORE_DEFAULT_MB 4096
#define PULSAR_KVSTORE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)

#define PULSAR_KVSTORE_EXT_TOOL_MAP          (1u << 0)
#define PULSAR_KVSTORE_EXT_RESPONSES_VISIBLE (1u << 1)
#define PULSAR_KVSTORE_EXT_THINKING_VISIBLE  (1u << 2)
#define PULSAR_KVSTORE_EXT_SESSION_TITLE     (1u << 3)

typedef enum {
    PULSAR_KVSTORE_REASON_UNKNOWN   = 0,
    PULSAR_KVSTORE_REASON_COLD      = 1,
    PULSAR_KVSTORE_REASON_CONTINUED = 2,
    PULSAR_KVSTORE_REASON_EVICT     = 3,
    PULSAR_KVSTORE_REASON_SHUTDOWN  = 4,
    PULSAR_KVSTORE_REASON_AGENT_SYSTEM  = 5,
    PULSAR_KVSTORE_REASON_AGENT_SESSION = 6,
    /* Truncated shared-preamble checkpoint (system prompt + tools, cut before
     * the first task message): the one snapshot every NEW conversation with
     * the same preamble can text-prefix restore from. Same concept as
     * AGENT_SYSTEM but written by the server's cold-anchor path. */
    PULSAR_KVSTORE_REASON_SYS_PREFIX    = 7,
} pulsar_kvstore_reason;

typedef enum {
    PULSAR_KVSTORE_LOG_DEFAULT,
    PULSAR_KVSTORE_LOG_KVCACHE,
    PULSAR_KVSTORE_LOG_WARNING,
} pulsar_kvstore_log_type;

typedef struct {
    /* The file name is the rendered byte prefix, not the token sequence. The
     * payload still carries the exact tokens and graph state; the hash only
     * answers "does this checkpoint represent the bytes at the front of the
     * incoming prompt?" */
    char sha[41];
    char *path;
    uint8_t quant_bits;
    /* Stored in header byte 7.  Flash is 0 for backward compatibility with
     * older cache files where this reserved byte was always written as zero. */
    uint8_t model_id;
    uint8_t reason;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint8_t ext_flags;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    uint64_t text_bytes;
    uint64_t file_size;
} pulsar_kvstore_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
} pulsar_kvstore_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    pulsar_kvstore_options opt;
    int continued_last_store_tokens;
    pulsar_kvstore_entry *entry;
    int len;
    int cap;
    const char *log_name;
    void *log_ud;
    void (*log)(void *ud, pulsar_kvstore_log_type type, const char *msg);
} pulsar_kvstore;

typedef struct {
    const char *text;
    size_t text_len;
    uint8_t model_id;
    uint8_t quant_bits;
    uint32_t ctx_size;
    bool reject_different_quant;
} pulsar_kvstore_eviction_context;

typedef struct {
    void *ud;
    uint8_t ext_flag;
    bool (*serialized_size)(void *ud, const char *text, uint64_t *bytes_out);
    bool (*write)(void *ud, FILE *fp, const char *text, uint64_t *written_bytes);
    int (*load)(void *ud, FILE *fp, const void *wanted);
    const void *load_wanted;
} pulsar_kvstore_trailer_hooks;

typedef struct {
    int tokens;
    uint32_t text_bytes;
    uint8_t quant_bits;
    uint8_t ext_flags;
    double load_ms;
    bool consumed;
    char *path;
} pulsar_kvstore_load_result;

pulsar_kvstore_options pulsar_kvstore_default_options(void);
uint8_t pulsar_kvstore_reason_code(const char *reason);
const char *pulsar_kvstore_key_kind(uint8_t ext_flags);

bool pulsar_kvstore_open(pulsar_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, pulsar_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, pulsar_kvstore_log_type type, const char *msg),
                      void *log_ud);
void pulsar_kvstore_close(pulsar_kvstore *kc);
void pulsar_kvstore_clear(pulsar_kvstore *kc);
void pulsar_kvstore_entry_free(pulsar_kvstore_entry *e);

char *pulsar_kvstore_render_tokens_text(pulsar_engine *engine,
                                     const pulsar_tokens *tokens,
                                     size_t *out_len);
bool pulsar_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
void pulsar_kvstore_tokens_copy_prefix(pulsar_tokens *dst, const pulsar_tokens *src, int n);
void pulsar_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        pulsar_engine *engine,
        const pulsar_tokens *exact_prefix,
        const char *suffix_text,
        pulsar_tokens *out);

int pulsar_kvstore_store_len(const pulsar_kvstore *kc, int tokens);
int pulsar_kvstore_chat_anchor_pos(const pulsar_kvstore *kc,
                                const pulsar_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id);
int pulsar_kvstore_continued_store_target(const pulsar_kvstore *kc, int live_tokens);
void pulsar_kvstore_note_store(pulsar_kvstore *kc, int tokens);
int pulsar_kvstore_suppress_continued_store(pulsar_kvstore *kc, int tokens);
void pulsar_kvstore_restore_suppressed_continued(pulsar_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens);

bool pulsar_kvstore_file_size_fits(const pulsar_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out);
double pulsar_kvstore_entry_eviction_score(const pulsar_kvstore_entry *e,
                                        const pulsar_tokens *live,
                                        uint64_t now,
                                        const pulsar_kvstore_eviction_context *incoming);
void pulsar_kvstore_evict(pulsar_kvstore *kc, const pulsar_tokens *live,
                       uint64_t extra_bytes,
                       const pulsar_kvstore_eviction_context *incoming);
int pulsar_kvstore_find_text_prefix(pulsar_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size);

bool pulsar_kvstore_store_live_prefix_text(pulsar_kvstore *kc,
                                        pulsar_engine *engine,
                                        pulsar_session *session,
                                        const pulsar_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const pulsar_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len);
bool pulsar_kvstore_store_live_prefix(pulsar_kvstore *kc,
                                   pulsar_engine *engine,
                                   pulsar_session *session,
                                   const pulsar_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const pulsar_kvstore_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len);
bool pulsar_kvstore_maybe_store_continued(pulsar_kvstore *kc,
                                       pulsar_engine *engine,
                                       pulsar_session *session,
                                       const pulsar_kvstore_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len);
int pulsar_kvstore_try_load_text(pulsar_kvstore *kc,
                              pulsar_engine *engine,
                              pulsar_session *session,
                              const char *prompt_text,
                              pulsar_tokens *effective_prompt,
                              pulsar_kvstore_load_result *result,
                              const pulsar_kvstore_trailer_hooks *hooks,
                              bool responses_protocol);
void pulsar_kvstore_load_result_free(pulsar_kvstore_load_result *result);

bool pulsar_kvstore_read_header(FILE *fp, pulsar_kvstore_entry *e,
                             uint32_t *text_bytes);
bool pulsar_kvstore_read_entry_file(const char *path, const char sha[41],
                                 pulsar_kvstore_entry *out);
void pulsar_kvstore_fill_header(uint8_t h[PULSAR_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes);
bool pulsar_kvstore_touch_file(const char *path, uint32_t hits);
bool pulsar_kvstore_sha_hex_name(const char *name, char sha[41]);
void pulsar_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
char *pulsar_kvstore_path_join(const char *dir, const char *name);
char *pulsar_kvstore_path_for_sha(pulsar_kvstore *kc, const char sha[41]);
void pulsar_kvstore_le_put32(uint8_t *p, uint32_t v);
uint32_t pulsar_kvstore_le_get32(const uint8_t *p);


#endif
