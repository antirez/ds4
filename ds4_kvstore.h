#ifndef DS4_KVSTORE_H
#define DS4_KVSTORE_H

#include "ds4.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define DS4_KVSTORE_FIXED_HEADER 48u
#define DS4_KVSTORE_DEFAULT_MB 4096
#define DS4_KVSTORE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)

#define DS4_KVSTORE_EXT_TOOL_MAP          (1u << 0)
#define DS4_KVSTORE_EXT_RESPONSES_VISIBLE (1u << 1)
#define DS4_KVSTORE_EXT_THINKING_VISIBLE  (1u << 2)
#define DS4_KVSTORE_EXT_SESSION_TITLE     (1u << 3)

typedef enum {
    DS4_KVSTORE_REASON_UNKNOWN   = 0,
    DS4_KVSTORE_REASON_COLD      = 1,
    DS4_KVSTORE_REASON_CONTINUED = 2,
    DS4_KVSTORE_REASON_EVICT     = 3,
    DS4_KVSTORE_REASON_SHUTDOWN  = 4,
    DS4_KVSTORE_REASON_AGENT_SYSTEM  = 5,
    DS4_KVSTORE_REASON_AGENT_SESSION = 6,
} ds4_kvstore_reason;

typedef enum {
    DS4_KVSTORE_LOG_DEFAULT,
    DS4_KVSTORE_LOG_KVCACHE,
    DS4_KVSTORE_LOG_WARNING,
} ds4_kvstore_log_type;

typedef struct {
    /* The file name is the rendered byte prefix, not the token sequence. The
     * payload still carries the exact tokens and graph state; the hash only
     * answers "does this checkpoint represent the bytes at the front of the
     * incoming prompt?" */
    char sha[41];
    char *path;
    uint8_t quant_bits;
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
} ds4_kvstore_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
} ds4_kvstore_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    ds4_kvstore_options opt;
    int continued_last_store_tokens;
    ds4_kvstore_entry *entry;
    int len;
    int cap;
    const char *log_name;
    void *log_ud;
    void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg);
} ds4_kvstore;

typedef struct {
    void *ud;
    uint8_t ext_flag;
    bool (*serialized_size)(void *ud, const char *text, uint64_t *bytes_out);
    bool (*write)(void *ud, FILE *fp, const char *text, uint64_t *written_bytes);
    int (*load)(void *ud, FILE *fp, const void *wanted);
    const void *load_wanted;
} ds4_kvstore_trailer_hooks;

typedef struct {
    int tokens;
    uint32_t text_bytes;
    uint8_t quant_bits;
    uint8_t ext_flags;
    double load_ms;
    bool consumed;
    char *path;
} ds4_kvstore_load_result;

ds4_kvstore_options ds4_kvstore_default_options(void);
uint8_t ds4_kvstore_reason_code(const char *reason);
const char *ds4_kvstore_key_kind(uint8_t ext_flags);

bool ds4_kvstore_open(ds4_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, ds4_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, ds4_kvstore_log_type type, const char *msg),
                      void *log_ud);
void ds4_kvstore_close(ds4_kvstore *kc);
void ds4_kvstore_clear(ds4_kvstore *kc);
void ds4_kvstore_entry_free(ds4_kvstore_entry *e);

char *ds4_kvstore_render_tokens_text(ds4_engine *engine,
                                     const ds4_tokens *tokens,
                                     size_t *out_len);
bool ds4_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
void ds4_kvstore_tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n);
void ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out);

int ds4_kvstore_store_len(const ds4_kvstore *kc, int tokens);
int ds4_kvstore_chat_anchor_pos(const ds4_kvstore *kc,
                                const ds4_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id);
int ds4_kvstore_continued_store_target(const ds4_kvstore *kc, int live_tokens);
void ds4_kvstore_note_store(ds4_kvstore *kc, int tokens);
int ds4_kvstore_suppress_continued_store(ds4_kvstore *kc, int tokens);
void ds4_kvstore_restore_suppressed_continued(ds4_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens);

bool ds4_kvstore_file_size_fits(const ds4_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out);
double ds4_kvstore_entry_eviction_score(const ds4_kvstore_entry *e,
                                        const ds4_tokens *live,
                                        const char *protected_sha,
                                        uint64_t now);
void ds4_kvstore_evict(ds4_kvstore *kc, const ds4_tokens *live,
                       const char *protected_sha);
int ds4_kvstore_find_text_prefix(ds4_kvstore *kc, const char *prompt_text,
                                 int quant_bits, int ctx_size);

bool ds4_kvstore_store_live_prefix_text(ds4_kvstore *kc,
                                        ds4_engine *engine,
                                        ds4_session *session,
                                        const ds4_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const ds4_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len);
bool ds4_kvstore_store_live_prefix(ds4_kvstore *kc,
                                   ds4_engine *engine,
                                   ds4_session *session,
                                   const ds4_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const ds4_kvstore_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len);
bool ds4_kvstore_maybe_store_continued(ds4_kvstore *kc,
                                       ds4_engine *engine,
                                       ds4_session *session,
                                       const ds4_kvstore_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len);
int ds4_kvstore_try_load_text(ds4_kvstore *kc,
                              ds4_engine *engine,
                              ds4_session *session,
                              const char *prompt_text,
                              ds4_tokens *effective_prompt,
                              ds4_kvstore_load_result *result,
                              const ds4_kvstore_trailer_hooks *hooks,
                              bool responses_protocol);
void ds4_kvstore_load_result_free(ds4_kvstore_load_result *result);

bool ds4_kvstore_read_header(FILE *fp, ds4_kvstore_entry *e,
                             uint32_t *text_bytes);
bool ds4_kvstore_read_entry_file(const char *path, const char sha[41],
                                 ds4_kvstore_entry *out);
void ds4_kvstore_fill_header(uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                             uint8_t quant_bits, uint8_t reason,
                             uint8_t ext_flags, uint32_t tokens,
                             uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes);
bool ds4_kvstore_touch_file(const char *path, uint32_t hits);
bool ds4_kvstore_sha_hex_name(const char *name, char sha[41]);
void ds4_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
char *ds4_kvstore_path_join(const char *dir, const char *name);
char *ds4_kvstore_path_for_sha(ds4_kvstore *kc, const char sha[41]);
void ds4_kvstore_le_put32(uint8_t *p, uint32_t v);
uint32_t ds4_kvstore_le_get32(const uint8_t *p);

#endif
