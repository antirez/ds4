#ifndef DS4_AGENT_CONTEXT_H
#define DS4_AGENT_CONTEXT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct ds4_agent_context_meta {
    char id[41];
    char *label;
    char *kv_file;
    char *memory_file;
    uint64_t created_at;
    uint64_t world_epoch;
    int transcript_tokens;
} ds4_agent_context_meta;

typedef struct ds4_agent_side_effect {
    uint64_t epoch;
    char *kind;
    char *detail;
    struct ds4_agent_side_effect *next;
} ds4_agent_side_effect;

typedef struct ds4_agent_side_effects {
    ds4_agent_side_effect *head;
    int count;
    uint64_t evicted_count;
    uint64_t latest_evicted_epoch;
} ds4_agent_side_effects;

void ds4_agent_context_meta_free(ds4_agent_context_meta *m);
bool ds4_agent_context_id_valid(const char *id);
bool ds4_agent_context_file_component_safe(const char *s);
char *ds4_agent_context_file_name(const char id[41], const char *suffix);
char *ds4_agent_context_path_for_file(const char *context_dir, const char *file);
char *ds4_agent_context_limited_strdup(const char *s, size_t max);
char *ds4_agent_context_oneline(const char *s, size_t max);

bool ds4_agent_context_write_meta(const ds4_agent_context_meta *m,
                                  const char *meta_path,
                                  char *err, size_t err_len);
bool ds4_agent_context_read_meta_file(const char *path,
                                      ds4_agent_context_meta *m,
                                      char *err, size_t err_len);
bool ds4_agent_context_meta_filename(const char *name);
int ds4_agent_context_count_checkpoints(const char *context_dir);
uint64_t ds4_agent_context_max_world_epoch(const char *context_dir);
char *ds4_agent_context_full_kv_path(const char *context_dir,
                                     const ds4_agent_context_meta *m);
char *ds4_agent_context_full_memory_path(const char *context_dir,
                                         const ds4_agent_context_meta *m);
bool ds4_agent_context_find_checkpoint(const char *context_dir,
                                       const char *prefix,
                                       ds4_agent_context_meta *found,
                                       char **meta_path_out,
                                       char **kv_path_out,
                                       char *err, size_t err_len);

void ds4_agent_side_effects_free(ds4_agent_side_effects *effects);
uint64_t ds4_agent_side_effects_note(ds4_agent_side_effects *effects,
                                     uint64_t current_epoch,
                                     const char *kind,
                                     const char *detail);
char *ds4_agent_side_effects_summary_since(const ds4_agent_side_effects *effects,
                                           uint64_t epoch);
bool ds4_agent_context_no_running_bash_guard(const char *action,
                                             int running_bash_jobs,
                                             char *err,
                                             size_t err_len);
bool ds4_agent_context_restore_epoch_guard(uint64_t current_epoch,
                                           uint64_t checkpoint_epoch,
                                           bool allow_side_effect_mismatch,
                                           char *err,
                                           size_t err_len);

#endif
