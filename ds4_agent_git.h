#ifndef DS4_AGENT_GIT_H
#define DS4_AGENT_GIT_H

#include <stdbool.h>
#include <stddef.h>

typedef struct ds4_agent_git_result {
    char *output;
    int exit_code;
    bool truncated;
} ds4_agent_git_result;

typedef struct ds4_agent_git_options {
    const char *repo;
    const char *action;
    const char *path;
    const char *ref;
    const char *base_ref;
    const char *target_ref;
    const char *range;
    const char *message;
    const char *remote;
    int limit;
    int start_line;
    int line_count;
    bool staged;
    bool stat;
    bool name_status;
    bool name_only;
    bool patch;
    bool follow;
    bool dry_run;
    bool all;
    bool confirm;
    size_t max_bytes;
} ds4_agent_git_options;

void ds4_agent_git_result_free(ds4_agent_git_result *r);

bool ds4_agent_git_run_options(const ds4_agent_git_options *opts,
                               ds4_agent_git_result *result,
                               char *err,
                               size_t err_len);

bool ds4_agent_git_run(const char *repo,
                       const char *action,
                       const char *path,
                       const char *ref,
                       int limit,
                       bool staged,
                       size_t max_bytes,
                       ds4_agent_git_result *result,
                       char *err,
                       size_t err_len);

#endif
