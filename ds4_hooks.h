#ifndef DS4_HOOKS_H
#define DS4_HOOKS_H

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    DS4_HOOK_BEFORE_RESPONSE = 0,
    DS4_HOOK_AFTER_RESPONSE = 1,
} ds4_hook_event;

typedef struct {
    const char *before_response_command;
    const char *after_response_command;
    int timeout_seconds;
} ds4_hook_config;

typedef struct {
    ds4_hook_event event;
    const char *model;
    const char *user_text;
    const char *response_text;
    int response_index;
    int generated_tokens;
    bool interrupted;
    bool tool_call;
} ds4_hook_payload;

typedef struct {
    bool attempted;
    bool timed_out;
    int exit_code;
    int term_signal;
    char error[160];
} ds4_hook_result;

const char *ds4_hook_event_name(ds4_hook_event event);
bool ds4_hook_command_enabled(const ds4_hook_config *config,
                              ds4_hook_event event);
char *ds4_hook_payload_json(const ds4_hook_payload *payload);
int ds4_hook_run(const ds4_hook_config *config,
                 const ds4_hook_payload *payload,
                 ds4_hook_result *result);

#endif
