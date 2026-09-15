#define _POSIX_C_SOURCE 200809L
#include "ds4.h"
#include "ds4_hooks.h"

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_agent_core_main(int argc, char **argv);

typedef struct {
    pthread_mutex_t mutex;
    ds4_hook_config config;
    ds4_tokens *transcript;
    ds4_engine *engine;
    char *user_text;
    char *response_text;
    size_t response_len;
    size_t response_cap;
    int response_index;
    int generated_tokens;
    bool response_open;
} ds4_agent_hook_state;

typedef struct {
    ds4_hook_payload payload;
    char *model;
    char *user_text;
    char *response_text;
} ds4_agent_hook_snapshot;

static ds4_agent_hook_state g_agent_hooks = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};

static char *hook_strdup(const char *text) {
    if (!text) return NULL;
    size_t len = strlen(text);
    char *copy = malloc(len + 1);
    if (!copy) return NULL;
    memcpy(copy, text, len + 1);
    return copy;
}

static bool hook_text_reserve(size_t add) {
    if (add > SIZE_MAX - g_agent_hooks.response_len - 1) return false;
    size_t needed = g_agent_hooks.response_len + add + 1;
    if (needed <= g_agent_hooks.response_cap) return true;
    size_t cap = g_agent_hooks.response_cap ? g_agent_hooks.response_cap : 256;
    while (cap < needed) cap *= 2;
    char *next = realloc(g_agent_hooks.response_text, cap);
    if (!next) return false;
    g_agent_hooks.response_text = next;
    g_agent_hooks.response_cap = cap;
    return true;
}

static void hook_text_reset_locked(void) {
    g_agent_hooks.response_len = 0;
    if (g_agent_hooks.response_text) g_agent_hooks.response_text[0] = '\0';
}

static void hook_text_append_locked(const char *text, size_t len) {
    if (!text || !len || !hook_text_reserve(len)) return;
    memcpy(g_agent_hooks.response_text + g_agent_hooks.response_len, text, len);
    g_agent_hooks.response_len += len;
    g_agent_hooks.response_text[g_agent_hooks.response_len] = '\0';
}

static void hook_snapshot_free(ds4_agent_hook_snapshot *snapshot) {
    if (!snapshot) return;
    free(snapshot->model);
    free(snapshot->user_text);
    free(snapshot->response_text);
    memset(snapshot, 0, sizeof(*snapshot));
}

static ds4_agent_hook_snapshot hook_snapshot_locked(ds4_hook_event event,
                                                     bool interrupted,
                                                     bool tool_call) {
    ds4_agent_hook_snapshot snapshot = {0};
    snapshot.model = hook_strdup(g_agent_hooks.engine ?
        ds4_engine_model_name(g_agent_hooks.engine) : NULL);
    snapshot.user_text = hook_strdup(g_agent_hooks.user_text);
    snapshot.response_text = event == DS4_HOOK_AFTER_RESPONSE ?
        hook_strdup(g_agent_hooks.response_text ? g_agent_hooks.response_text : "") : NULL;
    snapshot.payload.event = event;
    snapshot.payload.model = snapshot.model;
    snapshot.payload.user_text = snapshot.user_text;
    snapshot.payload.response_text = snapshot.response_text;
    snapshot.payload.response_index = g_agent_hooks.response_index;
    snapshot.payload.generated_tokens = g_agent_hooks.generated_tokens;
    snapshot.payload.interrupted = interrupted;
    snapshot.payload.tool_call = tool_call;
    return snapshot;
}

static void hook_run_snapshot(ds4_agent_hook_snapshot *snapshot) {
    ds4_hook_result result;
    int rc = ds4_hook_run(&g_agent_hooks.config, &snapshot->payload, &result);
    if (rc != 0 && result.attempted)
        fprintf(stderr, "ds4-agent: %s hook: %s\n",
                ds4_hook_event_name(snapshot->payload.event),
                result.error[0] ? result.error : "hook failed");
    hook_snapshot_free(snapshot);
}

static bool hook_response_looks_like_tool_call(const char *text) {
    return text && (strstr(text, "｜DSML｜") ||
                    strstr(text, "<tool_call") ||
                    strstr(text, "<function="));
}

static void hook_finish_response(bool interrupted, bool tool_call) {
    ds4_agent_hook_snapshot snapshot = {0};
    bool run = false;
    pthread_mutex_lock(&g_agent_hooks.mutex);
    if (g_agent_hooks.response_open) {
        snapshot = hook_snapshot_locked(DS4_HOOK_AFTER_RESPONSE, interrupted,
            tool_call || hook_response_looks_like_tool_call(g_agent_hooks.response_text));
        g_agent_hooks.response_open = false;
        run = ds4_hook_command_enabled(&g_agent_hooks.config, DS4_HOOK_AFTER_RESPONSE);
    }
    pthread_mutex_unlock(&g_agent_hooks.mutex);
    if (run) hook_run_snapshot(&snapshot);
    else hook_snapshot_free(&snapshot);
}

static void hook_flush_at_exit(void) {
    hook_finish_response(false, false);
    pthread_mutex_lock(&g_agent_hooks.mutex);
    free(g_agent_hooks.user_text);
    free(g_agent_hooks.response_text);
    g_agent_hooks.user_text = NULL;
    g_agent_hooks.response_text = NULL;
    pthread_mutex_unlock(&g_agent_hooks.mutex);
}

static void hook_print_help(FILE *fp) {
    fprintf(fp,
        "\nAgent hooks:\n"
        "  --before-response-hook CMD  Run CMD before each assistant response.\n"
        "  --after-response-hook CMD   Run CMD after each assistant response.\n"
        "  --hook-timeout SEC          Kill a hook after SEC seconds (default: 10).\n\n"
        "Hooks receive one JSON object on stdin. They are disabled by default.\n"
        "Hook failures are reported to stderr and do not abort the agent turn.\n");
}

static bool hook_parse_positive_int(const char *text, int *out) {
    if (!text || !text[0]) return false;
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno || !end || *end || value < 1 || value > 3600) return false;
    *out = (int)value;
    return true;
}

static const char *hook_option_value(const char *arg, const char *name) {
    size_t len = strlen(name);
    return strncmp(arg, name, len) == 0 && arg[len] == '=' ? arg + len + 1 : NULL;
}

static int hook_filter_options(int argc, char **argv, char ***filtered_out) {
    char **filtered = calloc((size_t)argc + 1, sizeof(*filtered));
    if (!filtered) return -1;
    int outc = 1;
    filtered[0] = argv[0];
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        const char *value = NULL;
        if (!strcmp(arg, "--help") && i + 1 < argc && !strcmp(argv[i + 1], "hooks")) {
            hook_print_help(stdout);
            free(filtered);
            return 0;
        }
        if (!strcmp(arg, "--before-response-hook")) {
            if (++i >= argc) goto missing;
            g_agent_hooks.config.before_response_command = argv[i];
            continue;
        }
        if ((value = hook_option_value(arg, "--before-response-hook"))) {
            g_agent_hooks.config.before_response_command = value;
            continue;
        }
        if (!strcmp(arg, "--after-response-hook")) {
            if (++i >= argc) goto missing;
            g_agent_hooks.config.after_response_command = argv[i];
            continue;
        }
        if ((value = hook_option_value(arg, "--after-response-hook"))) {
            g_agent_hooks.config.after_response_command = value;
            continue;
        }
        if (!strcmp(arg, "--hook-timeout")) {
            if (++i >= argc || !hook_parse_positive_int(argv[i], &g_agent_hooks.config.timeout_seconds))
                goto timeout_error;
            continue;
        }
        if ((value = hook_option_value(arg, "--hook-timeout"))) {
            if (!hook_parse_positive_int(value, &g_agent_hooks.config.timeout_seconds)) goto timeout_error;
            continue;
        }
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) hook_print_help(stdout);
        filtered[outc++] = argv[i];
    }
    filtered[outc] = NULL;
    *filtered_out = filtered;
    return outc;
missing:
    fprintf(stderr, "ds4-agent: hook option requires a command\n");
    free(filtered);
    return -1;
timeout_error:
    fprintf(stderr, "ds4-agent: --hook-timeout must be an integer from 1 to 3600\n");
    free(filtered);
    return -1;
}

int main(int argc, char **argv) {
    g_agent_hooks.config.timeout_seconds = 10;
    char **filtered = NULL;
    int filtered_argc = hook_filter_options(argc, argv, &filtered);
    if (filtered_argc == 0) return 0;
    if (filtered_argc < 0) return 2;
    (void)atexit(hook_flush_at_exit);
    int rc = ds4_agent_core_main(filtered_argc, filtered);
    free(filtered);
    return rc;
}

void ds4_hooks_chat_append_message(ds4_engine *engine, ds4_tokens *tokens,
                                   const char *role, const char *content) {
    if (role && !strcmp(role, "tool")) hook_finish_response(false, true);
    if (role && !strcmp(role, "user")) {
        pthread_mutex_lock(&g_agent_hooks.mutex);
        bool finish = g_agent_hooks.transcript == tokens && g_agent_hooks.response_open;
        pthread_mutex_unlock(&g_agent_hooks.mutex);
        if (finish) hook_finish_response(false, false);
    }
    ds4_chat_append_message(engine, tokens, role, content);
    if (role && !strcmp(role, "user")) {
        pthread_mutex_lock(&g_agent_hooks.mutex);
        if (!g_agent_hooks.transcript || g_agent_hooks.transcript == tokens) {
            g_agent_hooks.transcript = tokens;
            g_agent_hooks.engine = engine;
            free(g_agent_hooks.user_text);
            g_agent_hooks.user_text = hook_strdup(content ? content : "");
        }
        pthread_mutex_unlock(&g_agent_hooks.mutex);
    }
}

void ds4_hooks_chat_append_assistant_prefix(ds4_engine *engine, ds4_tokens *tokens,
                                            ds4_think_mode think_mode) {
    ds4_chat_append_assistant_prefix(engine, tokens, think_mode);
    ds4_agent_hook_snapshot snapshot = {0};
    bool run = false;
    pthread_mutex_lock(&g_agent_hooks.mutex);
    if (g_agent_hooks.transcript == tokens) {
        g_agent_hooks.engine = engine;
        g_agent_hooks.response_index++;
        g_agent_hooks.generated_tokens = 0;
        g_agent_hooks.response_open = true;
        hook_text_reset_locked();
        snapshot = hook_snapshot_locked(DS4_HOOK_BEFORE_RESPONSE, false, false);
        run = ds4_hook_command_enabled(&g_agent_hooks.config, DS4_HOOK_BEFORE_RESPONSE);
    }
    pthread_mutex_unlock(&g_agent_hooks.mutex);
    if (run) hook_run_snapshot(&snapshot);
    else hook_snapshot_free(&snapshot);
}

void ds4_hooks_tokens_push(ds4_tokens *tokens, int token) {
    ds4_tokens_push(tokens, token);
    pthread_mutex_lock(&g_agent_hooks.mutex);
    bool active = g_agent_hooks.response_open && g_agent_hooks.transcript == tokens;
    ds4_engine *engine = g_agent_hooks.engine;
    pthread_mutex_unlock(&g_agent_hooks.mutex);
    if (!active || !engine) return;
    if (token == ds4_token_eos(engine)) {
        hook_finish_response(false, false);
        return;
    }
    size_t text_len = 0;
    char *text = ds4_token_text(engine, token, &text_len);
    pthread_mutex_lock(&g_agent_hooks.mutex);
    if (g_agent_hooks.response_open && g_agent_hooks.transcript == tokens) {
        hook_text_append_locked(text, text_len);
        g_agent_hooks.generated_tokens++;
    }
    pthread_mutex_unlock(&g_agent_hooks.mutex);
    free(text);
}

bool ds4_hooks_token_is_stop_for_think_mode(ds4_engine *engine, int token,
                                             ds4_think_mode mode) {
    bool stop = ds4_token_is_stop_for_think_mode(engine, token, mode);
    if (stop) hook_finish_response(false, false);
    return stop;
}
