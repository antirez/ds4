#define _POSIX_C_SOURCE 200809L
#include "ds4_hooks.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define DS4_HOOK_DEFAULT_TIMEOUT_SECONDS 10
#define DS4_HOOK_MAX_TEXT_BYTES (1024u * 1024u)
#define DS4_HOOK_POLL_MILLISECONDS 20
#define DS4_HOOK_TERM_GRACE_MILLISECONDS 200

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} hook_buf;

static void hook_result_reset(ds4_hook_result *result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));
    result->exit_code = -1;
}

static void hook_result_error(ds4_hook_result *result, const char *fmt, ...) {
    if (!result) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(result->error, sizeof(result->error), fmt, ap);
    va_end(ap);
}

static bool hook_buf_reserve(hook_buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) return false;
    size_t needed = b->len + add + 1;
    if (needed <= b->cap) return true;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < needed) {
        if (cap > SIZE_MAX / 2) {
            cap = needed;
            break;
        }
        cap *= 2;
    }
    char *next = realloc(b->ptr, cap);
    if (!next) return false;
    b->ptr = next;
    b->cap = cap;
    return true;
}

static bool hook_buf_append(hook_buf *b, const void *data, size_t len) {
    if (!hook_buf_reserve(b, len)) return false;
    memcpy(b->ptr + b->len, data, len);
    b->len += len;
    b->ptr[b->len] = '\0';
    return true;
}

static bool hook_buf_puts(hook_buf *b, const char *text) {
    return hook_buf_append(b, text, strlen(text));
}

static bool hook_buf_printf(hook_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int needed = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (needed < 0 || !hook_buf_reserve(b, (size_t)needed)) {
        va_end(copy);
        return false;
    }
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, copy);
    va_end(copy);
    b->len += (size_t)needed;
    return true;
}

static bool hook_json_string_n(hook_buf *b, const char *text, size_t len) {
    if (!hook_buf_puts(b, "\"")) return false;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        switch (c) {
        case '"':
            if (!hook_buf_puts(b, "\\\"")) return false;
            break;
        case '\\':
            if (!hook_buf_puts(b, "\\\\")) return false;
            break;
        case '\b':
            if (!hook_buf_puts(b, "\\b")) return false;
            break;
        case '\f':
            if (!hook_buf_puts(b, "\\f")) return false;
            break;
        case '\n':
            if (!hook_buf_puts(b, "\\n")) return false;
            break;
        case '\r':
            if (!hook_buf_puts(b, "\\r")) return false;
            break;
        case '\t':
            if (!hook_buf_puts(b, "\\t")) return false;
            break;
        default:
            if (c < 0x20) {
                if (!hook_buf_printf(b, "\\u%04x", (unsigned)c)) return false;
            } else if (!hook_buf_append(b, &c, 1)) {
                return false;
            }
            break;
        }
    }
    return hook_buf_puts(b, "\"");
}

static bool hook_json_string(hook_buf *b, const char *text) {
    if (!text) return hook_buf_puts(b, "null");
    size_t len = strlen(text);
    if (len > DS4_HOOK_MAX_TEXT_BYTES) len = DS4_HOOK_MAX_TEXT_BYTES;
    return hook_json_string_n(b, text, len);
}

const char *ds4_hook_event_name(ds4_hook_event event) {
    switch (event) {
    case DS4_HOOK_BEFORE_RESPONSE:
        return "before_response";
    case DS4_HOOK_AFTER_RESPONSE:
        return "after_response";
    }
    return "unknown";
}

bool ds4_hook_command_enabled(const ds4_hook_config *config,
                              ds4_hook_event event) {
    if (!config) return false;
    const char *command = event == DS4_HOOK_BEFORE_RESPONSE ?
        config->before_response_command : config->after_response_command;
    return command && command[0] != '\0';
}

char *ds4_hook_payload_json(const ds4_hook_payload *payload) {
    if (!payload) return NULL;
    hook_buf b = {0};
    const char *user = payload->user_text;
    const char *response = payload->response_text;
    bool user_truncated = user && strlen(user) > DS4_HOOK_MAX_TEXT_BYTES;
    bool response_truncated = response &&
        strlen(response) > DS4_HOOK_MAX_TEXT_BYTES;

    bool ok = hook_buf_puts(&b, "{") &&
        hook_buf_puts(&b, "\"event\":") &&
        hook_json_string(&b, ds4_hook_event_name(payload->event)) &&
        hook_buf_printf(&b, ",\"pid\":%ld", (long)getpid()) &&
        hook_buf_puts(&b, ",\"model\":") &&
        hook_json_string(&b, payload->model) &&
        hook_buf_puts(&b, ",\"user_text\":") &&
        hook_json_string(&b, user) &&
        hook_buf_puts(&b, ",\"response_text\":") &&
        hook_json_string(&b, response) &&
        hook_buf_printf(&b, ",\"response_index\":%d", payload->response_index) &&
        hook_buf_printf(&b, ",\"generated_tokens\":%d", payload->generated_tokens) &&
        hook_buf_printf(&b, ",\"interrupted\":%s", payload->interrupted ? "true" : "false") &&
        hook_buf_printf(&b, ",\"tool_call\":%s", payload->tool_call ? "true" : "false") &&
        hook_buf_printf(&b, ",\"user_text_truncated\":%s", user_truncated ? "true" : "false") &&
        hook_buf_printf(&b, ",\"response_text_truncated\":%s", response_truncated ? "true" : "false") &&
        hook_buf_puts(&b, "}\n");

    if (!ok) {
        free(b.ptr);
        return NULL;
    }
    return b.ptr;
}

static double hook_monotonic_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void hook_sleep_milliseconds(int milliseconds) {
    struct timespec req = {
        .tv_sec = milliseconds / 1000,
        .tv_nsec = (long)(milliseconds % 1000) * 1000000L,
    };
    while (nanosleep(&req, &req) != 0 && errno == EINTR) {}
}

static int hook_set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static int hook_wait_child(pid_t pid, int *status, double deadline) {
    for (;;) {
        pid_t waited = waitpid(pid, status, WNOHANG);
        if (waited == pid) return 0;
        if (waited < 0 && errno != EINTR) return -1;
        if (hook_monotonic_seconds() >= deadline) return 1;
        hook_sleep_milliseconds(DS4_HOOK_POLL_MILLISECONDS);
    }
}

static void hook_terminate_child(pid_t pid) {
    (void)kill(-pid, SIGTERM);
    double deadline = hook_monotonic_seconds() +
        (double)DS4_HOOK_TERM_GRACE_MILLISECONDS / 1000.0;
    int status = 0;
    if (hook_wait_child(pid, &status, deadline) == 0) return;
    (void)kill(-pid, SIGKILL);
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
}

static int hook_write_payload(int fd, const char *json, size_t len,
                              double deadline) {
    size_t written = 0;
    while (written < len) {
        ssize_t n = write(fd, json + written, len - written);
        if (n > 0) {
            written += (size_t)n;
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            double remaining = deadline - hook_monotonic_seconds();
            if (remaining <= 0.0) return 1;
            int timeout_ms = (int)(remaining * 1000.0);
            if (timeout_ms > DS4_HOOK_POLL_MILLISECONDS)
                timeout_ms = DS4_HOOK_POLL_MILLISECONDS;
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            int rc = poll(&pfd, 1, timeout_ms);
            if (rc < 0 && errno == EINTR) continue;
            if (rc < 0) return -1;
            continue;
        }
        if (n < 0 && errno == EPIPE) return -2;
        return -1;
    }
    return 0;
}

int ds4_hook_run(const ds4_hook_config *config,
                 const ds4_hook_payload *payload,
                 ds4_hook_result *result) {
    hook_result_reset(result);
    if (!payload || !ds4_hook_command_enabled(config, payload->event)) return 0;

    const char *command = payload->event == DS4_HOOK_BEFORE_RESPONSE ?
        config->before_response_command : config->after_response_command;
    int timeout = config->timeout_seconds > 0 ?
        config->timeout_seconds : DS4_HOOK_DEFAULT_TIMEOUT_SECONDS;

    char *json = ds4_hook_payload_json(payload);
    if (!json) {
        hook_result_error(result, "failed to serialize hook payload");
        return -1;
    }

    int input_pipe[2];
    if (pipe(input_pipe) != 0) {
        hook_result_error(result, "pipe failed: %s", strerror(errno));
        free(json);
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        hook_result_error(result, "fork failed: %s", strerror(errno));
        close(input_pipe[0]);
        close(input_pipe[1]);
        free(json);
        return -1;
    }

    if (pid == 0) {
        (void)setpgid(0, 0);
        close(input_pipe[1]);
        if (dup2(input_pipe[0], STDIN_FILENO) < 0) _exit(126);
        close(input_pipe[0]);
        execl("/bin/sh", "sh", "-c", command, (char *)NULL);
        _exit(127);
    }

    if (result) result->attempted = true;
    close(input_pipe[0]);
    (void)setpgid(pid, pid);
    if (hook_set_nonblocking(input_pipe[1]) != 0) {
        hook_result_error(result, "failed to configure hook input: %s", strerror(errno));
        close(input_pipe[1]);
        hook_terminate_child(pid);
        free(json);
        return -1;
    }

    double deadline = hook_monotonic_seconds() + (double)timeout;
    sigset_t pipe_set;
    sigset_t old_set;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    int mask_rc = pthread_sigmask(SIG_BLOCK, &pipe_set, &old_set);
    int write_rc = hook_write_payload(input_pipe[1], json, strlen(json), deadline);
    close(input_pipe[1]);
    if (mask_rc == 0) {
        struct timespec zero = {0};
        siginfo_t ignored;
        while (sigtimedwait(&pipe_set, &ignored, &zero) >= 0) {}
        (void)pthread_sigmask(SIG_SETMASK, &old_set, NULL);
    }
    free(json);

    if (write_rc == 1) {
        if (result) result->timed_out = true;
        hook_result_error(result, "timed out while delivering hook payload");
        hook_terminate_child(pid);
        return -1;
    }
    if (write_rc < 0 && write_rc != -2) {
        hook_result_error(result, "failed to write hook payload: %s", strerror(errno));
        hook_terminate_child(pid);
        return -1;
    }

    int status = 0;
    int wait_rc = hook_wait_child(pid, &status, deadline);
    if (wait_rc == 1) {
        if (result) result->timed_out = true;
        hook_result_error(result, "hook timed out after %d second%s",
                          timeout, timeout == 1 ? "" : "s");
        hook_terminate_child(pid);
        return -1;
    }
    if (wait_rc < 0) {
        hook_result_error(result, "waitpid failed: %s", strerror(errno));
        return -1;
    }

    if (WIFEXITED(status)) {
        int exit_code = WEXITSTATUS(status);
        if (result) result->exit_code = exit_code;
        if (exit_code == 0) return 0;
        hook_result_error(result, "hook exited with status %d", exit_code);
        return -1;
    }
    if (WIFSIGNALED(status)) {
        int signal_number = WTERMSIG(status);
        if (result) result->term_signal = signal_number;
        hook_result_error(result, "hook terminated by signal %d", signal_number);
        return -1;
    }

    hook_result_error(result, "hook ended with an unknown process status");
    return -1;
}
