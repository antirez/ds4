#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ds4_agent_git.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define DS4_AGENT_GIT_DEFAULT_MAX_BYTES (64 * 1024)

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} ds4_agent_git_buf;

static void git_set_err(char *err, size_t err_len, const char *fmt, ...) {
    if (!err || err_len == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

static void *git_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "ds4-agent-git: out of memory\n");
        abort();
    }
    return p;
}

static void *git_xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) {
        fprintf(stderr, "ds4-agent-git: out of memory\n");
        abort();
    }
    return p;
}

static char *git_xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *out = git_xmalloc(n + 1);
    memcpy(out, s, n + 1);
    return out;
}

static void git_buf_append(ds4_agent_git_buf *b, const char *s, size_t n) {
    if (n == 0) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = git_xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void git_buf_append_capped(ds4_agent_git_buf *b, const char *s, size_t n,
                                  size_t max_bytes, bool *truncated) {
    if (!s || n == 0) return;
    if (!max_bytes) max_bytes = DS4_AGENT_GIT_DEFAULT_MAX_BYTES;
    if (b->len >= max_bytes) {
        if (truncated) *truncated = true;
        return;
    }
    size_t keep = max_bytes - b->len;
    if (keep > n) keep = n;
    git_buf_append(b, s, keep);
    if (keep < n && truncated) *truncated = true;
}

static void git_buf_puts_capped(ds4_agent_git_buf *b, const char *s,
                                size_t max_bytes, bool *truncated) {
    git_buf_append_capped(b, s ? s : "", s ? strlen(s) : 0,
                          max_bytes, truncated);
}

static char *git_buf_take(ds4_agent_git_buf *b) {
    if (!b->ptr) return git_xstrdup("");
    char *out = b->ptr;
    b->ptr = NULL;
    b->len = 0;
    b->cap = 0;
    return out;
}

static bool git_ref_safe(const char *ref) {
    if (!ref || !ref[0] || ref[0] == '-') return false;
    for (const unsigned char *p = (const unsigned char *)ref; *p; p++) {
        if (iscntrl(*p) || isspace(*p)) return false;
    }
    return true;
}

static void argv_add(const char **argv, int *argc, const char *s) {
    argv[(*argc)++] = s;
}

static int git_argv_base(const char **argv, const char *repo) {
    int argc = 0;
    if (!repo || !repo[0]) repo = ".";
    argv_add(argv, &argc, "git");
    argv_add(argv, &argc, "--no-pager");
    argv_add(argv, &argc, "-c");
    argv_add(argv, &argc, "color.ui=false");
    argv_add(argv, &argc, "-C");
    argv_add(argv, &argc, repo);
    return argc;
}

static bool run_argv(const char **argv, size_t max_bytes,
                     ds4_agent_git_result *result,
                     char *err, size_t err_len) {
    if (!max_bytes) max_bytes = DS4_AGENT_GIT_DEFAULT_MAX_BYTES;
    memset(result, 0, sizeof(*result));
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        git_set_err(err, err_len, "pipe: %s", strerror(errno));
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        git_set_err(err, err_len, "fork: %s", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        return false;
    }

    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execvp("git", (char * const *)argv);
        dprintf(STDERR_FILENO, "exec git: %s\n", strerror(errno));
        _exit(127);
    }

    close(pipefd[1]);
    ds4_agent_git_buf out = {0};
    char tmp[4096];
    while (true) {
        ssize_t n = read(pipefd[0], tmp, sizeof(tmp));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) {
            git_set_err(err, err_len, "read git output: %s", strerror(errno));
            close(pipefd[0]);
            int ignored = 0;
            waitpid(pid, &ignored, 0);
            free(out.ptr);
            return false;
        }
        if (n == 0) break;
        size_t got = (size_t)n;
        if (out.len < max_bytes) {
            size_t keep = max_bytes - out.len;
            if (keep > got) keep = got;
            git_buf_append(&out, tmp, keep);
        }
        if (out.len >= max_bytes) result->truncated = true;
    }
    close(pipefd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        git_set_err(err, err_len, "wait git: %s", strerror(errno));
        free(out.ptr);
        return false;
    }
    if (WIFEXITED(status)) result->exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) result->exit_code = 128 + WTERMSIG(status);
    else result->exit_code = 1;

    result->output = git_buf_take(&out);
    return true;
}

static bool run_git_tail(const char *repo, const char * const *tail,
                         size_t max_bytes, ds4_agent_git_result *result,
                         char *err, size_t err_len) {
    const char *argv[64];
    int argc = git_argv_base(argv, repo);
    for (int i = 0; tail && tail[i]; i++) argv_add(argv, &argc, tail[i]);
    argv[argc] = NULL;
    return run_argv(argv, max_bytes, result, err, err_len);
}

static char *git_first_line_value(const char *s) {
    if (!s) s = "";
    while (*s == '\n' || *s == '\r') s++;
    const char *end = s;
    while (*end && *end != '\n' && *end != '\r') end++;
    while (end > s && isspace((unsigned char)end[-1])) end--;
    size_t n = (size_t)(end - s);
    char *out = git_xmalloc(n + 1);
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static bool git_status_dirty(const char *status) {
    const char *p = status ? status : "";
    while (*p) {
        const char *line = p;
        while (*p && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len > 0 && !(len >= 3 && line[0] == '#' && line[1] == '#' && line[2] == ' '))
            return true;
        if (*p == '\n') p++;
    }
    return false;
}

static bool git_info_append_line(const char *repo, const char *key,
                                 const char * const *tail, bool required,
                                 ds4_agent_git_buf *out, size_t max_bytes,
                                 bool *truncated, int *exit_code,
                                 char **line_out, char *err, size_t err_len) {
    ds4_agent_git_result tmp = {0};
    if (!run_git_tail(repo, tail, max_bytes, &tmp, err, err_len)) return false;
    char *line = git_first_line_value(tmp.output);
    git_buf_puts_capped(out, key, max_bytes, truncated);
    git_buf_puts_capped(out, "=", max_bytes, truncated);
    if (tmp.exit_code == 0 && line[0]) {
        git_buf_puts_capped(out, line, max_bytes, truncated);
        if (line_out) *line_out = git_xstrdup(line);
    } else {
        git_buf_puts_capped(out, "(none)", max_bytes, truncated);
        if (required && exit_code && *exit_code == 0) *exit_code = tmp.exit_code;
    }
    git_buf_puts_capped(out, "\n", max_bytes, truncated);
    if (tmp.exit_code != 0 && required && line[0]) {
        git_buf_puts_capped(out, key, max_bytes, truncated);
        git_buf_puts_capped(out, "_error=", max_bytes, truncated);
        git_buf_puts_capped(out, line, max_bytes, truncated);
        git_buf_puts_capped(out, "\n", max_bytes, truncated);
    }
    if (tmp.truncated && truncated) *truncated = true;
    free(line);
    ds4_agent_git_result_free(&tmp);
    return true;
}

static bool git_info_append_section(const char *repo, const char *header,
                                    const char * const *tail, bool required,
                                    ds4_agent_git_buf *out, size_t max_bytes,
                                    bool *truncated, int *exit_code,
                                    ds4_agent_git_result *result_out,
                                    char *err, size_t err_len) {
    ds4_agent_git_result tmp = {0};
    if (!run_git_tail(repo, tail, max_bytes, &tmp, err, err_len)) return false;
    if (tmp.exit_code != 0 && required && exit_code && *exit_code == 0)
        *exit_code = tmp.exit_code;
    git_buf_puts_capped(out, header, max_bytes, truncated);
    git_buf_puts_capped(out, "\n", max_bytes, truncated);
    if (tmp.output && tmp.output[0])
        git_buf_puts_capped(out, tmp.output, max_bytes, truncated);
    else
        git_buf_puts_capped(out, "(no output)\n", max_bytes, truncated);
    if (out->len > 0 && out->ptr[out->len - 1] != '\n')
        git_buf_puts_capped(out, "\n", max_bytes, truncated);
    if (tmp.truncated && truncated) *truncated = true;
    if (result_out) {
        *result_out = tmp;
        memset(&tmp, 0, sizeof(tmp));
    }
    ds4_agent_git_result_free(&tmp);
    return true;
}

static bool git_run_info(const char *repo, size_t max_bytes,
                         ds4_agent_git_result *result,
                         char *err, size_t err_len) {
    if (!max_bytes) max_bytes = DS4_AGENT_GIT_DEFAULT_MAX_BYTES;
    memset(result, 0, sizeof(*result));
    ds4_agent_git_buf out = {0};
    bool truncated = false;
    int exit_code = 0;
    char *upstream = NULL;

    const char *root_args[] = {"rev-parse", "--show-toplevel", NULL};
    const char *branch_args[] = {"rev-parse", "--abbrev-ref", "HEAD", NULL};
    const char *head_args[] = {"rev-parse", "HEAD", NULL};
    const char *upstream_args[] = {
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", NULL
    };
    if (!git_info_append_line(repo, "repo_root", root_args, true, &out,
                              max_bytes, &truncated, &exit_code, NULL,
                              err, err_len))
        goto fail;
    if (!git_info_append_line(repo, "branch", branch_args, true, &out,
                              max_bytes, &truncated, &exit_code, NULL,
                              err, err_len))
        goto fail;
    if (!git_info_append_line(repo, "head", head_args, true, &out,
                              max_bytes, &truncated, &exit_code, NULL,
                              err, err_len))
        goto fail;
    if (!git_info_append_line(repo, "upstream", upstream_args, false, &out,
                              max_bytes, &truncated, &exit_code, &upstream,
                              err, err_len))
        goto fail;

    if (upstream && upstream[0]) {
        const char *ahead_args[] = {
            "rev-list", "--left-right", "--count", "@{u}...HEAD", NULL
        };
        if (!git_info_append_line(repo, "behind_ahead", ahead_args, false, &out,
                                  max_bytes, &truncated, &exit_code, NULL,
                                  err, err_len))
            goto fail;
    } else {
        git_buf_puts_capped(&out, "behind_ahead=(none)\n", max_bytes, &truncated);
    }

    const char *status_args[] = {"status", "--porcelain=v1", "--branch", NULL};
    ds4_agent_git_result status = {0};
    if (!git_info_append_section(repo, "status:", status_args, true, &out,
                                 max_bytes, &truncated, &exit_code, &status,
                                 err, err_len))
        goto fail;
    git_buf_puts_capped(&out, "dirty=", max_bytes, &truncated);
    git_buf_puts_capped(&out, git_status_dirty(status.output) ? "true\n" : "false\n",
                        max_bytes, &truncated);
    ds4_agent_git_result_free(&status);

    const char *remote_args[] = {"remote", "-v", NULL};
    if (!git_info_append_section(repo, "remotes:", remote_args, false, &out,
                                 max_bytes, &truncated, &exit_code, NULL,
                                 err, err_len))
        goto fail;

    result->output = git_buf_take(&out);
    result->exit_code = exit_code;
    result->truncated = truncated;
    free(upstream);
    return true;

fail:
    free(out.ptr);
    free(upstream);
    return false;
}

void ds4_agent_git_result_free(ds4_agent_git_result *r) {
    if (!r) return;
    free(r->output);
    memset(r, 0, sizeof(*r));
}

static bool git_optional_ref_safe(const char *name, const char *ref,
                                  char *err, size_t err_len) {
    if (!ref || !ref[0]) return true;
    if (git_ref_safe(ref)) return true;
    git_set_err(err, err_len, "unsafe git %s: %s", name, ref);
    return false;
}

static bool git_tree_ref_safe(const char *name, const char *ref,
                              char *err, size_t err_len) {
    if (!git_optional_ref_safe(name, ref, err, err_len)) return false;
    if (ref && strchr(ref, ':')) {
        git_set_err(err, err_len, "unsafe git %s: %s", name, ref);
        return false;
    }
    return true;
}

static bool git_path_required(const char *path, const char *action,
                              char *err, size_t err_len) {
    if (!path || !path[0]) {
        git_set_err(err, err_len, "git %s requires path", action);
        return false;
    }
    for (const unsigned char *p = (const unsigned char *)path; *p; p++) {
        if (iscntrl(*p)) {
            git_set_err(err, err_len, "unsafe git path for %s", action);
            return false;
        }
    }
    return true;
}

static bool git_path_or_all_required(const ds4_agent_git_options *opts,
                                     char *err, size_t err_len) {
    if (opts->all) return true;
    return git_path_required(opts->path, opts->action, err, err_len);
}

static bool git_message_valid(const char *what, const char *message,
                              char *err, size_t err_len) {
    if (!message || !message[0]) {
        git_set_err(err, err_len, "git %s requires message", what);
        return false;
    }
    size_t n = strlen(message);
    if (n > 256) {
        git_set_err(err, err_len, "git %s message is too long", what);
        return false;
    }
    for (const unsigned char *p = (const unsigned char *)message; *p; p++) {
        if (iscntrl(*p)) {
            git_set_err(err, err_len, "git %s message must be one line", what);
            return false;
        }
    }
    return true;
}

static bool git_stash_ref_safe(const char *ref, char *err, size_t err_len) {
    if (!ref || !ref[0]) return true;
    if (!git_ref_safe(ref) ||
        strncmp(ref, "stash@{", strlen("stash@{")) != 0 ||
        ref[strlen(ref) - 1] != '}') {
        git_set_err(err, err_len, "unsafe git stash ref: %s", ref);
        return false;
    }
    return true;
}

static bool git_remote_safe(const char *remote, char *err, size_t err_len) {
    if (!remote || !remote[0]) {
        git_set_err(err, err_len, "git remote is required");
        return false;
    }
    if (remote[0] == '-') {
        git_set_err(err, err_len, "unsafe git remote: %s", remote);
        return false;
    }
    for (const unsigned char *p = (const unsigned char *)remote; *p; p++) {
        if (iscntrl(*p) || isspace(*p)) {
            git_set_err(err, err_len, "unsafe git remote: %s", remote);
            return false;
        }
    }
    return true;
}

static bool git_push_ref_safe(const char *ref, char *err, size_t err_len) {
    if (!ref || !ref[0]) {
        git_set_err(err, err_len, "git push requires ref");
        return false;
    }
    if (!git_ref_safe(ref) || strchr(ref, ':')) {
        git_set_err(err, err_len, "unsafe git push ref: %s", ref);
        return false;
    }
    return true;
}

static bool git_confirmed_or_dry_run(const ds4_agent_git_options *opts,
                                     char *err, size_t err_len) {
    if (opts->dry_run || opts->confirm) return true;
    git_set_err(err, err_len, "git %s requires confirm=true or dry_run=true",
                opts->action ? opts->action : "action");
    return false;
}

static bool git_required_ref(const char *action, const char *name,
                             const char *ref, char *err, size_t err_len) {
    if (!ref || !ref[0]) {
        git_set_err(err, err_len, "git %s requires %s", action, name);
        return false;
    }
    return git_optional_ref_safe(name, ref, err, err_len);
}

static bool git_require_clean_worktree(const char *repo, const char *action,
                                       char *err, size_t err_len) {
    const char *status_args[] = {"status", "--porcelain=v1", NULL};
    ds4_agent_git_result st = {0};
    bool ok = run_git_tail(repo, status_args, DS4_AGENT_GIT_DEFAULT_MAX_BYTES,
                           &st, err, err_len);
    if (!ok) return false;
    if (st.exit_code != 0) {
        char *line = git_first_line_value(st.output);
        git_set_err(err, err_len, "git %s clean check failed: %s",
                    action, line[0] ? line : "status failed");
        free(line);
        ds4_agent_git_result_free(&st);
        return false;
    }
    if (st.output && st.output[0]) {
        git_set_err(err, err_len, "git %s requires a clean working tree", action);
        ds4_agent_git_result_free(&st);
        return false;
    }
    ds4_agent_git_result_free(&st);
    return true;
}

static char *git_blob_spec(const char *ref, const char *path) {
    size_t rn = strlen(ref);
    size_t pn = strlen(path);
    char *out = git_xmalloc(rn + 1 + pn + 1);
    memcpy(out, ref, rn);
    out[rn] = ':';
    memcpy(out + rn + 1, path, pn + 1);
    return out;
}

bool ds4_agent_git_run_options(const ds4_agent_git_options *opts,
                               ds4_agent_git_result *result,
                               char *err,
                               size_t err_len) {
    if (!result) {
        git_set_err(err, err_len, "git result is required");
        return false;
    }
    if (!opts || !opts->action || !opts->action[0]) {
        git_set_err(err, err_len, "git action is required");
        return false;
    }

    const char *repo = opts->repo && opts->repo[0] ? opts->repo : ".";
    const char *action = opts->action;
    if (!strcmp(action, "info"))
        return git_run_info(repo, opts->max_bytes, result, err, err_len);

    const char *argv[64];
    int argc = git_argv_base(argv, repo);

    char limit_arg[64];
    char range_arg[1024];
    char line_arg[64];
    char *owned_arg = NULL;
    if (!strcmp(action, "status")) {
        argv_add(argv, &argc, "status");
        argv_add(argv, &argc, "--porcelain=v1");
        argv_add(argv, &argc, "--branch");
    } else if (!strcmp(action, "merge_base")) {
        const char *base = opts->base_ref && opts->base_ref[0] ?
                           opts->base_ref : "HEAD";
        const char *target = opts->target_ref && opts->target_ref[0] ?
                             opts->target_ref : opts->ref;
        if (!git_required_ref(action, "base_ref", base, err, err_len) ||
            !git_required_ref(action, "target_ref", target, err, err_len))
            return false;
        argv_add(argv, &argc, "merge-base");
        argv_add(argv, &argc, base);
        argv_add(argv, &argc, target);
    } else if (!strcmp(action, "merge_preview")) {
        const char *target = opts->target_ref && opts->target_ref[0] ?
                             opts->target_ref : opts->ref;
        if (!git_required_ref(action, "target_ref", target, err, err_len))
            return false;
        argv_add(argv, &argc, "merge-tree");
        argv_add(argv, &argc, "--write-tree");
        argv_add(argv, &argc, "--messages");
        argv_add(argv, &argc, "HEAD");
        argv_add(argv, &argc, target);
    } else if (!strcmp(action, "merge")) {
        const char *target = opts->target_ref && opts->target_ref[0] ?
                             opts->target_ref : opts->ref;
        if (!git_required_ref(action, "target_ref", target, err, err_len) ||
            !git_confirmed_or_dry_run(opts, err, err_len))
            return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "merge-tree");
            argv_add(argv, &argc, "--write-tree");
            argv_add(argv, &argc, "--messages");
            argv_add(argv, &argc, "HEAD");
            argv_add(argv, &argc, target);
        } else {
            if (!git_require_clean_worktree(repo, action, err, err_len))
                return false;
            argv_add(argv, &argc, "merge");
            argv_add(argv, &argc, "--ff-only");
            argv_add(argv, &argc, target);
        }
    } else if (!strcmp(action, "merge_abort")) {
        if (!git_confirmed_or_dry_run(opts, err, err_len)) return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "status");
            argv_add(argv, &argc, "--porcelain=v1");
            argv_add(argv, &argc, "--branch");
        } else {
            argv_add(argv, &argc, "merge");
            argv_add(argv, &argc, "--abort");
        }
    } else if (!strcmp(action, "rebase_preview")) {
        const char *upstream = opts->base_ref && opts->base_ref[0] ?
                               opts->base_ref : opts->ref;
        if (!git_required_ref(action, "upstream", upstream, err, err_len))
            return false;
        argv_add(argv, &argc, "log");
        argv_add(argv, &argc, "--oneline");
        argv_add(argv, &argc, "--decorate");
        snprintf(range_arg, sizeof(range_arg), "%s..HEAD", upstream);
        argv_add(argv, &argc, range_arg);
    } else if (!strcmp(action, "rebase")) {
        const char *upstream = opts->base_ref && opts->base_ref[0] ?
                               opts->base_ref : opts->ref;
        if (!git_required_ref(action, "upstream", upstream, err, err_len) ||
            !git_confirmed_or_dry_run(opts, err, err_len))
            return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "log");
            argv_add(argv, &argc, "--oneline");
            argv_add(argv, &argc, "--decorate");
            snprintf(range_arg, sizeof(range_arg), "%s..HEAD", upstream);
            argv_add(argv, &argc, range_arg);
        } else {
            if (!git_require_clean_worktree(repo, action, err, err_len))
                return false;
            argv_add(argv, &argc, "rebase");
            argv_add(argv, &argc, upstream);
        }
    } else if (!strcmp(action, "rebase_abort")) {
        if (!git_confirmed_or_dry_run(opts, err, err_len)) return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "status");
            argv_add(argv, &argc, "--porcelain=v1");
            argv_add(argv, &argc, "--branch");
        } else {
            argv_add(argv, &argc, "rebase");
            argv_add(argv, &argc, "--abort");
        }
    } else if (!strcmp(action, "remote_list")) {
        argv_add(argv, &argc, "remote");
        argv_add(argv, &argc, "-v");
    } else if (!strcmp(action, "changed_files")) {
        argv_add(argv, &argc, "status");
        argv_add(argv, &argc, "--porcelain=v1");
        argv_add(argv, &argc, "-uall");
        argv_add(argv, &argc, "--");
        if (opts->path && opts->path[0]) argv_add(argv, &argc, opts->path);
    } else if (!strcmp(action, "file_at_ref")) {
        const char *ref = opts->ref && opts->ref[0] ? opts->ref : "HEAD";
        if (!git_path_required(opts->path, action, err, err_len) ||
            !git_tree_ref_safe("ref", ref, err, err_len))
            return false;
        owned_arg = git_blob_spec(ref, opts->path);
        argv_add(argv, &argc, "show");
        argv_add(argv, &argc, "--no-ext-diff");
        argv_add(argv, &argc, owned_arg);
    } else if (!strcmp(action, "stage")) {
        if (!git_path_or_all_required(opts, err, err_len)) return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "status");
            argv_add(argv, &argc, "--porcelain=v1");
            argv_add(argv, &argc, "-uall");
            if (!opts->all) {
                argv_add(argv, &argc, "--");
                argv_add(argv, &argc, opts->path);
            }
        } else {
            argv_add(argv, &argc, "add");
            if (opts->all) argv_add(argv, &argc, "-A");
            argv_add(argv, &argc, "--");
            argv_add(argv, &argc, opts->all ? "." : opts->path);
        }
    } else if (!strcmp(action, "unstage")) {
        if (!git_path_or_all_required(opts, err, err_len)) return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "diff");
            argv_add(argv, &argc, "--cached");
            argv_add(argv, &argc, "--name-status");
            if (!opts->all) {
                argv_add(argv, &argc, "--");
                argv_add(argv, &argc, opts->path);
            }
        } else {
            argv_add(argv, &argc, "restore");
            argv_add(argv, &argc, "--staged");
            argv_add(argv, &argc, "--");
            argv_add(argv, &argc, opts->all ? "." : opts->path);
        }
    } else if (!strcmp(action, "commit")) {
        if (!git_message_valid("commit", opts->message, err, err_len)) return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "diff");
            argv_add(argv, &argc, "--cached");
            argv_add(argv, &argc, "--stat");
        } else {
            argv_add(argv, &argc, "commit");
            argv_add(argv, &argc, "-m");
            argv_add(argv, &argc, opts->message);
        }
    } else if (!strcmp(action, "worktree_restore")) {
        const char *ref = opts->ref && opts->ref[0] ? opts->ref : "HEAD";
        if (!git_path_or_all_required(opts, err, err_len) ||
            !git_tree_ref_safe("ref", ref, err, err_len))
            return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "diff");
            argv_add(argv, &argc, "--name-status");
            argv_add(argv, &argc, ref);
            argv_add(argv, &argc, "--");
            argv_add(argv, &argc, opts->all ? "." : opts->path);
        } else {
            argv_add(argv, &argc, "restore");
            argv_add(argv, &argc, "--source");
            argv_add(argv, &argc, ref);
            argv_add(argv, &argc, "--");
            argv_add(argv, &argc, opts->all ? "." : opts->path);
        }
    } else if (!strcmp(action, "switch")) {
        const char *ref = opts->ref;
        if (!ref || !ref[0]) {
            git_set_err(err, err_len, "git switch requires ref");
            return false;
        }
        if (!git_optional_ref_safe("ref", ref, err, err_len)) return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "rev-parse");
            argv_add(argv, &argc, "--verify");
            argv_add(argv, &argc, ref);
        } else {
            argv_add(argv, &argc, "switch");
            argv_add(argv, &argc, "--no-guess");
            argv_add(argv, &argc, ref);
        }
    } else if (!strcmp(action, "stash_list")) {
        int limit = opts->limit;
        if (limit <= 0) limit = 20;
        if (limit > 100) limit = 100;
        snprintf(limit_arg, sizeof(limit_arg), "-%d", limit);
        argv_add(argv, &argc, "stash");
        argv_add(argv, &argc, "list");
        argv_add(argv, &argc, limit_arg);
    } else if (!strcmp(action, "stash_push")) {
        if (!git_message_valid("stash_push", opts->message, err, err_len))
            return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "status");
            argv_add(argv, &argc, "--porcelain=v1");
            argv_add(argv, &argc, "-uall");
            if (opts->path && opts->path[0]) {
                argv_add(argv, &argc, "--");
                argv_add(argv, &argc, opts->path);
            }
        } else {
            argv_add(argv, &argc, "stash");
            argv_add(argv, &argc, "push");
            argv_add(argv, &argc, "--message");
            argv_add(argv, &argc, opts->message);
            if (opts->all) argv_add(argv, &argc, "--include-untracked");
            if (opts->path && opts->path[0]) {
                argv_add(argv, &argc, "--");
                argv_add(argv, &argc, opts->path);
            }
        }
    } else if (!strcmp(action, "stash_show")) {
        const char *ref = opts->ref && opts->ref[0] ? opts->ref : "stash@{0}";
        if (!git_stash_ref_safe(ref, err, err_len)) return false;
        argv_add(argv, &argc, "stash");
        argv_add(argv, &argc, "show");
        if (opts->patch) argv_add(argv, &argc, "--patch");
        else argv_add(argv, &argc, "--stat");
        argv_add(argv, &argc, ref);
    } else if (!strcmp(action, "stash_apply") ||
               !strcmp(action, "stash_pop") ||
               !strcmp(action, "stash_drop")) {
        const char *ref = opts->ref && opts->ref[0] ? opts->ref : "stash@{0}";
        if (!git_stash_ref_safe(ref, err, err_len)) return false;
        if (opts->dry_run) {
            argv_add(argv, &argc, "stash");
            argv_add(argv, &argc, "show");
            argv_add(argv, &argc, "--stat");
            argv_add(argv, &argc, ref);
        } else {
            argv_add(argv, &argc, "stash");
            if (!strcmp(action, "stash_apply")) argv_add(argv, &argc, "apply");
            else if (!strcmp(action, "stash_pop")) argv_add(argv, &argc, "pop");
            else argv_add(argv, &argc, "drop");
            argv_add(argv, &argc, ref);
        }
    } else if (!strcmp(action, "fetch")) {
        if (!git_remote_safe(opts->remote, err, err_len) ||
            !git_optional_ref_safe("ref", opts->ref, err, err_len) ||
            !git_confirmed_or_dry_run(opts, err, err_len))
            return false;
        argv_add(argv, &argc, "fetch");
        argv_add(argv, &argc, "--prune");
        if (opts->dry_run) argv_add(argv, &argc, "--dry-run");
        argv_add(argv, &argc, opts->remote);
        if (opts->ref && opts->ref[0]) argv_add(argv, &argc, opts->ref);
    } else if (!strcmp(action, "push")) {
        if (!git_remote_safe(opts->remote, err, err_len) ||
            !git_push_ref_safe(opts->ref, err, err_len) ||
            !git_confirmed_or_dry_run(opts, err, err_len))
            return false;
        argv_add(argv, &argc, "push");
        if (opts->dry_run) argv_add(argv, &argc, "--dry-run");
        argv_add(argv, &argc, opts->remote);
        argv_add(argv, &argc, opts->ref);
    } else if (!strcmp(action, "blame")) {
        const char *ref = opts->ref && opts->ref[0] ? opts->ref : "HEAD";
        int start = opts->start_line > 0 ? opts->start_line : 1;
        int count = opts->line_count > 0 ? opts->line_count : 80;
        if (count > 1000) count = 1000;
        if (!git_path_required(opts->path, action, err, err_len) ||
            !git_optional_ref_safe("ref", ref, err, err_len))
            return false;
        snprintf(line_arg, sizeof(line_arg), "%d,+%d", start, count);
        argv_add(argv, &argc, "blame");
        argv_add(argv, &argc, "--no-progress");
        argv_add(argv, &argc, "--date=short");
        argv_add(argv, &argc, "-L");
        argv_add(argv, &argc, line_arg);
        argv_add(argv, &argc, ref);
        argv_add(argv, &argc, "--");
        argv_add(argv, &argc, opts->path);
    } else if (!strcmp(action, "path_history") || !strcmp(action, "log_path")) {
        const char *ref = opts->ref;
        int limit = opts->limit;
        if (limit <= 0) limit = 20;
        if (limit > 100) limit = 100;
        if (!git_path_required(opts->path, action, err, err_len) ||
            !git_optional_ref_safe("ref", ref, err, err_len))
            return false;
        snprintf(limit_arg, sizeof(limit_arg), "--max-count=%d", limit);
        argv_add(argv, &argc, "log");
        if (opts->follow) argv_add(argv, &argc, "--follow");
        argv_add(argv, &argc, "--oneline");
        argv_add(argv, &argc, "--decorate");
        argv_add(argv, &argc, "--name-status");
        argv_add(argv, &argc, limit_arg);
        if (ref && ref[0]) argv_add(argv, &argc, ref);
        argv_add(argv, &argc, "--");
        argv_add(argv, &argc, opts->path);
    } else if (!strcmp(action, "diff")) {
        if (opts->name_only && opts->name_status) {
            git_set_err(err, err_len, "diff cannot use both name_only and name_status");
            return false;
        }
        if (opts->range && opts->range[0] &&
            ((opts->base_ref && opts->base_ref[0]) ||
             (opts->target_ref && opts->target_ref[0]))) {
            git_set_err(err, err_len, "diff range cannot be combined with base_ref or target_ref");
            return false;
        }
        if (opts->staged &&
            ((opts->range && opts->range[0]) ||
             (opts->base_ref && opts->base_ref[0]) ||
             (opts->target_ref && opts->target_ref[0]))) {
            git_set_err(err, err_len, "staged diff cannot be combined with refs or range");
            return false;
        }
        if (!git_optional_ref_safe("range", opts->range, err, err_len) ||
            !git_optional_ref_safe("base_ref", opts->base_ref, err, err_len) ||
            !git_optional_ref_safe("target_ref", opts->target_ref, err, err_len))
            return false;

        const char *range = opts->range;
        if ((!range || !range[0]) && opts->base_ref && opts->base_ref[0]) {
            const char *target = opts->target_ref && opts->target_ref[0] ?
                                 opts->target_ref : "HEAD";
            size_t need = strlen(opts->base_ref) + strlen(target) + 4;
            if (need > sizeof(range_arg)) {
                git_set_err(err, err_len, "git diff range is too long");
                return false;
            }
            snprintf(range_arg, sizeof(range_arg), "%s...%s",
                     opts->base_ref, target);
            range = range_arg;
        } else if ((!range || !range[0]) && opts->target_ref && opts->target_ref[0]) {
            range = opts->target_ref;
        }

        argv_add(argv, &argc, "diff");
        argv_add(argv, &argc, "--no-ext-diff");
        argv_add(argv, &argc, "--find-renames");
        if (opts->staged) argv_add(argv, &argc, "--cached");
        if (opts->stat) argv_add(argv, &argc, "--stat");
        if (opts->name_status) argv_add(argv, &argc, "--name-status");
        if (opts->name_only) argv_add(argv, &argc, "--name-only");
        if (opts->patch) argv_add(argv, &argc, "--patch");
        if (range && range[0]) argv_add(argv, &argc, range);
        argv_add(argv, &argc, "--");
        if (opts->path && opts->path[0]) argv_add(argv, &argc, opts->path);
    } else if (!strcmp(action, "log")) {
        int limit = opts->limit;
        if (limit <= 0) limit = 10;
        if (limit > 100) limit = 100;
        snprintf(limit_arg, sizeof(limit_arg), "--max-count=%d", limit);
        argv_add(argv, &argc, "log");
        argv_add(argv, &argc, "--oneline");
        argv_add(argv, &argc, "--decorate");
        argv_add(argv, &argc, limit_arg);
        argv_add(argv, &argc, "--");
        if (opts->path && opts->path[0]) argv_add(argv, &argc, opts->path);
    } else if (!strcmp(action, "show")) {
        const char *ref = opts->ref;
        if (!ref || !ref[0]) ref = "HEAD";
        if (!git_ref_safe(ref)) {
            git_set_err(err, err_len, "unsafe git ref: %s", ref ? ref : "");
            return false;
        }
        argv_add(argv, &argc, "show");
        if (opts->stat || !opts->patch) argv_add(argv, &argc, "--stat");
        if (opts->patch) argv_add(argv, &argc, "--patch");
        argv_add(argv, &argc, "--oneline");
        argv_add(argv, &argc, "--decorate");
        argv_add(argv, &argc, "--no-ext-diff");
        argv_add(argv, &argc, ref);
        argv_add(argv, &argc, "--");
        if (opts->path && opts->path[0]) argv_add(argv, &argc, opts->path);
    } else if (!strcmp(action, "ls_files")) {
        argv_add(argv, &argc, "ls-files");
        argv_add(argv, &argc, "--");
        if (opts->path && opts->path[0]) argv_add(argv, &argc, opts->path);
    } else {
        git_set_err(err, err_len, "unknown git action: %s", action);
        return false;
    }
    argv[argc] = NULL;
    bool ok = run_argv(argv, opts->max_bytes, result, err, err_len);
    free(owned_arg);
    return ok;
}

bool ds4_agent_git_run(const char *repo,
                       const char *action,
                       const char *path,
                       const char *ref,
                       int limit,
                       bool staged,
                       size_t max_bytes,
                       ds4_agent_git_result *result,
                       char *err,
                       size_t err_len) {
    ds4_agent_git_options opts = {
        .repo = repo,
        .action = action,
        .path = path,
        .ref = ref,
        .limit = limit,
        .staged = staged,
        .max_bytes = max_bytes,
    };
    return ds4_agent_git_run_options(&opts, result, err, err_len);
}
