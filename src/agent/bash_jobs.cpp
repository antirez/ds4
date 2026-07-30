#include "pulsar_agent_internal.h"



int agent_bash_job::display_lines() const {
    const auto *job = this;
    if (!job || job->bytes == 0) return 0;
    return job->newline_count + (job->last_byte != '\n');
}



void agent_bash_job::note_output(const char *s, size_t n) {
    auto *job = this;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\n') job->newline_count++;
    }
    if (n) job->last_byte = s[n - 1];
    job->bytes += n;
}



void agent_bash_job::job_free() {
    auto *job = this;
    if (!job) return;
    if (job->running && job->pid > 0) {
        kill(-job->pid, SIGKILL);
        kill(job->pid, SIGKILL);
        waitpid(job->pid, NULL, 0);
    }
    if (job->pipe_fd >= 0) close(job->pipe_fd);
    if (job->tmp_fd >= 0) close(job->tmp_fd);
    /* Remove the mkstemp output file.  Only the three spawn-time error paths
     * unlinked it before, so every COMPLETED bash tool call left a
     * /tmp/pulsar_agent_output_* behind for the life of the box — a long agent
     * session orphaned one per command, sized by that command's full output. */
    if (job->path[0]) unlink(job->path);
    free(job->cmd);
    free(job);
}



void agent_bash_jobs_free(agent_worker *w) {
    agent_bash_job *job = w->bash_jobs;
    while (job) {
        agent_bash_job *next = job->next;
        job->job_free();
        job = next;
    }
    w->bash_jobs = NULL;
}



agent_bash_job *agent_bash_find_job(agent_worker *w, int id, pid_t pid) {
    for (agent_bash_job *job = w->bash_jobs; job; job = job->next) {
        if ((id > 0 && job->id == id) || (id <= 0 && pid > 0 && job->pid == pid))
            return job;
    }
    return NULL;
}



void agent_bash_remove_job(agent_worker *w, agent_bash_job *target) {
    agent_bash_job **link = &w->bash_jobs;
    while (*link) {
        if (*link == target) {
            *link = target->next;
            target->next = NULL;
            target->job_free();
            return;
        }
        link = &(*link)->next;
    }
}



void agent_bash_job::drain() {
    auto *job = this;
    if (!job || job->pipe_fd < 0) return;
    char tmp[4096];
    for (;;) {
        ssize_t n = read(job->pipe_fd, tmp, sizeof(tmp));
        if (n > 0) {
            job->note_output(tmp, (size_t)n);
            if (job->tmp_fd >= 0) write_all(job->tmp_fd, tmp, (size_t)n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}



static void agent_worker_note_terminal_mode_may_have_changed(agent_worker *w) {
    if (!w) return;
    pthread_mutex_lock(&w->mu);
    w->raw_mode_needs_restore = true;
    pthread_mutex_unlock(&w->mu);
}



void agent_bash_job::finalize(int status) {
    auto *job = this;
    job->drain();
    if (job->pipe_fd >= 0) {
        close(job->pipe_fd);
        job->pipe_fd = -1;
    }
    if (job->tmp_fd >= 0) {
        close(job->tmp_fd);
        job->tmp_fd = -1;
    }
    if (WIFEXITED(status)) job->exit_status = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) job->exit_status = 128 + WTERMSIG(status);
    else job->exit_status = -1;
    job->running = false;
    /* A child can still open /dev/tty directly and alter terminal state even
     * though its stdin is /dev/null.  Ask the UI thread to verify raw mode at
     * a safe point instead of touching linenoise from the worker path. */
    agent_worker_note_terminal_mode_may_have_changed(job->worker);
}



/* Drain available output, notice process exit, and enforce timeout.  This is
 * called opportunistically by status/wait/compaction instead of a background
 * reaper thread, keeping all bash job state owned by the agent worker. */
void agent_bash_job::poll() {
    auto *job = this;
    if (!job || !job->running) return;
    job->drain();

    int status = 0;
    pid_t rc = waitpid(job->pid, &status, WNOHANG);
    if (rc == job->pid) {
        job->finalize(status);
        return;
    }
    if (rc < 0 && errno != EINTR) {
        job->exit_status = -1;
        job->running = false;
        if (job->pipe_fd >= 0) {
            close(job->pipe_fd);
            job->pipe_fd = -1;
        }
        if (job->tmp_fd >= 0) {
            close(job->tmp_fd);
            job->tmp_fd = -1;
        }
        agent_worker_note_terminal_mode_may_have_changed(job->worker);
        return;
    }
    if (agent_now_sec() - job->start_time >= job->timeout_sec) {
        job->timed_out = true;
        kill(-job->pid, SIGKILL);
        kill(job->pid, SIGKILL);
        while (waitpid(job->pid, &status, 0) < 0 && errno == EINTR) {}
        job->finalize(status);
    }
}



/* Spawn a shell command into its own process group so bash_stop/timeout can
 * kill grandchildren created by the shell, not just the /bin/sh wrapper. */
agent_bash_job *agent_bash_start(agent_worker *w, const char *cmd,
                                        int timeout_sec, char *err, size_t err_len) {
    char tmp_path[] = "/tmp/ds4_agent_output_XXXXXX";
    int tmpfd = mkstemp(tmp_path);
    if (tmpfd < 0) {
        snprintf(err, err_len, "failed to create temporary output file: %s", strerror(errno));
        return NULL;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        snprintf(err, err_len, "failed to create pipe: %s", strerror(errno));
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, err_len, "failed to fork: %s", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    if (pid == 0) {
        setpgid(0, 0);
        close(tmpfd);
        /* The bash tool is not interactive.  Give the shell /dev/null as
         * stdin so it does not inherit the live linenoise terminal and reset
         * it from raw mode to cooked mode behind the agent's back. */
        int null_fd = open("/dev/null", O_RDONLY);
        if (null_fd >= 0) {
            if (dup2(null_fd, STDIN_FILENO) < 0)
                close(STDIN_FILENO);
            if (null_fd != STDIN_FILENO)
                close(null_fd);
        } else {
            close(STDIN_FILENO);
        }
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", cmd ? cmd : "", (char *)NULL);
        _exit(127);
    }

    close(pipefd[1]);
    setpgid(pid, pid);
    int old_flags;
    set_nonblock(pipefd[0], true, &old_flags);

    agent_bash_job *job = (agent_bash_job *)agent_xmalloc(sizeof(*job));
    memset(job, 0, sizeof(*job));
    if (w->next_bash_job_id <= 0) w->next_bash_job_id = 1;
    job->id = w->next_bash_job_id++;
    job->pid = pid;
    job->pipe_fd = pipefd[0];
    job->tmp_fd = tmpfd;
    snprintf(job->path, sizeof(job->path), "%s", tmp_path);
    job->cmd = xstrdup(cmd);
    job->start_time = agent_now_sec();
    job->timeout_sec = timeout_sec;
    job->exit_status = -1;
    job->running = true;
    job->worker = w;
    job->next = w->bash_jobs;
    w->bash_jobs = job;
    return job;
}



static void agent_tail_append(agent_buf *b, const char *s, size_t n, size_t max) {
    if (!n) return;
    agent_buf_append(b, s, n);
    if (b->len > max) {
        size_t drop = b->len - max;
        memmove(b->ptr, b->ptr + drop, b->len - drop + 1);
        b->len -= drop;
    }
}



/* Read the first max_lines from the output file, with a byte cap to avoid a
 * pathological single long line flooding the next model turn. */
char *agent_bash_job::read_head(int max_lines,
                                size_t max_bytes, int *lines_read,
                                bool *byte_limited) const {
    const auto *job = this;
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf out = {0};
    int lines = 0;
    while (lines < max_lines && out.len < max_bytes) {
        int c = fgetc(fp);
        if (c == EOF) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
        char ch = (char)c;
        agent_buf_append(&out, &ch, 1);
        if (ch == '\n') lines++;
    }
    if (out.len >= max_bytes && !feof(fp) && byte_limited) *byte_limited = true;
    fclose(fp);
    if (lines_read) *lines_read = lines + (out.len && out.ptr[out.len - 1] != '\n');
    if (!out.ptr) return xstrdup("");
    return agent_buf_take(&out);
}



/* Read the last max_lines from the full output file.  The model-visible label
 * says "tail -N <file>" so it is clear this is not the complete output. */
char *agent_bash_job::read_tail_lines(int max_lines) const {
    const auto *job = this;
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf tail = {0};
    char tmp[2048];
    for (;;) {
        size_t n = fread(tmp, 1, sizeof(tmp), fp);
        if (n) agent_tail_append(&tail, tmp, n, AGENT_BASH_TAIL_BYTES);
        if (n < sizeof(tmp)) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
    }
    fclose(fp);
    if (!tail.ptr) return xstrdup("");

    char *start = tail.ptr;
    int newlines = 0;
    for (char *p = tail.ptr + tail.len; p > tail.ptr; p--) {
        if (p[-1] == '\n' && ++newlines > max_lines) {
            start = p;
            break;
        }
    }
    char *out = xstrdup(start);
    free(tail.ptr);
    return out;
}



/* Build the tool result for a bash job.  mark_observed advances the per-job
 * cursor so the next status reports only fresh output. */
char *agent_bash_job::observation(bool mark_observed) {
    auto *job = this;
    job->poll();
    bool first_observation = !job->observed_once;
    int display_lines = job->display_lines();
    double elapsed = agent_now_sec() - job->start_time;

    agent_buf out = {0};
    char line[PATH_MAX + 256];
    if (job->running) {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=running elapsed_sec=%.1f timeout_sec=%.0f\n",
            job->id, (long)job->pid, elapsed, job->timeout_sec);
    } else {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=done elapsed_sec=%.1f timed_out=%d\n",
            job->id, (long)job->pid, elapsed, job->timed_out ? 1 : 0);
    }
    agent_buf_puts(&out, line);
    if (!job->running) {
        snprintf(line, sizeof(line), "exit_status=%d\n", job->exit_status);
        agent_buf_puts(&out, line);
    }

    if (job->bytes == 0) {
        agent_buf_puts(&out, "<output>\n</output>\n");
    } else if (first_observation) {
        int shown_lines = 0;
        bool byte_limited = false;
        char *head = job->read_head(AGENT_BASH_HEAD_LINES,
                                          AGENT_BASH_HEAD_BYTES,
                                          &shown_lines, &byte_limited);
        bool truncated = byte_limited || display_lines > shown_lines;
        if (!job->running && !truncated) {
            agent_buf_puts(&out, "<output>\n");
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</output>\n");
        } else {
            snprintf(line, sizeof(line),
                     "output_path=%s (%zu bytes, %d lines)\n",
                     job->path[0] ? job->path : "<unavailable>",
                     job->bytes, display_lines);
            agent_buf_puts(&out, line);
            snprintf(line, sizeof(line), "<head -%d %s>\n",
                     AGENT_BASH_HEAD_LINES, job->path);
            agent_buf_puts(&out, line);
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</head>\n");
        }
        free(head);
    } else {
        int tail_lines = job->running ? AGENT_BASH_PROGRESS_TAIL_LINES :
                                        AGENT_BASH_FINAL_TAIL_LINES;
        char *tail = job->read_tail_lines(tail_lines);
        snprintf(line, sizeof(line),
                 "output_path=%s (%zu bytes, %d lines)\n",
                 job->path[0] ? job->path : "<unavailable>",
                 job->bytes, display_lines);
        agent_buf_puts(&out, line);
        snprintf(line, sizeof(line), "<tail -%d %s>\n", tail_lines, job->path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, tail);
        if (tail[0] && tail[strlen(tail) - 1] != '\n') agent_buf_puts(&out, "\n");
        snprintf(line, sizeof(line), "</tail>\n");
        agent_buf_puts(&out, line);
        free(tail);
    }
    if (job->running) {
        snprintf(line, sizeof(line),
            "\nUse bash_status job=%d to get info before refresh time; use bash_stop job=%d to stop execution\n",
            job->id, job->id);
        agent_buf_puts(&out, line);
    }

    if (mark_observed) {
        job->observed_bytes = job->bytes;
        job->observed_display_lines = display_lines;
        job->observed_once = true;
    }
    return agent_buf_take(&out);
}



static void agent_bash_publish_observation(agent_worker *w, const char *obs) {
    if (!obs || !obs[0]) return;
    const char *body = NULL;
    const char *label = strstr(obs, "\n<head ");
    const char *close = NULL;
    if (label) {
        close = "</head>";
    } else {
        label = strstr(obs, "\n<tail ");
        if (label) close = "</tail>";
    }
    if (label) {
        const char *tag_end = strstr(label, ">\n");
        if (tag_end) {
            agent_publish(w, "\x1b[90m", 5);
            if (strstr(label, "\n<head ") == label)
                agent_publish(w, "[showing first output lines]\n",
                              strlen("[showing first output lines]\n"));
            else
                agent_publish(w, "[showing last output lines]\n",
                              strlen("[showing last output lines]\n"));
            agent_publish(w, "\x1b[0m", 4);
            body = tag_end + 2;
        }
    } else {
        label = strstr(obs, "\n<output>\n");
        if (label) {
            body = label + strlen("\n<output>\n");
            close = "</output>";
        }
    }
    if (!body || !body[0]) return;
    const char *end = close ? strstr(body, close) : NULL;
    size_t n = end ? (size_t)(end - body) : strlen(body);
    if (n) {
        bool failed = strstr(obs, "status=done") && !strstr(obs, "exit_status=0\n");
        if (failed) agent_publish(w, "\x1b[38;5;208m", 11);
        agent_publish(w, body, n);
        if (body[n - 1] != '\n') agent_publish(w, "\n", 1);
        if (failed) agent_publish(w, "\x1b[0m", 4);
    }
}



static void agent_bash_refresh_for(agent_worker *w, agent_bash_job *job,
                                   int refresh_sec) {
    double start = agent_now_sec();
    while (job->running && agent_now_sec() - start < refresh_sec) {
        if (worker_should_interrupt(w)) break;
        job->poll();
        if (!job->running) break;
        struct pollfd pfd = {.fd = job->pipe_fd, .events = POLLIN};
        poll(&pfd, 1, 100);
    }
    job->poll();
}



/* Common implementation for bash, bash_status, and bash_stop. */
char *agent_bash_job_tool_result(agent_worker *w, agent_bash_job *job,
                                        bool wait, int refresh_sec,
                                        bool stop, bool remove_if_done) {
    if (stop && job->running) {
        kill(-job->pid, SIGTERM);
        kill(job->pid, SIGTERM);
        double start = agent_now_sec();
        while (job->running && agent_now_sec() - start < 1.0) {
            job->poll();
            if (!job->running) break;
            usleep(20000);
        }
        if (job->running) {
            kill(-job->pid, SIGKILL);
            kill(job->pid, SIGKILL);
        }
    }
    if (wait || stop) agent_bash_refresh_for(w, job, refresh_sec);
    else job->poll();

    char *obs = job->observation(true);
    agent_bash_publish_observation(w, obs);
    if (remove_if_done && !job->running) agent_bash_remove_job(w, job);
    return obs;
}



int agent_tool_job_id(const agent_tool_call *call) {
    return agent_parse_int_default(agent_tool_arg_value(call, "job"), 0, 0, INT_MAX);
}



pid_t agent_tool_pid(const agent_tool_call *call) {
    return (pid_t)agent_parse_int_default(agent_tool_arg_value(call, "pid"), 0, 0, INT_MAX);
}

