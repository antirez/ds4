#include "ds4_agent_internal.h"



/* ============================================================================
 * Worker/UI Synchronization Helpers
 * ============================================================================
 */

int set_nonblock(int fd, bool on, int *old_flags) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (old_flags) *old_flags = flags;
    int next = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return fcntl(fd, F_SETFL, next);
}



/* Check and clear the raw_mode_needs_restore flag under the worker mutex.
 * Returns true if the UI thread should verify/reapply linenoise raw mode. */
bool worker_check_raw_mode_restore(agent_worker *w) {
    bool needs = false;
    pthread_mutex_lock(&w->mu);
    if (w->raw_mode_needs_restore) {
        w->raw_mode_needs_restore = false;
        needs = true;
    }
    pthread_mutex_unlock(&w->mu);
    return needs;
}



void drain_wake_fd(int fd) {
    char buf[128];
    for (;;) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n > 0) continue;
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}



/* Submit one user turn if the worker is idle.  Busy submissions are rejected so
 * the UI can keep the typed text editable instead of silently queueing it. */
bool worker_submit(agent_worker *w, const char *text) {
    pthread_mutex_lock(&w->mu);
    bool ok = w->initialized && w->status.state == AGENT_WORKER_IDLE && !w->cmd_text;
    if (ok) {
        w->cmd_text = xstrdup(text);
        /* A submitted turn is no longer idle, even if the worker thread has
         * not yet reached its real prefill accounting.  Non-interactive mode
         * depends on this to avoid exiting in the small handoff window between
         * accepting stdin and starting generation. */
        w->status.state = AGENT_WORKER_PREFILL;
        w->status.prefill_done = 0;
        w->status.prefill_total = 0;
        w->status.prefill_label = agent_next_prefill_label();
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        w->status.greedy_sampling = false;
        pthread_cond_signal(&w->cond);
    }
    pthread_mutex_unlock(&w->mu);
    return ok;
}



/* Request interruption at the next model/tool polling point. */
void worker_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->interrupt = true;
    pthread_mutex_unlock(&w->mu);
}



/* Stop the worker thread. */
void worker_stop(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->stop = true;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
}



/* The UI thread consumes output in batches.  Taking ownership of w->out under
 * the mutex keeps terminal writes outside the lock while preserving order. */
void worker_consume(agent_worker *w, char **out, size_t *out_len, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    if (out) {
        *out = w->out;
        *out_len = w->out_len;
        w->out = NULL;
        w->out_len = 0;
        w->out_cap = 0;
    }
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = w->cfg->gen.ctx_size;
    if (status) *status = w->status;
    w->wake_pending = false;
    pthread_mutex_unlock(&w->mu);
}



void worker_get_status(agent_worker *w, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = w->cfg->gen.ctx_size;
    *status = w->status;
    pthread_mutex_unlock(&w->mu);
}



bool worker_is_idle(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool idle = w->initialized &&
        (w->status.state == AGENT_WORKER_IDLE ||
         w->status.state == AGENT_WORKER_ERROR);
    pthread_mutex_unlock(&w->mu);
    return idle;
}



bool worker_is_initialized(agent_worker *w, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = w->cfg->gen.ctx_size;
    if (status) *status = w->status;
    bool initialized = w->initialized;
    pthread_mutex_unlock(&w->mu);
    return initialized;
}



bool stdout_is_tty(void) {
    return isatty(STDOUT_FILENO) != 0;
}



char *agent_format_user_prompt_echo(const char *text) {
    agent_buf b = {0};
    if (stdout_is_tty()) {
        agent_buf_puts(&b, "\x1b[1;91m*\x1b[1;97m ");
        agent_buf_puts(&b, text);
        agent_buf_puts(&b, "\x1b[0m\n\n");
    } else {
        agent_buf_puts(&b, "* ");
        agent_buf_puts(&b, text);
        agent_buf_puts(&b, "\n\n");
    }
    return agent_buf_take(&b);
}



void agent_echo_user_prompt(const char *text) {
    char *msg = agent_format_user_prompt_echo(text);
    printf("%s", msg);
    fflush(stdout);
    free(msg);
}

