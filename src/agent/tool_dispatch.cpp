#include "ds4_agent_internal.h"



/* ============================================================================
 * Tool Dispatch
 * ============================================================================
 */

/* Execute one parsed DSML tool call and return the text that will be appended as
 * the tool-role result.  UI visualization already happened while streaming; this
 * function is only about side effects and the model-visible observation. */
static char *agent_execute_tool_call(agent_worker *w, const agent_tool_call *call) {
    agent_buf result = {0};
    if (!call->name) return xstrdup("Tool error: missing tool name\n");

    if (!strcmp(call->name, "read")) return agent_tool_read(w, call);
    if (!strcmp(call->name, "more")) return agent_tool_more(w, call);
    if (!strcmp(call->name, "write")) return agent_tool_write(w, call);
    if (!strcmp(call->name, "list")) return agent_tool_list(call);
    if (!strcmp(call->name, "edit")) return agent_tool_edit(w, call);
    if (!strcmp(call->name, "search")) return agent_tool_search(w, call);

    if (!strcmp(call->name, "bash")) {
        const char *cmd = agent_tool_arg_value(call, "command");
        if (!cmd || !cmd[0]) return xstrdup("Tool error: bash requires command\n");
        int timeout = agent_parse_timeout(agent_tool_arg_value(call, "timeout_sec"));
        int refresh = agent_parse_int_default(agent_tool_arg_value(call, "refresh_sec"),
                                              60, 1, 3600);
        char err[160] = {0};
        agent_bash_job *job = agent_bash_start(w, cmd, timeout, err, sizeof(err));
        if (!job) {
            agent_buf_puts(&result, "Tool error: bash failed to start: ");
            agent_buf_puts(&result, err[0] ? err : "unknown error");
            agent_buf_puts(&result, "\n");
            return agent_buf_take(&result);
        }
        return agent_bash_job_tool_result(w, job, true, refresh, false, true);
    }

    if (!strcmp(call->name, "bash_status") ||
        !strcmp(call->name, "bash_stop"))
    {
        int job_id = agent_tool_job_id(call);
        pid_t pid = agent_tool_pid(call);
        agent_bash_job *job = agent_bash_find_job(w, job_id, pid);
        if (!job) {
            char msg[128];
            snprintf(msg, sizeof(msg), "Tool error: bash job not found: job=%d pid=%ld\n",
                     job_id, (long)pid);
            return xstrdup(msg);
        }
        int refresh = agent_parse_int_default(agent_tool_arg_value(call, "refresh_sec"),
                                              60, 1, 3600);
        bool stop = !strcmp(call->name, "bash_stop");
        bool wait = stop;
        return agent_bash_job_tool_result(w, job, wait, refresh, stop, true);
    }

    {
        char header[256];
        snprintf(header, sizeof(header), "\n[tool:%s] unknown tool\n", call->name);
        agent_publish(w, header, strlen(header));
        agent_buf_puts(&result, "Tool error: unknown tool: ");
        agent_buf_puts(&result, call->name);
        agent_buf_puts(&result, "\n");
        return agent_buf_take(&result);
    }
}



/* Execute all tool calls from one DSML block, preserving per-call labels in the
 * combined result so the model can associate observations with calls. */
char *agent_execute_tool_calls(agent_worker *w, const agent_tool_calls *calls) {
    agent_buf all = {0};
    for (int i = 0; i < calls->len; i++) {
        char *res = agent_execute_tool_call(w, &calls->v[i]);
        char hdr[128];
        snprintf(hdr, sizeof(hdr), "Tool result %d (%s):\n", i + 1,
                 calls->v[i].name ? calls->v[i].name : "unknown");
        agent_buf_puts(&all, hdr);
        agent_buf_puts(&all, res);
        if (res[0] && res[strlen(res) - 1] != '\n') agent_buf_puts(&all, "\n");
        free(res);
    }
    if (calls->len == 0) agent_buf_puts(&all, "Tool error: empty tool call block\n");
    return agent_buf_take(&all);
}



/* If compaction happens while a bash process is still alive, inject a small
 * tool-role reminder into the rebuilt transcript.  Otherwise the summary could
 * preserve the user's task but lose the fact that an external process still
 * needs status/wait/stop handling. */
char *agent_bash_jobs_compaction_observation(agent_worker *w) {
    if (!w->bash_jobs) return NULL;
    agent_buf out = {0};
    agent_buf_puts(&out,
        "Bash job update after context compaction. Running jobs still need explicit bash_status or bash_stop if relevant.\n");
    for (agent_bash_job *job = w->bash_jobs, *next = NULL; job; job = next) {
        next = job->next;
        char *obs = agent_bash_observation(job, true);
        char hdr[64];
        snprintf(hdr, sizeof(hdr), "\nJob %d:\n", job->id);
        agent_buf_puts(&out, hdr);
        agent_buf_puts(&out, obs);
        free(obs);
        if (!job->running) agent_bash_remove_job(w, job);
    }
    return agent_buf_take(&out);
}

