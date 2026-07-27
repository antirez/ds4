#include "ds4_agent_internal.h"



int set_nonblock(int fd, bool on, int *old_flags);


bool agent_parse_bool_default(const char *s, bool def);



unsigned agent_next_prefill_label(void);



volatile sig_atomic_t agent_sigint;


agent_worker *agent_completion_worker;



void agent_trace(agent_worker *w, const char *fmt, ...);


void agent_trace_text(agent_worker *w, const char *label,
                             const char *text, size_t len);


void agent_publish_system_status(agent_worker *w, const char *msg);


bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
                                     char *err, size_t err_len);


int agent_worker_sync_tokens(agent_worker *w, const ds4_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len);

