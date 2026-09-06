/* Hook-enabled ds4-agent translation unit.
 * The wrapper keeps the original implementation intact while interposing only
 * the chat/token boundaries needed to emit hook events. */
#define main ds4_agent_core_main
#define ds4_chat_append_message ds4_hooks_chat_append_message
#define ds4_chat_append_assistant_prefix ds4_hooks_chat_append_assistant_prefix
#define ds4_tokens_push ds4_hooks_tokens_push
#define ds4_token_is_stop_for_think_mode ds4_hooks_token_is_stop_for_think_mode
#include "ds4_agent.c"
#undef ds4_token_is_stop_for_think_mode
#undef ds4_tokens_push
#undef ds4_chat_append_assistant_prefix
#undef ds4_chat_append_message
#undef main

/* ds4.h was included while the interposition macros were active, so declare
 * the original functions explicitly for the wrapper implementation below. */
void ds4_chat_append_message(ds4_engine *, ds4_tokens *, const char *, const char *);
void ds4_chat_append_assistant_prefix(ds4_engine *, ds4_tokens *, ds4_think_mode);
void ds4_tokens_push(ds4_tokens *, int);
bool ds4_token_is_stop_for_think_mode(ds4_engine *, int, ds4_think_mode);

#include "ds4_hooks.c"
#include "ds4_agent_hooks.c"
