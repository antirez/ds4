/* pulsar_agent_internal.h — internal shared declarations for the agent sources.
 * Produced by the multi-TU split of pulsar_agent.c; edit freely (the
 * generator is not part of the build). */
#ifndef PULSAR_AGENT_INTERNAL_H
#define PULSAR_AGENT_INTERNAL_H

#include "pulsar.h"
#include "pulsar_help.h"
#include "pulsar_kvstore.h"
#include "linenoise.h"

#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <fnmatch.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <regex.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>


/* This is intentionally not in linenoise.h, but it is part of the existing
 * multiplexed editor implementation.  The agent uses it only to restore text
 * after Enter is pressed while the model is still busy. */
int linenoiseEditInsert(struct linenoiseState *l, const char *c, size_t clen);

/* ---- shared macros ---- */



#define AGENT_SYSTEM_PROMPT_REMINDER_TOKENS 50000


/* Poor man's code highlighter inspired by antirez/kilo: a tiny language table
 * plus one line-oriented tokenizer for comments, strings, numbers, and
 * separator-bounded keywords.  This is deliberately not a full parser; it is
 * only for making fenced Markdown code readable in the terminal. */
#define AGENT_HL_NORMAL 0
#define AGENT_HL_COMMENT 1
#define AGENT_HL_KEYWORD1 2
#define AGENT_HL_KEYWORD2 3
#define AGENT_HL_STRING 4
#define AGENT_HL_NUMBER 5

#define AGENT_SYNTAX_NUMBERS (1u<<0)
#define AGENT_SYNTAX_STRINGS (1u<<1)
#define AGENT_SYNTAX_BACKTICK_STRINGS (1u<<2)
#define AGENT_SYNTAX_CASE_INSENSITIVE (1u<<3)


#define AGENT_HISTORY_DEFAULT_TURNS 3
#define AGENT_HISTORY_MAX_TURNS 200
#define AGENT_HISTORY_ASSISTANT_MAX_LINES 80
#define AGENT_HISTORY_ASSISTANT_MAX_BYTES 12000


#define AGENT_FILE_MAX_BYTES (16*1024*1024)
#define AGENT_READ_DEFAULT_LINES 500
#define AGENT_TOOL_RESULT_RESERVE_TOKENS 1024
#define AGENT_EDIT_UPTO_MIN_PREFIX_BYTES 64
#define AGENT_EDIT_UPTO_MIN_PREFIX_LINES 2
#define AGENT_COMPACT_SOFT_PERCENT 85
#define AGENT_COMPACT_MIN_FREE_TOKENS 8192
#define AGENT_COMPACT_TAIL_DIVISOR 10
#define AGENT_COMPACT_TAIL_CAP_TOKENS 50000
#define AGENT_COMPACT_SUMMARY_MAX_TOKENS 4096




/* ============================================================================
 * Asynchronous Bash Jobs
 * ============================================================================
 *
 * Bash commands are tracked jobs, not blocking one-shot calls.  Each job owns a
 * process, a pipe, and a secure /tmp output file.  The first observation is
 * head-biased so headers and early errors are visible; later progress updates
 * are tail-biased and report how much output was added since the previous
 * observation.
 */

#define AGENT_BASH_HEAD_BYTES (8*1024)
#define AGENT_BASH_HEAD_LINES 100
#define AGENT_BASH_TAIL_BYTES (32*1024)
#define AGENT_BASH_PROGRESS_TAIL_LINES 4
#define AGENT_BASH_FINAL_TAIL_LINES 20

#define AGENT_INPUT_INITIAL_BUFLEN 4096
#define AGENT_INPUT_MAX_BUFLEN (1024*1024)
#define AGENT_STATUS_STYLE_START "\x1b[48;5;238;38;5;252m"
#define AGENT_STATUS_STYLE_END "\x1b[0m"
#define AGENT_STATUS_BAR_FILL "\x1b[48;5;238;38;5;201;1m"
#define AGENT_QUEUE_STYLE "\x1b[38;5;87;1m"
#define AGENT_STATUS_REDRAW_INTERVAL_SEC 0.20
#define AGENT_PROGRESS_BAR_WIDTH 32
#define AGENT_PROGRESS_BAR_MAX_BYTES 256

/* ---- shared types ---- */

/* ============================================================================
 * Configuration, Worker State, And Streaming Types
 * ============================================================================
 *
 * The agent is intentionally a single process: the UI thread owns terminal
 * input/output, while the worker thread owns the live DS4 session and KV state.
 * These types define the shared state and the small streaming state machines
 * used to render sampled assistant text and DSML tool calls as they arrive.
 */

typedef struct {
    const char *prompt;
    const char *system;
    const char *trace_path;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    pulsar_think_mode think_mode;
} agent_generation_options;

typedef struct {
    pulsar_engine_options engine;
    agent_generation_options gen;
    const char *chdir_path;
    bool non_interactive;
} agent_config;

typedef enum {
    AGENT_WORKER_IDLE,
    AGENT_WORKER_PREFILL,
    AGENT_WORKER_GENERATING,
    AGENT_WORKER_COMPACTING,
    AGENT_WORKER_SAVING,
    AGENT_WORKER_ERROR,
    AGENT_WORKER_STOPPED,
} agent_worker_state;

typedef struct {
    agent_worker_state state;
    int prefill_done;
    int prefill_total;
    unsigned prefill_label;
    double prefill_tps;
    int generated;
    double gen_tps;
    bool greedy_sampling;
    int ctx_used;
    int ctx_size;
    char error[256];
} agent_status;

typedef struct agent_bash_job agent_bash_job;

typedef struct {
    pulsar_engine *engine;
    agent_config *cfg;
    pulsar_session *session;
    pulsar_tokens transcript;
    char *cache_dir;
    char *sysprompt_path;
    char session_sha[41];
    char *session_title;
    uint64_t session_created_at;
    char *legacy_session_path_to_delete;
    bool user_activity;
    bool session_dirty;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cond;
    int wake_fd[2];
    FILE *trace;
    bool wake_pending;
    bool stop;
    bool interrupt;
    bool initialized;
    bool save_requested;
    bool compact_requested;
    int progress_base;
    double progress_started_at;
    int last_system_prompt_reminder_at;
    char *cmd_text;
    agent_status status;
    char *out;
    size_t out_len;
    size_t out_cap;
    bool queued_user_drain_pending;
    bool queued_user_drain_answered;
    char *queued_user_drain_text;
    bool datetime_context_injected;
    char more_path[PATH_MAX];
    int more_next_line;
    bool more_bare;
    bool more_valid;
    agent_bash_job *bash_jobs;
    int next_bash_job_id;
    bool raw_mode_needs_restore;
} agent_worker;

typedef struct agent_tail_capture {
    char *buf;
    size_t cap;
    size_t start;
    size_t len;
    size_t total;
} agent_tail_capture;

typedef enum {
    AGENT_MD_PENDING_NONE,
    AGENT_MD_PENDING_STAR,
    AGENT_MD_PENDING_BACKTICK,
} agent_markdown_pending;

typedef struct agent_syntax agent_syntax;

typedef struct {
    pulsar_engine *engine;
    agent_worker *worker;
    bool format_thinking;
    bool format_markdown;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    bool wrote_visible_output;
    bool md_bold;
    bool md_italic;
    bool md_inline_code;
    bool md_code_block;
    bool md_fence_info;
    bool md_code_line_start;
    bool md_code_in_ml_comment;
    bool md_syntax_silent;
    bool md_syntax_has_highlight;
    agent_markdown_pending md_pending;
    size_t md_pending_len;
    const agent_syntax *md_syntax;
    char md_fence_lang[32];
    size_t md_fence_lang_len;
    const char *md_code_line_prefix;
    const char *md_code_line_prefix_color;
    bool md_code_highlight_upto;
    char *md_code_line;
    size_t md_code_line_len;
    size_t md_code_line_cap;
    char pending[16];
    size_t pending_len;
    char utf8_pending[4];
    size_t utf8_pending_len;
    size_t utf8_pending_need;
    agent_tail_capture *capture;
} agent_token_renderer;

typedef struct {
    char *name;
    char *value;
    bool is_string;
} agent_tool_arg;

typedef struct {
    char *name;
    agent_tool_arg *args;
    int argc;
    int argcap;
} agent_tool_call;

typedef struct {
    agent_tool_call *v;
    int len;
    int cap;
} agent_tool_calls;

typedef enum {
    AGENT_DSML_SEARCH,
    AGENT_DSML_STRUCTURAL,
    AGENT_DSML_PARAM_VALUE,
    AGENT_DSML_DONE,
    AGENT_DSML_ERROR,
} agent_dsml_state;

typedef struct {
    agent_dsml_state state;
    char search_tail[64];
    size_t search_len;
    char *raw;
    size_t raw_len;
    size_t raw_cap;
    size_t parse_pos;
    agent_tool_call current;
    char *param_name;
    bool param_is_string;
    size_t param_value_start;
    bool param_close_prefix;
    agent_tool_calls calls;
    char error[160];
} agent_dsml_parser;

typedef enum {
    AGENT_TOOL_PARAM_NORMAL,
    AGENT_TOOL_PARAM_PATH,
    AGENT_TOOL_PARAM_OFFSET,
    AGENT_TOOL_PARAM_CONTENT,
    AGENT_TOOL_PARAM_DIFF_OLD,
    AGENT_TOOL_PARAM_DIFF_NEW,
    AGENT_TOOL_PARAM_BASH_COMMAND,
} agent_tool_param_kind;

typedef struct {
    bool active;
    bool tool_announced;
    bool param_active;
    bool at_line_start;
    bool last_output_newline;
    agent_tool_param_kind param_kind;
    char tool_name[64];
    char param_name[64];
    char param_end_tail[64];
    size_t param_end_len;
    bool read_style;
    bool read_prefix_rendered;
    bool read_line_rendered;
    char read_path[512];
    char read_start[32];
    char read_max[32];
    char read_whole[8];
    char tool_path[512];
    bool code_param_active;
} agent_tool_visualizer;

typedef struct {
    char tail[32];
    size_t len;
} agent_dsml_marker_detector;

typedef struct {
    agent_token_renderer *renderer;
    agent_dsml_parser *parser;
    agent_tool_visualizer viz;
    bool in_think;
    bool dsml_active;
    bool dsml_ignored;
    bool replay;
    char pending[16];
    size_t pending_len;
    char dsml_start_tail[64];
    size_t dsml_start_len;
    agent_dsml_marker_detector plain_dsml;
    agent_dsml_marker_detector think_dsml;
    bool dsml_in_think;
    bool dsml_in_think_reported;
    bool post_think_gap;
    bool tool_preflight_error;
    char tool_preflight_error_msg[256];
} agent_stream_renderer;

typedef struct {
    bool active;
    bool done;
} agent_edit_upto_forcer;

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} agent_input_buf;

struct agent_syntax {
    const char *name;
    const char *aliases;
    const char **keywords;
    const char *singleline_comments[3];
    const char *multiline_start;
    const char *multiline_end;
    unsigned flags;
};

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
    bool truncated;
} agent_buf;

typedef struct {
    bool has_title_trailer;
    bool legacy_identity;
    char *title;
    uint64_t created_at;
    char sha[41];
} agent_kv_session_meta;

typedef enum {
    AGENT_HISTORY_MARK_NONE,
    AGENT_HISTORY_MARK_USER,
    AGENT_HISTORY_MARK_ASSISTANT,
    AGENT_HISTORY_MARK_EOS,
} agent_history_mark;

typedef struct {
    const char **v;
    agent_history_mark *mark;
    int len;
    int cap;
} agent_history_ptrs;

typedef struct {
    pulsar_kvstore_entry entry;
    char *title;
} agent_session_list_item;

typedef struct {
    char sha[41];
    uint64_t last_used;
} agent_completion_session;

typedef struct {
    agent_completion_session *v;
    int len;
    int cap;
} agent_completion_sessions;

typedef struct {
    size_t start;
    size_t content_end;
    size_t end;
} agent_line_span;

typedef struct {
    agent_line_span *v;
    int len;
    int cap;
} agent_line_spans;

typedef struct {
    const char *query;
    const char *glob;
    regex_t regex;
    bool use_regex;
    bool regex_ready;
    bool case_sensitive;
    int context;
    int max_results;
    int results;
    agent_buf out;
} agent_search_ctx;

struct agent_bash_job {
    int id;
    pid_t pid;
    int pipe_fd;
    int tmp_fd;
    /* Always the mkstemp template "/tmp/pulsar_agent_output_XXXXXX" (27 chars +
     * NUL) from agent_bash_start — mkstemp only substitutes the X's, so it can
     * never lengthen.  Sized for exactly that, not PATH_MAX. */
    char path[32];
    char *cmd;
    double start_time;
    double timeout_sec;
    size_t bytes;
    int newline_count;
    char last_byte;
    size_t observed_bytes;
    int observed_display_lines;
    bool observed_once;
    int exit_status;
    bool running;
    bool timed_out;
    struct agent_bash_job *next;
    agent_worker *worker;  /* back-pointer for terminal state restoration */

    /* ---- methods (C++ port): 1:1 mirror of the agent_bash_* verb family;
     * bodies keep the auto *job = this alias, logic verbatim. ---- */
    int display_lines() const;
    void note_output(const char *s, size_t n);
    void job_free();
    void drain();
    void finalize(int status);
    void poll();
    char *read_head(int max_lines, size_t max_bytes, int *lines_read, bool *byte_limited) const;
    char *read_tail_lines(int max_lines) const;
    char *observation(bool mark_observed);
};

typedef struct {
    char **v;
    size_t len;
    size_t cap;
} agent_prompt_queue;

typedef struct {
    struct linenoiseState edit;
    char *input;
    char prompt[160];
    char status[4096];
    int old_stdin_flags;
    bool active;
    bool hidden;
    bool output_line_open;
    bool prompt_below_output;
    int output_col;
    bool scroll_region;
    int term_rows;
    int term_cols;
    int output_bottom;
    int prompt_row;
    int reserved_rows;
    bool output_cursor_saved;
    bool output_at_scroll_boundary;
    double last_prompt_redraw_time;
    char cpr_buf[32];
    size_t cpr_len;
    bool paste_open;
    bool paste_start_pending;
    char paste_tail[6];
    size_t paste_tail_len;
} agent_editor;

typedef enum {
    CPR_INVALID,
    CPR_PARTIAL,
    CPR_COMPLETE,
} cpr_state;

typedef enum {
    AGENT_YES_NO_AUTO_NONE,
    AGENT_YES_NO_AUTO_NO,
    AGENT_YES_NO_AUTO_YES,
} agent_yes_no_auto;

typedef struct {
    int timeout_sec;
    agent_yes_no_auto timeout_answer;
} agent_yes_no_options;

typedef enum {
    AGENT_EXIT_CANCEL,
    AGENT_EXIT_CLEAN,
    AGENT_EXIT_NOW,
} agent_exit_save_result;

/* ---- shared globals ---- */

extern volatile sig_atomic_t agent_sigint;
extern agent_worker *agent_completion_worker;
extern const char agent_dsml_syntax_reminder[];

/* ---- shared functions ---- */

void agent_sigint_handler(int sig);
void *agent_xmalloc(size_t n);
char *xstrdup(const char *s);
char *xstrndup(const char *s, size_t n);
void *agent_xrealloc(void *ptr, size_t n);
void write_all(int fd, const char *p, size_t n);
void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n);
char *agent_input_buf_take(agent_input_buf *b);
void agent_input_buf_free(agent_input_buf *b);
bool agent_slash_command_known(const char *cmd);
double agent_now_sec(void);
void usage(FILE *fp, const char *topic);
agent_config parse_options(int argc, char **argv);
void log_context_memory(pulsar_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk);
pulsar_think_mode effective_think_mode(const agent_config *cfg);
void agent_append_system_prompt(pulsar_engine *engine, pulsar_tokens *tokens,
                                       const char *extra);
void agent_worker_note_system_prompt_seen(agent_worker *w);
void agent_worker_maybe_append_datetime_context(agent_worker *w);
void agent_worker_maybe_append_system_prompt_reminder(agent_worker *w);
void agent_wake_locked(agent_worker *w);
void agent_publish(agent_worker *w, const char *s, size_t n);
void agent_publishf(agent_worker *w, const char *fmt, ...);
void agent_set_status(agent_worker *w, agent_worker_state state);
void agent_set_error(agent_worker *w, const char *msg);
void agent_trace(agent_worker *w, const char *fmt, ...);
void agent_trace_token(agent_worker *w, int token, const char *text,
                              size_t text_len, int index);
void agent_trace_tokens(agent_worker *w, const char *label,
                               const pulsar_tokens *tokens, int start);
void agent_trace_text(agent_worker *w, const char *label,
                             const char *text, size_t len);
bool bytes_has_prefix(const char *p, size_t n, const char *prefix);
bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix);
const char *agent_tool_arg_value(const agent_tool_call *call, const char *name);
void agent_dsml_parser_free(agent_dsml_parser *p);
void agent_dsml_parser_reset(agent_dsml_parser *p);
void agent_dsml_set_error(agent_dsml_parser *p, const char *msg);
bool agent_dsml_parameter_close_tail(const char *tail, size_t len,
                                            bool *complete);
void agent_dsml_start(agent_dsml_parser *p);
void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n);
char *agent_tail_capture_take(agent_tail_capture *t, size_t *len);
void renderer_write(agent_token_renderer *r, const char *s, size_t n);
void renderer_reset_color(agent_token_renderer *r);
void renderer_restore_text_attrs(agent_token_renderer *r);
const agent_syntax *agent_syntax_for_path(const char *path);
int renderer_terminal_cols(void);
void renderer_code_byte(agent_token_renderer *r, char c);
void renderer_code_stream_begin(agent_token_renderer *r,
                                       const agent_syntax *syntax);
void renderer_code_stream_set_prefix(agent_token_renderer *r,
                                            const char *prefix,
                                            const char *color);
void renderer_code_stream_set_upto_marker(agent_token_renderer *r,
                                                 bool enabled);
void renderer_code_end(agent_token_renderer *r);
void renderer_write_char(agent_token_renderer *r, char c);
void renderer_finish(agent_token_renderer *r);
void renderer_color(agent_token_renderer *r, const char *seq);
void renderer_plain(agent_token_renderer *r, const char *s, size_t n);
void agent_stream_text(agent_stream_renderer *sr, const char *text, size_t len, bool finish);
void worker_progress_cb(void *ud, const char *event, int current, int total);
bool worker_should_interrupt(agent_worker *w);
void worker_clear_interrupt(agent_worker *w);
bool agent_err_is_interrupted(const char *err);
bool worker_cancel_session_cb(void *ud);
void agent_buf_append(agent_buf *b, const char *s, size_t n);
void agent_buf_puts(agent_buf *b, const char *s);
char *agent_buf_take(agent_buf *b);
bool agent_tokens_equal(const pulsar_tokens *a, const pulsar_tokens *b);
bool agent_mkdir_p(const char *path);
char *agent_default_cache_dir(void);
char *agent_kv_path_for_sha(const char *dir, const char sha[41]);
void agent_session_identity_sha(const char *title, uint64_t created_at,
                                       char sha_out[41]);
void agent_worker_clear_session_identity(agent_worker *w);
void agent_kv_session_meta_free(agent_kv_session_meta *m);
bool agent_kv_read_text(FILE *fp, uint32_t text_bytes,
                               char **text_out, char *err, size_t err_len);
bool agent_kv_write_title_trailer(FILE *fp, const char *title,
                                         char *err, size_t err_len);
bool agent_kv_read_title_trailer(FILE *fp, const pulsar_kvstore_entry *hdr,
                                        char **title_out,
                                        char *err, size_t err_len);
void agent_kv_identity_sha(const pulsar_kvstore_entry *hdr,
                                  const char *text, uint32_t text_bytes,
                                  const char *title,
                                  char sha_out[41]);
bool agent_kv_load_path(agent_worker *w, const char *path,
                               const char *expected_sha,
                               const char *expected_text,
                               size_t expected_text_len,
                               pulsar_tokens *loaded_tokens,
                               agent_kv_session_meta *meta_out,
                               char *err, size_t err_len);
void agent_worker_build_system_tokens(agent_worker *w, pulsar_tokens *out);
void agent_publish_system_status(agent_worker *w, const char *msg);
void agent_publishf_system_status(agent_worker *w, const char *fmt, ...);
char *worker_request_queued_user_drain(agent_worker *w);
bool worker_take_queued_user_drain_request(agent_worker *w);
void worker_answer_queued_user_drain(agent_worker *w, char *text);
int agent_worker_sync_tokens(agent_worker *w, const pulsar_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len);
bool agent_worker_reset_to_sysprompt(agent_worker *w, char *err, size_t err_len);
bool agent_worker_has_user_session(agent_worker *w);
bool agent_worker_needs_save(agent_worker *w);
bool agent_worker_save_session_now(agent_worker *w, char sha_out[41],
                                          int *tokens_out,
                                          char *err, size_t err_len);
bool agent_worker_save_session(agent_worker *w, char *err, size_t err_len);
char *agent_session_title_from_prompt(const char *prompt,
                                             size_t max_bytes);
char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes);
bool agent_worker_show_history(agent_worker *w, int user_turns,
                                      char *err, size_t err_len);
void agent_worker_list_sessions(agent_worker *w);
void agent_switch_completion_callback(const char *buf,
                                             linenoiseCompletions *lc);
bool agent_worker_delete_session(agent_worker *w, const char *prefix,
                                        char sha_out[41],
                                        char *err, size_t err_len);
bool agent_worker_strip_session(agent_worker *w, const char *prefix,
                                       char sha_out[41],
                                       uint32_t *tokens_out,
                                       char *err, size_t err_len);
bool agent_worker_switch_session(agent_worker *w, const char *prefix,
                                        int history_turns,
                                        char *err, size_t err_len);
int agent_parse_timeout(const char *s);
int agent_parse_int_default(const char *s, int def, int min, int max);
bool agent_parse_bool_default(const char *s, bool def);
void agent_line_spans_free(agent_line_spans *spans);
void agent_split_lines(const char *data, size_t len, agent_line_spans *spans);
int agent_read_file_bytes(const char *path, char **data, size_t *len,
                                 char *err, size_t errlen);
bool agent_old_new_line_effect(const char *old_data, size_t old_len,
                                      const char *new_data, size_t new_len,
                                      size_t edit_offset, size_t replaced_len,
                                      int *start_line, int *end_line,
                                      int *delta);
char *agent_edit_result(const char *path,
                                       int start_line, int end_line, int delta,
                                       const char *new_data, size_t new_len,
                                       const char *kind);
bool agent_tool_result_fits_context(agent_worker *w, const char *result,
                                           int reserve_tokens,
                                           int *tokens_out);
char *agent_tool_read(agent_worker *w, const agent_tool_call *call);
char *agent_tool_more(agent_worker *w, const agent_tool_call *call);
char *agent_tool_write(agent_worker *w, const agent_tool_call *call);
char *agent_tool_list(const agent_tool_call *call);
void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end);
bool agent_edit_upto_forcer_should_replace(agent_edit_upto_forcer *forcer,
                                                  agent_dsml_parser *p,
                                                  const char *next_text,
                                                  size_t next_len);
bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
                                     char *err, size_t err_len);
char *agent_tool_edit(agent_worker *w, const agent_tool_call *call);
char *agent_tool_search(agent_worker *w, const agent_tool_call *call);
void agent_bash_jobs_free(agent_worker *w);
agent_bash_job *agent_bash_find_job(agent_worker *w, int id, pid_t pid);
void agent_bash_remove_job(agent_worker *w, agent_bash_job *target);
agent_bash_job *agent_bash_start(agent_worker *w, const char *cmd,
                                        int timeout_sec, char *err, size_t err_len);
char *agent_bash_job_tool_result(agent_worker *w, agent_bash_job *job,
                                        bool wait, int refresh_sec,
                                        bool stop, bool remove_if_done);
int agent_tool_job_id(const agent_tool_call *call);
pid_t agent_tool_pid(const agent_tool_call *call);
char *agent_execute_tool_calls(agent_worker *w, const agent_tool_calls *calls);
char *agent_bash_jobs_compaction_observation(agent_worker *w);
bool agent_worker_compact(agent_worker *w, const char *reason,
                                 char *err, size_t err_len);
bool agent_worker_compact_if_needed(agent_worker *w, const char *reason,
                                           char *err, size_t err_len);
int worker_accept_generated_token(agent_worker *w,
                                         int token,
                                         int *generated,
                                         double t0,
                                         agent_stream_renderer *stream,
                                         char *err,
                                         size_t err_len);
int worker_force_generated_text(agent_worker *w,
                                       const char *text,
                                       int max_tokens,
                                       int *generated,
                                       double t0,
                                       agent_stream_renderer *stream,
                                       char *err,
                                       size_t err_len);
void worker_request_save(agent_worker *w);
void worker_request_compact(agent_worker *w);
void *worker_main(void *arg);
int set_nonblock(int fd, bool on, int *old_flags);
bool worker_check_raw_mode_restore(agent_worker *w);
void drain_wake_fd(int fd);
bool worker_submit(agent_worker *w, const char *text);
void worker_interrupt(agent_worker *w);
void worker_stop(agent_worker *w);
void worker_consume(agent_worker *w, char **out, size_t *out_len, agent_status *status);
void worker_get_status(agent_worker *w, agent_status *status);
bool worker_is_idle(agent_worker *w);
bool worker_is_initialized(agent_worker *w, agent_status *status);
bool stdout_is_tty(void);
char *agent_format_user_prompt_echo(const char *text);
void agent_echo_user_prompt(const char *text);
void build_prompt_text(const agent_status *st, char *buf, size_t len);
unsigned agent_next_prefill_label(void);
void agent_prompt_queue_push(agent_prompt_queue *q, const char *text);
char *agent_prompt_queue_pop(agent_prompt_queue *q);
void agent_prompt_queue_push_front(agent_prompt_queue *q, char *text);
char *agent_prompt_queue_take_all(agent_prompt_queue *q);
char *agent_prompt_queue_take_all_echo(agent_prompt_queue *q);
void agent_prompt_queue_free(agent_prompt_queue *q);
void build_footer_text(const agent_status *st, const agent_prompt_queue *queue,
                              int cols, char *buf, size_t len);
void editor_read_stdin(agent_editor *ed);
bool editor_take_queued_byte(agent_editor *ed, unsigned char byte);
bool editor_take_bare_escape(agent_editor *ed);
void editor_replace_input(agent_editor *ed, const char *text);
void editor_restore_terminal_layout(agent_editor *ed);
int editor_start(agent_editor *ed, const char *prompt,
                        const char *status, const char *initial);
void editor_stop(agent_editor *ed);
void editor_show(agent_editor *ed);
void editor_set_prompt_status(agent_editor *ed, const char *prompt,
                                     const char *status);
void editor_write_async(agent_editor *ed, const char *text, size_t len,
                               const char *prompt, const char *status,
                               bool force_show);
void editor_cancel_input_with_hint(agent_editor *ed,
                                          const char *prompt,
                                          const char *status);
void runtime_help(void);
void editor_write_welcome_banner(agent_editor *editor,
                                        const agent_config *cfg,
                                        const char *prompt,
                                        const char *statusline);
int agent_worker_init(agent_worker *w, pulsar_engine *engine, agent_config *cfg);
void agent_worker_free(agent_worker *w);
bool agent_prompt_yes_no_ex(const char *prompt,
                                   const agent_yes_no_options *opts,
                                   bool *timed_out);
bool agent_maybe_save_before_leaving_session(agent_worker *w);
agent_exit_save_result agent_maybe_save_before_exiting(agent_worker *w);

/* ---- shared inline helpers ---- */


#endif /* PULSAR_AGENT_INTERNAL_H */
