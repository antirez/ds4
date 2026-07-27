#include "pulsar_agent_internal.h"



/* ============================================================================
 * Streaming Tool Visualization
 * ============================================================================
 *
 * Tool calls are parsed for execution later, but they are also visualized while
 * the model is still sampling.  This state machine suppresses raw DSML and
 * prints compact, tool-specific progress such as "$ command" or
 * "Reading file 1:500...".
 */

static bool streq_any(const char *s, const char *a, const char *b,
                      const char *c, const char *d) {
    return (a && !strcmp(s, a)) || (b && !strcmp(s, b)) ||
           (c && !strcmp(s, c)) || (d && !strcmp(s, d));
}



static agent_tool_param_kind agent_tool_param_kind_for(const char *tool, const char *param) {
    if (!tool) tool = "";
    if (!param) param = "";
    if (!strcmp(tool, "bash") && !strcmp(param, "command"))
        return AGENT_TOOL_PARAM_BASH_COMMAND;
    if (!strcmp(tool, "edit") && !strcmp(param, "old"))
        return AGENT_TOOL_PARAM_DIFF_OLD;
    if (!strcmp(tool, "edit") && !strcmp(param, "new"))
        return AGENT_TOOL_PARAM_DIFF_NEW;
    if (streq_any(param, "path", "file", "filename", NULL))
        return AGENT_TOOL_PARAM_PATH;
    if (streq_any(param, "line", "start_line", "end_line", "offset") ||
        streq_any(param, "start", "end", "count", "max_lines") ||
        streq_any(param, "timeout_sec", "refresh_sec", NULL, NULL))
        return AGENT_TOOL_PARAM_OFFSET;
    if (streq_any(param, "content", "text", NULL, NULL))
        return AGENT_TOOL_PARAM_CONTENT;
    return AGENT_TOOL_PARAM_NORMAL;
}



static const char *agent_tool_param_color(agent_tool_param_kind kind) {
    switch (kind) {
    case AGENT_TOOL_PARAM_PATH: return "\x1b[32m";
    case AGENT_TOOL_PARAM_OFFSET: return "\x1b[33m";
    case AGENT_TOOL_PARAM_CONTENT: return "\x1b[34m";
    case AGENT_TOOL_PARAM_DIFF_OLD: return "\x1b[31m";
    case AGENT_TOOL_PARAM_DIFF_NEW: return "\x1b[32m";
    case AGENT_TOOL_PARAM_BASH_COMMAND: return "\x1b[1;36m";
    default: return "\x1b[37m";
    }
}



static void agent_tool_viz_write(agent_stream_renderer *sr, const char *s, size_t n) {
    renderer_plain(sr->renderer, s, n);
    for (size_t i = 0; i < n; i++) sr->viz.last_output_newline = s[i] == '\n';
}



static void agent_tool_viz_puts(agent_stream_renderer *sr, const char *s) {
    agent_tool_viz_write(sr, s, strlen(s));
}



static void agent_tool_viz_start(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    bool line_open = !sr->renderer->last_output_newline;
    memset(v, 0, sizeof(*v));
    v->active = true;
    v->at_line_start = true;
    v->last_output_newline = true;
    if (sr->replay) {
        if (line_open) agent_tool_viz_puts(sr, "\n");
    } else if (sr->renderer->use_color) {
        /* The raw DSML start marker may arrive after ordinary text on the
         * current row.  Clear that row only for the live terminal UI; plain
         * stdout mode must never leak cursor-control escapes into pipes. */
        agent_tool_viz_puts(sr, "\r\x1b[2K");
    } else if (line_open) {
        agent_tool_viz_puts(sr, "\n");
    }
    v->last_output_newline = true;
}



static void agent_tool_viz_line_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    agent_tool_viz_puts(sr, "🛠️ ");
    v->at_line_start = false;
}



static const char *agent_tool_viz_prefix(const char *name) {
    if (!strcmp(name, "bash")) return "$ ";
    if (!strcmp(name, "read")) return "read ";
    if (!strcmp(name, "write")) return "write ";
    if (!strcmp(name, "edit")) return "edit ";
    if (!strcmp(name, "search")) return "search ";
    return NULL;
}



static void agent_tool_viz_tool(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->tool_announced && !strcmp(v->tool_name, name)) return;
    if (v->tool_announced && !v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    snprintf(v->tool_name, sizeof(v->tool_name), "%s", name ? name : "tool");
    v->tool_announced = true;
    v->read_style = !strcmp(v->tool_name, "read");
    agent_tool_viz_line_prefix(sr);
    if (v->read_style) {
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        v->read_prefix_rendered = true;
        return;
    }
    renderer_color(sr->renderer, !strcmp(v->tool_name, "bash") ?
                                "\x1b[1;36m" : "\x1b[1;37m");
    const char *prefix = agent_tool_viz_prefix(v->tool_name);
    if (prefix) {
        agent_tool_viz_puts(sr, prefix);
    } else {
        agent_tool_viz_puts(sr, v->tool_name);
        agent_tool_viz_puts(sr, " ");
    }
    renderer_color(sr->renderer, "\x1b[0m");
}



static void agent_tool_viz_append(char *dst, size_t cap, char c) {
    size_t len = strlen(dst);
    if (len + 1 >= cap) return;
    dst[len] = c;
    dst[len + 1] = '\0';
}



static void agent_tool_viz_read_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (!strcmp(v->param_name, "path")) {
        agent_tool_viz_append(v->read_path, sizeof(v->read_path), c);
        if (v->read_prefix_rendered) agent_tool_viz_write(sr, &c, 1);
    } else if (!strcmp(v->param_name, "start_line")) {
        agent_tool_viz_append(v->read_start, sizeof(v->read_start), c);
    } else if (!strcmp(v->param_name, "max_lines")) {
        agent_tool_viz_append(v->read_max, sizeof(v->read_max), c);
    } else if (!strcmp(v->param_name, "whole")) {
        agent_tool_viz_append(v->read_whole, sizeof(v->read_whole), c);
    }
}



static void agent_tool_viz_render_read(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->read_style || v->read_line_rendered) return;

    if (!v->read_prefix_rendered) {
        agent_tool_viz_line_prefix(sr);
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        agent_tool_viz_puts(sr, v->read_path[0] ? v->read_path : "<unknown>");
    } else if (!v->read_path[0]) {
        renderer_color(sr->renderer, "\x1b[32m");
        agent_tool_viz_puts(sr, "<unknown>");
    }
    renderer_color(sr->renderer, "\x1b[33m");
    bool whole = agent_parse_bool_default(v->read_whole, false);
    if (whole && (!v->read_start[0] || !strcmp(v->read_start, "1"))) {
        agent_tool_viz_puts(sr, " (whole file)");
    } else if (whole) {
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start);
        agent_tool_viz_puts(sr, ":EOF");
    } else {
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start[0] ? v->read_start : "1");
        agent_tool_viz_puts(sr, ":");
        agent_tool_viz_puts(sr, v->read_max[0] ? v->read_max : "500");
    }
    renderer_color(sr->renderer, "\x1b[1;37m");
    agent_tool_viz_puts(sr, "...");
    renderer_color(sr->renderer, "\x1b[0m");
    agent_tool_viz_puts(sr, "\n");
    v->read_line_rendered = true;
}



static bool agent_tool_viz_param_is_code_body(agent_tool_visualizer *v) {
    if (!strcmp(v->tool_name, "write") &&
        v->param_kind == AGENT_TOOL_PARAM_CONTENT)
        return true;
    if (!strcmp(v->tool_name, "edit") &&
        (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
         v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW ||
         v->param_kind == AGENT_TOOL_PARAM_CONTENT))
        return true;
    return false;
}



static const char *agent_tool_viz_diff_prefix(agent_tool_param_kind kind,
                                              const char **color) {
    if (color) *color = NULL;
    const char *prefix = NULL;
    if (kind == AGENT_TOOL_PARAM_DIFF_OLD) {
        prefix = "- ";
        if (color) *color = "\x1b[31m";
    } else if (kind == AGENT_TOOL_PARAM_DIFF_NEW) {
        prefix = "+ ";
        if (color) *color = "\x1b[32m";
    }
    return prefix;
}



static void agent_tool_viz_code_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->at_line_start) return;
    const char *color = NULL;
    const char *prefix = agent_tool_viz_diff_prefix(v->param_kind, &color);
    if (!prefix) return;
    renderer_color(sr->renderer, color);
    renderer_write(sr->renderer, prefix, strlen(prefix));
    renderer_color(sr->renderer, "\x1b[0m");
    sr->renderer->wrote_visible_output = true;
    sr->renderer->last_output_newline = false;
    v->last_output_newline = false;
    v->at_line_start = false;
}



static void agent_tool_viz_code_begin(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    const agent_syntax *syntax = agent_syntax_for_path(v->tool_path);
    renderer_code_stream_begin(sr->renderer, syntax);
    renderer_code_stream_set_upto_marker(sr->renderer,
        !strcmp(v->tool_name, "edit") &&
        v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD);
    v->code_param_active = true;
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        const char *color = NULL;
        const char *prefix = agent_tool_viz_diff_prefix(v->param_kind, &color);
        /* Diff prefixes are terminal UI, not code.  Keep them outside the
         * syntax buffer so a later row repaint preserves their red/green color
         * while highlighting only the actual edited line. */
        renderer_code_stream_set_prefix(sr->renderer, prefix, color);
        agent_tool_viz_code_prefix(sr);
    }
}



static void agent_tool_viz_code_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->code_param_active) return;
    renderer_code_end(sr->renderer);
    v->code_param_active = false;
    v->at_line_start = true;
    v->last_output_newline = sr->renderer->last_output_newline;
}



static void agent_tool_viz_code_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    agent_tool_viz_code_prefix(sr);
    renderer_code_byte(sr->renderer, c);
    v->last_output_newline = c == '\n';
    v->at_line_start = c == '\n';
}



static void agent_tool_viz_param_begin(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && sr->parser->current.name)
        agent_tool_viz_tool(sr, sr->parser->current.name);
    snprintf(v->param_name, sizeof(v->param_name), "%s", name ? name : "");
    v->param_kind = agent_tool_param_kind_for(v->tool_name, v->param_name);
    v->param_active = true;
    v->param_end_len = 0;

    if (v->read_style) return;

    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        v->at_line_start = true;
        agent_tool_viz_code_begin(sr);
        return;
    }

    if (v->param_kind == AGENT_TOOL_PARAM_CONTENT) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        if (strcmp(v->tool_name, "write")) {
            renderer_color(sr->renderer, "\x1b[1;37m");
            agent_tool_viz_puts(sr, v->param_name);
            agent_tool_viz_puts(sr, ":\n");
        }
        v->at_line_start = true;
        if (agent_tool_viz_param_is_code_body(v)) {
            agent_tool_viz_code_begin(sr);
        } else {
            renderer_color(sr->renderer, "\x1b[34m");
        }
        return;
    }

    if (v->param_kind != AGENT_TOOL_PARAM_BASH_COMMAND) {
        if (!v->at_line_start) agent_tool_viz_puts(sr, " ");
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, v->param_name);
        agent_tool_viz_puts(sr, "=");
    } else {
        renderer_color(sr->renderer, agent_tool_param_color(AGENT_TOOL_PARAM_BASH_COMMAND));
        return;
    }
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}



static void agent_tool_viz_param_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    v->param_end_len = 0;
    if (v->code_param_active) agent_tool_viz_code_end(sr);
    if (!v->read_style) renderer_color(sr->renderer, "\x1b[0m");
    v->param_active = false;
    v->param_name[0] = '\0';
}



static void agent_tool_viz_param_raw_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->read_style) {
        agent_tool_viz_read_value_byte(sr, c);
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_PATH) {
        agent_tool_viz_append(v->tool_path, sizeof(v->tool_path), c);
    }
    if (v->code_param_active) {
        agent_tool_viz_code_byte(sr, c);
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_BASH_COMMAND) {
        agent_tool_viz_write(sr, &c, 1);
        v->at_line_start = c == '\n';
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        agent_tool_viz_code_begin(sr);
        agent_tool_viz_code_byte(sr, c);
        return;
    }
    agent_tool_viz_write(sr, &c, 1);
    v->at_line_start = c == '\n';
}



static void agent_tool_viz_restore_param_color(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active || !v->param_active || v->read_style) return;
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}



/* Stream one DSML parameter byte into the visualizer.  The visualizer must not
 * wait for the whole parameter: large write/edit contents should show progress
 * as the model emits them, while still detecting the closing parameter tag. */
static void agent_tool_viz_param_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;

    if (v->param_end_len || c == '<') {
        if (v->param_end_len == sizeof(v->param_end_tail)) {
            size_t keep = v->param_end_len;
            v->param_end_len = 0;
            for (size_t i = 0; i < keep; i++)
                agent_tool_viz_param_raw_byte(sr, v->param_end_tail[i]);
            if (c != '<') {
                agent_tool_viz_param_raw_byte(sr, c);
                return;
            }
        }
        if (v->param_end_len < sizeof(v->param_end_tail))
            v->param_end_tail[v->param_end_len++] = c;
        bool complete = false;
        if (agent_dsml_parameter_close_tail(v->param_end_tail, v->param_end_len, &complete)) {
            if (complete) agent_tool_viz_param_end(sr);
            return;
        }
        size_t keep = v->param_end_len;
        v->param_end_len = 0;
        for (size_t i = 0; i < keep; i++)
            agent_tool_viz_param_raw_byte(sr, v->param_end_tail[i]);
        return;
    }
    agent_tool_viz_param_raw_byte(sr, c);
}



static void agent_tool_viz_finish(agent_stream_renderer *sr, const char *status) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;
    if (v->param_active) agent_tool_viz_param_end(sr);
    if (!status || !status[0]) agent_tool_viz_render_read(sr);
    if (status && status[0]) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        renderer_color(sr->renderer, "\x1b[90m");
        agent_tool_viz_puts(sr, status);
        renderer_color(sr->renderer, "\x1b[0m");
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    v->active = false;
}



static void agent_tool_viz_dump_invalid_dsml(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;

    /* The normal path hides DSML and paints a friendly semantic projection.  If
     * parsing fails, show the exact bytes we rejected so the next fix is based
     * on evidence instead of guessing from the projection. */
    if (v->param_active) {
        v->param_active = false;
        v->param_end_len = 0;
        v->param_name[0] = '\0';
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    renderer_color(sr->renderer, "\x1b[1;31m");
    if (sr->parser->raw && sr->parser->raw_len) {
        agent_tool_viz_write(sr, sr->parser->raw, sr->parser->raw_len);
    } else {
        agent_tool_viz_puts(sr, "<empty DSML>");
    }
    renderer_color(sr->renderer, "\x1b[0m");
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
}



static void agent_stream_finish_ignored_dsml(agent_stream_renderer *sr, const char *detail) {
    const char *msg =
        detail && detail[0] ? detail :
        "tool calling is not allowed inside <think></think>";
    sr->dsml_in_think = true;
    sr->dsml_in_think_reported = true;
    agent_trace(sr->renderer->worker, "dsml ignored inside thinking: %s", msg);
    if (!sr->renderer->last_output_newline)
        renderer_plain(sr->renderer, "\n", 1);
    renderer_color(sr->renderer, "\x1b[1;31m");
    renderer_plain(sr->renderer, "[tool call ignored: ", 20);
    renderer_plain(sr->renderer, msg, strlen(msg));
    renderer_plain(sr->renderer, "]\n", 2);
    renderer_color(sr->renderer, "\x1b[0m");
    agent_dsml_parser_reset(sr->parser);
    sr->dsml_active = false;
    sr->dsml_ignored = false;
}



static void agent_stream_malformed_dsml(agent_stream_renderer *sr,
                                        const char *detail) {
    const char *msg = detail && detail[0] ? detail :
        "DSML markup outside a valid tool_calls block";
    if (sr->parser->state == AGENT_DSML_ERROR) return;
    agent_dsml_set_error(sr->parser, msg);
    agent_trace(sr->renderer->worker, "malformed dsml in assistant output: %s", msg);
    if (!sr->renderer->last_output_newline)
        renderer_plain(sr->renderer, "\n", 1);
    renderer_color(sr->renderer, "\x1b[1;31m");
    renderer_plain(sr->renderer, "[invalid tool call: ", 20);
    renderer_plain(sr->renderer, msg, strlen(msg));
    renderer_plain(sr->renderer, "]\n", 2);
    renderer_color(sr->renderer, "\x1b[0m");
}



/* Mirror parser progress into the terminal visualizer.  Parser state is the
 * source of truth; this function only decides what the user should see. */
static void agent_stream_tool_events(agent_stream_renderer *sr) {
    agent_dsml_parser *p = sr->parser;
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && p->current.name)
        agent_tool_viz_tool(sr, p->current.name);
    if (v->tool_announced && !p->current.name && !v->param_active) {
        agent_tool_viz_render_read(sr);
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        v->read_style = false;
        v->read_prefix_rendered = false;
        v->read_line_rendered = false;
        v->read_path[0] = '\0';
        v->read_start[0] = '\0';
        v->read_max[0] = '\0';
        v->read_whole[0] = '\0';
        v->tool_announced = false;
    }
    if (!v->param_active && p->state == AGENT_DSML_PARAM_VALUE && p->param_name)
        agent_tool_viz_param_begin(sr, p->param_name);
}



static void agent_stream_preflight_closed_param(agent_stream_renderer *sr) {
    if (!sr || sr->replay || sr->dsml_ignored || sr->tool_preflight_error)
        return;
    agent_dsml_parser *p = sr->parser;
    agent_tool_visualizer *v = &sr->viz;
    if (!p || !v->param_active || strcmp(v->param_name, "old") != 0)
        return;
    if (!p->current.name || strcmp(p->current.name, "edit") != 0)
        return;

    char err[256] = {0};
    if (agent_preflight_edit_old(sr->renderer->worker, &p->current,
                                 err, sizeof(err)))
        return;

    sr->tool_preflight_error = true;
    snprintf(sr->tool_preflight_error_msg, sizeof(sr->tool_preflight_error_msg),
             "edit old selector failed before new was generated: %s",
             err[0] ? err : "old text is not a unique match");
    agent_trace(sr->renderer->worker, "edit old preflight failed: %s",
                sr->tool_preflight_error_msg);
}



static void agent_stream_feed_dsml_byte(agent_stream_renderer *sr, char c) {
    bool was_param = !sr->dsml_ignored && sr->viz.param_active;
    agent_dsml_feed(sr->parser, &c, 1);
    if (!sr->dsml_ignored) {
        agent_stream_tool_events(sr);
        if (was_param) agent_tool_viz_param_value_byte(sr, c);
        if (was_param && sr->parser->state != AGENT_DSML_PARAM_VALUE &&
            sr->viz.param_active)
        {
            agent_stream_preflight_closed_param(sr);
            agent_tool_viz_param_end(sr);
        }
    }
    if (sr->parser->state == AGENT_DSML_DONE) {
        if (sr->dsml_ignored) {
            agent_stream_finish_ignored_dsml(
                sr, "tool calling is not allowed inside <think></think>");
        } else {
            agent_trace(sr->renderer->worker, "dsml done calls=%d",
                        sr->parser->calls.len);
            agent_tool_viz_finish(sr, NULL);
            sr->dsml_active = false;
        }
    } else if (sr->parser->state == AGENT_DSML_ERROR) {
        if (sr->dsml_ignored) {
            agent_stream_finish_ignored_dsml(
                sr, "malformed tool call inside <think></think>");
        } else {
            char status[220];
            snprintf(status, sizeof(status), "[invalid tool call: %s]\n",
                     sr->parser->error[0] ? sr->parser->error : "parse error");
            agent_trace(sr->renderer->worker, "dsml error %s",
                        sr->parser->error[0] ? sr->parser->error : "parse error");
            agent_tool_viz_dump_invalid_dsml(sr);
            agent_tool_viz_finish(sr, status);
            sr->dsml_active = false;
        }
    }
}



/* Start a DSML block from the streaming detector.  The detector may accept a
 * known malformed opening form for robustness, but the parser is seeded with
 * canonical bytes so all later parsing remains strict. */
static void agent_stream_start_dsml(agent_stream_renderer *sr, bool ignored) {
    sr->dsml_active = true;
    sr->dsml_ignored = ignored;
    if (ignored) sr->dsml_in_think = true;
    sr->dsml_start_len = 0;
    sr->post_think_gap = false;
    agent_trace(sr->renderer->worker, "dsml start detected%s",
                ignored ? " inside thinking" : "");
    agent_dsml_start(sr->parser);
    if (!ignored) {
        agent_tool_viz_start(sr);
        agent_stream_tool_events(sr);
    }
}



static void agent_stream_note_plain_dsml_byte(agent_stream_renderer *sr, char c);



static void agent_stream_flush_start_tail(agent_stream_renderer *sr) {
    if (!sr->dsml_start_len) return;
    sr->post_think_gap = false;
    for (size_t i = 0; i < sr->dsml_start_len; i++) {
        renderer_write_char(sr->renderer, sr->dsml_start_tail[i]);
        agent_stream_note_plain_dsml_byte(sr, sr->dsml_start_tail[i]);
        if (sr->parser->state == AGENT_DSML_ERROR) break;
    }
    sr->dsml_start_len = 0;
}



static bool agent_stream_dsml_start_match(const char *tail, size_t len,
                                          bool *complete,
                                          bool *implicit_invoke) {
    static const char canonical[] = "<｜DSML｜tool_calls>";
    static const char missing_bar[] = "<DSML｜tool_calls>";
    static const char invoke[] = "<｜DSML｜invoke";
    static const char invoke_missing_bar[] = "<DSML｜invoke";
    struct {
        const char *text;
        bool implicit_invoke;
    } forms[] = {
        {canonical, false},
        {missing_bar, false},
        {invoke, true},
        {invoke_missing_bar, true},
    };
    *complete = false;
    *implicit_invoke = false;
    for (size_t i = 0; i < sizeof(forms)/sizeof(forms[0]); i++) {
        size_t form_len = strlen(forms[i].text);
        if (len <= form_len && memcmp(forms[i].text, tail, len) == 0) {
            *complete = len == form_len;
            *implicit_invoke = forms[i].implicit_invoke;
            return true;
        }
    }
    return false;
}



static bool agent_tail_matches(const char *tail, size_t len,
                               const char *needle, size_t needle_len) {
    return len >= needle_len &&
           memcmp(tail + len - needle_len, needle, needle_len) == 0;
}



/* Detect DSML-looking control markers in text that is not currently owned by
 * the executable DSML parser.  This helper intentionally has no policy: inside
 * <think> the marker means "tool call attempted too early", while in normal
 * assistant output it means malformed DSML that the model should see as a tool
 * error. */
static bool agent_dsml_marker_detector_feed(agent_dsml_marker_detector *d,
                                            char c) {
    if (d->len == sizeof(d->tail)) {
        memmove(d->tail, d->tail + 1, sizeof(d->tail) - 1);
        d->len--;
    }
    d->tail[d->len++] = c;

    static const char fullwidth_marker[] = "｜DSML｜";
    static const char ascii_marker[] = "|DSML|";
    static const char missing_open[] = "<DSML｜";
    static const char missing_close[] = "</DSML｜";
    return agent_tail_matches(d->tail, d->len,
                              fullwidth_marker, sizeof(fullwidth_marker) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              ascii_marker, sizeof(ascii_marker) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              missing_open, sizeof(missing_open) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              missing_close, sizeof(missing_close) - 1);
}



static void agent_stream_note_thinking_dsml_byte(agent_stream_renderer *sr,
                                                 char c) {
    if (!sr->in_think || sr->dsml_in_think) return;
    if (agent_dsml_marker_detector_feed(&sr->think_dsml, c))
        sr->dsml_in_think = true;
}



static void agent_stream_note_plain_dsml_byte(agent_stream_renderer *sr,
                                              char c) {
    if (sr->parser->state == AGENT_DSML_ERROR) return;
    if (sr->dsml_active || sr->in_think || sr->dsml_in_think) return;
    if (agent_dsml_marker_detector_feed(&sr->plain_dsml, c)) {
        agent_stream_malformed_dsml(
            sr, "DSML markup outside a valid tool_calls block");
    }
}



/* Route ordinary assistant bytes either to normal markdown rendering or into
 * the DSML detector.  The detector must hold short prefixes because the model
 * can split "<｜DSML｜tool_calls>" across arbitrary tokens. */
static void agent_stream_normal_byte(agent_stream_renderer *sr, char c) {
    static const char start[] = "<｜DSML｜tool_calls>";
    static const char canonical_invoke[] = "<｜DSML｜invoke";
    if (sr->parser->state == AGENT_DSML_ERROR) return;
    agent_stream_note_thinking_dsml_byte(sr, c);

    /* DeepSeek usually emits one or more blank lines after </think> before
     * either prose or a DSML tool stanza.  At that point the bytes are just a
     * visual gap between the hidden thinking phase and the real answer, and
     * printing them makes tool calls appear after odd empty lines.  We only
     * suppress whitespace in this very narrow post-thinking window; once the
     * first non-space byte arrives, normal rendering resumes. */
    if (sr->post_think_gap &&
        (c == ' ' || c == '\t' || c == '\r' || c == '\n'))
    {
        return;
    }

    if (sr->dsml_start_len || c == start[0]) {
        if (sr->dsml_start_len < sizeof(sr->dsml_start_tail))
            sr->dsml_start_tail[sr->dsml_start_len++] = c;
        bool complete = false, implicit_invoke = false;
        if (agent_stream_dsml_start_match(sr->dsml_start_tail, sr->dsml_start_len,
                                          &complete, &implicit_invoke))
        {
            if (complete) {
                /* Accept the common missing-leading-bar typo
                 * "<DSML｜tool_calls>" here, but seed the parser with the
                 * canonical marker so the rest of the DSML parser stays
                 * strict and simple.  Also accept a direct invoke opener as an
                 * implicit tool_calls block; the model often knows it wants a
                 * tool but forgets the outer wrapper. */
                agent_stream_start_dsml(sr, sr->in_think);
                if (implicit_invoke) {
                    for (size_t i = 0; i < sizeof(canonical_invoke) - 1; i++)
                        agent_stream_feed_dsml_byte(sr, canonical_invoke[i]);
                }
            }
            return;
        }
        if (sr->dsml_start_len > 1 &&
            sr->dsml_start_tail[sr->dsml_start_len - 1] == start[0])
        {
            sr->post_think_gap = false;
            size_t flush = sr->dsml_start_len - 1;
            for (size_t i = 0; i < flush; i++) {
                renderer_write_char(sr->renderer, sr->dsml_start_tail[i]);
                agent_stream_note_plain_dsml_byte(sr, sr->dsml_start_tail[i]);
                if (sr->parser->state == AGENT_DSML_ERROR) break;
            }
            if (sr->parser->state == AGENT_DSML_ERROR) {
                sr->dsml_start_len = 0;
                return;
            }
            sr->dsml_start_tail[0] = start[0];
            sr->dsml_start_len = 1;
            return;
        }
        agent_stream_flush_start_tail(sr);
        return;
    }

    sr->post_think_gap = false;
    renderer_write_char(sr->renderer, c);
    agent_stream_note_plain_dsml_byte(sr, c);
}



/* This is the single streaming display state machine for assistant output.  It
 * hides raw DSML as soon as the tool_calls marker is complete, lets the DSML
 * parser continue building executable calls, and paints semantic tool output
 * from parser state changes.  The sampled transcript remains unchanged: only
 * the terminal projection is rewritten. */
void agent_stream_text(agent_stream_renderer *sr, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = sr->pending_len + len;
    char *buf = (char *)agent_xmalloc(total ? total : 1);
    if (sr->pending_len) memcpy(buf, sr->pending, sr->pending_len);
    if (len) memcpy(buf + sr->pending_len, text, len);
    sr->pending_len = 0;

    /* The UI may reset terminal attributes while redrawing the editable prompt
     * between generated chunks.  If a DSML parameter is still streaming, make
     * each new token fragment self-contained by restoring the active parameter
     * color before visible bytes are projected.  This keeps the prompt normal
     * without sacrificing long write/edit content coloring. */
    if (len) agent_tool_viz_restore_param_color(sr);
    if (len && !sr->dsml_active) renderer_restore_text_attrs(sr->renderer);

    size_t i = 0;
    while (i < total) {
        char *cur = buf + i;
        size_t rem = total - i;
        if (!sr->dsml_active && bytes_has_prefix(cur, rem, think_open)) {
            agent_stream_flush_start_tail(sr);
            sr->post_think_gap = false;
            sr->in_think = true;
            sr->renderer->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (!sr->dsml_active && bytes_has_prefix(cur, rem, think_close)) {
            agent_stream_flush_start_tail(sr);
            sr->in_think = false;
            sr->renderer->in_think = false;
            renderer_reset_color(sr->renderer);
            if (!sr->renderer->last_output_newline)
                renderer_write(sr->renderer, "\n", 1);
            renderer_write(sr->renderer, "\n", 1);
            sr->renderer->last_output_newline = true;
            sr->post_think_gap = true;
            i += strlen(think_close);
            continue;
        }
        if (!finish && !sr->dsml_active && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(sr->pending)) {
                memcpy(sr->pending, cur, rem);
                sr->pending_len = rem;
            }
            break;
        }

        if (sr->dsml_active) {
            agent_stream_feed_dsml_byte(sr, cur[0]);
        } else if (sr->in_think) {
            /* Tool calls are executable only after thinking has closed.  Still
             * route thinking bytes through the DSML start detector so an
             * accidental in-think tool stanza can be suppressed cleanly instead
             * of being shown as raw markup or, worse, executed. */
            agent_stream_normal_byte(sr, cur[0]);
        } else {
            agent_stream_normal_byte(sr, cur[0]);
        }
        i++;
    }
    free(buf);

    if (finish) {
        agent_stream_flush_start_tail(sr);
        sr->post_think_gap = false;
        if (sr->dsml_active) {
            if (sr->dsml_ignored) {
                agent_stream_finish_ignored_dsml(
                    sr, "tool calling is not allowed inside <think></think>");
            } else {
                agent_tool_viz_finish(sr, sr->tool_preflight_error ?
                                      "[tool call stopped: edit old selector failed]\n" :
                                      "[tool call interrupted]\n");
                sr->dsml_active = false;
            }
        }
        if (sr->dsml_in_think && !sr->dsml_in_think_reported) {
            agent_stream_finish_ignored_dsml(
                sr, "tool calling is not allowed inside <think></think>");
        }
    }
}

