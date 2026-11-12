#ifndef DS4_CLI_UI_H
#define DS4_CLI_UI_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <termios.h>

struct linenoiseState;

typedef struct { const char *name, *description; } ds4_cli_ui_command;

typedef struct {
    const char *model;
    const char *backend;
    const char *thinking;
    const char *directory;
    int ctx_size;
    int ctx_used;
    int power_percent;
    bool ssd_streaming;
    bool vision;
    uint64_t cache_bytes;
} ds4_cli_ui_state;

typedef struct {
    FILE *fp;
    int width, margin, column, glyph_width, escape;
    bool enabled, line_start, joined, regional;
    char utf8[4];
    size_t utf8_len;
    char ansi[64];
    size_t ansi_len;
} ds4_cli_ui_output;

typedef struct {
    ds4_cli_ui_state state;
    int rows, columns, output_bottom, prompt_row;
    int inline_prefix, content_row;
    int logo_top, logo_left, logo_width, logo_height;
    bool active, alt_screen, braille, logo_pending, logo_visible, mouse_enabled;
    bool content_positioned;
    bool busy, external_footer;
    struct termios input_mode;
    struct linenoiseState *edit;
    size_t scroll_offset;
    int content_utf8;
    char busy_input[32];
    size_t busy_input_len;
    char start_input[512];
    size_t start_input_len;
    int message_kind, last_kind, trailing_newlines, content_escape;
    double animation_start, frame_time;
    FILE *terminal, *saved_errors, *content, *errors;
    char *history;
    size_t history_len, history_cap;
    ds4_cli_ui_output system_output;
    ds4_cli_ui_output text_output;
} ds4_cli_ui;

bool ds4_cli_ui_color_enabled(FILE *fp);
void ds4_cli_ui_output_start(ds4_cli_ui_output *out, FILE *fp);
void ds4_cli_ui_output_write(ds4_cli_ui_output *out, const char *text, size_t length);
void ds4_cli_ui_output_finish(ds4_cli_ui_output *out);
void ds4_cli_ui_start(ds4_cli_ui *ui, bool no_alt_screen);
void ds4_cli_ui_format_footer(const ds4_cli_ui_state *state, int columns, char *text, size_t length);
int ds4_cli_ui_edit_begin(ds4_cli_ui *ui, struct linenoiseState *edit,
                         char *buffer, size_t length, const char *footer);
void ds4_cli_ui_edit_end(ds4_cli_ui *ui, struct linenoiseState *edit, bool submitted);
void ds4_cli_ui_edit_tick(ds4_cli_ui *ui, struct linenoiseState *edit);
void ds4_cli_ui_print_user(ds4_cli_ui *ui, const char *text);
char *ds4_cli_ui_readline(ds4_cli_ui *ui, const ds4_cli_ui_state *state);
void ds4_cli_ui_stop(ds4_cli_ui *ui);
void ds4_cli_ui_print_status(FILE *fp, const ds4_cli_ui_state *state);
void ds4_cli_ui_print_commands(FILE *fp);
void ds4_cli_ui_print_command_list(FILE *fp, const ds4_cli_ui_command *commands, size_t count);
void ds4_cli_ui_print_notice(FILE *fp, const char *label, const char *format, ...);

#endif
