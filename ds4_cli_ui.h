#ifndef DS4_CLI_UI_H
#define DS4_CLI_UI_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    DS4_CLI_LOGO_AUTO,
    DS4_CLI_LOGO_IMAGE,
    DS4_CLI_LOGO_TEXT,
} ds4_cli_logo_mode;

typedef struct {
    const char *model;
    const char *backend;
    const char *thinking;
    int ctx_size;
    int power_percent;
    bool ssd_streaming;
    bool vision;
    uint64_t cache_bytes;
    ds4_cli_logo_mode logo_mode;
} ds4_cli_ui_state;

bool ds4_cli_ui_color_enabled(FILE *fp);
void ds4_cli_ui_print_welcome(FILE *fp, const ds4_cli_ui_state *state);
void ds4_cli_ui_print_status(FILE *fp, const ds4_cli_ui_state *state);
void ds4_cli_ui_print_commands(FILE *fp);
void ds4_cli_ui_print_notice(FILE *fp, const char *label, const char *format, ...);
void ds4_cli_ui_print_assistant_prompt(FILE *fp);
const char *ds4_cli_ui_user_prompt(FILE *fp);

#endif
