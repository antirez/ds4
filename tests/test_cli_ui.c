/* Model-free driver for the actual CLI editor and terminal layout. */
#include "ds4_cli_ui.h"
#include "linenoise.h"

#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile sig_atomic_t interrupted;
static void stop_generation(int signal) { (void)signal; interrupted = 1; }

int main(int argc, char **argv) {
    signal(SIGINT, stop_generation);
    bool no_alt_screen = argc == 2 && !strcmp(argv[1], "--no-alt-screen");
    ds4_cli_ui ui;
    ds4_cli_ui_state state = {
        .model = "Preview model", .backend = "metal", .thinking = "high",
        .ctx_size = 32768, .power_percent = 100,
    };
    ds4_cli_ui_start(&ui, no_alt_screen);
    for (;;) {
        errno = 0;
        char *line = ds4_cli_ui_readline(&ui, &state);
        if (!line) {
            if (errno == EAGAIN) continue;
            break;
        }
        if (*line) linenoiseHistoryAdd(line);
        if (!strcmp(line, "/quit")) {
            free(line);
            break;
        }
        if (!strcmp(line, "/help")) ds4_cli_ui_print_commands(stdout);
        else if (!strcmp(line, "/status")) ds4_cli_ui_print_status(stdout, &state);
        else if (!strcmp(line, "/context")) state.ctx_used = 8192;
        else if (!strncmp(line, "/power 50", 9) && !line[9 + strspn(line + 9, " ")])
            state.power_percent = 50;
        else if (!strcmp(line, "/system")) {
            fprintf(stderr, "\rprocessing 10 tokens: 5/10\x1b[K");
            fprintf(stderr, "\rprocessing 10 tokens: done\x1b[K\n");
            fprintf(stderr, "ds4: prefill: 120 t/s\n");
            fprintf(stderr, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\n");
        }
        else if (!strcmp(line, "/spacing")) {
            fprintf(stderr, "\r\x1b[90mprocessing: 1/2\x1b[K");
            fprintf(stderr, "\rprocessing: done\x1b[K\n\x1b[0m");
            ds4_cli_ui_output output;
            ds4_cli_ui_output_start(&output, stdout);
            ds4_cli_ui_output_write(&output, "answer text\n", 12);
            ds4_cli_ui_output_finish(&output);
            fprintf(stderr, "\x1b[90mds4: prefill: 120 t/s\n\x1b[0m");
        }
        else if (*line) {
            ds4_cli_ui_output output;
            ds4_cli_ui_output_start(&output, stdout);
            if (!strcmp(line, "/layout")) {
                const char *text = "answer first\nabcdefghijklmnopqrstuv\n中é\ntab\tend\n\x1b[90mcolored\x1b[0m\n";
                for (size_t i = 0; text[i]; i++) ds4_cli_ui_output_write(&output, text + i, 1);
            } else if (!strncmp(line, "/echo ", 6)) {
                ds4_cli_ui_output_write(&output, "received: ", 10);
                ds4_cli_ui_output_write(&output, line + 6, strlen(line + 6));
                ds4_cli_ui_output_write(&output, "\n", 1);
            } else {
                ds4_cli_ui_output_write(&output, "accepted: ", 10);
                ds4_cli_ui_output_write(&output, line, strlen(line));
                ds4_cli_ui_output_write(&output, "\n", 1);
                bool slow = !strcmp(line, "/slow");
                for (int row = 0; row < (slow ? 100 : 40) && !interrupted; row++) {
                    char text[64];
                    int length = snprintf(text, sizeof(text), "response row %d\n", row);
                    ds4_cli_ui_output_write(&output, text, (size_t)length);
                    fflush(stdout);
                    usleep(slow ? 15000 : 5000);
                }
                interrupted = 0;
            }
            ds4_cli_ui_output_finish(&output);
        }
        free(line);
    }
    ds4_cli_ui_stop(&ui);
    return 0;
}
