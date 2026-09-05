#include "ds4_cli_ui.h"
#include "ds4_cli_logo.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

typedef struct {
    int width;
    int margin;
    int rows;
    double cell_aspect;
    bool unicode;
    bool images;
    const char *off;
    const char *accent;
    const char *bold;
    const char *muted;
} cli_ui_style;

static const struct { const char *name, *description; } commands[] = {
    {"/help",       "Command guide"},
    {"/status",     "Session details"},
    {"/think",      "Normal reasoning"},
    {"/think-max",  "Max reasoning"},
    {"/nothink",    "Direct answers"},
    {"/ctx N",      "Restart context"},
    {"/power N",    "GPU duty cycle"},
    {"/steer F",    "FFN steering"},
    {"/read FILE",  "Read text/image"},
    {"/quit /exit", "Exit session"},
    {"Ctrl+C",      "Stop generation"},
};
enum { COMMAND_COUNT = sizeof(commands) / sizeof(commands[0]), SIDEBAR_ROWS = 7 + COMMAND_COUNT };

static int smaller(int a, int b) { return a < b ? a : b; }
static int larger(int a, int b) { return a > b ? a : b; }

static bool cli_ui_terminal(FILE *fp) {
    const char *term = getenv("TERM");
    return fp && isatty(fileno(fp)) && (!term || strcmp(term, "dumb"));
}

bool ds4_cli_ui_color_enabled(FILE *fp) {
    return cli_ui_terminal(fp) && getenv("NO_COLOR") == NULL;
}

static cli_ui_style cli_ui_make_style(FILE *fp) {
    cli_ui_style s = {.cell_aspect = 0.5, .off = "", .accent = "", .bold = "", .muted = ""};
    struct winsize size = {0};
    if (fp && isatty(fileno(fp))) ioctl(fileno(fp), TIOCGWINSZ, &size);
    int columns = size.ws_col ? size.ws_col : 80;
    s.rows = size.ws_row;
    s.margin = columns >= 12 ? 2 : 0;
    s.width = larger(1, smaller(108, columns - 2 * s.margin - 1));
    if (size.ws_xpixel && size.ws_ypixel && size.ws_col && size.ws_row) {
        double aspect = ((double)size.ws_xpixel / size.ws_col) /
                        ((double)size.ws_ypixel / size.ws_row);
        if (aspect >= 0.3 && aspect <= 1.0) s.cell_aspect = aspect;
    }
    const char *locale = getenv("LC_ALL");
    if (!locale || !*locale) locale = getenv("LC_CTYPE");
    if (!locale || !*locale) locale = getenv("LANG");
    s.unicode = cli_ui_terminal(fp) && (!locale || !*locale ||
                strstr(locale, "UTF-8") || strstr(locale, "utf-8") ||
                strstr(locale, "utf8") || strstr(locale, "UTF8"));
    if (ds4_cli_ui_color_enabled(fp)) {
        s.off = "\x1b[0m";
        s.accent = "\x1b[38;5;180m";
        s.bold = "\x1b[1m";
        s.muted = "\x1b[90m";
    }
    const char *term = getenv("TERM");
    const char *program = getenv("TERM_PROGRAM");
    // Conservative detection: never send graphics into pipes or multiplexers.
    // q=2 below avoids protocol replies becoming input to the chat prompt.
    s.images = ds4_cli_ui_color_enabled(fp) && !getenv("TMUX") &&
               !(term && (strstr(term, "screen") || strstr(term, "tmux"))) &&
               ((term && (!strcmp(term, "xterm-kitty") || !strcmp(term, "xterm-ghostty"))) ||
                (program && !strcmp(program, "ghostty")));
    return s;
}

static void spaces(FILE *fp, int count) {
    while (count-- > 0) fputc(' ', fp);
}

// Session labels, engine model names and command descriptions are ASCII.
// Unicode logo cells are written separately, so byte counts cannot skew them.
static void text_cell(FILE *fp, const cli_ui_style *s, const char *color,
                      const char *text, int width, bool pad) {
    int len = (int)strlen(text);
    int used = smaller(len, larger(0, width));
    fputs(color, fp);
    if (len > width && width >= 3) {
        fwrite(text, 1, (size_t)(width - 3), fp);
        fputs("...", fp);
    } else if (used > 0) {
        fwrite(text, 1, (size_t)used, fp);
    }
    fputs(s->off, fp);
    if (pad) spaces(fp, width - used);
}

static void rule(FILE *fp, const cli_ui_style *s) {
    spaces(fp, s->margin);
    fputs(s->muted, fp);
    for (int i = 0; i < s->width; i++) fputs(s->unicode ? "─" : "-", fp);
    fputs(s->off, fp);
    fputc('\n', fp);
}

static void heading(FILE *fp, const cli_ui_style *s, const char *title) {
    spaces(fp, s->margin);
    text_cell(fp, s, s->bold, title, s->width, false);
    fputc('\n', fp);
}

static void format_count(int value, char *buf, size_t len) {
    if (value >= 1000000)
        snprintf(buf, len, "%d,%03d,%03d", value / 1000000, (value / 1000) % 1000, value % 1000);
    else if (value >= 1000)
        snprintf(buf, len, "%d,%03d", value / 1000, value % 1000);
    else snprintf(buf, len, "%d", value);
}

static void session_lines(const ds4_cli_ui_state *state, char lines[5][160]) {
    char ctx[32];
    format_count(state->ctx_size, ctx, sizeof(ctx));
    snprintf(lines[0], 160, "%s", state->model ? state->model : "Unknown model");
    snprintf(lines[1], 160, "%s / %s ctx", state->backend ? state->backend : "unknown", ctx);
    snprintf(lines[2], 160, "%s", state->ssd_streaming ? "SSD streaming" : "Model in RAM");
    if (state->ssd_streaming && state->cache_bytes) {
        snprintf(lines[2], 160, "SSD streaming / %.0f GB cache", (double)state->cache_bytes / 1073741824.0);
    }
    snprintf(lines[3], 160, "thinking %s%s", state->thinking ? state->thinking : "unknown",
             state->vision ? " / vision on" : "");
    snprintf(lines[4], 160, "GPU power %d%%", state->power_percent);
}

static void command(FILE *fp, const cli_ui_style *s, int index, int width, bool pad) {
    int name_width = smaller(width, 13);
    text_cell(fp, s, s->accent, commands[index].name, name_width, width > name_width || pad);
    if (width > name_width) {
        fputc(' ', fp);
        text_cell(fp, s, s->muted, commands[index].description, width - name_width - 1, pad);
    }
}

static void command_rows(FILE *fp, const cli_ui_style *s) {
    int columns = s->width >= 70 ? 2 : 1;
    int count = (COMMAND_COUNT + columns - 1) / columns;
    int width = columns == 2 ? (s->width - 4) / 2 : s->width;
    for (int row = 0; row < count; row++) {
        spaces(fp, s->margin);
        if (width < 32) {
            text_cell(fp, s, s->accent, commands[row].name, width, false);
            fputc('\n', fp);
            spaces(fp, s->margin);
            text_cell(fp, s, s->muted, commands[row].description, width, false);
        } else {
            command(fp, s, row, width, columns == 2);
            if (columns == 2 && row + count < COMMAND_COUNT) {
                spaces(fp, 4);
                command(fp, s, row + count, width, false);
            }
        }
        fputc('\n', fp);
    }
}

void ds4_cli_ui_print_status(FILE *fp, const ds4_cli_ui_state *state) {
    cli_ui_style s = cli_ui_make_style(fp);
    char lines[5][160];
    session_lines(state, lines);
    heading(fp, &s, "Session");
    fputc('\n', fp);
    for (int i = 0; i < 5; i++) {
        spaces(fp, s.margin);
        text_cell(fp, &s, i == 0 ? s.bold : "", lines[i], s.width, false);
        fputc('\n', fp);
    }
}

void ds4_cli_ui_print_commands(FILE *fp) {
    cli_ui_style s = cli_ui_make_style(fp);
    fputc('\n', fp);
    heading(fp, &s, "Commands");
    fputc('\n', fp);
    command_rows(fp, &s);
    fputc('\n', fp);
}

static double logo_coverage(int x, int y, int width, int height) {
    int x0 = x * DS4_CLI_LOGO_WIDTH / width;
    int x1 = larger(x0 + 1, (x + 1) * DS4_CLI_LOGO_WIDTH / width);
    int y0 = y * DS4_CLI_LOGO_HEIGHT / height;
    int y1 = larger(y0 + 1, (y + 1) * DS4_CLI_LOGO_HEIGHT / height);
    int ink = 0;
    for (int sy = y0; sy < y1; sy++) {
        for (int sx = x0; sx < x1; sx++) {
            int bit = sy * DS4_CLI_LOGO_WIDTH + sx;
            ink += (ds4_cli_logo_bits[bit / 8] >> (bit % 8)) & 1;
        }
    }
    return (double)ink / ((x1 - x0) * (y1 - y0));
}

static void logo_row(FILE *fp, const cli_ui_style *s, int row, int width, int height) {
    static const int dots[4][2] = {{0, 3}, {1, 4}, {2, 5}, {6, 7}};
    static const char ramp[] = " .:-=+*#%@";
    fputs(s->accent, fp);
    for (int x = 0; x < width; x++) {
        if (!s->unicode) {
            int index = (int)(logo_coverage(x, row, width, height) * 9 + 0.5);
            fputc(ramp[index], fp);
            continue;
        }
        unsigned int bits = 0;
        for (int dy = 0; dy < 4; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                if (logo_coverage(2 * x + dx, 4 * row + dy, 2 * width, 4 * height) >= 0.35)
                    bits |= 1u << dots[dy][dx];
            }
        }
        if (!bits) fputc(' ', fp);
        else {
            unsigned int cp = 0x2800 + bits;
            fputc(0xe0 | (cp >> 12), fp);
            fputc(0x80 | ((cp >> 6) & 63), fp);
            fputc(0x80 | (cp & 63), fp);
        }
    }
    fputs(s->off, fp);
}

static void logo_image(FILE *fp, int up, int left, int width) {
    // Reserve space first so the terminal scrolls the image and text together.
    fprintf(fp, "\0337\033[%dA\r", up);
    if (left) fprintf(fp, "\033[%dC", left);
    size_t total = sizeof(ds4_cli_logo_png_base64) - 1;
    for (size_t offset = 0; offset < total; offset += 4096) {
        size_t count = total - offset;
        if (count > 4096) count = 4096;
        // Specify columns only: the terminal preserves the SVG aspect ratio
        // using its actual font metrics instead of stretching to a cell box.
        if (offset == 0)
            fprintf(fp, "\033_Ga=T,f=100,t=d,q=2,C=1,c=%d,m=%d;", width, offset + count < total);
        else fprintf(fp, "\033_Gq=2,m=%d;", offset + count < total);
        fwrite(ds4_cli_logo_png_base64 + offset, 1, count, fp);
        fputs("\033\\", fp);
    }
    fputs("\0338", fp);
}

void ds4_cli_ui_print_welcome(FILE *fp, const ds4_cli_ui_state *state) {
    cli_ui_style s = cli_ui_make_style(fp);
    char lines[5][160];
    session_lines(state, lines);
    bool side = s.width >= 74;
    int right_width = side ? (s.width >= 90 ? 36 : 32) : 0;
    int logo_width = smaller(64, side ? s.width - right_width - 4 : s.width);
    double aspect = (double)DS4_CLI_LOGO_HEIGHT / DS4_CLI_LOGO_WIDTH * s.cell_aspect;
    // Reserve space for the heading, footer and input on short screens.
    if (s.rows > 0 && s.rows < 34) {
        int available = larger(10, s.rows - 7);
        logo_width = smaller(logo_width, (int)(available / aspect));
    }
    int logo_height = larger(1, (int)(logo_width * aspect));
    if (logo_height < logo_width * aspect) logo_height++;
    int body_height = side ? larger(logo_height, SIDEBAR_ROWS) : logo_height;
    bool image = state->logo_mode != DS4_CLI_LOGO_TEXT && s.images &&
                 (!s.rows || body_height < s.rows);
    int logo_top = (body_height - logo_height) / 2;
    int details_top = (body_height - SIDEBAR_ROWS) / 2;
    int logo_left = s.margin + (side ? 0 : (s.width - logo_width) / 2);

    fputc('\n', fp);
    spaces(fp, s.margin);
    text_cell(fp, &s, s.bold, "DWARFSTAR", smaller(s.width, 9), false);
    if (s.width >= 30) {
        spaces(fp, 3);
        text_cell(fp, &s, s.muted, "ds4 / interactive", s.width - 12, false);
    }
    fputc('\n', fp);
    rule(fp, &s);
    fputc('\n', fp);
    for (int row = 0; row < body_height; row++) {
        spaces(fp, logo_left);
        if (!image && row >= logo_top && row < logo_top + logo_height)
            logo_row(fp, &s, row - logo_top, logo_width, logo_height);
        else spaces(fp, logo_width);
        if (side) {
            spaces(fp, 4);
            int line = row - details_top;
            if (line >= 0 && line < 5)
                text_cell(fp, &s, line == 0 ? s.bold : s.muted, lines[line], right_width, false);
            else if (line == 6) text_cell(fp, &s, s.bold, "Commands", right_width, false);
            else if (line >= 7 && line < SIDEBAR_ROWS) command(fp, &s, line - 7, right_width, false);
        }
        fputc('\n', fp);
    }
    if (image) logo_image(fp, body_height - logo_top, logo_left, logo_width);
    if (!side) {
        fputc('\n', fp);
        for (int i = 0; i < 5; i++) {
            spaces(fp, s.margin);
            text_cell(fp, &s, i == 0 ? s.bold : s.muted, lines[i], s.width, false);
            fputc('\n', fp);
        }
        ds4_cli_ui_print_commands(fp);
    } else fputc('\n', fp);
    rule(fp, &s);
    if (state->logo_mode == DS4_CLI_LOGO_IMAGE && !image) {
        spaces(fp, s.margin);
        text_cell(fp, &s, s.muted, "Image logo unavailable; using text.", s.width, false);
        fputc('\n', fp);
    }
    fputc('\n', fp);
    fflush(fp);
}

void ds4_cli_ui_print_notice(FILE *fp, const char *label, const char *format, ...) {
    cli_ui_style s = cli_ui_make_style(fp);
    fprintf(fp, "  %s%s%s  ", s.accent, label, s.off);
    va_list args;
    va_start(args, format);
    vfprintf(fp, format, args);
    va_end(args);
    fputc('\n', fp);
}

void ds4_cli_ui_print_assistant_prompt(FILE *fp) {
    cli_ui_style s = cli_ui_make_style(fp);
    fprintf(fp, "%sds4%s %s>%s ", s.accent, s.off, s.muted, s.off);
    fflush(fp);
}

const char *ds4_cli_ui_user_prompt(FILE *fp) {
    if (!ds4_cli_ui_color_enabled(fp)) return "you > ";
    return "\x1b[1myou\x1b[0m \x1b[38;5;180m>\x1b[0m ";
}
