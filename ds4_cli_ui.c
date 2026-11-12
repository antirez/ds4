#include "ds4_cli_ui.h"
#include "ds4_cli_logo.h"
#include "linenoise.h"

#include <errno.h>
#include <langinfo.h>
#include <locale.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#include <wchar.h>

typedef struct {
    int width;
    int margin;
    double cell_aspect;
    const char *off;
    const char *accent;
    const char *bold;
    const char *muted;
    const char *white;
} cli_ui_style;

static const ds4_cli_ui_command commands[] = {
    {"/help",       "Command guide"},
    {"/status",     "Session details"},
    {"/think [N]",  "Normal / V4.1 0..100"},
    {"/think-max",  "Max reasoning"},
    {"/nothink",    "Direct answers"},
    {"/ctx N",      "Restart context"},
    {"/power N",    "GPU duty cycle"},
    {"/steer F",    "FFN steering"},
    {"/read FILE",  "Read text/image"},
    {"/quit /exit", "Exit session"},
    {"Ctrl+C",      "Stop generation"},
    {"Shift+Enter", "New line"},
    {"Ctrl+J",      "New line fallback"},
};
enum { COMMAND_COUNT = sizeof(commands) / sizeof(commands[0]) };
enum { INPUT_PADDING = 1, INPUT_GAP = 1, EMPTY_INPUT_ROWS = 1 + 2 * INPUT_PADDING };
enum { CONTENT_TEXT, CONTENT_USER, CONTENT_SYSTEM };
enum { LOGO_ANIMATION_MS = 3000, LOGO_FADE_MS = 350, LOGO_FRAME_MS = 60 };
static const char user_background[] = "\x1b[48;2;42;48;58m\x1b[38;5;252m";
static const char input_placeholder[] = "Message DwarfStar...";
static __thread unsigned int painting;
static void cursor(FILE *fp, int row, int column);
static void repaint_content(ds4_cli_ui *ui, FILE *fp);
static void park_cursor(ds4_cli_ui *ui);
static void scroll_conversation(ds4_cli_ui *ui, int rows);
static void poll_busy_input(ds4_cli_ui *ui);
static void set_layout(ds4_cli_ui *ui, FILE *fp, size_t prompt_rows, size_t status_rows);
static void show_idle_input(ds4_cli_ui *ui);
static void erase_logo(ds4_cli_ui *ui, FILE *fp);

static int terminal_fd(FILE *fp) {
    return fp == stdout ? STDOUT_FILENO : fp == stderr ? STDERR_FILENO : fileno(fp);
}

static int smaller(int a, int b) { return a < b ? a : b; }
static int larger(int a, int b) { return a > b ? a : b; }

static bool cli_ui_terminal(FILE *fp) {
    const char *term = getenv("TERM");
    return fp && isatty(terminal_fd(fp)) && (!term ||
           (strcmp(term, "dumb") && strcmp(term, "cons25") && strcmp(term, "emacs")));
}

bool ds4_cli_ui_color_enabled(FILE *fp) {
    return cli_ui_terminal(fp) && getenv("NO_COLOR") == NULL;
}

static cli_ui_style cli_ui_make_style(FILE *fp) {
    cli_ui_style s = {.cell_aspect = 0.5, .off = "", .accent = "", .bold = "", .muted = "", .white = ""};
    struct winsize size = {0};
    if (fp && isatty(terminal_fd(fp))) ioctl(terminal_fd(fp), TIOCGWINSZ, &size);
    int columns = size.ws_col ? size.ws_col : 80;
    s.margin = columns >= 12 ? 2 : 0;
    s.width = larger(1, smaller(108, columns - 2 * s.margin - 1));
    if (size.ws_xpixel && size.ws_ypixel && size.ws_col && size.ws_row) {
        double aspect = ((double)size.ws_xpixel / size.ws_col) /
                        ((double)size.ws_ypixel / size.ws_row);
        if (aspect >= 0.3 && aspect <= 1.0) s.cell_aspect = aspect;
    }
    if (ds4_cli_ui_color_enabled(fp)) {
        s.off = "\x1b[0m";
        s.accent = "\x1b[38;5;180m";
        s.bold = "\x1b[1m";
        s.muted = "\x1b[90m";
        s.white = "\x1b[97m";
    }
    return s;
}

static void spaces(FILE *fp, int count) {
    while (count-- > 0) fputc(' ', fp);
}

static void user_row_start(FILE *fp) {
    if (ds4_cli_ui_color_enabled(fp)) fputs(user_background, fp);
    fputs("\x1b[?7l", fp);
}

static void user_row_end(FILE *fp, int used, int columns, bool newline) {
    spaces(fp, columns - used);
    if (ds4_cli_ui_color_enabled(fp)) fputs("\x1b[0m", fp);
    fputs("\x1b[?7h", fp);
    if (newline) fputs("\r\n", fp);
}

static void print_user_message(FILE *fp, const char *text, int columns) {
    int margin = columns >= 8 ? 2 : 0;
    int width = larger(1, columns - 2 * margin);
    for (int row = 0; row < INPUT_PADDING; row++) {
        user_row_start(fp);
        user_row_end(fp, 0, columns, true);
    }
    size_t left = strlen(text);
    do {
        user_row_start(fp);
        spaces(fp, margin);
        int used = 0;
        while (left && *text != '\n') {
            int cells;
            size_t n = linenoiseNextGrapheme(text, left, &cells);
            if (*text == '\t') cells = smaller(width, 4 - used % 4);
            if (used && used + cells > width) break;
            if (cells > width) { fputc('?', fp); cells = 1; }
            else if (*text == '\t') spaces(fp, cells);
            else fwrite(text, 1, n, fp);
            used += cells;
            text += n;
            left -= n;
        }
        user_row_end(fp, margin + used, columns, true);
        if (!left) break;
        if (*text == '\n') { text++; left--; }
    } while (true);
    for (int row = 0; row < INPUT_PADDING; row++) {
        user_row_start(fp);
        user_row_end(fp, 0, columns, true);
    }
    fputc('\n', fp); // Keep a plain gap between the panel and the answer.
}

void ds4_cli_ui_output_start(ds4_cli_ui_output *out, FILE *fp) {
    memset(out, 0, sizeof(*out));
    out->fp = fp;
    out->line_start = true;
    out->enabled = isatty(STDIN_FILENO) && cli_ui_terminal(fp);
    struct winsize size = {0};
    if (out->enabled) ioctl(terminal_fd(fp), TIOCGWINSZ, &size);
    int columns = size.ws_col ? size.ws_col : 80;
    out->margin = columns >= 8 ? 2 : 0;
    out->width = larger(1, columns - 2 * out->margin);
}

static void output_character(ds4_cli_ui_output *out, const char *text, size_t length, uint32_t cp) {
    if (cp == '\n' || cp == '\r') {
        fwrite(text, 1, length, out->fp);
        out->column = 0;
        out->line_start = true;
        out->joined = out->regional = false;
        out->glyph_width = 0;
        return;
    }
    if (cp == '\t') {
        int count = 4 - out->column % 4;
        while (count-- > 0) output_character(out, " ", 1, ' ');
        return;
    }
    int width = linenoiseCharacterWidth(cp);
    bool regional = cp >= 0x1f1e6 && cp <= 0x1f1ff;
    if (cp == 0x200d) out->joined = true;
    else if (cp == 0xfe0f && out->glyph_width == 1) { width = 1; out->glyph_width = 2; }
    else if (width) {
        if (out->joined || (regional && out->regional)) {
            int glyph_width = larger(out->glyph_width, width);
            width = glyph_width - out->glyph_width;
            out->glyph_width = glyph_width;
            out->joined = out->regional = false;
        } else {
            out->glyph_width = width;
            out->regional = regional;
        }
    }
    if (out->column + width > out->width) {
        fputc('\n', out->fp);
        out->column = 0;
        out->line_start = true;
    }
    if (out->line_start) { spaces(out->fp, out->margin); out->line_start = false; }
    fwrite(text, 1, length, out->fp);
    out->column += width;
}

/* Token boundaries may split UTF-8 or ANSI sequences. Keep display-width
 * accounting across chunks so wrapping never consumes either side margin. */
void ds4_cli_ui_output_write(ds4_cli_ui_output *out, const char *text, size_t length) {
    if (!out->enabled) { fwrite(text, 1, length, out->fp); return; }
    struct winsize size = {0};
    if (out->fp == stdout && ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col) {
        out->margin = size.ws_col >= 8 ? 2 : 0;
        out->width = larger(1, size.ws_col - 2 * out->margin);
    }
    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)text[i];
        if (out->escape) {
            if (out->ansi_len == sizeof(out->ansi)) { out->escape = 0; out->ansi_len = 0; continue; }
            out->ansi[out->ansi_len++] = (char)c;
            out->escape = out->escape == 1 ? (c == '[' ? 2 : 0) : (c >= '@' && c <= '~' ? 0 : 2);
            if (!out->escape) { fwrite(out->ansi, 1, out->ansi_len, out->fp); out->ansi_len = 0; }
            continue;
        }
        if (c == 0x1b && !out->utf8_len) {
            out->escape = 1;
            out->ansi[0] = (char)c;
            out->ansi_len = 1;
            continue;
        }
        if (out->utf8_len && (c & 0xc0) != 0x80) {
            output_character(out, "?", 1, '?');
            out->utf8_len = 0;
        }
        if (out->utf8_len || c >= 0x80) {
            out->utf8[out->utf8_len++] = (char)c;
            uint32_t cp;
            size_t n = linenoiseUtf8Decode(out->utf8, out->utf8_len, &cp);
            if (!n) continue;
            if (n != out->utf8_len || (n == 1 && cp >= 0x80)) output_character(out, "?", 1, '?');
            else output_character(out, out->utf8, n, cp);
            out->utf8_len = 0;
        } else output_character(out, text + i, 1, c);
    }
}

void ds4_cli_ui_output_finish(ds4_cli_ui_output *out) {
    if (out->utf8_len) output_character(out, "?", 1, '?');
    out->utf8_len = 0;
    out->escape = 0;
    out->ansi_len = 0;
}

static void save_content_cursor(ds4_cli_ui *ui) {
    if (!ui->content_positioned) return;
    fputs("\0337", ui->terminal);
    fflush(ui->terminal);
    ui->content_positioned = false;
}

static void track_content_spacing(ds4_cli_ui *ui, const char *text, size_t length) {
    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)text[i];
        if (ui->content_escape) {
            ui->content_escape = ui->content_escape == 1 ? (c == '[' ? 2 : 0) :
                                 (c >= '@' && c <= '~' ? 0 : 2);
        } else if (c == 0x1b) ui->content_escape = 1;
        else if (c == '\n') ui->trailing_newlines = smaller(2, ui->trailing_newlines + 1);
        else if (c != '\r') ui->trailing_newlines = 0;
        if ((c & 0xc0) == 0x80) { if (ui->content_utf8) ui->content_utf8--; }
        else ui->content_utf8 = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : c >= 0xc2 ? 1 : 0;
    }
}

typedef struct { size_t end, lines; bool failed; } content_extent;

/* Keep trailing line breaks logical until the next content arrives. The input
 * gap already supplies the outer margin; paragraph gaps still appear between
 * messages, and painted user-panel padding remains part of the content. */
static content_extent render_content(FILE *fp, const char *text, size_t length,
                                     size_t *pending, int *escape, int row, int bottom) {
    content_extent extent = {0};
    if (!length) return extent;
    size_t start = 0;
    for (size_t i = 0; i < length; i++) {
        unsigned char c = (unsigned char)text[i];
        if (*escape) {
            *escape = *escape == 1 ? (c == '[' ? 2 : 0) : (c >= '@' && c <= '~' ? 0 : 2);
        } else if (c == 0x1b) *escape = 1;
        else if (c == '\n') {
            if (fp && fwrite(text + start, 1, i - start, fp) != i - start) extent.failed = true;
            (*pending)++;
            start = i + 1;
        } else if (c != '\r') {
            extent.end = i + 1;
            if (*pending) {
                if (fp && fwrite(text + start, 1, i - start, fp) != i - start) extent.failed = true;
                extent.lines += *pending;
                if (fp) {
                    // Scrolled blank rows must use the default background,
                    // even when the next panel has already selected its color.
                    fputs("\0337\x1b[0m", fp);
                    for (size_t n = 0; n < *pending; n++)
                        if (fputs("\r\n", fp) == EOF) extent.failed = true;
                    size_t room = (size_t)larger(0, bottom - row);
                    row += (int)(*pending < room ? *pending : room);
                    fputs("\0338", fp);
                    cursor(fp, row, 1);
                }
                *pending = 0;
                start = i;
            }
        }
    }
    if (fp && fwrite(text + start, 1, length - start, fp) != length - start) extent.failed = true;
    return extent;
}

static content_extent history_extent(const ds4_cli_ui *ui) {
    size_t pending = 0;
    int escape = 0;
    return render_content(NULL, ui->history, ui->history_len, &pending, &escape, 0, 0);
}

/* Retain the actual styled output, including carriage-return progress lines.
 * Repainting a smaller viewport never destroys the conversation behind it. */
static ssize_t content_write(void *cookie, const char *text, size_t length) {
    ds4_cli_ui *ui = cookie;
    if (!painting && !ui->content_escape && !ui->content_utf8) poll_busy_input(ui);
    if (!painting && ui->logo_visible) {
        painting++;
        fputs("\x1b[?2026h\x1b[?25l", stdout);
        erase_logo(ui, stdout);
        park_cursor(ui);
        fputs("\x1b[?25h\x1b[?2026l", stdout);
        painting--;
    }
    size_t output_length = length;
    int escape = ui->content_escape;
    if (painting) save_content_cursor(ui);
    else {
        // One plain blank row separates panels, answers and system output.
        // Track actual newlines so existing gaps are reused, including after
        // carriage-return progress and ANSI color resets split across writes.
        if (ui->message_kind != ui->last_kind) {
            ui->last_kind = ui->message_kind;
            int missing = 2 - ui->trailing_newlines;
            if (missing && content_write(ui, "\n\n", (size_t)missing) < 0) return -1;
        }
        track_content_spacing(ui, text, length);
    }
    if (!painting) {
        if (length > (SIZE_MAX - ui->history_len - 1) / 2) { errno = ENOMEM; return -1; }
        size_t start = ui->history_len;
        size_t needed = start + 2 * length + 1;
        if (needed > ui->history_cap) {
            size_t capacity = needed <= SIZE_MAX / 2 ? needed * 2 : needed;
            char *history = realloc(ui->history, capacity);
            if (!history) return -1;
            ui->history = history;
            ui->history_cap = capacity;
        }
        // Editing disables terminal newline translation. Retain explicit CRLF
        // so replay has the same columns in both raw and cooked terminal mode.
        for (size_t i = 0; i < length; i++) {
            if (text[i] == '\n' && (!ui->history_len || ui->history[ui->history_len - 1] != '\r'))
                ui->history[ui->history_len++] = '\r';
            ui->history[ui->history_len++] = text[i];
        }
        ui->history[ui->history_len] = '\0';
        text = ui->history + start;
        output_length = ui->history_len - start;
        if (!ui->scroll_offset) {
            if (!ui->content_positioned) fputs("\x1b[?2026h\x1b[?25l\0338", ui->terminal);
            ui->content_positioned = true;
        }
    }
    content_extent rendered = {0};
    if (painting) rendered.failed = fwrite(text, 1, output_length, ui->terminal) != output_length;
    else {
        rendered = render_content(ui->scroll_offset ? NULL : ui->terminal, text, output_length,
                                  &ui->pending_newlines, &escape, ui->content_row, ui->output_bottom);
        size_t room = (size_t)larger(0, ui->output_bottom - ui->content_row);
        if (!ui->alt_screen && rendered.lines > room) {
            size_t scrolled = rendered.lines - room;
            ui->inline_prefix -= scrolled < (size_t)ui->inline_prefix ? (int)scrolled : ui->inline_prefix;
        }
        ui->content_row += (int)(rendered.lines < room ? rendered.lines : room);
        if (ui->scroll_offset) {
            ui->scroll_offset += rendered.lines;
            poll_busy_input(ui);
            return rendered.failed ? -1 : (ssize_t)length;
        }
    }
    if (!painting && !ui->content_escape && !ui->content_utf8) {
        fputs("\0337", ui->terminal);
        ui->content_positioned = false;
        park_cursor(ui);
        fputs("\x1b[?25h\x1b[?2026l", ui->terminal);
    }
    fflush(ui->terminal);
    if (!painting && !ui->content_escape && !ui->content_utf8) poll_busy_input(ui);
    return rendered.failed ? -1 : (ssize_t)length;
}

static ssize_t system_write(void *cookie, const char *text, size_t length) {
    ds4_cli_ui *ui = cookie;
    unsigned int was_painting = painting;
    int was_kind = ui->message_kind;
    painting = 0;
    ui->message_kind = CONTENT_SYSTEM;
    ds4_cli_ui_output_write(&ui->system_output, text, length);
    ui->message_kind = was_kind;
    painting = was_painting;
    return (ssize_t)length;
}

#ifdef __APPLE__
static int content_write_mac(void *cookie, const char *text, int length) {
    return (int)content_write(cookie, text, (size_t)length);
}
static int system_write_mac(void *cookie, const char *text, int length) {
    return (int)system_write(cookie, text, (size_t)length);
}
#endif

static FILE *content_stream(ds4_cli_ui *ui, bool system) {
#ifdef __APPLE__
    FILE *fp = funopen(ui, NULL, system ? system_write_mac : content_write_mac, NULL, NULL);
#else
    cookie_io_functions_t functions = {.write = system ? system_write : content_write};
    FILE *fp = fopencookie(ui, "w", functions);
#endif
    if (fp) setvbuf(fp, NULL, _IONBF, 0);
    return fp;
}

static size_t text_prefix(const char *text, int width, int *used) {
    size_t bytes = 0, len = strlen(text);
    *used = 0;
    while (bytes < len) {
        int cells;
        size_t n = linenoiseNextGrapheme(text + bytes, len - bytes, &cells);
        if (!n || *used + cells > width) break;
        *used += cells;
        bytes += n;
    }
    return bytes;
}

static void text_cell(FILE *fp, const cli_ui_style *s, const char *color,
                      const char *text, int width, bool pad) {
    int used;
    size_t bytes = text_prefix(text, larger(0, width), &used);
    fputs(color, fp);
    if (text[bytes] && width >= 3) {
        bytes = text_prefix(text, width - 3, &used);
        fwrite(text, 1, bytes, fp);
        fputs("...", fp);
        used += 3;
    } else fwrite(text, 1, bytes, fp);
    fputs(s->off, fp);
    if (pad) spaces(fp, width - used);
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

static void command(FILE *fp, const cli_ui_style *s, const ds4_cli_ui_command *entry, int width, bool pad) {
    int name_width = smaller(width, 13);
    text_cell(fp, s, s->accent, entry->name, name_width, width > name_width || pad);
    if (width > name_width) {
        fputc(' ', fp);
        text_cell(fp, s, s->muted, entry->description, width - name_width - 1, pad);
    }
}

static void command_rows(FILE *fp, const cli_ui_style *s, const ds4_cli_ui_command *entries, int total) {
    int columns = s->width >= 70 ? 2 : 1;
    int count = (total + columns - 1) / columns;
    int width = columns == 2 ? (s->width - 4) / 2 : s->width;
    for (int row = 0; row < count; row++) {
        spaces(fp, s->margin);
        if (width < 32) {
            text_cell(fp, s, s->accent, entries[row].name, width, false);
            fputc('\n', fp);
            spaces(fp, s->margin);
            text_cell(fp, s, s->muted, entries[row].description, width, false);
        } else {
            command(fp, s, entries + row, width, columns == 2);
            if (columns == 2 && row + count < total) {
                spaces(fp, 4);
                command(fp, s, entries + row + count, width, false);
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
    ds4_cli_ui_print_command_list(fp, commands, COMMAND_COUNT);
}

void ds4_cli_ui_print_command_list(FILE *fp, const ds4_cli_ui_command *entries, size_t count) {
    cli_ui_style s = cli_ui_make_style(fp);
    heading(fp, &s, "Commands");
    fputc('\n', fp);
    command_rows(fp, &s, entries, (int)count);
}

/* Protect the obelisk and wordmark; only the surrounding flames are warped.
 * These bounds follow the symbol in the embedded 256 x 201 source bitmap. */
static bool flame_pixel(int x, int y) {
    return y >= 30 && y < 164 &&
           (x < 92 - 10 * y / 164 || x > 164 + 8 * y / 164);
}

static int logo_ink(int x, int y, double phase, double motion) {
    if (motion > 0 && flame_pixel(x, y)) {
        double taper = (164.0 - y) / 134.0;
        int sx = x + (int)lround(3 * taper * motion * sin(y * 0.16 + phase * 5));
        int sy = y + (int)lround(5 * taper * motion * sin(x * 0.22 - phase * 7));
        if (flame_pixel(sx, sy)) { x = sx; y = sy; }
    }
    if (x < 0 || x >= DS4_CLI_LOGO_WIDTH || y < 0 || y >= DS4_CLI_LOGO_HEIGHT) return 0;
    int bit = y * DS4_CLI_LOGO_WIDTH + x;
    return (ds4_cli_logo_bits[bit / 8] >> (bit % 8)) & 1;
}

static double logo_coverage(int x, int y, int width, int height, double phase, double motion) {
    int x0 = x * DS4_CLI_LOGO_WIDTH / width;
    int x1 = larger(x0 + 1, (x + 1) * DS4_CLI_LOGO_WIDTH / width);
    int y0 = y * DS4_CLI_LOGO_HEIGHT / height;
    int y1 = larger(y0 + 1, (y + 1) * DS4_CLI_LOGO_HEIGHT / height);
    int ink = 0;
    for (int sy = y0; sy < y1; sy++) {
        for (int sx = x0; sx < x1; sx++) {
            ink += logo_ink(sx, sy, phase, motion);
        }
    }
    return (double)ink / ((x1 - x0) * (y1 - y0));
}

/* Directional bevels are sampled inside the original mask. The left face
 * stays darker, while the lettering and flame edges catch light from above. */
static double logo_depth(int x, int row, int width, int height) {
    double nw = logo_coverage(x - 1, row - 1, width, height, 0, 0);
    double se = logo_coverage(x + 1, row + 1, width, height, 0, 0);
    double depth = 0.93 + 0.21 * (se - nw) - 0.04 * ((double)x / width + (double)row / height);
    int sx = (2 * x + 1) * DS4_CLI_LOGO_WIDTH / (2 * width);
    int sy = (2 * row + 1) * DS4_CLI_LOGO_HEIGHT / (2 * height);
    if (sy > 12 && sy < 164 && sx >= 92 - 10 * sy / 164 && sx < 118) depth *= 0.82;
    return fmax(0.54, fmin(1.0, depth));
}

static void logo_braille_row(FILE *fp, const cli_ui_style *s, int row, int width, int height,
                              double phase, double motion) {
    static const int dots[4][2] = {{0, 3}, {1, 4}, {2, 5}, {6, 7}};
    const int gold[3] = {215, 175, 135}, shine[3] = {255, 210, 145};
    int previous[3] = {-1, -1, -1};
    fputs(s->accent, fp);
    for (int x = 0; x < width; x++) {
        unsigned int bits = 0;
        for (int dy = 0; dy < 4; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                if (logo_coverage(2 * x + dx, 4 * row + dy, 2 * width, 4 * height, phase, motion) >= 0.35)
                    bits |= 1u << dots[dy][dx];
            }
        }
        if (!bits) fputc(' ', fp);
        else {
            if (*s->off) {
                // Project onto the bottom-left to top-right diagonal. Two
                // highlights cross the whole drawing on the flames' clock.
                double light = 0;
                if (motion > 0) {
                    double pass = 2 * phase * 1000 / LOGO_ANIMATION_MS;
                    double center = -0.10 + 1.20 * (pass - floor(pass));
                    double position = 0.5 * ((x + 0.5) / width + 1 - (row + 0.5) / height);
                    light = fmax(0.0, 1 - fabs(position - center) / 0.10);
                    light = light * light * (3 - 2 * light) * motion;
                }
                double brightness = (0.5 + 0.5 * motion) * logo_depth(x, row, width, height);
                int rgb[3];
                for (int i = 0; i < 3; i++) {
                    double base = gold[i] * brightness;
                    rgb[i] = (int)lround(base + (shine[i] - base) * light);
                }
                if (memcmp(rgb, previous, sizeof(rgb))) {
                    fprintf(fp, "\x1b[38;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
                    memcpy(previous, rgb, sizeof(rgb));
                }
            }
            unsigned int cp = 0x2800 + bits;
            fputc(0xe0 | (cp >> 12), fp);
            fputc(0x80 | ((cp >> 6) & 63), fp);
            fputc(0x80 | (cp & 63), fp);
        }
    }
    fputs(s->off, fp);
}

/* A UTF-8 locale and single-cell Braille are the portable capability check.
 * Terminals do not expose reliable coverage information for the user's font. */
static bool braille_supported(void) {
    locale_t locale = newlocale(LC_CTYPE_MASK, "", (locale_t)0);
    if (!locale) return false;
    locale_t previous = uselocale(locale);
    const char *encoding = nl_langinfo(CODESET);
    bool utf8 = !strcmp(encoding, "UTF-8") || !strcmp(encoding, "UTF8");
    bool supported = utf8 && wcwidth(L'\u2801') == 1 && wcwidth(L'\u28ff') == 1;
    uselocale(previous);
    freelocale(locale);
    return supported;
}

void ds4_cli_ui_format_footer(const ds4_cli_ui_state *state, int columns, char *footer, size_t length) {
    char ctx[32], used_ctx[32], info[512], details[1024];
    format_count(state->ctx_size, ctx, sizeof(ctx));
    format_count(larger(0, state->ctx_used), used_ctx, sizeof(used_ctx));
    double percent = state->ctx_size > 0 ? 100.0 * larger(0, state->ctx_used) / state->ctx_size : 0;
    const char *help = columns >= 100 ? "type /help for commands" : "/help";
    int margin = columns >= 8 ? 2 : 0;
    int available = larger(0, columns - 2 * margin - (int)strlen(help) - 3);
    char directory[512] = "";
    if (state->directory && *state->directory) {
        const char *path = state->directory;
        const char *home = getenv("HOME");
        char short_path[PATH_MAX + 2];
        size_t n_home = home ? strlen(home) : 0;
        while (n_home > 1 && home[n_home - 1] == '/') n_home--;
        if (n_home && !strncmp(path, home, n_home) &&
            ((n_home == 1 && home[0] == '/') || !path[n_home] || path[n_home] == '/')) {
            snprintf(short_path, sizeof(short_path), "~/%s", path + n_home + (path[n_home] == '/'));
            path = short_path;
        }
        int width = larger(8, available / 3), used;
        size_t n = text_prefix(path, width, &used);
        bool clipped = path[n] != '\0';
        if (clipped) {
            while (*path) {
                n = text_prefix(path, width - 3, &used);
                if (!path[n]) break;
                path += linenoiseNextGrapheme(path, strlen(path), &used);
            }
        }
        snprintf(directory, sizeof(directory), "%s%s | ", clipped ? "..." : "", path);
    }
    snprintf(info, sizeof(info), "(%s) | %sctx %s / %s (%.0f%%)%s",
             state->thinking ? state->thinking : "unknown", directory, used_ctx, ctx, percent,
             state->vision ? " | vision" : "");
    int model_width = larger(3, available - (int)strlen(info) - 1);
    const char *model = state->model ? state->model : "Unknown model";
    int used;
    size_t bytes = text_prefix(model, model_width, &used);
    bool shorten = model[bytes] != '\0';
    if (shorten) bytes = text_prefix(model, model_width - 3, &used);
    snprintf(details, sizeof(details), "%.*s%s %s", (int)bytes, model,
             shorten ? "..." : "", info);
    size_t n = text_prefix(details, available, &used);
    bool clipped = details[n] != '\0';
    if (clipped && available >= 3) n = text_prefix(details, available - 3, &used);
    if (available)
        snprintf(footer, length, "%*s%.*s%s | %s", margin, "", (int)n, details,
                 clipped && available >= 3 ? "..." : "", help);
    else {
        n = text_prefix(help, larger(0, columns - 2 * margin), &used);
        snprintf(footer, length, "%*s%.*s", margin, "", (int)n, help);
    }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void cursor(FILE *fp, int row, int column) {
    fprintf(fp, "\x1b[%d;%dH", row, column);
}

static void clear_rows(FILE *fp, int first, int last) {
    for (int row = first; row <= last; row++) {
        cursor(fp, row, 1);
        fputs("\x1b[2K", fp);
    }
}

static void print_title(ds4_cli_ui *ui, FILE *fp) {
    if (!ui->alt_screen) return;
    cli_ui_style s = cli_ui_make_style(stdout);
    cursor(fp, 1, 1);
    fputs("\x1b[2K", fp);
    spaces(fp, ui->columns >= 8 ? 2 : 0);
    text_cell(fp, &s, s.accent, ui->title, larger(1, ui->columns - 4), false);
}

static void repaint_content(ds4_cli_ui *ui, FILE *fp) {
    flockfile(stdout);
    int top = ui->alt_screen ? 2 : ui->inline_prefix + 1;
    clear_rows(fp, top, ui->output_bottom);
    if (ds4_cli_ui_color_enabled(stdout)) fputs("\x1b[0m", fp);
    cursor(fp, top, 1);
    size_t end = ui->scroll_offset ? history_extent(ui).end : ui->history_len;
    for (size_t skip = ui->scroll_offset; skip && end; skip--) {
        if (ui->history[end - 1] == '\n') end--;
        while (end && ui->history[end - 1] != '\n') end--;
    }
    size_t pending = 0;
    int escape = 0;
    content_extent rendered = render_content(fp, ui->history, end, &pending, &escape,
                                            top, ui->output_bottom);
    size_t room = (size_t)larger(0, ui->output_bottom - top);
    ui->content_row = top + (int)(rendered.lines < room ? rendered.lines : room);
    if (!ui->scroll_offset) ui->pending_newlines = pending;
    fputs("\0337", fp);
    print_title(ui, fp);
    ui->content_positioned = false;
    funlockfile(stdout);
}

static void park_cursor(ds4_cli_ui *ui) {
    int row = ui->prompt_row + INPUT_PADDING, column = 3;
    if (ui->edit && ui->edit->screen_cursor_row > 0) {
        row = ui->edit->screen_cursor_row;
        column = ui->edit->screen_cursor_col;
    }
    cursor(ui->terminal, row, column);
}

static void scroll_conversation(ds4_cli_ui *ui, int rows) {
    size_t lines = 1 + history_extent(ui).lines;
    size_t height = (size_t)larger(1, ui->output_bottom - (ui->alt_screen ? 1 : ui->inline_prefix));
    size_t limit = lines > height ? lines - height : 0;
    size_t offset = rows > 0 ? ui->scroll_offset + (size_t)rows :
                    ui->scroll_offset > (size_t)-rows ? ui->scroll_offset - (size_t)-rows : 0;
    if (offset > limit) offset = limit;
    if (offset == ui->scroll_offset) return;
    ui->scroll_offset = offset;
    painting++;
    fputs("\x1b[?2026h", stdout);
    repaint_content(ui, stdout);
    park_cursor(ui);
    fputs("\x1b[?2026l", stdout);
    fflush(stdout);
    painting--;
}

static void mouse_tracking(ds4_cli_ui *ui, FILE *fp, bool enabled) {
    if (ui->mouse_enabled == enabled) return;
    fputs(enabled ? "\x1b[?1006h\x1b[?1000h" : "\x1b[?1000l\x1b[?1006l", fp);
    ui->mouse_enabled = enabled;
}

static void erase_logo(ds4_cli_ui *ui, FILE *fp) {
    if (ui->logo_visible)
        clear_rows(fp, ui->logo_top, ui->logo_top + ui->logo_height - 1);
    ui->logo_visible = false;
    ui->animation_start = 0;
}

static void print_logo(ds4_cli_ui *ui, FILE *fp) {
    cli_ui_style s = cli_ui_make_style(stdout);
    double aspect = (double)DS4_CLI_LOGO_HEIGHT / DS4_CLI_LOGO_WIDTH * s.cell_aspect;
    int available = larger(1, ui->output_bottom - 1);
    int width = smaller(64, smaller(ui->columns - 1, (int)(available / aspect)));
    if (width < 1) return;
    int height = larger(1, (int)ceil(width * aspect));
    ui->logo_top = 2 + (available - height) / 2;
    ui->logo_left = 1 + (ui->columns - width) / 2;
    ui->logo_width = width;
    ui->logo_height = height;

    double phase = 0, motion = 0;
    if (ui->animation_start) {
        phase = now_sec() - ui->animation_start;
        double elapsed = phase * 1000;
        if (elapsed >= LOGO_ANIMATION_MS) { ui->animation_start = 0; phase = 0; }
        else {
            motion = fmin(1.0, fmin(elapsed, LOGO_ANIMATION_MS - elapsed) / LOGO_FADE_MS);
            motion = motion * motion * (3 - 2 * motion);
        }
    }
    // Terminal text has no alpha channel. Dim the gold RGB value to emulate
    // 50% opacity, with a smooth fade to full brightness on activation.
    char gold[64];
    if (ds4_cli_ui_color_enabled(stdout)) {
        double brightness = 0.5 + 0.5 * motion;
        snprintf(gold, sizeof(gold), "\x1b[38;2;%d;%d;%dm",
                 (int)lround(215 * brightness), (int)lround(175 * brightness),
                 (int)lround(135 * brightness));
        s.accent = gold;
    }
    for (int row = 0; row < height; row++) {
        cursor(fp, ui->logo_top + row, ui->logo_left);
        logo_braille_row(fp, &s, row, width, height, phase, motion);
    }
    ui->frame_time = now_sec();
    ui->logo_visible = true;
    mouse_tracking(ui, fp, true);
}

/* Both buffers reserve the input, its outer gap and footer below the scrolling
 * output. Inline mode preserves the shell output above our own conversation. */
static void set_layout(ds4_cli_ui *ui, FILE *fp, size_t prompt_rows, size_t status_rows) {
    int gap = ui->rows >= 7 ? INPUT_GAP : 0;
    int bottom = larger(ui->alt_screen ? 2 : 1, ui->rows - (int)prompt_rows - (int)status_rows - gap);
    if (bottom != ui->output_bottom) {
        if (ui->logo_visible) erase_logo(ui, fp);
        fprintf(fp, "\x1b[%d;%dr", ui->alt_screen ? 2 : 1, bottom);
        if (!ui->alt_screen) {
            size_t lines = 1 + history_extent(ui).lines;
            int prefix = lines >= (size_t)bottom ? 0 : smaller(ui->inline_prefix, bottom - (int)lines);
            if (prefix < ui->inline_prefix) {
                cursor(fp, bottom, 1);
                for (int i = prefix; i < ui->inline_prefix; i++) fputs("\r\n", fp);
                ui->inline_prefix = prefix;
            }
        }
        ui->output_bottom = bottom;
        repaint_content(ui, fp);
        clear_rows(fp, bottom + 1, bottom + gap);
    }
    ui->prompt_row = bottom + gap + 1;
}

static int input_layout(struct linenoiseState *edit, size_t prompt_rows,
                        size_t status_rows, void *data) {
    ds4_cli_ui *ui = data;
    char *frame = NULL;
    size_t length = 0;
    FILE *fp = open_memstream(&frame, &length);
    if (!fp) return 0;
    set_layout(ui, fp, prompt_rows, status_rows);
    print_title(ui, fp);
    bool show_logo = ui->logo_pending && !ui->history_len && ui->braille && edit->len == 0 &&
                     !edit->paste_active && ui->rows >= 4 && ui->columns >= 12;
    if (show_logo) print_logo(ui, fp);
    else if (ui->logo_visible) erase_logo(ui, fp);
    cursor(fp, ui->prompt_row, 1);
    fclose(fp);
    linenoiseWrite(STDOUT_FILENO, frame, length);
    free(frame);
    return ui->prompt_row;
}

static void logo_click(struct linenoiseState *edit, unsigned int button,
                       unsigned int column, unsigned int row, int pressed, void *data) {
    (void)edit;
    ds4_cli_ui *ui = data;
    if (pressed && (button == 64 || button == 65)) {
        if (row >= (unsigned)(ui->alt_screen ? 2 : 1) && row <= (unsigned)ui->output_bottom)
            scroll_conversation(ui, button == 64 ? 3 : -3);
        return;
    }
    if (pressed && button == 0 && ui->logo_visible && !ui->animation_start &&
        column >= (unsigned)ui->logo_left && column < (unsigned)(ui->logo_left + ui->logo_width) &&
        row >= (unsigned)ui->logo_top && row < (unsigned)(ui->logo_top + ui->logo_height)) {
        ui->animation_start = now_sec();
        ui->frame_time = 0;
    }
}

static void poll_busy_input(ds4_cli_ui *ui) {
    if (!ui->busy) return;
    struct winsize size = {0};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_row && size.ws_col &&
        (size.ws_row != ui->rows || size.ws_col != ui->columns)) {
        painting++;
        fputs("\x1b[?2026h\x1b[?25l", stdout);
        ui->rows = size.ws_row;
        ui->columns = size.ws_col;
        ui->output_bottom = 0;
        set_layout(ui, stdout, EMPTY_INPUT_ROWS, 1);
        show_idle_input(ui);
        fputs("\x1b[?25h\x1b[?2026l", stdout);
        painting--;
    }
    struct pollfd input = {.fd = STDIN_FILENO, .events = POLLIN};
    char bytes[256];
    while (poll(&input, 1, 0) > 0 && (input.revents & POLLIN)) {
        ssize_t n = read(STDIN_FILENO, bytes, sizeof(bytes));
        if (n <= 0) break;
        for (ssize_t i = 0; i < n; i++) {
            char c = bytes[i];
            if (c == '\x1b') ui->busy_input_len = 0;
            if (c != '\x1b' && !ui->busy_input_len) continue;
            if (ui->busy_input_len == sizeof(ui->busy_input) - 1) { ui->busy_input_len = 0; continue; }
            ui->busy_input[ui->busy_input_len++] = c;
            ui->busy_input[ui->busy_input_len] = '\0';
            if (c == 'M' || c == 'm') {
                unsigned int button, column, row;
                char kind, extra;
                if (sscanf(ui->busy_input, "\x1b[<%5u;%5u;%5u%c%c", &button, &column, &row, &kind, &extra) == 4)
                    logo_click(NULL, button, column, row, kind == 'M', ui);
                ui->busy_input_len = 0;
            } else if (c == 'u' || c == '~') {
                unsigned int key, modifiers;
                if (linenoiseDecodeKey(ui->busy_input + 1, ui->busy_input_len - 1, &key, &modifiers) > 0 &&
                    ((modifiers - 1) & 4) && (key == 'c' || key == 'C')) raise(SIGINT);
                ui->busy_input_len = 0;
            }
        }
    }
}

static void set_busy(ds4_cli_ui *ui, bool busy) {
    if (!ui->active || busy == ui->busy) return;
    if (busy) {
        if (tcgetattr(STDIN_FILENO, &ui->input_mode)) return;
        struct termios mode = ui->input_mode;
        mode.c_lflag &= ~(ICANON | ECHO);
        mode.c_cc[VMIN] = mode.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &mode)) return;
        ui->busy = true;
    } else {
        poll_busy_input(ui);
        tcflush(STDIN_FILENO, TCIFLUSH);
        tcsetattr(STDIN_FILENO, TCSANOW, &ui->input_mode);
        ui->busy = false;
        ui->busy_input_len = 0;
    }
    fflush(ui->terminal);
}

static void update_size(ds4_cli_ui *ui, struct linenoiseState *edit) {
    struct winsize size = {0};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0) {
        if (size.ws_row) ui->rows = size.ws_row;
        if (size.ws_col) ui->columns = size.ws_col;
    }
    char footer[1024];
    ds4_cli_ui_format_footer(&ui->state, ui->columns, footer, sizeof(footer));
    cli_ui_style s = cli_ui_make_style(stdout);
    if (!ui->external_footer) linenoiseEditSetStatus(edit, footer, s.white, s.off);
}

static char *input_hint(const char *text, int *color, int *bold) {
    if (*text) return NULL;
    *color = ds4_cli_ui_color_enabled(stdout) ? 90 : -1;
    *bold = 0;
    return (char *)input_placeholder;
}

/* Leave a visible input bar and footer while synchronous generation writes
 * into the scroll region above them. The next read starts editing that bar. */
static void show_idle_input(ds4_cli_ui *ui) {
    cli_ui_style s = cli_ui_make_style(stdout);
    char footer[1024];
    ds4_cli_ui_format_footer(&ui->state, ui->columns, footer, sizeof(footer));
    for (int row = 0; row < EMPTY_INPUT_ROWS; row++) {
        cursor(stdout, ui->rows - EMPTY_INPUT_ROWS + row, 1);
        user_row_start(stdout);
        user_row_end(stdout, 0, ui->columns, false);
    }
    cursor(stdout, ui->rows - EMPTY_INPUT_ROWS + INPUT_PADDING, 3);
    if (ds4_cli_ui_color_enabled(stdout)) fputs(user_background, stdout);
    text_cell(stdout, &s, s.muted, input_placeholder, larger(1, ui->columns - 4), false);
    cursor(stdout, ui->rows, 1);
    fputs("\x1b[2K", stdout);
    text_cell(stdout, &s, s.white, footer, ui->columns - 1, false);
    park_cursor(ui);
    fflush(ui->terminal);
}

/* Query only the cursor row, keeping keystrokes that arrive with the reply.
 * This lets inline mode protect existing shell output without reading it. */
static int inline_cursor_row(ds4_cli_ui *ui) {
    struct termios original, raw;
    if (tcgetattr(STDIN_FILENO, &original)) return 1;
    raw = original;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw)) return 1;
    fputs("\x1b[6n", ui->terminal);
    fflush(ui->terminal);
    double deadline = now_sec() + .15;
    int row = 1;
    while (now_sec() < deadline && ui->start_input_len < sizeof(ui->start_input) - 1) {
        struct pollfd input = {.fd = STDIN_FILENO, .events = POLLIN};
        int ready = poll(&input, 1, larger(0, (int)((deadline - now_sec()) * 1000)) + 1);
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0 || !(input.revents & POLLIN)) break;
        char c;
        if (read(STDIN_FILENO, &c, 1) != 1) break;
        ui->start_input[ui->start_input_len++] = c;
        ui->start_input[ui->start_input_len] = '\0';
        if (c != 'R') continue;
        bool found = false;
        for (size_t i = 0; i < ui->start_input_len; i++) {
            int r, column, used = 0;
            if (sscanf(ui->start_input + i, "\x1b[%5d;%5dR%n", &r, &column, &used) == 2 &&
                used && i + (size_t)used == ui->start_input_len && r >= 1 && r <= ui->rows &&
                column >= 1 && column <= ui->columns) {
                row = r;
                ui->start_input_len = i;
                found = true;
                break;
            }
        }
        if (found) break;
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &original);
    return row;
}

void ds4_cli_ui_start(ds4_cli_ui *ui, bool no_alt_screen, const char *title) {
    memset(ui, 0, sizeof(*ui));
    ui->title = title;
    linenoiseSetMultiLine(1);
    struct winsize size = {0};
    if (!isatty(STDIN_FILENO) || !cli_ui_terminal(stdout) ||
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) != 0 || size.ws_row < 4 || size.ws_col < 12)
        return;
    ui->active = true;
    ui->alt_screen = !no_alt_screen;
    ui->rows = size.ws_row;
    ui->columns = size.ws_col;
    ui->logo_pending = ui->alt_screen;
    ui->braille = ui->alt_screen && braille_supported();
    ui->last_kind = -1;
    ui->trailing_newlines = 1;
    fflush(stdout);
    fflush(stderr);
    ui->terminal = stdout;
    ui->saved_errors = stderr;
    ui->content = content_stream(ui, false);
    if (!ui->content) { ui->active = false; return; }
    linenoiseSetHintsCallback(input_hint);
    stdout = ui->content;
    ds4_cli_ui_output_start(&ui->system_output, stdout);
    ds4_cli_ui_output_start(&ui->text_output, stdout);
    if (cli_ui_terminal(stderr)) {
        ui->errors = content_stream(ui, true);
        if (ui->errors) stderr = ui->errors;
    }
    painting++;
    if (ui->alt_screen) {
        fputs("\x1b[?1049h", stdout);
        set_layout(ui, stdout, EMPTY_INPUT_ROWS, 1);
    }
    else {
        cli_ui_style s = cli_ui_make_style(stdout);
        fprintf(stdout, "  %s%s%s\n", s.accent, ui->title, s.off);
        int reserved = EMPTY_INPUT_ROWS + 1 + (ui->rows >= 7 ? INPUT_GAP : 0);
        for (int i = 0; i < reserved; i++) fputs("\r\n", stdout);
        fprintf(stdout, "\x1b[%dA\r", reserved);
        ui->content_row = inline_cursor_row(ui);
        ui->inline_prefix = ui->content_row - 1;
        fputs("\0337", stdout);
        ui->output_bottom = larger(1, ui->rows - reserved);
        ui->prompt_row = ui->output_bottom + 1 + (ui->rows >= 7 ? INPUT_GAP : 0);
        fprintf(stdout, "\x1b[1;%dr", ui->output_bottom);
    }
    // Disambiguate modified Enter while retaining legacy-terminal fallbacks.
    fputs("\x1b[>4;2m\x1b[>1u", stdout);
    mouse_tracking(ui, stdout, true);
    fflush(stdout);
    painting--;
}

int ds4_cli_ui_edit_begin(ds4_cli_ui *ui, struct linenoiseState *edit,
                         char *buffer, size_t length, const char *footer) {
    set_busy(ui, false);
    ui->external_footer = footer != NULL;
    painting++;
    set_layout(ui, stdout, EMPTY_INPUT_ROWS, 1);
    cursor(stdout, ui->prompt_row, 1);
    fflush(stdout);
    if (linenoiseEditStart(edit, STDIN_FILENO, STDOUT_FILENO, buffer, length,
                           "  ") != 0) {
        painting--;
        return -1;
    }
    ui->edit = edit;
    edit->buflen_max = 1024 * 1024;
    bool color = ds4_cli_ui_color_enabled(stdout);
    linenoiseEditSetInputStyle(edit, color ? user_background : "",
                               color ? "\x1b[0m" : "", INPUT_PADDING);
    if (footer) linenoiseEditSetStatus(edit, footer, color ? "\x1b[97m" : "", color ? "\x1b[0m" : "");
    update_size(ui, edit);
    linenoiseEditSetLayoutCallback(edit, input_layout, ui);
    linenoiseEditSetMouseCallback(edit, logo_click, ui);
    if (ui->start_input_len) {
        linenoiseEditQueueInput(edit, ui->start_input, ui->start_input_len);
        ui->start_input_len = 0;
    }
    linenoiseShow(edit);
    fputs("\x1b[?25h", ui->terminal);
    fflush(ui->terminal);
    painting--;
    return 0;
}

void ds4_cli_ui_edit_tick(ds4_cli_ui *ui, struct linenoiseState *edit) {
    painting++;
    if (ui->animation_start && (now_sec() - ui->frame_time) * 1000 >= LOGO_FRAME_MS)
        linenoiseShow(edit);
    struct winsize size = {0};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 &&
        ((size.ws_row && size.ws_row != ui->rows) ||
         (size.ws_col && size.ws_col != ui->columns))) {
        linenoiseHide(edit);
        erase_logo(ui, stdout);
        clear_rows(stdout, ui->prompt_row, ui->rows);
        if (ui->logo_pending) clear_rows(stdout, 1, size.ws_row);
        ui->output_bottom = 0; // Reinstall scroll margins after resizing.
        fflush(stdout);
        update_size(ui, edit);
        linenoiseShow(edit);
    }
    painting--;
}

void ds4_cli_ui_edit_end(ds4_cli_ui *ui, struct linenoiseState *edit, bool submitted) {
    painting++;
    linenoiseHide(edit);
    erase_logo(ui, stdout);
    fflush(stdout);
    linenoiseEditStop(edit);
    ui->edit = NULL;
    if (submitted) ui->logo_pending = false;
    set_layout(ui, stdout, EMPTY_INPUT_ROWS, 1);
    show_idle_input(ui);
    painting--;
}

void ds4_cli_ui_print_user(ds4_cli_ui *ui, const char *text) {
    ui->logo_pending = false;
    painting++;
    erase_logo(ui, stdout);
    painting--;
    if (ui->scroll_offset) {
        ui->scroll_offset = 0;
        painting++;
        repaint_content(ui, stdout);
        park_cursor(ui);
        painting--;
    }
    ui->message_kind = CONTENT_USER;
    print_user_message(stdout, text, ui->columns);
    ui->message_kind = CONTENT_TEXT;
    ds4_cli_ui_output_start(&ui->text_output, stdout);
}

char *ds4_cli_ui_readline(ds4_cli_ui *ui, const ds4_cli_ui_state *state) {
    if (!ui->active) return linenoise("");
    ui->state = *state;
    struct linenoiseState edit;
    char *buffer = malloc(4096);
    if (!buffer) { errno = ENOMEM; return NULL; }
    if (ds4_cli_ui_edit_begin(ui, &edit, buffer, 4096, NULL)) { free(buffer); return NULL; }
    char *line = NULL;
    for (;;) {
        ds4_cli_ui_edit_tick(ui, &edit);
        struct pollfd input = {.fd = STDIN_FILENO, .events = POLLIN};
        int ready = linenoiseEditQueuedInput(&edit) ? 1 : poll(&input, 1, ui->animation_start ? LOGO_FRAME_MS : 100);
        if (ready < 0) { line = NULL; break; }
        if (!ready) continue;
        line = linenoiseEditFeed(&edit);
        if (line != linenoiseEditMore) break;
    }
    int saved_errno = errno;
    bool submitted = line && line[strspn(line, " \t\r\n")];
    ds4_cli_ui_edit_end(ui, &edit, submitted);
    free(edit.buf);
    if (line && line[0]) ds4_cli_ui_print_user(ui, line);
    if (submitted) set_busy(ui, true);
    fflush(stdout);
    errno = saved_errno;
    return line;
}

void ds4_cli_ui_stop(ds4_cli_ui *ui) {
    if (!ui->active) return;
    set_busy(ui, false);
    fputs("\x1b[?25h", ui->terminal);
    ds4_cli_ui_output_finish(&ui->system_output);
    linenoiseSetHintsCallback(NULL);
    painting++;
    fputs("\x1b[<u\x1b[>4m", stdout);
    mouse_tracking(ui, stdout, false);
    if (ui->alt_screen) {
        fputs("\x1b[r\x1b[?1049l", stdout);
    } else {
        if (ui->scroll_offset) { ui->scroll_offset = 0; repaint_content(ui, stdout); }
        clear_rows(stdout, ui->output_bottom + 1, ui->rows);
        fputs("\x1b[r\0338\x1b[0m", stdout);
        size_t lines = ui->pending_newlines ? ui->pending_newlines : 1;
        for (size_t i = 0; i < lines; i++) fputs("\r\n", stdout);
    }
    fflush(stdout);
    painting--;
    stdout = ui->terminal;
    stderr = ui->saved_errors;
    if (ui->errors) fclose(ui->errors);
    fclose(ui->content);
    free(ui->history);
    ui->history = NULL;
    ui->active = false;
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
