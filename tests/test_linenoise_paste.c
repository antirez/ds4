/* Unit tests for the paste and multi row handling of the bundled linenoise
 * fork: terminal geometry with hard newlines, numbered placeholders, expanding
 * a placeholder by pasting the same text again, and the inline hint.
 *
 * The editor is driven exactly like ds4-agent drives it: multi line mode on,
 * whole terminal byte sequences queued with linenoiseEditQueueInput() and
 * consumed by linenoiseEditFeed(). Rendering is a private detail of the line
 * editor, so the implementation is included instead of linked.
 *
 * Pure C99: no model, no GPU, no terminal. LINENOISE_ASSUME_TTY replaces the
 * isatty() gates, LINENOISE_COLS/LINENOISE_ROWS the terminal size, a non
 * blocking pipe stands in for the keyboard and a temporary file for the
 * screen, so what the editor paints can be read back. */

#include "../linenoise.c"

#include <fcntl.h>

static int g_failed = 0;
static int g_total  = 0;

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__);            \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

#define RUN(fn) do {                                                           \
    fprintf(stderr, "RUN: %s\n", #fn);                                         \
    int _before = g_failed;                                                    \
    (fn)();                                                                    \
    fprintf(stderr, "  %s\n", (_before == g_failed) ? "ok" : "FAIL");          \
} while (0)

/* Compare against an expected string, showing both sides when they differ:
 * a wrong placeholder is unreadable as a bare boolean. */
static void check_str(const char *got, const char *expected, const char *msg) {
    g_total++;
    if (got == NULL || strcmp(got, expected) != 0) {
        fprintf(stderr, "  FAIL: %s\n", msg);
        fprintf(stderr, "  -- expected --\n%s\n", expected);
        fprintf(stderr, "  -- got --\n%s\n", got ? got : "(null)");
        g_failed++;
    }
}

/* ============================ Editor fixture ============================== */

struct editor {
    struct linenoiseState l;
    int pipefd[2];
    FILE *screen;   /* Temporary file collecting everything painted. */
};

static void editor_open(struct editor *e) {
    char *buf = malloc(LINENOISE_INITIAL_BUFLEN);

    if (buf == NULL || pipe(e->pipefd) == -1) abort();
    /* A starved read must report EAGAIN so linenoiseEditFeed() returns
     * linenoiseEditMore instead of blocking the test forever. */
    fcntl(e->pipefd[0], F_SETFL, O_NONBLOCK);
    e->screen = tmpfile();
    if (e->screen == NULL) abort();
    linenoiseEditStart(&e->l, e->pipefd[0], fileno(e->screen), buf,
                       LINENOISE_INITIAL_BUFLEN, "> ");
    /* Let the buffer grow like the blocking API does, so a large paste is
     * limited by PASTE_MAX_BYTES and not by the initial allocation. */
    e->l.buflen_max = LINENOISE_MAX_LINE;
}

static void editor_close(struct editor *e) {
    linenoiseEditStop(&e->l);
    free(e->l.buf);
    close(e->pipefd[0]);
    close(e->pipefd[1]);
    fclose(e->screen);
}

/* Forget what has been painted so far, so the next assertion only sees the
 * bytes produced by the next keystroke. */
static void editor_screen_clear(struct editor *e) {
    int fd = fileno(e->screen);

    if (ftruncate(fd, 0) == -1) abort();
    if (lseek(fd, 0, SEEK_SET) == (off_t)-1) abort();
}

/* Everything painted since the last clear, nul terminated. Caller frees. */
static char *editor_screen_text(struct editor *e) {
    int fd = fileno(e->screen);
    off_t end = lseek(fd, 0, SEEK_END);
    char *out;

    if (end < 0) abort();
    out = malloc((size_t)end+1);
    if (out == NULL) abort();
    if (lseek(fd, 0, SEEK_SET) == (off_t)-1) abort();
    if (read(fd, out, (size_t)end) != (ssize_t)end) abort();
    if (lseek(fd, 0, SEEK_END) == (off_t)-1) abort();
    out[end] = '\0';
    return out;
}

/* Feed raw terminal bytes and let the editor consume all of them. Return the
 * submitted line if the bytes contained an ENTER, otherwise NULL. */
static char *editor_feed(struct editor *e, const char *data, size_t len) {
    if (linenoiseEditQueueInput(&e->l, data, len) == -1) abort();
    while (linenoiseEditQueuedInput(&e->l) > 0) {
        char *res = linenoiseEditFeed(&e->l);
        if (res != linenoiseEditMore) return res;
    }
    return NULL;
}

static char *editor_feed_str(struct editor *e, const char *s) {
    return editor_feed(e, s, strlen(s));
}

/* Send text the way a terminal in bracketed paste mode does. */
static void editor_paste(struct editor *e, const char *text, size_t len) {
    static const char START[] = "\x1b[200~", END[] = "\x1b[201~";
    size_t startlen = sizeof(START)-1, endlen = sizeof(END)-1;
    char *seq = malloc(startlen+len+endlen);

    if (seq == NULL) abort();
    memcpy(seq, START, startlen);
    memcpy(seq+startlen, text, len);
    memcpy(seq+startlen+len, END, endlen);
    editor_feed(e, seq, startlen+len+endlen);
    free(seq);
}

static void editor_paste_str(struct editor *e, const char *text) {
    editor_paste(e, text, strlen(text));
}

/* What the multi row refresh paints for the current buffer, with hard newlines
 * kept as real row breaks. Caller frees. */
static char *editor_render(struct editor *e, size_t *outpos) {
    char *render = NULL;
    size_t len = 0, pos = 0;

    if (linenoiseRenderBuffer(&e->l, &render, &len, &pos, 0) == -1) abort();
    if (outpos) *outpos = pos;
    return render;
}

/* The same, for the renderers confined to one row: newlines become markers. */
static char *editor_render_marked(struct editor *e, size_t *outpos) {
    char *render = NULL;
    size_t len = 0, pos = 0;

    if (linenoiseRenderBuffer(&e->l, &render, &len, &pos, 1) == -1) abort();
    if (outpos) *outpos = pos;
    return render;
}

/* Row and column of the cursor for the current buffer, exactly as
 * refreshMultiLine() computes them. */
static void editor_cursor(struct editor *e, int *row, int *col) {
    size_t pos = 0;
    char *render = editor_render(e, &pos);
    int pending;

    linenoiseGeometry(render, pos, utf8StrWidth(e->l.prompt, e->l.plen),
                      e->l.cols, row, col, &pending);
    free(render);
}

/* linenoiseBeep() writes BEL to stderr, where the test also reports its
 * failures: swallow it around the cases that expect one, so a suite run stays
 * readable and the terminal stays quiet. */
static int beep_saved_fd = -1;

static void beeps_mute(void) {
    int devnull = open("/dev/null", O_WRONLY);

    fflush(stderr);
    beep_saved_fd = dup(STDERR_FILENO);
    if (devnull != -1) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
    }
}

static void beeps_unmute(void) {
    fflush(stderr);
    if (beep_saved_fd == -1) return;
    dup2(beep_saved_fd, STDERR_FILENO);
    close(beep_saved_fd);
    beep_saved_fd = -1;
}

/* ============================ Text builders =============================== */

/* Build 'count' lines named after their index and padded to 'width' chars, so
 * that no two pastes in a test are ever the same text. */
static char *make_lines(int count, int width, int trailing_newline) {
    size_t cap = (size_t)count*((size_t)width+16)+2;
    char *s = malloc(cap);
    size_t n = 0;
    int i;

    if (s == NULL) abort();
    for (i = 0; i < count; i++) {
        int k = snprintf(s+n, cap-n, "line %d", i);
        while (k < width) s[n+k++] = '.';
        n += (size_t)k;
        if (i+1 < count || trailing_newline) s[n++] = '\n';
    }
    s[n] = '\0';
    return s;
}

/* The expected render of plain text: hard newlines become markers. */
static char *make_marked(const char *s) {
    size_t len = strlen(s), n = 0, i;
    char *out = malloc(len*LINENOISE_NEWLINE_MARKER_LEN+1);

    if (out == NULL) abort();
    for (i = 0; i < len; i++) {
        if (s[i] == '\n') {
            memcpy(out+n, LINENOISE_NEWLINE_MARKER, LINENOISE_NEWLINE_MARKER_LEN);
            n += LINENOISE_NEWLINE_MARKER_LEN;
        } else {
            out[n++] = s[i];
        }
    }
    out[n] = '\0';
    return out;
}

static char *make_repeated(char c, size_t len) {
    char *s = malloc(len+1);

    if (s == NULL) abort();
    memset(s, c, len);
    s[len] = '\0';
    return s;
}

/* ============================ Geometry tests ============================== */

static void check_geom(const char *render, size_t stop, size_t pwidth,
                       size_t cols, int erow, int ecol, int epending,
                       const char *msg) {
    int row = 0, col = 0, pending = 0;

    linenoiseGeometry(render, stop, pwidth, cols, &row, &col, &pending);
    g_total++;
    if (row != erow || col != ecol || pending != epending) {
        fprintf(stderr, "  FAIL: %s\n", msg);
        fprintf(stderr, "  -- expected -- row %d col %d pending %d\n",
                erow, ecol, epending);
        fprintf(stderr, "  -- got --      row %d col %d pending %d\n",
                row, col, pending);
        g_failed++;
    }
}

/* Where the cursor lands once hard newlines paint real rows. Columns are 10
 * wide here so the arithmetic stays readable. */
static void test_geometry_rows_and_columns(void) {
    /* No newline: the prompt simply shifts the first row. */
    check_geom("abc", 3, 2, 10, 1, 5, 0, "short text");
    check_geom("", 0, 2, 10, 1, 2, 0, "empty buffer");
    check_geom("abcdefgh", 8, 2, 10, 2, 0, 1, "text ending on the margin");
    check_geom("abcdefghi", 9, 2, 10, 2, 1, 0, "text past the margin");
    check_geom("0123456789012345678901234", 25, 0, 10, 3, 5, 0, "several wraps");
    check_geom("", 0, 10, 10, 2, 0, 1, "prompt ending on the margin");
    check_geom("abcdef", 6, 8, 10, 2, 4, 0, "prompt pushes the first row");

    /* Newlines: one row each, and none of them is skipped. */
    check_geom("abc\n", 4, 2, 10, 2, 0, 0, "trailing newline opens a row");
    check_geom("a\n\nb", 4, 2, 10, 3, 1, 0, "empty line keeps its row");
    check_geom("ab\ncd", 3, 2, 10, 2, 0, 0, "cursor right after a newline");
    check_geom("ab\ncd", 4, 2, 10, 2, 1, 0, "cursor inside the second row");
    check_geom("ab\ncd", 5, 2, 10, 2, 2, 0, "cursor at the end");
    check_geom("a\nb\nc", 5, 2, 10, 3, 1, 0, "three rows");

    /* A newline right after text that filled a row resolves the deferred
     * wrap: the terminal is already there, it must not drop another row. */
    check_geom("abcdefgh\n", 9, 2, 10, 2, 0, 0, "newline after a full row");
    check_geom("abcdefgh\nx", 10, 2, 10, 2, 1, 0, "text after a full row");

    /* Widths, not bytes, drive the wrapping. */
    check_geom("\344\270\255\344\270\255\344\270\255\344\270\255\344\270\255",
               15, 0, 10, 2, 0, 1, "five wide chars fill a row");
    check_geom("\344\270\255\344\270\255\344\270\255\344\270\255\344\270\255"
               "\344\270\255", 18, 0, 10, 2, 2, 0, "six wide chars wrap");
}

/* The geometry must reproduce the flat width arithmetic the editor used
 * before newlines became rows: on newline free input the two models can only
 * agree, and any drift there would move every cursor in the common case. */
static void test_geometry_matches_flat_formulas(void) {
    static const size_t pwidths[] = {1, 2, 5, 12, 79};
    static const size_t lens[] = {0, 1, 5, 68, 74, 78, 79, 80, 81, 160, 161};
    size_t cols = 80;
    char text[200];
    size_t pi, li, k;
    int mismatch = 0;

    memset(text, 'x', sizeof(text));
    for (pi = 0; pi < sizeof(pwidths)/sizeof(*pwidths); pi++) {
        size_t pwidth = pwidths[pi];
        for (li = 0; li < sizeof(lens)/sizeof(*lens); li++) {
            size_t len = lens[li];
            /* Interesting cursors: both ends, the middle, and the byte that
             * lands exactly on the right margin. */
            size_t stops[4];
            stops[0] = 0;
            stops[1] = len/2;
            stops[2] = len;
            stops[3] = cols > pwidth && cols-pwidth <= len ? cols-pwidth : len;
            for (k = 0; k < 4; k++) {
                size_t stop = stops[k];
                int end_row, end_col, end_pending;
                int cur_row, cur_col, cur_pending;
                int rows, wrap, old_rows, old_rpos2, old_col, old_wrap;

                linenoiseGeometry(text, len, pwidth, cols,
                                  &end_row, &end_col, &end_pending);
                linenoiseGeometry(text, stop, pwidth, cols,
                                  &cur_row, &cur_col, &cur_pending);
                /* Mirrors how refreshMultiLine() derives its numbers. */
                rows = end_row - (end_pending ? 1 : 0);
                wrap = end_pending && stop == len && stop != 0;
                if (wrap) rows++;

                old_rows = (int)((pwidth+len+cols-1)/cols);
                old_rpos2 = (int)((pwidth+stop+cols)/cols);
                old_col = (int)((pwidth+stop)%cols);
                old_wrap = stop && stop == len && (pwidth+stop)%cols == 0;
                if (old_wrap) old_rows++;

                if (rows == old_rows && cur_row == old_rpos2 &&
                    cur_col == old_col && wrap == old_wrap) continue;
                if (!mismatch)
                    fprintf(stderr, "  FAIL: pwidth %zu len %zu stop %zu: "
                            "rows %d/%d rpos2 %d/%d col %d/%d wrap %d/%d\n",
                            pwidth, len, stop, rows, old_rows, cur_row,
                            old_rpos2, cur_col, old_col, wrap, old_wrap);
                mismatch++;
            }
        }
    }
    g_total++;
    if (mismatch) {
        fprintf(stderr, "  FAIL: %d newline free geometry mismatches\n", mismatch);
        g_failed++;
    }
}

/* ================================ Tests =================================== */

/* A short single line paste is ordinary text: no placeholder. */
static void test_short_paste_is_inline(void) {
    struct editor e;
    char *render;

    editor_open(&e);
    editor_paste_str(&e, "hello world");
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_count == 0, "short paste is not folded");
    CHECK(e.l.len == 11, "short paste is fully inserted");
    check_str(render, "hello world", "short paste render");
    free(render);
    editor_close(&e);
}

/* A few pasted lines stay visible and occupy real terminal rows. */
static void test_multiline_paste_is_inline(void) {
    struct editor e;
    char *render, *screen;
    size_t pos = 0;
    int row = 0, col = 0;

    editor_open(&e);
    editor_screen_clear(&e);
    editor_paste_str(&e, "a\nb");
    render = editor_render(&e, &pos);
    CHECK(e.l.fold_count == 0, "three byte paste is not folded");
    CHECK(e.l.len == 3, "newlines are kept in the edit buffer");
    check_str(render, "a\nb", "newlines survive rendering");
    CHECK(pos == 3, "render position is a plain byte offset");
    free(render);

    editor_cursor(&e, &row, &col);
    CHECK(row == 2 && col == 1, "cursor sits on the second row");

    /* Raw mode has no output post processing: a bare LF would keep the
     * column, so the refresh has to write CR LF. */
    screen = editor_screen_text(&e);
    CHECK(strstr(screen, "\r\n") != NULL, "hard newlines are painted as CR LF");
    CHECK(strstr(screen, LINENOISE_NEWLINE_MARKER) == NULL,
          "multi row refresh paints no newline marker");
    free(screen);

    /* Moving left over 'b' must land at the start of the second row. */
    editor_feed_str(&e, "\x1b[D");
    editor_cursor(&e, &row, &col);
    CHECK(e.l.pos == 2, "cursor moved one byte left");
    CHECK(row == 2 && col == 0, "cursor is at the start of the second row");
    editor_close(&e);
}

/* Ctrl+J inserts a newline, which is the only way to type one: ENTER submits.
 * It must behave exactly like a pasted newline. */
static void test_ctrl_j_inserts_a_newline(void) {
    struct editor e;
    char *render, *screen;
    int row = 0, col = 0;

    editor_open(&e);
    editor_screen_clear(&e);
    editor_feed_str(&e, "ab\ncd");
    render = editor_render(&e, NULL);
    CHECK(e.l.len == 5, "the newline is stored in the buffer");
    check_str(render, "ab\ncd", "typed newline renders as a row break");
    free(render);
    editor_cursor(&e, &row, &col);
    CHECK(row == 2 && col == 2, "typing continues on the second row");
    screen = editor_screen_text(&e);
    CHECK(strstr(screen, "\r\n") != NULL, "typed newline is painted as CR LF");
    free(screen);

    /* Editing across the row break must not need any special handling. */
    editor_feed_str(&e, "\x7f\x7f\x7f"); /* Three backspaces. */
    CHECK(e.l.len == 2, "backspace deletes across the row break");
    editor_cursor(&e, &row, &col);
    CHECK(row == 1 && col == 4, "the prompt is back to one row");
    editor_close(&e);
}

/* The single row renderers cannot paint a row break, so there the newline
 * shows up as a marker. Both binaries run in multi line mode, but linenoise
 * still supports the single line one. */
static void test_single_line_render_uses_markers(void) {
    struct editor e;
    char *render, *screen;
    size_t pos = 0;

    char *expected = make_marked("a\nb");

    linenoiseSetMultiLine(0);
    editor_open(&e);
    editor_screen_clear(&e);
    editor_paste_str(&e, "a\nb");
    render = editor_render_marked(&e, &pos);
    check_str(render, expected, "newline marker render");
    CHECK(pos == 5, "marker render maps the cursor past the marker");
    free(render);
    free(expected);

    screen = editor_screen_text(&e);
    CHECK(strstr(screen, LINENOISE_NEWLINE_MARKER) != NULL,
          "single row refresh paints the marker");
    CHECK(strstr(screen, "\r\n") == NULL, "single row refresh paints no row break");
    free(screen);
    editor_close(&e);
    linenoiseSetMultiLine(1);
}

/* Long single line pastes fold into a numbered placeholder counting bytes. */
static void test_long_paste_folds_chars(void) {
    struct editor e;
    char *text = make_repeated('x', 300);
    char *render;

    editor_open(&e);
    editor_paste_str(&e, text);
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_count == 1, "long paste is folded");
    CHECK(e.l.len == 300, "folding keeps the real bytes");
    check_str(render, "[Pasted text #1 +300 chars] (paste again to expand)",
              "single line placeholder");
    free(render);
    editor_close(&e);
    free(text);
}

/* Many pasted lines fold into a placeholder counting lines, and a trailing
 * newline closes the last line instead of opening an empty one. */
static void test_multiline_paste_folds_lines(void) {
    struct editor e;
    char *text = make_lines(10, 0, 0);
    char *closed = make_lines(10, 0, 1);
    char *render;

    editor_open(&e);
    editor_paste_str(&e, text);
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_count == 1, "ten lines are folded");
    check_str(render, "[Pasted text #1 +10 lines] (paste again to expand)",
              "multiline placeholder");
    free(render);

    /* Ctrl+U clears the line, and with it the placeholder numbering. */
    editor_feed_str(&e, "\x15");
    editor_paste_str(&e, closed);
    render = editor_render(&e, NULL);
    check_str(render, "[Pasted text #1 +10 lines] (paste again to expand)",
              "trailing newline is not a line");
    free(render);
    editor_close(&e);
    free(text);
    free(closed);
}

/* Numbers follow creation order and never change, even when a later paste
 * lands before an earlier one in the buffer. */
static void test_placeholder_numbers_are_stable(void) {
    struct editor e;
    char *a = make_repeated('a', 250);
    char *b = make_repeated('b', 250);
    char *c = make_repeated('c', 250);
    char *render;

    editor_open(&e);
    editor_paste_str(&e, a);
    editor_paste_str(&e, b);
    editor_feed_str(&e, "\x1b[H"); /* Home. */
    editor_paste_str(&e, c);
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_count == 3, "three folds are remembered");
    CHECK(e.l.fold_id[0] == 3 && e.l.fold_id[1] == 1 && e.l.fold_id[2] == 2,
          "folds are sorted by offset, not by number");
    CHECK(e.l.fold_start[0] == 0 && e.l.fold_start[1] == 250 &&
          e.l.fold_start[2] == 500, "earlier folds moved with the insertion");
    check_str(render,
              "[Pasted text #3 +250 chars] (paste again to expand)"
              "[Pasted text #1 +250 chars]"
              "[Pasted text #2 +250 chars]",
              "stable numbering render");
    free(render);
    editor_close(&e);
    free(a);
    free(b);
    free(c);
}

/* Pasting the same text again reveals it instead of inserting a copy, and a
 * further paste folds the new copy under a fresh number. */
static void test_paste_again_expands(void) {
    struct editor e;
    char *text = make_lines(10, 0, 0);
    size_t len = strlen(text);
    char *render, *expected;
    int row = 0, col = 0;

    editor_open(&e);
    editor_paste_str(&e, text);
    CHECK(e.l.fold_count == 1, "first paste is folded");

    editor_paste_str(&e, text);
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_count == 0, "second paste expands the placeholder");
    CHECK(e.l.len == len, "second paste does not duplicate the text");
    CHECK(e.l.pos == len, "cursor stays at the end of the revealed text");
    check_str(render, text, "expanded render is the text itself");
    free(render);
    editor_cursor(&e, &row, &col);
    CHECK(row == 10, "expanded text occupies one row per line");

    /* With nothing left to match, the same bytes are a new paste. */
    editor_paste_str(&e, text);
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_count == 1, "third paste folds again");
    CHECK(e.l.len == len*2, "third paste is inserted");
    expected = malloc(len+80);
    if (expected == NULL) abort();
    sprintf(expected, "%s[Pasted text #2 +10 lines] (paste again to expand)",
            text);
    check_str(render, expected, "expanded text plus a new placeholder");
    free(expected);
    free(render);
    editor_close(&e);
    free(text);
}

/* Running out of fold slots must not lose the text: the extra paste simply
 * goes in unfolded. */
static void test_paste_past_fold_slots(void) {
    struct editor e;
    int i;

    editor_open(&e);
    for (i = 0; i < LINENOISE_MAX_FOLDS+1; i++) {
        char *text = make_repeated((char)('a'+i), 250);
        editor_paste_str(&e, text);
        free(text);
    }
    CHECK(e.l.fold_count == LINENOISE_MAX_FOLDS, "fold slots are all used");
    CHECK(e.l.len == 250*(LINENOISE_MAX_FOLDS+1), "no paste was dropped");
    CHECK(e.l.buf[0] == 'a' && e.l.buf[e.l.len-1] == 'a'+LINENOISE_MAX_FOLDS,
          "first and last paste are both in the buffer");
    editor_close(&e);
}

/* Expanding must refuse to grow the prompt past the terminal. */
static void test_expand_refused_when_too_tall(void) {
    struct editor e;
    char *text = make_lines(20, 40, 0);
    char *render;

    setenv("LINENOISE_ROWS", "6", 1);
    editor_open(&e);
    editor_paste_str(&e, text);
    CHECK(e.l.fold_count == 1, "twenty lines are folded");

    beeps_mute();
    editor_paste_str(&e, text);
    beeps_unmute();
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_count == 1, "the fold survives a refused expansion");
    CHECK(e.l.len == strlen(text), "a refused expansion inserts nothing");
    check_str(render, "[Pasted text #1 +20 lines] (paste again to expand)",
              "placeholder is unchanged");
    free(render);
    editor_close(&e);
    setenv("LINENOISE_ROWS", "40", 1);
    free(text);
}

/* The guard keeps two rows free, so text needing exactly the remaining rows
 * still expands and one row more does not. */
static void test_expand_fits_at_the_boundary(void) {
    struct editor e;
    char *fits = make_lines(8, 0, 0);
    char *toobig = make_lines(9, 0, 0);

    setenv("LINENOISE_ROWS", "10", 1);
    editor_open(&e);
    editor_paste_str(&e, fits);
    editor_paste_str(&e, fits);
    CHECK(e.l.fold_count == 0, "eight rows fit in ten minus two");
    editor_close(&e);

    editor_open(&e);
    editor_paste_str(&e, toobig);
    beeps_mute();
    editor_paste_str(&e, toobig);
    beeps_unmute();
    CHECK(e.l.fold_count == 1, "nine rows do not fit in ten minus two");
    CHECK(e.l.len == strlen(toobig), "the refused expansion inserts nothing");
    editor_close(&e);

    setenv("LINENOISE_ROWS", "40", 1);
    free(fits);
    free(toobig);
}

/* A paste larger than PASTE_MAX_BYTES is refused with a beep, leaving the
 * edited line untouched. */
static void test_huge_paste_is_refused(void) {
    struct editor e;
    char *text = make_repeated('z', PASTE_MAX_BYTES+16);

    editor_open(&e);
    beeps_mute();
    editor_paste(&e, text, PASTE_MAX_BYTES+16);
    beeps_unmute();
    CHECK(e.l.len == 0, "oversized paste is not inserted");
    CHECK(e.l.fold_count == 0, "oversized paste creates no fold");
    editor_close(&e);
    free(text);
}

/* A recalled history entry is folded without a number: it was not pasted on
 * this line. Submitting it still returns the real text. */
static void test_history_recall_fold(void) {
    struct editor e;
    char *text = make_lines(40, 0, 0);
    char *render, *line;

    linenoiseHistoryAdd(text);
    editor_open(&e);
    editor_feed_str(&e, "\x1b[A"); /* Up. */
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_count == 1, "recalled entry is folded");
    CHECK(e.l.fold_id[0] == 0, "history folds are unnumbered");
    check_str(render, "[... 40 lines ...]", "history placeholder");
    free(render);

    line = editor_feed_str(&e, "\r");
    CHECK(line != NULL, "enter submits the line");
    if (line) check_str(line, text, "submitted text is the real one");
    free(line);
    editor_close(&e);
    free(text);
}

/* The hint is an aid for the paste that just happened: any edit removes it,
 * and so does submitting the line. */
static void test_hint_is_transient(void) {
    struct editor e;
    char *text = make_repeated('x', 300);
    char *render, *line, *expected;

    editor_open(&e);
    editor_paste_str(&e, text);
    CHECK(e.l.fold_hint_id == 1, "the new fold carries the hint");

    editor_feed_str(&e, "y");
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_hint_id == 0, "typing clears the hint");
    check_str(render, "[Pasted text #1 +300 chars]y", "hint free placeholder");
    free(render);

    line = editor_feed_str(&e, "\r");
    CHECK(e.l.fold_hint_id == 0, "submitting clears the hint");
    expected = malloc(strlen(text)+2);
    if (expected == NULL) abort();
    sprintf(expected, "%sy", text);
    if (line) check_str(line, expected, "submitted text is the real one");
    free(line);
    free(expected);
    editor_close(&e);
    free(text);
}

int main(void) {
    /* Closing an editor prints the newline a real terminal needs after the
     * accepted line. The suite reports on stderr, so drop that stray output
     * instead of dotting the test log with blank lines. */
    if (freopen("/dev/null", "w", stdout) == NULL) return 1;
    setenv("LINENOISE_ASSUME_TTY", "1", 1);
    setenv("LINENOISE_COLS", "80", 1);
    setenv("LINENOISE_ROWS", "40", 1);
    /* Both binaries edit in multi line mode: test what they run. */
    linenoiseSetMultiLine(1);

    RUN(test_geometry_rows_and_columns);
    RUN(test_geometry_matches_flat_formulas);
    RUN(test_short_paste_is_inline);
    RUN(test_multiline_paste_is_inline);
    RUN(test_ctrl_j_inserts_a_newline);
    RUN(test_single_line_render_uses_markers);
    RUN(test_long_paste_folds_chars);
    RUN(test_multiline_paste_folds_lines);
    RUN(test_placeholder_numbers_are_stable);
    RUN(test_paste_again_expands);
    RUN(test_paste_past_fold_slots);
    RUN(test_expand_refused_when_too_tall);
    RUN(test_expand_fits_at_the_boundary);
    RUN(test_huge_paste_is_refused);
    RUN(test_history_recall_fold);
    RUN(test_hint_is_transient);

    fprintf(stderr, "\ntest_linenoise_paste: %d/%d checks passed (%d failed)\n",
            g_total - g_failed, g_total, g_failed);
    return g_failed == 0 ? 0 : 1;
}
