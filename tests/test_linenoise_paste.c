/* Unit tests for the paste handling of the bundled linenoise fork: newline
 * markers, numbered placeholders, expanding a placeholder by pasting the same
 * text again, and the inline hint.
 *
 * The editor is driven exactly like ds4-agent drives it: whole terminal byte
 * sequences are queued with linenoiseEditQueueInput() and consumed by
 * linenoiseEditFeed(). Rendering is a private detail of the line editor, so
 * the implementation is included instead of linked.
 *
 * Pure C99: no model, no GPU, no terminal. LINENOISE_ASSUME_TTY replaces the
 * isatty() gates, LINENOISE_COLS/LINENOISE_ROWS the terminal size, a non
 * blocking pipe stands in for the keyboard and /dev/null for the screen. */

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
    int devnull;
};

static void editor_open(struct editor *e) {
    char *buf = malloc(LINENOISE_INITIAL_BUFLEN);

    if (buf == NULL || pipe(e->pipefd) == -1) abort();
    /* A starved read must report EAGAIN so linenoiseEditFeed() returns
     * linenoiseEditMore instead of blocking the test forever. */
    fcntl(e->pipefd[0], F_SETFL, O_NONBLOCK);
    e->devnull = open("/dev/null", O_WRONLY);
    if (e->devnull == -1) abort();
    linenoiseEditStart(&e->l, e->pipefd[0], e->devnull, buf,
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
    close(e->devnull);
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

/* What the terminal would show for the current buffer. Caller frees. */
static char *editor_render(struct editor *e, size_t *outpos) {
    char *render = NULL;
    size_t len = 0, pos = 0;

    if (linenoiseRenderBuffer(&e->l, &render, &len, &pos) == -1) abort();
    if (outpos) *outpos = pos;
    return render;
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

/* A few pasted lines stay visible, with newlines shown as markers. The render
 * position must follow the substitution, not the byte offsets. */
static void test_multiline_paste_is_inline(void) {
    struct editor e;
    char *render;
    size_t pos = 0;

    editor_open(&e);
    editor_paste_str(&e, "a\nb");
    render = editor_render(&e, &pos);
    CHECK(e.l.fold_count == 0, "three byte paste is not folded");
    CHECK(e.l.len == 3, "newlines are kept in the edit buffer");
    check_str(render, "a" LINENOISE_NEWLINE_MARKER "b", "newline marker render");
    CHECK(pos == 5, "cursor maps past the marker");
    free(render);

    /* Moving left over 'b' must land right after the marker. */
    editor_feed_str(&e, "\x1b[D");
    render = editor_render(&e, &pos);
    CHECK(e.l.pos == 2, "cursor moved one byte left");
    CHECK(pos == 4, "cursor maps to the end of the marker");
    free(render);
    editor_close(&e);
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
    char *marked = make_marked(text);
    size_t len = strlen(text);
    char *render, *expected;

    editor_open(&e);
    editor_paste_str(&e, text);
    CHECK(e.l.fold_count == 1, "first paste is folded");

    editor_paste_str(&e, text);
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_count == 0, "second paste expands the placeholder");
    CHECK(e.l.len == len, "second paste does not duplicate the text");
    CHECK(e.l.pos == len, "cursor stays at the end of the revealed text");
    check_str(render, marked, "expanded render uses newline markers");
    free(render);

    /* With nothing left to match, the same bytes are a new paste. */
    editor_paste_str(&e, text);
    render = editor_render(&e, NULL);
    CHECK(e.l.fold_count == 1, "third paste folds again");
    CHECK(e.l.len == len*2, "third paste is inserted");
    expected = malloc(strlen(marked)+80);
    if (expected == NULL) abort();
    sprintf(expected, "%s[Pasted text #2 +10 lines] (paste again to expand)",
            marked);
    check_str(render, expected, "expanded text plus a new placeholder");
    free(expected);
    free(render);
    editor_close(&e);
    free(text);
    free(marked);
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

    RUN(test_short_paste_is_inline);
    RUN(test_multiline_paste_is_inline);
    RUN(test_long_paste_folds_chars);
    RUN(test_multiline_paste_folds_lines);
    RUN(test_placeholder_numbers_are_stable);
    RUN(test_paste_again_expands);
    RUN(test_paste_past_fold_slots);
    RUN(test_expand_refused_when_too_tall);
    RUN(test_huge_paste_is_refused);
    RUN(test_history_recall_fold);
    RUN(test_hint_is_transient);

    fprintf(stderr, "\ntest_linenoise_paste: %d/%d checks passed (%d failed)\n",
            g_total - g_failed, g_total, g_failed);
    return g_failed == 0 ? 0 : 1;
}
