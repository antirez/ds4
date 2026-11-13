/* Regression test for Alt/Option+arrow word-motion in linenoise.
 *
 * On macOS, Option+Left sends ESC b and Option+Right sends ESC f
 * (readline's backward-word / forward-word). Many Linux terminals instead
 * send CSI/SS3 modified-arrow sequences such as ESC [ 1 ; 3 C / D
 * (Alt+Right / Alt+Left) or ESC O 1 ; 3 D, where modifier 3 = Alt/Option and
 * 4 = Alt+Shift. These must move the cursor by whole words, not insert a
 * literal 'b'/'f' or move one character.
 *
 * This test drives linenoiseEditFeed through the multiplexed API using pipes
 * plus the LINENOISE_ASSUME_TTY / LINENOISE_COLS test mode, so it runs
 * without a real terminal. Build: make tests/test_linenoise_wordnav
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "linenoise.h"

static int failures = 0;

static void check(const char *what, struct linenoiseState *l, size_t pos, const char *buf) {
    if (l->pos != pos || strcmp(l->buf, buf) != 0) {
        failures++;
        fprintf(stderr, "FAIL %-28s got pos=%zu buf='%s'  want pos=%zu buf='%s'\n",
                what, l->pos, l->buf, pos, buf);
    }
}

static void feed_all(struct linenoiseState *l, const char *seq) {
    linenoiseEditQueueInput(l, seq, strlen(seq));
    while (linenoiseEditQueuedInput(l)) {
        char *r = linenoiseEditFeed(l);
        if (r != linenoiseEditMore) {
            if (r) free(r);
            return;
        }
    }
}

int main(void) {
    char buf[256];
    struct linenoiseState l;
    int pin[2], pout[2];

    if (pipe(pin) < 0 || pipe(pout) < 0) { perror("pipe"); return 2; }
    setenv("LINENOISE_ASSUME_TTY", "1", 1);
    setenv("LINENOISE_COLS", "80", 1);

    if (linenoiseEditStart(&l, pin[0], pout[1], buf, sizeof(buf), "> ") != 0) { perror("start"); return 2; }

    feed_all(&l, "alpha beta gamma");
    check("typed", &l, 16, "alpha beta gamma");

    /* ESC b / ESC f standalone (macOS Option+Left/Right, readline meta-b/f). */
    feed_all(&l, "\x1b""b");
    check("ESC b (prev word)", &l, 11, "alpha beta gamma");
    feed_all(&l, "\x1b""b");
    check("ESC b (prev word)", &l, 6, "alpha beta gamma");
    feed_all(&l, "\x1b""b");
    check("ESC b (prev word)", &l, 0, "alpha beta gamma");

    feed_all(&l, "\x1b""f");
    check("ESC f (next word)", &l, 6, "alpha beta gamma");
    feed_all(&l, "\x1b""f");
    check("ESC f (next word)", &l, 11, "alpha beta gamma");
    feed_all(&l, "\x1b""f");
    check("ESC f (no next word)", &l, 16, "alpha beta gamma");

    /* CSI modified arrows (Alt+Left/Right on many Linux terminals). */
    feed_all(&l, "\x1b[1;3D");
    check("ESC [ 1;3D (Alt+Left)", &l, 11, "alpha beta gamma");
    feed_all(&l, "\x1b[1;3C");
    check("ESC [ 1;3C (Alt+Right)", &l, 16, "alpha beta gamma");

    /* SS3 modified arrow form. */
    feed_all(&l, "\x1bO1;3D");
    check("ESC O 1;3D (Alt+Left)", &l, 11, "alpha beta gamma");

    /* Alt+Shift variants (modifier 4). */
    feed_all(&l, "\x1b[1;4C");
    check("ESC [ 1;4C (Alt+Shift+Right)", &l, 16, "alpha beta gamma");
    feed_all(&l, "\x1b[1;4D");
    check("ESC [ 1;4D (Alt+Shift+Left)", &l, 11, "alpha beta gamma");

    /* Plain arrows still move one character (not affected by the fix). */
    feed_all(&l, "\x1b[D");
    check("plain Left (one char)", &l, 10, "alpha beta gamma");
    feed_all(&l, "\x1b[C");
    check("plain Right (one char)", &l, 11, "alpha beta gamma");

    /* Ctrl (5) and iTerm Meta (9) modified arrows. */
    feed_all(&l, "\x1b[1;5D");
    check("ESC [ 1;5D (Ctrl+Left)", &l, 6, "alpha beta gamma");
    feed_all(&l, "\x1b[1;9C");
    check("ESC [ 1;9C (Meta+Right)", &l, 11, "alpha beta gamma");

    /* Uppercase meta variants. */
    feed_all(&l, "\x1b""B");
    check("ESC B", &l, 6, "alpha beta gamma");
    feed_all(&l, "\x1b""F");
    check("ESC F", &l, 11, "alpha beta gamma");

    /* Option+Backspace (ESC DEL) deletes the previous word. */
    feed_all(&l, "\x1b\x7f");
    check("ESC DEL (delete prev word)", &l, 6, "alpha gamma");

    linenoiseEditStop(&l);
    close(pin[0]); close(pin[1]);
    close(pout[0]); close(pout[1]);

    /* Fresh editor for multiline and UTF-8 buffers. */
    if (pipe(pin) < 0 || pipe(pout) < 0) { perror("pipe"); return 2; }
    if (linenoiseEditStart(&l, pin[0], pout[1], buf, sizeof(buf), "> ") != 0) { perror("start"); return 2; }

    /* A multiline paste is folded into one atomic unit by the editor, so word
     * motion must step over it as a whole rather than into its hidden lines. */
    feed_all(&l, "a \x1b[200~one\ntwo\x1b[201~ z");
    check("folded paste typed", &l, 11, "a one\ntwo z");
    feed_all(&l, "\x1b""b");
    check("prev word before fold", &l, 10, "a one\ntwo z");
    feed_all(&l, "\x1b""b");
    check("prev word skips fold atomically", &l, 2, "a one\ntwo z");
    feed_all(&l, "\x1b""b");
    check("prev word to start", &l, 0, "a one\ntwo z");
    feed_all(&l, "\x1b""f");
    check("next word", &l, 2, "a one\ntwo z");
    feed_all(&l, "\x1b""f");
    check("next word skips fold atomically", &l, 10, "a one\ntwo z");

    linenoiseEditStop(&l);
    close(pin[0]); close(pin[1]);
    close(pout[0]); close(pout[1]);
    if (pipe(pin) < 0 || pipe(pout) < 0) { perror("pipe"); return 2; }
    if (linenoiseEditStart(&l, pin[0], pout[1], buf, sizeof(buf), "> ") != 0) { perror("start"); return 2; }

    /* Multibyte characters must not be split. */
    feed_all(&l, "one \xc3\xa5\xc3\xa4\xc3\xb6 end");
    check("utf-8 typed", &l, 14, "one \xc3\xa5\xc3\xa4\xc3\xb6 end");
    feed_all(&l, "\x1b""b");
    check("utf-8 prev word (end)", &l, 11, "one \xc3\xa5\xc3\xa4\xc3\xb6 end");
    feed_all(&l, "\x1b""b");
    check("utf-8 prev word (multibyte)", &l, 4, "one \xc3\xa5\xc3\xa4\xc3\xb6 end");
    feed_all(&l, "\x1b""f");
    check("utf-8 next word", &l, 11, "one \xc3\xa5\xc3\xa4\xc3\xb6 end");

    linenoiseEditStop(&l);
    close(pin[0]); close(pin[1]);
    close(pout[0]); close(pout[1]);

    if (failures) { fprintf(stderr, "%d FAILURES\n", failures); return 1; }
    printf("linenoise word-motion: ALL PASS\n");
    return 0;
}
