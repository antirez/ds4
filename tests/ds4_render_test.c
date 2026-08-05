/* Unit tests for the streaming markdown renderer.
 *
 * Every corpus is rendered twice, once as a single write and once byte at a
 * time, and both runs must produce the same bytes: the renderer has to reach
 * the same state no matter how the model chunks its tokens. */

#define _POSIX_C_SOURCE 200809L

#include "ds4_render.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

/* Show escapes and newlines so a mismatch is readable in the test log. */
static char *show(const char *s) {
    size_t n = strlen(s);
    char *out = malloc(n * 4 + 1);
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == 0x1b) {
            memcpy(out + k, "\\e", 2);
            k += 2;
        } else if (c == '\n') {
            memcpy(out + k, "\\n", 2);
            k += 2;
        } else {
            out[k++] = (char)c;
        }
    }
    out[k] = '\0';
    return out;
}

static void fail(const char *what, const char *got, const char *want) {
    char *g = show(got);
    char *w = show(want);
    printf("FAIL %s\n  got:  %s\n  want: %s\n", what, g, w);
    free(g);
    free(w);
    g_failures++;
}

static char *render_once(const char *in, size_t len, bool color, bool thinking,
                         bool in_think, int cols, bool bytewise) {
    char *buf = NULL;
    size_t n = 0;
    FILE *fp = open_memstream(&buf, &n);
    if (!fp) {
        printf("FAIL open_memstream\n");
        g_failures++;
        return NULL;
    }
    ds4r r;
    ds4r_init(&r, fp, color, thinking);
    if (in_think) ds4r_set_in_think(&r, true);
    if (cols) ds4r_set_columns(&r, cols);
    if (bytewise) {
        for (size_t i = 0; i < len; i++) ds4r_write(&r, in + i, 1);
    } else {
        ds4r_write(&r, in, len);
    }
    ds4r_finish(&r);
    ds4r_free(&r);
    fclose(fp);
    return buf;
}

/* Render in both chunkings and return the whole-string result. */
static char *render_opt(const char *in, bool color, bool thinking,
                        bool in_think, int cols) {
    size_t len = strlen(in);
    char *whole = render_once(in, len, color, thinking, in_think, cols, false);
    char *split = render_once(in, len, color, thinking, in_think, cols, true);
    if (whole && split && strcmp(whole, split) != 0)
        fail("chunking is not stable", split, whole);
    free(split);
    return whole;
}

static char *render(const char *in) {
    return render_opt(in, true, false, false, 0);
}

static void expect(const char *what, char *got, const char *want) {
    if (!got) return;
    if (strcmp(got, want) != 0) fail(what, got, want);
    free(got);
}

static void expect_contains(const char *what, char *got, const char *needle) {
    if (!got) return;
    if (!strstr(got, needle)) fail(what, got, needle);
    free(got);
}

/* Visible width of one output line, escapes excluded. */
static int line_width(const char *s, int index) {
    int line = 0;
    const char *start = s;
    for (const char *p = s;; p++) {
        if (*p == '\n' || *p == '\0') {
            if (line == index) return ds4r_visible_width(start, (size_t)(p - start));
            if (*p == '\0') return -1;
            line++;
            start = p + 1;
        }
    }
}

/* No escape sequence may be written between the bytes of one UTF-8
 * character. */
static void check_utf8_integrity(const char *what, const char *s) {
    size_t n = strlen(s);
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        size_t need = 1;
        if (c >= 0xc2 && c <= 0xdf) need = 2;
        else if (c >= 0xe0 && c <= 0xef) need = 3;
        else if (c >= 0xf0 && c <= 0xf4) need = 4;
        for (size_t k = 1; k < need; k++) {
            if (i + k >= n || ((unsigned char)s[i + k] & 0xc0) != 0x80) {
                fail(what, s, "unbroken utf-8 sequences");
                return;
            }
        }
        i += need;
    }
}

static void test_inline(void) {
    expect("bold", render("**bold**\n"), "\x1b[1mbold\x1b[0m\n");
    expect("italic", render("*it*\n"), "\x1b[3mit\x1b[0m\n");
    expect("inline code", render("`code`\n"), "\x1b[36mcode\x1b[0m\n");
    expect("literal star", render("a * b\n"), "a * b\n");
    expect("bold inside text",
           render("say **hi** now\n"),
           "say \x1b[1mhi\x1b[0m now\n");
    /* A star run that opens nothing must survive as text. */
    expect("trailing star", render("2 * 3 = 6\n"), "2 * 3 = 6\n");
}

static void test_utf8(void) {
    char *got = render("**中文**测试\n");
    if (got) check_utf8_integrity("utf8 integrity", got);
    expect("cjk bold", got, "\x1b[1m中文\x1b[0m测试\n");
    expect("emoji passthrough", render("ok 🚀 done\n"), "ok 🚀 done\n");
}

static void test_think(void) {
    expect("think block",
           render_opt("<think>hidden</think>visible\n", true, true, false, 0),
           "\x1b[90mhidden\x1b[0m\nvisible\n");
    /* The CLI starts inside the think block when thinking is enabled. */
    expect("implicit think open",
           render_opt("reasoning</think>answer\n", true, true, true, 0),
           "\x1b[90mreasoning\x1b[0m\nanswer\n");
    expect("think without color",
           render_opt("<think>hidden</think>visible\n", false, true, false, 0),
           "hidden\nvisible\n");
    expect("markdown is inert inside think",
           render_opt("<think>**not bold**</think>ok\n", true, true, false, 0),
           "\x1b[90m**not bold**\x1b[0m\nok\n");
}

static void test_headings_lists(void) {
    expect("h1", render("# Title\n"), "\x1b[1m\x1b[4mTitle\x1b[0m\n");
    expect("h3", render("### Sub\n"), "\x1b[1mSub\x1b[0m\n");
    expect("not a heading", render("#tag\n"), "#tag\n");
    expect("bullet", render("- item\n"),
           "\x1b[38;5;244m\xe2\x80\xa2\x1b[0m item\n");
    expect("indented bullet", render("  * item\n"),
           "  \x1b[38;5;244m\xe2\x80\xa2\x1b[0m item\n");
    expect("ordered", render("1. one\n"),
           "\x1b[38;5;244m1.\x1b[0m one\n");
    expect("bold at line start", render("**b** x\n"),
           "\x1b[1mb\x1b[0m x\n");
    expect("quote", render("> hi\n"),
           "\x1b[38;5;244m\xe2\x94\x82 \x1b[0mhi\n");
    expect("nested quote", render(">> hi\n"),
           "\x1b[38;5;244m\xe2\x94\x82 \x1b[0m\x1b[38;5;244m\xe2\x94\x82 \x1b[0mhi\n");
    expect("dash text is not a rule", render("- - x\n"),
           "\x1b[38;5;244m\xe2\x80\xa2\x1b[0m - x\n");

    char *rule = render_opt("---\n", true, false, false, 20);
    expect("horizontal rule", rule,
           "\x1b[38;5;244m"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\x1b[0m\n");
    expect("star rule is a rule", render_opt("***\n", true, false, false, 6),
           "\x1b[38;5;244m\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\x1b[0m\n");
}

static void test_fences(void) {
    expect("plain fence keeps content verbatim",
           render("```\n+--+\n|ab|\n+--+\n```\n"),
           "```\n"
           "\x1b[38;5;75m+--+\x1b[0m\n"
           "\x1b[38;5;75m|ab|\x1b[0m\n"
           "\x1b[38;5;75m+--+\x1b[0m\n"
           "```\n");
    expect("text fence is not highlighted",
           render("```text\n1 + 1\n```\n"),
           "```text\n\x1b[38;5;75m1 + 1\x1b[0m\n```\n");
    expect_contains("language capture highlights keywords",
                    render("```c\nreturn 0;\n```\n"),
                    "\x1b[38;5;214mreturn");
    expect("unterminated fence flushes at finish",
           render("```\nabc"),
           "```\n\x1b[38;5;75mabc\x1b[0m");
    expect("fence closing at finish",
           render("```\nabc\n```"),
           "```\n\x1b[38;5;75mabc\x1b[0m\n```");
    /* Table pipes inside a fence must not start table buffering. */
    expect("pipes inside a fence stay literal",
           render("```\n| a | b |\n```\n"),
           "```\n\x1b[38;5;75m| a | b |\x1b[0m\n```\n");
}

static void test_tables(void) {
    static const char simple[] =
        "| a | bb |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n";
    expect("simple table", render(simple),
           "\x1b[38;5;244m\xe2\x94\x8c\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\xac\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x90\x1b[0m\n"
           "\x1b[38;5;244m\xe2\x94\x82\x1b[0m \x1b[1ma\x1b[22m "
           "\x1b[38;5;244m\xe2\x94\x82\x1b[0m \x1b[1mbb\x1b[22m "
           "\x1b[38;5;244m\xe2\x94\x82\x1b[0m\n"
           "\x1b[38;5;244m\xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\xbc\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\xa4\x1b[0m\n"
           "\x1b[38;5;244m\xe2\x94\x82\x1b[0m 1 "
           "\x1b[38;5;244m\xe2\x94\x82\x1b[0m 2  "
           "\x1b[38;5;244m\xe2\x94\x82\x1b[0m\n"
           "\x1b[38;5;244m\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\xb4\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x98\x1b[0m\n");

    /* Mixed CJK and ASCII: every rendered row must occupy the same columns. */
    static const char cjk[] =
        "| Name | 描述 |\n"
        "| --- | --- |\n"
        "| alpha | 中文说明 |\n"
        "| b | x |\n";
    char *t = render(cjk);
    if (t) {
        int w0 = line_width(t, 0);
        for (int i = 1; i < 6; i++) {
            int w = line_width(t, i);
            if (w != w0) {
                char msg[64];
                snprintf(msg, sizeof(msg), "cjk row %d width %d != %d",
                         i, w, w0);
                fail(msg, t, "equal visible widths");
                break;
            }
        }
        free(t);
    }

    /* Alignment row: left, center, right. */
    static const char aligned[] =
        "| a | b | c |\n"
        "| :--- | :---: | ---: |\n"
        "| xxxx | yyyy | zzzz |\n"
        "| 1 | 2 | 3 |\n";
    expect_contains("left alignment", render(aligned),
                    "\x1b[0m 1    \x1b[38;5;244m");
    expect_contains("center alignment", render(aligned),
                    "\x1b[0m  2   \x1b[38;5;244m");
    expect_contains("right alignment", render(aligned),
                    "\x1b[0m    3 \x1b[38;5;244m");

    expect_contains("bold inside a cell",
                    render("| a |\n| --- |\n| **x** |\n"),
                    "\x1b[1mx\x1b[22m");

    /* Narrow terminal: padding goes first, then the widest column is cut. */
    static const char wide[] =
        "| name | description |\n"
        "| --- | --- |\n"
        "| a | 0123456789 |\n";
    expect_contains("narrow table truncates", render_opt(wide, true, false, false, 16),
                    "\xe2\x80\xa6");
    /* A column that cannot fit at all falls back to the source text. */
    expect("hopeless table falls back to raw",
           render_opt(wide, true, false, false, 4), wide);

    expect("missing separator row falls back to raw",
           render("| a | b |\n| c | d |\n"),
           "| a | b |\n| c | d |\n");

    expect_contains("unterminated table renders at finish",
                    render("| a |\n| --- |\n| 1 |"),
                    "\xe2\x94\x94");

    /* Text after the table returns to the normal path. */
    expect_contains("text after a table", render("| a |\n| --- |\n| 1 |\nafter\n"),
                    "\nafter\n");
}

static void test_table_buffer_cap(void) {
    size_t rows = 600;
    char *in = malloc(rows * 4 + 1);
    size_t k = 0;
    for (size_t i = 0; i < rows; i++) {
        memcpy(in + k, "|a|\n", 4);
        k += 4;
    }
    in[k] = '\0';
    char *got = render(in);
    if (got && strcmp(got, in) != 0)
        fail("table buffer cap passes rows through raw", got, in);
    free(got);
    free(in);
}

static void test_plain_passthrough(void) {
    static const char corpus[] =
        "# Heading\n"
        "Some **bold**, *italic*, `code`, a * b.\n"
        "- bullet\n"
        "1. ordered\n"
        "> quote\n"
        "---\n"
        "| a | b |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "```c\nint main(void) { return 0; }\n```\n"
        "中文 mixed 内容 🚀\n";
    char *got = render_opt(corpus, false, false, false, 0);
    if (got && strcmp(got, corpus) != 0)
        fail("color=false passthrough", got, corpus);
    free(got);
}

static const char g_corpus[] =
    "# Heading 标题\n"
    "Text with **bold**, *italic*, `code`, and a * b.\n"
    "- bullet 中文\n"
    "1. ordered\n"
    "> quote\n"
    "---\n"
    "| a | 描述 |\n"
    "| --- | ---: |\n"
    "| 1 | 中文 |\n"
    "after\n"
    "```c\nint x = 1; /* c */\n```\n"
    "```\n+--+\n| ok |\n+--+\n```\n"
    "tail 🚀\n";

/* Chunk boundaries must not change the result, whatever the tokenizer emits. */
static void test_chunk_sweep(void) {
    size_t len = strlen(g_corpus);
    char *want = render_once(g_corpus, len, true, false, false, 40, false);
    if (!want) return;
    check_utf8_integrity("corpus utf8 integrity", want);
    for (size_t step = 1; step <= 7; step++) {
        char *buf = NULL;
        size_t n = 0;
        FILE *fp = open_memstream(&buf, &n);
        ds4r r;
        ds4r_init(&r, fp, true, false);
        ds4r_set_columns(&r, 40);
        for (size_t i = 0; i < len; i += step) {
            size_t k = len - i < step ? len - i : step;
            ds4r_write(&r, g_corpus + i, k);
        }
        ds4r_finish(&r);
        ds4r_free(&r);
        fclose(fp);
        if (strcmp(buf, want) != 0) {
            char msg[64];
            snprintf(msg, sizeof(msg), "chunk size %zu differs", step);
            fail(msg, buf, want);
            free(buf);
            break;
        }
        free(buf);
    }
    free(want);
}

/* --plain keeps the terminal colors for thinking but prints markdown source. */
static void test_plain_mode(void) {
    static const char in[] = "<think>x</think># H\n**b** | a |\n";
    char *buf = NULL;
    size_t n = 0;
    FILE *fp = open_memstream(&buf, &n);
    ds4r r;
    ds4r_init(&r, fp, true, true);
    ds4r_set_markdown(&r, false);
    ds4r_write(&r, in, strlen(in));
    ds4r_finish(&r);
    ds4r_free(&r);
    fclose(fp);
    if (strcmp(buf, "\x1b[90mx\x1b[0m\n# H\n**b** | a |\n") != 0)
        fail("plain mode", buf, "\\e[90mx\\e[0m\\n# H\\n**b** | a |\\n");
    free(buf);
}

static void test_widths(void) {
    if (ds4r_wcwidth('a') != 1) fail("wcwidth ascii", "x", "1");
    if (ds4r_wcwidth(0x4E2D) != 2) fail("wcwidth han", "x", "2");
    if (ds4r_wcwidth(0xFF21) != 2) fail("wcwidth fullwidth", "x", "2");
    if (ds4r_wcwidth(0x1F680) != 2) fail("wcwidth emoji", "x", "2");
    if (ds4r_wcwidth(0x0301) != 0) fail("wcwidth combining", "x", "0");
    if (ds4r_visible_width("\x1b[1mab\x1b[0m", strlen("\x1b[1mab\x1b[0m")) != 2)
        fail("visible width skips escapes", "x", "2");
    if (ds4r_visible_width("中a", 4) != 3)
        fail("visible width mixes scripts", "x", "3");
}

int main(void) {
    test_inline();
    test_utf8();
    test_think();
    test_headings_lists();
    test_fences();
    test_tables();
    test_table_buffer_cap();
    test_plain_passthrough();
    test_chunk_sweep();
    test_plain_mode();
    test_widths();
    if (g_failures) {
        printf("ds4-render-test: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("ds4-render-test: all tests passed\n");
    return 0;
}
