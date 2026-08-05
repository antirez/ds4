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
                         bool in_think, bool bytewise) {
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
                        bool in_think) {
    size_t len = strlen(in);
    char *whole = render_once(in, len, color, thinking, in_think, false);
    char *split = render_once(in, len, color, thinking, in_think, true);
    if (whole && split && strcmp(whole, split) != 0)
        fail("chunking is not stable", split, whole);
    free(split);
    return whole;
}

static char *render(const char *in) {
    return render_opt(in, true, false, false);
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
           render_opt("<think>hidden</think>visible\n", true, true, false),
           "\x1b[90mhidden\x1b[0m\nvisible\n");
    /* The CLI starts inside the think block when thinking is enabled. */
    expect("implicit think open",
           render_opt("reasoning</think>answer\n", true, true, true),
           "\x1b[90mreasoning\x1b[0m\nanswer\n");
    expect("think without color",
           render_opt("<think>hidden</think>visible\n", false, true, false),
           "hidden\nvisible\n");
    expect("markdown is inert inside think",
           render_opt("<think>**not bold**</think>ok\n", true, true, false),
           "\x1b[90m**not bold**\x1b[0m\nok\n");
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
    expect("pipes inside a fence stay literal",
           render("```\n| a | b |\n```\n"),
           "```\n\x1b[38;5;75m| a | b |\x1b[0m\n```\n");
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
    char *got = render_opt(corpus, false, false, false);
    if (got && strcmp(got, corpus) != 0)
        fail("color=false passthrough", got, corpus);
    free(got);
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

int main(void) {
    test_inline();
    test_utf8();
    test_think();
    test_fences();
    test_plain_passthrough();
    test_plain_mode();
    if (g_failures) {
        printf("ds4-render-test: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("ds4-render-test: all tests passed\n");
    return 0;
}
