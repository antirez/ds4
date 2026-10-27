#include "ds4_prompt_prefix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static void test_multiline(void) {
    static const char input[] =
        "USER: first line\n"
        "second line\n"
        " USER: this stays in the message\n"
        "ASSISTANT:\tfirst answer\n"
        "second answer\n";
    ds4_prompt_prefix prefix = {0};
    char error[256] = {0};
    CHECK(ds4_prompt_prefix_parse(&prefix, input, sizeof(input) - 1,
                                  error, sizeof(error)) == 0);
    CHECK(prefix.count == 2);
    if (prefix.count == 2) {
        CHECK(prefix.turns[0].role == DS4_PROMPT_PREFIX_USER);
        CHECK(!strcmp(prefix.turns[0].content,
                      "first line\nsecond line\n USER: this stays in the message"));
        CHECK(prefix.turns[1].role == DS4_PROMPT_PREFIX_ASSISTANT);
        CHECK(!strcmp(prefix.turns[1].content,
                      "first answer\nsecond answer"));
    }
    ds4_prompt_prefix_free(&prefix);
}

static void test_multiple_pairs_and_bom(void) {
    static const char input[] =
        "\xef\xbb\xbfUSER: one\r\n"
        "ASSISTANT: two\r\n"
        "USER: three\n"
        "ASSISTANT: four";
    ds4_prompt_prefix prefix = {0};
    char error[256] = {0};
    CHECK(ds4_prompt_prefix_parse(&prefix, input, sizeof(input) - 1,
                                  error, sizeof(error)) == 0);
    CHECK(prefix.count == 4);
    if (prefix.count == 4) {
        CHECK(!strcmp(prefix.turns[0].content, "one"));
        CHECK(!strcmp(prefix.turns[1].content, "two"));
        CHECK(!strcmp(prefix.turns[2].content, "three"));
        CHECK(!strcmp(prefix.turns[3].content, "four"));
    }
    ds4_prompt_prefix_free(&prefix);
}

static void expect_error(const char *input, const char *message) {
    ds4_prompt_prefix prefix = {0};
    char error[256] = {0};
    CHECK(ds4_prompt_prefix_parse(&prefix, input, strlen(input),
                                  error, sizeof(error)) != 0);
    CHECK(strstr(error, message) != NULL);
    CHECK(prefix.turns == NULL);
    CHECK(prefix.count == 0);
}

static void test_errors(void) {
    expect_error("", "empty");
    expect_error("preamble\nUSER: one\nASSISTANT: two", "line 1");
    expect_error("ASSISTANT: one", "expected USER");
    expect_error("USER: one\nUSER: two", "expected ASSISTANT");
    expect_error("USER: one", "must end with an ASSISTANT");
    expect_error("USER:\nASSISTANT: two", "empty USER");
    expect_error("USER: one\nASSISTANT:\n", "empty ASSISTANT");
}

static void test_load_file(void) {
    static const char input[] = "USER: loaded\nASSISTANT: yes\n";
    char path[] = "/tmp/ds4-prefix-test.XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(write(fd, input, sizeof(input) - 1) == (ssize_t)(sizeof(input) - 1));
    CHECK(close(fd) == 0);

    ds4_prompt_prefix prefix = {0};
    char error[256] = {0};
    CHECK(ds4_prompt_prefix_load(&prefix, path, error, sizeof(error)) == 0);
    CHECK(prefix.count == 2);
    if (prefix.count == 2) {
        CHECK(!strcmp(prefix.turns[0].content, "loaded"));
        CHECK(!strcmp(prefix.turns[1].content, "yes"));
    }
    ds4_prompt_prefix_free(&prefix);
    CHECK(unlink(path) == 0);
}

int main(void) {
    test_multiline();
    test_multiple_pairs_and_bom();
    test_errors();
    test_load_file();
    if (failures) {
        fprintf(stderr, "%d prefix parser test(s) failed\n", failures);
        return 1;
    }
    puts("prompt prefix tests passed");
    return 0;
}
