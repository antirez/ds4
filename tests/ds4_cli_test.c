#define main ds4_cli_program_main
#include "../ds4_cli.c"
#undef main

static int failures;

#define CHECK(condition) do {                                              \
    if (!(condition)) {                                                    \
        fprintf(stderr, "%s:%d: check failed: %s\n",                       \
                __FILE__, __LINE__, #condition);                           \
        failures++;                                                        \
    }                                                                      \
} while (0)

static char *capture_k3_projection(const char *const *chunks, size_t count) {
    char *output = NULL;
    size_t output_len = 0;
    FILE *fp = open_memstream(&output, &output_len);
    CHECK(fp != NULL);
    if (!fp) return NULL;

    token_printer printer = {
        .fp = fp,
        .k3 = true,
        .k3_at_response_start = true,
        .last_output_newline = true,
    };
    for (size_t i = 0; i < count; i++) {
        token_printer_write_text(&printer, chunks[i], strlen(chunks[i]));
    }
    token_printer_finish(&printer);
    fclose(fp);
    return output;
}

static void test_k3_projection_hides_fragmented_xtml(void) {
    const char *chunks[] = {
        "<think>reason",
        "ing</think><res",
        "ponse>Hello",
        "</res",
        "ponse><too",
        "ls><call tool=\"bash\" index=\"1\"></call></tools></message>",
    };
    char *output = capture_k3_projection(
        chunks, sizeof(chunks) / sizeof(chunks[0]));
    CHECK(output != NULL);
    CHECK(output && !strcmp(output, "reasoning\nHello"));
    free(output);
}

static void test_k3_projection_accepts_already_open_response(void) {
    const char *chunks[] = {
        "Direct answer",
        "</message>",
    };
    char *output = capture_k3_projection(
        chunks, sizeof(chunks) / sizeof(chunks[0]));
    CHECK(output != NULL);
    CHECK(output && !strcmp(output, "Direct answer"));
    free(output);
}

int main(void) {
    test_k3_projection_hides_fragmented_xtml();
    test_k3_projection_accepts_already_open_response();
    if (failures) {
        fprintf(stderr, "ds4 CLI tests: %d failed\n", failures);
        return 1;
    }
    puts("ds4 CLI tests: ok");
    return 0;
}
