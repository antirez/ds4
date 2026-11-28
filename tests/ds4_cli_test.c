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

static void test_k3_prompt_cache_prefixes(void) {
    int ids[] = {1, 23, 42};
    ds4_tokens tokens = {.v = ids, .len = 2};
    char *aligned = cli_prompt_cache_key(&tokens, 2);
    tokens.len = 3;
    char *full = cli_prompt_cache_key(&tokens, 2);
    char *other_chunk = cli_prompt_cache_key(&tokens, 3);
    tokens.len = 1;
    char *partial = cli_prompt_cache_key(&tokens, 2);
    tokens.len = 3;
    ids[1] = 24;
    char *different = cli_prompt_cache_key(&tokens, 2);
    CHECK(full && aligned && partial && different && other_chunk);
    if (full && aligned && partial && different && other_chunk) {
        CHECK(strncmp(full, aligned, strlen(aligned)) == 0);
        CHECK(strncmp(full, partial, strlen(partial)) != 0);
        CHECK(strncmp(different, aligned, strlen(aligned)) != 0);
        CHECK(strcmp(full, other_chunk) != 0);
        CHECK(cli_prompt_cache_key(&tokens, 0) == NULL);
        ids[1] = 23;
        char *again = cli_prompt_cache_key(&tokens, 2);
        CHECK(again && !strcmp(again, full));
        free(again);
    }
    free(full);
    free(different);
    free(partial);
    free(aligned);
    free(other_chunk);
}

static void test_k3_prompt_cache_file_option(void) {
    char path[] = "/tmp/ds4-cli-prompt.XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(write(fd, "hello", 5) == 5);
    CHECK(close(fd) == 0);
    char *argv[] = {"ds4", "--k3-prompt-cache", "cache", "--prompt-file", path};
    cli_config cfg = parse_options(5, argv);
    CHECK(cfg.gen.prompt_cache_dir && !strcmp(cfg.gen.prompt_cache_dir, "cache"));
    CHECK(cfg.gen.prompt && !strcmp(cfg.gen.prompt, "hello"));
    free(cfg.prompt_owned);
    ds4_dist_options_free(cfg.dist);
    ds4_prompt_prefix_free(&cfg.gen.prefix);
    unlink(path);
}

int main(void) {
    test_k3_prompt_cache_prefixes();
    test_k3_prompt_cache_file_option();
    test_k3_projection_hides_fragmented_xtml();
    test_k3_projection_accepts_already_open_response();
    if (failures) {
        fprintf(stderr, "ds4 CLI tests: %d failed\n", failures);
        return 1;
    }
    puts("ds4 CLI tests: ok");
    return 0;
}
