#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"

static int test_failures;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "TEST_ASSERT failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #cond); \
        test_failures++; \
    } \
} while (0)

static char *test_render_stream(const char *text, agent_stream_renderer *stream,
                                agent_dsml_state *state, size_t *out_len) {
    agent_tail_capture capture = {.cap = 8192};
    agent_token_renderer renderer = {
        .format_thinking = true,
        .format_markdown = false,
        .in_think = true,
        .last_output_newline = true,
        .capture = &capture,
    };
    agent_dsml_parser dsml = {.state = AGENT_DSML_SEARCH};
    *stream = (agent_stream_renderer){
        .renderer = &renderer,
        .parser = &dsml,
        .in_think = true,
    };

    agent_stream_text(stream, text, strlen(text), false);
    agent_stream_text(stream, NULL, 0, true);
    renderer_finish(&renderer);

    char *out = agent_tail_capture_take(&capture, out_len);
    if (state) *state = dsml.state;
    agent_dsml_parser_free(&dsml);
    return out;
}

static void test_ascii_dsml_mention_in_thinking_is_plain_text(void) {
    const char *text =
        "<think>The literal marker <|DSML|tool_calls> is documentation, "
        "not a tool call.</think>Done.";
    agent_stream_renderer stream = {0};
    agent_dsml_state state = AGENT_DSML_SEARCH;
    size_t out_len = 0;
    char *out = test_render_stream(text, &stream, &state, &out_len);

    TEST_ASSERT(out != NULL);
    TEST_ASSERT(!stream.dsml_in_think);
    TEST_ASSERT(state == AGENT_DSML_SEARCH);
    TEST_ASSERT(!strstr(out ? out : "", "[tool call ignored:"));
    TEST_ASSERT(strstr(out ? out : "", "<|DSML|tool_calls>") != NULL);
    TEST_ASSERT(strstr(out ? out : "", "Done.") != NULL);

    free(out);
    (void)out_len;
}

static void test_real_dsml_inside_thinking_is_ignored(void) {
    const char *text =
        "<think><｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"list\">\n"
        "<｜DSML｜parameter name=\"path\" string=\"true\">.</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls></think>Done.";
    agent_stream_renderer stream = {0};
    agent_dsml_state state = AGENT_DSML_SEARCH;
    size_t out_len = 0;
    char *out = test_render_stream(text, &stream, &state, &out_len);

    TEST_ASSERT(out != NULL);
    TEST_ASSERT(stream.dsml_in_think);
    TEST_ASSERT(strstr(out ? out : "", "[tool call ignored: tool calling is not allowed inside <think></think>]") != NULL);
    TEST_ASSERT(strstr(out ? out : "", "Done.") != NULL);
    TEST_ASSERT(state == AGENT_DSML_SEARCH);

    free(out);
    (void)out_len;
}

int main(void) {
    test_ascii_dsml_mention_in_thinking_is_plain_text();
    test_real_dsml_inside_thinking_is_ignored();

    if (test_failures) {
        fprintf(stderr, "ds4 agent tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4 agent tests: ok");
    return 0;
}
