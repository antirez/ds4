/* No generation: exercise the actual CLI transcript updater with the V4.1 fixture. */
#define main ds4_cli_main
#include "../ds4_cli.c"
#undef main
#include <assert.h>

/* The status panel must show numeric efforts instead of falling back to
 * "unknown" after the V4.1 thinking modes are merged into the CLI UI. */
static void test_thinking_ui(void) {
    const ds4_think_mode named[] = {DS4_THINK_NONE, DS4_THINK_HIGH, DS4_THINK_MAX};
    const char *names[] = {"off", "high", "max"};
    for (size_t i = 0; i < sizeof(named) / sizeof(*named); i++)
        assert(!strcmp(cli_ui_thinking_name(named[i]), names[i]));
    for (int level = 0; level <= 100; level++) {
        char input[4], expected[32], output[1024];
        snprintf(input, sizeof(input), "%d", level);
        ds4_think_mode mode;
        assert(ds4_think_mode_parse_level(input, &mode));
        ds4_cli_ui_state state = {.thinking = cli_ui_thinking_name(mode)};
        FILE *fp = tmpfile();
        assert(fp);
        ds4_cli_ui_print_status(fp, &state);
        rewind(fp);
        size_t n = fread(output, 1, sizeof(output) - 1, fp);
        output[n] = '\0';
        assert(!ferror(fp));
        fclose(fp);
        snprintf(expected, sizeof(expected), "thinking %d\n", level);
        assert(strstr(output, expected));
    }
    puts("CLI UI thinking levels 0..100: PASS");
}

static void test_output_padding(void) {
    FILE *fp = tmpfile();
    assert(fp);
    token_printer printer = {.fp = fp, .format_thinking = true, .last_output_newline = true};
    ds4_cli_ui_output_start(&printer.output, fp);
    printer.output.enabled = true;
    printer.output.width = 8;
    printer.output.margin = 2;
    const char *text = "<think>abc中def</think>abcdefghijklmnop";
    for (size_t i = 0; text[i]; i++) token_printer_write_text(&printer, text + i, 1);
    generation_done(&printer);
    rewind(fp);
    char output[256];
    size_t n = fread(output, 1, sizeof(output) - 1, fp);
    output[n] = '\0';
    assert(!strcmp(output, "  abc中def\n\n  abcdefgh\n  ijklmnop\n"));
    fclose(fp);
    puts("CLI padded streaming, split UTF-8 and thinking transitions: PASS");
}

static void test_thinking_spacing(void) {
    const char *gaps[] = {"", "\n", "\n\n", "\n\n\n"};
    for (size_t before = 0; before < 4; before++) {
        for (size_t after = 0; after < 4; after++) {
            char text[256];
            snprintf(text, sizeof(text), "<think>first\n\nlast%s</think>%sanswer\n\nparagraph",
                     gaps[before], gaps[after]);
            for (size_t split = 0; split <= strlen(text) + 1; split++) {
                FILE *fp = tmpfile();
                assert(fp);
                token_printer p = {.fp = fp, .format_thinking = true, .last_output_newline = true};
                if (split <= strlen(text)) {
                    token_printer_write_text(&p, text, split);
                    token_printer_write_text(&p, text + split, strlen(text) - split);
                } else {
                    for (size_t i = 0; text[i]; i++) token_printer_write_text(&p, text + i, 1);
                }
                generation_done(&p);
                rewind(fp);
                char output[256] = {0};
                assert(fread(output, 1, sizeof(output) - 1, fp) < sizeof(output) - 1);
                assert(!strcmp(output, "first\n\nlast\n\nanswer\n\nparagraph\n"));
                fclose(fp);
            }
        }
    }
    const char *edge[] = {"<think></think>\n\nanswer", "<think>reason\n\n", "reason\n\n</think>\n\nanswer"};
    const char *expected[] = {"answer\n", "reason\n", "reason\n\nanswer\n"};
    for (size_t i = 0; i < 3; i++) {
        FILE *fp = tmpfile();
        assert(fp);
        token_printer p = {.fp = fp, .format_thinking = true, .in_think = i == 2,
                           .last_output_newline = true};
        for (size_t j = 0; edge[i][j]; j++) token_printer_write_text(&p, edge[i] + j, 1);
        generation_done(&p);
        rewind(fp);
        char output[256] = {0};
        assert(fread(output, 1, sizeof(output) - 1, fp) < sizeof(output) - 1);
        assert(!strcmp(output, expected[i]));
        fclose(fp);
    }
    puts("CLI thinking gap and paragraph spacing across token splits: PASS");
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    if (!strcmp(argv[1], "--ui")) {
        test_thinking_ui();
        test_output_padding();
        test_thinking_spacing();
        return 0;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_experts = 512,
        .context_size = 256, .power_percent = 100};
    assert(ds4_engine_open(&engine, &opt) == 0);
    const ds4_think_mode modes[] = {DS4_THINK_NONE, DS4_THINK_HIGH,
        (ds4_think_mode)(DS4_THINK_LEVEL_BASE + 25), DS4_THINK_MAX,
        (ds4_think_mode)DS4_THINK_LEVEL_BASE, DS4_THINK_HIGH, DS4_THINK_NONE};
    for (int system = 0; system < 2; system++) {
        repl_chat chat = {.initial_system_text = system != 0};
        ds4_chat_begin(engine, &chat.transcript);
        chat.think_prefix_pos = chat.transcript.len;
        repl_chat_apply_think_prefix(engine, &chat, modes[0]);
        if (system) ds4_chat_append_message(engine, &chat.transcript, "system", "System text.");
        ds4_chat_append_message(engine, &chat.transcript, "user", "First question");
        ds4_chat_append_message(engine, &chat.transcript, "assistant", "First answer");
        for (size_t i = 0; i < sizeof(modes) / sizeof(*modes); i++) {
            assert(repl_chat_apply_think_prefix(engine, &chat, modes[i]));
            ds4_tokens expected = {0};
            ds4_chat_begin(engine, &expected);
            ds4_chat_append_think_prefix(engine, &expected, modes[i]);
            if (system) ds4_chat_append_message(engine, &expected, "system", "System text.");
            ds4_chat_append_message(engine, &expected, "user", "First question");
            ds4_chat_append_message(engine, &expected, "assistant", "First answer");
            assert(expected.len == chat.transcript.len &&
                   ds4_tokens_starts_with(&chat.transcript, &expected));
            ds4_tokens_free(&expected);
        }
        ds4_tokens before = {0};
        ds4_tokens_copy(&before, &chat.transcript);
        chat.ctx_size = chat.transcript.len + 1;
        assert(!repl_chat_apply_think_prefix(engine, &chat, DS4_THINK_MAX));
        assert(before.len == chat.transcript.len && ds4_tokens_starts_with(&chat.transcript, &before));
        ds4_tokens_free(&before);
        repl_chat_free(&chat);
    }
    ds4_engine_close(engine);
    puts("V4.1 CLI thinking prefix transitions: PASS");
    return 0;
}
