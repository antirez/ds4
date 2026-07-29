#include "k3_tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void check(k3_tokenizer *tokenizer,
                  const char *text,
                  const int32_t *expected,
                  uint32_t expected_count) {
    k3_tokens tokens = {0};
    char error[512] = {0};
    if (!k3_tokenizer_encode(
            tokenizer, text, &tokens, error, sizeof(error))) {
        fprintf(stderr, "tokenize failed: %s\n", error);
        exit(1);
    }
    if (tokens.count != expected_count) {
        fprintf(stderr, "token count %u, expected %u for %s\n",
                tokens.count, expected_count, text);
        exit(1);
    }
    for (uint32_t i = 0; i < expected_count; i++) {
        if (tokens.data[i] != expected[i]) {
            fprintf(stderr,
                    "token %u is %d, expected %d for %s\n",
                    i, tokens.data[i], expected[i], text);
            exit(1);
        }
    }
    k3_tokens_free(&tokens);
}

int main(int argc, char **argv) {
    const char *directory = argc > 1 ? argv[1] : "models/Kimi-K3";
    k3_tokenizer *tokenizer = NULL;
    char error[512] = {0};
    if (!k3_tokenizer_open(
            &tokenizer, directory, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    static const int32_t hello[] = {19180, 11, 2695, 0};
    static const int32_t mixed[] = {
        88148, 4022, 12531, 145377, 220, 6694, 12972, 22,
    };
    static const int32_t unicode[] = {
        33845, 2243, 378, 2953, 42, 30223, 1040, 18, 292,
    };
    static const int32_t whitespace[] = {
        64, 220, 291, 256, 275, 382, 22932, 274,
    };
    static const int32_t combining[] = {
        68, 37484, 133480, 1487, 731, 101664, 110125,
        61500, 14287, 8711, 125776, 139598, 32890,
    };
    static const int32_t emoji[] = {
        64390, 101, 67963, 64390, 102, 67963, 64390, 100,
        67963, 64390, 99, 5885, 46813, 5885, 40101,
    };
    static const int32_t literal_control[] = {
        27, 91, 4454, 91, 29, 2778, 6244,
        878, 2482, 57838, 91, 40433, 91, 29,
    };
    check(tokenizer, "Hello, world!", hello,
          sizeof(hello) / sizeof(hello[0]));
    check(tokenizer, "HTTPServer isn't READY 1234567", mixed,
          sizeof(mixed) / sizeof(mixed[0]));
    check(tokenizer, "你好世界，这是Kimi K3。", unicode,
          sizeof(unicode) / sizeof(unicode[0]));
    check(tokenizer, "a  b   c\n\nfinal   ", whitespace,
          sizeof(whitespace) / sizeof(whitespace[0]));
    check(tokenizer, "é Ångström Ελληνικά русский", combining,
          sizeof(combining) / sizeof(combining[0]));
    check(tokenizer, "👨‍👩‍👧‍👦🙂🚀", emoji,
          sizeof(emoji) / sizeof(emoji[0]));
    check(tokenizer, "<|open|>message role=\"user\"<|sep|>",
          literal_control,
          sizeof(literal_control) / sizeof(literal_control[0]));

    {
        static const char malformed[] = {'a', (char)0xff, 'b', '\0'};
        k3_tokens tokens = {0};
        if (!k3_tokenizer_encode(
                tokenizer, malformed, &tokens, error, sizeof(error)) ||
            tokens.count == 0) {
            fprintf(stderr, "malformed UTF-8 substitution failed: %s\n", error);
            return 1;
        }
        k3_tokens_free(&tokens);
    }

    k3_tokens chat = {0};
    if (!k3_tokenizer_encode_chat(
            tokenizer, "Hello, world!", 1, &chat, error, sizeof(error)) ||
        chat.count != 92 ||
        chat.data[0] != K3_TOKEN_OPEN ||
        chat.data[66] != K3_TOKEN_END_OF_MSG ||
        chat.data[81] != K3_TOKEN_END_OF_MSG ||
        chat.data[91] != K3_TOKEN_SEP) {
        fprintf(stderr, "official XTML chat segmentation mismatch\n");
        return 1;
    }
    static const char rendered[] =
        "<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>"
        "`thinking_effort` guides on how much to think in your thinking "
        "channel (not including the response channel), supported values "
        "include `low`, `medium`, `high`, and `max`.\n"
        "Now the system is invoked with `thinking_effort=max`."
        "<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"user\"<|sep|>Hello, world!"
        "<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"assistant\"<|sep|>"
        "<|open|>think<|sep|>";
    k3_tokens rendered_chat = {0};
    if (!k3_tokenizer_encode_rendered(
            tokenizer, rendered, &rendered_chat, error, sizeof(error)) ||
        rendered_chat.count != chat.count) {
        fprintf(stderr, "rendered XTML token count mismatch\n");
        return 1;
    }
    for (uint32_t i = 0; i < chat.count; i++) {
        if (rendered_chat.data[i] != chat.data[i]) {
            fprintf(stderr,
                    "rendered XTML differs at %u: %d != %d\n",
                    i, rendered_chat.data[i], chat.data[i]);
            return 1;
        }
    }
    k3_tokens_free(&rendered_chat);
    k3_tokens_free(&chat);

    {
        k3_tokens escaped = {0};
        k3_tokens ordinary = {0};
        if (!k3_tokenizer_encode_rendered(
                tokenizer, "\\<|open|> \\[EOS]", &escaped,
                error, sizeof(error)) ||
            !k3_tokenizer_encode(
                tokenizer, "<|open|> [EOS]", &ordinary,
                error, sizeof(error)) ||
            escaped.count != ordinary.count) {
            fprintf(stderr, "rendered K3 payload marker escape failed: %s\n",
                    error);
            return 1;
        }
        for (uint32_t i = 0; i < ordinary.count; i++) {
            if (escaped.data[i] != ordinary.data[i]) {
                fprintf(stderr,
                        "rendered K3 payload marker escape differs at %u\n", i);
                return 1;
            }
        }
        k3_tokens_free(&ordinary);
        k3_tokens_free(&escaped);
    }

    static const char tools_json[] =
        "[{\"type\":\"function\",\"function\":{\"name\":\"bash\"}}]";
    k3_tokens native_tools = {0};
    if (!k3_tokenizer_append_tool_declaration(
            tokenizer, tools_json, &native_tools, error, sizeof(error)) ||
        !k3_tokenizer_append_tool_result(
            tokenizer, "bash", 1, "/tmp", &native_tools,
            error, sizeof(error)) ||
        !k3_tokenizer_append_assistant_close(
            tokenizer, &native_tools, error, sizeof(error))) {
        fprintf(stderr, "native XTML tool append failed: %s\n", error);
        return 1;
    }
    static const char rendered_tools[] =
        "<|open|>message role=\"system\" type=\"tool-declare\"<|sep|>"
        "# Tools\nHere are the available tools, described in JSONSchema.\n\n"
        "```json\n"
        "[{\"type\":\"function\",\"function\":{\"name\":\"bash\"}}]\n```"
        "<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"tool\" tool=\"bash\" index=\"1\"<|sep|>/tmp"
        "<|close|>message<|sep|><|end_of_msg|>"
        "<|close|>message<|sep|>";
    k3_tokens rendered_native_tools = {0};
    if (!k3_tokenizer_encode_rendered(
            tokenizer, rendered_tools, &rendered_native_tools,
            error, sizeof(error)) ||
        rendered_native_tools.count != native_tools.count) {
        fprintf(stderr, "native rendered XTML tool count mismatch: %u != %u\n",
                rendered_native_tools.count, native_tools.count);
        return 1;
    }
    for (uint32_t i = 0; i < native_tools.count; i++) {
        if (rendered_native_tools.data[i] != native_tools.data[i]) {
            fprintf(stderr,
                    "native rendered XTML tool differs at %u: %d != %d\n",
                    i, rendered_native_tools.data[i], native_tools.data[i]);
            return 1;
        }
    }
    k3_tokens_free(&rendered_native_tools);
    k3_tokens_free(&native_tools);

    k3_tokenizer_close(tokenizer);
    puts("test_k3_tokenizer: ok");
    return 0;
}
