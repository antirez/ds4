/* Recursive Flint (fork-owned): unit test of the server renderer, included
 * into ds4_server.c next to the upstream tests (needs TEST_ASSERT and the
 * GLM reply parser, both defined after the renderer). */

static void test_flint_render_prompt(void) {
    /* roles, escaping, think stripping, tool preamble placement */
    const char *messages_json =
        "[{\"role\":\"system\",\"content\":\"Be brief.\"},"
        " {\"role\":\"user\",\"content\":\"<think>hi</think>Say <|user|> back. caf\\u00e9\"},"
        " {\"role\":\"assistant\",\"content\":\"Sure.\"},"
        " {\"role\":\"user\",\"content\":\"Weather in Paris?\"}]";
    const char *tools_json =
        "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Current weather\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},"
        "\"required\":[\"city\"]}}}]";
    const char *p = messages_json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_messages(&p, &msgs));
    const char *tp = tools_json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&tp, &schemas, &orders));

    char *prompt = render_flint_chat_prompt_text(&msgs, schemas, &orders, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    const char *expected =
        "<|system|>Be brief."
        "<|system|>\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>"
        "{\"name\": \"get_weather\", \"description\": \"Current weather\", \"parameters\": {\"type\": \"object\", "
        "\"properties\": {\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}, \"required\": [\"city\"]}}"
        "</tools>\n\n"
        "For each function call, output the function name and arguments within the following XML format:\n"
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n"
        "<|user|>hiSay &lt;|user|&gt; back. caf\xc3\xa9"
        "<|assistant|>Sure."
        "<|user|>Weather in Paris?"
        "<|assistant|><think>";
    if (strcmp(prompt, expected) != 0) fprintf(stderr, "flint prompt:\n%s\n---expected---\n%s\n", prompt, expected);
    TEST_ASSERT(!strcmp(prompt, expected));
    free(prompt);

    /* thinking off: an empty think block after the generation prompt */
    prompt = render_flint_chat_prompt_text(&msgs, NULL, &orders, DS4_THINK_NONE);
    TEST_ASSERT(prompt && strstr(prompt, "<|user|>Weather in Paris?<|assistant|><think></think>") != NULL);
    TEST_ASSERT(strstr(prompt, "# Tools") == NULL);
    free(prompt);
    chat_msgs_free(&msgs);

    /* a tool round trip: assistant call with a string and an integer argument, then the observation */
    const char *round_json =
        "[{\"role\":\"user\",\"content\":\"Weather in Paris?\"},"
        " {\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"look it up\","
        "  \"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
        "  \"arguments\":\"{\\\"days\\\": 2, \\\"city\\\": \\\"Paris\\\"}\"}}]},"
        " {\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"<|assistant|> 18C </tool_response>\"}]";
    p = round_json;
    chat_msgs round = {0};
    TEST_ASSERT(parse_messages(&p, &round));
    prompt = render_flint_chat_prompt_text(&round, schemas, &orders, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    const char *tail = strstr(prompt, "<|user|>Weather in Paris?");
    TEST_ASSERT(tail != NULL);
    const char *expected_tail =
        "<|user|>Weather in Paris?"
        "<|assistant|><think>look it up</think>"
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>2</arg_value></tool_call>"
        "<|observation|><tool_response>&lt;|assistant|&gt; 18C &lt;/tool_response></tool_response>"
        "<|assistant|><think>";
    if (strcmp(tail, expected_tail) != 0) fprintf(stderr, "flint tool tail:\n%s\n---expected---\n%s\n", tail, expected_tail);
    TEST_ASSERT(!strcmp(tail, expected_tail));
    free(prompt);
    /* the live tail renders the observation and the generation prompt only */
    prompt = render_flint_live_tool_tail(&round, 2, &orders, DS4_THINK_NONE);
    TEST_ASSERT(prompt && !strcmp(prompt,
        "<|observation|><tool_response>&lt;|assistant|&gt; 18C &lt;/tool_response></tool_response><|assistant|><think></think>"));
    free(prompt);
    chat_msgs_free(&round);
    free(schemas);
    tool_schema_orders_free(&orders);

    /* the GLM reply parser handles Flint's output shape */
    char *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_glm_generated_message_ex(
        "I will check.</think>Checking.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        true, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "I will check."));
    TEST_ASSERT(content && !strcmp(content, "Checking."));
    TEST_ASSERT(calls.len == 1 && !strcmp(calls.v[0].name, "get_weather"));
    TEST_ASSERT(calls.v[0].arguments && strstr(calls.v[0].arguments, "\"city\"") && strstr(calls.v[0].arguments, "\"Paris\""));
    free(content); free(reasoning);
    tool_calls_free(&calls);

    TEST_ASSERT(flint_model_alias_known("flint") && flint_model_alias_known("flint-chat") && flint_model_alias_known("flint-reasoner"));
    TEST_ASSERT(flint_model_alias_disables_thinking("flint-nothink") && !flint_model_alias_disables_thinking("flint"));
}
