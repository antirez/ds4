/* Recursive Flint (fork-owned): unit test of the server renderer, included
 * into ds4_server.c next to the upstream tests (needs TEST_ASSERT and the
 * GLM reply parser, both defined after the renderer). */

static void test_flint_visible_checkpoint(void);
static void test_flint_typed_arguments(void);

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
        true, &content, &reasoning, &calls, NULL));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "I will check."));
    TEST_ASSERT(content && !strcmp(content, "Checking."));
    TEST_ASSERT(calls.len == 1 && !strcmp(calls.v[0].name, "get_weather"));
    TEST_ASSERT(calls.v[0].arguments && strstr(calls.v[0].arguments, "\"city\"") && strstr(calls.v[0].arguments, "\"Paris\""));
    free(content); free(reasoning);
    tool_calls_free(&calls);

    TEST_ASSERT(flint_model_alias_known("flint") && flint_model_alias_known("flint-chat") && flint_model_alias_known("flint-reasoner"));
    TEST_ASSERT(flint_model_alias_disables_thinking("flint-nothink") && !flint_model_alias_disables_thinking("flint"));
    test_flint_visible_checkpoint();
    test_flint_typed_arguments();
}

/* Argument values follow the template's `tojson if not string` rule: a
 * schema-declared string stays text whatever it looks like, anything else
 * that parses as JSON (arrays, objects, numbers, booleans) is passed on as
 * JSON, and a value that is not JSON stays text. */
static void test_flint_typed_arguments(void) {
    const char *tools_json =
        "[{\"type\":\"function\",\"function\":{\"name\":\"return_items\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"order_id\":{\"type\":\"string\"},\"zip\":{\"type\":\"string\"},"
        "\"item_ids\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"count\":{\"type\":\"integer\"},"
        "\"note\":{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"null\"}]}}}}}]";
    const char *tp = tools_json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&tp, &schemas, &orders));
    const char *generated =
        "</think><tool_call>return_items"
        "<arg_key>order_id</arg_key><arg_value>#W6390527</arg_value>"
        "<arg_key>zip</arg_key><arg_value>12345</arg_value>"
        "<arg_key>item_ids</arg_key><arg_value>[\"8538875209\", \"1\"]</arg_value>"
        "<arg_key>count</arg_key><arg_value>2</arg_value>"
        "<arg_key>note</arg_key><arg_value>none</arg_value>"
        "<arg_key>extra</arg_key><arg_value>{\"a\": true}</arg_value>"
        "<arg_key>loose</arg_key><arg_value>[not json</arg_value>"
        "</tool_call>";
    const bool was_flint = g_flint_server;
    g_flint_server = true;
    char *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_glm_generated_message_ex(generated, true, &content, &reasoning, &calls, &orders));
    g_flint_server = was_flint;
    TEST_ASSERT(calls.len == 1 && !strcmp(calls.v[0].name, "return_items"));
    const char *expected_args =
        "{\"order_id\": \"#W6390527\", \"zip\": \"12345\", \"item_ids\": [\"8538875209\",\"1\"], \"count\": 2, "
        "\"note\": \"none\", \"extra\": {\"a\":true}, \"loose\": \"[not json\"}";
    if (strcmp(calls.v[0].arguments, expected_args) != 0)
        fprintf(stderr, "flint typed arguments:\n%s\n---expected---\n%s\n", calls.v[0].arguments, expected_args);
    TEST_ASSERT(!strcmp(calls.v[0].arguments, expected_args));
    free(content); free(reasoning);
    tool_calls_free(&calls);

    /* without the schema (or outside Flint) the upstream behaviour holds */
    TEST_ASSERT(parse_glm_generated_message_ex(generated, true, &content, &reasoning, &calls, NULL));
    TEST_ASSERT(calls.len == 1 && strstr(calls.v[0].arguments, "\"item_ids\": \"[\\\"8538875209\\\", \\\"1\\\"]\""));
    free(content); free(reasoning);
    tool_calls_free(&calls);
    free(schemas);
    tool_schema_orders_free(&orders);
}

/* The thinking-visible key: the bytes the next request renders for the turn
 * just generated when the client echoes content and tool calls but not the
 * reasoning, so the live KV continues with only the new suffix tokenized. */
static void test_flint_visible_checkpoint(void) {
    const char *tools_json =
        "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"city\":{\"type\":\"string\"}}}}}]";
    const char *tp = tools_json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&tp, &schemas, &orders));
    const char *first_json = "[{\"role\":\"user\",\"content\":\"Hi\"}]";
    const char *p = first_json;
    chat_msgs first = {0};
    TEST_ASSERT(parse_messages(&p, &first));

    /* a text answer in a tool conversation, thinking on */
    char *prompt = render_flint_chat_prompt_text(&first, schemas, &orders, DS4_THINK_HIGH);
    TEST_ASSERT(prompt && strstr(prompt, "<|user|>Hi<|assistant|><think>"));
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    r.model_syntax = SERVER_MODEL_SYNTAX_GLM;
    r.has_tools = true;
    r.prompt_text = prompt;
    TEST_ASSERT(flint_should_remember_visible_checkpoint(&r, false, "stop"));
    TEST_ASSERT(!flint_should_remember_visible_checkpoint(&r, true, "stop"));
    TEST_ASSERT(!flint_should_remember_visible_checkpoint(&r, false, "length"));
    char *visible = build_flint_visible_text(&r, "Hello!\nHow are you?", NULL);
    TEST_ASSERT(visible != NULL);
    const char *next_json =
        "[{\"role\":\"user\",\"content\":\"Hi\"},"
        " {\"role\":\"assistant\",\"content\":\"Hello!\\nHow are you?\"},"
        " {\"role\":\"user\",\"content\":\"Fine.\"}]";
    p = next_json;
    chat_msgs next = {0};
    TEST_ASSERT(parse_messages(&p, &next));
    char *future = render_flint_chat_prompt_text(&next, schemas, &orders, DS4_THINK_HIGH);
    TEST_ASSERT(future && strlen(future) > strlen(visible) && !memcmp(future, visible, strlen(visible)));
    TEST_ASSERT(!strcmp(future + strlen(visible), "<|user|>Fine.<|assistant|><think>"));
    free(future);
    /* reasoning echoed back is rendered and must not match the key */
    next.v[1].reasoning = xstrdup("greet");
    future = render_flint_chat_prompt_text(&next, schemas, &orders, DS4_THINK_HIGH);
    TEST_ASSERT(future && strncmp(future, visible, strlen(visible)) != 0);
    free(future);
    chat_msgs_free(&next);
    free(visible);

    /* thinking off: the generation prompt's empty think block is not in the history either */
    free(r.prompt_text);
    r.prompt_text = render_flint_chat_prompt_text(&first, schemas, &orders, DS4_THINK_NONE);
    r.think_mode = DS4_THINK_NONE;
    visible = build_flint_visible_text(&r, "Hello!", NULL);
    TEST_ASSERT(visible && strlen(visible) > 19 && !strcmp(visible + strlen(visible) - 19, "<|assistant|>Hello!"));
    TEST_ASSERT(strstr(visible, "<think>") == NULL);
    free(visible);

    /* a tool call: the sampled call text, replayed by id, is the frontier */
    free(r.prompt_text);
    r.prompt_text = render_flint_chat_prompt_text(&first, schemas, &orders, DS4_THINK_HIGH);
    r.think_mode = DS4_THINK_HIGH;
    const char *raw = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("c1");
    tc.name = xstrdup("get_weather");
    tc.arguments = xstrdup("{\"city\": \"Paris\"}");
    tool_calls_push(&calls, tc);
    TEST_ASSERT(build_flint_visible_text(&r, "", &calls) == NULL);   /* no raw text: the renderer would re-render */
    calls.raw_tool_text = xstrdup(raw);
    visible = build_flint_visible_text(&r, "", &calls);
    TEST_ASSERT(visible != NULL);
    const char *round_json =
        "[{\"role\":\"user\",\"content\":\"Hi\"},"
        " {\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\","
        "  \"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\": \\\"Paris\\\"}\"}}]},"
        " {\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"18C\"}]";
    p = round_json;
    chat_msgs round = {0};
    TEST_ASSERT(parse_messages(&p, &round));
    round.v[1].calls.raw_tool_text = xstrdup(raw);   /* what tool memory attaches by id */
    future = render_flint_chat_prompt_text(&round, schemas, &orders, DS4_THINK_HIGH);
    TEST_ASSERT(future && strlen(future) > strlen(visible) && !memcmp(future, visible, strlen(visible)));
    TEST_ASSERT(!strcmp(future + strlen(visible), "<|observation|><tool_response>18C</tool_response><|assistant|><think>"));
    free(future);
    chat_msgs_free(&round);
    free(visible);
    tool_calls_free(&calls);
    request_free(&r);
    chat_msgs_free(&first);
    free(schemas);
    tool_schema_orders_free(&orders);
}
