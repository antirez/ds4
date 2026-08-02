/* In-file unit tests extracted move-only from cli_main.c (the old
 * `#else` branch of its PULSAR_SERVER_TEST guard). Compiled ONLY inside
 * the tests/pulsar_test.c harness TU, which #includes every server .c
 * with PULSAR_SERVER_TEST defined and includes cli_main.c BEFORE this
 * file, so cli_main.c file-statics the tests poke (parse_options,
 * server_kv_budget_bytes, server_default_kv_disk_dir,
 * server_resolve_kv_disk_dir) remain reachable exactly as before.
 * A standalone compile (no PULSAR_SERVER_TEST) yields an empty object
 * for the pulsar-server link. */
#include "pulsar_server_internal.h"

#ifdef PULSAR_SERVER_TEST

static int test_failures = 0;



static void test_assert(bool cond, const char *file, int line, const char *expr) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    test_failures++;
}



#define TEST_ASSERT(expr) test_assert((expr), __FILE__, __LINE__, #expr)


static void test_tool_schema_order_from_anthropic_schema(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{\"type\":\"string\"},"
        "\"description\":{\"type\":\"string\"}}}}");
    const tool_schema_order *order = tool_schema_orders_find(&orders, "bash");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->len == 2);
    TEST_ASSERT(order && !strcmp(order->prop[0], "command"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "description"));
    tool_schema_orders_free(&orders);
}



static void test_tool_schema_order_from_openai_tools(void) {
    const char *json =
        "[{\"type\":\"function\",\"function\":{\"name\":\"edit\",\"parameters\":{"
        "\"type\":\"object\",\"properties\":{"
        "\"filePath\":{\"type\":\"string\"},"
        "\"oldString\":{\"type\":\"string\"},"
        "\"newString\":{\"type\":\"string\"}}}}}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"edit\""));
    const tool_schema_order *order = tool_schema_orders_find(&orders, "edit");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->len == 3);
    TEST_ASSERT(order && !strcmp(order->prop[0], "filePath"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "oldString"));
    TEST_ASSERT(order && !strcmp(order->prop[2], "newString"));
    free(schemas);
    tool_schema_orders_free(&orders);
}



static void test_tool_schema_order_from_responses_tool_search(void) {
    const char *json =
        "[{\"type\":\"tool_search\",\"execution\":\"client\","
        "\"description\":\"Search deferred tools\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"},"
        "\"limit\":{\"type\":\"number\"}},\"required\":[\"query\"]}}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"tool_search\""));
    TEST_ASSERT(schemas && strstr(schemas, "\"description\":\"Search deferred tools\""));
    const tool_schema_order *order = tool_schema_orders_find(&orders, "tool_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->responses_tool_search);
    TEST_ASSERT(order && order->len == 2);
    TEST_ASSERT(order && !strcmp(order->prop[0], "query"));
    TEST_ASSERT(order && !strcmp(order->prop[1], "limit"));
    free(schemas);
    tool_schema_orders_free(&orders);
}



static void test_responses_function_named_tool_search_stays_function_call(void) {
    const char *json =
        "[{\"type\":\"function\",\"function\":{\"name\":\"tool_search\","
        "\"description\":\"A normal user function that happens to use a reserved name\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"}}}}}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    const tool_schema_order *order = tool_schema_orders_find(&orders, "tool_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && !order->responses_tool_search);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_user_tool_search");
    tc.name = xstrdup("tool_search");
    tc.arguments = xstrdup("{\"query\":\"plain function\"}");
    tool_calls_push(&calls, tc);
    responses_tool_item item = {
        .fc_id = "fc_user_tool_search",
        .call_id = "call_user_tool_search",
        .is_custom = false,
        .output_index = 0,
    };

    buf out = {0};
    responses_append_function_call_item(&out, &calls.v[0], &item,
                                        "completed", true, &orders);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"function_call\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"tool_search_call\"") == NULL);

    buf_free(&out);
    tool_calls_free(&calls);
    free(schemas);
    tool_schema_orders_free(&orders);
}



static void test_responses_namespace_tool_schemas_restore_wire_namespace(void) {
    const char *json =
        "[{\"type\":\"namespace\",\"name\":\"mcp__perplexity__\","
        "\"description\":\"Perplexity tools\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"perplexity_search\","
        "\"description\":\"Search the web\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"},"
        "\"recency\":{\"type\":\"number\"}}}}]}]";
    const char *p = json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&p, &schemas, &orders));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"mcp__perplexity__perplexity_search\""));
    TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"perplexity_search\"") == NULL);

    const tool_schema_order *order =
        tool_schema_orders_find(&orders, "mcp__perplexity__perplexity_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->tool_namespace && !strcmp(order->tool_namespace, "mcp__perplexity__"));
    TEST_ASSERT(order && order->wire_name && !strcmp(order->wire_name, "perplexity_search"));
    TEST_ASSERT(order && order->len == 2);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_ns");
    tc.name = xstrdup("mcp__perplexity__perplexity_search");
    tc.arguments = xstrdup("{\"query\":\"deepseek\",\"recency\":7}");
    tool_calls_push(&calls, tc);
    responses_tool_item item = {
        .fc_id = "fc_ns",
        .call_id = "call_ns",
        .is_custom = false,
        .output_index = 0,
    };
    buf out = {0};
    responses_append_function_call_item(&out, &calls.v[0], &item,
                                        "completed", true, &orders);
    TEST_ASSERT(strstr(out.ptr, "\"name\":\"perplexity_search\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"namespace\":\"mcp__perplexity__\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "mcp__perplexity__perplexity_search") == NULL);

    buf_free(&out);
    tool_calls_free(&calls);
    free(schemas);
    tool_schema_orders_free(&orders);
}



static void test_responses_input_tool_search_output_loads_tools(void) {
    const char *json =
        "["
        "{\"type\":\"tool_search_call\",\"call_id\":\"call_search\","
        "\"execution\":\"client\",\"arguments\":{\"query\":\"perplexity\"}},"
        "{\"type\":\"tool_search_output\",\"call_id\":\"call_search\","
        "\"status\":\"completed\",\"execution\":\"client\",\"tools\":["
        "{\"type\":\"namespace\",\"name\":\"mcp__perplexity__\","
        "\"description\":\"Perplexity tools\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"perplexity_search\","
        "\"description\":\"Search with Perplexity\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"}}}}]}]}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    buf loaded = {0};
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_responses_input(&p, &msgs, &loaded, &orders));
    TEST_ASSERT(loaded.ptr && strstr(loaded.ptr, "\"name\":\"mcp__perplexity__perplexity_search\""));
    const tool_schema_order *order =
        tool_schema_orders_find(&orders, "mcp__perplexity__perplexity_search");
    TEST_ASSERT(order != NULL);
    TEST_ASSERT(order && order->tool_namespace && !strcmp(order->tool_namespace, "mcp__perplexity__"));
    TEST_ASSERT(order && order->wire_name && !strcmp(order->wire_name, "perplexity_search"));
    TEST_ASSERT(msgs.len == 2);
    TEST_ASSERT(msgs.v[0].calls.len == 1);
    TEST_ASSERT(!strcmp(msgs.v[0].calls.v[0].name, "tool_search"));
    TEST_ASSERT(strstr(msgs.v[1].content, "mcp__perplexity__") != NULL);

    buf_free(&loaded);
    tool_schema_orders_free(&orders);
    chat_msgs_free(&msgs);
}



static void test_responses_input_tool_search_output_rejects_bad_tools(void) {
    const char *json =
        "[{\"type\":\"tool_search_output\",\"call_id\":\"call_search\","
        "\"status\":\"completed\",\"tools\":{\"not\":\"a tool array\"}}]";
    const char *p = json;
    chat_msgs msgs = {0};
    buf loaded = {0};
    tool_schema_orders orders = {0};
    TEST_ASSERT(!parse_responses_input(&p, &msgs, &loaded, &orders));
    buf_free(&loaded);
    tool_schema_orders_free(&orders);
    chat_msgs_free(&msgs);
}



static void test_responses_input_function_call_namespace_round_trips_to_dsml(void) {
    const char *tools_json =
        "[{\"type\":\"namespace\",\"name\":\"mcp__perplexity__\","
        "\"tools\":[{\"type\":\"function\",\"name\":\"perplexity_search\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"}}}}]}]";
    const char *tools_p = tools_json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&tools_p, &schemas, &orders));

    const char *input_json =
        "[{\"type\":\"function_call\",\"call_id\":\"call_ns\","
        "\"name\":\"perplexity_search\",\"namespace\":\"mcp__perplexity__\","
        "\"arguments\":{\"query\":\"deepseek\"}}]";
    const char *input_p = input_json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_responses_input(&input_p, &msgs, NULL, NULL));
    TEST_ASSERT(msgs.len == 1);
    TEST_ASSERT(msgs.v[0].calls.len == 1);
    TEST_ASSERT(!strcmp(msgs.v[0].calls.v[0].name,
                        "mcp__perplexity__perplexity_search"));

    char *prompt = render_chat_prompt_text(&msgs, schemas, &orders, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt,
        "<｜DSML｜invoke name=\"mcp__perplexity__perplexity_search\">") != NULL);
    TEST_ASSERT(strstr(prompt, "<｜DSML｜invoke name=\"perplexity_search\">") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
    free(schemas);
    tool_schema_orders_free(&orders);
}



static void test_responses_output_sends_tool_search_call_item(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_search");
    tc.name = xstrdup("tool_search");
    tc.arguments = xstrdup("{\"limit\":3,\"query\":\"perplexity\"}");
    tool_calls_push(&calls, tc);
    const char *tools_json =
        "[{\"type\":\"tool_search\",\"execution\":\"client\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"query\":{\"type\":\"string\"},\"limit\":{\"type\":\"number\"}}}}]";
    const char *tools_p = tools_json;
    char *schemas = NULL;
    tool_schema_orders orders = {0};
    TEST_ASSERT(parse_tools_value(&tools_p, &schemas, &orders));
    responses_tool_item item = {
        .fc_id = "fc_search",
        .call_id = "call_search",
        .is_custom = false,
        .output_index = 0,
    };

    buf out = {0};
    responses_append_function_call_item(&out, &calls.v[0], &item,
                                        "completed", true, &orders);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"tool_search_call\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"execution\":\"client\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"status\":\"completed\"") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"arguments\":{\"limit\":3,\"query\":\"perplexity\"}") != NULL);
    TEST_ASSERT(strstr(out.ptr, "\"type\":\"function_call\"") == NULL);

    buf_free(&out);
    free(schemas);
    tool_schema_orders_free(&orders);
    tool_calls_free(&calls);
}



static tool_calls make_swapped_bash_call(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    tool_calls_push(&calls, tc);
    return calls;
}



static tool_schema_orders make_bash_order(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"bash\",\"input_schema\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{\"type\":\"string\"},"
        "\"description\":{\"type\":\"string\"}}}}");
    return orders;
}



static char *read_socket_text(int fd) {
    buf b = {0};
    char tmp[1024];
    ssize_t n;
    while ((n = read(fd, tmp, sizeof(tmp))) > 0) {
        buf_append(&b, tmp, (size_t)n);
    }
    return buf_take(&b);
}



static void test_context_length_error_uses_protocol_standard_shape(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.prompt.len = 16;
    TEST_ASSERT(request_exceeds_context(&r, 16));
    TEST_ASSERT(!request_exceeds_context(&r, 17));

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_error_context_length_exceeded(sv[0], false, &r, 16, 16));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 400") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"invalid_request_error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"code\":\"context_length_exceeded\"") != NULL);
        TEST_ASSERT(strstr(out, "\"param\":\"messages\"") != NULL);
        TEST_ASSERT(strstr(out, "\"n_prompt_tokens\":16") != NULL);
        TEST_ASSERT(strstr(out, "\"n_ctx\":16") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
    request_free(&r);

    request a;
    request_init(&a, REQ_CHAT, 128);
    a.api = API_ANTHROPIC;

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_error_context_length_exceeded(sv[0], false, &a, 20, 20));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "{\"type\":\"error\",\"error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"type\":\"invalid_request_error\"") != NULL);
        TEST_ASSERT(strstr(out, "\"n_prompt_tokens\":20") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
    request_free(&a);
}



static void test_cors_headers_are_opt_in(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_response(sv[0], false, 200, "application/json", "{}"));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "Access-Control-Allow-Origin") == NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] >= 0 && sv[1] >= 0) {
        TEST_ASSERT(http_response(sv[0], true, 200, "application/json", "{}"));
        shutdown(sv[0], SHUT_WR);
        char *out = read_socket_text(sv[1]);
        TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
        TEST_ASSERT(strstr(out, "Access-Control-Allow-Origin: *") != NULL);
        TEST_ASSERT(strstr(out, "Access-Control-Allow-Methods: GET, POST, OPTIONS") != NULL);
        TEST_ASSERT(strstr(out, "Access-Control-Allow-Headers: *") != NULL);
        free(out);
        close(sv[0]);
        close(sv[1]);
    }
}



static void test_cors_preflight_response_is_no_content(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    TEST_ASSERT(http_response(sv[0], true, 204, NULL, ""));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "HTTP/1.1 204 No Content") != NULL);
    TEST_ASSERT(strstr(out, "Content-Length: 0") != NULL);
    TEST_ASSERT(strstr(out, "Content-Type:") == NULL);
    TEST_ASSERT(strstr(out, "Access-Control-Allow-Origin: *") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);
}



static void test_cors_sse_headers(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    TEST_ASSERT(sse_headers(sv[0], true));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "HTTP/1.1 200 OK") != NULL);
    TEST_ASSERT(strstr(out, "Content-Type: text/event-stream") != NULL);
    TEST_ASSERT(strstr(out, "Access-Control-Allow-Origin: *") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);
}



static void test_anthropic_live_stream_sends_incremental_blocks(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.stream = true;
    r.think_mode = PULSAR_THINK_HIGH;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_test", 10, &st));
    const char *raw1 = "need a tool</think>Hello.\n\n";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_test", &st,
                                            raw1, strlen(raw1), false));

    const char *raw =
        "need a tool</think>Hello.\n\n"
        PULSAR_TOOL_CALLS_START "\n";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_test", &st,
                                            raw, strlen(raw), false));

    tool_calls calls = make_swapped_bash_call();
    TEST_ASSERT(anthropic_sse_finish_live(sv[0], NULL, &r, "msg_test", &st,
                                          raw, strlen(raw), &calls,
                                          "tool_calls", 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *msg_start = strstr(out, "event: message_start");
    const char *thinking = strstr(out, "\"thinking\":\"need a tool\"");
    const char *signature = strstr(out, "\"type\":\"signature_delta\"");
    const char *text = strstr(out, "\"text\":\"Hello.\"");
    const char *tool = strstr(out, "\"type\":\"tool_use\"");
    const char *stop = strstr(out, "event: message_stop");
    TEST_ASSERT(msg_start != NULL);
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(signature != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(stop != NULL);
    TEST_ASSERT(msg_start < thinking);
    TEST_ASSERT(thinking < signature);
    TEST_ASSERT(signature < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < stop);
    TEST_ASSERT(strstr(out, PULSAR_TOOL_CALLS_START) == NULL);

    free(out);
    tool_calls_free(&calls);
    anthropic_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_anthropic_tool_stream_sends_live_tool_use(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_tool", 7, &st));

    const char *raw =
        "Before.\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo partial";
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_tool", &st,
                                            raw, strlen(raw), false));

    const char *raw_complete =
        "Before.\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo partial done" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(anthropic_sse_stream_update(sv[0], NULL, &r, "msg_tool", &st,
                                            raw_complete, strlen(raw_complete), false));

    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(raw_complete, false, &parsed_content,
                                           &parsed_reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    apply_anthropic_stream_tool_ids(&calls, &st);
    TEST_ASSERT(calls.v[0].id != NULL);
    TEST_ASSERT(!strncmp(calls.v[0].id, "toolu_", 6));
    TEST_ASSERT(anthropic_sse_finish_live(sv[0], NULL, &r, "msg_tool", &st,
                                          raw_complete, strlen(raw_complete),
                                          &calls, "tool_calls", 5));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *text = strstr(out, "\"text\":\"Before.\"");
    const char *tool = strstr(out, "\"type\":\"tool_use\"");
    const char *key = strstr(out, "\\\"command\\\":\\\"");
    const char *partial = strstr(out, "\"partial_json\":\"echo partial\"");
    const char *rest = strstr(out, "\"partial_json\":\" done\"");
    const char *stop = strstr(out, "event: message_stop");
    int tool_use_count = 0;
    for (const char *p = out; (p = strstr(p, "\"type\":\"tool_use\"")) != NULL; p++) {
        tool_use_count++;
    }
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(key != NULL);
    TEST_ASSERT(partial != NULL);
    TEST_ASSERT(rest != NULL);
    TEST_ASSERT(stop != NULL);
    TEST_ASSERT(strstr(out, calls.v[0].id) != NULL);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < key);
    TEST_ASSERT(key < partial);
    TEST_ASSERT(partial < rest);
    TEST_ASSERT(rest < stop);
    TEST_ASSERT(tool_use_count == 1);
    TEST_ASSERT(strstr(out, PULSAR_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, PULSAR_PARAM_START) == NULL);

    free(out);
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&calls);
    anthropic_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_anthropic_usage_reports_cache_details(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.cache_read_tokens = 7;
    r.cache_write_tokens = 3;

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    TEST_ASSERT(anthropic_final_response(sv[0], false, &r, "msg_usage", "OK", NULL, NULL, "stop", 10, 2));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":0") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_read_input_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_creation_input_tokens\":3") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    anthropic_stream st;
    TEST_ASSERT(anthropic_sse_start_live(sv[0], &r, "msg_usage_stream", 10, &st));
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "event: message_start") != NULL);
    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":0") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":0") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_read_input_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_creation_input_tokens\":3") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);
    request_free(&r);
}



static void test_openai_tool_stream_sends_incremental_text(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_HIGH;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_test", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw1 = "<think>need a tool</think>Hello.\n\n";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st,
                                         raw1, strlen(raw1), false));

    const char *raw =
        "<think>need a tool</think>Hello.\n\n"
        PULSAR_TOOL_CALLS_START "\n";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_test", &st,
                                         raw, strlen(raw), false));

    tool_calls calls = make_swapped_bash_call();
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_test", &st,
                                       raw, strlen(raw), &calls,
                                       "tool_calls", 10, 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *role = strstr(out, "\"role\":\"assistant\"");
    const char *thinking = strstr(out, "\"reasoning_content\":\"need a tool\"");
    const char *text = strstr(out, "\"content\":\"Hello.\"");
    const char *tool = strstr(out, "\"tool_calls\"");
    const char *done = strstr(out, "data: [DONE]");
    TEST_ASSERT(role != NULL);
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(done != NULL);
    TEST_ASSERT(role < thinking);
    TEST_ASSERT(thinking < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < done);
    TEST_ASSERT(strstr(out, PULSAR_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, "<think>") == NULL);

    free(out);
    tool_calls_free(&calls);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_stream_usage_reports_cache_details(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.stream_include_usage = true;
    r.cache_read_tokens = 7;
    r.cache_write_tokens = 3;

    TEST_ASSERT(sse_done(sv[0], &r, "chatcmpl_usage", 10, 2));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"usage\":{\"prompt_tokens\":10") != NULL);
    TEST_ASSERT(strstr(out, "\"completion_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"total_tokens\":12") != NULL);
    TEST_ASSERT(strstr(out, "\"prompt_tokens_details\":{") != NULL);
    TEST_ASSERT(strstr(out, "\"cached_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_write_tokens\":3") != NULL);
    TEST_ASSERT(strstr(out, "data: [DONE]") != NULL);

    free(out);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_responses_usage_reports_cache_details(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.cache_read_tokens = 7;
    r.cache_write_tokens = 3;

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    TEST_ASSERT(responses_final_response(sv[0], false, &r, "resp_usage", "OK", NULL, NULL,
                                         "stop", 10, 2));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":10") != NULL);
    TEST_ASSERT(strstr(out, "\"input_tokens_details\":{") != NULL);
    TEST_ASSERT(strstr(out, "\"cached_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_write_tokens\":3") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"total_tokens\":12") != NULL);

    free(out);
    close(sv[0]);
    close(sv[1]);

    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) {
        request_free(&r);
        return;
    }

    responses_stream st;
    responses_stream_init(&r, &st);
    TEST_ASSERT(responses_sse_completed(sv[0], &r, &st, NULL, NULL,
                                        "stop", 10, 2, 1234));
    shutdown(sv[0], SHUT_WR);
    out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"type\":\"response.completed\"") != NULL);
    TEST_ASSERT(strstr(out, "\"usage\":{\"input_tokens\":10") != NULL);
    TEST_ASSERT(strstr(out, "\"input_tokens_details\":{") != NULL);
    TEST_ASSERT(strstr(out, "\"cached_tokens\":7") != NULL);
    TEST_ASSERT(strstr(out, "\"cache_write_tokens\":3") != NULL);
    TEST_ASSERT(strstr(out, "\"output_tokens\":2") != NULL);
    TEST_ASSERT(strstr(out, "\"total_tokens\":12") != NULL);

    free(out);
    responses_stream_free(&st);
    close(sv[0]);
    close(sv[1]);
    request_free(&r);
}



static void test_openai_chat_stream_splits_reasoning_without_tools(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_HIGH;
    r.has_tools = false;

    TEST_ASSERT(request_uses_structured_stream(&r));
    TEST_ASSERT(request_uses_openai_live_stream(&r));
    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_title", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw1 = "We need to generate a title";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_title", &st,
                                         raw1, strlen(raw1), false));

    const char *raw2 =
        "We need to generate a title</think>Free disk space check";
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_title", &st,
                                       raw2, strlen(raw2), NULL,
                                       "stop", 12, 8));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *role = strstr(out, "\"role\":\"assistant\"");
    const char *reasoning1 = strstr(out, "\"reasoning_content\":\"We need to generate \"");
    const char *reasoning2 = strstr(out, "\"reasoning_content\":\"a title\"");
    const char *content = strstr(out, "\"content\":\"Free disk space check\"");
    const char *done = strstr(out, "data: [DONE]");
    TEST_ASSERT(role != NULL);
    TEST_ASSERT(reasoning1 != NULL);
    TEST_ASSERT(reasoning2 != NULL);
    TEST_ASSERT(content != NULL);
    TEST_ASSERT(done != NULL);
    TEST_ASSERT(role < reasoning1);
    TEST_ASSERT(reasoning1 < reasoning2);
    TEST_ASSERT(reasoning2 < content);
    TEST_ASSERT(content < done);
    TEST_ASSERT(strstr(out, "\"content\":\"We need to generate a title") == NULL);
    TEST_ASSERT(strstr(out, "</think>") == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_sends_partial_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;
    r.tool_orders = make_bash_order();

    TEST_ASSERT(sse_chunk(sv[0], &r, "chatcmpl_partial_tool", NULL, NULL));

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        "Before.\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo partial";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                         raw, strlen(raw), false));

    const char *raw_complete =
        "Before.\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo partial done" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                         raw_complete, strlen(raw_complete), false));

    char *parsed_content = NULL;
    char *parsed_reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(raw_complete, false, &parsed_content, &parsed_reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    apply_openai_stream_tool_ids(&calls, &st);
    TEST_ASSERT(calls.v[0].id != NULL);
    TEST_ASSERT(!strncmp(calls.v[0].id, "call_", 5));
    TEST_ASSERT(openai_sse_finish_live(sv[0], NULL, &r, "chatcmpl_partial_tool", &st,
                                       raw_complete, strlen(raw_complete), &calls,
                                       "tool_calls", 10, 4));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    const char *text = strstr(out, "\"content\":\"Before.\"");
    const char *tool = strstr(out, "\"tool_calls\"");
    const char *key = strstr(out, "\\\"command\\\":\\\"");
    const char *partial = strstr(out, "\"arguments\":\"echo partial\"");
    const char *rest = strstr(out, "\"arguments\":\" done\"");
    int tool_id_count = 0;
    for (const char *p = out; (p = strstr(p, "\"id\":\"call_")) != NULL; p++) tool_id_count++;
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(key != NULL);
    TEST_ASSERT(partial != NULL);
    TEST_ASSERT(rest != NULL);
    TEST_ASSERT(strstr(out, calls.v[0].id) != NULL);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(tool < partial);
    TEST_ASSERT(partial < rest);
    TEST_ASSERT(tool_id_count == 1);
    TEST_ASSERT(strstr(out, PULSAR_TOOL_CALLS_START) == NULL);
    TEST_ASSERT(strstr(out, PULSAR_PARAM_START) == NULL);

    free(out);
    free(parsed_content);
    free(parsed_reasoning);
    tool_calls_free(&calls);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_waits_for_incomplete_tool_tags(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw_invoke = PULSAR_TOOL_CALLS_START "\n" PULSAR_INVOKE_START;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_incomplete_tool", &st,
                                         raw_invoke, strlen(raw_invoke), false));
    TEST_ASSERT(st.mode == OPENAI_STREAM_TOOL);
    TEST_ASSERT(st.tool.state == DSML_TOOL_BETWEEN_INVOKES);

    const char *raw_param =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_incomplete_tool", &st,
                                         raw_param, strlen(raw_param), false));
    TEST_ASSERT(st.mode == OPENAI_STREAM_TOOL);
    TEST_ASSERT(st.tool.state == DSML_TOOL_BETWEEN_PARAMS);

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);
    TEST_ASSERT(strstr(out, "\"name\":\"bash\"") != NULL);
    TEST_ASSERT(strstr(out, PULSAR_PARAM_START) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_sends_partial_raw_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"edits\" string=\"false\">[1,2,3";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_raw_tool", &st,
                                         raw, strlen(raw), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"name\":\"edit\"") != NULL);
    TEST_ASSERT(strstr(out, "\\\"edits\\\":") != NULL);
    TEST_ASSERT(strstr(out, "\"arguments\":\"[1,2,3\"") != NULL);
    TEST_ASSERT(strstr(out, PULSAR_TOOL_CALLS_START) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_holds_partial_dsml_entities(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw_partial =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo &amp";
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_entity_tool", &st,
                                         raw_partial, strlen(raw_partial), false));

    const char *raw_complete =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">echo &amp; done" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_entity_tool", &st,
                                         raw_complete, strlen(raw_complete), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"arguments\":\"echo \"") != NULL);
    TEST_ASSERT(strstr(out, "\"arguments\":\"& done\"") != NULL);
    TEST_ASSERT(strstr(out, "&amp") == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_holds_partial_utf8_arguments(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char prefix[] =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"write\">\n"
        PULSAR_PARAM_START " name=\"content\" string=\"true\">flag ";
    const char suffix[] =
        " done" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    const char flag_utf8[] = {(char)0xf0, (char)0x9f, (char)0x9a, (char)0xa9, 0};
    const char replacement[] = {(char)0xef, (char)0xbf, (char)0xbd, 0};

    buf partial = {0};
    buf_append(&partial, prefix, strlen(prefix));
    buf_putc(&partial, (char)0xf0);
    buf_putc(&partial, (char)0x9f);
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8_tool", &st,
                                         partial.ptr, partial.len, false));

    buf complete = {0};
    buf_append(&complete, prefix, strlen(prefix));
    buf_append(&complete, flag_utf8, 4);
    buf_append(&complete, suffix, strlen(suffix));
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8_tool", &st,
                                         complete.ptr, complete.len, false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"arguments\":\"flag \"") != NULL);
    TEST_ASSERT(strstr(out, flag_utf8) != NULL);
    TEST_ASSERT(strstr(out, replacement) == NULL);

    free(out);
    buf_free(&partial);
    buf_free(&complete);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_openai_tool_stream_handles_multiple_calls(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;
    r.has_tools = true;

    openai_stream st;
    openai_stream_start(&r, &st);
    const char *raw =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"read\">\n"
        PULSAR_PARAM_START " name=\"path\" string=\"true\">a.c" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">wc -l a.c" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_multi_tool", &st,
                                         raw, strlen(raw), false));

    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    int tool_id_count = 0;
    for (const char *p = out; (p = strstr(p, "\"id\":\"call_")) != NULL; p++) tool_id_count++;
    TEST_ASSERT(tool_id_count == 2);
    TEST_ASSERT(strstr(out, "\"name\":\"read\"") != NULL);
    TEST_ASSERT(strstr(out, "\"name\":\"bash\"") != NULL);
    TEST_ASSERT(strstr(out, "\\\"path\\\":") != NULL);
    TEST_ASSERT(strstr(out, "\\\"command\\\":") != NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_streaming_holds_partial_utf8(void) {
    const char partial[] = {'A', ' ', (char)0xf0, (char)0x9f, 0};
    const char complete[] = {'A', ' ', (char)0xf0, (char)0x9f,
                             (char)0x9a, (char)0xa9, ' ', 'd', 'o', 'n', 'e', 0};
    const char flag_done[] = {(char)0xf0, (char)0x9f,
                              (char)0x9a, (char)0xa9, ' ', 'd', 'o', 'n', 'e', 0};
    const char replacement[] = {(char)0xef, (char)0xbf, (char)0xbd, 0};

    TEST_ASSERT(utf8_stream_safe_len(partial, 0, strlen(partial), false) == 2);
    TEST_ASSERT(utf8_stream_safe_len(complete, 0, strlen(complete), false) == strlen(complete));

    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_OPENAI;
    r.stream = true;
    r.think_mode = PULSAR_THINK_NONE;

    openai_stream st;
    openai_stream_start(&r, &st);
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8", &st,
                                         partial, strlen(partial), false));
    TEST_ASSERT(openai_sse_stream_update(sv[0], NULL, &r, "chatcmpl_utf8", &st,
                                         complete, strlen(complete), false));
    shutdown(sv[0], SHUT_WR);
    char *out = read_socket_text(sv[1]);

    TEST_ASSERT(strstr(out, "\"content\":\"A \"") != NULL);
    TEST_ASSERT(strstr(out, flag_done) != NULL);
    TEST_ASSERT(strstr(out, replacement) == NULL);

    free(out);
    openai_stream_free(&st);
    request_free(&r);
    close(sv[0]);
    close(sv[1]);
}



static void test_request_defaults_use_min_p_filtering(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    TEST_ASSERT(r.think_mode == PULSAR_THINK_HIGH);
    TEST_ASSERT(r.temperature == PULSAR_DEFAULT_TEMPERATURE);
    TEST_ASSERT(r.top_p == PULSAR_DEFAULT_TOP_P);
    TEST_ASSERT(r.top_k == 0);
    TEST_ASSERT(r.min_p == PULSAR_DEFAULT_MIN_P);
    TEST_ASSERT(!r.has_temperature);
    TEST_ASSERT(!r.has_top_k);
    TEST_ASSERT(!r.has_top_p);
    TEST_ASSERT(!r.has_min_p);
    request_free(&r);
}



static void check_resolved_sampling(const request *r, float want_temp,
                                    int want_top_k, float want_top_p,
                                    float want_min_p) {
    float temperature = -1.0f, top_p = -1.0f, min_p = -1.0f;
    int top_k = -1;
    gen_resolve_sampling(r, &temperature, &top_k, &top_p, &min_p);
    TEST_ASSERT(temperature == want_temp);
    TEST_ASSERT(top_k == want_top_k);
    TEST_ASSERT(top_p == want_top_p);
    TEST_ASSERT(min_p == want_min_p);
}

/* The sampling contract: engine defaults apply only to parameters the request
 * left absent; anything the client sent explicitly reaches the sampler as-is,
 * including values that happen to equal the defaults. Exercises
 * gen_resolve_sampling (generate.c) over the full matrix of
 * {absent, explicit-nondefault, explicit-equal-to-default} per parameter,
 * with thinking on and off. "Absent" is simulated exactly as the parser
 * leaves it: request_init's default value with the has_ flag false. */
static void test_think_sampling_respects_explicit_params(void) {
    const pulsar_think_mode modes[2] = {PULSAR_THINK_HIGH, PULSAR_THINK_NONE};
    for (int m = 0; m < 2; m++) {
        for (int p = 0; p < 4; p++) {      /* param under test */
            for (int st = 0; st < 3; st++) { /* 0 absent, 1 explicit-nondefault,
                                              * 2 explicit-equal-to-default */
                request r;
                request_init(&r, REQ_CHAT, 128);
                r.think_mode = modes[m];
                float want_temp = PULSAR_DEFAULT_TEMPERATURE;
                int want_top_k = 0;
                float want_top_p = PULSAR_DEFAULT_TOP_P;
                float want_min_p = PULSAR_DEFAULT_MIN_P;
                if (st != 0) {
                    switch (p) {
                    case 0:
                        r.has_temperature = true;
                        r.temperature = st == 1 ? 0.35f : PULSAR_DEFAULT_TEMPERATURE;
                        want_temp = r.temperature;
                        break;
                    case 1:
                        r.has_top_k = true;
                        r.top_k = st == 1 ? 40 : 0;
                        want_top_k = r.top_k;
                        break;
                    case 2:
                        r.has_top_p = true;
                        r.top_p = st == 1 ? 0.9f : PULSAR_DEFAULT_TOP_P;
                        want_top_p = r.top_p;
                        break;
                    case 3:
                        r.has_min_p = true;
                        r.min_p = st == 1 ? 0.0f : PULSAR_DEFAULT_MIN_P;
                        want_min_p = r.min_p;
                        break;
                    }
                }
                check_resolved_sampling(&r, want_temp, want_top_k,
                                        want_top_p, want_min_p);
                request_free(&r);
            }
        }

        /* Explicit temperature EQUAL to the default alongside explicit
         * non-default knobs: the case the old value-only check could not
         * express — with thinking on it clobbered all four; the explicit
         * knobs must survive. */
        request r;
        request_init(&r, REQ_CHAT, 128);
        r.think_mode = modes[m];
        r.temperature = PULSAR_DEFAULT_TEMPERATURE;
        r.has_temperature = true;
        r.top_k = 40;
        r.has_top_k = true;
        r.top_p = 0.9f;
        r.has_top_p = true;
        r.min_p = 0.0f;
        r.has_min_p = true;
        check_resolved_sampling(&r, PULSAR_DEFAULT_TEMPERATURE, 40, 0.9f, 0.0f);

        /* Explicit temperature==0 (others absent): greedy decode must reach
         * the sampler so DSpark speculative decode can engage. */
        request greedy;
        request_init(&greedy, REQ_CHAT, 128);
        greedy.think_mode = modes[m];
        greedy.temperature = 0.0f;
        greedy.has_temperature = true;
        check_resolved_sampling(&greedy, 0.0f, 0, PULSAR_DEFAULT_TOP_P,
                                PULSAR_DEFAULT_MIN_P);
        request_free(&greedy);
        request_free(&r);
    }
}



static void test_reasoning_effort_mapping(void) {
    pulsar_think_mode mode = PULSAR_THINK_NONE;
    TEST_ASSERT(parse_reasoning_effort_name("low", &mode) && mode == PULSAR_THINK_HIGH);
    TEST_ASSERT(parse_reasoning_effort_name("medium", &mode) && mode == PULSAR_THINK_HIGH);
    TEST_ASSERT(parse_reasoning_effort_name("high", &mode) && mode == PULSAR_THINK_HIGH);
    TEST_ASSERT(parse_reasoning_effort_name("xhigh", &mode) && mode == PULSAR_THINK_HIGH);
    TEST_ASSERT(parse_reasoning_effort_name("max", &mode) && mode == PULSAR_THINK_MAX);
    TEST_ASSERT(!parse_reasoning_effort_name("banana", &mode));
    TEST_ASSERT(pulsar_think_mode_for_context(PULSAR_THINK_MAX, 32768) == PULSAR_THINK_HIGH);
    TEST_ASSERT(pulsar_think_mode_for_context(PULSAR_THINK_MAX,
                                           (int)pulsar_think_max_min_context()) == PULSAR_THINK_MAX);
}



static void test_api_thinking_controls_parse(void) {
    bool enabled = true;
    const char *thinking = "{\"type\":\"disabled\",\"budget_tokens\":1024}";
    TEST_ASSERT(parse_thinking_control_value(&thinking, &enabled));
    TEST_ASSERT(!enabled);
    thinking = "true";
    TEST_ASSERT(parse_thinking_control_value(&thinking, &enabled));
    TEST_ASSERT(enabled);

    pulsar_think_mode mode = PULSAR_THINK_HIGH;
    const char *anth_effort = "{\"effort\":\"max\",\"other\":true}";
    TEST_ASSERT(parse_output_config_effort(&anth_effort, &mode));
    TEST_ASSERT(mode == PULSAR_THINK_MAX);

    const char *openai_effort = "\"xhigh\"";
    mode = PULSAR_THINK_HIGH;
    TEST_ASSERT(parse_reasoning_effort_value(&openai_effort, &mode));
    TEST_ASSERT(mode == PULSAR_THINK_HIGH);
}



static void test_render_think_max_prompt_prefix(void) {
    chat_msgs msgs = {0};
    chat_msg sys = {0};
    sys.role = xstrdup("system");
    sys.content = xstrdup("You are terse.");
    chat_msgs_push(&msgs, sys);
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_MAX);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(!strncmp(prompt, "<｜begin▁of▁sentence｜>", strlen("<｜begin▁of▁sentence｜>")));
    TEST_ASSERT(strstr(prompt, pulsar_think_max_prefix()) != NULL);
    TEST_ASSERT(strstr(prompt, "You are terse.<｜User｜>Hello<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(prompt, "</think>") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_render_non_thinking_prompt_closes_think(void) {
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_NONE);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, pulsar_think_max_prefix()) == NULL);
    TEST_ASSERT(strstr(prompt, "<｜User｜>Hello<｜Assistant｜></think>") != NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_render_drops_old_reasoning_without_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("old hidden reasoning");
    assistant.content = xstrdup("first answer");
    chat_msgs_push(&msgs, assistant);
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("second");
    chat_msgs_push(&msgs, user2);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "old hidden reasoning") == NULL);
    TEST_ASSERT(strstr(prompt, "<｜Assistant｜></think>first answer") != NULL);
    TEST_ASSERT(strstr(prompt, "<｜User｜>second<｜Assistant｜><think>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_render_preserves_reasoning_with_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("tool reasoning");
    assistant.content = xstrdup("");
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    char *prompt = render_chat_prompt_text(&msgs, "{}", NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>tool reasoning</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>/tmp</tool_result>") != NULL);
    free(prompt);

    prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>tool reasoning</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>/tmp</tool_result>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_render_chat_prompt_text_renders_tools_before_system(void) {
    /* The tool-schema block must sit at the head of the system region so the
     * client's system content stays at the tail, right before <｜User｜>.
     * That keeps a per-request dynamic tail (e.g. a timestamp) out of the
     * cached prefix without losing the tool schemas to the trim. */
    chat_msgs msgs = {0};
    chat_msg sys = {0};
    sys.role = xstrdup("system");
    sys.content = xstrdup("CLIENT_SYSTEM_MARKER");
    chat_msgs_push(&msgs, sys);
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, "TOOL_SCHEMA_MARKER", NULL,
                                           PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    const char *tools  = strstr(prompt, "## Tools");
    const char *client = strstr(prompt, "CLIENT_SYSTEM_MARKER");
    const char *user_m = strstr(prompt, "<｜User｜>");
    TEST_ASSERT(tools && client && user_m);
    TEST_ASSERT(tools  < client);
    TEST_ASSERT(client < user_m);
    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_dsml_tool_args_preserve_call_order(void) {
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_dsml_tool_calls_text(&b, &calls);
    const char *command = strstr(b.ptr, "name=\"command\"");
    const char *description = strstr(b.ptr, "name=\"description\"");
    const char *timeout = strstr(b.ptr, "name=\"timeout\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(command < timeout);
    buf_free(&b);
    tool_calls_free(&calls);
}



static void test_openai_tool_args_preserve_call_order(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.tool_orders = make_bash_order();
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_tool_calls_json(&b, &calls, "test", &r.tool_orders);
    const char *command = strstr(b.ptr, "\\\"command\\\"");
    const char *description = strstr(b.ptr, "\\\"description\\\"");
    const char *timeout = strstr(b.ptr, "\\\"timeout\\\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(command < timeout);
    buf_free(&b);
    tool_calls_free(&calls);
    request_free(&r);
}



static void test_anthropic_thinking_and_tool_args_preserve_call_order(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.tool_orders = make_bash_order();
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_anthropic_content(&b, "done", "thinking text", &calls, "msg_1", &r.tool_orders);
    const char *thinking = strstr(b.ptr, "\"type\":\"thinking\"");
    const char *text = strstr(b.ptr, "\"type\":\"text\"");
    const char *tool = strstr(b.ptr, "\"type\":\"tool_use\"");
    const char *command = strstr(b.ptr, "\"command\"");
    const char *description = strstr(b.ptr, "\"description\"");
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(thinking < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(description < command);
    buf_free(&b);
    tool_calls_free(&calls);
    request_free(&r);
}



static void test_parse_short_dsml_and_canonical_suffix(void) {
    const char *generated =
        "<think>need a tool</think>"
        "<DSML｜tool_calls>\n"
        "<DSML｜invoke name=\"bash\">\n"
        "<DSML｜parameter name=\"description\" string=\"true\">list files</DSML｜parameter>\n"
        "<DSML｜parameter name=\"command\" string=\"true\">ls -la</DSML｜parameter>\n"
        "</DSML｜invoke>\n"
        "</DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a tool"));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.tool_orders = make_bash_order();
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    const char *command = strstr(suffix, "name=\"command\"");
    const char *description = strstr(suffix, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(strstr(suffix, "</think>") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜end▁of▁sentence｜>") != NULL);

    free(suffix);
    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    request_free(&r);
}



static void test_dsml_parser_recovers_loose_nested_parameters(void) {
    const char *generated =
        "review done\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"path\">/private/tmp/tetris.c" PULSAR_PARAM_END "\n"
        PULSAR_PARAM_START " name=\"edits\">\n"
        PULSAR_PARAM_START " name=\"oldText\" string=\"true\">old &lt;text&gt;" PULSAR_PARAM_END "\n"
        PULSAR_PARAM_START " name=\"newText\" string=\"true\">new text" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(content && !strcmp(content, "review done"));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "edit"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/private/tmp/tetris.c\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"edits\": {") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"oldText\":\"old <text>\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"newText\":\"new text\"") != NULL);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}



/* Verify that try_repair_dsml + parse_generated_message produces structurally
   valid tool calls for all three DSML styles and multiple truncation scenarios.
   Balanced but malformed DSML is not repaired: the model must retry it.
   This tests repair ACCURACY, not just that it doesn't crash. */
static void test_dsml_repair_produces_parseable_calls(void) {
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    buf repaired = {0};

    /* === TEST 1: Full DSML - missing </tool_calls> === */
    {
        const char *broken =
            "thinking done\n\n"
            PULSAR_TOOL_CALLS_START "\n"
            PULSAR_INVOKE_START " name=\"bash\">\n"
            PULSAR_PARAM_START " name=\"command\" string=\"true\">ls -la" PULSAR_PARAM_END "\n"
            PULSAR_INVOKE_END "\n";
        /* Missing: PULSAR_TOOL_CALLS_END */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"ls -la\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 2: Full DSML - missing </invoke> and </tool_calls> === */
    {
        const char *broken =
            "\n\n"
            PULSAR_TOOL_CALLS_START "\n"
            PULSAR_INVOKE_START " name=\"edit\">\n"
            PULSAR_PARAM_START " name=\"path\" string=\"true\">/tmp/test.c" PULSAR_PARAM_END "\n";
        /* Missing: PULSAR_INVOKE_END, PULSAR_TOOL_CALLS_END */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "edit"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/tmp/test.c\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 3: Full DSML - missing </parameter> === */
    {
        const char *broken =
            "\n\n"
            PULSAR_TOOL_CALLS_START "\n"
            PULSAR_INVOKE_START " name=\"bash\">\n"
            PULSAR_PARAM_START " name=\"command\" string=\"true\">echo hello";
        /* Missing: PULSAR_PARAM_END, PULSAR_INVOKE_END, PULSAR_TOOL_CALLS_END */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"echo hello\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 4: Short DSML - missing closing tags === */
    {
        const char *broken =
            "\n\n"
            PULSAR_TOOL_CALLS_START_SHORT "\n"
            PULSAR_INVOKE_START_SHORT " name=\"write_file\">\n"
            PULSAR_PARAM_START_SHORT " name=\"path\" string=\"true\">/tmp/out.txt" PULSAR_PARAM_END_SHORT "\n"
            PULSAR_PARAM_START_SHORT " name=\"content\" string=\"true\">hello world" PULSAR_PARAM_END_SHORT "\n"
            PULSAR_INVOKE_END_SHORT "\n";
        /* Missing: PULSAR_TOOL_CALLS_END_SHORT */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "write_file"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/tmp/out.txt\"") != NULL);
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"content\": \"hello world\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 5: Plain XML - missing closing tags === */
    {
        const char *broken =
            "\n\n"
            "<tool_calls>\n"
            "<invoke name=\"execute_command\">\n"
            "<parameter name=\"command\" string=\"true\">pwd</parameter>\n"
            "</invoke>\n";
        /* Missing: </tool_calls> */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "execute_command"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"pwd\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 6: Balanced text should NOT be modified === */
    {
        const char *balanced =
            "\n\n"
            PULSAR_TOOL_CALLS_START "\n"
            PULSAR_INVOKE_START " name=\"bash\">\n"
            PULSAR_PARAM_START " name=\"command\" string=\"true\">ls" PULSAR_PARAM_END "\n"
            PULSAR_INVOKE_END "\n"
            PULSAR_TOOL_CALLS_END;

        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced, strlen(balanced), &repaired));
        /* No repair needed */
    }

    /* === TEST 7: No DSML tags should return false === */
    {
        const char *no_dsml = "just plain text, no tools";
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(no_dsml, strlen(no_dsml), &repaired));
    }

    /* === TEST 8: Balanced DSML with no invoke is not repaired === */
    {
        const char *balanced_no_invoke =
            "Let me analyze this.\n\n"
            PULSAR_TOOL_CALLS_START
            "The write tool truncates this too, at what looks like the same content location."
            PULSAR_TOOL_CALLS_END;
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced_no_invoke, strlen(balanced_no_invoke), &repaired));
    }

    /* === TEST 9: Balanced short DSML with no invoke is not repaired === */
    {
        const char *balanced_short_no_invoke =
            "thinking...\n\n"
            PULSAR_TOOL_CALLS_START_SHORT
            "some content here"
            PULSAR_TOOL_CALLS_END_SHORT;
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced_short_no_invoke, strlen(balanced_short_no_invoke), &repaired));
    }

    /* === TEST 10: Balanced plain XML DSML with no invoke is not repaired === */
    {
        const char *balanced_xml_no_invoke =
            "Let me think.\n\n"
            "<tool_calls>"
            "I need to use a tool but I don't know which one."
            "</tool_calls>";
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced_xml_no_invoke, strlen(balanced_xml_no_invoke), &repaired));
    }

    /* === TEST 11: DSML mentioned inside thinking is not repaired === */
    {
        const char *thinking_quote =
            "<think>The protocol uses "
            PULSAR_TOOL_CALLS_START
            "some explanatory text"
            PULSAR_TOOL_CALLS_END
            ", but this is only a quote.</think>\nFinal answer.";
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(thinking_quote, strlen(thinking_quote), &repaired));
    }

    /* === TEST 12: Extra closing tags are unrecoverable, not truncation === */
    {
        const char *orphan_close =
            "done\n\n"
            PULSAR_TOOL_CALLS_START
            PULSAR_TOOL_CALLS_END
            PULSAR_TOOL_CALLS_END;
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(orphan_close, strlen(orphan_close), &repaired));
    }

    /* === TEST 13: Real DSML after thinking still repairs normally === */
    {
        const char *broken_after_think =
            "<think>"
            PULSAR_TOOL_CALLS_START
            "quoted DSML, not executable"
            PULSAR_TOOL_CALLS_END
            "</think>\n\n"
            PULSAR_TOOL_CALLS_START "\n"
            PULSAR_INVOKE_START " name=\"bash\">\n"
            PULSAR_PARAM_START " name=\"command\" string=\"true\">date" PULSAR_PARAM_END "\n"
            PULSAR_INVOKE_END "\n";
        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken_after_think, strlen(broken_after_think), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, true, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"date\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    buf_free(&repaired);
}



static void test_tool_parse_failure_returns_recoverable_finish(void) {
    const char *generated =
        "trying a tool\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START ">\n"
        PULSAR_TOOL_CALLS_END;

    char err[128] = {0};
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    const char *finish = "tool_calls";
    bool recovered = false;

    TEST_ASSERT(!parse_generated_message_for_response(generated,
                                                       true,
                                                       true,
                                                       false,
                                                       &finish,
                                                       err,
                                                       sizeof(err),
                                                       &content,
                                                       &reasoning,
                                                       &calls,
                                                       &recovered));
    TEST_ASSERT(recovered);
    TEST_ASSERT(!strcmp(finish, "stop"));
    TEST_ASSERT(!strcmp(err, "invalid tool call"));
    TEST_ASSERT(content && strstr(content, PULSAR_TOOL_CALLS_START) != NULL);
    TEST_ASSERT(reasoning == NULL);
    TEST_ASSERT(calls.len == 0);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}



static void test_invalid_dsml_tool_error_suffix_includes_system_prompt(void) {
    request r = {};
    r.think_mode = PULSAR_THINK_HIGH;
    r.prompt_text = xstrdup(
        "<｜begin▁of▁sentence｜>"
        "## Tools\nschema\n\nSystem rule\n\n"
        "<｜User｜>Hi<｜Assistant｜><think>");
    thinking_state st = {.inside = true};

    char *suffix = build_invalid_dsml_tool_error_suffix(&r, &st, "missing invoke name");
    TEST_ASSERT(suffix != NULL);
    TEST_ASSERT(strstr(suffix, "</think><｜end▁of▁sentence｜><｜User｜><tool_result>") == suffix);
    TEST_ASSERT(strstr(suffix, "Tool error: invalid DSML tool call: missing invoke name") != NULL);
    TEST_ASSERT(strstr(suffix, "The previous assistant output was not executed") != NULL);
    TEST_ASSERT(strstr(suffix, "System prompt reminder:\n## Tools\nschema\n\nSystem rule") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜User｜>Hi") == NULL);
    TEST_ASSERT(strstr(suffix, "</tool_result><｜Assistant｜><think>") != NULL);

    free(suffix);
    free(r.prompt_text);
}



static void test_thinking_dsml_is_not_executable_before_think_close(void) {
    const char *generated =
        "<think>I might mention a malformed or tentative tool call here:\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">true" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END
        "\nBut it is still reasoning, not an assistant action.</think>Final answer.";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, true,
                                           &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 0);
    TEST_ASSERT(reasoning && strstr(reasoning, PULSAR_TOOL_CALLS_START) != NULL);
    TEST_ASSERT(content && !strcmp(content, "Final answer."));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}



static void test_thinking_dsml_after_think_close_is_executable(void) {
    const char *generated =
        "<think>need a shell check</think>\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"bash\">\n"
        PULSAR_PARAM_START " name=\"command\" string=\"true\">pwd" PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, true,
                                           &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a shell check"));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"pwd\"") != NULL);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}



static void test_tool_checkpoint_suffix_is_future_prompt_canonical(void) {
    tool_schema_orders orders = make_bash_order();
    const char *tool_schemas =
        "{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{},\"description\":{},\"timeout\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("inspect");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, tool_schemas,
                                                &orders, PULSAR_THINK_HIGH);

    const char *generated =
        "need a tool</think>\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">cd /tmp && git diff 2>/dev/null</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(strstr(calls.v[0].arguments, "cd /tmp && git diff 2>/dev/null") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "&amp;&amp;") == NULL);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    TEST_ASSERT(strstr(suffix, "cd /tmp && git diff 2>/dev/null") != NULL);
    TEST_ASSERT(strstr(suffix, "&amp;&amp;") == NULL);
    TEST_ASSERT(strstr(suffix, "2&gt;/dev/null") == NULL);
    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("inspect");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);
    char *future_prompt = render_chat_prompt_text(&history_msgs, tool_schemas,
                                                  &r.tool_orders, PULSAR_THINK_HIGH);

    TEST_ASSERT(!strcmp(canonical.ptr, future_prompt));

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}



static void test_tool_checkpoint_minifies_json_parameters(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{},\"edits\":{}}}}");
    const char *tool_schemas =
        "{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{},\"edits\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("edit");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, tool_schemas,
                                                &orders, PULSAR_THINK_HIGH);

    const char *generated =
        "need edit</think>\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"edit\">\n"
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/file</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"edits\" string=\"false\">"
        "[{\"oldText\": \"status=created\", \"newText\": \"status=created\\nstatus2=resumed\"}]"
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("edit");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);
    char *future_prompt = render_chat_prompt_text(&history_msgs, tool_schemas,
                                                  &r.tool_orders, PULSAR_THINK_HIGH);

    TEST_ASSERT(!strcmp(canonical.ptr, future_prompt));

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}



static void test_tool_memory_replays_sampled_dsml(void) {
    const char *generated =
        "<think>need shell</think>\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls sampled = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &sampled));
    TEST_ASSERT(sampled.len == 1);

    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);
    s.assign_tool_call_ids(&sampled, API_OPENAI);
    TEST_ASSERT(sampled.v[0].id != NULL);
    TEST_ASSERT(!strncmp(sampled.v[0].id, "call_", 5));
    s.tool_memory_remember(&sampled);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    tool_call tc = {0};
    tc.id = xstrdup(sampled.v[0].id);
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    tool_replay_stats stats = {0};
    s.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(stats.mem == 1);
    TEST_ASSERT(stats.disk == 0);
    TEST_ASSERT(stats.canonical == 0);
    TEST_ASSERT(stats.missing_ids == 0);
    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    const char *command = strstr(prompt, "name=\"command\"");
    const char *timeout = strstr(prompt, "name=\"timeout\"");
    const char *description = strstr(prompt, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(command < timeout);
    TEST_ASSERT(timeout < description);

    free(prompt);
    chat_msgs_free(&msgs);
    free(content);
    free(reasoning);
    tool_calls_free(&sampled);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_anthropic_tool_memory_replays_sampled_dsml(void) {
    const char *sampled_dsml =
        "\n\n" PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"Bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        PULSAR_TOOL_CALLS_END;

    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);
    s.tool_memory_put("toolu_exact", sampled_dsml);

    const char *json =
        "["
        "{\"role\":\"assistant\",\"content\":["
        "{\"type\":\"tool_use\",\"id\":\"toolu_exact\",\"name\":\"Bash\","
        "\"input\":{\"description\":\"list files\",\"command\":\"ls -la\"}}"
        "]},"
        "{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_exact\",\"content\":\"ok\"}"
        "]}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 2);
    TEST_ASSERT(msgs.v[1].tool_call_id && !strcmp(msgs.v[1].tool_call_id, "toolu_exact"));

    stop_list ids = {0};
    collect_tool_call_ids(&msgs, &ids);
    TEST_ASSERT(id_list_contains(&ids, "toolu_exact"));
    id_list_free(&ids);

    tool_replay_stats stats = {0};
    s.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(stats.mem == 1);
    TEST_ASSERT(stats.canonical == 0);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    const char *command = strstr(prompt, "name=\"command\"");
    const char *description = strstr(prompt, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(command < description);

    free(prompt);
    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_anthropic_live_tail_renders_tool_results_only(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.think_mode = PULSAR_THINK_HIGH;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("toolu_live");
    tc.name = xstrdup("Bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>/tmp</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_live");
    chat_msgs_push(&msgs, user);

    /* Anthropic system text is parsed separately and appended to chat_msgs for
     * rendering.  The live-tail finder must ignore it when locating the final
     * tool_result run. */
    chat_msg system = {0};
    system.role = xstrdup("system");
    system.content = xstrdup("You are terse.");
    chat_msgs_push(&msgs, system);

    anthropic_prepare_live_continuation(&r, &msgs);
    TEST_ASSERT(r.anthropic_live_call_ids.len == 1);
    TEST_ASSERT(!strcmp(r.anthropic_live_call_ids.v[0], "toolu_live"));
    TEST_ASSERT(r.anthropic_live_suffix_text != NULL);
    TEST_ASSERT(!strncmp(r.anthropic_live_suffix_text,
                         "<｜end▁of▁sentence｜><｜User｜><tool_result>",
                         strlen("<｜end▁of▁sentence｜><｜User｜><tool_result>")));
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "/tmp</tool_result>") != NULL);
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "Bash") == NULL);

    chat_msgs_free(&msgs);
    request_free(&r);
}



static void test_anthropic_tool_result_id_validation(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>out</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_missing");
    chat_msgs_push(&msgs, user);

    char err[160] = {0};
    TEST_ASSERT(!s.anthropic_validate_tool_results(&msgs, NULL,
                                                 err, sizeof(err)));
    TEST_ASSERT(strstr(err, "Anthropic continuation state is not available") != NULL);

    pthread_mutex_lock(&s.tool_mu);
    s.n_slots = 1; /* live bindings are per-slot; has_call_id scans slots */
    s.slots[0].anthropic_live.valid = true;
    s.slots[0].anthropic_live.live_tokens = 10;
    id_list_push_unique(&s.slots[0].anthropic_live.call_ids, "toolu_missing");
    pthread_mutex_unlock(&s.tool_mu);
    bool needs_live_tool_state = false;
    err[0] = '\0';
    TEST_ASSERT(s.anthropic_validate_tool_results(&msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.slots[0].anthropic_live);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_anthropic_full_replay_allows_unknown_live_id(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("toolu_replay");
    tc.name = xstrdup("Bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>/tmp</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_replay");
    chat_msgs_push(&msgs, user);

    bool needs_live_tool_state = false;
    char err[160] = {0};
    TEST_ASSERT(s.anthropic_validate_tool_results(&msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);

    chat_msgs_free(&msgs);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_anthropic_tool_use_parses_before_role(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    /* GitHub #127 regression: Crush can replay full Anthropic history with
     * message objects serialized as {"content": ..., "role": ...}.  The parser
     * must still remember prior assistant tool_use ids, otherwise old
     * tool_result blocks are mistaken for live-only continuations and rejected
     * once the live frontier has moved on to newer tool calls. */
    pthread_mutex_lock(&s.tool_mu);
    s.n_slots = 1;
    s.slots[0].anthropic_live.valid = true;
    s.slots[0].anthropic_live.live_tokens = 100;
    id_list_push_unique(&s.slots[0].anthropic_live.call_ids, "toolu_current");
    pthread_mutex_unlock(&s.tool_mu);

    const char *json =
        "["
        "{\"content\":["
        "{\"type\":\"tool_use\",\"id\":\"toolu_old\",\"name\":\"Bash\","
        "\"input\":{\"command\":\"ls\"}}"
        "],\"role\":\"assistant\"},"
        "{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_old\",\"content\":\"ok\"}"
        "]},"
        "{\"role\":\"user\",\"content\":\"continue\"}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 3);
    TEST_ASSERT(msgs.v[0].calls.len == 1);
    TEST_ASSERT(msgs.v[0].calls.v[0].id &&
                !strcmp(msgs.v[0].calls.v[0].id, "toolu_old"));

    bool needs_live_tool_state = false;
    char err[160] = {0};
    TEST_ASSERT(s.anthropic_validate_tool_results(&msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.slots[0].anthropic_live);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_tool_checkpoint_canonicalization_gate_exact_replay(void) {
    server s;
    memset(&s, 0, sizeof(s));

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_exact");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{}");
    tool_calls_push(&calls, tc);
    calls.raw_dsml = xstrdup(
        "\n\n" PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "</｜DSML｜invoke>\n"
        PULSAR_TOOL_CALLS_END);

    TEST_ASSERT(!s.should_canonicalize_tool_checkpoint(&calls));

    s.disable_exact_dsml_tool_replay = true;
    TEST_ASSERT(s.should_canonicalize_tool_checkpoint(&calls));

    s.disable_exact_dsml_tool_replay = false;
    free(calls.raw_dsml);
    calls.raw_dsml = NULL;
    TEST_ASSERT(s.should_canonicalize_tool_checkpoint(&calls));

    tool_calls_free(&calls);
}



static void test_responses_live_tail_renders_tool_outputs_only(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.think_mode = PULSAR_THINK_HIGH;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_live");
    tc.name = xstrdup("exec_command");
    tc.arguments = xstrdup("{\"cmd\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_live");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    responses_prepare_live_continuation(&r, &msgs);
    TEST_ASSERT(r.responses_live_call_ids.len == 1);
    TEST_ASSERT(!strcmp(r.responses_live_call_ids.v[0], "call_live"));
    TEST_ASSERT(r.responses_live_suffix_text != NULL);
    TEST_ASSERT(!strncmp(r.responses_live_suffix_text,
                         "<｜end▁of▁sentence｜><｜User｜><tool_result>",
                         strlen("<｜end▁of▁sentence｜><｜User｜><tool_result>")));
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "/tmp</tool_result>") != NULL);
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "exec_command") == NULL);

    chat_msgs_free(&msgs);
    request_free(&r);
}



static void test_responses_tool_output_id_validation(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_missing");
    tool.content = xstrdup("out");
    chat_msgs_push(&msgs, tool);

    char err[160] = {0};
    TEST_ASSERT(!s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH, NULL, NULL,
                                                 err, sizeof(err)));
    TEST_ASSERT(strstr(err, "Responses continuation state is not available") != NULL);

    pthread_mutex_lock(&s.tool_mu);
    s.n_slots = 1;
    s.slots[0].responses_live.valid = true;
    s.slots[0].responses_live.live_tokens = 10;
    id_list_push_unique(&s.slots[0].responses_live.call_ids, "call_missing");
    pthread_mutex_unlock(&s.tool_mu);
    err[0] = '\0';
    bool needs_live_tool_state = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH,
                                                &needs_live_tool_state, NULL,
                                                err, sizeof(err)));
    TEST_ASSERT(needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.slots[0].responses_live);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_responses_stateless_tool_replay_requires_reasoning(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_replay");
    tc.name = xstrdup("exec_command");
    tc.arguments = xstrdup("{\"cmd\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_replay");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    char err[160] = {0};
    bool needs_live_reasoning = false;
    bool needs_live_tool_state = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(needs_live_reasoning);

    pthread_mutex_lock(&s.tool_mu);
    s.n_slots = 1;
    s.slots[0].responses_live.valid = true;
    s.slots[0].responses_live.live_tokens = 123;
    id_list_push_unique(&s.slots[0].responses_live.call_ids, "call_replay");
    pthread_mutex_unlock(&s.tool_mu);
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(needs_live_reasoning);

    free(msgs.v[0].reasoning);
    msgs.v[0].reasoning = xstrdup("replayed hidden reasoning");
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(!needs_live_reasoning);

    free(msgs.v[0].reasoning);
    msgs.v[0].reasoning = NULL;
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(s.responses_validate_tool_outputs(&msgs, PULSAR_THINK_NONE,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(!needs_live_reasoning);

    chat_msgs_free(&msgs);
    live_tool_state_free(&s.slots[0].responses_live);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_responses_visible_suffix_matches_client_replay(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.think_mode = PULSAR_THINK_HIGH;
    r.reasoning_summary_emit = true;

    char *suffix = build_responses_visible_assistant_suffix(&r, "5",
                                                            "hidden summary",
                                                            NULL);
    TEST_ASSERT(strstr(suffix, "hidden summary") == NULL);
    TEST_ASSERT(strstr(suffix, "</think>5") != NULL);
    free(suffix);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_live");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&calls, tc);

    suffix = build_responses_visible_assistant_suffix(&r, "",
                                                      "tool summary",
                                                      &calls);
    TEST_ASSERT(strstr(suffix, "tool summary</think>") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜DSML｜tool_calls>") != NULL);
    free(suffix);

    tool_calls_free(&calls);
    request_free(&r);
}



static void test_exact_dsml_tool_replay_can_be_disabled(void) {
    const char *dsml =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    s.tool_memory_put("call_disabled", dsml);
    s.disable_exact_dsml_tool_replay = true;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_disabled");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"canonical\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    tool_replay_stats stats = {0};
    s.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml == NULL);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);

    FILE *fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    uint64_t bytes = 123;
    TEST_ASSERT(s.kv_tool_map_write(fp, dsml, &bytes));
    TEST_ASSERT(bytes == 0);

    if (fp) fclose(fp);
    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_dsml_decode_state_separates_structure_and_payload(void) {
    dsml_decode_tracker tracker;
    dsml_decode_tracker_init(&tracker);

    const char *prefix =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n";
    TEST_ASSERT(dsml_decode_state_for_text(prefix, strlen(prefix)) ==
                DSML_DECODE_STRUCTURAL);
    dsml_decode_tracker_update(&tracker, prefix, strlen(prefix));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRUCTURAL);

    const char *path_param =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"path\" string=\"true\">/tmp/a.py";
    TEST_ASSERT(dsml_decode_state_for_text(path_param, strlen(path_param)) ==
                DSML_DECODE_STRING_BODY);
    dsml_decode_tracker_update(&tracker, path_param, strlen(path_param));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRING_BODY);

    const char *path_closing =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"path\" string=\"true\">/tmp/a.py</";
    TEST_ASSERT(dsml_decode_state_for_text(path_closing, strlen(path_closing)) ==
                DSML_DECODE_STRUCTURAL);
    dsml_decode_tracker_update(&tracker, path_closing, strlen(path_closing));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRUCTURAL);

    const char *json_struct =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"edits\" string=\"false\">[{";
    TEST_ASSERT(dsml_decode_state_for_text(json_struct, strlen(json_struct)) ==
                DSML_DECODE_JSON_STRUCTURAL);
    dsml_decode_tracker_init(&tracker);
    dsml_decode_tracker_update(&tracker, json_struct, strlen(json_struct));
    TEST_ASSERT(tracker.decode == DSML_DECODE_JSON_STRUCTURAL);

    const char *json_string =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"edits\" string=\"false\">[{\"newText\":\"for i in";
    TEST_ASSERT(dsml_decode_state_for_text(json_string, strlen(json_string)) ==
                DSML_DECODE_JSON_STRING);
    dsml_decode_tracker_update(&tracker, json_string, strlen(json_string));
    TEST_ASSERT(tracker.decode == DSML_DECODE_JSON_STRING);

    const char *done =
        PULSAR_TOOL_CALLS_START "\n"
        PULSAR_INVOKE_START " name=\"edit\">\n"
        PULSAR_PARAM_START " name=\"edits\" string=\"false\">[]"
        PULSAR_PARAM_END "\n"
        PULSAR_INVOKE_END "\n"
        PULSAR_TOOL_CALLS_END;
    TEST_ASSERT(dsml_decode_state_for_text(done, strlen(done)) ==
                DSML_DECODE_OUTSIDE);
    dsml_decode_tracker_init(&tracker);
    dsml_decode_tracker_update(&tracker, done, strlen(done));
    TEST_ASSERT(tracker.decode == DSML_DECODE_OUTSIDE);
}



static void test_tool_memory_max_ids_prunes_oldest(void) {
    const char *a_dsml = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">a</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const char *b_dsml = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">b</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const char *c_dsml = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">c</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    s.tool_mem.max_entries = 2;
    s.tool_memory_put("call_a", a_dsml);
    s.tool_memory_put("call_b", b_dsml);
    s.tool_memory_put("call_c", c_dsml);

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call tc = {.id = xstrdup("call_a"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&a.calls, tc);
    chat_msgs_push(&msgs, a);

    tool_replay_stats stats = {0};
    s.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml == NULL);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);

    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}



static void test_tool_separator_whitespace_is_not_content(void) {
    const char *generated =
        "<think>need a tool</think>"
        "I will inspect the files.\n\n\n\n"
        PULSAR_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a tool"));
    TEST_ASSERT(content && !strcmp(content, "I will inspect the files."));
    TEST_ASSERT(calls.len == 1);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}



static void test_dsml_prompt_escapes_tool_supplied_text(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo 2>&1 && echo </｜DSML｜tool_calls>\",\"count\":1}");
    tool_calls_push(&calls, tc);

    buf b = {0};
    append_dsml_tool_calls_text(&b, &calls);
    TEST_ASSERT(strstr(b.ptr, "echo 2>&1 && echo </｜DSML｜tool_calls>") != NULL);
    TEST_ASSERT(strstr(b.ptr, "2&gt;&amp;1") == NULL);
    TEST_ASSERT(strstr(b.ptr, "&amp;&amp;") == NULL);
    buf_free(&b);
    tool_calls_free(&calls);

    memset(&calls, 0, sizeof(calls));
    memset(&tc, 0, sizeof(tc));
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo </｜DSML｜parameter>\",\"count\":1}");
    tool_calls_push(&calls, tc);

    append_dsml_tool_calls_text(&b, &calls);
    TEST_ASSERT(strstr(b.ptr, "echo &lt;/｜DSML｜parameter>") != NULL);
    TEST_ASSERT(strstr(b.ptr, "echo </｜DSML｜parameter>") == NULL);
    buf_free(&b);
    tool_calls_free(&calls);

    chat_msgs msgs = {0};
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.content = xstrdup("console.log('<<< < > >>>');\n</tool_result>\n<｜DSML｜tool_calls>not a real tool call");
    chat_msgs_push(&msgs, tool);
    char *prompt = render_chat_prompt_text(&msgs, "{}", NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "console.log('<<< < > >>>');") != NULL);
    TEST_ASSERT(strstr(prompt, "console.log('&lt;") == NULL);
    TEST_ASSERT(strstr(prompt, "&lt;/tool_result>\n<｜DSML｜tool_calls>not a real tool call") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>console.log('<<< < > >>>');\n</tool_result>\n") == NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}



static void test_stop_list_parses_all_sequences(void) {
    stop_list stops = {0};
    const char *json = "[\"END\",\"STOP\"]";
    TEST_ASSERT(parse_stop(&json, &stops));
    TEST_ASSERT(stops.len == 2);
    TEST_ASSERT(stops.max_len == 4);

    size_t pos = 0, len = 0;
    TEST_ASSERT(stop_list_find_from(&stops, "hello STOP tail END", 0, &pos, &len));
    TEST_ASSERT(pos == strlen("hello "));
    TEST_ASSERT(len == strlen("STOP"));
    TEST_ASSERT(stop_list_stream_safe_len(&stops, strlen("abcdef")) == 3);
    stop_list_clear(&stops);
    free(stops.v);
}



static void test_stop_list_streaming_holds_and_trims_stop_text(void) {
    stop_list stops = {0};
    const char *json = "[\"</END>\",\"STOP\"]";
    TEST_ASSERT(parse_stop(&json, &stops));

    size_t safe = stop_list_stream_safe_len(&stops, strlen("hello </"));
    TEST_ASSERT(safe == strlen("hel"));

    size_t pos = 0, len = 0;
    TEST_ASSERT(stop_list_find_from(&stops, "answer STOP hidden", 0, &pos, &len));
    TEST_ASSERT(pos == strlen("answer "));
    TEST_ASSERT(len == strlen("STOP"));

    stop_list_clear(&stops);
    free(stops.v);
}



static char *test_nested_json_array(int depth) {
    buf b = {0};
    for (int i = 0; i < depth; i++) buf_putc(&b, '[');
    buf_putc(&b, '0');
    for (int i = 0; i < depth; i++) buf_putc(&b, ']');
    return buf_take(&b);
}



static void test_json_skip_has_nesting_limit(void) {
    char *ok = test_nested_json_array(JSON_MAX_NESTING);
    const char *p = ok;
    TEST_ASSERT(json_skip_value(&p));
    TEST_ASSERT(*p == '\0');
    free(ok);

    char *bad = test_nested_json_array(JSON_MAX_NESTING + 1);
    p = bad;
    TEST_ASSERT(!json_skip_value(&p));
    free(bad);
}



static void append_tool_heavy_schema(buf *b, int idx) {
    if (idx) buf_putc(b, ',');
    buf_puts(b, "{\"type\":\"function\",\"function\":{\"name\":");
    char name[64];
    snprintf(name, sizeof(name), "opencode_tool_%02d", idx);
    json_escape(b, name);
    buf_puts(b, ",\"description\":");
    json_escape(b, "Tool schema with many properties and escaped text.");
    buf_puts(b, ",\"parameters\":{\"type\":\"object\",\"properties\":{");
    for (int j = 0; j < 12; j++) {
        if (j) buf_putc(b, ',');
        char prop[64];
        snprintf(prop, sizeof(prop), "arg_%02d_%02d", idx, j);
        json_escape(b, prop);
        buf_puts(b, ":{\"type\":\"string\",\"description\":");
        json_escape(b, "argument description with \\\\ escapes, quotes, and unicode \\ud83d\\ude80");
        buf_putc(b, '}');
    }
    buf_puts(b, "},\"required\":[");
    for (int j = 0; j < 4; j++) {
        if (j) buf_putc(b, ',');
        char prop[64];
        snprintf(prop, sizeof(prop), "arg_%02d_%02d", idx, j);
        json_escape(b, prop);
    }
    buf_puts(b, "]}}}");
}



static void append_tool_heavy_messages(buf *b) {
    buf_putc(b, '[');
    buf_puts(b, "{\"role\":\"system\",\"content\":");
    json_escape(b, "You are running OpenCode with many local tools.");
    buf_puts(b, "},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":");
    json_escape(b, "Please inspect the repository, edit files, run tests, and report briefly.");
    buf_puts(b, "}]}");

    for (int turn = 0; turn < 24; turn++) {
        buf_puts(b, ",{\"role\":\"assistant\",\"reasoning_content\":");
        json_escape(b, "I need to inspect files, use tools, and keep track of changes.");
        buf_puts(b, ",\"content\":");
        json_escape(b, "I will use the available tools.");
        buf_puts(b, ",\"tool_calls\":[");
        for (int call = 0; call < 3; call++) {
            if (call) buf_putc(b, ',');
            char id[64], name[64];
            snprintf(id, sizeof(id), "call_%02d_%02d", turn, call);
            snprintf(name, sizeof(name), "opencode_tool_%02d", call);

            buf args = {0};
            buf_puts(&args, "{\"path\":\"/tmp/opencode/project/file.c\",");
            buf_printf(&args, "\"range\":\"%d:%d\",", 10 + turn, 14 + turn);
            buf_puts(&args, "\"old\":\"line one\\\\nline two with quotes \\\" and backslash \\\\\\\\ plus rocket ");
            buf_puts(&args, "\\ud83d\\ude80\",");
            buf_puts(&args, "\"new\":\"replacement text\\\\nwith several lines\\\\nand symbols <>&\"}");

            buf_puts(b, "{\"id\":");
            json_escape(b, id);
            buf_puts(b, ",\"type\":\"function\",\"function\":{\"name\":");
            json_escape(b, name);
            buf_puts(b, ",\"arguments\":");
            json_escape(b, args.ptr ? args.ptr : "");
            buf_puts(b, "}}");
            buf_free(&args);
        }
        buf_puts(b, "]}");

        for (int call = 0; call < 3; call++) {
            char id[64];
            snprintf(id, sizeof(id), "call_%02d_%02d", turn, call);
            buf_puts(b, ",{\"role\":\"tool\",\"tool_call_id\":");
            json_escape(b, id);
            buf_puts(b, ",\"content\":[{\"type\":\"text\",\"text\":");
            json_escape(b, "tool output first line\nsecond line with escaped JSON-looking text {\"ok\":true}");
            buf_puts(b, "}]}");
        }
    }
    buf_putc(b, ']');
}



static void test_json_parser_handles_tool_heavy_requests(void) {
    buf tools = {0};
    buf_putc(&tools, '[');
    for (int i = 0; i < 32; i++) append_tool_heavy_schema(&tools, i);
    buf_putc(&tools, ']');

    buf messages = {0};
    append_tool_heavy_messages(&messages);

    for (int i = 0; i < 32; i++) {
        const char *tp = tools.ptr;
        char *schemas = NULL;
        tool_schema_orders orders = {0};
        TEST_ASSERT(parse_tools_value(&tp, &schemas, &orders));
        json_ws(&tp);
        TEST_ASSERT(*tp == '\0');
        TEST_ASSERT(schemas && strstr(schemas, "\"name\":\"opencode_tool_00\""));
        TEST_ASSERT(tool_schema_orders_find(&orders, "opencode_tool_00") != NULL);
        free(schemas);
        tool_schema_orders_free(&orders);

        const char *mp = messages.ptr;
        chat_msgs msgs = {0};
        TEST_ASSERT(parse_messages(&mp, &msgs));
        json_ws(&mp);
        TEST_ASSERT(*mp == '\0');
        TEST_ASSERT(msgs.len == 98);
        TEST_ASSERT(msgs.v[2].calls.len == 3);
        TEST_ASSERT(msgs.v[2].calls.v[0].arguments != NULL);
        TEST_ASSERT(strstr(msgs.v[2].calls.v[0].arguments, "replacement text") != NULL);
        chat_msgs_free(&msgs);
    }

    buf_free(&messages);
    buf_free(&tools);
}



static void test_json_string_handles_surrogates(void) {
    const char *p = "\"paired \\ud83d\\ude80 lone \\ud83d text badlow \\ud83d\\u0041\"";
    char *s = NULL;
    TEST_ASSERT(json_string(&p, &s));
    TEST_ASSERT(s != NULL);
    TEST_ASSERT(strstr(s, "paired \xf0\x9f\x9a\x80") != NULL);
    TEST_ASSERT(strstr(s, "lone \xef\xbf\xbd text") != NULL);
    TEST_ASSERT(strstr(s, "badlow \xef\xbf\xbd" "A") != NULL);
    TEST_ASSERT(*p == '\0');
    free(s);
}



static void test_model_metadata_clamps_completion_to_context(void) {
    buf b = {0};
    append_model_json_values(&b, "deepseek-v4-flash", "DeepSeek V4 Flash",
                             32768, 393216);
    TEST_ASSERT(strstr(b.ptr, "\"id\":\"deepseek-v4-flash\"") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"name\":\"DeepSeek V4 Flash\"") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"context_length\":32768") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"max_completion_tokens\":32768") != NULL);
    buf_free(&b);

    append_model_json_values(&b, "deepseek-v4-pro", "DeepSeek V4 Pro",
                             100000, 4096);
    TEST_ASSERT(strstr(b.ptr, "\"id\":\"deepseek-v4-pro\"") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"name\":\"DeepSeek V4 Pro\"") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"context_length\":100000") != NULL);
    TEST_ASSERT(strstr(b.ptr, "\"max_completion_tokens\":4096") != NULL);
    buf_free(&b);
}



static void test_client_socket_nonblocking_flag(void) {
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    if (sv[0] < 0 || sv[1] < 0) return;
    set_client_socket_nonblocking(sv[0]);
    int flags = fcntl(sv[0], F_GETFL, 0);
    TEST_ASSERT(flags >= 0);
    TEST_ASSERT((flags & O_NONBLOCK) != 0);
    close(sv[0]);
    close(sv[1]);
}



static void test_thinking_state_tracks_prompt_and_generated_tags(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.prompt_text = xstrdup("<｜Assistant｜><think>");
    thinking_state st = thinking_state_from_prompt(&r);
    TEST_ASSERT(st.inside == true);
    st.feed("reasoning body", strlen("reasoning body"));
    TEST_ASSERT(st.inside == true);
    st.feed("</thi", strlen("</thi"));
    TEST_ASSERT(st.inside == true);
    st.feed("nk>answer", strlen("nk>answer"));
    TEST_ASSERT(st.inside == false);
    st.feed("<thi", strlen("<thi"));
    TEST_ASSERT(st.inside == false);
    st.feed("nk>more", strlen("nk>more"));
    TEST_ASSERT(st.inside == true);
    request_free(&r);

    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_NONE;
    r.prompt_text = xstrdup("<｜Assistant｜></think>");
    st = thinking_state_from_prompt(&r);
    TEST_ASSERT(st.inside == false);
    request_free(&r);
}



static void test_thinking_checkpoint_remember_gate(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    thinking_state st = {.inside = true};

    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "length"));
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "stop"));

    st.inside = false;
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "length"));
    TEST_ASSERT(should_remember_thinking_checkpoint(&r, &st, "stop"));

    r.prompt_preserves_reasoning = true;
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "stop"));
    r.prompt_preserves_reasoning = false;
    r.has_tools = true;
    /* has_tools is NOT a disqualifier since the openwebui/opencode replay
     * fixes: a client can advertise tools and still strip reasoning on
     * replay, so prompt_preserves_reasoning is the sole gate (see
     * should_remember_thinking_checkpoint). The old expectation here was
     * stale from before that change. */
    TEST_ASSERT(should_remember_thinking_checkpoint(&r, &st, "stop"));
    r.has_tools = false;
    r.think_mode = PULSAR_THINK_NONE;
    TEST_ASSERT(!should_remember_thinking_checkpoint(&r, &st, "stop"));

    request_free(&r);
}



static void test_tool_marker_state_ignores_orphan_end(void) {
    bool saw_start = false;
    bool saw_end = false;
    bool orphan_end = false;

    observe_tool_markers("reasoning\n" PULSAR_PARAM_END "\n" PULSAR_INVOKE_END "\n" PULSAR_TOOL_CALLS_END,
                         &saw_start, &saw_end, &orphan_end);
    TEST_ASSERT(!saw_start);
    TEST_ASSERT(!saw_end);
    TEST_ASSERT(orphan_end);

    orphan_end = false;
    observe_tool_markers(PULSAR_TOOL_CALLS_START "\n" PULSAR_INVOKE_START " name=\"bash\">",
                         &saw_start, &saw_end, &orphan_end);
    TEST_ASSERT(saw_start);
    TEST_ASSERT(!saw_end);
    TEST_ASSERT(!orphan_end);

    observe_tool_markers(PULSAR_INVOKE_END "\n" PULSAR_TOOL_CALLS_END,
                         &saw_start, &saw_end, &orphan_end);
    TEST_ASSERT(saw_start);
    TEST_ASSERT(saw_end);
}



static void test_canonical_rewrite_rebuilds_when_live_tail_changes(void) {
    /* Regression for the first canonical-KV rewrite attempt: replacing a small
     * live suffix looks tempting because the raw SWA ring may still contain the
     * needed rows, but compressed KV counters and compressor/indexer frontiers
     * are already past the shared prefix.  Until those graph frontiers can be
     * restored exactly, every rewrite behind the live end must rebuild or load a
     * disk checkpoint. */
    TEST_ASSERT(pulsar_session_rewrite_requires_rebuild(19296, 19290, 19081));
    TEST_ASSERT(pulsar_session_rewrite_requires_rebuild(1024, 1030, 1000));
    TEST_ASSERT(pulsar_session_rewrite_requires_rebuild(1024, 900, 900));

    TEST_ASSERT(!pulsar_session_rewrite_requires_rebuild(1024, 1024, 1024));
    TEST_ASSERT(!pulsar_session_rewrite_requires_rebuild(1024, 1100, 1024));
}



static void test_kv_cache_store_len_uses_configured_boundary(void) {
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();
    TEST_ASSERT(kv_cache_store_len(&kc, 11011) == 10240);
    TEST_ASSERT(kv_cache_store_len(&kc, 1695) == 1695);

    kc.opt.boundary_trim_tokens = 0;
    kc.opt.boundary_align_tokens = 1000;
    TEST_ASSERT(kv_cache_store_len(&kc, 3500) == 3000);

    kc.opt.boundary_align_tokens = 0;
    TEST_ASSERT(kv_cache_store_len(&kc, 3500) == 3500);
}



static void test_kv_cache_chat_anchor_uses_last_user_before_assistant(void) {
    const int user = 9001;
    const int assistant = 9002;
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();
    kc.opt.min_tokens = 4;

    pulsar_tokens codex = {0};
    pulsar_tokens_push(&codex, 1);     /* BOS / system */
    pulsar_tokens_push(&codex, 2);
    pulsar_tokens_push(&codex, user);  /* environment_context item */
    pulsar_tokens_push(&codex, 3);
    pulsar_tokens_push(&codex, 4);
    pulsar_tokens_push(&codex, user);  /* actual task starts here */
    pulsar_tokens_push(&codex, 5);
    pulsar_tokens_push(&codex, assistant);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &codex, user, assistant) == 5);

    pulsar_tokens claude = {0};
    pulsar_tokens_push(&claude, 1);
    pulsar_tokens_push(&claude, 2);
    pulsar_tokens_push(&claude, 3);
    pulsar_tokens_push(&claude, 4);
    pulsar_tokens_push(&claude, user); /* system reminder and task share a turn */
    pulsar_tokens_push(&claude, 5);
    pulsar_tokens_push(&claude, assistant);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &claude, user, assistant) == 4);

    pulsar_tokens_free(&codex);
    pulsar_tokens_free(&claude);
}



static void test_kv_cache_chat_anchor_ignores_multiturn_tail(void) {
    const int user = 9001;
    const int assistant = 9002;
    kv_disk_cache kc = {0};
    kc.opt = kv_cache_default_options();
    kc.opt.min_tokens = 2;

    pulsar_tokens prompt = {0};
    pulsar_tokens_push(&prompt, 1);
    pulsar_tokens_push(&prompt, 2);
    pulsar_tokens_push(&prompt, user);      /* first task */
    pulsar_tokens_push(&prompt, 3);
    pulsar_tokens_push(&prompt, assistant); /* stop scanning here */
    pulsar_tokens_push(&prompt, 4);
    pulsar_tokens_push(&prompt, user);      /* later turn: not a cold anchor */
    pulsar_tokens_push(&prompt, 5);
    pulsar_tokens_push(&prompt, assistant);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, user, assistant) == 2);

    kc.opt.min_tokens = 3;
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, user, assistant) == -1);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, -1, assistant) == -1);
    TEST_ASSERT(kv_cache_chat_anchor_pos(&kc, &prompt, user, -1) == -1);

    pulsar_tokens_free(&prompt);
}



static void test_kv_cache_continued_uses_aligned_frontiers(void) {
    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.opt = kv_cache_default_options();

    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10239) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);

    kc.continued_last_store_tokens = 4096;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);

    kc.continued_last_store_tokens = 24576;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 30720) == 30720);

    kc.continued_last_store_tokens = 10240;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 18432) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 20480) == 20480);

    kc.opt.boundary_align_tokens = 0;
    kc.continued_last_store_tokens = 20480;
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 29999) == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 30000) == 30000);
}



static void test_kv_cache_cold_store_suppresses_duplicate_continued_boundary(void) {
    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.opt = kv_cache_default_options();

    int old = kv_cache_suppress_continued_store(&kc, 10240);
    TEST_ASSERT(old == 0);
    TEST_ASSERT(kc.continued_last_store_tokens == 10240);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 0);

    kv_cache_restore_suppressed_continued(&kc, old, 10240);
    TEST_ASSERT(kc.continued_last_store_tokens == 0);
    TEST_ASSERT(kv_cache_continued_store_target(&kc, 10240) == 10240);
}



static void test_kv_cache_file_size_must_fit_budget(void) {
    kv_disk_cache kc = {0};
    kc.budget_bytes = 1100;

    TEST_ASSERT(kv_cache_file_size_fits(&kc, 100, 930, 0, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, 100, 938, 0, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, 100, 900, 40, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, UINT64_MAX, 1, 0, NULL, NULL));

    kc.budget_bytes = 0;
    TEST_ASSERT(kv_cache_file_size_fits(&kc, 100, 900, 40, NULL, NULL));
    TEST_ASSERT(!kv_cache_file_size_fits(&kc, UINT64_MAX, 1, 0, NULL, NULL));
}



static void test_sha1_bytes_hex_matches_known_vector(void) {
    char sha[41];
    sha1_bytes_hex("abc", 3, sha);
    TEST_ASSERT(!strcmp(sha, "a9993e364706816aba3e25717850c26c9cd0d89d"));
}



static void test_kv_stub_file(const char *dir, const char *sha,
                              uint8_t reason, uint32_t tokens, uint32_t hits,
                              uint64_t last_used, uint64_t payload_bytes) {
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) {
        free(path);
        return;
    }

    uint8_t h[KV_CACHE_FIXED_HEADER];
    kv_fill_header(h, 2, reason, 0, tokens, hits, 32768, 100, last_used, payload_bytes);
    uint8_t text_len[4] = {0};
    TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
    for (uint64_t i = 0; i < payload_bytes; i++) {
        TEST_ASSERT(fputc(0, fp) != EOF);
    }
    TEST_ASSERT(fclose(fp) == 0);
    free(path);
}



static void test_kv_text_stub_file_model(const char *dir, const char *text,
                                         uint8_t model_id, uint8_t reason,
                                         uint32_t tokens,
                                         uint64_t payload_bytes) {
    char sha[41];
    sha1_bytes_hex(text, strlen(text), sha);
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) {
        free(path);
        return;
    }

    uint8_t h[KV_CACHE_FIXED_HEADER];
    pulsar_kvstore_fill_header(h, model_id, 2, reason, 0, tokens, 0,
                            32768, 100, 100, payload_bytes);
    uint8_t text_len[4];
    le_put32(text_len, (uint32_t)strlen(text));
    TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
    TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
    for (uint64_t i = 0; i < payload_bytes; i++) {
        TEST_ASSERT(fputc(0, fp) != EOF);
    }
    TEST_ASSERT(fclose(fp) == 0);
    free(path);
}



static void test_kv_text_stub_file(const char *dir, const char *text,
                                   uint8_t reason,
                                   uint32_t tokens, uint64_t payload_bytes) {
    test_kv_text_stub_file_model(dir, text, 0, reason, tokens, payload_bytes);
}



static void test_kv_cache_lookup_uses_longest_text_prefix(void) {
    char tmpl[] = "/tmp/ds4-kv-text-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *short_text = "transcript prefix";
    const char *long_text = "transcript prefix with sampled token bytes";
    test_kv_text_stub_file(dir, short_text, KV_REASON_COLD, 512, 0);
    test_kv_text_stub_file(dir, long_text, KV_REASON_COLD, 768, 0);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();

    int idx = kv_cache_find_text_prefix(&kc,
        "transcript prefix with sampled token bytes and suffix",
        2, 32768);
    TEST_ASSERT(idx >= 0);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].tokens == 768);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].text_bytes == strlen(long_text));
    TEST_ASSERT(kv_cache_find_text_prefix(&kc, "transcript prefiX", 2, 32768) < 0);

    kv_cache_close(&kc);
    char short_sha[41], long_sha[41];
    sha1_bytes_hex(short_text, strlen(short_text), short_sha);
    sha1_bytes_hex(long_text, strlen(long_text), long_sha);
    char short_name[44], long_name[44];
    snprintf(short_name, sizeof(short_name), "%.40s.kv", short_sha);
    snprintf(long_name, sizeof(long_name), "%.40s.kv", long_sha);
    char *short_path = path_join(dir, short_name);
    char *long_path = path_join(dir, long_name);
    unlink(short_path);
    unlink(long_path);
    free(short_path);
    free(long_path);
    rmdir(dir);
}



static void test_kv_cache_lookup_rejects_wrong_model(void) {
    char tmpl[] = "/tmp/ds4-kv-model-id-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *text = "shared rendered prefix";
    test_kv_text_stub_file_model(dir, text, 1, KV_REASON_COLD, 512, 0);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();

    TEST_ASSERT(pulsar_kvstore_find_text_prefix(&kc, "shared rendered prefix and tail",
                                             0, 2, 32768) < 0);
    int idx = pulsar_kvstore_find_text_prefix(&kc, "shared rendered prefix and tail",
                                           1, 2, 32768);
    TEST_ASSERT(idx >= 0);
    TEST_ASSERT(idx >= 0 && kc.entry[idx].model_id == 1);

    kv_cache_close(&kc);
    char sha[41];
    sha1_bytes_hex(text, strlen(text), sha);
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    unlink(path);
    free(path);
    rmdir(dir);
}



static void test_kv_cache_lookup_rejects_stale_payload_abi(void) {
    char tmpl[] = "/tmp/ds4-kv-stale-abi-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *text = "stale rendered prefix";
    char sha[41];
    sha1_bytes_hex(text, strlen(text), sha);
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);

    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (fp) {
        uint8_t h[KV_CACHE_FIXED_HEADER];
        kv_fill_header(h, 2, KV_REASON_COLD, 0, 512, 0, 32768, 100, 100, 0);
        h[20] = 0; /* pre-ABI-guard files used this byte as reserved zero. */
        uint8_t text_len[4];
        le_put32(text_len, (uint32_t)strlen(text));
        TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
        TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
        TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
        TEST_ASSERT(fclose(fp) == 0);
    }

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();

    TEST_ASSERT(pulsar_kvstore_find_text_prefix(&kc, "stale rendered prefix and tail",
                                             0, 2, 32768) < 0);

    kv_cache_close(&kc);
    unlink(path);
    free(path);
    rmdir(dir);
}



static void test_kv_tool_map_filters_by_dsml_text(void) {
    const char *dsml_keep =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    const char *dsml_drop =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">zzzz</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    server src = {0}, dst = {0};
    pthread_mutex_init(&src.tool_mu, NULL);
    pthread_mutex_init(&dst.tool_mu, NULL);
    src.tool_memory_put("call_keep", dsml_keep);
    src.tool_memory_put("call_drop", dsml_drop);

    FILE *fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    uint64_t estimated_bytes = 0;
    TEST_ASSERT(src.kv_tool_map_serialized_size(dsml_keep, &estimated_bytes));
    uint64_t bytes = 0;
    TEST_ASSERT(src.kv_tool_map_write(fp, dsml_keep, &bytes));
    TEST_ASSERT(bytes > 0);
    TEST_ASSERT(estimated_bytes == bytes);
    rewind(fp);
    TEST_ASSERT(dst.kv_tool_map_load_from_pos(fp, NULL) == 1);

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call keep = {.id = xstrdup("call_keep"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&a.calls, keep);
    chat_msgs_push(&msgs, a);
    chat_msg b = {0};
    b.role = xstrdup("assistant");
    tool_call drop = {.id = xstrdup("call_drop"), .name = xstrdup("bash"), .arguments = xstrdup("{}")};
    tool_calls_push(&b.calls, drop);
    chat_msgs_push(&msgs, b);
    tool_replay_stats stats = {0};
    dst.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(msgs.v[1].calls.raw_dsml == NULL);
    TEST_ASSERT(stats.disk == 1);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);
    TEST_ASSERT(strstr(msgs.v[0].calls.raw_dsml, "pwd") != NULL);
    TEST_ASSERT(strstr(msgs.v[0].calls.raw_dsml, "zzzz") == NULL);

    chat_msgs_free(&msgs);
    if (fp) fclose(fp);
    tool_memory_free(&src.tool_mem);
    tool_memory_free(&dst.tool_mem);
    pthread_mutex_destroy(&src.tool_mu);
    pthread_mutex_destroy(&dst.tool_mu);
}



static void test_kv_tool_map_restores_before_prompt_render(void) {
    char tmpl[] = "/tmp/ds4-kv-tool-map-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *sha = "3333333333333333333333333333333333333333";
    char name[44];
    snprintf(name, sizeof(name), "%.40s.kv", sha);
    char *path = path_join(dir, name);
    const char *dsml =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">echo exact</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    const char *text = dsml;

    server src = {0};
    pthread_mutex_init(&src.tool_mu, NULL);
    src.tool_memory_put("call_disk", dsml);

    FILE *fp = fopen(path, "wb");
    TEST_ASSERT(fp != NULL);
    if (fp) {
        uint8_t h[KV_CACHE_FIXED_HEADER];
        kv_fill_header(h, 2, KV_REASON_CONTINUED, KV_EXT_TOOL_MAP, 512, 0, 32768, 100, 100, 0);
        uint8_t text_len[4];
        le_put32(text_len, (uint32_t)strlen(text));
        TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
        TEST_ASSERT(fwrite(text_len, 1, sizeof(text_len), fp) == sizeof(text_len));
        TEST_ASSERT(fwrite(text, 1, strlen(text), fp) == strlen(text));
        uint64_t ignored = 0;
        TEST_ASSERT(src.kv_tool_map_write(fp, dsml, &ignored));
        TEST_ASSERT(fclose(fp) == 0);
    }

    server dst = {0};
    pthread_mutex_init(&dst.tool_mu, NULL);
    dst.kv.enabled = true;
    dst.kv.dir = xstrdup(dir);
    dst.kv.opt = kv_cache_default_options();

    chat_msgs msgs = {0};
    chat_msg a = {0};
    a.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_disk");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"echo canonical\"}");
    tool_calls_push(&a.calls, tc);
    chat_msgs_push(&msgs, a);

    dst.kv_cache_restore_tool_memory_for_messages(&msgs);
    tool_replay_stats stats = {0};
    dst.tool_memory_attach_to_messages(&msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_dsml != NULL);
    TEST_ASSERT(stats.disk == 1);
    TEST_ASSERT(stats.canonical == 0);
    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(strstr(prompt, "echo exact") != NULL);
    TEST_ASSERT(strstr(prompt, "echo canonical") == NULL);

    free(prompt);
    chat_msgs_free(&msgs);
    kv_cache_close(&dst.kv);
    tool_memory_free(&src.tool_mem);
    tool_memory_free(&dst.tool_mem);
    pthread_mutex_destroy(&src.tool_mu);
    pthread_mutex_destroy(&dst.tool_mu);
    unlink(path);
    free(path);
    rmdir(dir);
}



static void test_kv_cache_eviction_values_fresh_snapshots(void) {
    char tmpl[] = "/tmp/ds4-kv-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_UNKNOWN, 512, 0, now, 4096);
    test_kv_stub_file(dir, new_sha, KV_REASON_UNKNOWN, 2048, 0, now, 2048);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);
    TEST_ASSERT(access(new_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_prefers_anchor_reason(void) {
    char tmpl[] = "/tmp/ds4-kv-anchor-reason-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *anchor_sha = "1111111111111111111111111111111111111111";
    const char *continued_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, anchor_sha, KV_REASON_COLD, 2048, 0, now, 2048);
    test_kv_stub_file(dir, continued_sha, KV_REASON_CONTINUED, 2048, 0, now, 2048);

    char anchor_name[44], continued_name[44];
    snprintf(anchor_name, sizeof(anchor_name), "%.40s.kv", anchor_sha);
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    char *anchor_path = path_join(dir, anchor_name);
    char *continued_path = path_join(dir, continued_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(anchor_path, F_OK) == 0);
    TEST_ASSERT(access(continued_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(anchor_path);
    unlink(continued_path);
    free(anchor_path);
    free(continued_path);
    rmdir(dir);
}



/* A sys-prefix checkpoint (truncated shared preamble) must outlive ordinary
 * cold anchors under budget pressure: it is the one file every NEW
 * conversation with the same system prompt can text-prefix restore from, and
 * it sits at hits=0 until the first restart needs it. */
static void test_kv_cache_eviction_prefers_sys_prefix_over_cold(void) {
    char tmpl[] = "/tmp/ds4-kv-sys-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *sys_sha  = "3333333333333333333333333333333333333333";
    const char *cold_sha = "4444444444444444444444444444444444444444";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, sys_sha, KV_REASON_SYS_PREFIX, 2048, 0, now, 2048);
    test_kv_stub_file(dir, cold_sha, KV_REASON_COLD, 2048, 0, now, 2048);

    char sys_name[44], cold_name[44];
    snprintf(sys_name, sizeof(sys_name), "%.40s.kv", sys_sha);
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    char *sys_path = path_join(dir, sys_name);
    char *cold_path = path_join(dir, cold_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(sys_path, F_OK) == 0);
    TEST_ASSERT(access(cold_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(sys_path);
    unlink(cold_path);
    free(sys_path);
    free(cold_path);
    rmdir(dir);

    TEST_ASSERT(pulsar_kvstore_reason_code("sys-prefix") ==
                PULSAR_KVSTORE_REASON_SYS_PREFIX);
}



static void test_kv_cache_eviction_makes_room_before_store(void) {
    char tmpl[] = "/tmp/ds4-kv-pre-store-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 4096, 0, now, 2048);

    char old_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    char *old_path = path_join(dir, old_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 4096u) + 16u;
    kv_cache_evict(&kc, NULL, KV_CACHE_FIXED_HEADER + 4u + 4096u, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(old_path);
    free(old_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_ignores_oversize_incoming(void) {
    char tmpl[] = "/tmp/ds4-kv-oversize-store-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 4096, 0, now, 1024);

    char old_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    char *old_path = path_join(dir, old_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 1024u) + 16u;
    kv_cache_evict(&kc, NULL, kc.budget_bytes + 1, NULL);

    TEST_ASSERT(access(old_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    free(old_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_prefers_superseded_continued_prefix(void) {
    char tmpl[] = "/tmp/ds4-kv-prefix-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *continued_text = "system: hello world";
    const char *cold_text = "different stable prefix";
    const char *incoming_text = "system: hello world\nuser: prompt";
    test_kv_text_stub_file(dir, continued_text, KV_REASON_CONTINUED, 4096, 2048);
    test_kv_text_stub_file(dir, cold_text, KV_REASON_COLD, 1024, 2048);

    char continued_sha[41], cold_sha[41];
    sha1_bytes_hex(continued_text, strlen(continued_text), continued_sha);
    sha1_bytes_hex(cold_text, strlen(cold_text), cold_sha);
    char continued_name[44], cold_name[44];
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    char *continued_path = path_join(dir, continued_name);
    char *cold_path = path_join(dir, cold_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    uint64_t incoming_bytes =
        KV_CACHE_FIXED_HEADER + 4u + strlen(incoming_text) + 2048u;
    kc.budget_bytes =
        incoming_bytes + KV_CACHE_FIXED_HEADER + 4u + strlen(cold_text) + 2048u;
    pulsar_kvstore_eviction_context incoming = {
        .text = incoming_text,
        .text_len = strlen(incoming_text),
        .model_id = 0,
        .quant_bits = 2,
        .ctx_size = 32768,
        .reject_different_quant = false,
    };
    kv_cache_evict(&kc, NULL, incoming_bytes, &incoming);

    TEST_ASSERT(access(continued_path, F_OK) != 0);
    TEST_ASSERT(access(cold_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(continued_path);
    unlink(cold_path);
    free(continued_path);
    free(cold_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_keeps_smaller_context_prefix(void) {
    char tmpl[] = "/tmp/ds4-kv-prefix-ctx-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *continued_text = "system: hello world";
    const char *cold_text = "different stable prefix";
    const char *incoming_text = "system: hello world\nuser: prompt";
    test_kv_text_stub_file(dir, continued_text, KV_REASON_CONTINUED, 4096, 2048);
    test_kv_text_stub_file(dir, cold_text, KV_REASON_COLD, 1024, 2048);

    char continued_sha[41], cold_sha[41];
    sha1_bytes_hex(continued_text, strlen(continued_text), continued_sha);
    sha1_bytes_hex(cold_text, strlen(cold_text), cold_sha);
    char continued_name[44], cold_name[44];
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    char *continued_path = path_join(dir, continued_name);
    char *cold_path = path_join(dir, cold_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    uint64_t incoming_bytes =
        KV_CACHE_FIXED_HEADER + 4u + strlen(incoming_text) + 2048u;
    kc.budget_bytes =
        incoming_bytes + KV_CACHE_FIXED_HEADER + 4u + strlen(continued_text) + 2048u;
    pulsar_kvstore_eviction_context incoming = {
        .text = incoming_text,
        .text_len = strlen(incoming_text),
        .model_id = 0,
        .quant_bits = 2,
        .ctx_size = 65536,
        .reject_different_quant = false,
    };
    kv_cache_evict(&kc, NULL, incoming_bytes, &incoming);

    TEST_ASSERT(access(continued_path, F_OK) == 0);
    TEST_ASSERT(access(cold_path, F_OK) != 0);

    kv_cache_close(&kc);
    unlink(continued_path);
    unlink(cold_path);
    free(continued_path);
    free(cold_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_score_decays_stale_hits(void) {
    /* stale: lower tokens-per-byte (e.g. tool-heavy prompt) but boosted by
     * 10 hits well in the past.  fresh: higher tokens-per-byte and zero hits,
     * just stored.  The stale hit bonus decays by inactivity, so fresh wins on
     * its better baseline even though stale once had more successful hits. */
    const uint64_t now = 1000u + 14u * KV_CACHE_HIT_HALF_LIFE_SECONDS;
    kv_entry stale = {.tokens = 1024, .hits = 10, .last_used = 1000, .file_size = 4096};
    kv_entry fresh = {.tokens = 2048, .hits = 0,  .last_used = now, .file_size = 4096};

    double s_on = kv_entry_eviction_score(&stale, NULL, now, NULL);
    double f_on = kv_entry_eviction_score(&fresh, NULL, now, NULL);
    TEST_ASSERT(s_on < f_on);

    /* A fresh entry's score never decays below its (0+1) * tokens/size floor,
     * regardless of how old another entry's hit history is. */
    TEST_ASSERT(f_on == 1.0 * (double)fresh.tokens / (double)fresh.file_size);
}



static void test_kv_cache_eviction_decayed_hits_tie_break_by_age(void) {
    char tmpl[] = "/tmp/ds4-kv-stale-hit-evict-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *old_sha = "1111111111111111111111111111111111111111";
    const char *new_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    uint64_t stale = now > KV_CACHE_HIT_HALF_LIFE_SECONDS * 14ull
        ? now - KV_CACHE_HIT_HALF_LIFE_SECONDS * 14ull
        : 1;
    test_kv_stub_file(dir, old_sha, KV_REASON_COLD, 2048, 15, stale, 2048);
    test_kv_stub_file(dir, new_sha, KV_REASON_COLD, 2048, 0, now, 2048);

    char old_name[44], new_name[44];
    snprintf(old_name, sizeof(old_name), "%.40s.kv", old_sha);
    snprintf(new_name, sizeof(new_name), "%.40s.kv", new_sha);
    char *old_path = path_join(dir, old_name);
    char *new_path = path_join(dir, new_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(old_path, F_OK) != 0);
    TEST_ASSERT(access(new_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(old_path);
    unlink(new_path);
    free(old_path);
    free(new_path);
    rmdir(dir);
}



static void test_kv_cache_eviction_keeps_aligned_continued_frontiers(void) {
    char tmpl[] = "/tmp/ds4-kv-live-prefix-test.XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (!dir) return;

    const char *cold_sha = "1111111111111111111111111111111111111111";
    const char *continued_sha = "2222222222222222222222222222222222222222";
    uint64_t now = (uint64_t)time(NULL);
    test_kv_stub_file(dir, cold_sha, KV_REASON_COLD, 512, 0, now, 2048);
    test_kv_stub_file(dir, continued_sha, KV_REASON_CONTINUED, 2048, 0, now, 2048);

    char cold_name[44], continued_name[44];
    snprintf(cold_name, sizeof(cold_name), "%.40s.kv", cold_sha);
    snprintf(continued_name, sizeof(continued_name), "%.40s.kv", continued_sha);
    char *cold_path = path_join(dir, cold_name);
    char *continued_path = path_join(dir, continued_name);

    kv_disk_cache kc = {0};
    kc.enabled = true;
    kc.dir = xstrdup(dir);
    kc.opt = kv_cache_default_options();
    kc.budget_bytes = (KV_CACHE_FIXED_HEADER + 4u + 2048u) + 16u;
    kv_cache_evict(&kc, NULL, 0, NULL);

    TEST_ASSERT(access(cold_path, F_OK) != 0);
    TEST_ASSERT(access(continued_path, F_OK) == 0);

    kv_cache_close(&kc);
    unlink(cold_path);
    unlink(continued_path);
    free(cold_path);
    free(continued_path);
    rmdir(dir);
}



static void test_thinking_checkpoint_canonical_matches_future_prompt(void) {
    /* Simulate: user sends a single message, thinking mode on, no tools.
     * Model generates reasoning + content.  The next request will drop the
     * reasoning from this turn.  Verify that:
     *   prompt_text[:-len("<think>")] + "</think>" + content + "<|eos|>"
     * equals what render_chat_prompt_text produces for the history. */

    chat_msgs prefix_msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("What is 2+2?");
    chat_msgs_push(&prefix_msgs, user1);

    /* This is what prompt_text looks like for the first generation */
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, NULL, NULL, PULSAR_THINK_HIGH);
    /* prompt_text should end with <think> */
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(pt_len >= 7);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* The model generates: reasoning + </think> + content */
    const char *reasoning = "Let me think... 2+2 = 4";
    const char *content = "The answer is 4.";

    /* Build the canonical checkpoint text (what we'd produce after canonicalization) */
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);  /* strip <think> */
    buf_puts(&canonical, "</think>");
    buf_puts(&canonical, content);
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = PULSAR_THINK_HIGH;
    r.prompt_text = xstrdup(prompt_text);
    char *visible = build_toolless_thinking_visible_text(&r, content);
    TEST_ASSERT(visible != NULL);
    TEST_ASSERT(!strcmp(visible, canonical.ptr));
    free(visible);
    request_free(&r);

    /* Now build what the NEXT request would render: history includes this
     * assistant message, plus a new user message.  Extract just the prefix
     * up to and including the eos of the assistant turn. */
    chat_msgs history_msgs = {0};
    chat_msg h_user1 = {0};
    h_user1.role = xstrdup("user");
    h_user1.content = xstrdup("What is 2+2?");
    chat_msgs_push(&history_msgs, h_user1);
    chat_msg h_asst = {0};
    h_asst.role = xstrdup("assistant");
    h_asst.reasoning = xstrdup(reasoning);
    h_asst.content = xstrdup(content);
    chat_msgs_push(&history_msgs, h_asst);
    chat_msg h_user2 = {0};
    h_user2.role = xstrdup("user");
    h_user2.content = xstrdup("Thanks!");
    chat_msgs_push(&history_msgs, h_user2);

    char *future_prompt = render_chat_prompt_text(&history_msgs, NULL, NULL, PULSAR_THINK_HIGH);

    /* The future prompt should START with our canonical text */
    size_t clen = canonical.len;
    TEST_ASSERT(strlen(future_prompt) > clen);
    TEST_ASSERT(!memcmp(future_prompt, canonical.ptr, clen));

    /* And what comes after is the new user turn + assistant prefix */
    const char *rest = future_prompt + clen;
    TEST_ASSERT(strstr(rest, "Thanks!") != NULL);
    TEST_ASSERT(strstr(rest, "<think>") != NULL);  /* new turn starts thinking */

    /* Verify reasoning is NOT in the future prompt for this turn */
    const char *asst_turn = strstr(future_prompt, "<" "\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c" ">");
    TEST_ASSERT(asst_turn != NULL);
    TEST_ASSERT(strstr(future_prompt, reasoning) == NULL);  /* reasoning dropped */

    free(future_prompt);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&prefix_msgs);
    chat_msgs_free(&history_msgs);
}



static void test_thinking_canonical_empty_content(void) {
    /* Edge case: model thinks but produces empty content (e.g. tool-less
     * thinking where answer is entirely in reasoning).  Canonical should
     * still be valid: prompt_text[:-7] + "</think><|eos|>" */
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Think about life");
    chat_msgs_push(&msgs, user);

    char *prompt_text = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);

    /* Build canonical with empty content */
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);
    buf_puts(&canonical, "</think>");
    /* empty content */
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    /* Future prompt with empty content assistant message */
    chat_msgs history = {0};
    chat_msg h_u = {0};
    h_u.role = xstrdup("user");
    h_u.content = xstrdup("Think about life");
    chat_msgs_push(&history, h_u);
    chat_msg h_a = {0};
    h_a.role = xstrdup("assistant");
    h_a.reasoning = xstrdup("Deep thoughts about existence...");
    h_a.content = xstrdup("");
    chat_msgs_push(&history, h_a);
    chat_msg h_u2 = {0};
    h_u2.role = xstrdup("user");
    h_u2.content = xstrdup("Continue");
    chat_msgs_push(&history, h_u2);

    char *future = render_chat_prompt_text(&history, NULL, NULL, PULSAR_THINK_HIGH);
    TEST_ASSERT(strlen(future) > canonical.len);
    TEST_ASSERT(!memcmp(future, canonical.ptr, canonical.len));
    /* reasoning dropped */
    TEST_ASSERT(strstr(future, "Deep thoughts") == NULL);

    free(future);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&msgs);
    chat_msgs_free(&history);
}



static void test_thinking_canonical_multi_turn(void) {
    /* Multi-turn: 3 user messages, 2 assistant responses with reasoning.
     * Both prior assistant turns should have reasoning dropped.
     * The canonical after the SECOND generation should produce text that
     * matches the start of a 3rd-turn future prompt. */
    chat_msgs turn2_prefix = {0};
    chat_msg u1 = {0};
    u1.role = xstrdup("user");
    u1.content = xstrdup("Hello");
    chat_msgs_push(&turn2_prefix, u1);
    chat_msg a1 = {0};
    a1.role = xstrdup("assistant");
    a1.reasoning = xstrdup("first reasoning");
    a1.content = xstrdup("Hi there");
    chat_msgs_push(&turn2_prefix, a1);
    chat_msg u2 = {0};
    u2.role = xstrdup("user");
    u2.content = xstrdup("How are you?");
    chat_msgs_push(&turn2_prefix, u2);

    /* prompt_text for the 2nd generation (includes 1st assistant turn) */
    char *prompt_text = render_chat_prompt_text(&turn2_prefix, NULL, NULL, PULSAR_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* 1st turn reasoning is already dropped in this prompt_text */
    TEST_ASSERT(strstr(prompt_text, "first reasoning") == NULL);
    TEST_ASSERT(strstr(prompt_text, "Hi there") != NULL);

    /* After 2nd generation: canonical drops 2nd reasoning too */
    const char *content2 = "I'm doing well";
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);
    buf_puts(&canonical, "</think>");
    buf_puts(&canonical, content2);
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    /* Future: 3rd user message arrives */
    chat_msgs future_msgs = {0};
    chat_msg fu1 = {0}; fu1.role = xstrdup("user"); fu1.content = xstrdup("Hello");
    chat_msgs_push(&future_msgs, fu1);
    chat_msg fa1 = {0}; fa1.role = xstrdup("assistant");
    fa1.reasoning = xstrdup("first reasoning");
    fa1.content = xstrdup("Hi there");
    chat_msgs_push(&future_msgs, fa1);
    chat_msg fu2 = {0}; fu2.role = xstrdup("user"); fu2.content = xstrdup("How are you?");
    chat_msgs_push(&future_msgs, fu2);
    chat_msg fa2 = {0}; fa2.role = xstrdup("assistant");
    fa2.reasoning = xstrdup("second reasoning");
    fa2.content = xstrdup(content2);
    chat_msgs_push(&future_msgs, fa2);
    chat_msg fu3 = {0}; fu3.role = xstrdup("user"); fu3.content = xstrdup("Great");
    chat_msgs_push(&future_msgs, fu3);

    char *future = render_chat_prompt_text(&future_msgs, NULL, NULL, PULSAR_THINK_HIGH);
    /* Both reasonings dropped */
    TEST_ASSERT(strstr(future, "first reasoning") == NULL);
    TEST_ASSERT(strstr(future, "second reasoning") == NULL);
    /* Canonical is a prefix of future */
    TEST_ASSERT(strlen(future) > canonical.len);
    TEST_ASSERT(!memcmp(future, canonical.ptr, canonical.len));

    free(future);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&turn2_prefix);
    chat_msgs_free(&future_msgs);
}



static void test_thinking_canonical_with_tools_preserves_reasoning(void) {
    /* When tools ARE present, reasoning is preserved in re-render.
     * The toolless thinking live binding should NOT fire (has_tools gate),
     * and the tool-call replay path handles it.  Verify the template
     * preserves reasoning when tool_context is true. */
    const char *tool_schemas = "{\"name\":\"bash\"}";

    chat_msgs msgs = {0};
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup("run ls");
    chat_msgs_push(&msgs, u);

    char *prompt_text = render_chat_prompt_text(&msgs, tool_schemas, NULL, PULSAR_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* With tools, next render KEEPS reasoning */
    chat_msgs history = {0};
    chat_msg hu = {0}; hu.role = xstrdup("user"); hu.content = xstrdup("run ls");
    chat_msgs_push(&history, hu);
    chat_msg ha = {0}; ha.role = xstrdup("assistant");
    ha.reasoning = xstrdup("I should run bash");
    ha.content = xstrdup("Here you go");
    chat_msgs_push(&history, ha);
    chat_msg hu2 = {0}; hu2.role = xstrdup("user"); hu2.content = xstrdup("thanks");
    chat_msgs_push(&history, hu2);

    char *future = render_chat_prompt_text(&history, tool_schemas, NULL, PULSAR_THINK_HIGH);
    /* Reasoning IS preserved when tools present */
    TEST_ASSERT(strstr(future, "I should run bash") != NULL);
    TEST_ASSERT(strstr(future, "<think>I should run bash</think>") != NULL);

    free(future);
    free(prompt_text);
    chat_msgs_free(&msgs);
    chat_msgs_free(&history);
}



static void test_thinking_canonical_non_thinking_mode_noop(void) {
    /* When thinking is disabled (deepseek-chat), prompt_text ends with
     * </think> not <think>.  The toolless thinking live binding is a no-op
     * (early return on memcmp check). */
    chat_msgs msgs = {0};
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup("Hello");
    chat_msgs_push(&msgs, u);

    char *prompt_text = render_chat_prompt_text(&msgs, NULL, NULL, PULSAR_THINK_NONE);
    size_t pt_len = strlen(prompt_text);
    /* Should end with </think>, not <think> */
    TEST_ASSERT(pt_len >= 8);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 8, "</think>", 8));
    /* Does NOT end with <think> */
    TEST_ASSERT(memcmp(prompt_text + pt_len - 7, "<think>", 7) != 0);

    free(prompt_text);
    chat_msgs_free(&msgs);
}



static void test_unterminated_think_stays_off_content(void) {
    /* Generation that ends inside the think block (token cap / stop) must
     * surface as reasoning_content, never as visible content — clients that
     * score or display the answer channel would otherwise receive raw
     * chain-of-thought (the tool-eval-bench MMLU/IFEval artifact). Covers
     * both the generated "<think>" opener and the prompt-pre-opened form. */
    const char *cases[] = {
        "<think>We need to compute the index of the subgroup",
        "We need to compute the index of the subgroup",
    };
    for (size_t i = 0; i < 2; i++) {
        char *content = NULL, *reasoning = NULL;
        tool_calls calls = {0};
        TEST_ASSERT(parse_generated_message_ex(cases[i], true, &content,
                                               &reasoning, &calls));
        TEST_ASSERT(content && content[0] == '\0');
        TEST_ASSERT(reasoning &&
                    !strcmp(reasoning, "We need to compute the index of the subgroup"));
        TEST_ASSERT(calls.len == 0);
        free(content);
        free(reasoning);
        tool_calls_free(&calls);
    }
}



static void test_kv_admission_budget_math(void) {
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;

    /* Budget = usable - weights - overhead - free floor, clamped at 0 (the
     * floor term is the 2026-07-13 lockup fix: a fully committed budget must
     * still leave PULSAR_SERVER_MEM_FLOOR_BYTES of the machine free). */
    TEST_ASSERT(server_kv_budget_bytes(91ull * GiB) ==
                PULSAR_SERVER_USABLE_BYTES - 91ull * GiB -
                PULSAR_SERVER_PROCESS_OVERHEAD_BYTES - PULSAR_SERVER_MEM_FLOOR_BYTES);
    TEST_ASSERT(server_kv_budget_bytes(200ull * GiB) == 0);  /* weights > usable: no underflow */
    /* Reserves alone (no weights) must also clamp, not underflow. */
    TEST_ASSERT(server_kv_budget_bytes(PULSAR_SERVER_USABLE_BYTES) == 0);

    /* Admission: committed + incoming <= budget, with overflow-safe compare. */
    TEST_ASSERT(server_kv_admits(26ull * GiB, 0, 20ull * GiB));
    TEST_ASSERT(server_kv_admits(26ull * GiB, 20ull * GiB, 6ull * GiB));   /* exact fit */
    TEST_ASSERT(!server_kv_admits(26ull * GiB, 20ull * GiB, 7ull * GiB));  /* over by 1 GiB */
    TEST_ASSERT(!server_kv_admits(26ull * GiB, 0, 27ull * GiB));           /* lone over-budget */

    /* GB10 production shape (2026-07-15 re-measure: 18 GiB steady-state
     * process overhead, 4 GiB kernel-breathing-room floor):
     * usable 121 GiB − weights ~85.4 GiB − overhead 18 GiB − floor 4 GiB
     * ⇒ budget ~13.6 GiB. Slot 0 at ctx=65536/pc=4096 costs ~4.6 GiB
     * (measured) and must admit at startup; a doubled ~9.2 GiB session (the
     * ctx=131072 upper bound) must also admit alone; the THIRD 4.6 GiB slot
     * (13.8 GiB committed) must be refused — the 2026-07-13 incident shape
     * admitted three. */
    const uint64_t MiB = 1024ull * 1024ull;
    const uint64_t gb10_weights = 87450ull * MiB;              /* ~85.4 GiB */
    const uint64_t gb10_budget = server_kv_budget_bytes(gb10_weights);
    TEST_ASSERT(gb10_budget == PULSAR_SERVER_USABLE_BYTES - gb10_weights -
                PULSAR_SERVER_PROCESS_OVERHEAD_BYTES - PULSAR_SERVER_MEM_FLOOR_BYTES);
    TEST_ASSERT(gb10_budget > 13ull * GiB && gb10_budget < 14ull * GiB);   /* ~13.6 GiB */
    const uint64_t slot64k = 4710ull * MiB;                    /* ~4.6 GiB @ ctx 64k */
    TEST_ASSERT(server_kv_admits(gb10_budget, 0, slot64k));               /* slot 0, 64k */
    TEST_ASSERT(server_kv_admits(gb10_budget, 0, 2ull * slot64k));        /* slot 0, 128k bound */
    TEST_ASSERT(server_kv_admits(gb10_budget, slot64k, slot64k));         /* second slot */
    TEST_ASSERT(!server_kv_admits(gb10_budget, 2ull * slot64k, slot64k)); /* third refused */

    /* Packed estimate vs the sizeof(float) upper bound. The raw SWA ring packs
     * to f16 regardless of a loaded model (it depends only on the static shape),
     * so it is strictly smaller here. The compressed term depends on the model's
     * per-layer compress ratios (g_pulsar_compress_ratios, populated at engine open)
     * and is therefore 0 in this no-model unit context — its packing is exercised
     * live at server startup and reported by the KV-admission log. We assert the
     * model-independent invariants: internal consistency, unpacked scratch, and
     * monotonic (packed <= f32) bounds. */
    pulsar_context_memory f32 =
        pulsar_context_memory_estimate_with_prefill(PULSAR_BACKEND_CUDA, 32768, 0);
    pulsar_context_memory pk =
        pulsar_context_memory_estimate_packed(PULSAR_BACKEND_CUDA, 32768, 0);
    TEST_ASSERT(pk.total_bytes == pk.raw_bytes + pk.compressed_bytes + pk.scratch_bytes);
    TEST_ASSERT(pk.raw_bytes > 0 && pk.raw_bytes < f32.raw_bytes);  /* f16 raw ring */
    TEST_ASSERT(pk.scratch_bytes == f32.scratch_bytes);             /* scratch is not packed */
    TEST_ASSERT(pk.compressed_bytes <= f32.compressed_bytes);       /* packed <= f32 upper bound */
    TEST_ASSERT(pk.total_bytes < f32.total_bytes);
}



/* MemAvailable floor backstop (2026-07-15 re-measure). The floor is kernel
 * breathing room ONLY: process-fixed costs are the ledger's 18 GiB overhead
 * reserve, and the ~8.7 GiB lazy first-request CUDA allocations erode
 * MemAvailable INSIDE that already-subtracted reserve. The old 6 GiB floor
 * double-counted that caution and vetoed sessions the ledger legally
 * admitted. */
static void test_mem_floor_admits_warmed_box_shape(void) {
    const uint64_t MiB = 1024ull * 1024ull;
    const uint64_t GiB = 1024ull * MiB;

    /* The 2026-07-14 Tier-1 exit-gate incident: warmed box, two sessions
     * live, third 2.5 GiB session sees MemAvailable 8.39 GiB. The 6 GiB
     * floor demanded 8.50 and refused (a 0.11 GiB miss with ~5.9 GiB truly
     * free at full commit); the 4 GiB floor must admit — post-admission
     * MemAvailable stays >= ~5.89 GiB, well above the backstop. */
    const uint64_t est = 2560ull * MiB;                     /* 2.5 GiB session */
    TEST_ASSERT(server_mem_floor_admits(8590ull * MiB, est));  /* 8.39 GiB avail */

    /* Warmed-box full-commit steady state measured 2026-07-15: 5.96 GiB
     * avail with three live sessions. A further session would leave ~3.4,
     * below the backstop: refuse. */
    TEST_ASSERT(!server_mem_floor_admits(6103ull * MiB, est)); /* 5.96 GiB avail */

    /* A genuinely tight box must always refuse. */
    TEST_ASSERT(!server_mem_floor_admits(4ull * GiB, est));
    /* Exact boundary: est + floor. */
    TEST_ASSERT(server_mem_floor_admits(est + PULSAR_SERVER_MEM_FLOOR_BYTES, est));
    TEST_ASSERT(!server_mem_floor_admits(est + PULSAR_SERVER_MEM_FLOOR_BYTES - 1, est));
    /* Unreadable /proc/meminfo (avail == 0) fails closed. */
    TEST_ASSERT(!server_mem_floor_admits(0, est));
    /* Overflow guard: absurd estimate must refuse, not wrap. */
    TEST_ASSERT(!server_mem_floor_admits(UINT64_MAX, UINT64_MAX - 1ull * GiB));
}



/* Multi-session increment 4: eviction is the first runtime session-free path,
 * so the ledger must balance EXACTLY across provision→evict→provision cycles
 * — each provisioning commits the session's ACTUAL allocator bytes and each
 * eviction releases that same stored value. */
static void test_session_eviction_ledger_math(void) {
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;
    const uint64_t MiB = 1024ull * 1024ull;
    const uint64_t budget = server_kv_budget_bytes(87450ull * MiB); /* ~13.6 GiB */
    const uint64_t slot0 = 4710ull * MiB;   /* startup slot, ctx 64k measured */
    const uint64_t a = 2560ull * MiB;       /* lazy 64k slot (pc 2048 shape) */
    const uint64_t b = 2571ull * MiB;       /* same shape, distinct actual */

    /* provision slot0 + a + b, filling most of the budget */
    uint64_t committed = slot0;
    TEST_ASSERT(server_kv_admits(budget, committed, a));
    committed += a;
    TEST_ASSERT(server_kv_admits(budget, committed, b));
    committed += b;
    /* budget full for another slot0-sized session: admission refuses
     * (9841 + 4710 = 14551 MiB > ~13.6 GiB budget) */
    TEST_ASSERT(!server_kv_admits(budget, committed, slot0));

    /* evict a: the exact committed value comes back, and the freed budget
     * admits an equal-shape provisioning again */
    committed = server_ledger_release(committed, a);
    TEST_ASSERT(committed == slot0 + b);
    TEST_ASSERT(server_kv_admits(budget, committed, a));
    committed += a;
    TEST_ASSERT(committed == slot0 + b + a);

    /* evict everything back down to slot 0: balance is exact, not approximate */
    committed = server_ledger_release(committed, b);
    committed = server_ledger_release(committed, a);
    TEST_ASSERT(committed == slot0);
    committed = server_ledger_release(committed, slot0);
    TEST_ASSERT(committed == 0);

    /* releasing more than is committed means the pairing broke: clamp to 0
     * (warns loudly; the MemAvailable floor backstops the over-admission) */
    TEST_ASSERT(server_ledger_release(1ull * GiB, 2ull * GiB) == 0);
    TEST_ASSERT(server_ledger_release(0, 1) == 0);
}



/* Victim selection: LRU over IDLE provisioned slots, slot 0 pinned, active
 * and protected slots skipped, ties broken by smallest committed bytes
 * (cheapest to bring back). Pure host-field selection — no session is ever
 * touched. */
static void test_session_eviction_victim_selection(void) {
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;
    session_slot slots[PULSAR_SESSION_POOL_CAP];
    memset(slots, 0, sizeof(slots));
    for (int i = 0; i < PULSAR_SESSION_POOL_CAP; i++) slots[i].provisioned = true;
    slots[0].last_serviced_us = 1;              /* oldest of all — but pinned */
    slots[0].est_cost_bytes = 9ull * GiB;
    slots[1].last_serviced_us = 100;
    slots[1].est_cost_bytes = 3ull * GiB;
    slots[2].last_serviced_us = 50;
    slots[2].est_cost_bytes = 2ull * GiB;
    slots[3].last_serviced_us = 50;             /* LRU tie with slot 2 */
    slots[3].est_cost_bytes = 1ull * GiB;       /* ...but cheaper to restore */

    TEST_ASSERT(server_evict_pick_victim(slots, 4, NULL) == 3); /* LRU tie-break */
    bool protect[PULSAR_SESSION_POOL_CAP] = {0};
    protect[3] = true;
    TEST_ASSERT(server_evict_pick_victim(slots, 4, protect) == 2); /* protected skipped */
    slots[2].active_job = (struct job *)&slots;                    /* busy skipped */
    TEST_ASSERT(server_evict_pick_victim(slots, 4, protect) == 1);
    slots[1].provisioned = false;                                  /* hole skipped */
    TEST_ASSERT(server_evict_pick_victim(slots, 4, protect) == -1);
    protect[3] = false;
    TEST_ASSERT(server_evict_pick_victim(slots, 4, protect) == 3);
    /* n_slots bounds the scan: slot 3 invisible when only 3 are published */
    TEST_ASSERT(server_evict_pick_victim(slots, 3, protect) == -1);
    /* slot 0 alone is never a victim */
    TEST_ASSERT(server_evict_pick_victim(slots, 1, NULL) == -1);
}



/* Routing decision (task #30): the choose-vs-provision gate. Through v0.2.0
 * the gate was best_common == 0, unreachable for rendered chat traffic
 * (every rendered prompt shares the template header — measured common = 4-9
 * tokens across DISTINCT conversations in the task-#24 bounce repro), so
 * sequential conversations always clobbered slot 0 and the pool provisioned
 * zero slots. The classifier reclassifies header-deep matches as no-match
 * for the routing decision only. T mirrors the startup-derived threshold
 * (template header tokens + PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS);
 * the decision must hold for any plausible derivation, so the matrix uses
 * the allowance floor. Owner routing (live tool-state continuations) sits
 * UPSTREAM of this classifier in choose_slot_for_job and is untouched;
 * its slot lookup needs a live session frontier, so it is exercised by the
 * e2e gates rather than here. */
static void test_slot_route_trivial_match_decision(void) {
    const int T = PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS;
    /* zero common vs a warm 5.2k-token conversation: provision (the one
     * case the v0.2.0 gate did handle — behavior kept) */
    TEST_ASSERT(server_slot_match_is_trivial(0, 5200, T));
    /* trivial common (template header only, the measured bounce shape):
     * THE FIX — a different conversation must not clobber a warm slot */
    TEST_ASSERT(server_slot_match_is_trivial(4, 5200, T));
    TEST_ASSERT(server_slot_match_is_trivial(9, 5200, T));
    TEST_ASSERT(server_slot_match_is_trivial(T - 1, 5200, T));
    /* real common (long shared prefix: a client resending a longer version
     * of the same prompt, or the documented stateless-continuation
     * pattern): reuse the warm slot, never provision away from it */
    TEST_ASSERT(!server_slot_match_is_trivial(T, 5200, T));
    TEST_ASSERT(!server_slot_match_is_trivial(5175, 5900, T));
    /* empty slot: nothing to protect, reuse it (never "clobbers") */
    TEST_ASSERT(!server_slot_match_is_trivial(0, 0, T));
    /* short same-conversation continuation: common covers nearly the whole
     * slot state — stay on the warm slot even though common < T */
    TEST_ASSERT(!server_slot_match_is_trivial(38, 40, T));
    /* sub-threshold warm tail past the match: clobbering costs a sub-second
     * re-prefill, a fresh provisioning costs seconds — reuse (deliberate
     * semantic change from v0.2.0, which provisioned for pos in (0, T)) */
    TEST_ASSERT(!server_slot_match_is_trivial(0, T - 1, T));
    /* boundary: destroyed tail exactly at the threshold provisions */
    TEST_ASSERT(server_slot_match_is_trivial(0, T, T));
    TEST_ASSERT(server_slot_match_is_trivial(T - 1, 2 * T - 1, T));
    TEST_ASSERT(!server_slot_match_is_trivial(T - 1, 2 * T - 2, T));
}



/* Routing probe for thinking chats (task #30): a slot whose live thinking
 * binding byte-matches the request's visible transcript is that
 * conversation's warm continuation, even when the token common prefix is
 * header-short (the client replays visible content; the slot's sampled
 * frontier holds the hidden reasoning). choose_slot_for_job must route such
 * a request back to its slot instead of provisioning a "fresh conversation"
 * slot for it. Host fields only — the session is never touched (the caller
 * passes the live position). */
static void test_thinking_binding_routes_visible_continuation(void) {
    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);

    char vis[] = "<BOS>sys<U>hi<A></think>hello<EOS>";
    session_slot sl;
    memset(&sl, 0, sizeof(sl));
    sl.thinking_live.valid = true;
    sl.thinking_live.visible_text = vis;
    sl.thinking_live.visible_len = strlen(vis);
    sl.thinking_live.live_tokens = 40;

    char prompt[] = "<BOS>sys<U>hi<A></think>hello<EOS><U>again";
    request req;
    memset(&req, 0, sizeof(req));
    req.kind = REQ_CHAT;
    req.api = API_OPENAI;
    req.prompt_text = prompt;

    /* the continuation matches its slot at the remembered frontier */
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == strlen(vis));
    /* frontier moved (slot served someone else meanwhile): stale, no match */
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 41) == 0);
    /* a different conversation sharing only the header must not match —
     * longer than the binding key so the BYTE COMPARE is what rejects it,
     * not the visible_len < prompt_len guard (2026-07-16 review) */
    char other[] = "<BOS>sys<U>completely different much longer conversation";
    req.prompt_text = other;
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == 0);
    /* an exact replay (no new suffix) is not a continuation */
    char exact[] = "<BOS>sys<U>hi<A></think>hello<EOS>";
    req.prompt_text = exact;
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == 0);
    /* owner-routed protocols resolve via live call ids upstream; the probe
     * must not claim them */
    req.prompt_text = prompt;
    req.api = API_RESPONSES;
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == 0);
    req.api = API_OPENAI;
    req.kind = REQ_COMPLETION;
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == 0);
    /* invalidated binding (clobbered/evicted slot) never matches */
    req.kind = REQ_CHAT;
    sl.thinking_live.valid = false;
    TEST_ASSERT(s.thinking_live_binds_prompt(&sl, &req, 40) == 0);

    pthread_mutex_destroy(&s.tool_mu);
}



/* Multi-session increment 2: while a job is bound to a slot, the worker's
 * send_all() routes through a slot_writer — writes that do not fit the socket
 * buffer defer instead of blocking, and flushes deliver every byte in order.
 * The wire stream must be identical to the blocking path. */
static void test_slot_writer_defers_and_preserves_order(void) {
    signal(SIGPIPE, SIG_IGN);
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    int small = 4096;
    setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
    setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &small, sizeof(small));
    set_client_socket_nonblocking(sv[0]);
    set_client_socket_nonblocking(sv[1]);

    slot_writer w;
    slot_writer_init(&w, sv[0]);
    slot_writer_install(&w);

    const size_t total = 512 * 1024;
    char *pattern = (char *)server_xmalloc(total);
    for (size_t i = 0; i < total; i++) {
        pattern[i] = (char)((i * 31u + (i >> 8)) & 0xff);
    }
    char *received = (char *)server_xmalloc(total);
    size_t sent = 0, got = 0;
    bool deferred = false;

    /* The peer reads nothing during the sends, so the tiny kernel buffers fill
     * and everything past them must defer into the writer queue. */
    while (sent < total) {
        size_t nchunk = 700 + (sent % 900); /* odd sizes straddle buffers */
        if (nchunk > total - sent) nchunk = total - sent;
        TEST_ASSERT(send_all(sv[0], pattern + sent, nchunk)); /* defers, never fails */
        sent += nchunk;
        if (w.pending.len > w.off) deferred = true;
    }
    TEST_ASSERT(deferred);

    /* Drain: alternate the worker-side flush with peer reads. Bounded so a
     * writer regression that stops delivering (without setting failed) shows
     * up as an assertion instead of a hung test suite. */
    int stagnant = 0;
    while (got < total) {
        TEST_ASSERT(slot_writer_flush(&w));
        char tmp[8192];
        ssize_t r = recv(sv[1], tmp, sizeof(tmp), MSG_DONTWAIT);
        if (r > 0) {
            TEST_ASSERT(got + (size_t)r <= total);
            memcpy(received + got, tmp, (size_t)r);
            got += (size_t)r;
            stagnant = 0;
        } else if (++stagnant >= 100000) {
            TEST_ASSERT(!"slot_writer drain made no progress");
            break;
        }
    }
    TEST_ASSERT(!w.failed);
    TEST_ASSERT(w.pending.len == w.off); /* everything reached the wire */
    TEST_ASSERT(memcmp(pattern, received, total) == 0);

    slot_writer_free(&w); /* also uninstalls */
    free(pattern);
    free(received);
    close(sv[0]);
    close(sv[1]);
}



/* A peer that accepts no bytes past the stall deadline fails the stream, and
 * every later write on the failed writer reports failure — matching the old
 * blocking send_all semantics that the generation loop depends on. */
static void test_slot_writer_stall_times_out(void) {
    signal(SIGPIPE, SIG_IGN);
    int sv[2];
    TEST_ASSERT(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    int small = 4096;
    setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
    setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &small, sizeof(small));
    set_client_socket_nonblocking(sv[0]);

    slot_writer w;
    slot_writer_init(&w, sv[0]);
    slot_writer_install(&w);

    char blob[8192];
    memset(blob, 'q', sizeof(blob));
    for (int i = 0; i < 64; i++) {
        TEST_ASSERT(send_all(sv[0], blob, sizeof(blob))); /* peer never reads: defer */
    }
    TEST_ASSERT(w.pending.len > w.off);
    /* Force the deadline instead of sleeping PULSAR_SERVER_SEND_STALL_TIMEOUT_MS. */
    w.stall_deadline_ms = 1;
    TEST_ASSERT(!slot_writer_flush(&w));
    TEST_ASSERT(w.failed);
    TEST_ASSERT(!send_all(sv[0], "x", 1));
    TEST_ASSERT(!slot_writer_drain(&w));

    slot_writer_free(&w);
    close(sv[0]);
    close(sv[1]);
}



/* ---- disk-KV default-on resolution matrix (task #31) ---- */

static char *test_env_save(const char *name) {
    const char *v = getenv(name);
    return v ? xstrdup(v) : NULL;
}

static void test_env_restore(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}



static void test_kv_disk_default_dir_resolution(void) {
    char *old_xdg = test_env_save("XDG_CACHE_HOME");
    char *old_home = test_env_save("HOME");

    /* XDG_CACHE_HOME wins; nonexistent model path falls back to its raw
     * basename; the .gguf extension is stripped case-insensitively. */
    setenv("XDG_CACHE_HOME", "/tmp/ds4-kvtest-xdg", 1);
    char *d = server_default_kv_disk_dir("/no/such/dir/ds4flash-test.gguf");
    TEST_ASSERT(d && !strcmp(d, "/tmp/ds4-kvtest-xdg/ds4/kv-ds4flash-test"));
    free(d);

    /* The gguf/model.gguf ACTIVE-POINTER symlink must key by the versioned
     * artifact it points at, not by the pointer name. */
    char tmpl[] = "/tmp/ds4-kvtest-XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (dir) {
        char artifact[PATH_MAX], pointer[PATH_MAX];
        snprintf(artifact, sizeof(artifact), "%s/ds4flash-v9-test.gguf", dir);
        snprintf(pointer, sizeof(pointer), "%s/model.gguf", dir);
        FILE *fp = fopen(artifact, "w");
        TEST_ASSERT(fp != NULL);
        if (fp) fclose(fp);
        TEST_ASSERT(symlink(artifact, pointer) == 0);
        d = server_default_kv_disk_dir(pointer);
        TEST_ASSERT(d && !strcmp(d, "/tmp/ds4-kvtest-xdg/ds4/kv-ds4flash-v9-test"));
        free(d);
        unlink(pointer);
        unlink(artifact);
        rmdir(dir);
    }

    /* Without XDG_CACHE_HOME, fall back to ~/.cache; shell-hostile bytes in
     * the model name are sanitized. */
    unsetenv("XDG_CACHE_HOME");
    setenv("HOME", "/tmp/ds4-kvtest-home", 1);
    d = server_default_kv_disk_dir("weird name!.GGUF");
    TEST_ASSERT(d && !strcmp(d, "/tmp/ds4-kvtest-home/.cache/ds4/kv-weird_name_"));
    free(d);

    /* A relative XDG_CACHE_HOME is ignored per the XDG spec (it would key
     * the cache off the current working directory). */
    setenv("XDG_CACHE_HOME", "relative-cache", 1);
    d = server_default_kv_disk_dir("x.gguf");
    TEST_ASSERT(d && !strcmp(d, "/tmp/ds4-kvtest-home/.cache/ds4/kv-x"));
    free(d);
    unsetenv("XDG_CACHE_HOME");

    /* No cache home at all: resolution reports NULL (server then runs with
     * the disk cache disabled instead of guessing a path). */
    unsetenv("HOME");
    TEST_ASSERT(server_default_kv_disk_dir("x.gguf") == NULL);

    test_env_restore("XDG_CACHE_HOME", old_xdg);
    test_env_restore("HOME", old_home);
}



static void test_kv_disk_flag_matrix(void) {
    char *old_xdg = test_env_save("XDG_CACHE_HOME");
    setenv("XDG_CACHE_HOME", "/tmp/ds4-kvtest-xdg", 1);

    /* Unset: the default directory is resolved (default-on). */
    {
        char *argv[] = {(char *)"pulsar-server"};
        server_config c = parse_options(1, argv);
        server_resolve_kv_disk_dir(&c);
        TEST_ASSERT(c.kv_disk_dir != NULL);
        TEST_ASSERT(c.kv_disk_dir &&
                    !strncmp(c.kv_disk_dir, "/tmp/ds4-kvtest-xdg/ds4/kv-",
                             strlen("/tmp/ds4-kvtest-xdg/ds4/kv-")));
        free((char *)c.kv_disk_dir);
    }

    /* Explicit path: used verbatim, exactly as before. */
    {
        char *argv[] = {(char *)"pulsar-server",
                        (char *)"--kv-disk-dir", (char *)"/tmp/explicit-kv"};
        server_config c = parse_options(3, argv);
        server_resolve_kv_disk_dir(&c);
        TEST_ASSERT(c.kv_disk_dir && !strcmp(c.kv_disk_dir, "/tmp/explicit-kv"));
    }

    /* Empty value: opt-out. */
    {
        char *argv[] = {(char *)"pulsar-server",
                        (char *)"--kv-disk-dir", (char *)""};
        server_config c = parse_options(3, argv);
        server_resolve_kv_disk_dir(&c);
        TEST_ASSERT(c.kv_disk_dir == NULL);
    }

    /* --no-kv-disk: opt-out. */
    {
        char *argv[] = {(char *)"pulsar-server", (char *)"--no-kv-disk"};
        server_config c = parse_options(2, argv);
        server_resolve_kv_disk_dir(&c);
        TEST_ASSERT(c.kv_disk_dir == NULL);
    }

    /* Last kv-disk flag wins: opt-out then explicit path re-enables. */
    {
        char *argv[] = {(char *)"pulsar-server", (char *)"--no-kv-disk",
                        (char *)"--kv-disk-dir", (char *)"/tmp/explicit-kv"};
        server_config c = parse_options(4, argv);
        server_resolve_kv_disk_dir(&c);
        TEST_ASSERT(c.kv_disk_dir && !strcmp(c.kv_disk_dir, "/tmp/explicit-kv"));
    }

    test_env_restore("XDG_CACHE_HOME", old_xdg);
}



static void test_kv_cache_open_unusable_dir_disables(void) {
    /* Uncreatable path: open fails, cache stays disabled, no crash. */
    kv_disk_cache kc = {0};
    TEST_ASSERT(!kv_cache_open(&kc, "/proc/ds4-kvtest-nope/kv", 64, false,
                               kv_cache_default_options()));
    TEST_ASSERT(!kc.enabled);
    kv_cache_close(&kc);

    /* Pre-existing read-only directory: refused up front (writability probe)
     * instead of failing store-by-store later.  Mode bits do not bind root. */
    char tmpl[] = "/tmp/ds4-kvtest-ro-XXXXXX";
    char *dir = mkdtemp(tmpl);
    TEST_ASSERT(dir != NULL);
    if (dir) {
        TEST_ASSERT(chmod(dir, 0500) == 0);
        kv_disk_cache ro = {0};
        bool opened = kv_cache_open(&ro, dir, 64, false,
                                    kv_cache_default_options());
        if (geteuid() == 0) {
            TEST_ASSERT(opened);
        } else {
            TEST_ASSERT(!opened);
            TEST_ASSERT(!ro.enabled);
        }
        kv_cache_close(&ro);
        chmod(dir, 0700);
        rmdir(dir);
    }
}



static void pulsar_server_unit_tests_run(void) {
    test_kv_disk_default_dir_resolution();
    test_kv_disk_flag_matrix();
    test_kv_cache_open_unusable_dir_disables();
    test_kv_admission_budget_math();
    test_mem_floor_admits_warmed_box_shape();
    test_session_eviction_ledger_math();
    test_session_eviction_victim_selection();
    test_slot_route_trivial_match_decision();
    test_thinking_binding_routes_visible_continuation();
    test_slot_writer_defers_and_preserves_order();
    test_slot_writer_stall_times_out();
    test_unterminated_think_stays_off_content();
    test_request_defaults_use_min_p_filtering();
    test_think_sampling_respects_explicit_params();
    test_reasoning_effort_mapping();
    test_api_thinking_controls_parse();
    test_render_think_max_prompt_prefix();
    test_render_non_thinking_prompt_closes_think();
    test_render_drops_old_reasoning_without_tools();
    test_render_preserves_reasoning_with_tools();
    test_render_chat_prompt_text_renders_tools_before_system();
    test_tool_schema_order_from_anthropic_schema();
    test_tool_schema_order_from_openai_tools();
    test_tool_schema_order_from_responses_tool_search();
    test_responses_function_named_tool_search_stays_function_call();
    test_responses_namespace_tool_schemas_restore_wire_namespace();
    test_responses_input_tool_search_output_loads_tools();
    test_responses_input_tool_search_output_rejects_bad_tools();
    test_responses_input_function_call_namespace_round_trips_to_dsml();
    test_responses_output_sends_tool_search_call_item();
    test_dsml_tool_args_preserve_call_order();
    test_openai_tool_args_preserve_call_order();
    test_anthropic_thinking_and_tool_args_preserve_call_order();
    test_context_length_error_uses_protocol_standard_shape();
    test_cors_headers_are_opt_in();
    test_cors_preflight_response_is_no_content();
    test_cors_sse_headers();
    test_anthropic_live_stream_sends_incremental_blocks();
    test_anthropic_usage_reports_cache_details();
    test_anthropic_tool_stream_sends_live_tool_use();
    test_openai_tool_stream_sends_incremental_text();
    test_openai_stream_usage_reports_cache_details();
    test_responses_usage_reports_cache_details();
    test_openai_chat_stream_splits_reasoning_without_tools();
    test_openai_tool_stream_sends_partial_arguments();
    test_openai_tool_stream_waits_for_incomplete_tool_tags();
    test_openai_tool_stream_sends_partial_raw_arguments();
    test_openai_tool_stream_holds_partial_dsml_entities();
    test_openai_tool_stream_holds_partial_utf8_arguments();
    test_openai_tool_stream_handles_multiple_calls();
    test_streaming_holds_partial_utf8();
    test_parse_short_dsml_and_canonical_suffix();
    test_dsml_parser_recovers_loose_nested_parameters();
    test_dsml_repair_produces_parseable_calls();
    test_tool_parse_failure_returns_recoverable_finish();
    test_invalid_dsml_tool_error_suffix_includes_system_prompt();
    test_thinking_dsml_is_not_executable_before_think_close();
    test_thinking_dsml_after_think_close_is_executable();
    test_tool_checkpoint_suffix_is_future_prompt_canonical();
    test_tool_checkpoint_minifies_json_parameters();
    test_tool_memory_replays_sampled_dsml();
    test_anthropic_tool_memory_replays_sampled_dsml();
    test_anthropic_live_tail_renders_tool_results_only();
    test_anthropic_tool_result_id_validation();
    test_anthropic_full_replay_allows_unknown_live_id();
    test_anthropic_tool_use_parses_before_role();
    test_tool_checkpoint_canonicalization_gate_exact_replay();
    test_responses_live_tail_renders_tool_outputs_only();
    test_responses_tool_output_id_validation();
    test_responses_stateless_tool_replay_requires_reasoning();
    test_responses_visible_suffix_matches_client_replay();
    test_exact_dsml_tool_replay_can_be_disabled();
    test_dsml_decode_state_separates_structure_and_payload();
    test_tool_memory_max_ids_prunes_oldest();
    test_kv_tool_map_filters_by_dsml_text();
    test_kv_tool_map_restores_before_prompt_render();
    test_thinking_checkpoint_canonical_matches_future_prompt();
    test_thinking_canonical_empty_content();
    test_thinking_canonical_multi_turn();
    test_thinking_canonical_with_tools_preserves_reasoning();
    test_thinking_canonical_non_thinking_mode_noop();
    test_tool_separator_whitespace_is_not_content();
    test_dsml_prompt_escapes_tool_supplied_text();
    test_stop_list_parses_all_sequences();
    test_stop_list_streaming_holds_and_trims_stop_text();
    test_json_skip_has_nesting_limit();
    test_json_parser_handles_tool_heavy_requests();
    test_json_string_handles_surrogates();
    test_model_metadata_clamps_completion_to_context();
    test_client_socket_nonblocking_flag();
    test_thinking_state_tracks_prompt_and_generated_tags();
    test_thinking_checkpoint_remember_gate();
    test_tool_marker_state_ignores_orphan_end();
    test_canonical_rewrite_rebuilds_when_live_tail_changes();
    test_kv_cache_store_len_uses_configured_boundary();
    test_kv_cache_chat_anchor_uses_last_user_before_assistant();
    test_kv_cache_chat_anchor_ignores_multiturn_tail();
    test_kv_cache_continued_uses_aligned_frontiers();
    test_kv_cache_cold_store_suppresses_duplicate_continued_boundary();
    test_kv_cache_file_size_must_fit_budget();
    test_sha1_bytes_hex_matches_known_vector();
    test_kv_cache_lookup_uses_longest_text_prefix();
    test_kv_cache_lookup_rejects_wrong_model();
    test_kv_cache_lookup_rejects_stale_payload_abi();
    test_kv_cache_eviction_values_fresh_snapshots();
    test_kv_cache_eviction_prefers_anchor_reason();
    test_kv_cache_eviction_prefers_sys_prefix_over_cold();
    test_kv_cache_eviction_makes_room_before_store();
    test_kv_cache_eviction_ignores_oversize_incoming();
    test_kv_cache_eviction_prefers_superseded_continued_prefix();
    test_kv_cache_eviction_keeps_smaller_context_prefix();
    test_kv_cache_eviction_score_decays_stale_hits();
    test_kv_cache_eviction_decayed_hits_tie_break_by_age();
    test_kv_cache_eviction_keeps_aligned_continued_frontiers();
}



#ifndef PULSAR_SERVER_TEST_NO_MAIN

int main(void) {
    pulsar_server_unit_tests_run();
    if (test_failures) {
        fprintf(stderr, "pulsar-server tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("pulsar-server tests: ok");
    return 0;
}


#endif


#endif /* PULSAR_SERVER_TEST */
