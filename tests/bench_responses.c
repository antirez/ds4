/* Compare full-history and response-ID parsing, without model inference. */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#include <assert.h>

ds4_engine *responses_bench_engine_open(const char *path);
void responses_bench_engine_close(ds4_engine *e);

static volatile int parsed_tokens;

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : "ds4flash.gguf";
    const char *path = argc > 2 ? argv[2] :
        "tests/test-vectors/flash-0731/prompts/long_code_audit.txt";
    int iterations = argc > 3 ? atoi(argv[3]) : 100;
    if (iterations < 1) return 2;
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return 1; }
    buf input = {0};
    char chunk[4096];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), fp))) buf_append(&input, chunk, n);
    if (ferror(fp)) { fclose(fp); return 1; }
    fclose(fp);

    buf body = {0};
    buf_puts(&body, "{\"input\":[{\"role\":\"user\",\"content\":");
    json_escape(&body, input.ptr ? input.ptr : "");
    buf seed = {0};
    buf_append(&seed, body.ptr, body.len);
    buf_puts(&seed, "}]}");
    buf_puts(&body, "},{\"role\":\"assistant\",\"content\":\"A\"},"
                    "{\"role\":\"user\",\"content\":\"Continue.\"}]}");
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    s.engine = responses_bench_engine_open(model);
    s.ctx_size = INT_MAX;
    request initial;
    char err[160];
    assert(parse_responses_request(s.engine, &s, seed.ptr, 128, s.ctx_size,
                                   &initial, err, sizeof(err)));
    /* Represent the completed Flash response before the new user turn. */
    ds4_tokens completion = {0};
    ds4_tokenize_rendered_chat(s.engine, "</think>A", &completion);
    response_state_entry *entry = response_state_remember(
        &s, 0, initial.prompt.len + completion.len, 1,
        "resp_benchmark", &initial, NULL, false);
    ds4_tokens_free(&completion);
    buf_free(&seed);
    assert(entry);
    response_state_release(&s, entry);
    request_free(&initial);

    const char *delta = "{\"previous_response_id\":\"resp_benchmark\","
                        "\"input\":\"Continue.\"}";
    const char *bodies[] = {body.ptr, delta};
    const char *labels[] = {"full replay", "response id"};
    for (int arm = 0; arm < 2; arm++) {
        double start = now_sec();
        for (int i = 0; i < iterations; i++) {
            request r;
            assert(parse_responses_request(s.engine, &s, bodies[arm], 128, s.ctx_size,
                                           &r, err, sizeof(err)));
            parsed_tokens += r.prompt.len;
            request_free(&r);
        }
        printf("%s: %.0f ns/op (%d iterations)\n", labels[arm],
               (now_sec() - start) * 1e9 / iterations, iterations);
    }
    response_state_index_free(&s.response_states);
    pthread_mutex_destroy(&s.tool_mu);
    responses_bench_engine_close(s.engine);
    buf_free(&body);
    buf_free(&input);
    return 0;
}
