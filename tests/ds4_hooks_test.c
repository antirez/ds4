#define _POSIX_C_SOURCE 200809L
#include "../ds4_hooks.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;
#define TEST_ASSERT(expr) do { if (!(expr)) { \
    fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #expr); \
    failures++; \
} } while (0)

static ds4_hook_payload test_payload(ds4_hook_event event) {
    ds4_hook_payload payload = {
        .event = event,
        .model = "test-model",
        .user_text = "say \"hello\"\nnext",
        .response_text = event == DS4_HOOK_AFTER_RESPONSE ? "hello\\world\n" : NULL,
        .response_index = 2,
        .generated_tokens = 3,
    };
    return payload;
}

static void test_payload_json(void) {
    ds4_hook_payload payload = test_payload(DS4_HOOK_AFTER_RESPONSE);
    char *json = ds4_hook_payload_json(&payload);
    TEST_ASSERT(json != NULL);
    if (json) {
        TEST_ASSERT(strstr(json, "\"event\":\"after_response\"") != NULL);
        TEST_ASSERT(strstr(json, "say \\\"hello\\\"\\nnext") != NULL);
        TEST_ASSERT(strstr(json, "hello\\\\world\\n") != NULL);
        TEST_ASSERT(strstr(json, "\"generated_tokens\":3") != NULL);
    }
    free(json);
}

static void test_disabled_hook(void) {
    ds4_hook_config config = {0};
    ds4_hook_payload payload = test_payload(DS4_HOOK_BEFORE_RESPONSE);
    ds4_hook_result result;
    TEST_ASSERT(ds4_hook_run(&config, &payload, &result) == 0);
    TEST_ASSERT(!result.attempted);
}

static void test_hook_stdin(void) {
    char path[] = "/tmp/ds4-hook-test-XXXXXX";
    int fd = mkstemp(path);
    TEST_ASSERT(fd >= 0);
    if (fd < 0) return;
    close(fd);
    char command[512];
    snprintf(command, sizeof(command), "cat > '%s'", path);
    ds4_hook_config config = {.after_response_command = command, .timeout_seconds = 2};
    ds4_hook_payload payload = test_payload(DS4_HOOK_AFTER_RESPONSE);
    ds4_hook_result result;
    TEST_ASSERT(ds4_hook_run(&config, &payload, &result) == 0);
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (fp) {
        char data[2048];
        size_t n = fread(data, 1, sizeof(data) - 1, fp);
        data[n] = '\0';
        fclose(fp);
        TEST_ASSERT(strstr(data, "\"model\":\"test-model\"") != NULL);
    }
    unlink(path);
}

static void test_failures(void) {
    ds4_hook_payload before = test_payload(DS4_HOOK_BEFORE_RESPONSE);
    ds4_hook_config bad = {.before_response_command = "exit 7", .timeout_seconds = 2};
    ds4_hook_result result;
    TEST_ASSERT(ds4_hook_run(&bad, &before, &result) != 0);
    TEST_ASSERT(result.exit_code == 7);

    ds4_hook_payload after = test_payload(DS4_HOOK_AFTER_RESPONSE);
    ds4_hook_config slow = {.after_response_command = "sleep 5", .timeout_seconds = 1};
    TEST_ASSERT(ds4_hook_run(&slow, &after, &result) != 0);
    TEST_ASSERT(result.timed_out);
}

int main(void) {
    test_payload_json();
    test_disabled_hook();
    test_hook_stdin();
    test_failures();
    if (failures) return 1;
    puts("ds4 hook tests: ok");
    return 0;
}
