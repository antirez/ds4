#define DS4_AGENT_TEST
#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"

int main(void) {
    ds4_agent_unit_tests_run();
    if (agent_test_failures) {
        fprintf(stderr, "ds4-agent tests: %d failure(s)\n",
                agent_test_failures);
        return 1;
    }
    puts("ds4-agent tests: ok");
    return 0;
}
