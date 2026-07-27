#define DS4_AGENT_TEST
#define DS4_AGENT_TEST_NO_MAIN
#include "../src/agent/config_types.cpp"
#include "../src/agent/util_cli.cpp"
#include "../src/agent/prompt_queue.cpp"
#include "../src/agent/trace.cpp"
#include "../src/agent/dsml_parser.cpp"
#include "../src/agent/markdown.cpp"
#include "../src/agent/tool_viz.cpp"
#include "../src/agent/progress.cpp"
#include "../src/agent/kvstore_session.cpp"
#include "../src/agent/session_ui.cpp"
#include "../src/agent/file_tools.cpp"
#include "../src/agent/edit_tools.cpp"
#include "../src/agent/bash_jobs.cpp"
#include "../src/agent/tool_dispatch.cpp"
#include "../src/agent/compaction.cpp"
#include "../src/agent/worker.cpp"
#include "../src/agent/worker_sync.cpp"
#include "../src/agent/terminal_ui.cpp"
#include "../src/agent/runtime.cpp"

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
