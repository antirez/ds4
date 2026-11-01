#!/usr/bin/env bash
# Pull request gate: checks that need neither a model nor a GPU backend.
set -euo pipefail

cd "$(dirname "$0")/.."

# Objects that are already built emit no warnings and would pass unchecked.
if compgen -G '*.o' >/dev/null || compgen -G 'tests/*.o' >/dev/null; then
    echo "ci_gate: run make clean first" >&2
    exit 2
fi

BUILD_LOG=$(mktemp)
JOBS=$(getconf _NPROCESSORS_ONLN)

build() {
    make -j"$JOBS" "$@" 2>&1 | tee -a "$BUILD_LOG"
}

run() {
    echo "--- $*"
    "$@"
}

UNIT_TESTS="tests/test_q4k_dot tests/test_mxfp4_dot tests/test_engram
    tests/test_session_state tests/test_tp_commands tests/test_tp_rdma
    tests/test_tp_tcp tests/test_layer_pack tests/test_prompt_prefix
    tests/test_gpu_args tests/test_image_decode tests/test_engine_mgpu_placement
    tests/test_deepseek4_vision_image tests/test_sampling"

# Only the build differs: CI hosts have Metal on macOS but no nvcc on Linux,
# where the runners link the CPU backend. Both run the same tests below.
if [ "$(uname -s)" = Darwin ]; then
    build ds4 ds4-server ds4-bench ds4-agent ds4-eval ds4_test ds4_agent_test
    SERVER_TEST=./ds4_test
    AGENT_TEST=./ds4_agent_test
else
    # Compile check only: the CPU backend is reference code.
    build cpu ds4_test_cpu ds4_agent_test_cpu
    SERVER_TEST=./ds4_test_cpu
    AGENT_TEST=./ds4_agent_test_cpu
fi

build tests/test_web_recovery $UNIT_TESTS

# Only build output counts: some tests print runtime warnings on purpose.
if grep -E 'warning( #[0-9]+)?:' "$BUILD_LOG"; then
    echo "ci_gate: compiler warnings are build failures" >&2
    exit 1
fi

# Tests run one at a time, so timing-sensitive ones do not compete for CPU.
run ./ds4-eval --validate-cases
run ./ds4-eval --self-test-extractors
run "$SERVER_TEST" --server
run "$AGENT_TEST"

for bin in $UNIT_TESTS; do
    run "./$bin"
done

run python3 tests/test_web_recovery.py ./tests/test_web_recovery
run ./tests/test_gpu_args_cli.sh

for bin in ds4 ds4-server ds4-agent; do
    run "./$bin" --help >/dev/null
done
