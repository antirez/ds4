#!/usr/bin/env bash
# Pull request gate: checks that need neither a model nor a GPU backend.
# Run from a clean tree: objects that are already built emit no warnings.
set -euo pipefail

cd "$(dirname "$0")/.."

LOG=$(mktemp)
JOBS=$(getconf _NPROCESSORS_ONLN)

build() {
    make -j"$JOBS" "$@" 2>&1 | tee -a "$LOG"
}

run() {
    echo "--- $*"
    "$@" 2>&1 | tee -a "$LOG"
}

UNIT_TESTS="tests/test_layer_pack tests/test_prompt_prefix tests/test_gpu_args
    tests/test_image_decode tests/test_engine_mgpu_placement
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

build q4k-dot-test mxfp4-dot-test test-engram test-session-state \
    test-web-recovery $UNIT_TESTS

run ./ds4-eval --validate-cases
run ./ds4-eval --self-test-extractors
run "$SERVER_TEST" --server
run "$AGENT_TEST"

for bin in $UNIT_TESTS; do
    run "./$bin"
done

run ./tests/test_gpu_args_cli.sh

for bin in ds4 ds4-server ds4-agent; do
    run "./$bin" --help >/dev/null
done

if grep -E 'warning( #[0-9]+)?:' "$LOG"; then
    echo "ci_gate: compiler warnings are build failures" >&2
    exit 1
fi
