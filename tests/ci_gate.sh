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

# Linked without a GPU runtime, so these build and run on every host.
PORTABLE_TESTS="q4k-dot-test mxfp4-dot-test test-engram test-web-recovery"
PORTABLE_BINS="tests/test_layer_pack tests/test_prompt_prefix tests/test_gpu_args tests/test_image_decode"

build $PORTABLE_TESTS $PORTABLE_BINS

if [ "$(uname -s)" = Darwin ]; then
    # Linux links the same tests through nvcc, which CI hosts do not have.
    build test-session-state tests/test_engine_mgpu_placement \
        tests/test_deepseek4_vision_image tests/test_sampling \
        ds4 ds4-server ds4-bench ds4-agent ds4-eval ds4_test ds4_agent_test
    PORTABLE_BINS="$PORTABLE_BINS tests/test_engine_mgpu_placement"
    PORTABLE_BINS="$PORTABLE_BINS tests/test_deepseek4_vision_image tests/test_sampling"
    run ./ds4-eval --validate-cases
    run ./ds4-eval --self-test-extractors
    run ./ds4_test --server
    run ./ds4_agent_test
else
    # Compile check only: the CPU backend is reference code.
    build cpu
fi

for bin in $PORTABLE_BINS; do
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
