#!/usr/bin/env bash
# Pull request gate: checks that need no model.
# Usage: tests/ci_gate.sh [build|test]. With no argument it does both.
set -euo pipefail

cd "$(dirname "$0")/.."

UNIT_TESTS="tests/test_q4k_dot tests/test_mxfp4_dot tests/test_engram
    tests/test_session_state tests/test_tp_commands tests/test_tp_rdma
    tests/test_tp_tcp tests/test_layer_pack tests/test_prompt_prefix
    tests/test_gpu_args tests/test_image_decode tests/test_engine_mgpu_placement
    tests/test_deepseek4_vision_image tests/test_sampling tests/test_linux_memory
    tests/test_qwen4_memory tests/test_qwen4_mtp_policy
    tests/test_metal_device_policy"

# Only this block differs by platform. CI hosts have Metal on macOS but no
# nvcc on Linux, where the runners link the CPU backend. Every platform runs
# the tests above; macOS also runs the Metal tests, as the Makefile targets
# that hold their arguments and environment.
if [ "$(uname -s)" = Darwin ]; then
    BINARIES="ds4 ds4-server ds4-bench ds4-agent ds4-eval ds4_test ds4_agent_test"
    SERVER_TEST=./ds4_test
    AGENT_TEST=./ds4_agent_test
    PLATFORM_TESTS="tests/test_qwen4_ssd_experts tests/test_qwen4_resident_arithmetic
        tests/test_metal_q8_reduction tests/test_metal_q8_gemv_reference
        tests/test_metal_qwen4_moe_half tests/test_metal_qwen4_hc
        tests/test_metal_qwen4_moe_mid tests/test_metal_qwen4_moe_down"
    PLATFORM_TARGETS="test-qwen4-ssd-experts test-qwen4-resident-arithmetic
        check-metal-q8-reduction check-metal-q8-gemv-reference
        check-metal-qwen4-moe-half check-metal-qwen4-hc
        check-metal-qwen4-moe-mid check-metal-qwen4-moe-down"
else
    # Compile check only: the CPU backend is reference code.
    BINARIES="cpu ds4_test_cpu ds4_agent_test_cpu"
    SERVER_TEST=./ds4_test_cpu
    AGENT_TEST=./ds4_agent_test_cpu
    PLATFORM_TESTS=
    PLATFORM_TARGETS=
fi

build_all() {
    # Objects that are already built emit no warnings and would pass unchecked.
    if compgen -G '*.o' >/dev/null || compgen -G 'tests/*.o' >/dev/null; then
        echo "ci_gate: run make clean first" >&2
        exit 2
    fi

    local log jobs
    log=$(mktemp)
    jobs=$(getconf _NPROCESSORS_ONLN)
    make -j"$jobs" $BINARIES 2>&1 | tee -a "$log"
    make -j"$jobs" tests/test_web_recovery $UNIT_TESTS $PLATFORM_TESTS 2>&1 | tee -a "$log"

    # Only build output counts: some tests print runtime warnings on purpose.
    if grep -E 'warning( #[0-9]+)?:' "$log"; then
        echo "ci_gate: compiler warnings are build failures" >&2
        exit 1
    fi
}

run() {
    echo "--- $*"
    "$@"
}

# Tests run one at a time, so timing-sensitive ones do not compete for CPU.
run_tests() {
    run ./ds4-eval --validate-cases
    run ./ds4-eval --self-test-extractors
    run "$SERVER_TEST" --server
    run "$AGENT_TEST"

    for bin in $UNIT_TESTS; do
        run "./$bin"
    done

    for target in $PLATFORM_TARGETS; do
        run make -s "$target"
    done

    run python3 tests/test_qwen4_moe_mm_compact.py
    run python3 tests/test_web_recovery.py ./tests/test_web_recovery
    run ./tests/test_gpu_args_cli.sh

    for bin in ds4 ds4-server ds4-agent; do
        echo "--- ./$bin --help"
        "./$bin" --help >/dev/null
    done
}

case "${1:-all}" in
    build) build_all ;;
    test) run_tests ;;
    all) build_all; run_tests ;;
    *) echo "usage: tests/ci_gate.sh [build|test]" >&2; exit 2 ;;
esac
