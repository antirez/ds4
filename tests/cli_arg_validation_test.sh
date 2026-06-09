#!/bin/sh
set -eu

tmp="${TMPDIR:-/tmp}/ds4-cli-arg-validation-test.$$"
trap 'rm -f "$tmp"' EXIT HUP INT TERM

fail() {
    echo "cli_arg_validation_test: $*" >&2
    if [ -s "$tmp" ]; then
        cat "$tmp" >&2
    fi
    exit 1
}

expect_ok() {
    if ! "$@" >"$tmp" 2>&1; then
        fail "expected success: $*"
    fi
}

expect_fail_contains() {
    needle="$1"
    shift
    if "$@" >"$tmp" 2>&1; then
        fail "expected failure: $*"
    fi
    if ! grep -F -- "$needle" "$tmp" >/dev/null; then
        fail "missing error text '$needle': $*"
    fi
}

for bin in ./ds4 ./ds4-agent ./ds4-eval; do
    expect_fail_contains "invalid value for --seed" "$bin" --seed -1 --help
    expect_fail_contains "invalid value for --seed" "$bin" --seed " -1" --help
    expect_fail_contains "invalid value for --seed" "$bin" --seed +1 --help
    expect_fail_contains "invalid value for --seed" "$bin" --seed 0 --help
    expect_fail_contains "invalid value for --seed" "$bin" --seed 18446744073709551616 --help
    expect_ok "$bin" --seed 1 --help
    expect_ok "$bin" --seed 18446744073709551615 --help
done

expect_fail_contains "--port must be between 1 and 65535" ./ds4-server --port 0
expect_fail_contains "--port must be between 1 and 65535" ./ds4-server --port -1
expect_fail_contains "--port must be between 1 and 65535" ./ds4-server --port 65536
expect_ok ./ds4-server --port 65535 --help

echo "cli-arg-validation-test: OK"
