#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test_log=$(mktemp)
trap 'rm -f "$test_log"' EXIT

./ds4-bench --help benchmark >"$test_log" 2>&1
if ! grep -q -- "--gen-warmup-tokens N" "$test_log"; then
    echo "FAIL: benchmark help omits --gen-warmup-tokens" >&2
    exit 1
fi

set +e
./ds4-bench --prompt-file /dev/null -m /dev/null \
    --gen-tokens 16 --gen-warmup-tokens 32 >"$test_log" 2>&1
test_rc=$?
set -e
if [ "$test_rc" -ne 2 ] ||
   ! grep -q -- "--gen-warmup-tokens must be <= --gen-tokens" "$test_log"; then
    echo "FAIL: inconsistent warmup was not rejected during option parsing" >&2
    sed -n '1,20p' "$test_log" >&2
    exit 1
fi

set +e
./ds4-bench --prompt-file /dev/null -m /dev/null \
    --gen-warmup-tokens -1 >"$test_log" 2>&1
test_rc=$?
set -e
if [ "$test_rc" -ne 2 ] ||
   ! grep -q -- "invalid value for --gen-warmup-tokens" "$test_log"; then
    echo "FAIL: negative warmup was not rejected during option parsing" >&2
    sed -n '1,20p' "$test_log" >&2
    exit 1
fi

echo "test_bench_metrics_cli: PASS"
