#!/bin/sh
# One process per cell releases session allocations between measurements.
# This does not reset the OS file cache or the machine's thermal state.
#
# usage: speed-bench/concurrency_sweep.sh CSV [extra session_concurrency_bench options]

set -e
CSV=${1:?usage: concurrency_sweep.sh CSV [options]}
shift || true
BENCH=./speed-bench/session_concurrency_bench
CONCURRENCY=${CONCURRENCY:-"1 2 4 8 16"}
CONTEXTS=${CONTEXTS:-"0 4096 16384 32768 65536 131072"}
failed=0

for ctx in $CONTEXTS; do
    for conc in $CONCURRENCY; do
        echo "=== concurrency $conc, context $ctx ==="
        if "$BENCH" --concurrency "$conc" --ctx "$ctx" --csv "$CSV" "$@"; then
            :
        else
            echo "cell concurrency=$conc ctx=$ctx failed with $?" >&2
            failed=1
        fi
    done
done
echo
echo "wrote $CSV"
exit "$failed"
