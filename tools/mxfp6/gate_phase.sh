#!/bin/bash
# One GPU lock-hold phase: run prefill-only frontier-logit dumps for each
# MODEL:OUTDIR pair given as arguments. Invoked as:
#   flock -w SECS /home/tyler/Projects/AI/temp/gpu.lock bash gate_phase.sh LOGFILE M1:D1 [M2:D2 ...]
# Before every model load: drop_caches + require 95 GiB MemAvailable
# (model is 91 GB; we never touch processes we did not start).
set -u
WT=/home/tyler/Projects/AI/temp/wt-mxfp6
BENCH="$WT/pulsar-bench"
PROMPT="$WT/speed-bench/promessi_sposi.txt"
LOG="$1"; shift

say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

wait_for_ram() {
    for _ in $(seq 60); do
        sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
        avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
        [ "$avail_kb" -gt $((95 * 1024 * 1024)) ] && return 0
        say "RAM gate: only $((avail_kb / 1024 / 1024)) GiB available, waiting 60s"
        sleep 60
    done
    return 1
}

say "GPU lock acquired: $# dump run(s)"
for pair in "$@"; do
    model="${pair%%:*}"; outdir="${pair#*:}"
    mkdir -p "$outdir"
    if ! wait_for_ram; then say "ABORT $outdir: RAM never freed"; exit 1; fi
    say "dump: $model -> $outdir"
    "$BENCH" -m "$model" --cuda --prompt-file "$PROMPT" \
        --ctx-start 64 --ctx-max 2048 --gen-tokens 0 \
        --dump-frontier-logits-dir "$outdir" \
        --csv "$outdir/bench.csv" >> "$LOG" 2>&1
    rc=$?
    say "dump rc=$rc: $outdir"
    [ $rc -ne 0 ] && exit $rc
done
exit 0
