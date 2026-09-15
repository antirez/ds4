#!/bin/bash
# DS4 Pro Phase 1: decode schedule A/B grid for M2 Ultra
# Phase A: short-context window (prefix 2048, q2 2/32 active)
# Phase B: long-context region (prefix 8192, fallback 4/0 active) - second-split sweep
cd ~/Work/mlx/ds4 || exit 1
BENCH=./speed-bench/metal_decode_schedule_bench
OUT=speed-bench/m2u_scan
mkdir -p "$OUT"
LOG=$OUT/decode_scan.log

run_ab() { # name cf cs vf vs prefix
  local name=$1 cf=$2 cs=$3 vf=$4 vs=$5 prefix=$6
  local ctx=$(( prefix + 1024 ))
  echo "" >> "$LOG"
  echo "### $name | control=$cf/$cs candidate=$vf/$vs prefix=$prefix | $(date '+%H:%M:%S')" >> "$LOG"
  $BENCH -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
    --prefix-tokens "$prefix" --ctx "$ctx" --warmup 16 --tokens 512 \
    --control-first "$cf" --control-second "$cs" \
    --candidate-first "$vf" --candidate-second "$vs" \
    --include-selection 2>&1 | grep -E "^variant=|^exact_|^metal-decode" >> "$LOG"
  echo "### done $name rc=$?" >> "$LOG"
}

echo "==== PHASE A: prefix 2048 (q2 window, control 2/32) ====" >> "$LOG"
run_ab A1 2 32 3 32 2048
run_ab A2 2 32 2 28 2048
run_ab A3 2 32 2 36 2048
run_ab A4 2 32 2 24 2048

echo "==== PHASE B: prefix 8192 (fallback 4/0, second-split sweep) ====" >> "$LOG"
run_ab B1 4 0 4 8  8192
run_ab B2 4 0 4 12 8192
run_ab B3 4 0 4 16 8192
run_ab B4 4 0 4 20 8192
run_ab B5 4 0 4 24 8192
run_ab B6 4 0 4 32 8192

echo "==== SCAN PHASE 1 COMPLETE $(date) ====" >> "$LOG"
