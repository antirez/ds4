#!/bin/bash
# DS4 Pro Phase 2: prefill chunk size sweep for M2 Ultra
# baseline default is 4096 (ds4-bench log: prefill_chunk=4096)
# dual-die M2 Ultra may favor different chunk: test 2048..16384
cd ~/Work/mlx/ds4 || exit 1
OUT=speed-bench/m2u_scan
mkdir -p "$OUT"
LOG=$OUT/prefill_scan.log

for chunk in 2048 3072 4096 6144 8192 12288 16384; do
  echo "" >> "$LOG"
  echo "### chunk=$chunk | $(date '+%H:%M:%S')" >> "$LOG"
  ./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128 \
    --prefill-chunk "$chunk" --csv "$OUT/prefill_chunk_${chunk}.csv" 2>&1 | \
    grep -E "prefill_chunk|error|die" >> "$LOG"
  echo "### done chunk=$chunk rc=$? | $(date '+%H:%M:%S')" >> "$LOG"
done

echo "==== PREFILL SCAN COMPLETE $(date) ====" >> "$LOG"
