#!/bin/bash
# DS4 Pro Phase 3: verify M2 whitelist patches (bit-exact + perf)
# control = patched binary (M2 enabled), candidate = same binary with the
# disable env set (= exact pre-patch behavior). exact_rows must be full match.
cd ~/Work/mlx/ds4 || exit 1
BENCH=./speed-bench/metal_decode_schedule_bench
OUT=speed-bench/m2u_scan
LOG=$OUT/whitelist_verify.log

echo "==== V1: rope-fuse whitelist (decode) | $(date '+%H:%M:%S')" >> "$LOG"
$BENCH -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --prefix-tokens 2048 --ctx 3072 --warmup 16 --tokens 512 \
  --candidate-env DS4_METAL_DISABLE_AFFINE_ROPE_PAIR \
  --include-selection 2>&1 | grep -E "^variant=|^exact_|^metal-decode" >> "$LOG"
echo "### V1 done $(date '+%H:%M:%S')" >> "$LOG"

echo "==== V2: prefill mask-cache whitelist (bit-exact via decode rows) | $(date '+%H:%M:%S')" >> "$LOG"
$BENCH -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --prefix-tokens 8192 --ctx 9216 --warmup 16 --tokens 256 \
  --candidate-env DS4_METAL_DISABLE_ZERO_PREFIX_PREFILL_MASK_CACHE \
  --include-selection 2>&1 | grep -E "^variant=|^exact_|^metal-decode" >> "$LOG"
echo "### V2 done $(date '+%H:%M:%S')" >> "$LOG"

echo "==== V3: mask-cache PREFILL perf (pure prefill, ctx 16384) | $(date '+%H:%M:%S')" >> "$LOG"
echo "### V3a cache ON (no env)" >> "$LOG"
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 16384 --ctx-max 16384 --gen-tokens 0 --csv "$OUT/v3a.csv" 2>&1 | \
  grep -E "prefill_chunk|error" >> "$LOG"
echo "### V3b cache OFF (env set)" >> "$LOG"
DS4_METAL_DISABLE_ZERO_PREFIX_PREFILL_MASK_CACHE=1 ./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 16384 --ctx-max 16384 --gen-tokens 0 --csv "$OUT/v3b.csv" 2>&1 | \
  grep -E "prefill_chunk|error" >> "$LOG"
echo "==== VERIFY COMPLETE $(date) ====" >> "$LOG"
