#!/bin/bash
# Prefill composition profiling on the post-widening dev build (77d579d).
# nsys kernel trace + PULSAR_MOE_TIME per-format split, at prefill-chunk 2048 and 8192.
set -u
WT=/home/tyler/Projects/AI/temp/wt-prof
OUT=/home/tyler/Projects/AI/temp
MODEL=/home/tyler/Projects/AI/ds4-gb10/gguf/model.gguf
PROMPT=$WT/speed-bench/promessi_sposi.txt
cd "$WT" || exit 1

run_depth () {
  D=$1
  echo "===== depth $D : $(date +%T) ====="
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  free -g | head -2
  PULSAR_MOE_TIME=1 nsys profile --trace=cuda --sample=none \
      -o "$OUT/prof_p${D}" --force-overwrite true \
      ./pulsar-bench -m "$MODEL" --prompt-file "$PROMPT" --cuda \
      --ctx-start $D --ctx-max $D --gen-tokens 0 --prefill-chunk $D \
      --csv "$OUT/p${D}.csv" > "$OUT/moetime_p${D}.log" 2>&1
  echo "exit=$? depth $D done $(date +%T)"
  tail -12 "$OUT/moetime_p${D}.log"
}

run_depth 2048
run_depth 8192
echo "ALL DONE $(date +%T)"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
