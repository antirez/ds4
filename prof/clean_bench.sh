#!/bin/bash
set -u
WT=/home/tyler/Projects/AI/temp/wt-prof
MODEL=/home/tyler/Projects/AI/ds4-gb10/gguf/model.gguf
cd "$WT"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
./pulsar-bench -m "$MODEL" --prompt-file speed-bench/promessi_sposi.txt --cuda \
   --ctx-start 2048 --ctx-max 8192 --step-incr 6144 --gen-tokens 0 \
   --csv /home/tyler/Projects/AI/temp/clean_prefill.csv 2>/home/tyler/Projects/AI/temp/clean_bench.log
echo "exit=$?"
cat /home/tyler/Projects/AI/temp/clean_prefill.csv
