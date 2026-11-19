#!/bin/bash
# Long-context needle retrieval gate (skill stability gate).
# Greedy, thinking off; both codes must appear exactly in the answer.
set -e
cd /Users/aparuael/workspace/ds4-kolibri-1
M=gguf/Kolibri-1-Q8.gguf
python3 opt/make_needles.py

echo "=== needle @ 2K (code near 90%) ==="
timeout 600 ./ds4 -m $M --ctx 4096 --temp 0 --nothink --prompt-file opt/runs/needle-2k.txt 2>/dev/null \
  | tee /tmp/needle-2k-out.txt | tail -2
grep -q "ZQ-7741-KATZE" /tmp/needle-2k-out.txt && echo "2K NEEDLE: PASS" || echo "2K NEEDLE: FAIL"

echo "=== needles @ ~28K (50% and 95%) ==="
timeout 900 ./ds4 -m $M --ctx 32768 --temp 0 --nothink --prompt-file opt/runs/needle-30k.txt 2>/dev/null \
  | tee /tmp/needle-30k-out.txt | tail -2
grep -q "ZQ-7741-KATZE" /tmp/needle-30k-out.txt && echo "30K needle1: PASS" || echo "30K needle1: FAIL"
grep -q "LT-9082-VOGEL" /tmp/needle-30k-out.txt && echo "30K needle2: PASS" || echo "30K needle2: FAIL"
