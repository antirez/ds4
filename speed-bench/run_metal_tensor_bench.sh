#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROMPT_FILE="${PROMPT_FILE:-speed-bench/promessi_sposi.txt}"
CTX_START="${CTX_START:-512}"
CTX_MAX="${CTX_MAX:-65536}"
STEP_MUL="${STEP_MUL:-2}"
GEN_TOKENS="${GEN_TOKENS:-128}"
OUT_DIR="${OUT_DIR:-/tmp/ds4-bench-runs}"
PYTHON="${PYTHON:-python3}"
OPEN_CHART="${OPEN_CHART:-1}"

mkdir -p "$OUT_DIR"

QUALITY_CSV="$OUT_DIR/ds4_bench_quality_${GEN_TOKENS}.csv"
STANDARD_CSV="$OUT_DIR/ds4_bench_standard_metal_${GEN_TOKENS}.csv"
TENSOR_CSV="$OUT_DIR/ds4_bench_tensor_metal_${GEN_TOKENS}.csv"
CHART="$OUT_DIR/ds4_bench_standard_quality_tensor_${GEN_TOKENS}.png"

COMMON_ARGS=(
  --prompt-file "$PROMPT_FILE"
  --ctx-start "$CTX_START"
  --ctx-max "$CTX_MAX"
  --step-mul "$STEP_MUL"
  --gen-tokens "$GEN_TOKENS"
)

echo "1/3 Quality Metal -> $QUALITY_CSV"
./ds4-bench --quality "${COMMON_ARGS[@]}" --csv "$QUALITY_CSV"

echo "2/3 Standard Metal -> $STANDARD_CSV"
./ds4-bench -mt off "${COMMON_ARGS[@]}" --csv "$STANDARD_CSV"

echo "3/3 Tensor Metal -> $TENSOR_CSV"
./ds4-bench -mt auto "${COMMON_ARGS[@]}" --csv "$TENSOR_CSV"

echo "Comparing runs -> $CHART"
"$PYTHON" speed-bench/compare_bench.py \
  "$STANDARD_CSV" \
  "$QUALITY_CSV" \
  "$TENSOR_CSV" \
  --labels "Standard Metal" "Quality Metal" "Tensor Metal" \
  --title "ds4-bench: Standard vs Quality vs Tensor (${GEN_TOKENS} generated tokens)" \
  -o "$CHART"

echo
echo "Wrote:"
echo "  $QUALITY_CSV"
echo "  $STANDARD_CSV"
echo "  $TENSOR_CSV"
echo "  $CHART"

if [[ "$OPEN_CHART" != "0" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "$CHART"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$CHART" >/dev/null 2>&1 &
  else
    echo "No opener found; set OPEN_CHART=0 to skip this step."
  fi
fi
