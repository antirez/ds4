#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROMPT_FILE="${PROMPT_FILE:-speed-bench/promessi_sposi.txt}"
CTX_START="${CTX_START:-512}"
CTX_MAX="${CTX_MAX:-65536}"
STEP_MUL="${STEP_MUL:-2}"
GEN_TOKENS="${GEN_TOKENS:-128}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-speed-bench/local-runs/${RUN_ID}-metal-tensor-bench}"
PYTHON="${PYTHON:-python3}"
OPEN_CHART="${OPEN_CHART:-1}"
ALLOW_STALE_BINARY="${ALLOW_STALE_BINARY:-0}"

if [[ "$ALLOW_STALE_BINARY" != "1" ]]; then
  if [[ ! -x ./ds4-bench ]]; then
    echo "error: ./ds4-bench does not exist or is not executable; run make ds4-bench first" >&2
    exit 1
  fi
  stale_source="$(
    {
      printf '%s\n' ds4.c ds4.h ds4_gpu.h ds4_bench.c ds4_metal.m
      find metal -type f -name '*.metal'
    } 2>/dev/null | while IFS= read -r path; do
      if [[ "$path" -nt ./ds4-bench ]]; then
        printf '%s\n' "$path"
        break
      fi
    done
  )"
  if [[ -n "$stale_source" ]]; then
    echo "error: ./ds4-bench is stale; $stale_source is newer" >&2
    echo "       rebuild first, or set ALLOW_STALE_BINARY=1 to summarize old artifacts intentionally" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR"

ARTIFACT_PREFIX="${RUN_ID}_gen${GEN_TOKENS}"
QUALITY_CSV="$OUT_DIR/${ARTIFACT_PREFIX}_ds4_bench_quality.csv"
STANDARD_CSV="$OUT_DIR/${ARTIFACT_PREFIX}_ds4_bench_standard_metal.csv"
TENSOR_CSV="$OUT_DIR/${ARTIFACT_PREFIX}_ds4_bench_tensor_metal.csv"
CHART="$OUT_DIR/${ARTIFACT_PREFIX}_ds4_bench_standard_quality_tensor.png"

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
