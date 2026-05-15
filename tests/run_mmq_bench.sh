#!/usr/bin/env bash
# tests/run_mmq_bench.sh
#
# Sprint 0 bench harness. Runs ds4-bench N+W times against a fixed prompt and
# context-frontier set, writing each run to its own CSV under --out-dir.
# Companion analysis lives in tests/mmq_bench_stats.py.
#
# The goal is variance < 1% on prefill and gen at the headline configuration
# (ctx=2048, gen-tokens=128). Knobs are limited on purpose; this script is the
# locked baseline.
set -euo pipefail

BENCH_BIN_DEFAULT=./ds4-bench
PROMPT_DEFAULT=tests/long_context_story_prompt.txt
MODEL_DEFAULT=ds4flash.gguf
BACKEND_DEFAULT=--cuda
OUT_ROOT_DEFAULT=tests/bench-results

label=""
runs=5
warmup=1
gen_tokens=128
ctx_start=2048
ctx_max=16384
step_mul=2
step_incr=2048
warmup_ctx=1024  # extra frontier at the start of each process to warm GPU clocks; filtered out of combined.csv
bench_bin="$BENCH_BIN_DEFAULT"
prompt="$PROMPT_DEFAULT"
model="$MODEL_DEFAULT"
backend="$BACKEND_DEFAULT"
out_root="$OUT_ROOT_DEFAULT"
ds4_use_mmq=""
extra_args=()

usage() {
    cat <<EOF
Usage: $0 --label NAME [options] [-- extra-args-passed-to-ds4-bench]

  --label NAME           Output directory name under --out-dir. Required.
  --runs N               Measurement runs (default 5).
  --warmup K             Warm-up runs discarded (default 1).
  --gen-tokens N         Decode tokens per frontier (default 128).
  --ctx-start N          First measured frontier (default 2048).
  --ctx-max N            Last measured frontier (default 16384).
  --step-mul F           Multiplicative step (default 2).
  --step-incr N          Linear step when --step-mul is 1 (default 2048).
  --warmup-ctx N         Per-process warm-up frontier prepended before ctx-start
                         (default 1024; set to 0 to disable). Filtered out of
                         combined.csv. Needed because the Runpod container
                         denies nvidia-smi -lgc and the GPU is at idle clocks
                         between bench invocations.
  --bench-bin PATH       ds4-bench binary (default $BENCH_BIN_DEFAULT).
  --prompt-file PATH     Prompt file (default $PROMPT_DEFAULT).
  --model PATH           GGUF path (default $MODEL_DEFAULT).
  --backend FLAG         Backend flag (default $BACKEND_DEFAULT). Use e.g. "--cuda".
  --use-mmq 0|1          Set DS4_CUDA_USE_MMQ for the runs (default: unset, i.e. opt-out off=default-on).
  --out-dir DIR          Output root (default $OUT_ROOT_DEFAULT).
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)        label="$2"; shift 2 ;;
        --runs)         runs="$2"; shift 2 ;;
        --warmup)       warmup="$2"; shift 2 ;;
        --gen-tokens)   gen_tokens="$2"; shift 2 ;;
        --ctx-start)    ctx_start="$2"; shift 2 ;;
        --ctx-max)      ctx_max="$2"; shift 2 ;;
        --step-mul)     step_mul="$2"; shift 2 ;;
        --step-incr)    step_incr="$2"; shift 2 ;;
        --warmup-ctx)   warmup_ctx="$2"; shift 2 ;;
        --bench-bin)    bench_bin="$2"; shift 2 ;;
        --prompt-file)  prompt="$2"; shift 2 ;;
        --model)        model="$2"; shift 2 ;;
        --backend)      backend="$2"; shift 2 ;;
        --use-mmq)      ds4_use_mmq="$2"; shift 2 ;;
        --out-dir)      out_root="$2"; shift 2 ;;
        --) shift; extra_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$label" ]]; then
    echo "error: --label is required" >&2
    exit 2
fi
if [[ ! -x "$bench_bin" ]]; then
    echo "error: bench binary not executable: $bench_bin" >&2
    exit 2
fi
if [[ ! -f "$prompt" ]]; then
    echo "error: prompt file not found: $prompt" >&2
    exit 2
fi

out_dir="$out_root/$label"
mkdir -p "$out_dir"

# Save manifest so future me knows exactly what produced these CSVs.
{
    echo "label=$label"
    echo "date_utc=$(date -u +%FT%TZ)"
    echo "host=$(hostname)"
    echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "runs=$runs"
    echo "warmup=$warmup"
    echo "gen_tokens=$gen_tokens"
    echo "ctx_start=$ctx_start"
    echo "ctx_max=$ctx_max"
    echo "step_mul=$step_mul"
    echo "step_incr=$step_incr"
    echo "warmup_ctx=$warmup_ctx"
    echo "bench_bin=$bench_bin"
    echo "prompt=$prompt"
    echo "model=$model"
    echo "backend=$backend"
    echo "ds4_use_mmq=${ds4_use_mmq:-<unset>}"
    echo "extra_args=${extra_args[*]:-}"
} > "$out_dir/manifest.txt"

# When warmup_ctx is set and below ctx_start, prepend it so the bench sweeps
# {warmup_ctx, ctx_start, ...}. The warmup row is filtered out of combined.csv
# in the concat step below.
effective_ctx_start="$ctx_start"
if [[ "$warmup_ctx" -gt 0 ]] && [[ "$warmup_ctx" -lt "$ctx_start" ]]; then
    effective_ctx_start="$warmup_ctx"
fi

bench_cmd=(
    "$bench_bin"
    "$backend"
    --model "$model"
    --prompt-file "$prompt"
    --ctx-start "$effective_ctx_start"
    --ctx-max "$ctx_max"
    --step-mul "$step_mul"
    --step-incr "$step_incr"
    --gen-tokens "$gen_tokens"
)
if [[ ${#extra_args[@]} -gt 0 ]]; then
    bench_cmd+=("${extra_args[@]}")
fi

run_once() {
    local idx="$1" kind="$2"
    local csv="$out_dir/${kind}-$(printf '%02d' "$idx").csv"
    local log="$out_dir/${kind}-$(printf '%02d' "$idx").log"
    local t0 t1
    t0=$(date +%s)
    if [[ -n "$ds4_use_mmq" ]]; then
        DS4_CUDA_USE_MMQ="$ds4_use_mmq" "${bench_cmd[@]}" --csv "$csv" >"$log" 2>&1
    else
        "${bench_cmd[@]}" --csv "$csv" >"$log" 2>&1
    fi
    t1=$(date +%s)
    echo "  $kind run $idx -> $csv  ($((t1 - t0))s)"
}

echo "ds4 mmq bench harness"
echo "  label    : $label"
echo "  out_dir  : $out_dir"
echo "  warmup   : $warmup"
echo "  runs     : $runs"
echo "  ctx pts  : start=$ctx_start max=$ctx_max step_mul=$step_mul step_incr=$step_incr (warmup_ctx=$warmup_ctx)"
echo "  gen_toks : $gen_tokens"
echo "  use_mmq  : ${ds4_use_mmq:-<unset>}"
echo "  cmd      : ${bench_cmd[*]}"

for ((i = 1; i <= warmup; i++)); do
    run_once "$i" warmup
done
for ((i = 1; i <= runs; i++)); do
    run_once "$i" run
done

# Concatenate measurement runs into one combined CSV with a run column for
# downstream analysis tools that prefer long-form. The warmup-ctx row (first
# frontier of each process) is dropped so the cold-clock sample doesn't pollute
# the stats.
combined="$out_dir/combined.csv"
filter_ctx=""
if [[ "$effective_ctx_start" != "$ctx_start" ]]; then
    filter_ctx="$effective_ctx_start"
fi
{
    echo "run,ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes"
    for ((i = 1; i <= runs; i++)); do
        csv="$out_dir/run-$(printf '%02d' "$i").csv"
        if [[ -n "$filter_ctx" ]]; then
            tail -n +2 "$csv" | awk -F, -v skip="$filter_ctx" -v run="$i" '$1 != skip { print run","$0 }'
        else
            tail -n +2 "$csv" | sed "s|^|$i,|"
        fi
    done
} > "$combined"

echo
echo "done. combined CSV: $combined"
echo "next: python3 tests/mmq_bench_stats.py aggregate $out_dir"
