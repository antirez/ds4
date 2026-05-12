#!/bin/bash
# usage: ./bench/profile_h200.sh [ncu|nsys] [ctx_start] [ctx_max]
# examples:
#   ./bench/profile_h200.sh ncu 32768
#   ./bench/profile_h200.sh nsys 65536 131072
MODE=${1:-ncu}
CTX_START=${2:-32768}
CTX_MAX=${3:-$CTX_START}
BENCH=${BENCH:-./ds4-bench}
MODEL=${MODEL:-ds4flash.gguf}

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL (set MODEL= env var)" >&2
    exit 1
fi
if [ ! -x "$BENCH" ]; then
    echo "Bench binary not found: $BENCH (run make first)" >&2
    exit 1
fi

BENCH_ARGS="$BENCH -m $MODEL --ctx-start $CTX_START --ctx-max $CTX_MAX --step-incr $CTX_START --gen-tokens 64"

if [ "$MODE" = "ncu" ]; then
    ncu \
      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes.sum,dram__bytes.sum,sm__cycles_elapsed.avg,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum \
      --target-processes all \
      --print-summary per-kernel \
      $BENCH_ARGS
elif [ "$MODE" = "nsys" ]; then
    OUTFILE="nsys_ctx${CTX_START}_$(date +%Y%m%d_%H%M%S)"
    nsys profile \
      --stats=true \
      --trace=cuda,nvtx \
      --output="$OUTFILE" \
      $BENCH_ARGS
    echo "Profile saved: ${OUTFILE}.nsys-rep"
else
    echo "Unknown mode: $MODE. Use ncu or nsys." >&2
    exit 1
fi
