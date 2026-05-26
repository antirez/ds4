#!/usr/bin/env bash
# test_distributed_correctness.sh — Verify distributed ds4 produces identical
# output to single-node.
#
# Phase 1 (automated): Run single-node CPU baseline and capture output.
# Phase 2 (manual):    Run the same prompt distributed, then compare.
#
# Requires: ds4 binary built with JACCL=1 (make JACCL=1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DS4_ROOT="$(dirname "$SCRIPT_DIR")"
DS4_BIN="${DS4_ROOT}/ds4"
MODEL="${DS4_ROOT}/ds4flash.gguf"

PROMPT="The capital of France is"
CTX=4096
N_PREDICT=32

BASELINE_FILE="/tmp/ds4_correctness_baseline.txt"
DISTRIBUTED_FILE="/tmp/ds4_correctness_distributed.txt"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo "=== ds4 distributed correctness test ==="
echo "Binary:  $DS4_BIN"
echo "Model:   $MODEL"
echo "Prompt:  \"$PROMPT\""
echo "Context: $CTX"
echo "Predict: $N_PREDICT tokens"
echo ""

# --- Check prerequisites ---
if [[ ! -x "$DS4_BIN" ]]; then
    echo -e "${RED}Error: ds4 binary not found at $DS4_BIN${NC}"
    echo "Build with: make JACCL=1"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo -e "${RED}Error: model not found at $MODEL${NC}"
    echo "Symlink or copy your GGUF model to $MODEL"
    exit 1
fi

# ========================================
# Phase 1: Single-node CPU baseline
# ========================================
echo "--- Phase 1: Single-node CPU baseline ---"
echo "Running: $DS4_BIN --cpu -c $CTX -n $N_PREDICT -p \"$PROMPT\""
echo ""

"$DS4_BIN" --cpu -c "$CTX" -n "$N_PREDICT" -p "$PROMPT" 2>/dev/null \
    | tee "$BASELINE_FILE"

echo ""
echo -e "${GREEN}Baseline captured to $BASELINE_FILE${NC}"
echo ""

# ========================================
# Phase 2: Distributed (manual instructions)
# ========================================
echo "--- Phase 2: Distributed run (manual) ---"
echo ""
echo "To run the same prompt distributed across 2 nodes, execute on each node:"
echo ""
echo "  Node 0 (coordinator):"
echo "    export JACCL_RANK=0"
echo "    export JACCL_WORLD_SIZE=2"
echo "    export JACCL_COORDINATOR=<rank0_lan_ip>"
echo "    export JACCL_IBV_DEVICES='[[null, \"rdma_enX\"], [\"rdma_enY\", null]]'"
echo "    $DS4_BIN --distributed --cpu -c $CTX -n $N_PREDICT -p \"$PROMPT\" > $DISTRIBUTED_FILE 2>/dev/null"
echo ""
echo "  Node 1:"
echo "    export JACCL_RANK=1"
echo "    export JACCL_WORLD_SIZE=2"
echo "    export JACCL_COORDINATOR=<rank0_lan_ip>"
echo "    export JACCL_IBV_DEVICES='[[null, \"rdma_enX\"], [\"rdma_enY\", null]]'"
echo "    $DS4_BIN --distributed --cpu -c $CTX -n $N_PREDICT -p \"$PROMPT\" > /dev/null 2>/dev/null"
echo ""
echo "  Or use the launch script:"
echo "    ./distributed_launch.sh --nodes hub,m3u4 --model $MODEL --extra \"--cpu -n $N_PREDICT -p '$PROMPT'\""
echo ""
echo "Then run this script with --compare to check results:"
echo "    $0 --compare"
echo ""

# ========================================
# Phase 3: Compare outputs
# ========================================
if [[ "${1:-}" == "--compare" ]]; then
    echo "--- Phase 3: Comparing outputs ---"

    if [[ ! -f "$BASELINE_FILE" ]]; then
        echo -e "${RED}Error: baseline file not found at $BASELINE_FILE${NC}"
        echo "Run this script without --compare first to generate the baseline."
        exit 1
    fi

    if [[ ! -f "$DISTRIBUTED_FILE" ]]; then
        echo -e "${RED}Error: distributed output not found at $DISTRIBUTED_FILE${NC}"
        echo "Run the distributed command from Phase 2, saving rank 0's output to $DISTRIBUTED_FILE"
        exit 1
    fi

    echo "Baseline output:"
    cat "$BASELINE_FILE"
    echo ""
    echo "Distributed output:"
    cat "$DISTRIBUTED_FILE"
    echo ""

    if diff -q "$BASELINE_FILE" "$DISTRIBUTED_FILE" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS: Outputs are identical.${NC}"
        exit 0
    else
        echo -e "${RED}FAIL: Outputs differ.${NC}"
        echo ""
        echo "Diff:"
        diff --color=auto "$BASELINE_FILE" "$DISTRIBUTED_FILE" || true
        exit 1
    fi
fi
