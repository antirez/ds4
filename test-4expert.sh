#!/bin/bash
# Test 4Expert PR end-to-end on a fresh Linux machine.
#
# Usage:
#   bash test-4expert.sh                      # all defaults
#   bash test-4expert.sh /path/to/weights 16   # custom weight dir and threads
#
# This script:
#   1. Clones and builds ds4 (4expert branch)
#   2. Downloads 4Expert safetensors (if not already present)
#   3. Generates GGUF template from safetensors metadata
#   4. Quantizes to Q4_K GGUF (~153 GiB output)
#   5. Links GGUF and runs a test inference
set -euo pipefail

WEIGHTS_DIR="${1:-DeepSeek-V4-Flash-4Expert}"
THREADS="${2:-$(nproc)}"
OUT_GGUF="ds4flash-4expert.gguf"
TEMPLATE="template.gguf"
BRANCH="4expert"

echo "============================================"
echo " 4Expert PR End-to-End Test"
echo "============================================"
echo " weights dir : $WEIGHTS_DIR"
echo " threads     : $THREADS"
echo " output      : $OUT_GGUF"
echo ""

# ── Step 1: Clone and build ──
if [ ! -f ds4.c ]; then
    echo "==> [1/5] Cloning ds4 ($BRANCH branch) ..."
    git clone https://github.com/yuhai-china/ds4
    cd ds4
    git checkout "$BRANCH"
else
    echo "==> [1/5] Already in ds4 repo, building ..."
fi

echo "     Building gguf-tools ..."
make -C gguf-tools -j"$THREADS"
echo "     Building ds4 (CPU) ..."
make cpu -j"$THREADS"
echo ""

# ── Step 2: Download weights if needed ──
if [ ! -f "$WEIGHTS_DIR/model.safetensors.index.json" ]; then
    echo "==> [2/5] Downloading 4Expert safetensors ..."
    pip install -q huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('cloudyu/DeepSeek-V4-Flash-4Expert', local_dir='$WEIGHTS_DIR')
"
else
    echo "==> [2/5] Weights found at $WEIGHTS_DIR"
fi
echo ""

# ── Step 3: Generate template ──
if [ ! -f "$TEMPLATE" ]; then
    echo "==> [3/5] Generating GGUF template ..."
    python3 gguf-tools/gen_gguf_template.py --hf "$WEIGHTS_DIR" --out "$TEMPLATE"
else
    echo "==> [3/5] Template already exists: $TEMPLATE"
fi
echo ""

# ── Step 4: Quantize ──
echo "==> [4/5] Quantizing ($THREADS threads, ~153 GiB output) ..."
./gguf-tools/deepseek4-quantize \
    --hf "$WEIGHTS_DIR" \
    --template "$TEMPLATE" \
    --out "$OUT_GGUF" \
    --experts q4_k \
    --attention-proj q8_0 \
    --attention f16 \
    --shared q8_0 \
    --output q8_0 \
    --embedding f16 \
    --dense f16 \
    --threads "$THREADS"
echo ""

# ── Step 5: Test inference ──
echo "==> [5/5] Running test inference ..."
ln -sfn "$OUT_GGUF" ds4flash.gguf
./ds4 -p "The weather is great today" -n 100

echo ""
echo "============================================"
echo " Test complete. If you see output above,"
echo " 4Expert support is working."
echo "============================================"
