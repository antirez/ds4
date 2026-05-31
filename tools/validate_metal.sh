#!/bin/bash
# Validate ds4 Metal shaders offline using xcrun metal.
# Requires: Metal Toolchain (xcodebuild -downloadComponent MetalToolchain)
# Usage: tools/validate_metal.sh

set -euo pipefail

if ! xcrun --sdk macosx metal --version &>/dev/null; then
    echo "Error: Metal compiler not found. Install with:"
    echo "  xcodebuild -downloadComponent MetalToolchain"
    exit 1
fi

cd "$(dirname "$0")/.."
OUT=/tmp/ds4_shader_check.air

# Base header matching ds4_gpu_source in ds4_metal.m
cat > /tmp/ds4_metal_base.h << 'EOF'
#include <metal_stdlib>
using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

enum ds4_sort_order {
    DS4_SORT_ORDER_ASC,
    DS4_SORT_ORDER_DESC,
};

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};
EOF

echo "// === base header ===" > /tmp/ds4_full.metal
cat /tmp/ds4_metal_base.h >> /tmp/ds4_full.metal

for f in \
    metal/flash_attn.metal \
    metal/dense.metal \
    metal/moe.metal \
    metal/dsv4_hc.metal \
    metal/unary.metal \
    metal/dsv4_kv.metal \
    metal/dsv4_rope.metal \
    metal/dsv4_misc.metal \
    metal/argsort.metal \
    metal/cpy.metal \
    metal/concat.metal \
    metal/get_rows.metal \
    metal/sum_rows.metal \
    metal/softmax.metal \
    metal/repeat.metal \
    metal/glu.metal \
    metal/norm.metal \
    metal/bin.metal \
    metal/set_rows.metal; do
    echo "" >> /tmp/ds4_full.metal
    echo "// === $f ===" >> /tmp/ds4_full.metal
    cat "$f" >> /tmp/ds4_full.metal
done

LINES=$(wc -l < /tmp/ds4_full.metal)
echo "Compiling $LINES lines of Metal shader source..."

WARNINGS=0
if xcrun -sdk macosx metal -std=metal3.1 -c /tmp/ds4_full.metal -o "$OUT" 2>&1 | tee /tmp/ds4_metal_warnings.txt; then
    WARNINGS=$(grep -c "warning:" /tmp/ds4_metal_warnings.txt || true)
    echo ""
    echo "SUCCESS: All Metal shaders compiled ($WARNINGS warnings)"
    ls -la "$OUT"
    rm -f /tmp/ds4_metal_base.h /tmp/ds4_full.metal /tmp/ds4_metal_warnings.txt
else
    echo ""
    echo "FAILED: See errors above"
    exit 1
fi
