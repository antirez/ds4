#!/usr/bin/env bash
set -euo pipefail

# Manual regression smoke for the Pro Q4 local multi-GPU path. It compares the
# new MFMA route with the fallback route for top-token stability; the synthetic
# rocm-q8-mfma-correctness test is the numerical correctness test.

skip() {
    echo "SKIP rocm pro q4 logits smoke: $*"
    exit 0
}

DS4_BIN=${DS4_BIN:-./ds4}
MODEL0=${DS4_PRO_Q4_MODEL0:-gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf}
MODEL1=${DS4_PRO_Q4_MODEL1:-gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf}
GPUS=${DS4_GPUS:-0,1,2,3,4,5,6,7}
CTX=${DS4_CTX:-262144}
PREFILL_CHUNK=${DS4_PREFILL_CHUNK:-4096}
DIST_PREFILL_CHUNK=${DS4_DIST_PREFILL_CHUNK:-4096}
DIST_PREFILL_WINDOW=${DS4_DIST_PREFILL_WINDOW:-8}

[[ -x "$DS4_BIN" ]] || skip "$DS4_BIN is not executable"
[[ -f "$MODEL0" ]] || skip "$MODEL0 is missing"
[[ -f "$MODEL1" ]] || skip "$MODEL1 is missing"
if [[ ! -e /dev/kfd && ! -e /dev/dri ]]; then
    skip "ROCm device nodes are not available"
fi

REQ_GPUS=$(python3 - "$GPUS" <<'PY'
import sys
print(len([p for p in sys.argv[1].split(",") if p.strip()]))
PY
)
if command -v rocminfo >/dev/null 2>&1; then
    HAVE_GPUS=$(rocminfo 2>/dev/null | grep -E 'Name:[[:space:]]+gfx' | wc -l | tr -d ' ')
    if [[ "$HAVE_GPUS" != "0" && "$HAVE_GPUS" -lt "$REQ_GPUS" ]]; then
        skip "requested $REQ_GPUS GPUs but rocminfo reports $HAVE_GPUS"
    fi
fi

OUT_A=${DS4_LOGITS_COMPARE_A:-$(mktemp -t ds4-rocm-pro-q4-mfma.XXXXXX.json)}
OUT_B=${DS4_LOGITS_COMPARE_B:-$(mktemp -t ds4-rocm-pro-q4-fallback.XXXXXX.json)}
LOG_A=${DS4_LOGITS_COMPARE_LOG_A:-$(mktemp -t ds4-rocm-pro-q4-mfma.XXXXXX.log)}
LOG_B=${DS4_LOGITS_COMPARE_LOG_B:-$(mktemp -t ds4-rocm-pro-q4-fallback.XXXXXX.log)}
PROMPT_FILE=${DS4_LOGITS_COMPARE_PROMPT_FILE:-$(mktemp -t ds4-rocm-pro-q4-prompt.XXXXXX.txt)}

if [[ -z "${DS4_LOGITS_COMPARE_PROMPT_FILE:-}" ]]; then
    python3 - "$PROMPT_FILE" <<'PY'
from pathlib import Path
text = (
    "Summarize the design tradeoffs in local distributed inference. "
    "Focus on memory placement, activation transport, and deterministic testing. "
)
Path(__import__("sys").argv[1]).write_text(text * 96, encoding="utf-8")
PY
fi

run_case() {
    local label=$1
    local out=$2
    local log=$3
    shift 3
    if ! env "$@" "$DS4_BIN" --rocm \
        -m "$MODEL0" \
        -m "$MODEL1" \
        --gpus "$GPUS" \
        --ctx "$CTX" \
        --prefill-chunk "$PREFILL_CHUNK" \
        --dist-prefill-chunk "$DIST_PREFILL_CHUNK" \
        --dist-prefill-window "$DIST_PREFILL_WINDOW" \
        --nothink \
        --temp 0 \
        --dump-logits "$out" \
        --prompt-file "$PROMPT_FILE" >"$log" 2>&1; then
        echo "FAIL $label logits smoke run failed; log: $log" >&2
        tail -80 "$log" >&2 || true
        exit 1
    fi
}

run_case "mfma" "$OUT_A" "$LOG_A"
run_case "fallback" "$OUT_B" "$LOG_B" DS4_ROCM_DISABLE_Q8_BATCH_MFMA=1

python3 - "$OUT_A" "$OUT_B" "$LOG_A" "$LOG_B" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

out_a, out_b, log_a, log_b = sys.argv[1:5]

bad_needles = [
    "Kernel Name:",
    "HSA_STATUS_ERROR_EXCEPTION",
    "unspecified launch failure",
    "prompt processing failed",
    "Aborted",
]
for log_path in (log_a, log_b):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    for needle in bad_needles:
        if needle in text:
            print(f"FAIL logits smoke saw {needle!r} in {log_path}", file=sys.stderr)
            sys.exit(1)

def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    logits = data.get("logits")
    if not isinstance(logits, list) or not logits:
        raise SystemExit(f"FAIL {path} has no logits array")
    vals = []
    for i, v in enumerate(logits):
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise SystemExit(f"FAIL {path} has non-finite logit at {i}")
        vals.append(float(v))
    arg = data.get("argmax_token", {}).get("id")
    if not isinstance(arg, int):
        raise SystemExit(f"FAIL {path} has no argmax token id")
    return vals, arg

a, arg_a = load(out_a)
b, arg_b = load(out_b)
if len(a) != len(b):
    print(f"FAIL vocab mismatch {len(a)} != {len(b)}", file=sys.stderr)
    sys.exit(1)

diffs = [abs(x - y) for x, y in zip(a, b)]
max_abs = max(diffs)
rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
top_a = sorted(range(len(a)), key=a.__getitem__, reverse=True)[:5]
top_b = sorted(range(len(b)), key=b.__getitem__, reverse=True)[:5]
overlap = len(set(top_a) & set(top_b))

max_abs_limit = float(os.getenv("DS4_LOGITS_COMPARE_MAX_ABS", "5.0"))
rms_limit = float(os.getenv("DS4_LOGITS_COMPARE_RMS", "0.75"))
min_top5_overlap = int(os.getenv("DS4_LOGITS_COMPARE_MIN_TOP5", "3"))

if arg_a != arg_b or overlap < min_top5_overlap or max_abs > max_abs_limit or rms > rms_limit:
    print(
        "FAIL logits smoke "
        f"argmax={arg_a}/{arg_b} top5_overlap={overlap}/5 "
        f"max_abs={max_abs:.6g} rms={rms:.6g}",
        file=sys.stderr,
    )
    print(f"mfma logits: {out_a}", file=sys.stderr)
    print(f"fallback logits: {out_b}", file=sys.stderr)
    print(f"mfma log: {log_a}", file=sys.stderr)
    print(f"fallback log: {log_b}", file=sys.stderr)
    sys.exit(1)

print(
    "PASS logits smoke "
    f"argmax={arg_a} top5_overlap={overlap}/5 "
    f"max_abs={max_abs:.6g} rms={rms:.6g}"
)
PY

echo "mfma logits: $OUT_A"
echo "fallback logits: $OUT_B"
echo "mfma log: $LOG_A"
echo "fallback log: $LOG_B"
