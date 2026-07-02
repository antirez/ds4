#!/usr/bin/env bash
set -euo pipefail

# Manual hardware validation for a local multi-GPU ROCm Pro Q4 setup. This
# script self-skips when the private shard files or requested GPU count are not
# present; it is not intended as a portable CI test.

skip() {
    echo "SKIP rocm pro q4 smoke: $*"
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
PROMPT=${DS4_SMOKE_PROMPT:-tell me about yourself}

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

OUT=${DS4_SMOKE_LOGPROBS:-$(mktemp -t ds4-rocm-pro-q4-logprobs.XXXXXX.json)}
LOG=${DS4_SMOKE_LOG:-$(mktemp -t ds4-rocm-pro-q4.XXXXXX.log)}

if ! "$DS4_BIN" --rocm \
    -m "$MODEL0" \
    -m "$MODEL1" \
    --gpus "$GPUS" \
    --ctx "$CTX" \
    --prefill-chunk "$PREFILL_CHUNK" \
    --dist-prefill-chunk "$DIST_PREFILL_CHUNK" \
    --dist-prefill-window "$DIST_PREFILL_WINDOW" \
    --nothink \
    --temp 0 \
    --tokens 1 \
    --dump-logprobs "$OUT" \
    -p "$PROMPT" >"$LOG" 2>&1; then
    echo "FAIL rocm pro q4 smoke run failed; log: $LOG" >&2
    tail -80 "$LOG" >&2 || true
    exit 1
fi

python3 - "$OUT" "$LOG" "$REQ_GPUS" <<'PY'
import json
import math
import sys
from pathlib import Path

out_path, log_path, req_gpus_s = sys.argv[1:4]
req_gpus = int(req_gpus_s)
log = Path(log_path).read_text(encoding="utf-8", errors="replace")
bad = [
    "Kernel Name:",
    "HSA_STATUS_ERROR_EXCEPTION",
    "unspecified launch failure",
    "prompt processing failed",
    "decode failed",
    "Aborted",
    "unable to connect to 127.0.0.1",
]
for needle in bad:
    if needle in log:
        print(f"FAIL smoke saw {needle!r}", file=sys.stderr)
        print(f"log: {log_path}", file=sys.stderr)
        sys.exit(1)

if req_gpus > 1 and "local GPU worker: coordinator disconnected; exiting" not in log:
    print("FAIL smoke did not observe local worker shutdown", file=sys.stderr)
    print(f"log: {log_path}", file=sys.stderr)
    sys.exit(1)

with open(out_path, "r", encoding="utf-8") as f:
    data = json.load(f)
root = data[0] if isinstance(data, list) else data
steps = root.get("steps")
if not isinstance(steps, list) or not steps:
    print("FAIL smoke logprobs has no steps", file=sys.stderr)
    sys.exit(1)
step = steps[0]
selected = step.get("selected", {})
top = step.get("top_logprobs", [])
if not isinstance(selected.get("id"), int):
    print("FAIL smoke selected token id missing", file=sys.stderr)
    sys.exit(1)
if not isinstance(top, list) or not top:
    print("FAIL smoke top_logprobs missing", file=sys.stderr)
    sys.exit(1)
for item in top[:5]:
    for key in ("logit", "logprob"):
        value = item.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            print(f"FAIL smoke non-finite {key}", file=sys.stderr)
            sys.exit(1)

print(f"PASS first token id={selected['id']} top_k={len(top)}")
PY

echo "logprobs: $OUT"
echo "log: $LOG"
