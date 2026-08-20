#!/usr/bin/env bash
# test_qwen38_bench.sh — correctness + bench harness for qwen38.gguf
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${DS4_TEST_MODEL:-$ROOT/qwen38.gguf}"
PROMPT="$ROOT/speed-bench/promessi_sposi.txt"
CSV="/tmp/qwen38_test_$$.csv"
LOG="/tmp/qwen38_test_$$.log"
fail(){ echo "FAIL: $*" >&2; exit 1; }
ok(){ echo "ok: $*"; }
# 1. Verify quants.h DS4Q_TYPE_Q4_64A=36 COUNT 43 no collision
python3 << 'PY'
import re
with open("gguf-tools/quants.h") as f: txt=f.read()
import re
m=re.search(r'DS4Q_TYPE_Q4_64A\s*=\s*(\d+)',txt)
n=re.search(r'DS4Q_TYPE_COUNT\s*=\s*(\d+)',txt)
assert m and n, "missing Q4_64A or COUNT"
q4=int(m.group(1)); cnt=int(n.group(1))
assert q4==36, f"Q4_64A expected 36 got {q4}"
assert cnt==43, f"COUNT expected 43 got {cnt}"
others=re.findall(r'DS4Q_TYPE_\w+\s*=\s*36',txt)
assert len(others)==1, f"collision at 36: {others}"
types=[int(x) for x in re.findall(r'DS4Q_TYPE_\w+\s*=\s*(\d+)',txt) if int(x)<43]
assert max(types)==41, f"max 41 got {max(types)}"
assert cnt>max(types)+1
print("ok quants.h: Q4_64A=36 COUNT=43 no collision (max 41 spare 42)")
PY
ok "quants.h verified"
if [[ ! -f "$MODEL" ]]; then
  echo "skip: qwen38 model not found at $MODEL" >&2
  if [[ -x "$ROOT/ds4-bench" ]]; then
    "$ROOT/ds4-bench" --help | grep -q "model" || fail "help missing"
    ok "ds4-bench help ok"
  fi
  exit 0
fi
if [[ ! -x "$ROOT/ds4-bench" ]]; then make -C "$ROOT" ds4-bench -j4 >/dev/null 2>&1 || make -C "$ROOT" ds4-bench >/dev/null; fi
if [[ ! -x "$ROOT/ds4" ]]; then make -C "$ROOT" ds4 -j4 >/dev/null 2>&1 || make -C "$ROOT" ds4 >/dev/null; fi
python3 << PY
import struct
with open("$MODEL","rb") as f:
    assert f.read(4)==b'GGUF', "not GGUF"
    ver=struct.unpack('<I',f.read(4))[0]
    nt=struct.unpack('<Q',f.read(8))[0]
    print(f"ok GGUF header ver={ver} tensors={nt}")
    data=f.read(4096)
    if b'qwen' in data.lower(): print("ok arch qwen")
PY
echo "testing ds4 --inspect -m $MODEL ..."
"$ROOT/ds4" -m "$MODEL" --inspect > "$LOG" 2>&1 || { cat "$LOG" >&2; fail "inspect failed"; }
ok "ds4 --inspect ok"
echo "testing ds4-bench --model qwen38.gguf --cpu ..."
rm -f "$CSV"
DS4_BENCH_DISABLE_SNAPSHOT=1 "$ROOT/ds4-bench" --model "$MODEL" --cpu --prompt-file "$PROMPT" --ctx-start 32 --ctx-max 64 --step-incr 32 --gen-tokens 4 --csv "$CSV" > "$LOG" 2>&1 || { cat "$LOG" >&2; fail "bench cpu failed"; }
[[ -f "$CSV" ]] || fail "csv missing"
head -n1 "$CSV" | grep -q "prefill_tps" || fail "csv missing prefill"
lines=$(wc -l < "$CSV" | tr -d ' ')
[[ "$lines" -ge 2 ]] || fail "csv short"
ok "ds4-bench cpu smoke ok ($lines lines)"
echo "testing metal (fallback) ..."
METAL_CSV="/tmp/qwen38_metal_$$.csv"; rm -f "$METAL_CSV"
if DS4_BENCH_DISABLE_SNAPSHOT=1 "$ROOT/ds4-bench" --model "$MODEL" --metal --prompt-file "$PROMPT" --ctx-start 32 --ctx-max 64 --step-incr 32 --gen-tokens 4 --csv "$METAL_CSV" > "$LOG" 2>&1; then
  ok "metal ok (fallback)"
  grep -q "Qwen family uses CPU backend" "$LOG" && echo "  (Metal->CPU fallback confirmed)" || true
else echo "warn: metal failed but cpu passed — ok non-Metal" >&2; fi
python3 << PY
import csv
with open("$CSV") as f:
    for r in csv.DictReader(f):
        print(f"  ctx {r['ctx_tokens']:>5} prefill {float(r['prefill_tps']):6.1f} decode {float(r['gen_tps']):6.1f} steady {float(r.get('gen_steady_tps',0)):6.1f}")
PY
rm -f "$CSV" "$METAL_CSV"
ok "qwen38 harness complete"
