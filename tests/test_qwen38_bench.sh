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
# 1. Verify quants.h custom quant ids and enum bound
python3 - "$ROOT/gguf-tools/quants.h" << 'PY'
import re
import sys
with open(sys.argv[1]) as f: txt=f.read()
ids=dict((n,int(v)) for n,v in re.findall(r'DS4Q_TYPE_(\w+)\s*=\s*(\d+)',txt))
cnt=ids.pop("COUNT",None)
assert cnt is not None, "missing DS4Q_TYPE_COUNT"
assert ids.get("Q4_64A")==36, f"Q4_64A expected 36 got {ids.get('Q4_64A')}"
assert ids.get("Q2_64A")==37, f"Q2_64A expected 37 got {ids.get('Q2_64A')}"
for want in (36,37):
    dup=[n for n,v in ids.items() if v==want]
    assert len(dup)==1, f"collision at {want}: {dup}"
assert max(ids.values())==41, f"max type expected 41 got {max(ids.values())}"
assert cnt==max(ids.values())+1, f"COUNT expected {max(ids.values())+1} got {cnt}"
print(f"ok quants.h: Q4_64A=36 Q2_64A=37 COUNT={cnt} no collision")
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
METAL_CSV="/tmp/qwen38_metal_$$.csv"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "skip: Metal section (not macOS)"
else
  echo "testing ds4-bench --metal ..."
  rm -f "$METAL_CSV"
  DS4_BENCH_DISABLE_SNAPSHOT=1 "$ROOT/ds4-bench" --model "$MODEL" --metal --prompt-file "$PROMPT" --ctx-start 32 --ctx-max 64 --step-incr 32 --gen-tokens 4 --csv "$METAL_CSV" > "$LOG" 2>&1 || { cat "$LOG" >&2; fail "bench metal failed"; }
  [[ -f "$METAL_CSV" ]] || fail "metal csv missing"
  head -n1 "$METAL_CSV" | grep -q "prefill_tps" || fail "metal csv missing prefill"
  mlines=$(wc -l < "$METAL_CSV" | tr -d ' ')
  [[ "$mlines" -ge 2 ]] || fail "metal csv short"
  ok "ds4-bench metal smoke ok ($mlines lines)"
fi
python3 << PY
import csv
with open("$CSV") as f:
    for r in csv.DictReader(f):
        print(f"  ctx {r['ctx_tokens']:>5} prefill {float(r['prefill_tps']):6.1f} decode {float(r['gen_tps']):6.1f} steady {float(r.get('gen_steady_tps',0)):6.1f}")
PY
rm -f "$CSV" "$METAL_CSV"
ok "qwen38 harness complete"
