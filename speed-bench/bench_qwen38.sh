#!/usr/bin/env bash
# bench_qwen38.sh - Report tok/s for qwen38.gguf (prefill and decode) on CPU and Metal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${QWEN38_MODEL:-$ROOT/qwen38.gguf}"
PROMPT_FILE="${QWEN38_PROMPT:-$ROOT/speed-bench/promessi_sposi.txt}"
OUT_DIR="${QWEN38_OUT_DIR:-/tmp/qwen38-bench-$$}"
QUICK=1
GEN_TOKENS=32
usage() {
  cat <<EOF
Usage: $0 [--quick] [--full] [--model FILE] [--prompt FILE] [--out DIR] [--gen N]
  --quick     Small sweep: --ctx-start 512 --ctx-max 2048 --step-incr 512 --gen 32 (default)
  --full      Larger sweep: --ctx-start 2048 --ctx-max 16384 --step-incr 2048 --gen 128
  --model     GGUF path (default: $MODEL, env QWEN38_MODEL)
  --prompt    Prompt file (default: $PROMPT_FILE, env QWEN38_PROMPT)
  --out       Output dir for CSVs (default: temp)
  --gen       Tokens per frontier (default: $GEN_TOKENS)
Reports prefill tok/s and decode tok/s (steady-state) for both CPU and Metal.
EOF
}
CTX_START=512
CTX_MAX=2048
STEP_INCR=512
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --quick) QUICK=1; CTX_START=512; CTX_MAX=2048; STEP_INCR=512; GEN_TOKENS=32; shift ;;
    --full) QUICK=0; CTX_START=2048; CTX_MAX=16384; STEP_INCR=2048; GEN_TOKENS=128; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompt) PROMPT_FILE="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --gen) GEN_TOKENS="$2"; shift 2 ;;
    --ctx-start) CTX_START="$2"; shift 2 ;;
    --ctx-max) CTX_MAX="$2"; shift 2 ;;
    --step-incr) STEP_INCR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done
if [[ ! -f "$MODEL" ]]; then
  echo "bench_qwen38: model not found: $MODEL" >&2; exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "bench_qwen38: prompt not found: $PROMPT_FILE" >&2; exit 1
fi
if [[ ! -x "$ROOT/ds4-bench" ]]; then
  echo "bench_qwen38: building ds4-bench..." >&2
  make -C "$ROOT" ds4-bench -j4 >/dev/null 2>&1 || make -C "$ROOT" ds4-bench >/dev/null
fi
mkdir -p "$OUT_DIR"
CPU_CSV="$OUT_DIR/qwen38_cpu.csv"
METAL_CSV="$OUT_DIR/qwen38_metal.csv"
run_one() {
  local backend="$1" csv="$2"
  local extra=()
  if [[ "$backend" == "cpu" ]]; then extra=(--cpu); else extra=(--metal); fi
  echo "bench_qwen38: running $backend: --ctx-start $CTX_START --ctx-max $CTX_MAX --gen $GEN_TOKENS" >&2
  set +e
  DS4_BENCH_DISABLE_SNAPSHOT=1 "$ROOT/ds4-bench" --model "$MODEL" "${extra[@]}" \
    --prompt-file "$PROMPT_FILE" \
    --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --step-incr "$STEP_INCR" \
    --gen-tokens "$GEN_TOKENS" --csv "$csv" 2> "$OUT_DIR/${backend}.stderr.log"
  local rc=$?; set -e
  if [[ $rc -ne 0 ]]; then
    echo "bench_qwen38: $backend failed rc=$rc see $OUT_DIR/${backend}.stderr.log" >&2
    cat "$OUT_DIR/${backend}.stderr.log" >&2 || true
    return $rc
  fi
  echo "bench_qwen38: $backend csv: $csv" >&2
}
run_one cpu "$CPU_CSV"
if uname -s | grep -q Darwin; then
  run_one metal "$METAL_CSV" || { echo "bench_qwen38: metal failed, continuing cpu only" >&2; METAL_CSV=""; }
else
  echo "bench_qwen38: non-Darwin skip Metal" >&2; METAL_CSV=""
fi
python3 - "$CPU_CSV" "$METAL_CSV" << 'PYEOF'
import csv, sys, json, os
def summarize(p):
    if not p or not os.path.exists(p): return None
    rows=[]
    with open(p) as f:
        for r in csv.DictReader(f): rows.append(r)
    if not rows: return None
    def avg(k):
        v=[float(x[k]) for x in rows if x.get(k) not in (None,"")]
        return sum(v)/len(v) if v else 0
    return {"path":p,"frontiers":len(rows),"prefill":avg("prefill_tps"),"gen":avg("gen_tps"),"steady":avg("gen_steady_tps") if "gen_steady_tps" in rows[0] else 0,"rows":rows}
cpu=summarize(sys.argv[1]) if len(sys.argv)>1 else None
metal=summarize(sys.argv[2]) if len(sys.argv)>2 else None
def fmt(v): return f"{v:.2f}" if v else "n/a"
print("")
print("="*78)
print("Qwen3.8 27B (qwen38.gguf) — tok/s summary")
print("-"*78)
print(f"{'backend':<12} {'frontiers':<10} {'prefill_tps':<14} {'decode_tps':<14} {'steady_tps':<14}")
print("-"*78)
for lbl, d in [("CPU",cpu),("Metal",metal)]:
    if d is None: print(f"{lbl:<12} {'--':<10} {'n/a':<14} {'n/a':<14} {'n/a':<14}  (skipped)")
    else: print(f"{lbl:<12} {d['frontiers']:<10} {fmt(d['prefill']):<14} {fmt(d['gen']):<14} {fmt(d['steady']):<14}")
print("-"*78)
if cpu: print(f"CPU  CSV: {cpu['path']}")
if metal:
    print(f"Metal CSV: {metal['path']}")
    try:
        with open(metal["path"].replace("metal.csv","metal.stderr.log")) as f:
    except: pass
print("="*78)
out_dir=os.path.dirname(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1] else "/tmp"
try:
    j=os.path.join(out_dir,"summary.json")
    with open(j,"w") as o: json.dump({"cpu":cpu,"metal":metal},o,indent=2)
    print(f"JSON: {j}")
except: pass
PYEOF
echo ""
echo "CSVs: $OUT_DIR"
ls -lh "$OUT_DIR"/*.csv 2>/dev/null || true
