#!/bin/sh
# M5 TensorOps accumulate-precision probe (experiment branch only).
#
# Runs the same greedy logprob dump three ways on an M5-class GPU:
#   reference : legacy simdgroup kernels (DS4_METAL_DISABLE_METAL4=1)
#   accumulate: shipped MPP kernels, mode::multiply_accumulate chain
#   muladd    : MPP kernels with mode::multiply + explicit fp32 adds
#               (DS4_METAL_MPP_MOE_MULADD=1)
#
# If muladd matches reference and accumulate does not, the M5 drift lives in
# the TensorOps multiply_accumulate path and the explicit-add schedule is a
# candidate kernel-side fix.  If muladd still drifts, the per-tile product
# itself is lossy and the automatic tensor route must stay withheld.

LOCK=${DS4_LOCK_FILE:-/tmp/ds4.lock}
wait_lock() {
    i=0
    while [ $i -lt 90 ]; do
        if /usr/bin/python3 -c "import fcntl,sys; fcntl.flock(open('$LOCK','a'), fcntl.LOCK_EX|fcntl.LOCK_NB)" 2>/dev/null; then
            return 0
        fi
        [ $i -eq 0 ] && echo "waiting for ds4 instance lock ($LOCK)..."
        sleep 10; i=$((i+1))
    done
    return 1
}
MODEL=${1:-gguf/GLM-5.3-Flash-Q2.gguf}
PROMPT=tests/test-vectors/glm-openrouter/prompts/long_code_audit.txt
OUT=/tmp/ds4-mpp-probe
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== building ds4 (incremental) =="
make ds4 >/dev/null

echo "== preparing prompts (58 and 309 tokens) =="
head -c 250  "$PROMPT" > "$OUT/p250.txt"
head -c 1500 "$PROMPT" > "$OUT/p1500.txt"

run_dump() { # label envflag promptfile
    label=$1; envflag=$2; pf=$3
    wait_lock || { echo "ds4 lock stayed busy for 15 min; aborting"; exit 1; }
    # shellcheck disable=SC2086
    env $envflag ./ds4 -m "$MODEL" --metal --nothink -sys "" --temp 0 \
        -n 2 --ctx 32768 --prompt-file "$pf" \
        --dump-logprobs "$OUT/${label}_$(basename "$pf" .txt).json" \
        --logprobs-top-k 20 > "$OUT/${label}_$(basename "$pf" .txt).log" 2>&1 \
        || { echo "run $label failed:"; tail -3 "$OUT/${label}_$(basename "$pf" .txt).log"; exit 1; }
}

echo "== GPU check =="
grep -m1 "Metal device" "$OUT"/*.log 2>/dev/null || true
run_dump probe "" "$OUT/p250.txt"
DEV=$(grep -m1 "Metal device" "$OUT/probe_p250.log" | sed 's/.*Metal device //')
echo "device: $DEV"
case "$DEV" in
    *M5*|*M6*|*A19*|*A20*) ;;
    *) echo "WARNING: not an M5-class device; the tensor route will not engage and all rows will coincide." ;;
esac
if ! grep -q "tensor_matmul=on" "$OUT/probe_p250.log"; then
    echo "WARNING: tensor route did not engage (tensor_matmul=off in the log); results are not meaningful."
fi

for pf in "$OUT/p250.txt" "$OUT/p1500.txt"; do
    run_dump reference  "DS4_METAL_DISABLE_METAL4=1" "$pf"
    run_dump accumulate ""                         "$pf"
    run_dump muladd     "DS4_METAL_MPP_MOE_MULADD=1" "$pf"
    run_dump k16        "DS4_METAL_MPP_MOE_K16=1"    "$pf"
done

# fast-math lowering check on the shipped accumulate route (env only)
run_dump mathsafe "DS4_METAL_MATH_SAFE=1" "$OUT/p250.txt"

echo
echo "== results (vs reference; max |logit delta| over common top-k, argmax match) =="
python3 - "$OUT" <<'EOF'
import json, sys, glob, os
out = sys.argv[1]
def load(p):
    with open(p) as f: return json.load(f)["steps"]
def compare(a_path, b_path):
    a, b = load(a_path), load(b_path)
    maxd, div = 0.0, 0
    for sa, sb in zip(a, b):
        if sa["selected"]["id"] != sb["selected"]["id"]: div += 1
        ta = {t["token"]["id"]: t["logit"] for t in sa["top_logprobs"]}
        tb = {t["token"]["id"]: t["logit"] for t in sb["top_logprobs"]}
        for k in set(ta) & set(tb):
            maxd = max(maxd, abs(ta[k] - tb[k]))
    return maxd, div, len(a)
for stem in ("p250", "p1500"):
    ref = os.path.join(out, f"reference_{stem}.json")
    row = [stem]
    for label in ("accumulate", "muladd", "k16"):
        p = os.path.join(out, f"{label}_{stem}.json")
        if not os.path.exists(p):
            row.append(f"{label}: MISSING"); continue
        maxd, div, n = compare(ref, p)
        verdict = "MATCH" if (maxd == 0 and div == 0) else ("close" if maxd < 0.01 else "DRIFT")
        row.append(f"{label}: max|d|={maxd:.6g} argmax_div={div}/{n} [{verdict}]")
    print("  ".join(row))

ref = os.path.join(out, "reference_p250.json")
p = os.path.join(out, "mathsafe_p250.json")
if os.path.exists(ref) and os.path.exists(p):
    maxd, div, n = compare(ref, p)
    verdict = "MATCH" if (maxd == 0 and div == 0) else ("close" if maxd < 0.01 else "DRIFT")
    print(f"p250  mathsafe: max|d|={maxd:.6g} argmax_div={div}/{n} [{verdict}]")
EOF
echo
echo "Verdict guide:"
echo "  muladd MATCH + accumulate DRIFT -> cross-tile accumulate is the loss; explicit-add schedule is a viable kernel fix."
echo "  k16 ~ 2x muladd drift           -> per-op-run truncation; larger K tiles reduce it but parity needs huge K."
echo "  k16 ~ muladd drift              -> per-multiply/per-add internal precision; not fixable from MSL."
echo "  mathsafe MATCH                  -> shader fast-math lowering was the loss."
echo "  all DRIFT                       -> keep the tensor route withheld."
echo "Raw dumps and logs: $OUT"
