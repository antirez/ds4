#!/bin/sh
# M5 mixed-staging MPP arms: end-to-end logprob drift vs the legacy reference
# (same methodology as m5-tensor-precision-probe.sh; arms w32stage/a32stage).
#
#   reference : legacy simdgroup kernels (DS4_METAL_DISABLE_METAL4=1)
#   accumulate: shipped MPP kernels (default tensor route)
#   f32stage  : MPP with both operand tiles staged fp32
#   w32stage  : MPP, fp32 weight tile / binary16 activation tile
#   a32stage  : MPP, binary16 weight tile / fp32 activation tile
#
# Run only while no bench/GPU job is active (timing skew + lock contention).

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
OUT=/tmp/ds4-mixstage-probe
rm -rf "$OUT"; mkdir -p "$OUT"

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

for pf in "$OUT/p250.txt" "$OUT/p1500.txt"; do
    run_dump reference  "DS4_METAL_DISABLE_METAL4=1" "$pf"
    run_dump accumulate ""                         "$pf"
    run_dump f32stage   "DS4_METAL_MPP_MOE_F32STAGE=1" "$pf"
    run_dump w32stage   "DS4_METAL_MPP_MOE_W32STAGE=1" "$pf"
    run_dump a32stage   "DS4_METAL_MPP_MOE_A32STAGE=1" "$pf"
done

echo
echo "== results (vs legacy reference; max |logit delta| over common top-k, argmax match) =="
python3 - "$OUT" <<'EOF'
import json, sys, os, math
out = sys.argv[1]
def load(p):
    with open(p) as f: return json.load(f)["steps"]
def compare(a_path, b_path):
    a, b = load(a_path), load(b_path)
    deltas = []
    div = 0
    for sa, sb in zip(a, b):
        if sa["selected"]["id"] != sb["selected"]["id"]: div += 1
        ta = {t["token"]["id"]: t["logit"] for t in sa["top_logprobs"]}
        tb = {t["token"]["id"]: t["logit"] for t in sb["top_logprobs"]}
        for k in set(ta) & set(tb):
            deltas.append(ta[k] - tb[k])
    rms = math.sqrt(sum(d*d for d in deltas)/len(deltas)) if deltas else 0.0
    return (max(abs(d) for d in deltas), rms, div, len(a))
for stem in ("p250", "p1500"):
    ref = os.path.join(out, f"reference_{stem}.json")
    row = [stem]
    for label in ("accumulate", "f32stage", "w32stage", "a32stage"):
        p = os.path.join(out, f"{label}_{stem}.json")
        if not os.path.exists(p):
            row.append(f"{label}: MISSING"); continue
        maxd, rms, div, n = compare(ref, p)
        verdict = "MATCH" if (maxd == 0 and div == 0) else ("close" if maxd < 0.01 else "DRIFT")
        row.append(f"{label}: max|d|={maxd:.4g} rms={rms:.4g} argmax_div={div}/{n} [{verdict}]")
    print("  ".join(row))
EOF
echo "Raw dumps and logs: $OUT"
