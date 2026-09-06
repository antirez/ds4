#!/bin/bash
# balanced_ab.sh - paired A/B harness for ds4-bench with the arm order alternated.
#
# WHY THIS EXISTS
#
# Running arm A for a while, then arm B for a while, and dividing the two
# medians gives a biased ratio whenever throughput drifts inside a session.
# On a GB10 we measured that drift directly: across a 24-value sweep, every
# single value was lower than the one before it (24/24, p = 2^-24 under the
# null of no trend). With a fixed arm order the arm measured last is
# systematically penalised, which moved the A/B ratio by 0.38-0.55 percentage
# points - the same order of magnitude as the effects being measured.
#
# Three properties fix it, and this script enforces all three:
#
#   1. The two arms are ADJACENT IN TIME. No third arm, no other work between
#      them, so both see nearly the same machine state.
#   2. The ORDER ALTERNATES between rounds: round 1 runs A,B and round 2 runs
#      B,A. Any monotone drift contributes with opposite sign in the two rounds
#      and cancels in the mean.
#   3. Ratios are formed WITHIN a round, never across rounds, and clock and
#      temperature are recorded before and after every run so a thermal
#      explanation can be checked rather than assumed.
#
# CAVEAT WORTH KNOWING BEFORE YOU COMPARE ANYTHING
#
# ds4-bench prefills only the STEP INCREMENT at each point of a sweep, reusing
# the KV cache built by the previous point. prefill_tokens equals ctx_tokens
# only at the first point:
#
#     --ctx-start 2048 --ctx-max 8192 --step-incr 2048
#         ctx 2048 -> prefill 2048     ctx 4096 -> prefill 2048
#         ctx 6144 -> prefill 2048     ctx 8192 -> prefill 2048
#
#     --ctx-start 8192 --ctx-max 65536 --step-mul 2
#         ctx 8192 -> prefill 8192     ctx 16384 -> prefill 8192
#         ctx 32768 -> prefill 16384   ctx 65536 -> prefill 32768
#
# So "prefill_tps at ctx 8192" is not one quantity. It is a cold 8192-token
# prefill in the first sweep, and a 2048-token increment onto a warm cache in
# the second. We measured the difference: Q4 leads Q8 by about 0.4% on the cold
# full prefill and by about 2.3% on the warm increment, at the same ctx.
#
# Only compare runs with the same --ctx-start and the same step. The summary
# CSV records prefill_tokens next to prefill_tps so this stays checkable.
#
# USAGE
#
#   ./balanced_ab.sh --bench ./ds4-bench \
#                    --a  /path/model-A.gguf --a-label Q4 \
#                    --b  /path/model-B.gguf --b-label Q8 \
#                    --prompt /path/prompt.txt \
#                    --blocks 5 --out results
#
# Optional:
#   --ctx-start N --ctx-max N --step-incr N   context sweep (default 2048..8192 by 2048)
#   --gen-tokens N                            passed through to ds4-bench (default 128)
#   --gap-seconds N                           idle time between blocks (default 300)
#   --before-cmd 'CMD'                        run before each measurement block
#   --after-cmd  'CMD'                        run after each measurement block
#
# --before-cmd / --after-cmd exist because a benchmark should not share the GPU
# with anything else. If the machine also serves a model, stop it in
# --before-cmd and start it again in --after-cmd; the script then only takes the
# GPU for the length of one block instead of the whole session. Both commands
# are run through the shell; a non-zero exit from --before-cmd skips the block.
#
# OUTPUT
#
# One tidy CSV row per context point per run, in $OUT/paired.csv:
#
#   block,round,position,arm,ctx,prefill_tokens,prefill_tps,gen_steady_tps,
#   sm_mhz_before,temp_c_before,sm_mhz_after,temp_c_after,status
#
# Pair on (block, round, ctx): the two rows sharing those three values are the
# paired A/B sample. Raw ds4-bench CSVs and stdout are kept alongside it.

set -u

BENCH="" A_MODEL="" B_MODEL="" A_LABEL="A" B_LABEL="B" PROMPT=""
CTX_START=2048 CTX_MAX=8192 STEP_INCR=2048 GEN_TOKENS=128
BLOCKS=5 GAP=300 OUT="balanced_ab_out" BEFORE_CMD="" AFTER_CMD=""

die() { echo "balanced_ab: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --bench)       BENCH=$2; shift 2 ;;
    --a)           A_MODEL=$2; shift 2 ;;
    --b)           B_MODEL=$2; shift 2 ;;
    --a-label)     A_LABEL=$2; shift 2 ;;
    --b-label)     B_LABEL=$2; shift 2 ;;
    --prompt)      PROMPT=$2; shift 2 ;;
    --ctx-start)   CTX_START=$2; shift 2 ;;
    --ctx-max)     CTX_MAX=$2; shift 2 ;;
    --step-incr)   STEP_INCR=$2; shift 2 ;;
    --gen-tokens)  GEN_TOKENS=$2; shift 2 ;;
    --blocks)      BLOCKS=$2; shift 2 ;;
    --gap-seconds) GAP=$2; shift 2 ;;
    --out)         OUT=$2; shift 2 ;;
    --before-cmd)  BEFORE_CMD=$2; shift 2 ;;
    --after-cmd)   AFTER_CMD=$2; shift 2 ;;
    -h|--help)     sed -n '2,60p' "$0"; exit 0 ;;
    *)             die "unknown option: $1" ;;
  esac
done

[ -n "$BENCH" ]    || die "--bench is required"
[ -x "$BENCH" ]    || die "not executable: $BENCH"
[ -n "$A_MODEL" ]  || die "--a is required"
[ -n "$B_MODEL" ]  || die "--b is required"
[ -s "$A_MODEL" ]  || die "missing or empty: $A_MODEL"
[ -s "$B_MODEL" ]  || die "missing or empty: $B_MODEL"
[ -n "$PROMPT" ]   || die "--prompt is required"
[ -s "$PROMPT" ]   || die "missing or empty: $PROMPT"
[ "$A_MODEL" != "$B_MODEL" ] || die "--a and --b are the same file"

mkdir -p "$OUT" || die "cannot create $OUT"
SUMMARY="$OUT/paired.csv"
LOG="$OUT/run.log"

if [ ! -s "$SUMMARY" ]; then
  echo "block,round,position,arm,ctx,prefill_tokens,prefill_tps,gen_steady_tps,sm_mhz_before,temp_c_before,sm_mhz_after,temp_c_after,status" > "$SUMMARY"
fi

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Identify exactly what was measured. A benchmark result without the checksum
# of the binary that produced it cannot be reproduced or contested later.
sum_of() { (md5sum "$1" 2>/dev/null || md5 -q "$1" 2>/dev/null) | awk '{print $1}'; }

log "balanced_ab starting"
log "  bench    $BENCH  md5 $(sum_of "$BENCH")"
log "  arm $A_LABEL  $A_MODEL"
log "  arm $B_LABEL  $B_MODEL"
log "  prompt   $PROMPT  md5 $(sum_of "$PROMPT")  $(wc -c < "$PROMPT") bytes"
log "  sweep    ctx $CTX_START..$CTX_MAX step $STEP_INCR, gen-tokens $GEN_TOKENS"
log "  blocks   $BLOCKS, gap ${GAP}s"

gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
      | head -1 | tr -d ' '
  else
    echo ","
  fi
}

run_arm() {
  local blk=$1 rnd=$2 pos=$3 arm=$4 model=$5
  local csv="$OUT/b${blk}-r${rnd}-p${pos}-${arm}.csv"
  local g1 sm1 t1 g2 sm2 t2 rc
  g1=$(gpu); sm1=${g1%,*}; t1=${g1#*,}
  "$BENCH" --prompt-file "$PROMPT" \
           --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --step-incr "$STEP_INCR" \
           --gen-tokens "$GEN_TOKENS" -m "$model" --csv "$csv" \
           > "$OUT/b${blk}-r${rnd}-p${pos}-${arm}.out" 2>&1
  rc=$?
  g2=$(gpu); sm2=${g2%,*}; t2=${g2#*,}
  log "  block $blk round $rnd pos $pos $arm rc=$rc  sm ${sm1}->${sm2} MHz  temp ${t1}->${t2} C"
  if [ $rc -eq 0 ] && [ -s "$csv" ]; then
    tail -n +2 "$csv" | while IFS=, read -r ctx ptok ptps gtok gtps gfirst gsteady gstps kv; do
      echo "$blk,$rnd,$pos,$arm,$ctx,$ptok,$ptps,$gstps,$sm1,$t1,$sm2,$t2,ok" >> "$SUMMARY"
    done
  else
    echo "$blk,$rnd,$pos,$arm,,,,,$sm1,$t1,$sm2,$t2,failed_rc$rc" >> "$SUMMARY"
  fi
}

cleanup() {
  if [ -n "$AFTER_CMD" ]; then
    log "cleanup: running --after-cmd"
    sh -c "$AFTER_CMD" || log "  --after-cmd exited non-zero"
  fi
}
trap cleanup EXIT
trap 'log "signal received, exiting"; exit 1' INT TERM HUP

blk=0
while [ "$blk" -lt "$BLOCKS" ]; do
  blk=$((blk + 1))
  log "---- block $blk of $BLOCKS ----"

  if [ -n "$BEFORE_CMD" ]; then
    if ! sh -c "$BEFORE_CMD"; then
      log "  --before-cmd failed, skipping this block"
      continue
    fi
  fi

  # Round 1: A then B.  Round 2: B then A.  Any monotone drift within the
  # block enters the two rounds with opposite sign and cancels in the mean.
  run_arm "$blk" 1 1 "$A_LABEL" "$A_MODEL"
  run_arm "$blk" 1 2 "$B_LABEL" "$B_MODEL"
  run_arm "$blk" 2 1 "$B_LABEL" "$B_MODEL"
  run_arm "$blk" 2 2 "$A_LABEL" "$A_MODEL"

  if [ -n "$AFTER_CMD" ]; then
    sh -c "$AFTER_CMD" || log "  --after-cmd exited non-zero"
  fi

  if [ "$blk" -lt "$BLOCKS" ] && [ "$GAP" -gt 0 ]; then
    log "  gap ${GAP}s"
    sleep "$GAP"
  fi
done

log "done: $blk blocks, $(( $(wc -l < "$SUMMARY") - 1 )) rows in $SUMMARY"
