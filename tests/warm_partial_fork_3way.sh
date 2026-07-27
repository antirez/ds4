#!/bin/bash
# plan-33 inc D gate — THE server-fork VALUE + integration proof: PARTIAL-prefix
# fork routing on the DOMINANT shape (a LARGE shared text prefix + divergent user
# tails), the work-skipped teeth, and the TTFT payoff number.
#
# CONSTRUCTION: the partial fork only fires, and only matters, on a LARGE shared
# prefix. Appended detokenized GENERATION diverges at the trunk/gen junction, so
# the shared region must be literal TEXT, identical across branches, no generated
# text inside it. Then P tokenizes stably and common ~= |P| > warm_partial_min.
#   P            : ~1900-token shared system+few-shot TEXT block (literal, verbatim).
#   R1 = P+tail0 : establishes the trunk (bank A), commits -> frontier > |P|.
#   R2 = P+tailA : common ~= |P|, 256 <= common < frontier -> PARTIAL fork A->B.
#   R3 = P+tailB : A still intact -> SECOND partial fork A->C (double-fork case).
#
# CORRECTNESS ORACLE — self-consistency, NOT fork-vs-cold text (MEASURED FINDING
# 2026-07-20): a partial fork resumes at a 128-aligned R and re-prefills an
# UNALIGNED suffix, which carries the accepted warm-continuation last-ulp KV delta
# (fork-gate P5: "KV byte-identical: no — delta expected on unaligned"). The reused
# prefix [0,R) IS byte-identical (fork-gate P6), and the ANSWER is correct, but
# under greedy argmax that ulp delta can flip a generated token — sometimes LATE
# (branch "two primes": 189 identical chars then drift) and sometimes the VERY
# FIRST token when the top-2 logits are near-tied (branch "European river": both
# name the Danube but open with a different reasoning sentence). So fork output is
# NOT bit-identical to a COLD prefill and asserting that would flag a correct fork
# as a failure. The NUMERICAL correctness (reused prefix byte-identity, suffix
# coherence) is the ENGINE fork gate's job (bank_fork_gate P2/P5/P6). This SERVER
# gate proves VALUE + INTEGRATION via a drift-IMMUNE invariant: the SAME forked
# request is DETERMINISTIC and SIBLING-INDEPENDENT — forking a trunk twice (with an
# intervening sibling fork) yields byte-identical output to forking it once.
#
# ASSERTS:
#  (i)   WORK ACTUALLY SKIPPED: >=2 warm-fork-partial fired, each `resume R>0` (never
#        `resume 0` = the value bug), trunk preserved.
#  (ii)  SELF-CONSISTENCY: double-fork branch output == single-fork branch output
#        (byte-identical) -> the fork is deterministic and a sibling fork does not
#        corrupt the shared trunk. Plus branches diverged (A output != B output).
#  (iii) TTFT payoff: warm partial fork vs cold (max_tokens=1 wall time) >= 2x.
# Run manually under GPU discipline (flock, drop_caches, watchdog, one load).
#   usage: tests/warm_partial_fork_3way.sh [MODEL] [PORT]
set -u
MODEL=${1:-gguf/model.gguf}; PORT=${2:-8902}
DIR=$(mktemp -d /tmp/warmpartial.XXXX); LOG=$DIR/server.log
P=$(python3 -c "
hdr='You are a precise assistant. Follow the format exactly.\n'
shot='Example %d: given input, respond with a single factual sentence and stop.\n'
print(hdr + ''.join(shot%i for i in range(1,120)))")
T0="$P
Task-0: name a color."
TA="$P
Question-Alpha: name two prime numbers."
TB="$P
Question-Bravo: name a European river."
TT="$P
Probe-Gamma: name a mountain."     # dedicated TTFT tail (fresh partial fork)
jreq(){ python3 -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'max_tokens':int(sys.argv[2]),'temperature':0.0,'stream':False}))" "$1" "$2"; }
# 4 banks: trunk + 2 branches fill 3, leaving a FREE bank so the TTFT probe forks.
start(){ DS4_WARM_FORK=$1 DS4_MSEQ_BANKS=4 DS4_WARM_PARTIAL_MIN=256 ./pulsar-server -m "$MODEL" \
        --host 127.0.0.1 --port $PORT -c 32768 --kv-disk-dir "" >"$LOG" 2>&1 & SP=$!
  for i in $(seq 1 300); do [ "$(curl -s -m2 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && return 0; sleep 1; done; return 1; }
stop(){ kill -INT $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; }
gen(){ curl -s -m 600 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
        -d "$(jreq "$1" "$2")" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['text'])"; }
ttft(){ curl -s -m 600 -o /dev/null -w '%{time_total}' http://127.0.0.1:$PORT/v1/completions \
        -H 'content-type: application/json' -d "$(jreq "$1" 1)"; }

# (F) fork run: trunk, branch A, branch B (DOUBLE fork off intact trunk), TTFT probe.
echo "[Dgate] (F) warm-partial-fork (branching + double fork)"; start 1 || { echo FAIL-start; exit 1; }
gen "$T0" 24 >/dev/null                                    # R1: build trunk A
FA=$(gen "$TA" 48)                                          # R2: partial fork A->B
FB=$(gen "$TB" 48)                                          # R3: A intact -> partial fork A->C
T_WARM=$(ttft "$TT")                                        # TTFT: warm partial fork off A (free bank)
PF=$(grep -c "warm-fork-partial:" "$LOG")
PRES=$(grep -c "trunk preserved" "$LOG")
BADRESUME=$(grep -c "resume 0)" "$LOG")                    # value-bug signature
GOODRESUME=$(grep -cE "warm-fork-partial: .*resume [1-9][0-9]*\)" "$LOG")
stop

# (S) single-fork run: fork branch B off the trunk with NO intervening sibling.
echo "[Dgate] (S) single fork (self-consistency control)"; start 1 || { echo FAIL-start; exit 1; }
gen "$T0" 24 >/dev/null; FBS=$(gen "$TB" 48); stop

# (Ccold) cold TTFT baseline: fresh server, probe is the first (cold-prefill) request.
echo "[Dgate] (Ccold) cold TTFT baseline"; start 0 || { echo FAIL-start; exit 1; }; T_COLD=$(ttft "$TT"); stop

fail=0
echo "partial forks fired: $PF (trunk-preserved: $PRES, resume>0: $GOODRESUME, resume0-BUG: $BADRESUME)"
# (i) WORK SKIPPED teeth.
[ "$PF"        -ge 2 ] || { echo "EXPECTED >=2 partial forks, got $PF"; fail=1; }
[ "$PRES"      -ge 1 ] || { echo "trunk never reported preserved"; fail=1; }
[ "$BADRESUME" -eq 0 ] || { echo "WORK-SKIPPED BUG: a fork logged 'resume 0)' (re-prefills the whole prompt)"; fail=1; }
[ "$GOODRESUME" -ge 1 ] || { echo "no partial fork logged resume>0 (work not actually skipped)"; fail=1; }
# (ii) self-consistency: double-fork == single-fork (drift-immune correctness).
[ -n "$FA" ] && [ -n "$FB" ] || { echo "a fork produced EMPTY output"; fail=1; }
[ "$FA" != "$FB" ] || { echo "branches did NOT diverge (both forks returned identical output)"; fail=1; }
if [ "$FB" = "$FBS" ]; then echo "self-consistency: double-fork == single-fork (byte-identical) OK"
else echo "SELF-CONSISTENCY FAIL: double-fork branch output != single-fork (sibling fork corrupted the trunk)"; fail=1; fi
# (iii) TTFT payoff.
SPEEDUP=$(python3 -c "w=$T_WARM;c=$T_COLD;print('%.3f'%(c/w) if w>0 else 999)")
echo "TTFT payoff: warm-fork ${T_WARM}s vs cold ${T_COLD}s -> ${SPEEDUP}x faster"
python3 -c "import sys; sys.exit(0 if $T_COLD/$T_WARM >= 2.0 else 1)" || {
    echo "TTFT payoff below 2x ($SPEEDUP x) — fork is not materially skipping prefill"; fail=1; }
echo "WARM-PARTIAL-FORK GATE: $([ $fail = 0 ] && echo PASS || echo FAIL)"; exit $fail
