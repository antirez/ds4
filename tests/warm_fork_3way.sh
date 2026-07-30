#!/bin/bash
# plan-33 inc B gate: warm FULL-PREFIX FORK routing, BRANCHING (fan-out) shape.
#
# Design (settled): warm-fork is a BRANCHING / trunk-preservation optimization.
# It fires on a request whose token prefix equals a warm bank's FULL committed
# frontier (a full-prefix match — NOT a partial/sibling match, which is inc C).
# The value shows only under BRANCHING: two divergent continuations of ONE trunk.
# In-place continuation would let the first branch CONSUME the trunk, forcing the
# second to cold-prefill; forking preserves the trunk so BOTH branches reuse it.
#
# Shape:
#   R1  : prompt = TRUNK              -> bank A, frontier = tok(TRUNK)+gen1  (G1)
#   R2  : prompt = TRUNK + G1 + " A?" -> FULL-prefix match on A -> fork A->B
#   R3  : prompt = TRUNK + G1 + " B?" -> A still intact -> FULL match -> fork A->C
# Two forks fire; A is never consumed. We assert the R2/R3 OUTPUT TEXT is
# identical across three routings (per the gate-A finding, KV bytes are only
# comparable at chunk-aligned cuts; full-match forks continue from the trunk's
# exact frontier, so OUTPUT tokens are the oracle):
#   (iii) warm-fork  PULSAR_WARM_FORK=1  -> forks fire, trunk preserved
#   (ii)  warm cont. PULSAR_WARM_FORK=0  -> in-place: R2 consumes A, R3 rewinds A
#   (i)   cold       PULSAR_WARM_FORK=0, fresh server per branch (no reuse at all)
# plus: >=2 forks fired (routing alive) and >=1 "trunk preserved" line.
# Run manually under GPU discipline (flock, drop_caches, watchdog, one load).
#   usage: tests/warm_fork_3way.sh [MODEL] [PORT]
set -u
MODEL=${1:-gguf/model.gguf}; PORT=${2:-8901}
DIR=$(mktemp -d /tmp/warmfork.XXXX); LOG=$DIR/server.log
TRUNK=$(python3 -c "print('Shared preamble sentence. '*380)")   # ~2.7k-token trunk
jreq(){ python3 -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'max_tokens':int(sys.argv[2]),'temperature':0.0,'stream':False}))" "$1" "$2"; }
start(){ PULSAR_WARM_FORK=$1 PULSAR_MSEQ_BANKS=3 ./pulsar-server -m "$MODEL" --host 127.0.0.1 --port $PORT \
        -c 32768 --kv-disk-dir "" >"$LOG" 2>&1 & SP=$!
  for i in $(seq 1 240); do [ "$(curl -s -m2 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && return 0; sleep 1; done; return 1; }
stop(){ kill -INT $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; }
gen(){ curl -s -m 600 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
        -d "$(jreq "$1" "$2")" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['text'])"; }

# (iii) warm-fork: establish trunk then two branches from its FULL frontier.
echo "[3way] (iii) warm-fork"; start 1 || { echo FAIL-start; exit 1; }
G1=$(gen "$TRUNK" 12)                                  # trunk generation, extends A
Fa=$(gen "$TRUNK$G1 Question A?" 40)                   # branch A: full-match on A -> fork
Fb=$(gen "$TRUNK$G1 Question B?" 40)                   # branch B: A intact -> fork
FORKS=$(grep -c "warm-fork: trunk" "$LOG"); PRES=$(grep -c "trunk preserved" "$LOG")
stop

# (ii) warm continuation, fork disabled: same three requests, in-place routing.
echo "[3way] (ii) warm continuation (fork off)"; start 0 || { echo FAIL-start; exit 1; }
G1b=$(gen "$TRUNK" 12); Wa=$(gen "$TRUNK$G1b Question A?" 40); Wb=$(gen "$TRUNK$G1b Question B?" 40)
stop
[ "$G1" = "$G1b" ] || echo "WARN: trunk generation differs across runs (greedy nondeterminism?)"

# (i) cold: fresh server for each branch (no bank reuse whatsoever).
echo "[3way] (i) cold per branch"
start 0 || { echo FAIL-start; exit 1; }; Ca=$(gen "$TRUNK$G1 Question A?" 40); stop
start 0 || { echo FAIL-start; exit 1; }; Cb=$(gen "$TRUNK$G1 Question B?" 40); stop

fail=0
[ "$Fa" = "$Wa" ] || { echo "MISMATCH branch A: fork vs warm"; fail=1; }
[ "$Fa" = "$Ca" ] || { echo "MISMATCH branch A: fork vs cold"; fail=1; }
[ "$Fb" = "$Wb" ] || { echo "MISMATCH branch B: fork vs warm"; fail=1; }
[ "$Fb" = "$Cb" ] || { echo "MISMATCH branch B: fork vs cold"; fail=1; }
echo "forks fired: $FORKS (trunk-preserved lines: $PRES)"
[ "$FORKS" -ge 2 ] || { echo "EXPECTED >=2 forks (both branches), got $FORKS"; fail=1; }
[ "$PRES"  -ge 1 ] || { echo "trunk never reported preserved"; fail=1; }
echo "WARM-FORK 3WAY (branching): $([ $fail = 0 ] && echo PASS || echo FAIL)"; exit $fail
