#!/usr/bin/env bash
# scripts/tp_ab.sh — two-node NHI tensor-parallel run for the persistent
# gate POC A/B (DS4_TP_PERSIST_GATE).  Leader (rank 0) runs ds4-bench on
# max2 (10.99.0.2), worker (rank 1) runs the ds4 CLI on max.
#
# Usage: scripts/tp_ab.sh [persist 0|1] [ctx N] [gen-tokens N] [tag]
#   persist=0 baseline (legacy launch path), persist=1 persistent kernel.
#
# Outputs: both node logs, the bench CSV line, and the two NHI stats lines.

set -u

PERSIST="${1:-0}"
CTX="${2:-197}"
GEN="${3:-512}"
TAG="${4:-}"
if [ -z "$TAG" ]; then TAG="persist${PERSIST}-c${CTX}"; fi

LEADER_SSH_HOST="${TP_LEADER_SSH:-max2}"
WORKER_SSH_HOST="${TP_WORKER_SSH:-max}"
SSH_USER="${TP_SSH_USER:-jryates}"
LEADER_IP="${TP_LEADER_IP:-10.99.0.2}"
PORT="${TP_PORT:-18128}"
DIR="${TP_DIR:-/home/${SSH_USER}/ds4-persist-gate}"
ROCMLIB="/opt/rocm-therock/lib"
MODEL="${TP_MODEL:-/home/${SSH_USER}/models/ds4-pr-20260801/DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf}"
NHI_DEV="/run/ds4-tbstream/device"
LEADER_LOG="${DIR}/tpab-${TAG}-leader.log"
WORKER_LOG="${DIR}/tpab-${TAG}-worker.log"
PROMPT="${DIR}/tp-ab-prompt.txt"
# The shard load is ~77 GiB per rank and is NOT page-cache resident between
# runs, so a cold start (post-reboot) can take >15 min before the worker
# connects.  Keep the leader's accept window comfortably above that.
TP_TIMEOUT_SEC="${TP_TIMEOUT_SEC:-2400}"
# The leader sizes its context at CTX+GEN; give the worker the same.
WORKER_CTX="${TP_WORKER_CTX:-$((CTX + GEN))}"
LOAD_WAIT_SEC="${TP_LOAD_WAIT_SEC:-1800}"
# Extra VAR=value pairs for both ranks.  DS4_TP_GATE_STATS prints the
# per-gate GPU-timeline cost summary at shutdown on both paths.
TP_EXTRA_ENV="${TP_EXTRA_ENV:-DS4_TP_GATE_STATS=1}"
# Per-gate trace CSVs (DS4_TP_GATE_TRACE, one per rank); TP_TRACE=0 disables.
# Fetched into $TP_OUT_DIR after the run and decomposed by
# docs/tp_persist_gate/tp_gate_trace.py (one-way latency vs rank skew).
TP_TRACE="${TP_TRACE:-1}"
TP_OUT_DIR="${TP_OUT_DIR:-/tmp/tpab}"
LEADER_TRACE="${DIR}/tpab-${TAG}-leader.trace.csv"
WORKER_TRACE="${DIR}/tpab-${TAG}-worker.trace.csv"
LEADER_TRACE_ENV=""; WORKER_TRACE_ENV=""
if [ "$TP_TRACE" = "1" ]; then
    LEADER_TRACE_ENV="DS4_TP_GATE_TRACE=$LEADER_TRACE"
    WORKER_TRACE_ENV="DS4_TP_GATE_TRACE=$WORKER_TRACE"
fi

sshx() { ssh -o BatchMode=yes "${SSH_USER}@$1" "${@:2}"; }

# Launch a long-running command on a node and return immediately.  The
# child must not inherit any of the ssh session's fds (stdin from
# /dev/null, stdout/stderr to the log) and the `&` must apply to the nohup
# command ALONE: `cd DIR && nohup CMD &` backgrounds the whole AND-list in a
# subshell that still holds the session's stdout and waits for CMD, so ssh
# never returns, the worker is never started, and the leader dies on its
# accept timeout (the 09-03 "nodes in a bad state" failure).
launch_detached() { # host log pid cmd...
    local host="$1" log="$2" pid="$3"
    shift 3
    sshx "$host" "cd $DIR || exit 1; rm -f '$log'; nohup $* > '$log' 2>&1 < /dev/null & echo \$! > '$pid'"
}

die() { printf 'tp_ab: %s\n' "$*" >&2; exit 1; }

wait_log_line() { # host file pattern timeout label
    local host="$1" file="$2" pat="$3" tmo="${4:-240}" label="$5"
    local i=0
    while [ $i -lt "$tmo" ]; do
        if sshx "$host" "grep -q '$pat' '$file' 2>/dev/null"; then
            printf 'tp_ab: %s: %s\n' "$label" "$(sshx "$host" "grep -m1 \"$pat\" '$file'")"
            return 0
        fi
        sleep 2; i=$((i + 2))
    done
    printf 'tp_ab: %s: timeout after %ss waiting for %s\n' \
        "$label" "$tmo" "'$pat'"
    return 1
}

wait_leader_done() { # poll until the leader process exits
    local i=0 tmo="${1:-1800}"
    while [ $i -lt "$tmo" ]; do
        if sshx "$LEADER_SSH_HOST" "! kill -0 \$(cat ${DIR}/tpab-${TAG}-leader.pid) 2>/dev/null"; then
            return 0
        fi
        sleep 3; i=$((i + 3))
    done
    die "leader did not finish within ${tmo}s"
}

cleanup_pair() { # hard-stop anything from this harness
    sshx "$LEADER_SSH_HOST" \
        "pkill -INT -x ds4-bench 2>/dev/null; pkill -INT -f 'ds4.*role coordinator' 2>/dev/null; sleep 2; pkill -9 -x ds4-bench 2>/dev/null; pkill -9 -f 'ds4.*role coordinator' 2>/dev/null; true"
    sshx "$WORKER_SSH_HOST" \
        "pkill -INT -f 'ds4.*role worker' 2>/dev/null; sleep 2; pkill -9 -f 'ds4.*role worker' 2>/dev/null; true"
    sleep 3
}

preflight() { # both nodes idle, NHI endpoint ready, prompt in place
    local h
    for h in "$LEADER_SSH_HOST" "$WORKER_SSH_HOST"; do
        sshx "$h" "test -c $NHI_DEV" \
            || die "$h: NHI device $NHI_DEV missing (ds4-tbstream-reconcile?)"
        sshx "$h" "test -x $DIR/ds4 && test -x $DIR/ds4-bench" \
            || die "$h: $DIR/ds4 or ds4-bench missing"
        # TBSTREAM_ZC_IMPORT needs CAP_SYS_RAWIO; a relink produces a new
        # inode without the file capability (the 09-03 "Operation not
        # permitted" run after the persist rebuild).  Restore it.
        local b
        for b in ds4 ds4-bench; do
            if ! sshx "$h" "getcap $DIR/$b | grep -q cap_sys_rawio"; then
                sshx "$h" "sudo -n setcap cap_sys_rawio+ep $DIR/$b" \
                    || die "$h: $DIR/$b lacks cap_sys_rawio and setcap failed"
                printf 'tp_ab: %s: restored cap_sys_rawio on %s\n' "$h" "$b"
            fi
        done
    done
    # ds4-bench takes the first CTX tokens of the prompt file; the repo's
    # standard bench text is long enough for every ctx the A/B uses.
    if ! sshx "$LEADER_SSH_HOST" "test -s $PROMPT"; then
        sshx "$LEADER_SSH_HOST" \
            "cp $DIR/speed-bench/promessi_sposi.txt $PROMPT" \
            || die "$LEADER_SSH_HOST: no $PROMPT and no speed-bench/promessi_sposi.txt to seed it"
        printf 'tp_ab: seeded %s from speed-bench/promessi_sposi.txt\n' "$PROMPT"
    fi
}

run_once() { # returns 0 on a full bound-and-finished run
    cleanup_pair

    printf 'tp_ab: persist=%s ctx=%s gen=%s tag=%s\n' \
        "$PERSIST" "$CTX" "$GEN" "$TAG"

    # ---- leader first (P0.2 startup order) -------------------------------
    launch_detached "$LEADER_SSH_HOST" "tpab-${TAG}-leader.log" "tpab-${TAG}-leader.pid" \
        env LD_LIBRARY_PATH=$ROCMLIB DS4_TP_PERSIST_GATE=$PERSIST DS4_TP_TIMEOUT_SEC=$TP_TIMEOUT_SEC $TP_EXTRA_ENV $LEADER_TRACE_ENV \
        ./ds4-bench --prompt-file $PROMPT --ctx-start $CTX --ctx-max $CTX --gen-tokens $GEN --show-output \
        --tensor-parallel --role coordinator --listen $LEADER_IP $PORT \
        --transport nhi --nhi-device $NHI_DEV -m $MODEL

    if ! wait_log_line "$LEADER_SSH_HOST" "$LEADER_LOG" \
        "waiting for worker" "$LOAD_WAIT_SEC" "leader listening"; then
        cleanup_pair
        return 1
    fi

    # ---- then worker ------------------------------------------------------
    launch_detached "$WORKER_SSH_HOST" "tpab-${TAG}-worker.log" "tpab-${TAG}-worker.pid" \
        env LD_LIBRARY_PATH=$ROCMLIB DS4_TP_PERSIST_GATE=$PERSIST DS4_TP_TIMEOUT_SEC=$TP_TIMEOUT_SEC $TP_EXTRA_ENV $WORKER_TRACE_ENV \
        ./ds4 --tensor-parallel --role worker --coordinator $LEADER_IP $PORT \
        --transport nhi --nhi-device $NHI_DEV -m $MODEL -c $WORKER_CTX

    # The worker only connects after its own shard load, so this wait has
    # to cover a cold load too.
    if ! wait_log_line "$LEADER_SSH_HOST" "$LEADER_LOG" \
        "worker connected" "$LOAD_WAIT_SEC" "TP bound"; then
        cleanup_pair
        return 1
    fi
    wait_log_line "$WORKER_SSH_HOST" "$WORKER_LOG" \
        "leader connected" 120 "worker bound" || true

    printf 'tp_ab: pair bound, running benchmark (ctx=%s gen=%s)\n' \
        "$CTX" "$GEN"
    wait_leader_done 1800
    return 0
}

report() { # returns 1 when the run produced no bench CSV line
    printf '\n===== bench CSV (leader) =====\n'
    local csv
    csv="$(sshx "$LEADER_SSH_HOST" "grep -E 'ctx_tokens|^[0-9]+,' $LEADER_LOG | tail -2")"
    printf '%s\n' "$csv"
    printf '\n===== gate stats (DS4_TP_GATE_STATS) =====\n'
    sshx "$LEADER_SSH_HOST" "grep -m1 'TP gate stats' $LEADER_LOG" || true
    sshx "$WORKER_SSH_HOST" "grep -m1 'TP gate stats' $WORKER_LOG" || true
    if [ "$TP_TRACE" = "1" ]; then
        printf '\n===== gate trace (DS4_TP_GATE_TRACE) =====\n'
        sshx "$LEADER_SSH_HOST" "grep -m1 'TP gate trace' $LEADER_LOG" || true
        sshx "$WORKER_SSH_HOST" "grep -m1 'TP gate trace' $WORKER_LOG" || true
        mkdir -p "$TP_OUT_DIR"
        scp -q -o BatchMode=yes "${SSH_USER}@${LEADER_SSH_HOST}:$LEADER_TRACE" \
            "$TP_OUT_DIR/${TAG}-leader.trace.csv" 2>/dev/null || true
        scp -q -o BatchMode=yes "${SSH_USER}@${WORKER_SSH_HOST}:$WORKER_TRACE" \
            "$TP_OUT_DIR/${TAG}-worker.trace.csv" 2>/dev/null || true
        if [ -s "$TP_OUT_DIR/${TAG}-leader.trace.csv" ] && \
           [ -s "$TP_OUT_DIR/${TAG}-worker.trace.csv" ] && \
           command -v python3 >/dev/null; then
            python3 "$(dirname "$0")/../docs/tp_persist_gate/tp_gate_trace.py" \
                "$TP_OUT_DIR/${TAG}-leader.trace.csv" \
                "$TP_OUT_DIR/${TAG}-worker.trace.csv" || true
        fi
    fi
    printf '\n===== generated output (token-exact A/B snapshot) =====\n'
    sshx "$LEADER_SSH_HOST" "grep -m2 'decoded text' $LEADER_LOG" || true
    printf '\n===== NHI stats (leader) =====\n'
    sshx "$LEADER_SSH_HOST" "grep -m2 'TP NHI stats' $LEADER_LOG" || true
    printf '\n===== NHI stats (worker) =====\n'
    sshx "$WORKER_SSH_HOST" "grep -m2 'TP NHI stats' $WORKER_LOG" || true
    printf '\n===== persist log line =====\n'
    sshx "$LEADER_SSH_HOST" "grep -m1 'persistent TP gate' $LEADER_LOG" || true
    sshx "$WORKER_SSH_HOST" "grep -m1 'persistent TP gate' $WORKER_LOG" || true
    printf '\n===== errors =====\n'
    sshx "$LEADER_SSH_HOST" "grep -m3 -E 'failed|error' $LEADER_LOG" || true
    sshx "$WORKER_SSH_HOST" "grep -m3 -E 'failed|error' $WORKER_LOG" || true
    if ! printf '%s\n' "$csv" | grep -qE '^[0-9]+,'; then
        printf '\ntp_ab: FAILED (%s): no bench CSV line in %s\n' "$TAG" "$LEADER_LOG"
        return 1
    fi
    printf '\ntp_ab: done (%s)\n' "$TAG"
}

preflight
ATTEMPTS="${TP_AB_ATTEMPTS:-3}"
attempt=1
while [ $attempt -le "$ATTEMPTS" ]; do
    printf 'tp_ab: attempt %d/%d\n' "$attempt" "$ATTEMPTS"
    if run_once; then
        report
        exit $?
    fi
    attempt=$((attempt + 1))
    sleep 15
done

printf 'tp_ab: all %s attempts failed\n' "$ATTEMPTS" >&2
report
exit 1