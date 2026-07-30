#!/bin/bash
# Confidence-head retrain, stage 1: run the live engine (sampled temperature-
# matched drafting) over the collect_driver.py mixture, dumping lean
# confidence records + fused stats for offline labeling.
#
# Usage: tools/confhead/collect.sh <out_dir> [model.gguf] [port] [ctx] [driver args...]
#   out_dir gets: dump.bin, server.log, driver.log, manifest.jsonl, DONE
#
# Successor of temp/conf_collect2.sh (the argmax-era collector). Differences:
# runs the merged v5mx artifact (drafter auto-detected, no --dspark), samples
# at request temperature (labels = real p/q accept outcomes), and spans the
# temp x workload x depth mixture instead of greedy-only ~1k ctx.
#
# GB10 discipline: refuses to start if any pulsar-server is already running
# (never kills a process it did not start), drops caches before the load,
# and runs a sustained-breach MemAvailable watchdog (4 consecutive 5s samples
# < 5 GiB) that kills only the server it spawned. Callers serialize GPU use
# by running this under flock on the gpu lock.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${1:?usage: collect.sh <out_dir> [model] [port] [ctx] [driver args...]}"
MODEL="${2:-gguf/model.gguf}"
PORT="${3:-8077}"
CTX="${4:-36864}"
shift $(( $# < 4 ? $# : 4 ))
mkdir -p "$OUT"

if pgrep -x pulsar-server >/dev/null; then
    echo "FATAL: a pulsar-server is already running; refusing to proceed" >&2
    exit 1
fi
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
free -g | sed -n 2p
# GB10 model-load gate: the 91 GB load fails (or worse, lands and immediately
# watchdogs) if something else still holds unified memory -- which is often
# invisible to RSS and survives drop_caches.
avail_gb=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ "${avail_gb:-0}" -lt 100 ]; then
    echo "FATAL: only ${avail_gb} GiB available after drop_caches; box not clean" >&2
    exit 1
fi

export DS4_DSPARK_STATS=1
export DS4_DSPARK_CONF_SCHED=off       # unbiased labels: verify all n_draft
export DS4_DSPARK_DUMP="$OUT/dump.bin"
export DS4_DSPARK_DUMP_LEAN=1
export DS4_DSPARK_DUMP_STEPS=200000

./pulsar-server -m "$MODEL" --dspark-draft 5 -c "$CTX" --port "$PORT" \
    > "$OUT/server.log" 2>&1 &
SRV=$!
echo "server pid $SRV"

# MemAvailable watchdog: sustained breach only (a transient dip during load
# is normal on GB10). Kills the server we spawned, by explicit pid.
(
    breach=0
    while kill -0 "$SRV" 2>/dev/null; do
        avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
        if [ "$avail_kb" -lt $((5 * 1024 * 1024)) ]; then
            breach=$((breach + 1))
            if [ "$breach" -ge 4 ]; then
                echo "WATCHDOG: MemAvailable < 5 GiB for 4 samples; killing $SRV" >&2
                kill "$SRV"
                break
            fi
        else
            breach=0
        fi
        sleep 5
    done
) &
DOG=$!

ready=0
for i in $(seq 1 900); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then break; fi
    sleep 1
done
if [ "$ready" != 1 ]; then
    echo "FATAL: server never became ready" >&2
    kill "$SRV" 2>/dev/null; kill "$DOG" 2>/dev/null
    exit 1
fi
echo "ready after ${i}s"

python3 tools/confhead/collect_driver.py --port "$PORT" \
    --manifest "$OUT/manifest.jsonl" "$@" > "$OUT/driver.log" 2>&1
RC=$?

kill -INT "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null
kill "$DOG" 2>/dev/null

ls -la "$OUT/dump.bin" 2>/dev/null
grep -c "dspark fused" "$OUT/server.log" || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
[ "$RC" = 0 ] && touch "$OUT/DONE"
echo "COLLECT-DONE rc=$RC"
exit "$RC"
