#!/bin/bash
# Sanity-belt smoke: serve MODEL at conf-sched TAU and run the short
# tool-eval-bench scenario suite against it (15 scenarios, ~10 min).
#
# Usage: tools/confhead/smoke.sh <out_dir> <model.gguf> <tau> [port]
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${1:?usage: smoke.sh <out_dir> <model> <tau> [port]}"
MODEL="${2:?}"
TAU="${3:?}"
PORT="${4:-8077}"
mkdir -p "$OUT"

if pgrep -x pulsar-server >/dev/null; then
    echo "FATAL: a pulsar-server is already running; refusing to proceed" >&2
    exit 1
fi
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
avail_gb=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ "${avail_gb:-0}" -lt 100 ]; then
    echo "FATAL: only ${avail_gb} GiB available after drop_caches; box not clean" >&2
    exit 1
fi

DS4_DSPARK_CONF_SCHED="$TAU" ./pulsar-server -m "$MODEL" --dspark-draft 5 \
    -c 36864 --port "$PORT" > "$OUT/server.log" 2>&1 &
SRV=$!
echo "server pid $SRV (tau=$TAU)"

(
    breach=0
    while kill -0 "$SRV" 2>/dev/null; do
        avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
        if [ "$avail_kb" -lt $((5 * 1024 * 1024)) ]; then
            breach=$((breach + 1))
            [ "$breach" -ge 4 ] && { echo "WATCHDOG: killing $SRV" >&2; kill "$SRV"; break; }
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

/home/tyler/.local/bin/tool-eval-bench --short \
    --base-url "http://127.0.0.1:$PORT/v1" --model d > "$OUT/smoke.log" 2>&1
RC=$?

kill -INT "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null
kill "$DOG" 2>/dev/null
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
echo "SMOKE-DONE rc=$RC"
[ "$RC" = 0 ] && touch "$OUT/DONE"
tail -20 "$OUT/smoke.log"
exit "$RC"
