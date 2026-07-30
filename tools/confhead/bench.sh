#!/bin/bash
# One bench leg: launch pulsar-server on MODEL with conf-sched TAU, run a
# fixed-seed suite, tear down. Callers serialize via flock on the gpu lock.
#
# Usage: tools/confhead/bench.sh <out_dir> <model.gguf> <tau> <suite> [runs] [port] [ctx]
#   suite: sweep | ab   (see bench_driver.py)
#   out_dir gets: server.log, driver.log, manifest.jsonl, texts/ (greedy), DONE
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${1:?usage: bench.sh <out_dir> <model> <tau> <suite> [runs] [port]}"
MODEL="${2:?}"
TAU="${3:?}"
SUITE="${4:?}"
RUNS="${5:-1}"
PORT="${6:-8077}"
CTX="${7:-36864}"
mkdir -p "$OUT"

if pgrep -x pulsar-server >/dev/null; then
    echo "FATAL: a pulsar-server is already running; refusing to proceed" >&2
    exit 1
fi
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
free -g | sed -n 2p
avail_gb=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
if [ "${avail_gb:-0}" -lt 100 ]; then
    echo "FATAL: only ${avail_gb} GiB available after drop_caches; box not clean" >&2
    exit 1
fi

export PULSAR_DSPARK_STATS=1
export PULSAR_DSPARK_CONF_SCHED="$TAU"

./pulsar-server -m "$MODEL" --dspark-draft 5 -c "$CTX" --port "$PORT" \
    > "$OUT/server.log" 2>&1 &
SRV=$!
echo "server pid $SRV (tau=$TAU model=$MODEL)"

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

python3 tools/confhead/bench_driver.py --port "$PORT" --suite "$SUITE" \
    --runs "$RUNS" --manifest "$OUT/manifest.jsonl" --texts "$OUT/texts" \
    > "$OUT/driver.log" 2>&1
RC=$?

kill -INT "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null
kill "$DOG" 2>/dev/null
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
[ "$RC" = 0 ] && touch "$OUT/DONE"
echo "BENCH-DONE rc=$RC tau=$TAU suite=$SUITE"
exit "$RC"
