#!/bin/bash
# Conditional step 4 of the MXFP6 pre-check: tool-eval-bench --short smoke on a
# spliced artifact. Run ONLY if the frontier-logit drift is borderline-or-better,
# and ONLY under the GPU flock:
#   flock -w 28800 /home/tyler/Projects/AI/temp/gpu.lock \
#       bash tools/mxfp6/run_teb_smoke.sh MODEL.gguf TAG
# Expect ~97 if quality holds (v5mx --short precedent).
set -u
WT=/home/tyler/Projects/AI/temp/wt-mxfp6
ART=/home/tyler/Projects/AI/temp/mxfp6-artifacts
MODEL="${1:?model gguf required}"
TAG="${2:?tag required}"
PORT=8077
LOG="$ART/teb-$TAG.log"

# never touch a server we did not start; refuse if one is up
if pgrep -f '[.]/pulsar-server' >/dev/null || pgrep -x pulsar-server >/dev/null; then
    echo "FATAL: a pulsar-server is already running (not ours); refusing" | tee -a "$LOG"
    exit 1
fi

sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
if [ "$avail_kb" -le $((95 * 1024 * 1024)) ]; then
    echo "FATAL: only $((avail_kb / 1024 / 1024)) GiB available" | tee -a "$LOG"
    exit 1
fi

cd "$WT"
./pulsar-server -m "$MODEL" --port $PORT > "$ART/teb-$TAG-server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT

for _ in $(seq 240); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "FATAL: server died during load" | tee -a "$LOG"; exit 1; }
    sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "FATAL: server never became ready" | tee -a "$LOG"; exit 1; }

cd /home/tyler/Projects/AI/tool-eval-bench
uv run tool-eval-bench --base-url "http://127.0.0.1:$PORT/v1" --short --no-live \
    --json-file "$ART/teb-$TAG.json" > "$LOG" 2>&1
rc=$?
echo "TEB-SHORT-EXIT=$rc" >> "$LOG"
exit $rc
