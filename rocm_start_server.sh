#!/bin/bash
# Unified DS4 Start Script (ROCm/HIP optimized)
# Fuses cleanup, memory flushing, and server execution.

set -e

echo "--- Step 1: Killing Stale Server Instances ---"
pkill -9 -x ds4-server || true
rm -f /tmp/ds4.lock

echo "--- Step 2: Cleaning System Cache Memory ---"
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "--- Step 3: Setting ROCm Environment ---"
# For HSA/Strix Halo, unsetting COPY_MODEL enables optimal Zero-Copy path
unset DS4_HIP_COPY_MODEL
export DS4_HIP_PREFILL_CHUNK=4096

echo "--- Step 4: Starting DS4 Server ---"
# We run this in the background if it's called with --bg, otherwise we tail the log.
# For simplicity, we'll always use a log file to track initialization.
LOG_FILE="/tmp/ds4-server.log"

cd "$(dirname "$0")"
nohup ./ds4-server --rocm --ctx 65536 \
    --warm-weights \
    --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192 \
    > "$LOG_FILE" 2>&1 &

echo "--- Step 5: Waiting for Initialization ---"
sleep 2
tail -f "$LOG_FILE" | sed '/listening on/q'

echo "--- DONE: Server is running at http://127.0.0.1:8000 ---"
