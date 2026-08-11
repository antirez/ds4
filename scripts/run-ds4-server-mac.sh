#!/usr/bin/env bash
# Start ds4-server on macOS with a RAM-safe --ctx (total RAM - 6 GiB ceiling).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RESERVE_GIB="${DS4_RAM_RESERVE_GIB:-6}"
HOST="${DS4_HOST:-127.0.0.1}"
PORT="${DS4_PORT:-8000}"
KV_DIR="${DS4_KV_DISK_DIR:-$HOME/.ds4/server-kv}"
KV_MB="${DS4_KV_DISK_SPACE_MB:-8192}"

if [[ ! -x ./ds4-server ]]; then
  echo "ds4-server not found; build with: make" >&2
  exit 1
fi

SAFE_CTX="$(python3 "$ROOT/scripts/safe_ctx.py" --reserve-gib "$RESERVE_GIB" --print-ctx)"
if [[ -z "$SAFE_CTX" || "$SAFE_CTX" -le 0 ]]; then
  echo "No safe --ctx under RAM-${RESERVE_GIB}GiB policy. Free memory or use a smaller model." >&2
  python3 "$ROOT/scripts/safe_ctx.py" --reserve-gib "$RESERVE_GIB" >&2 || true
  exit 1
fi

# Allow explicit override, but warn if it exceeds the safe budget.
CTX="${DS4_CTX:-$SAFE_CTX}"
if ! python3 "$ROOT/scripts/safe_ctx.py" --reserve-gib "$RESERVE_GIB" --ctx "$CTX" >/dev/null; then
  echo "warning: --ctx $CTX exceeds safe RAM-${RESERVE_GIB}GiB budget; refusing to start." >&2
  python3 "$ROOT/scripts/safe_ctx.py" --reserve-gib "$RESERVE_GIB" --ctx "$CTX" >&2 || true
  exit 2
fi

python3 "$ROOT/scripts/safe_ctx.py" --reserve-gib "$RESERVE_GIB" --ctx "$CTX"
echo "starting: ./ds4-server --metal --host $HOST --port $PORT --ctx $CTX --kv-disk-dir $KV_DIR --kv-disk-space-mb $KV_MB"
exec ./ds4-server --metal --host "$HOST" --port "$PORT" --ctx "$CTX" \
  --kv-disk-dir "$KV_DIR" --kv-disk-space-mb "$KV_MB" "$@"
