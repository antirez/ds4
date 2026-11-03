#!/bin/sh
set -eu

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<'USAGE'
Usage: tests/kv_restart_restore_smoke.sh [MODEL]

End-to-end regression for issue #1053: a tool-using Chat Completions
conversation, started warm (another conversation was live in the same
process), must be restored from the disk KV checkpoint written at graceful
shutdown.  The client never sends reasoning_content back, like every
OpenAI-compatible client.

Flow: start ds4-server with an empty --kv-disk-dir, run conversation A then
conversation B (with tools), SIGTERM, restart with the same dir, resend B's
history plus one user turn, and require that (almost) the whole prompt is
served from the cache.  Also checks that the shutdown checkpoint was keyed by
the client-visible transcript and that B's turns continued from live KV.

Skips (exit 0) when the model file is missing.  Needs python3 and ~10 minutes
on a Flash class model (DeepSeek, DeepSeek 4.1, GLM, Qwen3.8).

Environment:
  DS4_SERVER_BIN=./ds4-server
  DS4_TEST_MODEL=ds4flash.gguf
  DS4_KV_RESTORE_PORT=8199
  DS4_KV_RESTORE_CTX=32768
  DS4_KV_RESTORE_PREFILL_CHUNK=1024 set empty to omit --prefill-chunk, needed
                                    for syntaxes that reject it (GLM selects its
                                    own graph prefill chunks)
  DS4_KV_RESTORE_EXTRA_ARGS=""      (extra ds4-server args, e.g. "--gpu-devices 0")
  DS4_KV_RESTORE_KEEP=1             keep the temp dir (logs, kv files)
USAGE
    exit 0
fi

bin=${DS4_SERVER_BIN:-./ds4-server}
model=${1:-${DS4_TEST_MODEL:-ds4flash.gguf}}
port=${DS4_KV_RESTORE_PORT:-8199}
ctx=${DS4_KV_RESTORE_CTX:-32768}
extra=${DS4_KV_RESTORE_EXTRA_ARGS:-}
# GLM 5.x rejects --prefill-chunk, so let callers omit it for those models.
chunk=${DS4_KV_RESTORE_PREFILL_CHUNK-1024}
if [ -n "$chunk" ]; then chunk_arg="--prefill-chunk $chunk"; else chunk_arg=""; fi
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
client="$here/kv_restart_restore_client.py"

if [ ! -f "$model" ]; then
    echo "kv-restart-restore: skipped, missing model $model"
    exit 0
fi
if [ ! -x "$bin" ]; then
    echo "kv-restart-restore: missing $bin (run make ds4-server)" >&2
    exit 1
fi

# The Metal backend resolves its kernels as metal/<name>.metal relative to the
# working directory, so the server has to start from the repository root.
# Make both paths absolute first: either may be relative to the caller, and the
# checks above deliberately ran against the caller's view of them.
case "$bin" in
    /*) ;;
    *) bin="$(pwd)/${bin#./}" ;;
esac
case "$model" in
    /*) ;;
    *) model="$(pwd)/$model" ;;
esac

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-kv-restart.XXXXXX")
server_pid=""
cleanup() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [ "${DS4_KV_RESTORE_KEEP:-0}" = "1" ]; then
        echo "kv-restart-restore: kept $tmpdir"
    else
        rm -rf "$tmpdir"
    fi
}
trap cleanup EXIT INT HUP TERM

failures=""
fail() {
    echo "kv-restart-restore: FAIL: $*" >&2
    failures="${failures}x"
}

abort() {
    echo "kv-restart-restore: ABORT: $*" >&2
    for l in "$tmpdir/server1.log" "$tmpdir/server2.log"; do
        [ -f "$l" ] || continue
        echo "--- $l (kv lines) ---" >&2
        grep -E "kv cache|kv store|thinking live|tool-turn|live kv" "$l" >&2 || true
    done
    exit 1
}

start_server() {
    log=$1
    # shellcheck disable=SC2086
    ( cd "$root" && DS4_LOCK_FILE="$tmpdir/ds4.lock" exec "$bin" -m "$model" -c "$ctx" --port "$port" \
        $chunk_arg --kv-disk-dir "$tmpdir/kv" --kv-disk-space-mb 4096 \
        $extra ) >"$log" 2>&1 &
    server_pid=$!
    i=0
    while ! grep -q "listening on" "$log"; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            cat "$log" >&2
            abort "server exited during startup"
        fi
        i=$((i + 1))
        [ "$i" -gt 240 ] && abort "server did not start within 20 minutes"
        sleep 5
    done
}

stop_server() {
    kill -TERM "$server_pid"
    wait "$server_pid" || true
    server_pid=""
}

base="http://127.0.0.1:$port"
mkdir -p "$tmpdir/kv"

echo "kv-restart-restore: run 1 (conversation A, then B with tools)"
start_server "$tmpdir/server1.log"
python3 "$client" conversation "$base" "$tmpdir/b_history.json" || abort "conversation client failed"
stop_server

grep -q "reason=shutdown" "$tmpdir/server1.log" || abort "no shutdown checkpoint stored"
# The shutdown checkpoint must be keyed by the client-visible transcript.  A
# token-text key embeds the sampled hidden reasoning that the client never
# replays, so it can never match after a restart (issue #1053 log signature).
grep -q "reason=shutdown key=thinking-visible" "$tmpdir/server1.log" ||
    fail "shutdown checkpoint was not keyed by the visible transcript: $(grep 'reason=shutdown' "$tmpdir/server1.log")"
# B's turns (with tools) must continue from live KV rather than re-prefill.
if grep -q "TOOLS.*finish=" "$tmpdir/server1.log"; then
    grep -q "tool-turn visible checkpoint remembered" "$tmpdir/server1.log" ||
        fail "tool-call turn left no visible checkpoint"
fi

echo "kv-restart-restore: run 2 (restart, replay B + one turn)"
start_server "$tmpdir/server2.log"
result=$(python3 "$client" replay "$base" "$tmpdir/b_history.json") || abort "replay client failed"
stop_server

prompt=$(printf '%s' "$result" | sed -n 's/.*prompt=\([0-9]*\).*/\1/p')
cached=$(printf '%s' "$result" | sed -n 's/.*cached=\([0-9]*\).*/\1/p')
[ -n "$prompt" ] && [ -n "$cached" ] || abort "could not parse replay usage: $result"
# Only the new user turn plus the assistant prefix may be prefilled.
missing=$((prompt - cached))
if [ "$missing" -gt 64 ]; then
    fail "restart restored only $cached of $prompt prompt tokens ($missing re-prefilled)"
fi
grep -q "kv cache hit text .*key=thinking-visible" "$tmpdir/server2.log" ||
    fail "replay did not hit a visible-keyed checkpoint: $(grep -E 'kv cache hit|cache_source' "$tmpdir/server2.log" | head -1)"

if [ -n "$failures" ]; then
    echo "kv-restart-restore: FAILED (${#failures} check(s)); restored $cached of $prompt tokens after restart" >&2
    for l in "$tmpdir/server1.log" "$tmpdir/server2.log"; do
        echo "--- $l (kv lines) ---" >&2
        grep -E "kv cache|kv store|thinking live|tool-turn|live kv" "$l" >&2 || true
    done
    exit 1
fi
echo "kv-restart-restore: OK (restored $cached of $prompt tokens after restart)"
