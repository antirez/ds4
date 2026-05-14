#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: tests/verify_turbo_server.sh [options]

Runs live ds4-server verification for TurboQuant KV modes:
  - /v1/models smoke check
  - non-streaming and streaming /v1/chat/completions
  - memory-token cache reuse
  - memory-text cache reuse across a tokenizer-boundary prompt extension
  - cold/continued/evict KV disk checkpoint writes
  - restart and disk-text KV checkpoint load
  - optional deterministic output comparison across FP8, turbo3, turbo4
  - optional long-prompt no-cache performance benchmark across FP8, turbo3, turbo4

Options:
  --server FILE       Server binary. Default: ./ds4-server
  --model FILE        GGUF model. Default: ds4flash.gguf
  --ctx N             Server context. Default: 8192
  --port N            Preferred port. Default: 1233
  --host HOST         Bind host. Default: 127.0.0.1
  --kv-dir DIR        Base KV cache directory. Default: /tmp/ds4-kv-turbo4
  --out-dir DIR       Log/trace output directory. Default: /tmp/ds4-turbo-verify.$$
  --primary MODE      Cache validation mode: 4, 3, fp8, or 0. Default: 4
  --no-cache          Skip primary cache validation phase.
  --compare MODES     Comma-separated comparison modes. Default: fp8,3,4
  --no-compare        Skip comparison phase.
  --bench MODES       Comma-separated benchmark modes. Default: fp8,3,4
  --no-bench          Skip performance benchmark phase.
  --bench-repeats N   Benchmark prompt paragraph repeats. Default: 240
  --bench-tokens N    Benchmark generated tokens. Default: 64
  --long-repeats N    Long prompt paragraph repeats. Default: 80
  --max-tokens N      Generated tokens for decode checks. Default: 16
  --start-timeout N   Seconds to wait for server startup. Default: 600
  --keep-server       Leave the last server running on failure.
  -h, --help          Show this help.

Environment overrides use the DS4_VERIFY_* prefix, for example:
  DS4_VERIFY_CTX=384000 DS4_VERIFY_KV_DIR=/tmp/ds4-kv-turbo4 tests/verify_turbo_server.sh
USAGE
}

SERVER="${DS4_VERIFY_SERVER:-./ds4-server}"
MODEL="${DS4_VERIFY_MODEL:-ds4flash.gguf}"
HOST="${DS4_VERIFY_HOST:-127.0.0.1}"
PORT="${DS4_VERIFY_PORT:-1233}"
CTX="${DS4_VERIFY_CTX:-8192}"
KV_BASE="${DS4_VERIFY_KV_DIR:-/tmp/ds4-kv-turbo4}"
OUT_DIR="${DS4_VERIFY_OUT_DIR:-/tmp/ds4-turbo-verify.$$}"
PRIMARY_MODE="${DS4_VERIFY_PRIMARY_MODE:-4}"
CACHE="${DS4_VERIFY_CACHE:-1}"
COMPARE_MODES="${DS4_VERIFY_COMPARE_MODES:-fp8,3,4}"
COMPARE="${DS4_VERIFY_COMPARE:-1}"
BENCH_MODES="${DS4_VERIFY_BENCH_MODES:-fp8,3,4}"
BENCH="${DS4_VERIFY_BENCH:-1}"
BENCH_REPEATS="${DS4_VERIFY_BENCH_REPEATS:-240}"
BENCH_TOKENS="${DS4_VERIFY_BENCH_TOKENS:-64}"
LONG_REPEATS="${DS4_VERIFY_LONG_REPEATS:-80}"
MAX_TOKENS="${DS4_VERIFY_MAX_TOKENS:-16}"
START_TIMEOUT="${DS4_VERIFY_START_TIMEOUT:-600}"
KV_SPACE_MB="${DS4_VERIFY_KV_SPACE_MB:-4096}"
KV_MIN_TOKENS="${DS4_VERIFY_KV_MIN_TOKENS:-64}"
KV_COLD_MAX_TOKENS="${DS4_VERIFY_KV_COLD_MAX_TOKENS:-4096}"
KV_CONTINUED_TOKENS="${DS4_VERIFY_KV_CONTINUED_TOKENS:-512}"
KV_TRIM_TOKENS="${DS4_VERIFY_KV_TRIM_TOKENS:-8}"
KV_ALIGN_TOKENS="${DS4_VERIFY_KV_ALIGN_TOKENS:-64}"
KEEP_SERVER=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --server) SERVER="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --kv-dir) KV_BASE="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --primary) PRIMARY_MODE="$2"; shift 2 ;;
        --no-cache) CACHE=0; shift ;;
        --compare) COMPARE_MODES="$2"; COMPARE=1; shift 2 ;;
        --no-compare) COMPARE=0; shift ;;
        --bench) BENCH_MODES="$2"; BENCH=1; shift 2 ;;
        --no-bench) BENCH=0; shift ;;
        --bench-repeats) BENCH_REPEATS="$2"; shift 2 ;;
        --bench-tokens) BENCH_TOKENS="$2"; shift 2 ;;
        --long-repeats) LONG_REPEATS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --start-timeout) START_TIMEOUT="$2"; shift 2 ;;
        --keep-server) KEEP_SERVER=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "verify_turbo_server: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

require_file() {
    if [ ! -e "$1" ]; then
        echo "verify_turbo_server: missing $2: $1" >&2
        exit 2
    fi
}

require_file "$SERVER" "server binary"
require_file "$MODEL" "model"
command -v curl >/dev/null 2>&1 || { echo "verify_turbo_server: curl is required" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "verify_turbo_server: python3 is required" >&2; exit 2; }

mkdir -p "$OUT_DIR"

SERVER_PID=""
SERVER_LOG=""
SERVER_TRACE=""
SERVER_PORT=""

cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        if [ "$KEEP_SERVER" = 1 ]; then
            echo "verify_turbo_server: leaving server pid $SERVER_PID running at http://$HOST:$SERVER_PORT" >&2
            return
        fi
        kill -TERM "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

choose_port() {
    python3 - "$HOST" "$PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
preferred = int(sys.argv[2])

def can_bind(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()

if can_bind(preferred):
    print(preferred)
else:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, 0))
    print(s.getsockname()[1])
    s.close()
PY
}

mode_name() {
    case "$1" in
        ""|0|fp8|FP8) echo "fp8" ;;
        3|turbo3) echo "turbo3" ;;
        4|turbo4) echo "turbo4" ;;
        *) echo "$1" ;;
    esac
}

mode_env_prefix() {
    case "$1" in
        ""|0|fp8|FP8) echo "env -u DS4_KV_TURBO" ;;
        3|turbo3) echo "env DS4_KV_TURBO=3" ;;
        4|turbo4) echo "env DS4_KV_TURBO=4" ;;
        *) echo "env DS4_KV_TURBO=$1" ;;
    esac
}

start_server() {
    local mode="$1"
    local phase="$2"
    local name
    name="$(mode_name "$mode")"
    SERVER_PORT="$(choose_port)"
    if [ "$SERVER_PORT" != "$PORT" ]; then
        echo "verify_turbo_server: preferred port $PORT is busy; using $SERVER_PORT" >&2
    fi

    local kv_dir="$KV_BASE.$name"
    mkdir -p "$kv_dir"
    SERVER_LOG="$OUT_DIR/$name.$phase.server.log"
    SERVER_TRACE="$OUT_DIR/$name.$phase.trace"
    echo "verify_turbo_server: starting $name server ($phase) on http://$HOST:$SERVER_PORT" >&2

    local env_prefix
    env_prefix="$(mode_env_prefix "$mode")"
    # shellcheck disable=SC2086
    $env_prefix "$SERVER" \
        --model "$MODEL" \
        --ctx "$CTX" \
        --tokens "$MAX_TOKENS" \
        --host "$HOST" \
        --port "$SERVER_PORT" \
        --trace "$SERVER_TRACE" \
        --kv-disk-dir "$kv_dir" \
        --kv-disk-space-mb "$KV_SPACE_MB" \
        --kv-cache-min-tokens "$KV_MIN_TOKENS" \
        --kv-cache-cold-max-tokens "$KV_COLD_MAX_TOKENS" \
        --kv-cache-continued-interval-tokens "$KV_CONTINUED_TOKENS" \
        --kv-cache-boundary-trim-tokens "$KV_TRIM_TOKENS" \
        --kv-cache-boundary-align-tokens "$KV_ALIGN_TOKENS" \
        >"$SERVER_LOG" 2>&1 &
    SERVER_PID="$!"
    wait_ready "$SERVER_PID" "http://$HOST:$SERVER_PORT"
}

start_server_bench() {
    local mode="$1"
    local phase="$2"
    local name
    name="$(mode_name "$mode")"
    SERVER_PORT="$(choose_port)"
    if [ "$SERVER_PORT" != "$PORT" ]; then
        echo "verify_turbo_server: preferred port $PORT is busy; using $SERVER_PORT" >&2
    fi

    SERVER_LOG="$OUT_DIR/$name.$phase.server.log"
    SERVER_TRACE="$OUT_DIR/$name.$phase.trace"
    echo "verify_turbo_server: starting $name server ($phase, no kv disk cache) on http://$HOST:$SERVER_PORT" >&2

    local env_prefix
    env_prefix="$(mode_env_prefix "$mode")"
    # shellcheck disable=SC2086
    $env_prefix "$SERVER" \
        --model "$MODEL" \
        --ctx "$CTX" \
        --tokens "$BENCH_TOKENS" \
        --host "$HOST" \
        --port "$SERVER_PORT" \
        --trace "$SERVER_TRACE" \
        >"$SERVER_LOG" 2>&1 &
    SERVER_PID="$!"
    wait_ready "$SERVER_PID" "http://$HOST:$SERVER_PORT"
}

stop_server() {
    if [ -z "${SERVER_PID:-}" ]; then
        return
    fi
    if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        kill -TERM "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
    SERVER_PID=""
}

wait_ready() {
    local pid="$1"
    local base_url="$2"
    local deadline=$((SECONDS + START_TIMEOUT))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            echo "verify_turbo_server: server exited during startup; log: $SERVER_LOG" >&2
            tail -80 "$SERVER_LOG" >&2 || true
            exit 1
        fi
        if curl -fsS --max-time 2 "$base_url/v1/models" >"$OUT_DIR/.ready.json" 2>/dev/null; then
            return
        fi
        sleep 1
    done
    echo "verify_turbo_server: timed out waiting for server; log: $SERVER_LOG" >&2
    tail -80 "$SERVER_LOG" >&2 || true
    exit 1
}

client_json() {
    local base_url="$1"
    local action="$2"
    local out="$3"
    python3 - "$base_url" "$action" "$out" "$LONG_REPEATS" "$MAX_TOKENS" "$BENCH_REPEATS" "$BENCH_TOKENS" <<'PY'
import json
import sys
import urllib.error
import urllib.request

base_url, action, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
long_repeats, max_tokens = int(sys.argv[4]), int(sys.argv[5])
bench_repeats, bench_tokens = int(sys.argv[6]), int(sys.argv[7])

def request_json(path, payload=None, stream=False):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base_url + path, data=data)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            if stream:
                chunks = []
                saw_done = False
                for raw in resp:
                    line = raw.decode("utf-8", "replace").rstrip("\n")
                    chunks.append(line)
                    if line.strip() == "data: [DONE]":
                        saw_done = True
                text = "\n".join(chunks) + "\n"
                if not saw_done:
                    raise RuntimeError("stream ended without [DONE]")
                return text
            body = resp.read().decode("utf-8")
            return body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e

def chat_payload(content, tokens=max_tokens, stream=False):
    return {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": content}],
        "think": False,
        "temperature": 0,
        "top_p": 1,
        "seed": 20260513,
        "max_tokens": tokens,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }

def long_prompt():
    para = (
        "Cache verification record {i}: the prefix must remain byte-stable, "
        "the model should continue after a restored KV checkpoint, and every "
        "line includes deterministic filler for token volume.\n"
    )
    return "".join(para.format(i=i) for i in range(long_repeats))

def bench_prompt():
    para = (
        "Performance benchmark record {i}: this stable long-context prefix "
        "forces the attention path to read compressed KV rows during decode, "
        "so FP8 and TurboQuant cache formats can be compared directly.\n"
    )
    return "".join(para.format(i=i) for i in range(bench_repeats))

if action == "models":
    body = request_json("/v1/models")
    obj = json.loads(body)
    assert obj.get("object") == "list"
elif action == "short":
    body = request_json("/v1/chat/completions", chat_payload("Define a KV cache hit in one short sentence."))
    obj = json.loads(body)
    choice = obj["choices"][0]
    assert choice["message"].get("content", "") is not None
elif action == "stream":
    body = request_json("/v1/chat/completions", chat_payload("Reply with exactly three words about local inference.", stream=True), stream=True)
    assert "data: " in body
elif action == "memory0":
    body = request_json("/v1/chat/completions", chat_payload("Keep this prompt resident for a zero-token cache probe.", tokens=0))
    obj = json.loads(body)
    assert obj["choices"][0]["finish_reason"] in ("length", "stop")
elif action == "memory_text":
    prompt = "Write one common English word related to caching."
    first = request_json("/v1/chat/completions", chat_payload(prompt, tokens=1))
    first_obj = json.loads(first)
    piece = first_obj["choices"][0]["message"].get("content", "")
    if not piece:
        raise RuntimeError("memory-text probe generated no assistant bytes")
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": piece + "s"},
        ],
        "think": False,
        "temperature": 0,
        "top_p": 1,
        "seed": 20260513,
        "max_tokens": 0,
        "stream": False,
    }
    body = request_json("/v1/chat/completions", payload)
    obj = json.loads(body)
    assert obj["choices"][0]["finish_reason"] in ("length", "stop")
elif action == "long":
    body = request_json("/v1/chat/completions", chat_payload(long_prompt()))
    obj = json.loads(body)
    msg = obj["choices"][0]["message"]
    assert "content" in msg
elif action == "evict":
    body = request_json("/v1/chat/completions", chat_payload("Unrelated short prompt used to evict the previous live KV session.", tokens=0))
    obj = json.loads(body)
    assert obj["choices"]
elif action == "compare":
    body = request_json("/v1/chat/completions", chat_payload("In one sentence, describe why disk KV cache reuse helps long chats.", tokens=24))
    obj = json.loads(body)
    content = obj["choices"][0]["message"].get("content", "")
    if not content.strip():
        raise RuntimeError("empty comparison content")
    words = content.split()
    run = 1
    for prev, cur in zip(words, words[1:]):
        run = run + 1 if cur == prev else 1
        if run >= 8:
            raise RuntimeError("comparison content appears repetitively collapsed")
elif action == "bench":
    body = request_json("/v1/chat/completions", chat_payload(bench_prompt(), tokens=bench_tokens))
    obj = json.loads(body)
    content = obj["choices"][0]["message"].get("content", "")
    if not content.strip():
        raise RuntimeError("empty benchmark content")
else:
    raise RuntimeError(f"unknown action: {action}")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(body)
PY
}

require_trace_pattern() {
    local pattern="$1"
    local file="$2"
    local label="$3"
    if ! grep -q "$pattern" "$file"; then
        echo "verify_turbo_server: missing $label in $file" >&2
        tail -120 "$file" >&2 || true
        exit 1
    fi
}

require_log_pattern() {
    local pattern="$1"
    local file="$2"
    local label="$3"
    if ! grep -q "$pattern" "$file"; then
        echo "verify_turbo_server: missing $label in $file" >&2
        tail -120 "$file" >&2 || true
        exit 1
    fi
}

count_pattern() {
    local pattern="$1"
    local file="$2"
    if [ -f "$file" ]; then
        grep -c "$pattern" "$file" || true
    else
        echo 0
    fi
}

summarize_phase() {
    local mode="$1"
    local phase="$2"
    local log="$OUT_DIR/$(mode_name "$mode").$phase.server.log"
    local trace="$OUT_DIR/$(mode_name "$mode").$phase.trace"
    {
        echo "mode=$(mode_name "$mode") phase=$phase"
        echo "log=$log"
        echo "trace=$trace"
        echo "memory-token=$(count_pattern 'cache_source: memory-token' "$trace")"
        echo "memory-text=$(count_pattern 'cache_source: memory-text' "$trace")"
        echo "disk-text=$(count_pattern 'cache_source: disk-text' "$trace")"
        echo "cold-store=$(count_pattern 'reason=cold' "$log")"
        echo "continued-store=$(count_pattern 'reason=continued' "$log")"
        echo "evict-store=$(count_pattern 'reason=evict' "$log")"
        grep -E 'context buffers|prefill chunk|prompt done|decoding chunk|kv cache (stored|hit|load failed|evicted)' "$log" || true
    } >"$OUT_DIR/$(mode_name "$mode").$phase.summary"
}

run_primary_cache_validation() {
    local mode="$1"
    local name
    name="$(mode_name "$mode")"
    local base_url
    rm -rf "$KV_BASE.$name"
    mkdir -p "$KV_BASE.$name"

    start_server "$mode" "cache1"
    base_url="http://$HOST:$SERVER_PORT"
    client_json "$base_url" models "$OUT_DIR/$name.models.json"
    client_json "$base_url" short "$OUT_DIR/$name.short.json"
    client_json "$base_url" stream "$OUT_DIR/$name.stream.sse"
    client_json "$base_url" memory0 "$OUT_DIR/$name.memory0.a.json"
    client_json "$base_url" memory0 "$OUT_DIR/$name.memory0.b.json"
    client_json "$base_url" memory_text "$OUT_DIR/$name.memory-text.json"
    client_json "$base_url" long "$OUT_DIR/$name.long.first.json"
    client_json "$base_url" evict "$OUT_DIR/$name.evict.json"
    stop_server

    summarize_phase "$mode" "cache1"
    require_trace_pattern 'cache_source: memory-token' "$OUT_DIR/$name.cache1.trace" "memory-token cache source"
    require_trace_pattern 'cache_source: memory-text' "$OUT_DIR/$name.cache1.trace" "memory-text cache source"
    require_log_pattern 'reason=cold' "$OUT_DIR/$name.cache1.server.log" "cold KV store"
    require_log_pattern 'reason=continued' "$OUT_DIR/$name.cache1.server.log" "continued KV store"
    require_log_pattern 'reason=evict' "$OUT_DIR/$name.cache1.server.log" "evict KV store"

    local kv_count
    kv_count="$(find "$KV_BASE.$name" -name '*.kv' -type f | wc -l | tr -d ' ')"
    if [ "$kv_count" = "0" ]; then
        echo "verify_turbo_server: no KV checkpoint files were created in $KV_BASE.$name" >&2
        exit 1
    fi

    start_server "$mode" "cache2"
    base_url="http://$HOST:$SERVER_PORT"
    client_json "$base_url" long "$OUT_DIR/$name.long.second.json"
    stop_server

    summarize_phase "$mode" "cache2"
    require_trace_pattern 'cache_source: disk-text' "$OUT_DIR/$name.cache2.trace" "disk-text cache source after restart"
    require_log_pattern 'kv cache hit text' "$OUT_DIR/$name.cache2.server.log" "KV disk cache hit"
}

run_compare_mode() {
    local mode="$1"
    local name
    name="$(mode_name "$mode")"
    start_server "$mode" "compare"
    client_json "http://$HOST:$SERVER_PORT" compare "$OUT_DIR/$name.compare.json"
    stop_server
    summarize_phase "$mode" "compare"
}

run_comparison() {
    local modes_csv="$1"
    local modes
    modes="${modes_csv//,/ }"
    for mode in $modes; do
        run_compare_mode "$mode"
    done
    python3 - "$OUT_DIR" $modes <<'PY'
import json
import os
import sys

out_dir = sys.argv[1]
modes = sys.argv[2:]
for mode in modes:
    name = {"0": "fp8", "fp8": "fp8", "FP8": "fp8", "3": "turbo3", "turbo3": "turbo3", "4": "turbo4", "turbo4": "turbo4"}.get(mode, mode)
    path = os.path.join(out_dir, f"{name}.compare.json")
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    content = obj["choices"][0]["message"].get("content", "").strip().replace("\n", "\\n")
    print(f"{name}: {content}")
PY
}

run_bench_mode() {
    local mode="$1"
    local name
    name="$(mode_name "$mode")"
    start_server_bench "$mode" "bench"
    client_json "http://$HOST:$SERVER_PORT" bench "$OUT_DIR/$name.bench.json"
    stop_server
    summarize_phase "$mode" "bench"
}

run_benchmark() {
    local modes_csv="$1"
    local modes
    modes="${modes_csv//,/ }"
    for mode in $modes; do
        run_bench_mode "$mode"
    done
    python3 - "$OUT_DIR" $modes <<'PY'
import os
import re
import sys

out_dir = sys.argv[1]
modes = sys.argv[2:]

def mode_name(mode):
    return {"0": "fp8", "fp8": "fp8", "FP8": "fp8", "3": "turbo3", "turbo3": "turbo3", "4": "turbo4", "turbo4": "turbo4"}.get(mode, mode)

print("mode\tcontext_mib\tprompt_tokens\tprefill_sec\tprefill_tps\tdecode_tokens\tdecode_tps")
for mode in modes:
    name = mode_name(mode)
    path = os.path.join(out_dir, f"{name}.bench.server.log")
    text = open(path, encoding="utf-8", errors="replace").read()

    mem = ""
    m = re.search(r"context buffers ([0-9.]+) MiB", text)
    if m:
        mem = m.group(1)

    prompt_tokens = ""
    prefill_sec = ""
    prefill_tps = ""
    prompt_matches = list(re.finditer(r"ctx=([0-9]+)\.\.([0-9]+):([0-9]+) prompt done ([0-9.]+)s", text))
    if prompt_matches:
        m = prompt_matches[-1]
        prompt_tokens = m.group(3)
        prefill_sec = m.group(4)
        try:
            prefill_tps = f"{float(prompt_tokens) / float(prefill_sec):.2f}"
        except ZeroDivisionError:
            prefill_tps = "inf"

    decode_tokens = ""
    decode_tps = ""
    decode_matches = list(re.finditer(r"gen=([0-9]+) decoding chunk=[0-9.]+ t/s avg=([0-9.]+) t/s", text))
    if decode_matches:
        m = decode_matches[-1]
        decode_tokens = m.group(1)
        decode_tps = m.group(2)

    print(f"{name}\t{mem}\t{prompt_tokens}\t{prefill_sec}\t{prefill_tps}\t{decode_tokens}\t{decode_tps}")
PY
}

echo "verify_turbo_server: logs: $OUT_DIR" >&2

if [ "$CACHE" = 1 ]; then
    run_primary_cache_validation "$PRIMARY_MODE"
fi

if [ "$COMPARE" = 1 ]; then
    run_comparison "$COMPARE_MODES" >"$OUT_DIR/compare.outputs"
fi

if [ "$BENCH" = 1 ]; then
    run_benchmark "$BENCH_MODES" >"$OUT_DIR/bench.tsv"
fi

{
    echo "verify_turbo_server: OK"
    echo "output_dir=$OUT_DIR"
    echo "kv_base=$KV_BASE"
    echo "primary=$(mode_name "$PRIMARY_MODE")"
    if [ "$CACHE" = 1 ]; then
        echo "cache_summaries:"
        ls "$OUT_DIR"/*.summary 2>/dev/null || true
    fi
    if [ -f "$OUT_DIR/compare.outputs" ]; then
        echo "compare_outputs:"
        cat "$OUT_DIR/compare.outputs"
    fi
    if [ -f "$OUT_DIR/bench.tsv" ]; then
        echo "benchmark:"
        cat "$OUT_DIR/bench.tsv"
    fi
} | tee "$OUT_DIR/summary.txt"
