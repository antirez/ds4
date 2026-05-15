#!/bin/bash
# bench_pool_scaling.sh — Demonstrate session pool value at increasing context sizes
#
# The key insight: without a pool, every agent switch forces re-prefill of the
# entire conversation history. At 50K tokens, that's ~15s of wasted compute on
# every turn. With the pool, the KV state is already in memory — only the new
# turn's ~50-100 tokens need prefill (~0.05s).
#
# This benchmark measures TTFT at multiple system prompt sizes to show how
# pool value scales linearly with prompt size.
#
# Usage:
#   ./tests/bench_pool_scaling.sh --model /path/to/model.gguf
#   ./tests/bench_pool_scaling.sh --model ./ds4flash.gguf --sizes 1000,5000,10000,25000
#   DS4_MODEL=/path/to/model.gguf ./tests/bench_pool_scaling.sh

set -euo pipefail

# --- Configuration ---
MODEL="${DS4_MODEL:-}"
SIZES="1000,5000,10000,25000,50000"
CTX=65536

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --sizes) SIZES="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        *) if [[ -z "$MODEL" ]]; then MODEL="$1"; fi; shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: No model specified. Use --model <path> or set DS4_MODEL env var."
    exit 1
fi

PORT_POOL=8201
PORT_SINGLE=8202
HOST="127.0.0.1"
SESSIONS_POOL=4
PIDS=()
LOGFILE_POOL="/tmp/ds4_bench_pool_$$.log"
LOGFILE_SINGLE="/tmp/ds4_bench_single_$$.log"
RESULTS_CSV="/tmp/ds4_bench_results_$$.csv"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# --- Helpers ---
cleanup() {
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do
        kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
    done
    rm -f "$LOGFILE_POOL" "$LOGFILE_SINGLE" /tmp/ds4_bench_prompt_*.txt \
          /tmp/ds4_bench_*.lock /tmp/ds4_bench_msg_*.json
}
trap cleanup EXIT

info() { echo -e "  ${CYAN}→${NC} $1" >&2; }
dim() { echo -e "  ${DIM}$1${NC}" >&2; }

# --- Generate system prompt of target token size ---
# Concatenates repo source files to build a realistic prompt of the desired size.
# No static fixture needed — the repo's own code is a perfect stand-in for a
# real system prompt (varied text, code-like structure, ~3.8 chars/token).
generate_prompt() {
    local target_tokens="$1"
    local outfile="$2"

    python3 << PYEOF
import os, glob

target_tokens = ${target_tokens}
target_chars = int(target_tokens * 3.8)
repo_dir = "${REPO_DIR}"

# Gather source files from the repo itself
source_parts = []
for pattern in ["*.c", "*.h", "*.md"]:
    for path in sorted(glob.glob(os.path.join(repo_dir, pattern))):
        try:
            with open(path) as f:
                source_parts.append(f"// --- {os.path.basename(path)} ---\n" + f.read())
        except:
            pass

source = "\n\n".join(source_parts)
if not source:
    # Fallback: synthetic tool definitions
    source = "You are an AI coding assistant.\n\nTOOL DEFINITIONS:\n"
    source += "\n".join([
        f"{i}. tool_{i}(param_a: string, param_b: int) - Operation {i} on given parameters."
        for i in range(1, 500)
    ])

if len(source) >= target_chars:
    prompt = source[:target_chars]
else:
    repeats = (target_chars // len(source)) + 1
    parts = []
    for i in range(repeats):
        if i > 0:
            parts.append(f"\n\n// --- repeated section {i+1} ---\n\n")
        parts.append(source)
    prompt = "".join(parts)[:target_chars]

with open("${outfile}", "w") as f:
    f.write(prompt)

print(f"{len(prompt)}|{len(prompt)/3.8:.0f}")
PYEOF
}

# --- Send a conversation turn ---
# Reads env: DS4_PORT, DS4_HOST, DS4_SESSION, DS4_MESSAGES_FILE, DS4_MAX_TOKENS, DS4_STREAM
# Output: ttft|total|content_text
#   DS4_STREAM=0: non-streaming (content = full API response content field)
#   DS4_STREAM=1: streaming (measures TTFT from first delta)
send_turn() {
    python3 << 'PYEOF'
import time, http.client, json, os

host = os.environ.get('DS4_HOST', '127.0.0.1')
port = int(os.environ['DS4_PORT'])
session_id = os.environ['DS4_SESSION']
messages_file = os.environ['DS4_MESSAGES_FILE']
max_tokens = int(os.environ.get('DS4_MAX_TOKENS', '1'))
use_stream = os.environ.get('DS4_STREAM', '1') == '1'

with open(messages_file) as f:
    messages = json.load(f)

conn = http.client.HTTPConnection(host, port, timeout=900)
body = json.dumps({
    'model': 'ds4',
    'messages': messages,
    'max_tokens': max_tokens,
    'stream': use_stream,
    'temperature': 0
})
headers = {
    'Content-Type': 'application/json',
    'X-Session-Id': session_id
}

t0 = time.perf_counter()
conn.request('POST', '/v1/chat/completions', body, headers)
resp = conn.getresponse()

if not use_stream:
    # Non-streaming: content field has the full text (matches canonical checkpoint)
    data = json.loads(resp.read())
    total = time.perf_counter() - t0
    conn.close()
    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
    # Write content to file to avoid shell quoting issues
    content_file = os.environ.get('DS4_CONTENT_FILE', '')
    if content_file:
        with open(content_file, 'w') as f:
            f.write(content)
    print(f'{total:.4f}|{total:.4f}')
else:
    # Streaming: measure TTFT
    ttft = None
    total = None
    while True:
        line = resp.readline().decode('utf-8', errors='replace')
        if not line:
            total = time.perf_counter() - t0
            break
        if line.startswith('data: [DONE]'):
            total = time.perf_counter() - t0
            break
        if line.startswith('data: '):
            try:
                chunk = json.loads(line[6:])
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                has_output = ('reasoning_content' in delta and delta['reasoning_content']) or \
                             ('content' in delta and delta['content'])
                if has_output and ttft is None:
                    ttft = time.perf_counter() - t0
            except:
                pass
    conn.close()
    if ttft is None:
        ttft = total if total else -1.0
    if total is None:
        total = ttft
    print(f'{ttft:.4f}|{total:.4f}|')
PYEOF
}

# --- Start server ---
start_server() {
    local port="$1"
    local sessions="$2"
    local logfile="$3"
    local label="$4"

    info "Starting server: ${label} (port=${port}, sessions=${sessions}, ctx=${CTX})..."

    DS4_LOCK_FILE="/tmp/ds4_bench_${port}.lock" \
    ./ds4-server --model "$MODEL" --ctx "$CTX" --sessions "$sessions" \
        --port "$port" --host "$HOST" > "$logfile" 2>&1 &
    local pid=$!
    PIDS+=("$pid")

    for i in $(seq 1 120); do
        if curl -s "http://${HOST}:${port}/v1/models" >/dev/null 2>&1; then
            info "  Ready (PID $pid)"
            echo "$pid"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}Server died during startup. Last 30 lines:${NC}" >&2
            tail -30 "$logfile" >&2
            exit 1
        fi
        sleep 1
    done
    echo -e "${RED}Server failed to start within 120s${NC}" >&2
    exit 1
}

# --- Run one size test ---
# Pattern: A.turn1(cold) → B.turn1(cold, forces switch) → A.turn2(switch back)
# We capture actual content from A.turn1 so A.turn2's tokens align with cache.
run_size_test() {
    local port="$1"
    local prompt_file="$2"

    local msg_file="/tmp/ds4_bench_msg_${port}_$$.json"

    local content_file="/tmp/ds4_bench_content_${port}_$$.txt"

    # --- A.turn1: cold start, NON-STREAMING to capture content for echo-back ---
    # Non-streaming returns the full text (incl. mid-think text) in content field,
    # which is what canonicalize_thinking_checkpoint uses for the canonical suffix.
    DS4_PROMPT_FILE="$prompt_file" python3 -c "
import json, os
with open(os.environ['DS4_PROMPT_FILE']) as f:
    sp = f.read()
msgs = [
    {'role': 'system', 'content': sp},
    {'role': 'user', 'content': 'What is 2+2? Answer briefly.'}
]
with open('$msg_file', 'w') as f:
    json.dump(msgs, f)
"
    local a1_result
    a1_result=$(DS4_PORT="$port" DS4_HOST="$HOST" DS4_SESSION="alpha-${port}-$$" \
                DS4_MESSAGES_FILE="$msg_file" DS4_MAX_TOKENS=20 DS4_STREAM=0 \
                DS4_CONTENT_FILE="$content_file" send_turn)
    local a1_ttft=$(echo "$a1_result" | cut -d'|' -f1)

    # --- B.turn1: different agent, forces context switch (streaming, measure TTFT) ---
    DS4_PROMPT_FILE="$prompt_file" python3 -c "
import json, os
with open(os.environ['DS4_PROMPT_FILE']) as f:
    sp = f.read()
msgs = [
    {'role': 'system', 'content': sp},
    {'role': 'user', 'content': 'What is 3+3? Answer briefly.'}
]
with open('$msg_file', 'w') as f:
    json.dump(msgs, f)
"
    local b1_result
    b1_result=$(DS4_PORT="$port" DS4_HOST="$HOST" DS4_SESSION="beta-${port}-$$" \
                DS4_MESSAGES_FILE="$msg_file" DS4_MAX_TOKENS=1 DS4_STREAM=1 send_turn)
    local b1_ttft=$(echo "$b1_result" | cut -d'|' -f1)

    # --- A.turn2: switch BACK to A — the key measurement (streaming TTFT) ---
    # Read content from file (avoids shell quoting issues with quotes/pipes in text)
    DS4_PROMPT_FILE="$prompt_file" DS4_CONTENT_FILE="$content_file" python3 -c "
import json, os
with open(os.environ['DS4_PROMPT_FILE']) as f:
    sp = f.read()
cf = os.environ.get('DS4_CONTENT_FILE', '')
a1 = ''
if cf and os.path.exists(cf):
    with open(cf) as f:
        a1 = f.read()
msgs = [
    {'role': 'system', 'content': sp},
    {'role': 'user', 'content': 'What is 2+2? Answer briefly.'},
    {'role': 'assistant', 'content': a1},
    {'role': 'user', 'content': 'Now what is 5+5?'}
]
with open('$msg_file', 'w') as f:
    json.dump(msgs, f)
"
    local a2_result
    a2_result=$(DS4_PORT="$port" DS4_HOST="$HOST" DS4_SESSION="alpha-${port}-$$" \
                DS4_MESSAGES_FILE="$msg_file" DS4_MAX_TOKENS=1 DS4_STREAM=1 send_turn)
    local a2_ttft=$(echo "$a2_result" | cut -d'|' -f1)

    rm -f "$msg_file" "$content_file"

    # Output: cold_avg|switch_ttft|a1_ttft|b1_ttft
    python3 -c "
a1=${a1_ttft}; b1=${b1_ttft}; a2=${a2_ttft}
cold_avg = (a1 + b1) / 2
print(f'{cold_avg:.4f}|{a2:.4f}|{a1:.4f}|{b1:.4f}')
"
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}${YELLOW}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${YELLOW} ds4-server Session Pool Scaling Benchmark${NC}"
echo -e "${BOLD}${YELLOW}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Model:    $MODEL"
echo -e "  Ctx:      $CTX tokens"
echo -e "  Sizes:    $SIZES tokens"
echo -e "  Pattern:  A.cold → B.cold (force switch) → A.switch (measure)"
echo -e "  Pool:     ${SESSIONS_POOL} sessions vs 1 session"
echo -e "  Thinking: enabled (default — exercises canonicalization cache)"
echo ""
echo -e "  ${DIM}Pool keeps each agent's KV state across switches.${NC}"
echo -e "  ${DIM}Without pool, switching re-prefills entire conversation.${NC}"
echo -e "  ${DIM}Savings scale linearly with prompt size.${NC}"
echo ""


# --- Start both servers ---
echo -e "${CYAN}━━━ Starting servers ━━━${NC}"
PID_POOL=$(start_server "$PORT_POOL" "$SESSIONS_POOL" "$LOGFILE_POOL" "pool (${SESSIONS_POOL} sessions)")
PID_SINGLE=$(start_server "$PORT_SINGLE" 1 "$LOGFILE_SINGLE" "no-pool (1 session)")
echo ""

# --- Run tests at each size ---
IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"
echo "size,pool_cold,pool_switch,single_cold,single_switch" > "$RESULTS_CSV"

for size in "${SIZE_ARRAY[@]}"; do
    if (( size + 500 > CTX )); then
        echo -e "  ${YELLOW}⚠ Skipping ${size} tokens (exceeds --ctx ${CTX})${NC}"
        continue
    fi

    echo -e "${CYAN}━━━ Testing: ${size} token system prompt ━━━${NC}"

    # Generate prompt
    local_prompt="/tmp/ds4_bench_prompt_${size}_$$.txt"
    prompt_info=$(generate_prompt "$size" "$local_prompt")
    local_chars=$(echo "$prompt_info" | cut -d'|' -f1)
    local_tokens=$(echo "$prompt_info" | cut -d'|' -f2)
    dim "Generated ${local_chars} chars (~${local_tokens} tokens)"

    # Test with pool
    dim "Pool (${SESSIONS_POOL} sessions)..."
    pool_result=$(run_size_test "$PORT_POOL" "$local_prompt")
    pool_cold_v=$(echo "$pool_result" | cut -d'|' -f1)
    pool_switch_v=$(echo "$pool_result" | cut -d'|' -f2)
    printf "    Pool:    cold=%ss  switch=%ss\n" "$pool_cold_v" "$pool_switch_v"

    # Test without pool
    dim "No pool (1 session)..."
    single_result=$(run_size_test "$PORT_SINGLE" "$local_prompt")
    single_cold_v=$(echo "$single_result" | cut -d'|' -f1)
    single_switch_v=$(echo "$single_result" | cut -d'|' -f2)
    printf "    No pool: cold=%ss  switch=%ss\n" "$single_cold_v" "$single_switch_v"

    # Speedup
    speedup=$(python3 -c "p=${pool_switch_v}; s=${single_switch_v}; print(f'{s/p:.1f}' if p > 0.001 else 'N/A')")
    echo -e "    ${GREEN}Switch speedup: ${speedup}x${NC}"
    echo ""

    echo "${size},${pool_cold_v},${pool_switch_v},${single_cold_v},${single_switch_v}" >> "$RESULTS_CSV"
    rm -f "$local_prompt"
done

# --- Kill servers ---
kill "$PID_POOL" 2>/dev/null; wait "$PID_POOL" 2>/dev/null || true
kill "$PID_SINGLE" 2>/dev/null; wait "$PID_SINGLE" 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BOLD}${YELLOW}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${YELLOW} RESULTS: Pool vs No-Pool TTFT on Agent Switch${NC}"
echo -e "${BOLD}${YELLOW}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

python3 << PYEOF
import csv

with open('${RESULTS_CSV}') as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("  No results collected.")
    exit(0)

sizes = [int(r['size']) for r in rows]
pool_cold = [float(r['pool_cold']) for r in rows]
pool_switch = [float(r['pool_switch']) for r in rows]
single_cold = [float(r['single_cold']) for r in rows]
single_switch = [float(r['single_switch']) for r in rows]

print(f"  {'Prompt Size':>12}  {'Pool TTFT':>10}  {'No-Pool TTFT':>13}  {'Speedup':>8}  {'Saved':>8}")
print(f"  {'─'*12}  {'─'*10}  {'─'*13}  {'─'*8}  {'─'*8}")

for i, size in enumerate(sizes):
    ps = pool_switch[i]
    ss = single_switch[i]
    speedup = ss / ps if ps > 0.001 else 0
    saved = ss - ps
    size_str = f"{size:,} tok"
    print(f"  {size_str:>12}  {ps:>9.3f}s  {ss:>12.3f}s  {speedup:>7.1f}x  {saved:>7.2f}s")

print()
if len(sizes) >= 2 and pool_switch[-1] > 0.001:
    peak = single_switch[-1] / pool_switch[-1]
    print(f"  📊 At {sizes[-1]:,} tokens: {peak:.0f}x faster switch with pool")
    print(f"     Pool TTFT stays ~constant (only new turn tokens prefilled)")
    print(f"     No-pool TTFT grows linearly (re-prefills entire conversation)")
print()
PYEOF

# --- Log analysis ---
echo -e "${CYAN}━━━ Cache spans (pool server) ━━━${NC}"
grep -o "ctx=[0-9]*\.\.[0-9]*:[0-9]*" "$LOGFILE_POOL" 2>/dev/null | tail -15 | while read -r line; do
    echo "    $line"
done
echo ""
echo -e "${CYAN}━━━ Cache spans (no-pool server) ━━━${NC}"
grep -o "ctx=[0-9]*\.\.[0-9]*:[0-9]*" "$LOGFILE_SINGLE" 2>/dev/null | tail -15 | while read -r line; do
    echo "    $line"
done

echo ""
echo -e "  ${GREEN}Done.${NC} Results: $RESULTS_CSV"
echo -e "  Logs: $LOGFILE_POOL / $LOGFILE_SINGLE"
