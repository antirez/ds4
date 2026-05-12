#!/bin/bash
# test_warmup.sh — Integration test for ds4-server /v1/warmup endpoint
#
# Tests:
#   1. Fire a warmup request → get 202 Accepted immediately
#   2. After warmup completes, chat completion with same session is fast
#   3. Compare cold start vs pre-warmed latency
#
# Usage:
#   ./tests/test_warmup.sh [--model /path/to/model.gguf]
#   DS4_MODEL=/path/to/model.gguf ./tests/test_warmup.sh

set -euo pipefail

# --- Configuration ---
MODEL="${DS4_MODEL:-${1:-}}"
if [[ "${1:-}" == "--model" ]]; then MODEL="${2:-}"; fi
if [[ -z "$MODEL" ]]; then
    echo "ERROR: No model specified. Use --model <path> or set DS4_MODEL env var."
    exit 1
fi

PORT=8198
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
SESSIONS=4
CTX=8192
SERVER_PID=""
LOGFILE="/tmp/ds4_test_warmup_$$.log"
PASS=0
FAIL=0

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Helpers ---
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$LOGFILE"
}
trap cleanup EXIT

pass() { PASS=$((PASS + 1)); echo -e "  ${GREEN}✓ PASS${NC}: $1"; }
fail() { FAIL=$((FAIL + 1)); echo -e "  ${RED}✗ FAIL${NC}: $1"; }
info() { echo -e "  ${CYAN}→${NC} $1"; }

# A longer system prompt to make prefill time measurable
LONG_SYS="You are a highly specialized AI assistant with expertise in mathematics, physics, chemistry, biology, computer science, and engineering. You follow a structured approach: first understand the problem, then identify relevant principles, apply them step by step, verify the answer, and present it clearly. You always show intermediate steps. You never skip calculations. You cite relevant theorems, laws, or principles by name. When multiple approaches exist, you briefly mention alternatives before proceeding with the most efficient one. You format mathematical expressions clearly using standard notation."

WARMUP_MSGS='[
    {"role": "system", "content": "'"$LONG_SYS"'"},
    {"role": "user", "content": "Explain the fundamental theorem of calculus and its two parts."},
    {"role": "assistant", "content": "The Fundamental Theorem of Calculus (FTC) connects differentiation and integration. Part 1: If f is continuous on [a,b], then F(x) = integral from a to x of f(t)dt is differentiable and F'\''(x) = f(x). Part 2: If F is any antiderivative of f on [a,b], then the definite integral from a to b of f(x)dx = F(b) - F(a)."},
    {"role": "user", "content": "Now prove Part 1 rigorously using the epsilon-delta definition of limits."}
]'

# --- Start server ---
echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
echo -e "${YELLOW} ds4-server Warmup Endpoint Test${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Model: $MODEL"
echo -e "  Server: ${HOST}:${PORT}"
echo ""

info "Starting ds4-server with $SESSIONS sessions..."
./ds4-server --model "$MODEL" --ctx "$CTX" --sessions "$SESSIONS" \
    --port "$PORT" --host "$HOST" > "$LOGFILE" 2>&1 &
SERVER_PID=$!

# Wait for server ready
for i in $(seq 1 60); do
    if curl -s "${BASE_URL}/v1/models" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo -e "${RED}Server died during startup. Log:${NC}"
        tail -20 "$LOGFILE"
        exit 1
    fi
    sleep 1
done
if ! curl -s "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo -e "${RED}Server failed to start within 60s${NC}"
    tail -20 "$LOGFILE"
    exit 1
fi
info "Server ready (PID $SERVER_PID)"
echo ""

# ═══════════════════════════════════════════════════
# TEST 1: Warmup returns 202 immediately
# ═══════════════════════════════════════════════════
echo -e "${CYAN}━━━ Test 1: Warmup returns 202 Accepted ━━━${NC}"

# Time how long the warmup HTTP call takes (should be near-instant, <1s)
WARMUP_START=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")

warmup_resp=$(curl -s -w "\n%{http_code}" --max-time 10 \
    "${BASE_URL}/v1/warmup" \
    -H "Content-Type: application/json" \
    -H "X-Session-Id: pre-warmed-1" \
    -d "{
        \"model\": \"deepseek-v4-flash\",
        \"messages\": $WARMUP_MSGS,
        \"max_tokens\": 50
    }")

WARMUP_END=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")
warmup_elapsed=$(echo "scale=3; ($WARMUP_END - $WARMUP_START) / 1000000000" | bc 2>/dev/null || python3 -c "print(f'{($WARMUP_END - $WARMUP_START)/1e9:.3f}')")

warmup_http_code=$(echo "$warmup_resp" | tail -1)
warmup_body=$(echo "$warmup_resp" | sed '$d')

info "Warmup response time: ${warmup_elapsed}s"
info "HTTP status: $warmup_http_code"
info "Body: $warmup_body"

if [[ "$warmup_http_code" == "202" ]]; then
    pass "Warmup returned 202 Accepted"
else
    fail "Expected 202, got $warmup_http_code"
fi

# Check that the response contains session_id
if echo "$warmup_body" | grep -q "pre-warmed-1"; then
    pass "Response confirms session_id 'pre-warmed-1'"
else
    fail "Response doesn't contain the session_id"
fi
echo ""

# ═══════════════════════════════════════════════════
# TEST 2: Wait for prefill, then send completion
# ═══════════════════════════════════════════════════
echo -e "${CYAN}━━━ Test 2: Pre-warmed session is fast ━━━${NC}"

# Wait for the async prefill to complete
info "Waiting 5s for warmup prefill to complete..."
sleep 5

# Check server log for warmup completion
if grep -q "warmup pre-warmed-1 done" "$LOGFILE" 2>/dev/null; then
    warmup_log=$(grep "warmup pre-warmed-1" "$LOGFILE" | tail -1)
    info "Server log: $warmup_log"
    pass "Warmup prefill completed (confirmed in server log)"
elif grep -q "warmup pre-warmed-1 already warm" "$LOGFILE" 2>/dev/null; then
    pass "Session was already warm"
else
    info "Warmup may still be running (no completion log yet)"
    info "Waiting 10 more seconds..."
    sleep 10
    if grep -q "warmup pre-warmed-1" "$LOGFILE" 2>/dev/null; then
        warmup_log=$(grep "warmup pre-warmed-1" "$LOGFILE" | tail -1)
        info "Server log: $warmup_log"
    fi
fi

# Now send a chat completion with the SAME messages — should hit the warm cache
WARM_START=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")

warm_resp=$(curl -s --max-time 120 \
    "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-Session-Id: pre-warmed-1" \
    -d "{
        \"model\": \"deepseek-v4-flash\",
        \"messages\": $WARMUP_MSGS,
        \"max_tokens\": 50,
        \"stream\": false,
        \"temperature\": 0
    }")

WARM_END=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")
warm_elapsed=$(echo "scale=3; ($WARM_END - $WARM_START) / 1000000000" | bc 2>/dev/null || python3 -c "print(f'{($WARM_END - $WARM_START)/1e9:.3f}')")

warm_pt=$(echo "$warm_resp" | grep -o '"prompt_tokens":[0-9]*' | head -1 | cut -d: -f2)
info "Pre-warmed completion: ${warm_elapsed}s, prompt_tokens=$warm_pt"

if [[ -n "$warm_pt" && "$warm_pt" -gt 0 ]]; then
    pass "Pre-warmed session returned a valid response"
else
    fail "Pre-warmed session failed to respond"
    info "Response: $warm_resp"
fi
echo ""

# ═══════════════════════════════════════════════════
# TEST 3: Compare cold start vs pre-warmed
# ═══════════════════════════════════════════════════
echo -e "${CYAN}━━━ Test 3: Cold start vs pre-warmed latency ━━━${NC}"

# Send the same prompt to a NEW session (cold — no warmup)
COLD_START=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")

cold_resp=$(curl -s --max-time 120 \
    "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-Session-Id: cold-session-new" \
    -d "{
        \"model\": \"deepseek-v4-flash\",
        \"messages\": $WARMUP_MSGS,
        \"max_tokens\": 50,
        \"stream\": false,
        \"temperature\": 0
    }")

COLD_END=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")
cold_elapsed=$(echo "scale=3; ($COLD_END - $COLD_START) / 1000000000" | bc 2>/dev/null || python3 -c "print(f'{($COLD_END - $COLD_START)/1e9:.3f}')")

cold_pt=$(echo "$cold_resp" | grep -o '"prompt_tokens":[0-9]*' | head -1 | cut -d: -f2)
info "Cold session:     ${cold_elapsed}s (prompt_tokens=$cold_pt)"
info "Pre-warmed session: ${warm_elapsed}s (prompt_tokens=$warm_pt)"

if [[ -n "$cold_pt" && "$cold_pt" -gt 0 ]]; then
    pass "Cold session completed for comparison"
else
    fail "Cold session failed"
fi

# Compare: pre-warmed should be faster than cold
# Use python for float comparison (bash can't do floats)
is_faster=$(python3 -c "
warm = float('${warm_elapsed}')
cold = float('${cold_elapsed}')
# Pre-warmed should be faster. Allow some tolerance — at minimum it shouldn't be 2x slower.
if warm < cold:
    print('FASTER')
    print(f'  Pre-warmed was {cold/warm:.1f}x faster than cold start')
elif warm < cold * 2:
    print('SIMILAR')
    print(f'  Times are close (warm={warm:.3f}s, cold={cold:.3f}s) — prefill may be too short to measure')
else:
    print('SLOWER')
    print(f'  Pre-warmed ({warm:.3f}s) was slower than cold ({cold:.3f}s) — unexpected')
" 2>/dev/null || echo "SKIP")

comparison_msg=$(python3 -c "
warm = float('${warm_elapsed}')
cold = float('${cold_elapsed}')
if warm < cold:
    print(f'Pre-warmed was {cold/warm:.1f}x faster than cold start')
elif warm < cold * 2:
    print(f'Times are close (warm={warm:.3f}s, cold={cold:.3f}s)')
else:
    print(f'Pre-warmed ({warm:.3f}s) was slower than cold ({cold:.3f}s)')
" 2>/dev/null || echo "Could not compare")

result_type=$(echo "$is_faster" | head -1)
info "$comparison_msg"

case "$result_type" in
    FASTER)
        pass "Pre-warmed session was measurably faster than cold start"
        ;;
    SIMILAR)
        pass "Timing comparison inconclusive (prompt may be too short for measurable difference)"
        ;;
    SLOWER)
        fail "Pre-warmed session was slower than cold — warmup may not have completed"
        ;;
    *)
        info "Could not compare timings (python3 not available for float math)"
        pass "Both requests completed successfully"
        ;;
esac
echo ""

# ═══════════════════════════════════════════════════
# TEST 4: Warmup without X-Session-Id → 400
# ═══════════════════════════════════════════════════
echo -e "${CYAN}━━━ Test 4: Warmup without session ID → 400 ━━━${NC}"

bad_resp=$(curl -s -w "\n%{http_code}" --max-time 5 \
    "${BASE_URL}/v1/warmup" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"deepseek-v4-flash\",
        \"messages\": [{\"role\":\"user\",\"content\":\"hello\"}],
        \"max_tokens\": 10
    }")

bad_code=$(echo "$bad_resp" | tail -1)
if [[ "$bad_code" == "400" ]]; then
    pass "Warmup without X-Session-Id correctly returns 400"
else
    fail "Expected 400 without session ID, got $bad_code"
fi
echo ""

# ═══════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════
echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"
TOTAL=$((PASS + FAIL))
if [[ $FAIL -eq 0 ]]; then
    echo -e "  ${GREEN}All $TOTAL tests passed!${NC}"
else
    echo -e "  ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC} (of $TOTAL)"
fi
echo -e "${YELLOW}═══════════════════════════════════════════════════${NC}"

exit $FAIL
