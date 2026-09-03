#!/usr/bin/env bash
# CLI option smoke tests. Run from the repo root via `make test`.
# These do not exercise CUDA or tensor-parallel hardware.
set -uo pipefail

cd "$(dirname "$0")/.."

PASS=0
FAIL=0
LOG=$(mktemp)

ok()   { PASS=$((PASS+1)); echo "ok $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL $1"; }

assert_grep() {
    # $1 = name, $2 = pattern, $3 = file
    if grep -q -- "$2" "$3" 2>/dev/null; then ok "$1"; else
        fail "$1 (pattern not in $3)"
        echo "    --- content of $3 ---"
        head -20 "$3" | sed 's/^/    /'
    fi
}

assert_not_grep() {
    # $1 = name, $2 = pattern, $3 = file
    if grep -q -- "$2" "$3" 2>/dev/null; then
        fail "$1 (obsolete pattern found in $3)"
    else
        ok "$1"
    fi
}

# Binaries to check
BINS=(./ds4 ./ds4-server ./ds4-bench ./ds4-agent)
NAMES=(ds4 ds4-server ds4-bench ds4-agent)

# 1: each binary's --help mentions both flags.
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    if [ ! -x "$bin" ]; then
        fail "$name not built — skipping help check"
        continue
    fi
    "$bin" --help > "$LOG" 2>&1 || true
    if grep -q -- "--rocm" "$LOG"; then
        assert_grep "$name --help mentions --rocm" "--rocm" "$LOG"
        assert_not_grep "$name ROCm help omits --cuda-tensor-parallel" \
            "--cuda-tensor-parallel" "$LOG"
    else
        assert_grep "$name --help mentions --gpu-vram" "gpu-vram" "$LOG"
        assert_grep "$name --help mentions --gpu-devices" "gpu-devices" "$LOG"
        assert_grep "$name --help mentions --cuda-tensor-parallel" \
            "cuda-tensor-parallel" "$LOG"
    fi
    if [ "$name" != "ds4-bench" ]; then
        "$bin" --help runtime > "$LOG" 2>&1 || true
        assert_grep "$name --help runtime mentions --mtp" \
            "--mtp" "$LOG"
        assert_grep "$name --help runtime mentions --mtp-model" \
            "--mtp-model" "$LOG"
        assert_grep "$name --help runtime mentions --mtp-exact-sampling" \
            "mtp-exact-sampling" "$LOG"
        assert_not_grep "$name --help runtime omits --glm-mtp" \
            "--glm-mtp" "$LOG"
    else
        "$bin" --help runtime > "$LOG" 2>&1 || true
        assert_grep "$name --help runtime mentions --mtp-model" \
            "--mtp-model FILE" "$LOG"
        assert_grep "$name --help runtime mentions --dspark" "--dspark" "$LOG"
        assert_grep "$name --help runtime mentions --dspark-confidence" \
            "--dspark-confidence" "$LOG"
    fi
    if [ "$name" = "ds4" ]; then
        "$bin" --help distributed > "$LOG" 2>&1 || true
        assert_grep "$name --help distributed mentions --tensor-parallel-token-prefill" \
            "tensor-parallel-token-prefill" "$LOG"
        assert_not_grep "$name --help distributed omits old --tp spellings" "--tp-" "$LOG"
    fi
done

if [ -x ./ds4-eval ]; then
    ./ds4-eval --help runtime > "$LOG" 2>&1 || true
    assert_grep "ds4-eval --help runtime mentions --mtp-model" \
        "--mtp-model" "$LOG"
    ./ds4-eval --mtp-model /dev/null -m /dev/null --questions 1 \
        > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && ! grep -q "unknown option" "$LOG"; then
        ok "ds4-eval parses --mtp-model"
    else
        fail "ds4-eval rejected --mtp-model in option parsing"
    fi
    ./ds4-eval --mtp /dev/null -m /dev/null --questions 1 > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q "unknown option" "$LOG"; then
        ok "ds4-eval rejects obsolete --mtp FILE spelling"
    else
        fail "ds4-eval retained obsolete --mtp FILE spelling"
    fi
fi

# 1b: ds4-bench accepts the basic DSpark options before model loading.
if [ -x ./ds4-bench ]; then
    ./ds4-bench --dspark --dspark-confidence 0 --mtp-model /dev/null \
        -m /dev/null --prompt-file /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && ! grep -q "unknown option" "$LOG"; then
        ok "ds4-bench parses basic DSpark options"
    else
        fail "ds4-bench rejected basic DSpark options in option parsing"
    fi

    ./ds4-bench --dspark -m /dev/null --prompt-file /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 2 ] && grep -q -- \
        "--dspark requires --mtp-model FILE" "$LOG"; then
        ok "ds4-bench requires a DSpark support model"
    else
        fail "ds4-bench did not reject --dspark without --mtp-model"
    fi
fi

# 1c: exact DSpark sampling is parsed by every frontend that can run DSpark.
for i in 0 1 3; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    "$bin" --mtp-exact-sampling --mtp-model /dev/null -m /dev/null \
        > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && ! grep -q "unknown option" "$LOG"; then
        ok "$name parses --mtp-exact-sampling"
    else
        fail "$name rejected --mtp-exact-sampling in option parsing"
    fi
done

# Embedded MTP uses one family-neutral switch; external support has its own
# file option. The old GLM-specific spellings are intentionally rejected.
for i in 0 1 3; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    for flag in --mtp --mtp-timing; do
        "$bin" "$flag" -m /dev/null > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] && ! grep -q "unknown option" "$LOG"; then
            ok "$name parses $flag"
        else
            fail "$name rejected $flag in option parsing"
        fi
    done
    for flag in --glm-mtp --glm-mtp-timing; do
        "$bin" "$flag" -m /dev/null > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] && grep -q "unknown option" "$LOG"; then
            ok "$name rejects obsolete $flag"
        else
            fail "$name retained obsolete $flag"
        fi
    done
done

# The provisional DSpark-specific spelling was never released.
for i in 0 1 3; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    "$bin" --dspark-exact-sampling -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q "unknown option" "$LOG"; then
        ok "$name rejects provisional --dspark-exact-sampling"
    else
        fail "$name retained provisional --dspark-exact-sampling"
    fi
done

# 2: parser error on syntactically invalid value. For ds4-bench, we
# also pass --prompt-file /dev/null so it doesn't exit on the
# "specify exactly one of --prompt-file or --chat-prompt-file" check
# before the gpu-vram parser is reached.
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    if [ "$name" = "ds4-bench" ]; then
        "$bin" --gpu-vram abc -m /dev/null --prompt-file /dev/null > "$LOG" 2>&1
    else
        "$bin" --gpu-vram abc -m /dev/null > "$LOG" 2>&1
    fi
    rc=$?
    if [ $rc -eq 0 ]; then
        fail "$name --gpu-vram abc should exit non-zero (got 0)"
    else
        ok "$name --gpu-vram abc exits non-zero ($rc)"
    fi
    # Confirm the shared value parser was reached, not merely the binary's
    # unknown-option fallback.
    if grep -q -- "--gpu-vram: not a number" "$LOG" 2>/dev/null &&
       ! grep -q "unknown option" "$LOG" 2>/dev/null; then
        ok "$name --gpu-vram abc reaches shared parser"
    else
        fail "$name --gpu-vram abc did not reach shared parser"
        head -10 "$LOG" | sed 's/^/    /'
    fi
done

# 3: count mismatch.
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    if [ "$name" = "ds4-bench" ]; then
        "$bin" --gpu-vram 40,12 --gpu-devices 0 -m /dev/null \
            --prompt-file /dev/null > "$LOG" 2>&1
    else
        "$bin" --gpu-vram 40,12 --gpu-devices 0 -m /dev/null > "$LOG" 2>&1
    fi
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q -- "--gpu-devices count (1) does not match --gpu-vram count (2)" "$LOG" &&
       ! grep -q "unknown option" "$LOG"; then
        ok "$name count-mismatch reaches shared parser ($rc)"
    else
        fail "$name count-mismatch did not reach shared parser"
        head -10 "$LOG" | sed 's/^/    /'
    fi
done

# 4: --cuda --help still works (the flag alone shouldn't break parsing).
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    [ -x "$bin" ] || continue
    "$bin" --cuda --help > "$LOG" 2>&1 || true
    # Servers may print a usage banner; check help still surfaced.
    if grep -qE "Usage:|usage:|--help" "$LOG"; then
        ok "$name --cuda --help still prints help"
    else
        fail "$name --cuda --help did not print help text"
    fi
done

# 5: --gpu-vram 0 short-circuit. We use ds4 (CLI) specifically because
# it produces predictable stdout/stderr.
if [ -x ./ds4 ]; then
    ./ds4 --gpu-vram 0 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        ok "ds4 --gpu-vram 0 exits non-zero (expected: model-load fail)"
    else
        fail "ds4 --gpu-vram 0 returned 0 — unexpected"
    fi
    # The layout line must NOT appear (short-circuit happens before).
    if grep -q "GPU config:" "$LOG"; then
        fail "ds4 --gpu-vram 0 should NOT print GPU layout line"
        head -10 "$LOG" | sed 's/^/    /'
    else
        ok "ds4 --gpu-vram 0 does not print GPU layout (short-circuit reached)"
    fi
fi

# 6: tensor parallelism reuses the distributed role and address options, but
# owns the split and therefore rejects --layers.
if [ -x ./ds4 ]; then
    ./ds4 --metal --tensor-parallel --role coordinator --listen 127.0.0.1 9911 \
        --layers 0:1 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q "always uses one 50/50 worker" "$LOG"; then
        ok "tensor parallel rejects explicit layer slices"
    else
        fail "tensor parallel accepted --layers or returned the wrong error"
    fi

    ./ds4 --tensor-parallel --role worker -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q "requires --coordinator HOST PORT" "$LOG"; then
        ok "tensor-parallel worker requires coordinator address"
    else
        fail "tensor-parallel worker returned the wrong missing-address error"
    fi

    ./ds4 --metal --tensor-parallel --role coordinator --listen 127.0.0.1 9911 \
        --transport tcp --tensor-parallel-token-prefill --debug-hash 2 \
        --rdma-device rdma-test --rdma-gid-index 0 \
        --inspect -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -qE "model file is too small|another ds4 process is already running" "$LOG" &&
       ! grep -q "requires --layers" "$LOG"; then
        ok "tensor-parallel common options reach model loading"
    else
        fail "tensor-parallel common options did not reach model loading"
    fi

    for old_arg in \
        "--tp-coordinator 9911" \
        "--tp-lead 9911" \
        "--tp-coordinator-host 127.0.0.1" \
        "--tp-lead-host 127.0.0.1" \
        "--tp-worker 127.0.0.1 9911" \
        "--tp-transport tcp" \
        "--tp-debug-hash 2" \
        "--tp-token-prefill"
    do
        # Word splitting is intentional: each item contains one old option
        # and its former arguments.
        ./ds4 $old_arg -m /dev/null > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] && grep -q "unknown option" "$LOG"; then
            ok "obsolete ${old_arg%% *} is rejected"
        else
            fail "obsolete ${old_arg%% *} was not rejected"
        fi
    done
fi

# 6b: ds4-agent accepts the same TP coordinator flags as ds4. The worker role
# remains a serving mode and is intentionally handled by ./ds4.
if [ -x ./ds4-agent ]; then
    if ./ds4-agent --help 2>&1 | grep -q -- "--rocm"; then
        AGENT_GPU_FLAG=--rocm
        AGENT_GPU_BACKEND=rocm
    else
        AGENT_GPU_FLAG=--cuda
        AGENT_GPU_BACKEND=cuda
    fi
    ./ds4-agent "$AGENT_GPU_FLAG" --non-interactive -p test \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && ! grep -q "unknown option" "$LOG"; then
        ok "ds4-agent parses advertised $AGENT_GPU_FLAG"
    else
        fail "ds4-agent rejected advertised $AGENT_GPU_FLAG"
    fi
    ./ds4-agent --backend "$AGENT_GPU_BACKEND" --non-interactive -p test \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       ! grep -qE "unknown option|invalid backend" "$LOG"; then
        ok "ds4-agent parses --backend $AGENT_GPU_BACKEND"
    else
        fail "ds4-agent rejected --backend $AGENT_GPU_BACKEND"
    fi

    ./ds4-agent --metal --tensor-parallel --role coordinator \
        --listen 127.0.0.1 9911 --transport tcp \
        --tensor-parallel-token-prefill --debug-hash 2 \
        --rdma-device rdma-test --rdma-gid-index 0 \
        --non-interactive -p test -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -qE "model file is too small|another ds4 process is already running" "$LOG" &&
       ! grep -q "unknown option" "$LOG"; then
        ok "ds4-agent tensor-parallel coordinator options reach model loading"
    else
        fail "ds4-agent tensor-parallel coordinator options did not reach model loading"
        head -10 "$LOG" | sed 's/^/    /'
    fi

    ./ds4-agent --metal --tensor-parallel --role worker \
        --coordinator 127.0.0.1 9911 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q "serving mode; start workers with ./ds4" "$LOG"; then
        ok "ds4-agent tensor-parallel worker directs users to ds4"
    else
        fail "ds4-agent tensor-parallel worker returned the wrong error"
        head -10 "$LOG" | sed 's/^/    /'
    fi

    PROMPT_FILE=$(mktemp)
    printf 'prompt file parser smoke\n' > "$PROMPT_FILE"
    ./ds4-agent --non-interactive --prompt-file "$PROMPT_FILE" \
        --raw-prompt -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -qE "model file is too small|another ds4 process is already running" "$LOG" &&
       ! grep -q "unknown option" "$LOG"; then
        ok "ds4-agent --prompt-file reaches model loading"
    else
        fail "ds4-agent --prompt-file did not reach model loading"
        head -10 "$LOG" | sed 's/^/    /'
    fi

    ./ds4-agent --non-interactive -p inline --prompt-file "$PROMPT_FILE" \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q "specify only one" "$LOG"; then
        ok "ds4-agent rejects two initial prompt sources"
    else
        fail "ds4-agent accepted two initial prompt sources"
    fi
    rm -f "$PROMPT_FILE"
fi

# 7: --gpu-vram 40,12 layout line.
if [ -x ./ds4 ]; then
    ./ds4 --gpu-vram 40,12 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if grep -q "GPU config: 2 devices \[0,1\] requested, budgets 40,12 GB" "$LOG"; then
        ok "ds4 --gpu-vram 40,12 prints expected layout line"
    else
        fail "ds4 --gpu-vram 40,12 missing or malformed layout line"
        head -10 "$LOG" | sed 's/^/    /'
    fi
fi

rm -f "$LOG"

echo ""
echo "test_gpu_args_cli: PASS=$PASS FAIL=$FAIL"
if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
