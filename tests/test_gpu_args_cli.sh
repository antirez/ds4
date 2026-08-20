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

# Binaries exposing the shared CUDA placement parser.
BINS=(./ds4 ./ds4-server ./ds4-bench ./ds4-agent)
NAMES=(ds4 ds4-server ds4-bench ds4-agent)

# Every executable that can coordinate network expert/tensor parallelism.
TP_BINS=(./ds4 ./ds4-server ./ds4-bench ./ds4-agent ./ds4-eval)
TP_NAMES=(ds4 ds4-server ds4-bench ds4-agent ds4-eval)

# Network-parallel validation is backend-specific. CUDA accepts two/four-rank
# EP or TP and eight-rank EP over NCCL; Metal accepts a two-rank TCP topology.
if [ "$(uname -s)" = "Darwin" ]; then
    NETWORK_BACKEND=--metal
    NETWORK_WORLD=2
    NETWORK_TRANSPORT=tcp
else
    NETWORK_BACKEND=--cuda
    NETWORK_WORLD=4
    NETWORK_TRANSPORT=nccl
fi

# 1: each binary's --help mentions both flags.
for i in "${!BINS[@]}"; do
    name=${NAMES[$i]}; bin=${BINS[$i]}
    if [ ! -x "$bin" ]; then
        fail "$name not built — skipping help check"
        continue
    fi
    "$bin" --help > "$LOG" 2>&1 || true
    assert_grep "$name --help mentions --gpu-vram" "gpu-vram" "$LOG"
    assert_grep "$name --help mentions --gpu-devices" "gpu-devices" "$LOG"
    assert_grep "$name --help mentions --cuda-tensor-parallel" "cuda-tensor-parallel" "$LOG"
done

for i in "${!TP_BINS[@]}"; do
    name=${TP_NAMES[$i]}; bin=${TP_BINS[$i]}
    if [ ! -x "$bin" ]; then
        fail "$name not built — skipping network help check"
        continue
    fi
    "$bin" --help distributed > "$LOG" 2>&1 || true
    assert_grep "$name --help distributed mentions --expert-parallel" \
        "--expert-parallel" "$LOG"
    assert_grep "$name --help distributed mentions --tensor-parallel" \
        "--tensor-parallel" "$LOG"
    assert_grep "$name --help distributed mentions --tensor-parallel-world" \
        "--tensor-parallel-world" "$LOG"
    assert_grep "$name --help distributed mentions --tensor-parallel-rank" \
        "--tensor-parallel-rank" "$LOG"
    assert_grep "$name --help distributed mentions --tensor-parallel-token-prefill" \
        "tensor-parallel-token-prefill" "$LOG"
    assert_not_grep "$name --help distributed omits old --tp spellings" "--tp-" "$LOG"
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
    if [ $rc -ne 0 ] && grep -q "owns model placement; omit --layers" "$LOG"; then
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
       grep -qE "model file is too small|another ds4 process is already running|requires the CUDA backend on Linux" "$LOG" &&
       ! grep -q "requires --layers" "$LOG"; then
        ok "tensor-parallel common options reach model loading"
    else
        fail "tensor-parallel common options did not reach model loading"
    fi

    ./ds4 --expert-parallel --tensor-parallel -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q -- "--expert-parallel and --tensor-parallel are exclusive" "$LOG"; then
        ok "expert and tensor modes are mutually exclusive"
    else
        fail "expert/tensor mode conflict was not rejected"
    fi

    if [ "$(uname -s)" = "Linux" ]; then
        ./ds4 --cuda --expert-parallel --glm-mtp \
            --role coordinator --listen 127.0.0.1 9911 \
            -m /dev/null > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] &&
           grep -q "CUDA network parallelism does not yet support --glm-mtp" "$LOG"; then
            ok "CUDA network parallel rejects GLM MTP"
        else
            fail "CUDA network parallel accepted GLM MTP"
        fi

        ./ds4 --cuda --expert-parallel --mtp /dev/null --dspark \
            --role coordinator --listen 127.0.0.1 9911 \
            -m /dev/null > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] &&
           grep -q "does not yet support MTP/DSpark speculative drafting" "$LOG"; then
            ok "CUDA network parallel rejects external MTP/DSpark"
        else
            fail "CUDA network parallel accepted external MTP/DSpark"
        fi

        ./ds4 --cuda --expert-parallel \
            --dir-steering-file /dev/null \
            --role coordinator --listen 127.0.0.1 9911 \
            -m /dev/null > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] &&
           grep -q "does not yet support directional steering" "$LOG"; then
            ok "CUDA network parallel rejects directional steering"
        else
            fail "CUDA network parallel accepted directional steering"
        fi

        ./ds4-bench --cuda --expert-parallel --transport tcp \
            --role coordinator --listen 127.0.0.1 9911 \
            --prompt-file /dev/null -m /dev/null > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] &&
           grep -q "CUDA network parallelism requires --transport nccl" "$LOG"; then
            ok "ds4-bench validates CUDA network transport before model loading"
        else
            fail "ds4-bench skipped CUDA network transport validation"
        fi
    fi

    ./ds4 --expert-parallel --role worker \
        --tensor-parallel-world 3 --coordinator 127.0.0.1 9911 \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] && grep -q -- "--tensor-parallel-world must be 2, 4, or 8" "$LOG"; then
        ok "network parallel rejects a three-rank world"
    else
        fail "network parallel accepted a three-rank world"
    fi

    ./ds4 --expert-parallel --role worker \
        --tensor-parallel-world 4 --coordinator 127.0.0.1 9911 \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q "4-rank workers require an explicit --tensor-parallel-rank in 1..3" "$LOG"; then
        ok "four-rank workers require an explicit rank"
    else
        fail "four-rank worker accepted an implicit rank"
    fi

    ./ds4 --expert-parallel --role worker \
        --tensor-parallel-world 8 --coordinator 127.0.0.1 9911 \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q "8-rank workers require an explicit --tensor-parallel-rank in 1..7" "$LOG"; then
        ok "eight-rank workers require an explicit rank"
    else
        fail "eight-rank worker accepted an implicit rank"
    fi

    if [ "$(uname -s)" = "Linux" ]; then
        ./ds4 --cuda --tensor-parallel --tensor-parallel-world 8 \
            --role coordinator --listen 127.0.0.1 9911 \
            --transport nccl -m /dev/null > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] && grep -q "TP8 unsupported; use EP8" "$LOG"; then
            ok "full TP8 is rejected with EP8 guidance"
        else
            fail "full TP8 returned the wrong validation result"
        fi
    fi

    ./ds4 --expert-parallel --role worker \
        --tensor-parallel-world 2 --tensor-parallel-rank 2 \
        --coordinator 127.0.0.1 9911 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q "worker --tensor-parallel-rank 2 is outside 1..1" "$LOG"; then
        ok "two-rank worker rejects a rank outside its world"
    else
        fail "two-rank worker accepted rank 2"
    fi

    for old_arg in \
        "--tp-coordinator 9911" \
        "--tp-lead 9911" \
        "--tp-coordinator-host 127.0.0.1" \
        "--tp-lead-host 127.0.0.1" \
        "--tp-worker 127.0.0.1 9911" \
        "--tp-transport tcp" \
        "--tp-debug-hash 2" \
        "--tp-token-prefill" \
        "--tp-world 2" \
        "--tp-rank 1"
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

# 6b: every long-lived frontend accepts network leader/worker options and
# reaches engine loading. These use /dev/null, so no socket or GPU work starts.
FRONTEND_BINS=(./ds4-server ./ds4-agent ./ds4-eval)
FRONTEND_NAMES=(ds4-server ds4-agent ds4-eval)
for i in "${!FRONTEND_BINS[@]}"; do
    name=${FRONTEND_NAMES[$i]}; bin=${FRONTEND_BINS[$i]}
    [ -x "$bin" ] || continue

    "$bin" "$NETWORK_BACKEND" --ctx 128 --tensor-parallel --role worker \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q "requires --coordinator HOST PORT" "$LOG"; then
        ok "$name network worker requires coordinator address"
    else
        fail "$name returned the wrong missing worker-address error"
    fi

    "$bin" "$NETWORK_BACKEND" --ctx 128 --tensor-parallel \
        --tensor-parallel-world "$NETWORK_WORLD" --tensor-parallel-rank 1 \
        --role worker --coordinator 127.0.0.1 9911 \
        --transport "$NETWORK_TRANSPORT" \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q "model file is too small to be GGUF" "$LOG" &&
       ! grep -qE "unknown option|start workers with ./ds4" "$LOG"; then
        ok "$name network worker reaches engine loading"
    else
        fail "$name network worker did not reach engine loading"
        head -10 "$LOG" | sed 's/^/    /'
    fi

    "$bin" "$NETWORK_BACKEND" --ctx 128 \
        --expert-parallel --tensor-parallel \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q -- "--expert-parallel and --tensor-parallel are exclusive" "$LOG"; then
        ok "$name rejects conflicting network modes"
    else
        fail "$name accepted conflicting network modes"
    fi

    if [ "$(uname -s)" = "Linux" ]; then
        "$bin" --cuda --ctx 128 --expert-parallel --transport tcp \
            --role coordinator --listen 127.0.0.1 9911 \
            -m /dev/null > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] &&
           grep -q "CUDA network parallelism requires --transport nccl" "$LOG"; then
            ok "$name validates CUDA network transport before model loading"
        else
            fail "$name skipped CUDA network transport validation"
        fi
    fi
done

if [ -x ./ds4-eval ]; then
    ./ds4-eval "$NETWORK_BACKEND" --expert-parallel --role coordinator \
        --listen 127.0.0.1 9911 -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q "network parallelism requires an explicit --ctx" "$LOG"; then
        ok "ds4-eval network mode requires explicit context capacity"
    else
        fail "ds4-eval accepted automatic context sizing in network mode"
    fi
fi

# 6c: the official-continuation scorer uses the same network parser and
# validates the rank topology before loading a model. It keeps its historical
# positional MODEL/manifest/output interface, so /dev/null fills those slots.
QUALITY_SCORER=./gguf-tools/quality-testing/score_official
if [ ! -x "$QUALITY_SCORER" ]; then
    fail "score_official not built — skipping network parser checks"
else
    "$QUALITY_SCORER" > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q -- "--expert-parallel" "$LOG" &&
       grep -q -- "--tensor-parallel" "$LOG"; then
        ok "score_official usage advertises network EP/TP"
    else
        fail "score_official usage omits network EP/TP"
    fi

    "$QUALITY_SCORER" /dev/null /dev/null /dev/null 1024 \
        --tensor-parallel --role worker > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q "requires --coordinator HOST PORT" "$LOG"; then
        ok "score_official network worker requires coordinator address"
    else
        fail "score_official returned the wrong missing worker-address error"
    fi

    "$QUALITY_SCORER" /dev/null /dev/null /dev/null 1024 \
        --expert-parallel --tensor-parallel > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q -- "--expert-parallel and --tensor-parallel are exclusive" "$LOG"; then
        ok "score_official rejects conflicting network modes"
    else
        fail "score_official accepted conflicting network modes"
    fi

    if [ "$(uname -s)" = "Linux" ]; then
        "$QUALITY_SCORER" /dev/null /dev/null /dev/null 1024 \
            --expert-parallel --transport tcp \
            --role coordinator --listen 127.0.0.1 9911 > "$LOG" 2>&1
        rc=$?
        if [ $rc -ne 0 ] &&
           grep -q "CUDA network parallelism requires --transport nccl" "$LOG"; then
            ok "score_official validates CUDA network transport before model loading"
        else
            fail "score_official skipped CUDA network transport validation"
        fi
    fi

    "$QUALITY_SCORER" /dev/null /dev/null /dev/null 1024 \
        --tensor-parallel --tensor-parallel-world "$NETWORK_WORLD" \
        --tensor-parallel-rank 1 --role worker \
        --coordinator 127.0.0.1 9911 \
        --transport "$NETWORK_TRANSPORT" > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q "model file is too small to be GGUF" "$LOG" &&
       ! grep -qE "unknown option|start pipeline workers with ./ds4" "$LOG"; then
        ok "score_official network worker reaches engine loading"
    else
        fail "score_official network worker did not reach engine loading"
        head -10 "$LOG" | sed 's/^/    /'
    fi
fi

if [ -x ./ds4-server ]; then
    ./ds4-server "$NETWORK_BACKEND" --ctx 128 --batched-session 2 \
        --tensor-parallel --tensor-parallel-world "$NETWORK_WORLD" \
        --tensor-parallel-rank 1 --role worker \
        --coordinator 127.0.0.1 9911 --transport "$NETWORK_TRANSPORT" \
        -m /dev/null > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] &&
       grep -q "model file is too small to be GGUF" "$LOG" &&
       ! grep -q "batched-session" "$LOG"; then
        ok "ds4-server accepts native batching in network worker mode"
    else
        fail "ds4-server rejected native batching in network mode"
    fi
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
