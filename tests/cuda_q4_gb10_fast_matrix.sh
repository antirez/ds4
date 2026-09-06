#!/bin/sh
# Isolate the numerical effect of each AProjQ4 GB10 fast-path component.
#
# Usage:
#   tests/cuda_q4_gb10_fast_matrix.sh MODEL MANIFEST [OUTPUT_DIR]
#
# Every non-oracle arm is run through score_official.  The script deliberately
# uses separate processes:
# these switches are cached during CUDA/MMQ initialization and cannot be
# compared safely in one process.

set -eu

usage() {
    cat >&2 <<'EOF'
usage: tests/cuda_q4_gb10_fast_matrix.sh MODEL MANIFEST [OUTPUT_DIR]

Environment:
  DS4_BIN                         ds4 executable (default: ./ds4)
  DS4_CUDA_Q4_MATRIX_SCORER       score_official executable
  DS4_CUDA_Q4_MATRIX_CTX          context size (default: 4096)
  DS4_CUDA_Q4_MATRIX_TOKENS       smoke continuation length (default: 32)
  DS4_CUDA_Q4_MATRIX_TOP_K        smoke top-logprobs (default: 128)
  DS4_CUDA_Q4_MATRIX_PROMPT       deterministic smoke prompt
  DS4_CUDA_Q4_MATRIX_SSD_STREAMING  0 or 1 (default: 0)
  DS4_CUDA_Q4_MATRIX_SSD_CACHE    expert count or NGB; required with streaming
  DS4_CUDA_Q4_MATRIX_SSD_PRELOAD  optional expert preload count
  DS4_CUDA_Q4_MATRIX_DECODE_GRAPHS default, 0, or 1
  DS4_CUDA_Q4_MATRIX_SKIP_PARITY  0 or 1 (default: 0; skip is incomplete QA)

The output directory must not already contain matrix results.  Exit status is
zero only when tensor-oracle coverage passes, scorer rows match, and every
recorded smoke result is byte-identical to the umbrella rollback.  A numerical
difference is preserved in *.diff and causes a nonzero exit after all arms
have completed.
EOF
}

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    usage
    exit 2
fi

MODEL=$1
MANIFEST=$2
OUT_DIR=${3:-}
DS4_BIN=${DS4_BIN:-./ds4}
SCORER=${DS4_CUDA_Q4_MATRIX_SCORER:-gguf-tools/quality-testing/score_official}
CTX=${DS4_CUDA_Q4_MATRIX_CTX:-4096}
TOKENS=${DS4_CUDA_Q4_MATRIX_TOKENS:-32}
TOP_K=${DS4_CUDA_Q4_MATRIX_TOP_K:-128}
PROMPT=${DS4_CUDA_Q4_MATRIX_PROMPT:-Write a complete Python quicksort function with comments.}
SSD_STREAMING=${DS4_CUDA_Q4_MATRIX_SSD_STREAMING:-0}
SSD_CACHE=${DS4_CUDA_Q4_MATRIX_SSD_CACHE:-}
SSD_PRELOAD=${DS4_CUDA_Q4_MATRIX_SSD_PRELOAD:-}
DECODE_GRAPHS=${DS4_CUDA_Q4_MATRIX_DECODE_GRAPHS:-default}
SKIP_PARITY=${DS4_CUDA_Q4_MATRIX_SKIP_PARITY:-0}

case "$CTX:$TOKENS:$TOP_K" in
    *[!0-9:]*|:*|*::*|*:|0:*|*:0:*|*:0)
        echo "q4-gb10-matrix: ctx, tokens, and top-k must be positive integers" >&2
        exit 2
        ;;
esac
if [ "$TOP_K" -gt 128 ]; then
    echo "q4-gb10-matrix: top-k cannot exceed the ds4 dump limit (128)" >&2
    exit 2
fi
case "$SSD_STREAMING" in
    0|1) ;;
    *) echo "q4-gb10-matrix: SSD_STREAMING must be 0 or 1" >&2; exit 2 ;;
esac
case "$SKIP_PARITY" in
    0|1) ;;
    *) echo "q4-gb10-matrix: SKIP_PARITY must be 0 or 1" >&2; exit 2 ;;
esac
case "$DECODE_GRAPHS" in
    default|0|1) ;;
    *) echo "q4-gb10-matrix: DECODE_GRAPHS must be default, 0, or 1" >&2; exit 2 ;;
esac
if [ "$SSD_STREAMING" = 1 ] && [ -z "$SSD_CACHE" ]; then
    echo "q4-gb10-matrix: set DS4_CUDA_Q4_MATRIX_SSD_CACHE when streaming" >&2
    exit 2
fi
if [ "$SSD_STREAMING" = 0 ] && { [ -n "$SSD_CACHE" ] || [ -n "$SSD_PRELOAD" ]; }; then
    echo "q4-gb10-matrix: SSD cache/preload requires SSD_STREAMING=1" >&2
    exit 2
fi
if [ ! -x "$DS4_BIN" ]; then
    echo "q4-gb10-matrix: ds4 executable not found: $DS4_BIN" >&2
    exit 2
fi
if [ ! -r "$MODEL" ]; then
    echo "q4-gb10-matrix: model is not readable: $MODEL" >&2
    exit 2
fi
if [ ! -r "$MANIFEST" ]; then
    echo "q4-gb10-matrix: manifest is not readable: $MANIFEST" >&2
    exit 2
fi
if [ ! -x "$SCORER" ]; then
    echo "q4-gb10-matrix: scorer not found: $SCORER" >&2
    echo "build it with: make gguf-tools/quality-testing/score_official" >&2
    exit 2
fi

if [ -z "$OUT_DIR" ]; then
    OUT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ds4-q4-gb10-matrix.XXXXXX")
else
    mkdir -p "$OUT_DIR"
fi
if [ -e "$OUT_DIR/umbrella_control.log" ] ||
   [ -e "$OUT_DIR/umbrella_control.json" ] ||
   [ -e "$OUT_DIR/umbrella_control.tsv" ]; then
    echo "q4-gb10-matrix: output directory already contains matrix results: $OUT_DIR" >&2
    exit 2
fi

case "$DECODE_GRAPHS" in
    default) GRAPH_ENV=; GRAPH_LOG_ENV= ;;
    0) GRAPH_ENV=DS4_CUDA_DECODE_GRAPHS=0; GRAPH_LOG_ENV= ;;
    1) GRAPH_ENV=DS4_CUDA_DECODE_GRAPHS=1; GRAPH_LOG_ENV=DS4_CUDA_DECODE_GRAPH_LOG=1 ;;
esac

# Strip every selector that could leak from the caller and turn an apparently
# isolated arm into a compound experiment.  Arm-specific assignments follow
# these -u options.
clean_env() {
    env \
        -u DS4_CUDA_MMQ \
        -u DS4_CUDA_DISABLE_Q4_DENSE_PAIR \
        -u DS4_CUDA_NO_Q4_GB10_FAST \
        -u DS4_CUDA_NO_Q4_DENSE_SCRATCH \
        -u DS4_CUDA_NO_Q4_GROUPED_ATTN_A \
        -u DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH \
        -u DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH \
        -u DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_BATCH \
        -u DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_PREFILL \
        -u DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL \
        -u DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_PREFILL \
        -u DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_SINGLE_GRID \
        -u DS4_CUDA_DISABLE_Q4_GROUPED_ATTN_A_SINGLE_GRID \
        -u DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_SINGLE_GRID \
        -u DS4_CUDA_NO_Q4_GROUPED_ATTN_A_Q81 \
        -u DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_Q81 \
        -u DS4_CUDA_Q4_GROUPED_ATTN_A_ORACLE \
        -u DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE \
        -u DS4_CUDA_ENABLE_Q4_ATTN_OUT_HC_FUSE \
        -u DS4_CUDA_Q4_ATTN_OUT_HC_Q8K_EXPERIMENT \
        -u DS4_CUDA_Q4_ATTN_OUT_HC_ORACLE \
        -u DS4_CUDA_NO_Q4_K1024_PERSISTENT \
        -u DS4_CUDA_ENABLE_Q4_K1024_PERSISTENT \
        -u DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT \
        -u DS4_CUDA_Q4_MMQ_16WARP \
        -u DS4_CUDA_NO_Q4_MMQ_16WARP \
        -u DS4_CUDA_REQUIRE_Q4_MMQ_16WARP \
        -u DS4_CUDA_DECODE_GRAPHS \
        -u DS4_CUDA_DECODE_GRAPH_LOG \
        "$@"
}

check_gb10_log() {
    log=$1
    cuda_count=$(grep -c 'ds4: CUDA backend initialized on ' "$log" || true)
    if [ "$cuda_count" -ne 1 ] ||
       ! grep -Eq 'ds4: CUDA backend initialized on .*\(sm_121\) dev=' "$log"; then
        echo "q4-gb10-matrix: $log did not initialize exactly one sm_121 CUDA GPU" >&2
        return 1
    fi
}

check_graph_log() {
    arm=$1
    log=$2
    case "$arm" in
        grouped_oracle|hc_oracle)
            if ! grep -q 'decode graph capture disabled for Q4 attention oracle' "$log"; then
                echo "q4-gb10-matrix: $arm did not prove oracle graph exclusion" >&2
                return 1
            fi
            ;;
        *)
            case "$DECODE_GRAPHS" in
                0)
                    if ! grep -q 'decode graph capture disabled' "$log"; then
                        echo "q4-gb10-matrix: $arm did not prove graphs-off dispatch" >&2
                        return 1
                    fi
                    ;;
                1)
                    if ! grep -q 'ds4: decode graph captured ' "$log"; then
                        echo "q4-gb10-matrix: $arm had zero decode-graph captures" >&2
                        return 1
                    fi
                    if grep -Eq 'decode graph (capture|instantiate|first launch|replay) failed' "$log"; then
                        echo "q4-gb10-matrix: $arm reported a decode-graph failure" >&2
                        return 1
                    fi
                    ;;
            esac
            ;;
    esac
}

run_smoke() {
    arm=$1
    shift
    json="$OUT_DIR/$arm.json"
    stdout="$OUT_DIR/$arm.stdout"
    log="$OUT_DIR/$arm.log"
    echo "q4-gb10-matrix: smoke arm=$arm"

    if [ "$SSD_STREAMING" = 1 ]; then
        if [ -n "$SSD_PRELOAD" ]; then
            clean_env ${GRAPH_ENV:+"$GRAPH_ENV"} ${GRAPH_LOG_ENV:+"$GRAPH_LOG_ENV"} \
                "$@" "$DS4_BIN" \
                --cuda -m "$MODEL" --ctx "$CTX" --tokens "$TOKENS" \
                --nothink --temp 0 --dump-logprobs "$json" \
                --logprobs-top-k "$TOP_K" --ssd-streaming \
                --ssd-streaming-cache-experts "$SSD_CACHE" \
                --ssd-streaming-preload-experts "$SSD_PRELOAD" \
                -p "$PROMPT" >"$stdout" 2>"$log"
        else
            clean_env ${GRAPH_ENV:+"$GRAPH_ENV"} ${GRAPH_LOG_ENV:+"$GRAPH_LOG_ENV"} \
                "$@" "$DS4_BIN" \
                --cuda -m "$MODEL" --ctx "$CTX" --tokens "$TOKENS" \
                --nothink --temp 0 --dump-logprobs "$json" \
                --logprobs-top-k "$TOP_K" --ssd-streaming \
                --ssd-streaming-cache-experts "$SSD_CACHE" \
                -p "$PROMPT" >"$stdout" 2>"$log"
        fi
    else
        clean_env ${GRAPH_ENV:+"$GRAPH_ENV"} ${GRAPH_LOG_ENV:+"$GRAPH_LOG_ENV"} \
            "$@" "$DS4_BIN" \
            --cuda -m "$MODEL" --ctx "$CTX" --tokens "$TOKENS" \
            --nothink --temp 0 --dump-logprobs "$json" \
            --logprobs-top-k "$TOP_K" -p "$PROMPT" \
            >"$stdout" 2>"$log"
    fi
    check_gb10_log "$log"
    check_graph_log "$arm" "$log"
    if [ ! -s "$json" ]; then
        echo "q4-gb10-matrix: arm $arm produced no logprob dump" >&2
        return 1
    fi
}

run_score() {
    arm=$1
    shift
    tsv="$OUT_DIR/$arm.tsv"
    log="$OUT_DIR/$arm.score.log"
    echo "q4-gb10-matrix: quality arm=$arm"

    if [ "$SSD_STREAMING" = 1 ]; then
        if [ -n "$SSD_PRELOAD" ]; then
            clean_env ${GRAPH_ENV:+"$GRAPH_ENV"} ${GRAPH_LOG_ENV:+"$GRAPH_LOG_ENV"} \
                "$@" "$SCORER" \
                "$MODEL" "$MANIFEST" "$tsv" "$CTX" --ssd-streaming \
                --ssd-streaming-cache-experts "$SSD_CACHE" \
                --ssd-streaming-preload-experts "$SSD_PRELOAD" \
                >"$OUT_DIR/$arm.score.stdout" 2>"$log"
        else
            clean_env ${GRAPH_ENV:+"$GRAPH_ENV"} ${GRAPH_LOG_ENV:+"$GRAPH_LOG_ENV"} \
                "$@" "$SCORER" \
                "$MODEL" "$MANIFEST" "$tsv" "$CTX" --ssd-streaming \
                --ssd-streaming-cache-experts "$SSD_CACHE" \
                >"$OUT_DIR/$arm.score.stdout" 2>"$log"
        fi
    else
        clean_env ${GRAPH_ENV:+"$GRAPH_ENV"} ${GRAPH_LOG_ENV:+"$GRAPH_LOG_ENV"} \
            "$@" "$SCORER" \
            "$MODEL" "$MANIFEST" "$tsv" "$CTX" \
            >"$OUT_DIR/$arm.score.stdout" 2>"$log"
    fi
    check_gb10_log "$log"
    check_graph_log "$arm" "$log"
    if [ ! -s "$tsv" ]; then
        echo "q4-gb10-matrix: quality arm $arm produced no TSV" >&2
        return 1
    fi
}

field() {
    printf '%s\n' "$1" | tr ' ' '\n' | awk -F= -v key="$2" '
        $1 == key { gsub(/[^0-9].*$/, "", $2); print $2; exit }
    '
}

check_grouped_oracle() {
    log=$1
    line=$(grep 'ds4: CUDA Q4 grouped attention-A oracle:' "$log" | tail -n 1 || true)
    calls=$(field "$line" calls)
    mismatches=$(field "$line" mismatches)
    skips=$(field "$line" skips)
    if [ -z "$line" ] || [ -z "$calls" ] || [ "$calls" -le 0 ] ||
       [ "$skips" != 0 ]; then
        echo "q4-gb10-matrix: grouped oracle coverage failed: ${line:-missing summary}" >&2
        return 1
    fi
    if [ "$mismatches" != 0 ]; then
        echo "q4-gb10-matrix: grouped oracle found mismatches: $line" >&2
        oracle_mismatch_arms="$oracle_mismatch_arms grouped"
    fi
}

check_hc_oracle() {
    log=$1
    line=$(grep 'ds4: CUDA Q4 attention-output/HC oracle:' "$log" | tail -n 1 || true)
    calls=$(field "$line" calls)
    mismatches=$(field "$line" epilogue_mismatches)
    skips=$(field "$line" skips)
    if [ -z "$line" ] || [ -z "$calls" ] || [ "$calls" -le 0 ] ||
       [ "$skips" != 0 ]; then
        echo "q4-gb10-matrix: HC epilogue oracle coverage failed: ${line:-missing summary}" >&2
        return 1
    fi
    if [ "$mismatches" != 0 ]; then
        echo "q4-gb10-matrix: HC epilogue oracle found mismatches: $line" >&2
        oracle_mismatch_arms="$oracle_mismatch_arms hc_epilogue"
    fi
}

LOCAL_ROLLBACK="DS4_CUDA_NO_Q4_DENSE_SCRATCH=1
DS4_CUDA_NO_Q4_GROUPED_ATTN_A=1
DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1
DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL=1
DS4_CUDA_NO_Q4_GROUPED_ATTN_A_Q81=1
DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE=1
DS4_CUDA_NO_Q4_K1024_PERSISTENT=1"

# Do not stop at the first numerical difference: completing all arms is what
# identifies a single culprit versus an interaction.  Execution/coverage
# failures still stop immediately because subsequent comparisons would lie.
if [ "$SKIP_PARITY" = 0 ]; then
    echo "q4-gb10-matrix: synthetic MMQ parity"
    clean_env make test-mmq-parity-cuda CUDA_ARCH=sm_121 \
        >"$OUT_DIR/mmq-parity.log" 2>&1
else
    echo "q4-gb10-matrix: WARNING synthetic parity skipped (result is incomplete)" >&2
fi

# `set -- $LOCAL_ROLLBACK` intentionally splits the newline-delimited list
# into environment assignments. Values and names contain no shell metacharacters.
set -- $LOCAL_ROLLBACK
run_smoke umbrella_control DS4_CUDA_NO_Q4_GB10_FAST=1
run_smoke local_control "$@"
run_smoke scratch_only \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL=1 \
    DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1
run_smoke grouped_only \
    DS4_CUDA_NO_Q4_DENSE_SCRATCH=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1 \
    DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1
run_smoke hc_only \
    DS4_CUDA_NO_Q4_DENSE_SCRATCH=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1
run_smoke default_fast DS4_CUDA_NO_Q4_K1024_PERSISTENT=1
run_smoke grouped_q81_rollback \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_Q81=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1
run_smoke grouped_prefill_rollback \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1

status=0
oracle_mismatch_arms=
run_smoke grouped_oracle \
    DS4_CUDA_NO_Q4_DENSE_SCRATCH=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1 \
    DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1 \
    DS4_CUDA_Q4_GROUPED_ATTN_A_ORACLE=1
check_grouped_oracle "$OUT_DIR/grouped_oracle.log"

run_smoke hc_oracle \
    DS4_CUDA_NO_Q4_DENSE_SCRATCH=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1 \
    DS4_CUDA_Q4_ATTN_OUT_HC_ORACLE=1
check_hc_oracle "$OUT_DIR/hc_oracle.log"

if [ -n "$oracle_mismatch_arms" ]; then
    status=1
fi
changed_smoke=
compare_smoke() {
    arm=$1
    if cmp -s "$OUT_DIR/umbrella_control.json" "$OUT_DIR/$arm.json"; then
        echo "q4-gb10-matrix: smoke $arm: EXACT"
    else
        echo "q4-gb10-matrix: smoke $arm: DIFFERENT"
        diff -u "$OUT_DIR/umbrella_control.json" "$OUT_DIR/$arm.json" \
            >"$OUT_DIR/$arm.diff" || true
        changed_smoke="$changed_smoke $arm"
        status=1
    fi
}

for arm in local_control scratch_only grouped_only hc_only default_fast \
           grouped_q81_rollback grouped_prefill_rollback \
           grouped_oracle hc_oracle; do
    compare_smoke "$arm"
done

changed_quality=
run_score umbrella_control DS4_CUDA_NO_Q4_GB10_FAST=1
set -- $LOCAL_ROLLBACK
run_score local_control "$@"
run_score scratch_only \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL=1 \
    DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1
run_score grouped_only \
    DS4_CUDA_NO_Q4_DENSE_SCRATCH=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1 \
    DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1
run_score hc_only \
    DS4_CUDA_NO_Q4_DENSE_SCRATCH=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1 \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1
run_score default_fast DS4_CUDA_NO_Q4_K1024_PERSISTENT=1
run_score grouped_q81_rollback \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_Q81=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1
run_score grouped_prefill_rollback \
    DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL=1 \
    DS4_CUDA_NO_Q4_K1024_PERSISTENT=1

for arm in local_control scratch_only grouped_only hc_only default_fast \
           grouped_q81_rollback grouped_prefill_rollback; do
    if cmp -s "$OUT_DIR/umbrella_control.tsv" "$OUT_DIR/$arm.tsv"; then
        echo "q4-gb10-matrix: quality $arm: EXACT"
    else
        echo "q4-gb10-matrix: quality $arm: DIFFERENT"
        python3 gguf-tools/quality-testing/compare_scores.py \
            "$OUT_DIR/umbrella_control.tsv" "$OUT_DIR/$arm.tsv" \
            >"$OUT_DIR/$arm.comparison.txt"
        changed_quality="$changed_quality $arm"
        status=1
    fi
done

if [ "$SKIP_PARITY" != 0 ]; then
    status=1
fi

{
    echo "output_dir=$OUT_DIR"
    echo "decode_graphs=$DECODE_GRAPHS"
    echo "smoke_differences=${changed_smoke# }"
    echo "quality_differences=${changed_quality# }"
    echo "oracle_mismatches=${oracle_mismatch_arms# }"
    echo "quality_manifest=$MANIFEST"
    if [ "$SKIP_PARITY" = 0 ]; then
        echo "synthetic_parity=pass"
    else
        echo "synthetic_parity=skipped"
    fi
    if [ "$status" -eq 0 ]; then
        echo "promotion_gate=pass"
    else
        echo "promotion_gate=blocked"
    fi
} | tee "$OUT_DIR/summary.txt"

exit "$status"
