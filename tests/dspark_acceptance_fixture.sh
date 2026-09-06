#!/bin/sh
set -eu

DS4_BIN=${DS4_BIN:-./ds4}
MODEL=${DS4_DSPARK_MODEL:-${DS4_TEST_MODEL:-./ds4flash.gguf}}
SUPPORT=${DS4_DSPARK_SUPPORT:-gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}
TOKENS=${DS4_DSPARK_FIXTURE_TOKENS:-32}
REQUIRE_PARTIAL=${DS4_DSPARK_FIXTURE_REQUIRE_PARTIAL:-0}
REQUIRE_DIRECT=${DS4_DSPARK_FIXTURE_REQUIRE_DIRECT_COMMIT:-1}
REQUIRE_IDENTICAL=${DS4_DSPARK_FIXTURE_REQUIRE_IDENTICAL:-0}
PROPOSAL_QUALITY_GUARD=${DS4_DSPARK_FIXTURE_REQUIRE_PROPOSAL_QUALITY:-auto}
C_ADD_MIN_ACCEPTED=${DS4_DSPARK_FIXTURE_C_ADD_MIN_ACCEPTED:-8}
CONFIDENCE=${DS4_DSPARK_FIXTURE_CONFIDENCE:-}
TEMPERATURE=${DS4_DSPARK_FIXTURE_TEMPERATURE:-0}
TOP_P=${DS4_DSPARK_FIXTURE_TOP_P:-0.95}
MIN_P=${DS4_DSPARK_FIXTURE_MIN_P:-0.05}
SEED=${DS4_DSPARK_FIXTURE_SEED:-12345}
EXACT_SAMPLING=${DS4_DSPARK_FIXTURE_EXACT_SAMPLING:-0}
exact_sampling_arg=
if [ "$EXACT_SAMPLING" != 0 ]; then
    exact_sampling_arg=--mtp-exact-sampling
fi
partial_cases=0
direct_partial_cases=0
direct_commits=0
BACKEND=${DS4_DSPARK_FIXTURE_BACKEND:-auto}
SSD_STREAMING=${DS4_DSPARK_FIXTURE_SSD_STREAMING:-0}
SSD_CACHE_EXPERTS=${DS4_DSPARK_FIXTURE_SSD_STREAMING_CACHE_EXPERTS:-}
REQUIRE_ACTIVE=${DS4_DSPARK_FIXTURE_REQUIRE_ACTIVE:-1}
REQUIRE_EXACT2=${DS4_DSPARK_FIXTURE_REQUIRE_EXACT2:-0}
REQUIRE_CUDA_EXACTN=${DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN:-0}
REQUIRE_CUDA_EXACTN_BATCH_HEAD=${DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN_BATCH_HEAD:-0}
REQUIRE_CUDA_EXACTN_GRAPHS=${DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN_GRAPHS:-0}
REQUIRE_CUDA_DEVICE_PROPOSER=${DS4_DSPARK_FIXTURE_REQUIRE_CUDA_DEVICE_PROPOSER:-0}
REQUIRE_METAL_EXACTN_BATCH_HEAD=${DS4_DSPARK_FIXTURE_REQUIRE_METAL_EXACTN_BATCH_HEAD:-0}
REQUIRE_METAL_EXACTN_PARTIAL=${DS4_DSPARK_FIXTURE_REQUIRE_METAL_EXACTN_PARTIAL:-0}
REQUIRE_METAL_DEVICE_PROPOSER=${DS4_DSPARK_FIXTURE_REQUIRE_METAL_DEVICE_PROPOSER:-0}
total_proposed=0
total_accepted_draft=0
total_exact2_attempt=0
total_exact2_fallback=0
total_cuda_exactn_attempt=0
total_cuda_exactn_fallback=0
total_cuda_exactn_error_fallback=0
total_cuda_exactn_batch_head_attempt=0
total_cuda_exactn_batch_head_use=0
total_cuda_exactn_batch_head_fallback=0
total_cuda_exactn_graph_attempt=0
total_cuda_exactn_graph_use=0
total_cuda_exactn_graph_capture=0
total_cuda_exactn_graph_replay=0
total_cuda_exactn_graph_warm=0
total_cuda_exactn_graph_no_slot=0
total_cuda_exactn_graph_failure=0
total_cuda_device_proposer_attempt=0
total_cuda_device_proposer_use=0
total_cuda_device_proposer_fallback=0
total_cuda_device_proposer_policy_mismatch=0
total_exactn_union_error_fallback=0
total_exactn_union_partial_replay=0
total_exactn_union_verify_skip=0
total_metal_exactn_batch_head_attempt=0
total_metal_exactn_batch_head_use=0
total_metal_exactn_batch_head_fallback=0
total_metal_device_proposer_attempt=0
total_metal_device_proposer_use=0
total_metal_device_proposer_fallback=0
total_metal_device_proposer_policy_mismatch=0

stats_field() {
    printf '%s\n' "$1" | awk -v key="$2" '
        { prefix = key "="
          for (i = 1; i <= NF; i++) {
              if (index($i, prefix) == 1) {
                  print substr($i, length(prefix) + 1)
                  exit
              }
          }
        }'
}

case "$BACKEND" in
auto|metal|cuda|rocm) ;;
*)
    echo "dspark-fixture: invalid DS4_DSPARK_FIXTURE_BACKEND=$BACKEND" >&2
    exit 1
    ;;
esac
case "$SSD_STREAMING" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_SSD_STREAMING must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUIRE_ACTIVE" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_ACTIVE must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUIRE_EXACT2" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_EXACT2 must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUIRE_CUDA_EXACTN" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUIRE_CUDA_EXACTN_BATCH_HEAD" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN_BATCH_HEAD must be 0 or 1" >&2
    exit 1
    ;;
esac
if [ "$REQUIRE_CUDA_EXACTN_BATCH_HEAD" != 0 ]; then
    REQUIRE_CUDA_EXACTN=1
fi
case "$REQUIRE_CUDA_EXACTN_GRAPHS" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN_GRAPHS must be 0 or 1" >&2
    exit 1
    ;;
esac
if [ "$REQUIRE_CUDA_EXACTN_GRAPHS" != 0 ]; then
    REQUIRE_CUDA_EXACTN=1
fi
case "$REQUIRE_CUDA_DEVICE_PROPOSER" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_CUDA_DEVICE_PROPOSER must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUIRE_METAL_EXACTN_BATCH_HEAD" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_METAL_EXACTN_BATCH_HEAD must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUIRE_METAL_EXACTN_PARTIAL" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_METAL_EXACTN_PARTIAL must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUIRE_METAL_DEVICE_PROPOSER" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_METAL_DEVICE_PROPOSER must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUIRE_DIRECT" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_DIRECT_COMMIT must be 0 or 1" >&2
    exit 1
    ;;
esac
case "$REQUIRE_IDENTICAL" in
0|1) ;;
*)
    echo "dspark-fixture: DS4_DSPARK_FIXTURE_REQUIRE_IDENTICAL must be 0 or 1" >&2
    exit 1
    ;;
esac
if [ "$REQUIRE_EXACT2" != 0 ] || [ "$REQUIRE_CUDA_EXACTN" != 0 ] ||
   [ "$REQUIRE_CUDA_DEVICE_PROPOSER" != 0 ] ||
   [ "$REQUIRE_METAL_EXACTN_BATCH_HEAD" != 0 ] ||
   [ "$REQUIRE_METAL_EXACTN_PARTIAL" != 0 ] ||
   [ "$REQUIRE_METAL_DEVICE_PROPOSER" != 0 ]; then
    REQUIRE_IDENTICAL=1
fi
case "$SSD_CACHE_EXPERTS" in
""|*[!0-9]*)
    if [ -n "$SSD_CACHE_EXPERTS" ]; then
        echo "dspark-fixture: invalid SSD streaming expert count $SSD_CACHE_EXPERTS" >&2
        exit 1
    fi
    ;;
esac
if [ "$SSD_STREAMING" = 0 ] && [ -n "$SSD_CACHE_EXPERTS" ]; then
    echo "dspark-fixture: SSD cache experts requires DS4_DSPARK_FIXTURE_SSD_STREAMING=1" >&2
    exit 1
fi

proposal_quality_guard_enabled() {
    case "$PROPOSAL_QUALITY_GUARD" in
    0|false|no|off)
        return 1
        ;;
    1|true|yes|on)
        return 0
        ;;
    auto|"")
        [ "$REQUIRE_PARTIAL" = 0 ] || return 1
        [ -z "$CONFIDENCE" ] || return 1
        [ "$TOKENS" -ge 32 ] 2>/dev/null || return 1
        return 0
        ;;
    *)
        echo "dspark-fixture: invalid DS4_DSPARK_FIXTURE_REQUIRE_PROPOSAL_QUALITY=$PROPOSAL_QUALITY_GUARD" >&2
        exit 1
        ;;
    esac
}

case "$C_ADD_MIN_ACCEPTED" in
""|*[!0-9]*)
    echo "dspark-fixture: invalid DS4_DSPARK_FIXTURE_C_ADD_MIN_ACCEPTED=$C_ADD_MIN_ACCEPTED" >&2
    exit 1
    ;;
esac

PROPOSAL_QUALITY_GUARD_ACTIVE=0
if proposal_quality_guard_enabled; then
    PROPOSAL_QUALITY_GUARD_ACTIVE=1
fi

file_bytes() {
    if stat -L -f %z "$1" >/dev/null 2>&1; then
        stat -L -f %z "$1"
    elif stat -Lc %s "$1" >/dev/null 2>&1; then
        stat -Lc %s "$1"
    elif stat -f %z "$1" >/dev/null 2>&1; then
        stat -f %z "$1"
    elif stat -c %s "$1" >/dev/null 2>&1; then
        stat -c %s "$1"
    else
        echo unknown
    fi
}

git_commit_label() {
    commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
    if [ "$commit" != unknown ] && ! git diff --quiet -- . 2>/dev/null; then
        commit="${commit}+dirty"
    fi
    echo "$commit"
}

print_metadata() {
    hw_os=$(uname -sm 2>/dev/null || echo unknown)
    hw_model=$(sysctl -n hw.model 2>/dev/null || true)
    hw_cpu=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)
    confidence=${CONFIDENCE:-default}
    exact2_cuda=${DS4_CUDA_DSPARK_EXACT2:-unset}
    exact2_metal=${DS4_METAL_DSPARK_EXACT2:-unset}
    exactn_cuda=${DS4_CUDA_DSPARK_EXACTN:-unset}
    exactn_cuda_disable=${DS4_CUDA_DISABLE_DSPARK_EXACTN:-unset}
    exactn_cuda_batch_head=${DS4_CUDA_DSPARK_EXACTN_BATCH_HEAD:-unset}
    exactn_cuda_batch_head_disable=${DS4_CUDA_DISABLE_DSPARK_EXACTN_BATCH_HEAD:-unset}
    exactn_cuda_graphs=${DS4_CUDA_DSPARK_EXACTN_GRAPHS:-unset}
    exactn_cuda_graphs_disable=${DS4_CUDA_DISABLE_DSPARK_EXACTN_GRAPHS:-unset}
    cuda_device_proposer=${DS4_CUDA_DSPARK_DEVICE_PROPOSER:-unset}
    cuda_device_proposer_disable=${DS4_CUDA_DSPARK_NO_DEVICE_PROPOSER:-unset}
    exactn_union_metal=${DS4_METAL_DSPARK_EXACTN_UNION:-unset}
    exactn_metal_batch_head=${DS4_METAL_DSPARK_EXACTN_BATCH_HEAD:-unset}
    exactn_metal_batch_head_disable=${DS4_METAL_DISABLE_DSPARK_EXACTN_BATCH_HEAD:-unset}
    metal_device_proposer=${DS4_METAL_DSPARK_DEVICE_PROPOSER:-unset}
    metal_device_proposer_disable=${DS4_METAL_DSPARK_NO_DEVICE_PROPOSER:-unset}
    noncausal_online_cuda=${DS4_CUDA_ENABLE_DSPARK_NONCAUSAL_ONLINE:-unset}
    noncausal_online_cuda_disable=${DS4_CUDA_DISABLE_DSPARK_NONCAUSAL_ONLINE:-unset}
    verify_noncausal=${DS4_DSPARK_VERIFY_NONCAUSAL:-unset}
    exact_rows_async_tails_metal=${DS4_METAL_DSPARK_EXACT_ROWS_ASYNC_TAILS:-unset}
    proposer_cap_cuda=${DS4_CUDA_DSPARK_PROPOSER_BLOCK_MAX:-unset}
    proposer_cap_metal=${DS4_METAL_DSPARK_PROPOSER_BLOCK_MAX:-unset}
    verifier_cap=${DS4_DSPARK_SSD_VERIFY_BLOCK_MAX:-unset}

    printf '# commit=%s\n' "$(git_commit_label)"
    printf '# hardware_os=%s hardware_model=%s hardware_cpu=%s\n' \
        "$hw_os" "${hw_model:-unknown}" "${hw_cpu:-unknown}"
    printf '# model=%s model_bytes=%s support=%s support_bytes=%s\n' \
        "$MODEL" "$(file_bytes "$MODEL")" \
        "$SUPPORT" "$(file_bytes "$SUPPORT")"
    printf '# tokens=%s ctx=default flags="--temp %s --top-p %s --min-p %s --seed %s --nothink" exact_sampling=%s confidence=%s proposal_quality_guard=%s proposal_quality_active=%s c_add_min_accepted=%s require_direct=%s require_identical=%s\n' \
        "$TOKENS" "$TEMPERATURE" "$TOP_P" "$MIN_P" "$SEED" \
        "$EXACT_SAMPLING" "$confidence" "$PROPOSAL_QUALITY_GUARD" \
        "$PROPOSAL_QUALITY_GUARD_ACTIVE" "$C_ADD_MIN_ACCEPTED" \
        "$REQUIRE_DIRECT" "$REQUIRE_IDENTICAL"
    printf '# backend=%s ssd_streaming=%s ssd_cache_experts=%s require_active=%s\n' \
        "$BACKEND" "$SSD_STREAMING" "${SSD_CACHE_EXPERTS:-auto}" "$REQUIRE_ACTIVE"
    printf '# baseline_command=%s -m %s --tokens %s --temp %s --top-p %s --min-p %s --seed %s --nothink -p <fixture-prompt>\n' \
        "$DS4_BIN" "$MODEL" "$TOKENS" "$TEMPERATURE" "$TOP_P" "$MIN_P" "$SEED"
    printf '# dspark_command=DS4_DSPARK_STATS=1 %s --dspark%s%s -m %s --mtp-model %s --tokens %s --temp %s --top-p %s --min-p %s --seed %s --nothink -p <fixture-prompt>\n' \
        "$DS4_BIN" "${exact_sampling_arg:+ $exact_sampling_arg}" \
        "${CONFIDENCE:+ --dspark-confidence $CONFIDENCE}" \
        "$MODEL" "$SUPPORT" "$TOKENS" "$TEMPERATURE" "$TOP_P" "$MIN_P" "$SEED"
    printf '# exact2_cuda=%s exact2_metal=%s proposer_block_max_cuda=%s proposer_block_max_metal=%s verifier_block_max=%s require_exact2=%s\n' \
        "$exact2_cuda" "$exact2_metal" "$proposer_cap_cuda" \
        "$proposer_cap_metal" "$verifier_cap" "$REQUIRE_EXACT2"
    printf '# exactn_cuda=%s exactn_cuda_disable=%s require_cuda_exactn=%s exactn_cuda_batch_head=%s exactn_cuda_batch_head_disable=%s require_cuda_exactn_batch_head=%s cuda_device_proposer=%s cuda_device_proposer_disable=%s require_cuda_device_proposer=%s exactn_union_metal=%s noncausal_online_cuda=%s noncausal_online_cuda_disable=%s verify_noncausal=%s exact_rows_async_tails_metal=%s\n' \
        "$exactn_cuda" "$exactn_cuda_disable" "$REQUIRE_CUDA_EXACTN" \
        "$exactn_cuda_batch_head" "$exactn_cuda_batch_head_disable" \
        "$REQUIRE_CUDA_EXACTN_BATCH_HEAD" \
        "$cuda_device_proposer" "$cuda_device_proposer_disable" \
        "$REQUIRE_CUDA_DEVICE_PROPOSER" \
        "$exactn_union_metal" "$noncausal_online_cuda" \
        "$noncausal_online_cuda_disable" "$verify_noncausal" \
        "$exact_rows_async_tails_metal"
    printf '# exactn_cuda_graphs=%s exactn_cuda_graphs_disable=%s require_cuda_exactn_graphs=%s\n' \
        "$exactn_cuda_graphs" "$exactn_cuda_graphs_disable" \
        "$REQUIRE_CUDA_EXACTN_GRAPHS"
    printf '# exactn_metal_batch_head=%s exactn_metal_batch_head_disable=%s require_metal_exactn_batch_head=%s require_metal_exactn_partial=%s metal_device_proposer=%s metal_device_proposer_disable=%s require_metal_device_proposer=%s\n' \
        "$exactn_metal_batch_head" "$exactn_metal_batch_head_disable" \
        "$REQUIRE_METAL_EXACTN_BATCH_HEAD" "$REQUIRE_METAL_EXACTN_PARTIAL" \
        "$metal_device_proposer" "$metal_device_proposer_disable" \
        "$REQUIRE_METAL_DEVICE_PROPOSER"
}

if [ ! -x "$DS4_BIN" ]; then
    echo "dspark-fixture: skipped, missing executable $DS4_BIN" >&2
    exit 0
fi
if [ ! -f "$MODEL" ]; then
    echo "dspark-fixture: skipped, missing model $MODEL" >&2
    exit 0
fi
if [ ! -f "$SUPPORT" ]; then
    echo "dspark-fixture: skipped, missing DSpark support model $SUPPORT" >&2
    exit 0
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-dspark-fixture.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM

run_logged() {
    out=$1
    err=$2
    shift 2
    if "$@" >"$out" 2>"$err"; then
        return 0
    else
        status=$?
        cat "$err" >&2
        return "$status"
    fi
}

run_variant() {
    mode=$1
    prompt=$2
    stdout_file=$3
    stderr_file=$4

    set -- "$DS4_BIN"
    case "$BACKEND" in
    metal) set -- "$@" --metal ;;
    cuda)  set -- "$@" --cuda ;;
    rocm)  set -- "$@" --rocm ;;
    esac
    if [ "$SSD_STREAMING" = 1 ]; then
        set -- "$@" --ssd-streaming
        if [ -n "$SSD_CACHE_EXPERTS" ]; then
            set -- "$@" --ssd-streaming-cache-experts "$SSD_CACHE_EXPERTS"
        fi
    fi
    if [ "$mode" = dspark ]; then
        set -- "$@" --dspark --mtp-model "$SUPPORT"
        if [ "$EXACT_SAMPLING" != 0 ]; then
            set -- "$@" --mtp-exact-sampling
        fi
        if [ -n "$CONFIDENCE" ]; then
            set -- "$@" --dspark-confidence "$CONFIDENCE"
        fi
    fi
    set -- "$@" -m "$MODEL" --tokens "$TOKENS" \
        --temp "$TEMPERATURE" --top-p "$TOP_P" --min-p "$MIN_P" \
        --seed "$SEED" --nothink -p "$prompt"

    if [ "$mode" = dspark ]; then
        DS4_DSPARK_STATS=1 run_logged "$stdout_file" "$stderr_file" "$@"
    else
        run_logged "$stdout_file" "$stderr_file" "$@"
    fi
}

run_case() {
    id=$1
    prompt=$2
    base_out="$tmpdir/$id.baseline.out"
    base_err="$tmpdir/$id.baseline.err"
    dspark_out="$tmpdir/$id.dspark.out"
    dspark_err="$tmpdir/$id.dspark.err"

    run_variant baseline "$prompt" "$base_out" "$base_err"
    run_variant dspark "$prompt" "$dspark_out" "$dspark_err"

    output_match=1
    if ! cmp -s "$base_out" "$dspark_out"; then
        output_match=0
        if [ "$REQUIRE_IDENTICAL" != 0 ]; then
            echo "dspark-fixture: output mismatch for $id" >&2
            echo "baseline:" >&2
            sed 's/^/  /' "$base_out" >&2
            echo "dspark:" >&2
            sed 's/^/  /' "$dspark_out" >&2
            return 1
        fi
    fi

    base_tps=$(sed -n 's/.*generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$base_err" | tail -n 1)
    dspark_tps=$(sed -n 's/.*generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$dspark_err" | tail -n 1)
    stats=$(grep 'DSpark stats' "$dspark_err" | tail -n 1 | sed 's/^ds4: DSpark stats //')
    if [ -z "$stats" ]; then
        echo "dspark-fixture: missing DSpark stats for $id" >&2
        return 1
    fi

    partial=$(stats_field "$stats" partial)
    errors=$(stats_field "$stats" errors)
    verifier_unavailable=$(stats_field "$stats" verifier_unavailable)
    proposed=$(stats_field "$stats" proposed)
    accepted_draft=$(stats_field "$stats" accepted_draft)
    direct_full=$(stats_field "$stats" direct_full)
    direct_partial=$(stats_field "$stats" direct_partial)
    exact2_attempt=$(stats_field "$stats" exact2_attempt)
    exact2_fallback=$(stats_field "$stats" exact2_fallback)
    cuda_exactn_attempt=$(stats_field "$stats" cuda_exactn_attempt)
    cuda_exactn_fallback=$(stats_field "$stats" cuda_exactn_fallback)
    cuda_exactn_error_fallback=$(stats_field "$stats" cuda_exactn_error_fallback)
    cuda_exactn_batch_head_attempt=$(stats_field "$stats" cuda_exactn_batch_head_attempt)
    cuda_exactn_batch_head_use=$(stats_field "$stats" cuda_exactn_batch_head_use)
    cuda_exactn_batch_head_fallback=$(stats_field "$stats" cuda_exactn_batch_head_fallback)
    cuda_exactn_graph_attempt=$(stats_field "$stats" cuda_exactn_graph_attempt)
    cuda_exactn_graph_use=$(stats_field "$stats" cuda_exactn_graph_use)
    cuda_exactn_graph_capture=$(stats_field "$stats" cuda_exactn_graph_capture)
    cuda_exactn_graph_replay=$(stats_field "$stats" cuda_exactn_graph_replay)
    cuda_exactn_graph_warm=$(stats_field "$stats" cuda_exactn_graph_warm)
    cuda_exactn_graph_no_slot=$(stats_field "$stats" cuda_exactn_graph_no_slot)
    cuda_exactn_graph_failure=$(stats_field "$stats" cuda_exactn_graph_failure)
    cuda_device_proposer_attempt=$(stats_field "$stats" cuda_device_proposer_attempt)
    cuda_device_proposer_use=$(stats_field "$stats" cuda_device_proposer_use)
    cuda_device_proposer_fallback=$(stats_field "$stats" cuda_device_proposer_fallback)
    cuda_device_proposer_policy_mismatch=$(stats_field "$stats" cuda_device_proposer_policy_mismatch)
    exactn_union_error_fallback=$(stats_field "$stats" exactn_union_error_fallback)
    exactn_union_partial_replay=$(stats_field "$stats" exactn_union_partial_replay)
    exactn_union_verify_skip=$(stats_field "$stats" exactn_union_verify_skip)
    metal_exactn_batch_head_attempt=$(stats_field "$stats" metal_exactn_batch_head_attempt)
    metal_exactn_batch_head_use=$(stats_field "$stats" metal_exactn_batch_head_use)
    metal_exactn_batch_head_fallback=$(stats_field "$stats" metal_exactn_batch_head_fallback)
    metal_device_proposer_attempt=$(stats_field "$stats" metal_device_proposer_attempt)
    metal_device_proposer_use=$(stats_field "$stats" metal_device_proposer_use)
    metal_device_proposer_fallback=$(stats_field "$stats" metal_device_proposer_fallback)
    metal_device_proposer_policy_mismatch=$(stats_field "$stats" metal_device_proposer_policy_mismatch)
    exact2_full=$(stats_field "$stats" exact2_full)
    cuda_exactn_full=$(stats_field "$stats" cuda_exactn_full)
    exactn_union_full=$(stats_field "$stats" exactn_union_full)
    exactn_full=$(stats_field "$stats" exactn_full)
    first_tokens=$(stats_field "$stats" first_tokens)
    seed_batches=$(stats_field "$stats" seed_batches)
    partial=${partial:-0}
    errors=${errors:-0}
    verifier_unavailable=${verifier_unavailable:-0}
    proposed=${proposed:-0}
    accepted_draft=${accepted_draft:-0}
    first_tokens=${first_tokens:-0}
    seed_batches=${seed_batches:-0}
    direct_full=${direct_full:-0}
    direct_partial=${direct_partial:-0}
    exact2_attempt=${exact2_attempt:-0}
    exact2_fallback=${exact2_fallback:-0}
    cuda_exactn_attempt=${cuda_exactn_attempt:-0}
    cuda_exactn_fallback=${cuda_exactn_fallback:-0}
    cuda_exactn_error_fallback=${cuda_exactn_error_fallback:-0}
    cuda_exactn_batch_head_attempt=${cuda_exactn_batch_head_attempt:-0}
    cuda_exactn_batch_head_use=${cuda_exactn_batch_head_use:-0}
    cuda_exactn_batch_head_fallback=${cuda_exactn_batch_head_fallback:-0}
    cuda_exactn_graph_attempt=${cuda_exactn_graph_attempt:-0}
    cuda_exactn_graph_use=${cuda_exactn_graph_use:-0}
    cuda_exactn_graph_capture=${cuda_exactn_graph_capture:-0}
    cuda_exactn_graph_replay=${cuda_exactn_graph_replay:-0}
    cuda_exactn_graph_warm=${cuda_exactn_graph_warm:-0}
    cuda_exactn_graph_no_slot=${cuda_exactn_graph_no_slot:-0}
    cuda_exactn_graph_failure=${cuda_exactn_graph_failure:-0}
    cuda_device_proposer_attempt=${cuda_device_proposer_attempt:-0}
    cuda_device_proposer_use=${cuda_device_proposer_use:-0}
    cuda_device_proposer_fallback=${cuda_device_proposer_fallback:-0}
    cuda_device_proposer_policy_mismatch=${cuda_device_proposer_policy_mismatch:-0}
    exactn_union_error_fallback=${exactn_union_error_fallback:-0}
    exactn_union_partial_replay=${exactn_union_partial_replay:-0}
    exactn_union_verify_skip=${exactn_union_verify_skip:-0}
    metal_exactn_batch_head_attempt=${metal_exactn_batch_head_attempt:-0}
    metal_exactn_batch_head_use=${metal_exactn_batch_head_use:-0}
    metal_exactn_batch_head_fallback=${metal_exactn_batch_head_fallback:-0}
    metal_device_proposer_attempt=${metal_device_proposer_attempt:-0}
    metal_device_proposer_use=${metal_device_proposer_use:-0}
    metal_device_proposer_fallback=${metal_device_proposer_fallback:-0}
    metal_device_proposer_policy_mismatch=${metal_device_proposer_policy_mismatch:-0}
    exact2_full=${exact2_full:-0}
    cuda_exactn_full=${cuda_exactn_full:-0}
    exactn_union_full=${exactn_union_full:-0}
    exactn_full=${exactn_full:-0}
    if [ "$errors" -ne 0 ]; then
        echo "dspark-fixture: verifier errors for $id: $stats" >&2
        return 1
    fi
    if [ "$verifier_unavailable" -ne 0 ]; then
        echo "dspark-fixture: verifier unavailable for $id: $stats" >&2
        return 1
    fi
    total_proposed=$((total_proposed + proposed))
    total_accepted_draft=$((total_accepted_draft + accepted_draft))
    total_exact2_attempt=$((total_exact2_attempt + exact2_attempt))
    total_exact2_fallback=$((total_exact2_fallback + exact2_fallback))
    total_cuda_exactn_attempt=$((total_cuda_exactn_attempt + cuda_exactn_attempt))
    total_cuda_exactn_fallback=$((total_cuda_exactn_fallback + cuda_exactn_fallback))
    total_cuda_exactn_error_fallback=$((total_cuda_exactn_error_fallback + cuda_exactn_error_fallback))
    total_cuda_exactn_batch_head_attempt=$((total_cuda_exactn_batch_head_attempt + cuda_exactn_batch_head_attempt))
    total_cuda_exactn_batch_head_use=$((total_cuda_exactn_batch_head_use + cuda_exactn_batch_head_use))
    total_cuda_exactn_batch_head_fallback=$((total_cuda_exactn_batch_head_fallback + cuda_exactn_batch_head_fallback))
    total_cuda_exactn_graph_attempt=$((total_cuda_exactn_graph_attempt + cuda_exactn_graph_attempt))
    total_cuda_exactn_graph_use=$((total_cuda_exactn_graph_use + cuda_exactn_graph_use))
    total_cuda_exactn_graph_capture=$((total_cuda_exactn_graph_capture + cuda_exactn_graph_capture))
    total_cuda_exactn_graph_replay=$((total_cuda_exactn_graph_replay + cuda_exactn_graph_replay))
    total_cuda_exactn_graph_warm=$((total_cuda_exactn_graph_warm + cuda_exactn_graph_warm))
    total_cuda_exactn_graph_no_slot=$((total_cuda_exactn_graph_no_slot + cuda_exactn_graph_no_slot))
    total_cuda_exactn_graph_failure=$((total_cuda_exactn_graph_failure + cuda_exactn_graph_failure))
    total_cuda_device_proposer_attempt=$((total_cuda_device_proposer_attempt + cuda_device_proposer_attempt))
    total_cuda_device_proposer_use=$((total_cuda_device_proposer_use + cuda_device_proposer_use))
    total_cuda_device_proposer_fallback=$((total_cuda_device_proposer_fallback + cuda_device_proposer_fallback))
    total_cuda_device_proposer_policy_mismatch=$((total_cuda_device_proposer_policy_mismatch + cuda_device_proposer_policy_mismatch))
    total_exactn_union_error_fallback=$((total_exactn_union_error_fallback + exactn_union_error_fallback))
    total_exactn_union_partial_replay=$((total_exactn_union_partial_replay + exactn_union_partial_replay))
    total_exactn_union_verify_skip=$((total_exactn_union_verify_skip + exactn_union_verify_skip))
    total_metal_exactn_batch_head_attempt=$((total_metal_exactn_batch_head_attempt + metal_exactn_batch_head_attempt))
    total_metal_exactn_batch_head_use=$((total_metal_exactn_batch_head_use + metal_exactn_batch_head_use))
    total_metal_exactn_batch_head_fallback=$((total_metal_exactn_batch_head_fallback + metal_exactn_batch_head_fallback))
    total_metal_device_proposer_attempt=$((total_metal_device_proposer_attempt + metal_device_proposer_attempt))
    total_metal_device_proposer_use=$((total_metal_device_proposer_use + metal_device_proposer_use))
    total_metal_device_proposer_fallback=$((total_metal_device_proposer_fallback + metal_device_proposer_fallback))
    total_metal_device_proposer_policy_mismatch=$((total_metal_device_proposer_policy_mismatch + metal_device_proposer_policy_mismatch))
    if [ "$REQUIRE_EXACT2" != 0 ] && [ "$exact2_fallback" -ne 0 ]; then
        echo "dspark-fixture: exact2 fallback for $id: $stats" >&2
        return 1
    fi
    if [ "$REQUIRE_CUDA_EXACTN" != 0 ] &&
       [ "$cuda_exactn_error_fallback" -ne 0 ]; then
        echo "dspark-fixture: CUDA exact-N error fallback for $id: $stats" >&2
        return 1
    fi
    if [ "$REQUIRE_CUDA_EXACTN_BATCH_HEAD" != 0 ] &&
       [ "$cuda_exactn_batch_head_fallback" -ne 0 ]; then
        echo "dspark-fixture: CUDA exact-N batch-head fallback for $id: $stats" >&2
        return 1
    fi
    if [ "$REQUIRE_CUDA_DEVICE_PROPOSER" != 0 ] &&
       { [ "$cuda_device_proposer_fallback" -ne 0 ] ||
         [ "$cuda_device_proposer_policy_mismatch" -ne 0 ]; }; then
        echo "dspark-fixture: CUDA device proposer fallback/mismatch for $id: $stats" >&2
        return 1
    fi
    if [ "$REQUIRE_METAL_EXACTN_BATCH_HEAD" != 0 ] &&
       [ "$metal_exactn_batch_head_fallback" -ne 0 ]; then
        echo "dspark-fixture: Metal exact-N batch-head fallback for $id: $stats" >&2
        return 1
    fi
    if [ "$REQUIRE_METAL_EXACTN_PARTIAL" != 0 ] &&
       { [ "$exactn_union_error_fallback" -ne 0 ] ||
         [ "$exactn_union_partial_replay" -ne "$exactn_union_verify_skip" ]; }; then
        echo "dspark-fixture: Metal exact-N partial replay/skip mismatch for $id: $stats" >&2
        return 1
    fi
    if [ "$REQUIRE_METAL_DEVICE_PROPOSER" != 0 ] &&
       { [ "$metal_device_proposer_fallback" -ne 0 ] ||
         [ "$metal_device_proposer_policy_mismatch" -ne 0 ]; }; then
        echo "dspark-fixture: Metal device proposer fallback/mismatch for $id: $stats" >&2
        return 1
    fi
    if [ "$accepted_draft" -gt "$proposed" ] || [ "$seed_batches" -gt "$first_tokens" ]; then
        echo "dspark-fixture: inconsistent seed/draft accounting for $id: $stats" >&2
        return 1
    fi
    if [ "$PROPOSAL_QUALITY_GUARD_ACTIVE" -ne 0 ] && [ "$id" = c_add ] &&
        [ "$accepted_draft" -lt "$C_ADD_MIN_ACCEPTED" ]; then
        echo "dspark-fixture: c_add accepted_draft $accepted_draft below required $C_ADD_MIN_ACCEPTED: $stats" >&2
        return 1
    fi
    if [ "$REQUIRE_PARTIAL" != 0 ] && [ "$partial" -gt 0 ]; then
        partial_cases=$((partial_cases + 1))
    fi
    if [ "$direct_partial" -gt 0 ]; then
        direct_partial_cases=$((direct_partial_cases + 1))
    fi
    direct_commits=$((direct_commits + direct_full + direct_partial + \
        exact2_full + cuda_exactn_full + exactn_union_full + exactn_full))

    printf '%s\toutput_match=%s\tbaseline_tps=%s\tdspark_tps=%s\t%s\n' \
        "$id" "$output_match" "${base_tps:-n/a}" "${dspark_tps:-n/a}" "$stats"
}

print_metadata
echo "id	output_match	baseline_tps	dspark_tps	dspark_stats"
run_case hello 'Hello'
run_case redis 'Explain Redis in one sentence.'
run_case math 'What is 17 times 23?'
run_case python_reverse 'Write a Python function that reverses a string.'
run_case c_add 'Complete this C function: int add(int a, int b) {'

if [ "$REQUIRE_PARTIAL" != 0 ] && [ "$partial_cases" -eq 0 ]; then
    echo "dspark-fixture: expected at least one partial accept case" >&2
    exit 1
fi
if [ "$REQUIRE_PARTIAL" != 0 ] && [ "$direct_partial_cases" -eq 0 ]; then
    echo "dspark-fixture: expected at least one direct partial commit" >&2
    exit 1
fi
if [ "$REQUIRE_DIRECT" != 0 ] && [ "$direct_commits" -eq 0 ]; then
    echo "dspark-fixture: expected at least one direct verifier-state commit" >&2
    exit 1
fi
if [ "$REQUIRE_ACTIVE" != 0 ] &&
   { [ "$total_proposed" -eq 0 ] || [ "$total_accepted_draft" -eq 0 ]; }; then
    echo "dspark-fixture: DSpark runtime was not active (proposed=$total_proposed accepted_draft=$total_accepted_draft)" >&2
    exit 1
fi
if [ "$REQUIRE_EXACT2" != 0 ] && [ "$total_exact2_attempt" -eq 0 ]; then
    echo "dspark-fixture: exact2 was required but never attempted" >&2
    exit 1
fi
if [ "$REQUIRE_EXACT2" != 0 ] && [ "$total_exact2_fallback" -ne 0 ]; then
    echo "dspark-fixture: exact2 fallback count=$total_exact2_fallback" >&2
    exit 1
fi
if [ "$REQUIRE_CUDA_EXACTN" != 0 ]; then
    printf '# cuda_exactn_attempt=%s cuda_exactn_fallback=%s cuda_exactn_error_fallback=%s\n' \
        "$total_cuda_exactn_attempt" "$total_cuda_exactn_fallback" \
        "$total_cuda_exactn_error_fallback"
fi
if [ "$REQUIRE_CUDA_EXACTN_BATCH_HEAD" != 0 ]; then
    printf '# cuda_exactn_batch_head_attempt=%s cuda_exactn_batch_head_use=%s cuda_exactn_batch_head_fallback=%s\n' \
        "$total_cuda_exactn_batch_head_attempt" \
        "$total_cuda_exactn_batch_head_use" \
        "$total_cuda_exactn_batch_head_fallback"
fi
if [ "$REQUIRE_CUDA_EXACTN_GRAPHS" != 0 ]; then
    printf '# cuda_exactn_graph_attempt=%s cuda_exactn_graph_use=%s cuda_exactn_graph_capture=%s cuda_exactn_graph_replay=%s cuda_exactn_graph_warm=%s cuda_exactn_graph_no_slot=%s cuda_exactn_graph_failure=%s\n' \
        "$total_cuda_exactn_graph_attempt" \
        "$total_cuda_exactn_graph_use" \
        "$total_cuda_exactn_graph_capture" \
        "$total_cuda_exactn_graph_replay" \
        "$total_cuda_exactn_graph_warm" \
        "$total_cuda_exactn_graph_no_slot" \
        "$total_cuda_exactn_graph_failure"
fi
if [ "$REQUIRE_CUDA_DEVICE_PROPOSER" != 0 ]; then
    printf '# cuda_device_proposer_attempt=%s cuda_device_proposer_use=%s cuda_device_proposer_fallback=%s cuda_device_proposer_policy_mismatch=%s\n' \
        "$total_cuda_device_proposer_attempt" \
        "$total_cuda_device_proposer_use" \
        "$total_cuda_device_proposer_fallback" \
        "$total_cuda_device_proposer_policy_mismatch"
fi
if [ "$REQUIRE_METAL_EXACTN_BATCH_HEAD" != 0 ]; then
    printf '# metal_exactn_batch_head_attempt=%s metal_exactn_batch_head_use=%s metal_exactn_batch_head_fallback=%s\n' \
        "$total_metal_exactn_batch_head_attempt" \
        "$total_metal_exactn_batch_head_use" \
        "$total_metal_exactn_batch_head_fallback"
fi
if [ "$REQUIRE_METAL_EXACTN_PARTIAL" != 0 ]; then
    printf '# exactn_union_error_fallback=%s exactn_union_partial_replay=%s exactn_union_verify_skip=%s\n' \
        "$total_exactn_union_error_fallback" \
        "$total_exactn_union_partial_replay" \
        "$total_exactn_union_verify_skip"
fi
if [ "$REQUIRE_METAL_DEVICE_PROPOSER" != 0 ]; then
    printf '# metal_device_proposer_attempt=%s metal_device_proposer_use=%s metal_device_proposer_fallback=%s metal_device_proposer_policy_mismatch=%s\n' \
        "$total_metal_device_proposer_attempt" \
        "$total_metal_device_proposer_use" \
        "$total_metal_device_proposer_fallback" \
        "$total_metal_device_proposer_policy_mismatch"
fi
if [ "$REQUIRE_CUDA_EXACTN" != 0 ] &&
   [ "$total_cuda_exactn_attempt" -eq 0 ]; then
    echo "dspark-fixture: CUDA exact-N was required but never attempted" >&2
    exit 1
fi
if [ "$REQUIRE_CUDA_EXACTN" != 0 ] &&
   [ "$total_cuda_exactn_error_fallback" -ne 0 ]; then
    echo "dspark-fixture: CUDA exact-N error fallback count=$total_cuda_exactn_error_fallback" >&2
    exit 1
fi
if [ "$REQUIRE_CUDA_EXACTN_BATCH_HEAD" != 0 ] &&
   { [ "$total_cuda_exactn_batch_head_attempt" -eq 0 ] ||
     [ "$total_cuda_exactn_batch_head_use" -eq 0 ] ||
     [ "$total_cuda_exactn_batch_head_fallback" -ne 0 ]; }; then
    echo "dspark-fixture: CUDA exact-N batch head not cleanly exercised (attempt=$total_cuda_exactn_batch_head_attempt use=$total_cuda_exactn_batch_head_use fallback=$total_cuda_exactn_batch_head_fallback)" >&2
    exit 1
fi
if [ "$REQUIRE_CUDA_EXACTN_GRAPHS" != 0 ] &&
   { [ "$total_cuda_exactn_graph_attempt" -eq 0 ] ||
     [ "$total_cuda_exactn_graph_use" -eq 0 ] ||
     [ "$total_cuda_exactn_graph_capture" -eq 0 ] ||
     [ "$total_cuda_exactn_graph_replay" -eq 0 ] ||
     [ "$total_cuda_exactn_graph_no_slot" -ne 0 ] ||
     [ "$total_cuda_exactn_graph_failure" -ne 0 ]; }; then
    echo "dspark-fixture: CUDA exact-N graphs were not cleanly replayed after warmup (attempt=$total_cuda_exactn_graph_attempt use=$total_cuda_exactn_graph_use warm=$total_cuda_exactn_graph_warm capture=$total_cuda_exactn_graph_capture replay=$total_cuda_exactn_graph_replay no_slot=$total_cuda_exactn_graph_no_slot failure=$total_cuda_exactn_graph_failure)" >&2
    exit 1
fi
if [ "$REQUIRE_CUDA_DEVICE_PROPOSER" != 0 ] &&
   { [ "$total_cuda_device_proposer_attempt" -eq 0 ] ||
     [ "$total_cuda_device_proposer_use" -eq 0 ] ||
     [ "$total_cuda_device_proposer_attempt" -ne "$total_cuda_device_proposer_use" ] ||
     [ "$total_cuda_device_proposer_fallback" -ne 0 ] ||
     [ "$total_cuda_device_proposer_policy_mismatch" -ne 0 ]; }; then
    echo "dspark-fixture: CUDA device proposer not cleanly exercised (attempt=$total_cuda_device_proposer_attempt use=$total_cuda_device_proposer_use fallback=$total_cuda_device_proposer_fallback policy_mismatch=$total_cuda_device_proposer_policy_mismatch)" >&2
    exit 1
fi
if [ "$REQUIRE_METAL_EXACTN_BATCH_HEAD" != 0 ] &&
   { [ "$total_metal_exactn_batch_head_attempt" -eq 0 ] ||
     [ "$total_metal_exactn_batch_head_use" -eq 0 ] ||
     [ "$total_metal_exactn_batch_head_fallback" -ne 0 ]; }; then
    echo "dspark-fixture: Metal exact-N batch head not cleanly exercised (attempt=$total_metal_exactn_batch_head_attempt use=$total_metal_exactn_batch_head_use fallback=$total_metal_exactn_batch_head_fallback)" >&2
    exit 1
fi
if [ "$REQUIRE_METAL_EXACTN_PARTIAL" != 0 ] &&
   { [ "$total_exactn_union_partial_replay" -eq 0 ] ||
     [ "$total_exactn_union_partial_replay" -ne "$total_exactn_union_verify_skip" ] ||
     [ "$total_exactn_union_error_fallback" -ne 0 ]; }; then
    echo "dspark-fixture: Metal exact-N partial path not cleanly exercised (replay=$total_exactn_union_partial_replay verify_skip=$total_exactn_union_verify_skip error_fallback=$total_exactn_union_error_fallback)" >&2
    exit 1
fi
if [ "$REQUIRE_METAL_DEVICE_PROPOSER" != 0 ] &&
   { [ "$total_metal_device_proposer_attempt" -eq 0 ] ||
     [ "$total_metal_device_proposer_use" -eq 0 ] ||
     [ "$total_metal_device_proposer_attempt" -ne "$total_metal_device_proposer_use" ] ||
     [ "$total_metal_device_proposer_fallback" -ne 0 ] ||
     [ "$total_metal_device_proposer_policy_mismatch" -ne 0 ]; }; then
    echo "dspark-fixture: Metal device proposer not cleanly exercised (attempt=$total_metal_device_proposer_attempt use=$total_metal_device_proposer_use fallback=$total_metal_device_proposer_fallback policy_mismatch=$total_metal_device_proposer_policy_mismatch)" >&2
    exit 1
fi
