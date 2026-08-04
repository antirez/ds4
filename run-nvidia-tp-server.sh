#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$ROOT/$(basename -- "${BASH_SOURCE[0]}")"
cd -- "$ROOT"

MODEL="${DS4_MODEL:-/home/antirez/models/deepseek-v4-gguf/DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf}"
CTX="${DS4_CTX:-100000}"
SERVER_HOST="${DS4_SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${DS4_SERVER_PORT:-8000}"
KV_DIR="${DS4_KV_DIR:-/data/ds4-kv}"
KV_SPACE_MB="${DS4_KV_SPACE_MB:-8192}"
BATCHED_SESSIONS="${DS4_BATCHED_SESSIONS:-16}"
LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4.lock}"
LOG_FILE="${DS4_SERVER_LOG:-/tmp/ds4-server.log}"
START_TIMEOUT="${DS4_START_TIMEOUT:-180}"
STOP_TIMEOUT="${DS4_STOP_TIMEOUT:-120}"

die() {
    printf 'run-nvidia-tp-server: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage:
  $(basename -- "$0") [ds4-server options...]       Run in the foreground
  $(basename -- "$0") start [ds4-server options...] Start detached
  $(basename -- "$0") restart [ds4-server options...] Gracefully restart detached
  $(basename -- "$0") stop                          Gracefully stop
  $(basename -- "$0") status                        Show server status

Service log: $LOG_FILE
EOF
}

check_timeout() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] ||
        die "$1 must be a positive integer, got: $2"
}

check_prerequisites() {
    [[ -x ./ds4-server ]] ||
        die './ds4-server is not built; run `make cuda-generic` first'
    [[ -f "$MODEL" ]] ||
        die "model not found: $MODEL"
}

process_is_alive() {
    local pid="$1"
    local state

    state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
    [[ -n "$state" && "$state" != "Z" ]]
}

is_managed_server() {
    local pid="$1"
    local cwd arg
    local -a argv=()

    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" || return 1
    [[ "$cwd" == "$ROOT" ]] || return 1
    mapfile -d '' -t argv < "/proc/$pid/cmdline" || return 1
    ((${#argv[@]} > 0)) || return 1
    [[ "${argv[0]}" == "./ds4-server" ||
       "${argv[0]}" == "$ROOT/ds4-server" ]] || return 1
    for arg in "${argv[@]}"; do
        [[ "$arg" == "--cuda-tensor-parallel" ]] && return 0
    done
    return 1
}

LOCK_OWNER_PID=""
LOCK_OWNER_KIND="none"

inspect_lock_owner() {
    local pid

    LOCK_OWNER_PID=""
    LOCK_OWNER_KIND="none"
    if [[ -e "$LOCK_FILE" && (! -r "$LOCK_FILE" || ! -w "$LOCK_FILE") ]]; then
        die "cannot access $LOCK_FILE; fix its ownership before managing the server"
    fi
    [[ -r "$LOCK_FILE" ]] || return 0
    IFS= read -r pid < "$LOCK_FILE" || return 0
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 0
    process_is_alive "$pid" || return 0

    LOCK_OWNER_PID="$pid"
    if is_managed_server "$pid"; then
        LOCK_OWNER_KIND="managed"
    else
        LOCK_OWNER_KIND="other"
    fi
}

stop_server() {
    local deadline

    check_timeout DS4_STOP_TIMEOUT "$STOP_TIMEOUT"
    inspect_lock_owner
    case "$LOCK_OWNER_KIND" in
        none)
            printf 'run-nvidia-tp-server: server is not running\n'
            return 0
            ;;
        other)
            die "$LOCK_FILE is held by pid $LOCK_OWNER_PID, which is not this CUDA-TP server"
            ;;
    esac

    printf 'run-nvidia-tp-server: stopping pid %s\n' "$LOCK_OWNER_PID"
    kill -TERM "$LOCK_OWNER_PID"
    deadline=$((SECONDS + STOP_TIMEOUT))
    while process_is_alive "$LOCK_OWNER_PID"; do
        if ((SECONDS >= deadline)); then
            die "pid $LOCK_OWNER_PID did not stop within ${STOP_TIMEOUT}s"
        fi
        sleep 1
    done
    printf 'run-nvidia-tp-server: stopped\n'
}

run_server() {
    check_prerequisites
    exec ./ds4-server \
        --cuda \
        --cuda-tensor-parallel \
        --gpu-devices 0,2,4,6,1,3,5,7 \
        --gpu-vram auto \
        --model "$MODEL" \
        --ctx "$CTX" \
        --host "$SERVER_HOST" \
        --port "$SERVER_PORT" \
        --batched-session "$BATCHED_SESSIONS" \
        --kv-disk-dir "$KV_DIR" \
        --kv-disk-space-mb "$KV_SPACE_MB" \
        "$@"
}

start_server() {
    local pid="" deadline

    check_timeout DS4_START_TIMEOUT "$START_TIMEOUT"
    check_prerequisites
    command -v setsid >/dev/null 2>&1 ||
        die 'setsid is required for detached server startup'
    inspect_lock_owner
    case "$LOCK_OWNER_KIND" in
        managed)
            printf 'run-nvidia-tp-server: already running (pid %s)\n' "$LOCK_OWNER_PID"
            return 0
            ;;
        other)
            die "$LOCK_FILE is held by pid $LOCK_OWNER_PID, which is not this CUDA-TP server"
            ;;
    esac

    : > "$LOG_FILE"
    setsid -f "$SCRIPT" foreground "$@" > "$LOG_FILE" 2>&1 < /dev/null
    printf 'run-nvidia-tp-server: starting; log: %s\n' "$LOG_FILE"

    deadline=$((SECONDS + START_TIMEOUT))
    while ((SECONDS < deadline)); do
        inspect_lock_owner
        if [[ "$LOCK_OWNER_KIND" == "other" ]]; then
            die "$LOCK_FILE is held by pid $LOCK_OWNER_PID, which is not this CUDA-TP server"
        fi
        if [[ "$LOCK_OWNER_KIND" == "managed" ]]; then
            pid="$LOCK_OWNER_PID"
        fi
        if [[ -n "$pid" ]] &&
           process_is_alive "$pid" &&
           grep -Fq 'ds4-server: listening on ' "$LOG_FILE"; then
            printf 'run-nvidia-tp-server: ready on %s:%s (pid %s)\n' \
                "$SERVER_HOST" "$SERVER_PORT" "$pid"
            return 0
        fi
        sleep 1
    done

    tail -n 20 "$LOG_FILE" >&2 || true
    die "server did not become ready within ${START_TIMEOUT}s; see $LOG_FILE"
}

if (($# == 0)); then
    run_server
fi

command="$1"
shift
case "$command" in
    foreground|run)
        run_server "$@"
        ;;
    start)
        start_server "$@"
        ;;
    restart)
        stop_server
        start_server "$@"
        ;;
    stop)
        (($# == 0)) || die 'stop does not accept ds4-server options'
        stop_server
        ;;
    status)
        (($# == 0)) || die 'status does not accept ds4-server options'
        inspect_lock_owner
        case "$LOCK_OWNER_KIND" in
            managed)
                printf 'run-nvidia-tp-server: running (pid %s)\n' "$LOCK_OWNER_PID"
                ;;
            other)
                die "$LOCK_FILE is held by pid $LOCK_OWNER_PID, which is not this CUDA-TP server"
                ;;
            none)
                printf 'run-nvidia-tp-server: not running\n'
                exit 1
                ;;
        esac
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        run_server "$command" "$@"
        ;;
esac
