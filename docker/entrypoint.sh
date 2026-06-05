#!/bin/sh
set -eu

DS4_GGUF_DIR=${DS4_GGUF_DIR:-/models}
DS4_MODEL=${DS4_MODEL:-q2-imatrix}
DS4_ENABLE_MTP=${DS4_ENABLE_MTP:-0}
DS4_CTX=${DS4_CTX:-100000}
DS4_KV_DISK_DIR=${DS4_KV_DISK_DIR:-/kv-cache}
DS4_KV_DISK_SPACE_MB=${DS4_KV_DISK_SPACE_MB:-8192}
DS4_HOST=${DS4_HOST:-0.0.0.0}
DS4_PORT=${DS4_PORT:-8000}
DS4_MTP_DRAFT=${DS4_MTP_DRAFT:-2}

export DS4_GGUF_DIR

mkdir -p "$DS4_GGUF_DIR" "$DS4_KV_DISK_DIR"

if [ -n "$DS4_MODEL" ] && [ "$DS4_MODEL" != "none" ]; then
    /app/download_model.sh "$DS4_MODEL"
fi

set -- \
    --host "$DS4_HOST" \
    --port "$DS4_PORT" \
    --ctx "$DS4_CTX" \
    --kv-disk-dir "$DS4_KV_DISK_DIR" \
    --kv-disk-space-mb "$DS4_KV_DISK_SPACE_MB" \
    "$@"

case "$DS4_ENABLE_MTP" in
    1|true|TRUE|yes|YES|on|ON)
        /app/download_model.sh mtp
        set -- --mtp "$DS4_GGUF_DIR/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf" --mtp-draft "$DS4_MTP_DRAFT" "$@"
        if [ -n "${DS4_MTP_MARGIN:-}" ]; then
            set -- --mtp-margin "$DS4_MTP_MARGIN" "$@"
        fi
        ;;
esac

if [ -n "${DS4_MODEL_PATH:-}" ]; then
    set -- --model "$DS4_MODEL_PATH" "$@"
fi

if [ -n "${DS4_THREADS:-}" ]; then
    set -- --threads "$DS4_THREADS" "$@"
fi

if [ -n "${DS4_EXTRA_ARGS:-}" ]; then
    # Intentionally split DS4_EXTRA_ARGS like a shell command line for advanced flags.
    # shellcheck disable=SC2086
    set -- "$@" $DS4_EXTRA_ARGS
fi

exec /app/ds4-server "$@"
