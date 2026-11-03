#!/bin/sh
# Fetches the prebuilt llama.cpp web UI (the same public artifact llama.cpp's
# own build downloads) and unpacks it into webui/, so ds4-server can serve it
# via --webui-dir (default: webui) without needing node/npm on this machine.
#
#   tools/fetch_webui.sh            fetch only if webui/index.html is missing
#   tools/fetch_webui.sh --force    re-fetch even if already present
#
# Env vars:
#   DS4_WEBUI_DIR      output directory (default: webui, relative to repo root)
#   DS4_UI_HF_BUCKET   Hugging Face bucket (default: ggml-org/llama-ui)
#   DS4_UI_VERSION     version tag to fetch (default: latest)
#   HF_TOKEN           optional bearer token, for higher rate limits
#
# Never fatal to the build: on any failure this prints a warning and exits 0,
# leaving webui/ as it was so ds4-server still builds and runs (just without
# a UI to serve) and the next `make` retries the fetch.

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WEBUI_DIR=${DS4_WEBUI_DIR:-"$ROOT/webui"}
HF_BUCKET=${DS4_UI_HF_BUCKET:-"ggml-org/llama-ui"}
VERSION=${DS4_UI_VERSION:-"latest"}
TOKEN=${HF_TOKEN:-}
STAMP_FILE="$WEBUI_DIR/.ui-version"

warn() { echo "webui: $*" >&2; }

if [ "$1" != "--force" ] && [ -f "$WEBUI_DIR/index.html" ]; then
    if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE" 2>/dev/null)" = "$VERSION" ]; then
        echo "webui: already have '$VERSION' assets in $WEBUI_DIR, skipping (tools/fetch_webui.sh --force to refetch)"
        exit 0
    fi
    warn "assets present but not stamped as '$VERSION'; refetching"
fi

dl() {
    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -H "Authorization: Bearer $TOKEN" -o "$2" "$1"
    else
        curl -fL --progress-meter -o "$2" "$1"
    fi
}

TMP=$(mktemp -d) || { warn "mktemp failed"; exit 0; }
trap 'rm -rf "$TMP"' EXIT

BASE="https://huggingface.co/buckets/$HF_BUCKET/resolve/$VERSION"
echo "webui: downloading prebuilt llama.cpp UI ('$VERSION') from $BASE"

if ! dl "$BASE/dist.tar.gz?download=true" "$TMP/dist.tar.gz"; then
    warn "download of dist.tar.gz failed; building without a web UI"
    warn "retry later with: tools/fetch_webui.sh --force"
    exit 0
fi
if ! dl "$BASE/dist.tar.gz.sha256?download=true" "$TMP/dist.tar.gz.sha256"; then
    warn "download of dist.tar.gz.sha256 failed; building without a web UI"
    exit 0
fi

expected=$(awk '{print $1}' "$TMP/dist.tar.gz.sha256")
actual=$(sha256sum "$TMP/dist.tar.gz" | awk '{print $1}')
if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
    warn "checksum mismatch for dist.tar.gz (expected $expected, got $actual); building without a web UI"
    exit 0
fi

mkdir -p "$TMP/extracted"
if ! tar -xzf "$TMP/dist.tar.gz" -C "$TMP/extracted"; then
    warn "failed to extract dist.tar.gz; building without a web UI"
    exit 0
fi
if [ ! -f "$TMP/extracted/index.html" ]; then
    warn "archive is missing index.html; building without a web UI"
    exit 0
fi

rm -rf "$WEBUI_DIR"
mkdir -p "$(dirname -- "$WEBUI_DIR")"
mv "$TMP/extracted" "$WEBUI_DIR"
echo "$VERSION" > "$STAMP_FILE"
echo "webui: ready in $WEBUI_DIR ($VERSION)"
