#!/bin/sh
set -e

REPO="antirez/deepseek-v4-gguf"
Q2_IMATRIX_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
Q2_Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf"
PRO_Q2_IMATRIX_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"
PRO_Q4_LAYERS00_30_FILE="DeepSeek-V4-Pro-Q4K-Layers00-30.gguf"
PRO_Q4_LAYERS31_OUTPUT_FILE="DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf"
MTP_FILE="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
DeepSeek V4 GGUF downloader

Usage:
  ./download_model.sh q2-imatrix [--token TOKEN]
  ./download_model.sh q2-q4-imatrix [--token TOKEN]
  ./download_model.sh q4-imatrix [--token TOKEN]
  ./download_model.sh pro-q2-imatrix [--token TOKEN]
  ./download_model.sh pro-q4-layers00-30 [--token TOKEN]
  ./download_model.sh pro-q4-layers31-output [--token TOKEN]
  ./download_model.sh pro-q4-split [--token TOKEN]
  ./download_model.sh mtp [--token TOKEN]

Targets:

  q2-imatrix
       2-bit routed experts, about 81 GB on disk.
       Recommended model for 96 and 128 GB RAM machines.

  q2-q4-imatrix
       Mixed Flash quant: mostly q2 routed experts, with the last 6 layers
       using q4 routed experts. About 98 GB on disk. Good for higher
       quality inference for 128 GB MacBooks. Works on DGX Spark but loading
       may struggle compared to q2-imatrix.

  q4-imatrix
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  pro-q2-imatrix
       DeepSeek V4 PRO q2 imatrix quant, as a single GGUF file. About 430 GB
       on disk; intended for 512 GB RAM machines.

  pro-q4-layers00-30
       First half of the DeepSeek V4 PRO Q4 routed-expert quant, layers 0..30.
       Use on the coordinator in a two-Mac-Studio distributed run. About 426 GB.

  pro-q4-layers31-output
       Second half of the DeepSeek V4 PRO Q4 routed-expert quant, layers
       31..output. Use on the worker in a two-Mac-Studio distributed run.
       About 412 GB.

  pro-q4-split
       Downloads both PRO Q4 split files into the download directory. About
       838 GB total. This target does not update ./ds4flash.gguf.

  mtp  Optional speculative decoding component, about 3.5 GB on disk.
       It is useful with q2-imatrix, q2-q4-imatrix, and q4-imatrix, but must be
       enabled explicitly with --mtp when running ds4 or ds4-server.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache is used if present.

Environment:
  DS4_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

After main-model downloads the script updates:
  ./ds4flash.gguf -> <download directory>/<selected model>

Then the default commands work:
  ./ds4 -p "Hello"
  ./ds4-server --ctx 100000

After downloading mtp, enable it explicitly, for example:
  ./ds4 --mtp <download directory>/$MTP_FILE --mtp-draft 2

PRO files are downloaded with the official Hugging Face downloader because
they are too large for the curl path used by the smaller GGUF files.
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift
MODEL_FILES=
LINK_MODEL=1

case "$MODEL" in
    q2-imatrix) MODEL_FILE=$Q2_IMATRIX_FILE ;;
    q2-q4-imatrix) MODEL_FILE=$Q2_Q4_IMATRIX_FILE ;;
    q4-imatrix) MODEL_FILE=$Q4_IMATRIX_FILE ;;
    pro-q2-imatrix) MODEL_FILE=$PRO_Q2_IMATRIX_FILE ;;
    pro-q4-layers00-30) MODEL_FILE=$PRO_Q4_LAYERS00_30_FILE; LINK_MODEL=0 ;;
    pro-q4-layers31-output) MODEL_FILE=$PRO_Q4_LAYERS31_OUTPUT_FILE; LINK_MODEL=0 ;;
    pro-q4-split)
        MODEL_FILES="$PRO_Q4_LAYERS00_30_FILE $PRO_Q4_LAYERS31_OUTPUT_FILE"
        LINK_MODEL=0
        ;;
    mtp) MODEL_FILE=$MTP_FILE; LINK_MODEL=0 ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

needs_hf_download() {
    case "$1" in
        "$PRO_Q2_IMATRIX_FILE"|"$PRO_Q4_LAYERS00_30_FILE"|"$PRO_Q4_LAYERS31_OUTPUT_FILE")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

find_hf_command() {
    if command -v hf >/dev/null 2>&1; then
        printf '%s\n' hf
        return 0
    fi
    return 1
}

download_one_hf() {
    file=$1
    out="$OUT_DIR/$file"
    part="$out.part"

    mkdir -p "$OUT_DIR"

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    if [ -e "$part" ]; then
        echo "Found curl partial download: $part" >&2
        echo "The Hugging Face downloader cannot resume curl .part files." >&2
        echo "Move or remove that partial download before retrying this PRO target." >&2
        exit 1
    fi

    HF_CMD=$(find_hf_command || true)
    if [ -z "$HF_CMD" ]; then
        echo "PRO downloads require the official Hugging Face CLI." >&2
        echo "Install it with:" >&2
        echo "  python3 -m pip install -U huggingface_hub hf_xet" >&2
        exit 1
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "using $HF_CMD download"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        "$HF_CMD" download "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR" --token "$TOKEN"
    else
        "$HF_CMD" download "$REPO" "$file" --repo-type model --local-dir "$OUT_DIR"
    fi

    if [ ! -s "$out" ]; then
        echo "Hugging Face download finished but expected file is missing: $out" >&2
        exit 1
    fi
}

download_one() {
    file=$1
    out="$OUT_DIR/$file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$REPO/resolve/main/$file"

    if needs_hf_download "$file"; then
        download_one_hf "$file"
        return
    fi

    mkdir -p "$OUT_DIR"

    if [ -e "$aria2_part" ]; then
        echo "Found incomplete aria2 download sidecar: $aria2_part" >&2
        echo "Finish or remove that partial download before using this curl downloader." >&2
        exit 1
    fi

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$REPO"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

if [ -n "$MODEL_FILES" ]; then
    for file in $MODEL_FILES; do
        download_one "$file"
    done
else
    download_one "$MODEL_FILE"
fi

if [ "$MODEL" = "mtp" ]; then
    echo
    echo "MTP is an optional component for q2-imatrix, q2-q4-imatrix, and q4-imatrix."
    echo "Enable it explicitly, for example:"
    echo "  ./ds4 --mtp $OUT_DIR/$MTP_FILE --mtp-draft 2"
elif [ "$MODEL" = "pro-q4-layers00-30" ] || [ "$MODEL" = "pro-q4-layers31-output" ] || [ "$MODEL" = "pro-q4-split" ]; then
    echo
    echo "Downloaded PRO Q4 distributed split file(s). Use them with --layers,"
    echo "for example coordinator layers 0:30 and worker layers 31:output."
elif [ "$LINK_MODEL" -eq 1 ]; then
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
    echo "Linked ./ds4flash.gguf -> $OUT_DIR/$MODEL_FILE"
fi

echo
echo "Done."
