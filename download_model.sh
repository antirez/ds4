#!/bin/sh
set -e

GLM_UNSLOTH_REPO="unsloth/GLM-5.2-GGUF"
GLM_ANTIREZ_REPO="antirez/GLM-5.2-GGUF"
GLM53_REPO="antirez/glm-5.3-flash-gguf"
GLM53_FULL_REPO="antirez/glm-5.3-gguf"
REPO="antirez/deepseek-v4-gguf"
DS41_REPO="antirez/deepseek-v4.1-flash-gguf"
DS41_Q2_FILE="DeepSeek-V4.1-Flash-Q2.gguf"
DS41_Q4_FILE="DeepSeek-V4.1-Flash-Q4.gguf"
DS41_VISION_FILE="DeepSeek-V4.1-Flash-Vision.gguf"
DS4F_Q2_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
DS4F_Q4_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf"
DS4F_MXFP4_FILE="DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf"
DS4F_Q2_Q4_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"
PRO_Q2_IMATRIX_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix-0813.gguf"
PRO_Q4_LAYERS00_30_FILE="DeepSeek-V4-Pro-Q4K-Layers00-30.gguf"
PRO_Q4_LAYERS31_OUTPUT_FILE="DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf"
DS4F_DSPARK_FILE="DeepSeek-V4-Flash-DSpark-support-0731.gguf"
DS4F_VISION_Q2_FILE="DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf"
DS4F_VISION_Q2_Q4_FILE="DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf"
DS4F_VISION_MXFP4_FILE="DeepSeek-V4-Flash-Vision-Exp-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out.gguf"
DS4F_VISION_ENCODER_FILE="DeepSeek-V4-Flash-Vision-Encoder.gguf"
DS4F_VISION_DSPARK_FILE="DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf"
GLM_UNSLOTH_Q4_REMOTE_BASE="UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL"
GLM_UNSLOTH_Q4_LOCAL_BASE="GLM-5.2-UD-Q4_K_XL"
GLM_UNSLOTH_Q4_FIRST_FILE="$GLM_UNSLOTH_Q4_LOCAL_BASE-00001-of-00011.gguf"
GLM_ANTIREZ_IQ2XXS_FILE="GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
GLM_ANTIREZ_Q2_FILE="GLM-5.2-UD-Q2_K_RoutedQ2K.gguf"
GLM_ANTIREZ_Q4_FILE="GLM-5.2-UD-Q4_K_RoutedQ4K.gguf"
GLM53_FULL_Q2_FILE="GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
GLM53_Q2_FILE="GLM-5.3-Flash-Q2.gguf"
GLM53_Q4_FILE="GLM-5.3-Flash-Q4_K.gguf"
GLM53_FP8_FILE="GLM-5.3-Flash-FP8.gguf"
GLM53_VISION_FILE="GLM-5.3-Flash-Vision-Encoder.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
DwarfStar GGUF downloader

Usage:
  ./download_model.sh ds4f-q2 [--token TOKEN]
  ./download_model.sh ds4f-q2-q4 [--token TOKEN]
  ./download_model.sh ds4f-q4 [--token TOKEN]
  ./download_model.sh ds4f-mxfp4 [--token TOKEN]
  ./download_model.sh ds4f-dspark [--token TOKEN]
  ./download_model.sh ds4f-vision-q2 [--token TOKEN]
  ./download_model.sh ds4f-vision-q2-q4 [--token TOKEN]
  ./download_model.sh ds4f-vision-mxfp4 [--token TOKEN]
  ./download_model.sh ds4f-vision-encoder [--token TOKEN]
  ./download_model.sh ds4f-vision-dspark [--token TOKEN]
  ./download_model.sh ds41f-q2 [--token TOKEN]
  ./download_model.sh ds41f-q4 [--token TOKEN]
  ./download_model.sh ds41f-vision [--token TOKEN]
  ./download_model.sh pro-q2-imatrix [--token TOKEN]
  ./download_model.sh pro-q4-layers00-30 [--token TOKEN]
  ./download_model.sh pro-q4-layers31-output [--token TOKEN]
  ./download_model.sh pro-q4-split [--token TOKEN]
  ./download_model.sh glm-unsloth-q4 [--token TOKEN]
  ./download_model.sh glm-antirez-iq2xxs [--token TOKEN]
  ./download_model.sh glm-antirez-q2 [--token TOKEN]
  ./download_model.sh glm-antirez-q4 [--token TOKEN]
  ./download_model.sh glm53-full-q2 [--token TOKEN]
  ./download_model.sh glm53-q2 [--token TOKEN]
  ./download_model.sh glm53-q4 [--token TOKEN]
  ./download_model.sh glm53-fp8 [--token TOKEN]
  ./download_model.sh glm53-vision [--token TOKEN]

Targets:

  ds4f-q2
       2-bit routed experts, about 81 GB on disk.
       Recommended model for 96 and 128 GB RAM machines.

  ds4f-q2-q4
       Mixed Flash quant: mostly q2 routed experts, with the last 6 layers
       using q4 routed experts. About 98 GB on disk. Good for higher
       quality inference for 128 GB MacBooks. Works on DGX Spark but loading
       may struggle compared to ds4f-q2.

  ds4f-q4
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  ds4f-mxfp4
       Native DeepSeek V4 Flash MXFP4 routed experts, about 156 GB on disk.
       Supported by Metal and CUDA; Blackwell uses FP4 tensor cores for batched
       expert work, while CUDA decode keeps Q8 activations.

  ds4f-dspark
       Optional DSpark speculative decoding support GGUF for Flash 0731, about
       6 GB. Enable it with --dspark and --mtp-model when running ds4 or ds4-server.

  ds4f-vision-q2
       DeepSeek V4 Flash Vision Experimental with 2-bit routed experts, about
       81 GiB, plus its vision encoder. Recommended for 96 and 128 GB Macs.

  ds4f-vision-q2-q4
       Mixed Vision Experimental quant with routed experts in layers 37-42 at
       Q4_K and the others at 2 bits, about 91 GiB, plus its vision encoder.

  ds4f-vision-mxfp4
       Native MXFP4 Vision Experimental model, about 145 GiB, plus its vision
       encoder. Intended for CUDA or two 128 GB Macs using tensor parallelism.

  ds4f-vision-encoder
       Standalone Vision Experimental encoder, about 0.9 GiB. Use this when
       the matching language GGUF is already present.

  ds4f-vision-dspark
       Matching DSpark speculative decoding support for Vision Experimental,
       about 5.6 GiB. It is not compatible with the 0731 language checkpoint.

  ds41f-q2
       DeepSeek V4.1 Flash calibrated Q2, about 341 GiB on disk. Main weights
       occupy 152 GiB; Engram tables stay on disk. Metal only: use SSD streaming
       on one 128 GB Mac, tensor parallelism on two, or a larger resident Mac.

  ds41f-vision
       Matching V4.1 Flash vision encoder, about 0.9 GiB. Add --vision FILE
       to the language-model command. Does not update ./ds4flash.gguf.

  ds41f-q4
       DeepSeek V4.1 Flash calibrated Q4, about 483 GiB on disk. Main weights
       occupy 294 GiB; Engram tables stay on disk. Metal only: use SSD streaming
       on smaller Macs or full residency on a 512 GB Mac. Downloads two parts
       and joins them automatically; allow another 37 GiB of free disk space.

  pro-q2-imatrix
       DeepSeek V4 PRO 0813 q2 imatrix quant, as a single GGUF file. About
       430 GB on disk; intended for 512 GB RAM machines.

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

  glm-unsloth-q4
       GLM 5.2 Unsloth UD-Q4_K_XL quant from unsloth/GLM-5.2-GGUF.
       Downloads all 11 shards and links ./ds4flash.gguf to the first shard.

  glm-antirez-iq2xxs
       GLM 5.2 antirez routed IQ2_XXS GGUF from antirez/GLM-5.2-GGUF.
       Includes Q2_K block 78 and is intended for reduced-memory testing.

  glm-antirez-q2
       GLM 5.2 antirez routed Q2_K GGUF from antirez/GLM-5.2-GGUF.
       About 262 GB on disk.

  glm-antirez-q4
       GLM 5.2 antirez routed Q4_K GGUF from antirez/GLM-5.2-GGUF.
       About 434 GB on disk.

  glm53-full-q2
       Full GLM 5.3 routed IQ2_XXS/Q2_K GGUF, about 197 GiB on disk.
       Intended for 256 GB machines or SSD streaming on smaller systems.

  glm53-q2
       GLM 5.3 Flash imatrix Q2 GGUF, about 90 GiB on disk. Intended for
       resident inference on 128 GB Macs.

  glm53-q4
       GLM 5.3 Flash Q4_K GGUF, about 178 GiB on disk. Intended for two-Mac
       tensor parallelism or single-Mac SSD streaming.

  glm53-fp8
       Text-only GLM 5.3 Flash native FP8 GGUF, about 305 GiB on disk. It
       preserves the released weights without requantization. DwarfStar
       inference support for this paired FP8-code/scale format is pending.

  glm53-vision
       GLM 5.3 Flash vision encoder, about 1.1 GB on disk. Load it separately
       with --vision; this target does not update ./ds4flash.gguf.

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

After downloading DSpark support, enable it explicitly:
  ./ds4 --dspark --mtp-model <download directory>/$DS4F_DSPARK_FILE

PRO, V4.1 and GLM files use the official Hugging Face downloader
because they are too large, sharded, or nested for the curl path used by the
smaller DeepSeek Flash GGUF files.
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
FORCE_HF_DOWNLOAD=0
FLATTEN_DOWNLOADS=0

case "$MODEL" in
    ds4f-q2) MODEL_FILE=$DS4F_Q2_FILE ;;
    ds4f-q2-q4) MODEL_FILE=$DS4F_Q2_Q4_FILE ;;
    ds4f-q4) MODEL_FILE=$DS4F_Q4_FILE ;;
    ds4f-mxfp4) MODEL_FILE=$DS4F_MXFP4_FILE; FORCE_HF_DOWNLOAD=1 ;;
    ds4f-dspark) MODEL_FILE=$DS4F_DSPARK_FILE; LINK_MODEL=0 ;;
    ds4f-vision-q2)
        MODEL_FILE=$DS4F_VISION_Q2_FILE
        MODEL_FILES="$MODEL_FILE $DS4F_VISION_ENCODER_FILE"
        FORCE_HF_DOWNLOAD=1
        ;;
    ds4f-vision-q2-q4)
        MODEL_FILE=$DS4F_VISION_Q2_Q4_FILE
        MODEL_FILES="$MODEL_FILE $DS4F_VISION_ENCODER_FILE"
        FORCE_HF_DOWNLOAD=1
        ;;
    ds4f-vision-mxfp4)
        MODEL_FILE=$DS4F_VISION_MXFP4_FILE
        MODEL_FILES="$MODEL_FILE $DS4F_VISION_ENCODER_FILE"
        FORCE_HF_DOWNLOAD=1
        ;;
    ds4f-vision-encoder)
        MODEL_FILE=$DS4F_VISION_ENCODER_FILE
        FORCE_HF_DOWNLOAD=1
        LINK_MODEL=0
        ;;
    ds4f-vision-dspark)
        MODEL_FILE=$DS4F_VISION_DSPARK_FILE
        FORCE_HF_DOWNLOAD=1
        LINK_MODEL=0
        ;;
    ds41f-q2)
        REPO=$DS41_REPO
        MODEL_FILE=$DS41_Q2_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    ds41f-q4)
        REPO=$DS41_REPO
        MODEL_FILE=$DS41_Q4_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    ds41f-vision)
        REPO=$DS41_REPO
        MODEL_FILE=$DS41_VISION_FILE
        FORCE_HF_DOWNLOAD=1
        LINK_MODEL=0
        ;;
    pro-q2-imatrix) MODEL_FILE=$PRO_Q2_IMATRIX_FILE ;;
    pro-q4-layers00-30) MODEL_FILE=$PRO_Q4_LAYERS00_30_FILE; LINK_MODEL=0 ;;
    pro-q4-layers31-output) MODEL_FILE=$PRO_Q4_LAYERS31_OUTPUT_FILE; LINK_MODEL=0 ;;
    pro-q4-split)
        MODEL_FILES="$PRO_Q4_LAYERS00_30_FILE $PRO_Q4_LAYERS31_OUTPUT_FILE"
        LINK_MODEL=0
        ;;
    glm-unsloth-q4)
        REPO=$GLM_UNSLOTH_REPO
        MODEL_FILE=$GLM_UNSLOTH_Q4_FIRST_FILE
        MODEL_FILES=
        for part in 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010 00011; do
            MODEL_FILES="$MODEL_FILES $GLM_UNSLOTH_Q4_REMOTE_BASE-${part}-of-00011.gguf"
        done
        FORCE_HF_DOWNLOAD=1
        FLATTEN_DOWNLOADS=1
        ;;
    glm-antirez-q2)
        REPO=$GLM_ANTIREZ_REPO
        MODEL_FILE=$GLM_ANTIREZ_Q2_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    glm-antirez-iq2xxs)
        REPO=$GLM_ANTIREZ_REPO
        MODEL_FILE=$GLM_ANTIREZ_IQ2XXS_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    glm-antirez-q4)
        REPO=$GLM_ANTIREZ_REPO
        MODEL_FILE=$GLM_ANTIREZ_Q4_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    glm53-full-q2)
        REPO=$GLM53_FULL_REPO
        MODEL_FILE=$GLM53_FULL_Q2_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    glm53-q2)
        REPO=$GLM53_REPO
        MODEL_FILE=$GLM53_Q2_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    glm53-q4)
        REPO=$GLM53_REPO
        MODEL_FILE=$GLM53_Q4_FILE
        FORCE_HF_DOWNLOAD=1
        ;;
    glm53-fp8)
        REPO=$GLM53_REPO
        MODEL_FILE=$GLM53_FP8_FILE
        FORCE_HF_DOWNLOAD=1
        LINK_MODEL=0
        ;;
    glm53-vision)
        REPO=$GLM53_REPO
        MODEL_FILE=$GLM53_VISION_FILE
        FORCE_HF_DOWNLOAD=1
        LINK_MODEL=0
        ;;
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
    if [ "${FORCE_HF_DOWNLOAD:-0}" -eq 1 ]; then
        return 0
    fi
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
    for dir in "$HOME"/Library/Python/*/bin "$HOME"/.local/bin; do
        if [ -x "$dir/hf" ]; then
            printf '%s\n' "$dir/hf"
            return 0
        fi
    done
    return 1
}

local_download_name() {
    if [ "${FLATTEN_DOWNLOADS:-0}" -eq 1 ]; then
        basename "$1"
    else
        printf '%s\n' "$1"
    fi
}

artifact_identity() {
    case "$1" in
        "$DS41_Q2_FILE")
            expected_bytes=365713686528
            expected_sha=1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42
            ;;
        "$DS41_Q4_FILE")
            expected_bytes=518596067328
            expected_sha=a5e2e2c3ada4b2e98d9f9e4b50f6d9c2a12c2c96f5da165c07e13aff9264984e
            ;;
        "$DS41_Q4_FILE.part1")
            expected_bytes=480000000000
            expected_sha=6442b1f9224079662c02003c0ef9ef6be6e2aff509510f681dab9e6cc41df246
            ;;
        "$DS41_Q4_FILE.part2")
            expected_bytes=38596067328
            expected_sha=7c3e10646c918eeaffbc39305a75ec96117450262c61454ff194cef00d7617f0
            ;;
        "$DS41_VISION_FILE")
            expected_bytes=970555552
            expected_sha=cc283f032b3e8b8d78aeb5fccaa14e97b859b0c53aae3cd6bffa690ddf0e9e15
            ;;
        *) return 1 ;;
    esac
}

verify_download() {
    artifact_identity "$1" || return 0
    if [ "$(wc -c < "$2")" -ne "$expected_bytes" ]; then
        echo "Incorrect file size: $2. Move the incomplete file aside and retry." >&2
        exit 1
    fi
    echo "Verifying SHA-256: $2"
    if command -v sha256sum >/dev/null 2>&1; then
        actual_sha=$(sha256sum < "$2")
    else
        actual_sha=$(shasum -a 256 < "$2")
    fi
    actual_sha=${actual_sha%% *}
    if [ "$actual_sha" != "$expected_sha" ]; then
        echo "Checksum mismatch: $2. The file was not accepted." >&2
        exit 1
    fi
}

download_one_hf() {
    file=$1
    local_file=$(local_download_name "$file")
    out="$OUT_DIR/$local_file"
    hf_out="$OUT_DIR/$file"
    part="$out.part"

    mkdir -p "$(dirname "$out")"

    if [ -s "$out" ]; then
        verify_download "$file" "$out"
        echo "Already downloaded: $out"
        return
    fi

    if [ -e "$part" ]; then
        echo "Found curl partial download: $part" >&2
        echo "The Hugging Face downloader cannot resume curl .part files." >&2
        echo "Move or remove that partial download before retrying this target." >&2
        exit 1
    fi

    HF_CMD=$(find_hf_command || true)
    if [ -z "$HF_CMD" ]; then
        echo "Large GGUF downloads require the official Hugging Face CLI." >&2
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

    if [ "$hf_out" != "$out" ] && [ -s "$hf_out" ]; then
        mv "$hf_out" "$out"
        rmdir "$(dirname "$hf_out")" 2>/dev/null || true
    fi

    if [ ! -s "$out" ]; then
        echo "Hugging Face download finished but expected file is missing: $out" >&2
        exit 1
    fi
    verify_download "$file" "$out"
}

download_one() {
    file=$1
    local_file=$(local_download_name "$file")
    out="$OUT_DIR/$local_file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$REPO/resolve/main/$file"

    if needs_hf_download "$file"; then
        download_one_hf "$file"
        return
    fi

    mkdir -p "$(dirname "$out")"

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

download_ds41_q4() {
    q4_out="$OUT_DIR/$DS41_Q4_FILE"
    if [ -e "$q4_out" ]; then
        verify_download "$DS41_Q4_FILE" "$q4_out"
        echo "Already downloaded: $q4_out"
        return
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Joining the Q4 download requires Python 3." >&2
        exit 1
    fi
    if [ ! -e "$q4_out.assembling" ]; then
        download_one_hf "$DS41_Q4_FILE.part1"
    fi
    download_one_hf "$DS41_Q4_FILE.part2"
    artifact_identity "$DS41_Q4_FILE"
    q4_bytes=$expected_bytes
    q4_sha=$expected_sha
    artifact_identity "$DS41_Q4_FILE.part1"
    python3 - "$q4_out" "$expected_bytes" "$q4_bytes" "$q4_sha" <<'PY'
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import sys

out = Path(sys.argv[1])
boundary, total = map(int, sys.argv[2:4])
expected = sys.argv[4]
first, second, pending = (Path(str(out) + suffix)
                          for suffix in (".part1", ".part2", ".assembling"))

def verify(path):
    if path.stat().st_size != total:
        sys.exit("Incorrect file size: " + str(path))
    print("Verifying SHA-256: " + str(path), flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(16 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        sys.exit("Checksum mismatch: " + str(path) +
                 ". Move this file aside before retrying; it was not accepted.")

# Keep this lock file: unlinking it could allow two different locks for the
# same download. The first part becomes the output without a second full copy.
with Path(str(pending) + ".lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if out.exists():
        verify(out)
        sys.exit(0)
    if not pending.exists():
        if first.stat().st_size != boundary:
            sys.exit("Incorrect file size: " + str(first))
        first.rename(pending)
    size = pending.stat().st_size
    if not boundary <= size <= total:
        sys.exit("Invalid partial assembly: " + str(pending) +
                 ". Move it aside before retrying.")
    if second.stat().st_size != total - boundary:
        sys.exit("Incorrect file size: " + str(second))
    # An interrupted append restarts only the smaller tail, never the prefix.
    with pending.open("r+b") as dst:
        dst.truncate(boundary)
        dst.seek(boundary)
        if shutil.disk_usage(out.parent).free < total - boundary + (1 << 30):
            sys.exit("Not enough disk space to join Q4; keep the parts and retry.")
        print("Joining Q4 download parts", flush=True)
        with second.open("rb") as src:
            shutil.copyfileobj(src, dst, 16 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    verify(pending)
    pending.rename(out)
    second.unlink()
PY
}

if [ "$MODEL" = "ds41f-q4" ]; then
    download_ds41_q4
elif [ -n "$MODEL_FILES" ]; then
    for file in $MODEL_FILES; do
        download_one "$file"
    done
else
    download_one "$MODEL_FILE"
fi

if [ "$MODEL" = "ds4f-dspark" ]; then
    echo
    echo "DSpark support downloaded. Enable it explicitly:"
    echo "  ./ds4 --dspark -m ./ds4flash.gguf --mtp-model $OUT_DIR/$DS4F_DSPARK_FILE"
elif [ "$MODEL" = "ds4f-vision-dspark" ]; then
    echo
    echo "Vision Experimental DSpark support downloaded. Use it only with the matching checkpoint:"
    echo "  ./ds4 --dspark -m ./ds4flash.gguf --mtp-model $OUT_DIR/$DS4F_VISION_DSPARK_FILE"
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
