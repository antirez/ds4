#!/bin/sh
set -e

GLM_UNSLOTH_REPO="unsloth/GLM-5.2-GGUF"
GLM_ANTIREZ_REPO="antirez/GLM-5.2-GGUF"
QWEN_GGML_REPO="ggml-org/Qwen3.8-27B-GGUF"
QWEN_DFLASH_REPO="z-lab/Qwen3.8-27B-DFlash2-GGUF"
REPO="antirez/deepseek-v4-gguf"
HEADROOM128_REPO="apetersson/DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128"
HEADROOM128_FILE="DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128.gguf"
HEADROOM128_DSPARK_SUPPORT_FILE="DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128-DSpark-support.gguf"
DS4F_Q2_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
DS4F_Q4_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf"
DS4F_MXFP4_FILE="DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf"
DS4F_Q2_Q4_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"
PRO_Q2_IMATRIX_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"
PRO_Q4_LAYERS00_30_FILE="DeepSeek-V4-Pro-Q4K-Layers00-30.gguf"
PRO_Q4_LAYERS31_OUTPUT_FILE="DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf"
DS4F_DSPARK_FILE="DeepSeek-V4-Flash-DSpark-support-0731.gguf"
GLM_UNSLOTH_Q4_REMOTE_BASE="UD-Q4_K_XL/GLM-5.2-UD-Q4_K_XL"
GLM_UNSLOTH_Q4_LOCAL_BASE="GLM-5.2-UD-Q4_K_XL"
GLM_UNSLOTH_Q4_FIRST_FILE="$GLM_UNSLOTH_Q4_LOCAL_BASE-00001-of-00011.gguf"
GLM_ANTIREZ_IQ2XXS_FILE="GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
GLM_ANTIREZ_Q2_FILE="GLM-5.2-UD-Q2_K_RoutedQ2K.gguf"
GLM_ANTIREZ_Q4_FILE="GLM-5.2-UD-Q4_K_RoutedQ4K.gguf"
QWEN38_Q4_FILE="Qwen3.8-27B-Q4_K_M.gguf"
QWEN38_Q8_FILE="Qwen3.8-27B-Q8_0.gguf"
QWEN38_DFLASH_Q4_FILE="Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
QWEN38_DFLASH_Q8_FILE="Qwen3.8-27B-DFlash2-Q8_0.gguf"
QWEN38_DFLASH_BF16_FILE="Qwen3.8-27B-DFlash2-BF16.gguf"
ORNITH_REPO="ornith-ai/Ornith-1.5-35B-A3B-GGUF"
ORNITH_Q4_FILE="Ornith-1.5-35B-Q4_K_M.gguf"
ORNITH_Q5_FILE="Ornith-1.5-35B-Q5_K_M.gguf"
ORNITH_Q6_FILE="Ornith-1.5-35B-Q6_K.gguf"
ORNITH_Q8_FILE="Ornith-1.5-35B-Q8_0.gguf"
ORNITH_BF16_FILE="Ornith-1.5-35B-BF16.gguf"
ORNITH_MMPROJ_FILE="mmproj-Ornith-1.5-35B-BF16.gguf"
ORNITH9_REPO="ornith-ai/Ornith-1.5-9B-GGUF"
ORNITH9_Q4_FILE="Ornith-1.5-9B-Q4_K_M.gguf"
ORNITH9_Q5_FILE="Ornith-1.5-9B-Q5_K_M.gguf"
ORNITH9_Q6_FILE="Ornith-1.5-9B-Q6_K.gguf"
ORNITH9_Q8_FILE="Ornith-1.5-9B-Q8_0.gguf"
ORNITH9_BF16_FILE="Ornith-1.5-9B-BF16.gguf"
ORNITH9_MMPROJ_FILE="mmproj-Ornith-1.5-9B-BF16.gguf"
ORNITH9_DFLASH_REPO="audreyt/Ornith-1.5-9B-DFlash-GGUF"
ORNITH9_DFLASH_Q4_FILE="ornith1.5-9b-dflash-bf16-projection-Q4_K_M.gguf"
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
  ./download_model.sh headroom128 [--token TOKEN]
  ./download_model.sh preferred [--token TOKEN]
  ./download_model.sh headroom128-dspark-support [--token TOKEN]
  ./download_model.sh ds4f-q2 [--token TOKEN]
  ./download_model.sh ds4f-q2-q4 [--token TOKEN]
  ./download_model.sh ds4f-q4 [--token TOKEN]
  ./download_model.sh ds4f-mxfp4 [--token TOKEN]
  ./download_model.sh ds4f-dspark [--token TOKEN]
  ./download_model.sh q2-imatrix [--token TOKEN]
  ./download_model.sh q2-q4-imatrix [--token TOKEN]
  ./download_model.sh q4-imatrix [--token TOKEN]
  ./download_model.sh pro-q2-imatrix [--token TOKEN]
  ./download_model.sh pro-q4-layers00-30 [--token TOKEN]
  ./download_model.sh pro-q4-layers31-output [--token TOKEN]
  ./download_model.sh pro-q4-split [--token TOKEN]
  ./download_model.sh glm-unsloth-q4 [--token TOKEN]
  ./download_model.sh glm-antirez-iq2xxs [--token TOKEN]
  ./download_model.sh glm-antirez-q2 [--token TOKEN]
  ./download_model.sh glm-antirez-q4 [--token TOKEN]
  ./download_model.sh qwen-q4 [--token TOKEN]
  ./download_model.sh qwen [--token TOKEN]
  ./download_model.sh qwen3.8 [--token TOKEN]
  ./download_model.sh qwen-dflash [--token TOKEN]
  ./download_model.sh qwen-dflash-support [--token TOKEN]
  ./download_model.sh qwen-q8 [--token TOKEN]
  ./download_model.sh qwen-dflash-q8 [--token TOKEN]
  ./download_model.sh ornith-q4 [--token TOKEN]
  ./download_model.sh ornith [--token TOKEN]
  ./download_model.sh ornith-q6 [--token TOKEN]
  ./download_model.sh ornith-q8 [--token TOKEN]
  ./download_model.sh ornith-mmproj [--token TOKEN]
  ./download_model.sh ornith9-q4 [--token TOKEN]
  ./download_model.sh ornith9 [--token TOKEN]
  ./download_model.sh ornith9-q6 [--token TOKEN]
  ./download_model.sh ornith9-q8 [--token TOKEN]
  ./download_model.sh ornith9-mmproj [--token TOKEN]
  ./download_model.sh ornith9-dflash [--token TOKEN]
  ./download_model.sh ornith9-dflash-support [--token TOKEN]
Targets:

  headroom128 / preferred
       Preferred Flash GGUF for 96/128 GB machines on this fork.
       apetersson/DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128
       (~81 GiB / ~87 GB). Abliterated 0731 DS4 headroom build; links
       ./ds4flash.gguf.

  headroom128-dspark-support
       Matching DSpark support GGUF for headroom128 from the same
       apetersson Headroom128 repo, about 5.6 GiB. Enable with --dspark and
       --mtp when running the Headroom128 main model.

  ds4f-q2 / q2-imatrix
       2-bit routed experts, about 81 GB on disk.
       Official antirez stock 0731 imatrix for 96 and 128 GB RAM machines.

  ds4f-q2-q4 / q2-q4-imatrix
       Mixed Flash quant: mostly q2 routed experts, with the last 6 layers
       using q4 routed experts. About 98 GB on disk. Good for higher
       quality inference for 128 GB MacBooks. Works on DGX Spark but loading
       may struggle compared to ds4f-q2.

  ds4f-q4 / q4-imatrix
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  ds4f-mxfp4
       Native DeepSeek V4 Flash MXFP4 routed experts, about 156 GB on disk.
       Supported by Metal and CUDA; Blackwell uses FP4 tensor cores for batched
       expert work, while CUDA decode keeps Q8 activations.

  ds4f-dspark
       Optional DSpark speculative decoding support GGUF for Flash 0731, about
       6 GB. Enable it with --dspark and --mtp when running ds4 or ds4-server.

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

  qwen-q4 / qwen / qwen3.8 / qwen-base / qwen38
       Preferred default Qwen 3.8 27B Q4_K_M base model GGUF from
       ggml-org/Qwen3.8-27B-GGUF (~17.7 GiB). Links ./ds4flash.gguf and
       ./qwen38.gguf.

  qwen-dflash / qwen-combo
       Optional Qwen + DFlash2 combination. Downloads both the matching
       Qwen 3.8 27B Q4_K_M base model from ggml-org/Qwen3.8-27B-GGUF (~17.7 GiB)
       and the DFlash2 Q4_K_M draft model from z-lab/Qwen3.8-27B-DFlash2-GGUF
       (~1.06 GiB). Links ./ds4flash.gguf and ./qwen38.gguf.

  qwen-dflash-support / dflash2 / dflash
       Matching DFlash2 Q4_K_M draft GGUF from z-lab/Qwen3.8-27B-DFlash2-GGUF
       (~1.06 GiB). Enable with --dflash when running the Qwen base model.

  qwen-q8
       Qwen 3.8 27B Q8_0 base model GGUF from ggml-org/Qwen3.8-27B-GGUF
       (~26.6 GiB). Links ./ds4flash.gguf and ./qwen38.gguf.

  qwen-dflash-q8
       DFlash2 Q8_0 draft GGUF from z-lab/Qwen3.8-27B-DFlash2-GGUF (~1.92 GiB).

  ornith-q4 / ornith / ornith-1.5 / ornith-35b / ornith-1.5-35b / ornith-1.5-35b-a3b
       Ornith 1.5 35B A3B Q4_K_M text-only base model GGUF from
       ornith-ai/Ornith-1.5-35B-A3B-GGUF (~20.2 GiB / ~21.7 GB).
       Links ./ds4flash.gguf and ./ornith35.gguf.

  ornith-q6 / ornith-1.5-q6
       Ornith 1.5 35B A3B Q6_K text-only GGUF (~27.2 GiB).

  ornith-q8 / ornith-1.5-q8
       Ornith 1.5 35B A3B Q8_0 text-only GGUF (~35.2 GiB).

  ornith-mmproj / ornith-vision
       Optional CLIP vision encoder projector for Ornith 1.5
       (mmproj-Ornith-1.5-35B-BF16.gguf, ~0.84 GiB). Not required for
       text-only inference.

  ornith9-q4 / ornith9 / ornith-9b / ornith-1.5-9b / ornith-1.5-9b-q4
       Ornith 1.5 9B Q4_K_M text-only base model GGUF from
       ornith-ai/Ornith-1.5-9B-GGUF (~5.24 GiB / ~5.63 GB).
       Links ./ds4flash.gguf and ./ornith9.gguf.

  ornith9-q6 / ornith-9b-q6 / ornith-1.5-9b-q6
       Ornith 1.5 9B Q6_K text-only GGUF from ornith-ai/Ornith-1.5-9B-GGUF
       (~6.85 GiB / ~7.36 GB). Links ./ds4flash.gguf and ./ornith9.gguf.

  ornith9-q8 / ornith-9b-q8 / ornith-1.5-9b-q8
       Ornith 1.5 9B Q8_0 text-only GGUF from ornith-ai/Ornith-1.5-9B-GGUF
       (~8.87 GiB / ~9.53 GB). Links ./ds4flash.gguf and ./ornith9.gguf.

  ornith9-mmproj / ornith9-vision / ornith-9b-vision
       Optional CLIP vision encoder projector for Ornith 1.5 9B
       (mmproj-Ornith-1.5-9B-BF16.gguf, ~0.86 GiB / ~0.92 GB). Not required
       for text-only inference.
  ornith9-dflash / ornith-9b-dflash
       Preferred Ornith 1.5 9B speculative combination. Downloads the Ornith
       Q4_K_M target and the Ornith-specific distilled DFlash Q4_K_M draft from
       audreyt/Ornith-1.5-9B-DFlash-GGUF. Links ./ds4flash.gguf and
       ./ornith9.gguf to the target.

  ornith9-dflash-support
       Ornith-specific distilled DFlash Q4_K_M draft only. Enable it with
       --dflash when running the Ornith 1.5 9B target.

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

After downloading Headroom128 DSpark support, enable it explicitly in greedy mode:
  ./ds4 --dspark -m ./ds4flash.gguf --mtp <download directory>/$HEADROOM128_DSPARK_SUPPORT_FILE --temp 0

After downloading the official antirez DSpark support, enable it explicitly in greedy mode:
  ./ds4 --dspark --mtp <download directory>/$DS4F_DSPARK_FILE --temp 0

After downloading Qwen with DFlash2 draft support, run speculative decoding:
  ./ds4 -m ./ds4flash.gguf --dflash <download directory>/$QWEN38_DFLASH_Q4_FILE -p "Hello"

Or run Qwen server with DFlash2:
  ./ds4-server -m ./ds4flash.gguf --dflash <download directory>/$QWEN38_DFLASH_Q4_FILE --ctx 32768

PRO and GLM files are downloaded with the official Hugging Face downloader
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
DOWNLOAD_ITEMS=
LINK_MODEL=1
FORCE_HF_DOWNLOAD=0
FLATTEN_DOWNLOADS=0
case "$MODEL" in
    headroom128|preferred)
        REPO=$HEADROOM128_REPO
        MODEL_FILE=$HEADROOM128_FILE
        MODEL=headroom128
        ;;
    headroom128-dspark-support)
        REPO=$HEADROOM128_REPO
        MODEL_FILE=$HEADROOM128_DSPARK_SUPPORT_FILE
        LINK_MODEL=0
        ;;
    ds4f-q2|q2-imatrix) MODEL_FILE=$DS4F_Q2_FILE; MODEL=ds4f-q2 ;;
    ds4f-q2-q4|q2-q4-imatrix) MODEL_FILE=$DS4F_Q2_Q4_FILE; MODEL=ds4f-q2-q4 ;;
    ds4f-q4|q4-imatrix) MODEL_FILE=$DS4F_Q4_FILE; MODEL=ds4f-q4 ;;
    ds4f-mxfp4) MODEL_FILE=$DS4F_MXFP4_FILE; FORCE_HF_DOWNLOAD=1 ;;
    ds4f-dspark) MODEL_FILE=$DS4F_DSPARK_FILE; LINK_MODEL=0 ;;
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
    qwen-q4|qwen|qwen3.8|qwen-3.8|qwen38|qwen38-q4|qwen-base|qwen38-base)
        REPO=$QWEN_GGML_REPO
        MODEL_FILE=$QWEN38_Q4_FILE
        MODEL=qwen-q4
        ;;
    qwen-dflash|qwen-combo|qwen38-combo|qwen-dflash2|qwen38-dflash)
        DOWNLOAD_ITEMS="$QWEN_GGML_REPO:$QWEN38_Q4_FILE $QWEN_DFLASH_REPO:$QWEN38_DFLASH_Q4_FILE"
        MODEL_FILE=$QWEN38_Q4_FILE
        MODEL=qwen-dflash
        ;;
    qwen-dflash-support|qwen38-dflash-support|qwen-dflash2-support|dflash2|dflash|qwen-dflash-q4|qwen38-dflash-q4)
        REPO=$QWEN_DFLASH_REPO
        MODEL_FILE=$QWEN38_DFLASH_Q4_FILE
        LINK_MODEL=0
        MODEL=qwen-dflash-support
        ;;
    qwen-dflash-q8|qwen38-dflash-q8)
        REPO=$QWEN_DFLASH_REPO
        MODEL_FILE=$QWEN38_DFLASH_Q8_FILE
        LINK_MODEL=0
        MODEL=qwen-dflash-q8
        ;;
    qwen-dflash-bf16|qwen38-dflash-bf16)
        echo "Error: DFlash2 BF16 draft is not supported by ds4 runtime (use Q4_K_M or Q8_0 draft)" >&2
        exit 1
        ;;
    qwen-q8|qwen38-q8)
        REPO=$QWEN_GGML_REPO
        MODEL_FILE=$QWEN38_Q8_FILE
        MODEL=qwen-q8
        ;;
    ornith-q4|ornith|ornith-1.5|ornith-35b|ornith-1.5-35b|ornith-1.5-35b-a3b|ornith-1.5-35b-q4|ornith-35b-q4|ornith-q4_k_m|ornith-1.5-q4)
        REPO=$ORNITH_REPO
        MODEL_FILE=$ORNITH_Q4_FILE
        MODEL=ornith-q4
        ;;
    ornith-q5|ornith-1.5-q5|ornith-35b-q5|ornith-1.5-35b-q5|ornith-q5_k_m)
        echo "Error: Ornith 35B Q5_K_M is not supported by ds4 runtime (dense layers require Q4_K_M, Q6_K, or Q8_0)" >&2
        exit 1
        ;;
    ornith-q6|ornith-1.5-q6|ornith-35b-q6|ornith-1.5-35b-q6|ornith-q6_k)
        REPO=$ORNITH_REPO
        MODEL_FILE=$ORNITH_Q6_FILE
        MODEL=ornith-q6
        ;;
    ornith-q8|ornith-1.5-q8|ornith-35b-q8|ornith-1.5-35b-q8|ornith-q8_0)
        REPO=$ORNITH_REPO
        MODEL_FILE=$ORNITH_Q8_FILE
        MODEL=ornith-q8
        ;;
    ornith-bf16|ornith-1.5-bf16|ornith-35b-bf16|ornith-1.5-35b-bf16)
        echo "Error: Ornith 35B BF16 is not supported by ds4 runtime (dense layers require Q4_K_M, Q6_K, or Q8_0)" >&2
        exit 1
        ;;
    ornith-mmproj|ornith-1.5-mmproj|ornith-vision|ornith-clip)
        REPO=$ORNITH_REPO
        MODEL_FILE=$ORNITH_MMPROJ_FILE
        LINK_MODEL=0
        MODEL=ornith-mmproj
        ;;
    ornith9-dflash|ornith-9b-dflash|ornith9-combo)
        DOWNLOAD_ITEMS="$ORNITH9_REPO:$ORNITH9_Q4_FILE $ORNITH9_DFLASH_REPO:$ORNITH9_DFLASH_Q4_FILE"
        MODEL_FILE=$ORNITH9_Q4_FILE
        MODEL=ornith9-dflash
        ;;
    ornith9-dflash-support|ornith-9b-dflash-support)
        REPO=$ORNITH9_DFLASH_REPO
        MODEL_FILE=$ORNITH9_DFLASH_Q4_FILE
        LINK_MODEL=0
        MODEL=ornith9-dflash-support
        ;;
    ornith9-q4|ornith9|ornith-9b|ornith-1.5-9b|ornith-1.5-9b-q4|ornith-9b-q4|ornith9-q4_k_m)
        REPO=$ORNITH9_REPO
        MODEL_FILE=$ORNITH9_Q4_FILE
        MODEL=ornith9-q4
        ;;
    ornith9-q5|ornith-1.5-9b-q5|ornith-9b-q5|ornith9-q5_k_m)
        echo "Error: Ornith 9B Q5_K_M is not supported by ds4 runtime (dense layers require Q4_K_M, Q6_K, or Q8_0)" >&2
        exit 1
        ;;
    ornith9-q6|ornith-1.5-9b-q6|ornith-9b-q6|ornith9-q6_k)
        REPO=$ORNITH9_REPO
        MODEL_FILE=$ORNITH9_Q6_FILE
        MODEL=ornith9-q6
        ;;
    ornith9-q8|ornith-1.5-9b-q8|ornith-9b-q8|ornith9-q8_0)
        REPO=$ORNITH9_REPO
        MODEL_FILE=$ORNITH9_Q8_FILE
        MODEL=ornith9-q8
        ;;
    ornith9-bf16|ornith-1.5-9b-bf16|ornith-9b-bf16)
        echo "Error: Ornith 9B BF16 is not supported by ds4 runtime (dense layers require Q4_K_M, Q6_K, or Q8_0)" >&2
        exit 1
        ;;
    ornith9-mmproj|ornith-1.5-9b-mmproj|ornith9-vision|ornith-9b-vision)
        REPO=$ORNITH9_REPO
        MODEL_FILE=$ORNITH9_MMPROJ_FILE
        LINK_MODEL=0
        MODEL=ornith9-mmproj
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

download_one_hf() {
    repo=$1
    file=$2
    local_file=$(local_download_name "$file")
    out="$OUT_DIR/$local_file"
    hf_out="$OUT_DIR/$file"
    part="$out.part"

    mkdir -p "$(dirname "$out")"

    if [ -s "$out" ]; then
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
    echo "from https://huggingface.co/$repo"
    echo "using $HF_CMD download"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        "$HF_CMD" download "$repo" "$file" --repo-type model --local-dir "$OUT_DIR" --token "$TOKEN"
    else
        "$HF_CMD" download "$repo" "$file" --repo-type model --local-dir "$OUT_DIR"
    fi

    if [ "$hf_out" != "$out" ] && [ -s "$hf_out" ]; then
        mv "$hf_out" "$out"
        rmdir "$(dirname "$hf_out")" 2>/dev/null || true
    fi

    if [ ! -s "$out" ]; then
        echo "Hugging Face download finished but expected file is missing: $out" >&2
        exit 1
    fi
}

download_one() {
    repo=$1
    file=$2
    local_file=$(local_download_name "$file")
    out="$OUT_DIR/$local_file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$repo/resolve/main/$file"

    if needs_hf_download "$file"; then
        download_one_hf "$repo" "$file"
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
    echo "from https://huggingface.co/$repo"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

if [ -n "$DOWNLOAD_ITEMS" ]; then
    for item in $DOWNLOAD_ITEMS; do
        item_repo=${item%%:*}
        item_file=${item#*:}
        download_one "$item_repo" "$item_file"
    done
elif [ -n "$MODEL_FILES" ]; then
    for file in $MODEL_FILES; do
        download_one "$REPO" "$file"
    done
else
    download_one "$REPO" "$MODEL_FILE"
fi

if [ "$MODEL" = "ds4f-dspark" ] || [ "$MODEL" = "headroom128-dspark-support" ]; then
    echo
    echo "DSpark support downloaded. Enable it explicitly in greedy mode:"
    if [ "$MODEL" = "headroom128-dspark-support" ]; then
        echo "  ./ds4 --dspark -m ./ds4flash.gguf --mtp $OUT_DIR/$HEADROOM128_DSPARK_SUPPORT_FILE --temp 0"
    else
        echo "  ./ds4 --dspark -m ./ds4flash.gguf --mtp $OUT_DIR/$DS4F_DSPARK_FILE --temp 0"
    fi
elif [ "$MODEL" = "pro-q4-layers00-30" ] || [ "$MODEL" = "pro-q4-layers31-output" ] || [ "$MODEL" = "pro-q4-split" ]; then
    echo
    echo "Downloaded PRO Q4 distributed split file(s). Use them with --layers,"
    echo "for example coordinator layers 0:30 and worker layers 31:output."
elif [ "$LINK_MODEL" -eq 1 ]; then
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
    echo "Linked ./ds4flash.gguf -> $OUT_DIR/$MODEL_FILE"
    case "$MODEL" in
        qwen*)
            ln -sfn "$OUT_DIR/$MODEL_FILE" qwen38.gguf
            echo "Linked ./qwen38.gguf -> $OUT_DIR/$MODEL_FILE"
            ;;
        ornith9*)
            ln -sfn "$OUT_DIR/$MODEL_FILE" ornith9.gguf
            echo "Linked ./ornith9.gguf -> $OUT_DIR/$MODEL_FILE"
            ;;
        ornith*)
            ln -sfn "$OUT_DIR/$MODEL_FILE" ornith35.gguf
            echo "Linked ./ornith35.gguf -> $OUT_DIR/$MODEL_FILE"
            ;;
    esac
fi

if [ "$MODEL" = "headroom128" ]; then
    echo
    echo "Headroom128 has a matching DSpark support GGUF. Download it with:"
    echo "  ./download_model.sh headroom128-dspark-support"
    echo "Then enable DSpark explicitly in greedy mode:"
    echo "  ./ds4 --dspark -m ./ds4flash.gguf --mtp $OUT_DIR/$HEADROOM128_DSPARK_SUPPORT_FILE --temp 0"
elif [ "$MODEL" = "qwen-dflash" ]; then
    echo
    echo "Qwen 3.8 combination with DFlash2 draft model downloaded."
    echo "Run with DFlash2 block-diffusion speculative decoding:"
    echo "  ./ds4 -m ./ds4flash.gguf --dflash $OUT_DIR/$QWEN38_DFLASH_Q4_FILE -p \"Hello\""
    echo
    echo "Or start the server:"
    echo "  ./ds4-server -m ./ds4flash.gguf --dflash $OUT_DIR/$QWEN38_DFLASH_Q4_FILE --ctx 32768"
elif [ "$MODEL" = "qwen-dflash-support" ]; then
    echo
    echo "DFlash2 draft model downloaded. Run speculative decoding with:"
    echo "  ./ds4 -m ./ds4flash.gguf --dflash $OUT_DIR/$QWEN38_DFLASH_Q4_FILE -p \"Hello\""
elif [ "$MODEL" = "qwen-dflash-q8" ]; then
    echo
    echo "DFlash2 Q8_0 draft model downloaded. Run speculative decoding with:"
    echo "  ./ds4 -m ./ds4flash.gguf --dflash $OUT_DIR/$QWEN38_DFLASH_Q8_FILE -p \"Hello\""
elif [ "$MODEL" = "qwen-dflash-bf16" ]; then
    echo
    echo "DFlash2 BF16 draft model downloaded. Run speculative decoding with:"
    echo "  ./ds4 -m ./ds4flash.gguf --dflash $OUT_DIR/$QWEN38_DFLASH_BF16_FILE -p \"Hello\""
elif [ "$MODEL" = "qwen-q4" ] || [ "$MODEL" = "qwen-q8" ]; then
    echo
    echo "Qwen base model downloaded. For speculative decoding, download the matching DFlash2 draft model with:"
    echo "  ./download_model.sh qwen-dflash-support"
    echo "Then run:"
    echo "  ./ds4 -m ./ds4flash.gguf --dflash $OUT_DIR/$QWEN38_DFLASH_Q4_FILE -p \"Hello\""
elif [ "$MODEL" = "ornith9-dflash" ]; then
    echo
    echo "Ornith 1.5 9B and its distilled DFlash Q4_K_M draft downloaded."
    echo "Run speculative decoding with:"
    echo "  ./ds4 -m ./ornith9.gguf --dflash $OUT_DIR/$ORNITH9_DFLASH_Q4_FILE -p \"Hello\""
    echo "Or start the server:"
    echo "  ./ds4-server -m ./ornith9.gguf --dflash $OUT_DIR/$ORNITH9_DFLASH_Q4_FILE --ctx 32768"
elif [ "$MODEL" = "ornith9-dflash-support" ]; then
    echo
    echo "Ornith-specific distilled DFlash Q4_K_M draft downloaded. Run with:"
    echo "  ./ds4 -m ./ornith9.gguf --dflash $OUT_DIR/$ORNITH9_DFLASH_Q4_FILE -p \"Hello\""
elif [ "$MODEL" = "ornith9-q4" ] || [ "$MODEL" = "ornith9-q5" ] || [ "$MODEL" = "ornith9-q6" ] || [ "$MODEL" = "ornith9-q8" ] || [ "$MODEL" = "ornith9-bf16" ]; then
    echo
    echo "Ornith 1.5 9B text-only model downloaded."
    echo "Run with ds4:"
    echo "  ./ds4 -m ./ornith9.gguf -p \"The capital of France is\""
    echo "Or start the server:"
    echo "  ./ds4-server -m ./ornith9.gguf --ctx 32768"
    echo
    echo "Or compare with reference llama-cli:"
    echo "  llama-cli -m $OUT_DIR/$MODEL_FILE -p \"The capital of France is\" -n 32 --temp 0.0 --top-k 1 --top-p 1.0 -ngl 999 --no-warmup"
elif [ "$MODEL" = "ornith9-mmproj" ]; then
    echo
    echo "Ornith 1.5 9B vision projector (mmproj) downloaded ($OUT_DIR/$ORNITH9_MMPROJ_FILE)."
    echo "Note: Text-only ds4 inference does not require mmproj."
elif [ "$MODEL" = "ornith-q4" ] || [ "$MODEL" = "ornith-q5" ] || [ "$MODEL" = "ornith-q6" ] || [ "$MODEL" = "ornith-q8" ] || [ "$MODEL" = "ornith-bf16" ]; then
    echo
    echo "Ornith 1.5 35B text-only model downloaded."
    echo "Run with ds4:"
    echo "  ./ds4 -m ./ornith35.gguf -p \"The capital of France is\""
    echo "Or start the server:"
    echo "  ./ds4-server -m ./ornith35.gguf --ctx 32768"
    echo
    echo "Or compare with reference llama-cli:"
    echo "  llama-cli -m $OUT_DIR/$MODEL_FILE -p \"The capital of France is\" -n 32 --temp 0.0 --top-k 1 --top-p 1.0 -ngl 999 --no-warmup"
elif [ "$MODEL" = "ornith-mmproj" ]; then
    echo
    echo "Ornith 1.5 35B vision projector (mmproj) downloaded ($OUT_DIR/$ORNITH_MMPROJ_FILE)."
    echo "Note: Text-only ds4 inference does not require mmproj."
fi
echo
echo "Done."
