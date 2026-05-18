# DS4 GGUF Tools

This directory contains the offline tools used to build and evaluate DeepSeek
V4 Flash GGUF files for `ds4`.

The important pieces are:

- `deepseek4-quantize.c`: C HF-safetensors to GGUF quantizer.
- `quants.[ch]`: the deliberately small local quantization implementation used
  by the quantizer.  It implements the DS4 output formats we actually ship:
  `q8_0`, `q4_K`, `q2_K`, and `iq2_xxs`.
- `imatrix/`: dataset and instructions for collecting routed-MoE activation
  importance with `ds4`.
- `quality-testing/`: prompts and scripts used to compare local GGUF variants
  against official DeepSeek V4 Flash continuations.

## Build

```sh
make -C gguf-tools
```

The quantizer is plain C and does not link GGML.  GGUF metadata handling,
safetensors loading, FP4/FP8 dequantization, and the quantizers used by our Q2
and Q4 recipes live in this directory.

## Generate An Imatrix

First regenerate or inspect the calibration dataset:

```sh
python3 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

Then collect activation statistics with the DS4 runtime:

```sh
./ds4 \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat \
  --ctx 32768
```

The imatrix file is useful immediately with this DS4 quantizer.  Generic GGUF
tools need DS4-specific tensor-name mapping and per-expert slicing before they
can use it correctly.  The accepted imatrix format is the legacy llama.cpp
binary `.dat` file emitted by `ds4 --imatrix-out`.

Generating this `.dat` file locally is possible, but slow: it runs the DS4
prefill graph over the full calibration corpus and reads routed-MoE activation
statistics back from the GPU.  The latest published imatrix-generated GGUF files
are available in the antirez Hugging Face repository:

```text
https://huggingface.co/antirez/deepseek-v4-gguf/tree/main
```

## Generate Q2 And Q4 GGUFs

The template GGUF supplies metadata, tokenizer, tensor order, and logical
shapes.  Tensor bytes are regenerated from the Hugging Face safetensors.  Full
generation is intentionally offline and heavy: expect roughly 80-90 GB outputs
for the 2-bit template family and roughly 150-170 GB for the 4-bit routed-expert
family, plus enough free disk for the temporary output.  Use `--dry-run` and
`--compare-tensor` before starting a full write, and use `--overwrite` only when
you really mean to replace an existing GGUF.

Q2 routed experts with imatrix:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

Q4 routed experts with imatrix:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

You can override tensor families:

```sh
--experts iq2_xxs
--routed-w2 q2_k
--attention-proj q8_0
--shared q8_0
--output q8_0
```

Useful checks before writing a full model:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template MODEL.gguf \
  --compare-tensor blk.0.attn_q_a.weight
```

`--compare-tensor` regenerates a single tensor and byte-compares it against the
template or `--compare-gguf`.  `--threads N` controls routed-expert workers.

## When No Imatrix Is Given

`iq2_xxs` requires an importance vector.  If `--imatrix` is not provided and
the target type requires one, `deepseek4-quantize` computes a synthetic fallback
from the dequantized weight itself:

```text
importance[column] = sum(row[column]^2) over all rows
```

This is a weight-energy heuristic.  It is not as good as measuring real DS4
activations, but it gives the quantizer a stable column weighting and was good
enough for the first working 2-bit GGUFs.

## Quality Testing

See `quality-testing/README.md`.  The short version is:

```sh
python3 gguf-tools/quality-testing/collect_official.py
make -C gguf-tools quality-score
gguf-tools/quality-testing/score_official MODEL.gguf gguf-tools/quality-testing/data/manifest.tsv /tmp/model.tsv 4096
python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```
