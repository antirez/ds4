# GGUF Conversion Guide

Complete workflow for generating quantized GGUF files from DeepSeek V4 Flash safetensors weights.

## Quick Start (one command)

```bash
# From the ds4 repo root:
bash test-4expert.sh /path/to/DeepSeek-V4-Flash-4Expert $(nproc)
```

This generates `template.gguf`, runs the quantizer, and creates the final GGUF. See the next section for what happens under the hood.

## Manual Steps

### 1. Build

```bash
# Linux:
make cpu -j$(nproc)
make -C gguf-tools -j$(nproc)
# macOS:
make -j$(sysctl -n hw.ncpu)
make -C gguf-tools -j$(sysctl -n hw.ncpu)
```

### 2. Generate GGUF Template

```bash
python3 gguf-tools/gen_gguf_template.py \
  --hf /path/to/DeepSeek-V4-Flash-4Expert \
  --out /tmp/template.gguf
```

The template (~5.6 MB) contains complete metadata, tokenizer data, and tensor descriptors. It does **not** contain weight data — only tensor names, shapes, and types.

### 3. Quantize and Convert

```bash
./gguf-tools/deepseek4-quantize \
  --hf /path/to/DeepSeek-V4-Flash-4Expert \
  --template /tmp/template.gguf \
  --out /path/to/output/model-q4k.gguf \
  --experts q4_k \
  --attention-proj q8_0 \
  --attention f16 \
  --shared q8_0 \
  --output q8_0 \
  --embedding f16 \
  --dense f16 \
  --n-experts 256 \
  --threads 12
```

> **Note**: If the output file already exists, you must add `--overwrite` or the tool will error.

#### Quantization Options Reference

| Flag | Typical value | Description |
|------|--------------|-------------|
| `--experts` | `q4_k` | Routed experts MoE FFN (w1/w2/w3) |
| `--attention-proj` | `q8_0` | Attention projection matrices (q/kv/output_a/output_b) |
| `--attention` | `f16` | Other 2D attention/compressor/indexer tensors |
| `--shared` | `q8_0` | Shared expert FFN |
| `--output` | `q8_0` | Output projection (output.*) |
| `--embedding` | `f16` | Token embedding layer |
| `--dense` | `f16` | Remaining 2D+ tensors not matched above |
| `--n-experts` | from template | Number of routed experts (read from template metadata if omitted) |
| `--threads` | `8` | Parallel worker count |

### 4. Test

```bash
ln -sfn /path/to/output/model-q4k.gguf ds4flash.gguf
./ds4 -p "Hello" -n 100
```

## How It Works

### Template Generation (gen_gguf_template.py)

1. Reads `model.safetensors.index.json` for tensor names and shapes
2. Maps HF tensor names to GGUF names using the same `layer_map` as `deepseek4-quantize.c`
3. Sets regular tensor types to F32 (routed expert tensors to F16). 1D tensors (norms, scales, biases) remain F32. 2D+ tensors get their type overridden by the quantizer policy.
4. Writes a GGUF file containing metadata + tokenizer + tensor descriptors

### Quantizer (deepseek4-quantize)

1. Loads the template to obtain all tensor descriptors
2. For each tensor: determines the final type using the user-specified quantization policy
3. Reads safetensors weights, performs quantization, writes to the output GGUF
4. Produces a ready-to-use GGUF file

## Notes

- **Regenerate the template** whenever the model's tensor set changes (step 1)
- **Type conversion**: `gen_gguf_template.py` automatically handles I64 → I32 conversion (for the `tid2eid` routing table)
- **1D tensors** (norms, scales, biases) are always stored as F32 and never quantized
- **Large model**: Q4_K output is approximately 153 GiB; ensure sufficient disk space
