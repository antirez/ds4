# Testing the 4Expert PR on a Fresh Linux Machine

## Quick Start (single command)

```bash
git clone https://github.com/yuhai-china/ds4 && cd ds4 && git checkout 4expert
bash test-4expert.sh
```

This runs all 5 steps:
1. Build ds4 + gguf-tools
2. Download 4Expert safetensors weights (~130 GiB)
3. Generate GGUF template from metadata
4. Quantize to Q4_K GGUF (~153 GiB output, ~30 min with 20 threads)
5. Link GGUF and run `./ds4 -p "..." -n 100`

After step 4 completes, the GGUF file is reusable — skip re-conversion on subsequent runs.

## Custom Paths

```bash
bash test-4expert.sh /path/to/existing/weights 16
```

- 1st arg: path to safetensors directory (skips download if `model.safetensors.index.json` exists)
- 2nd arg: number of threads (default all cores)

## Pre-Quantized (skip conversion, download GGUF directly)

```bash
git clone https://github.com/yuhai-china/ds4 && cd ds4 && git checkout 4expert
make cpu -j$(nproc)
make -C gguf-tools -j$(nproc)

pip install -q huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('cloudyu/DeepSeek-V4-Flash-4Expert-GGUF', 'DeepSeek-V4-Flash-4Expert-Q4K.gguf', local_dir='.')
"

ln -sfn DeepSeek-V4-Flash-4Expert-Q4K.gguf ds4flash.gguf
./ds4 -p "Hello" -n 100
```

## Manual Steps (if the one-click script doesn't work)

### 1. Build

```bash
make -C gguf-tools -j$(nproc)
make cpu -j$(nproc)
```

### 2. Download weights

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('cloudyu/DeepSeek-V4-Flash-4Expert', local_dir='./DeepSeek-V4-Flash-4Expert')
"
```

### 3. Generate template + quantize

```bash
python3 gguf-tools/gen_gguf_template.py \
  --hf ./DeepSeek-V4-Flash-4Expert \
  --out template.gguf

./gguf-tools/deepseek4-quantize \
  --hf ./DeepSeek-V4-Flash-4Expert \
  --template template.gguf \
  --out ds4flash-4expert.gguf \
  --experts q4_k \
  --attention-proj q8_0 \
  --attention f16 \
  --shared q8_0 \
  --output q8_0 \
  --embedding f16 \
  --dense f16 \
  --threads $(nproc)
```

### 4. Test

```bash
ln -sfn ds4flash-4expert.gguf ds4flash.gguf
./ds4 -p "The weather is great today" -n 100
```
