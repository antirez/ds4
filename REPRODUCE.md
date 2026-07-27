# Reproducing the v5mx build

This is the real end-to-end build DAG for the shipped
`ds4flash-v5mx-reap25-type40-mxfp8lt-dspark-v1.gguf` (91 GB). It is **not** the
naive "prune → quantize" order the README sketches: REAP compaction happens on
an already-quantized *oracle* intermediate, and the final mixed-quant pass uses
that compacted file as its template. The full ordering is below.

Everything here is offline (no GPU needed) except the imatrix collection in
stage (d), which runs the pulsar prefill graph.

## Two external inputs

Everything else in this repo is reproducible from these two artifacts:

1. **DeepSeek-V4-Flash-DSpark** (MIT) — the source checkpoint, FP8/FP4 QAT
   weights plus the DSpark speculative-decoding drafter.
   `hf download deepseek-ai/DeepSeek-V4-Flash-DSpark` (the full 48-shard main
   model; the DSpark drafter shards ride in the same repo).
2. **eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4** — the REAP-25
   (LiveCodeBench-50-sample-calibrated) expert-survivor prune. We do **not**
   download this 68 GB GGUF: its survivor map is vendored in this repo as
   `gguf-tools/reap/reap25-lcb50-survivors.json` (derived integer index
   metadata, recovered from the router tensors — see `gguf-tools/reap/README.md`).
   If you would rather regenerate it, `gguf-tools/reap/recover_survivors.py`
   range-fetches only the small router tensors (~140 MB, not the full file).

Let `HF=/path/to/DeepSeek-V4-Flash-DSpark` (the safetensors dir) below.

## Build the tools

```sh
make -C gguf-tools            # builds gguf-tools/deepseek4-quantize
```

## The DAG

### (a) Templates — chat/tokenizer/tensor-order metadata

The quantizer needs a template GGUF for metadata, tensor order, tokenizer, and
logical shapes. Build the main and DSpark templates from the HF checkpoint:

```sh
python3 gguf-tools/build_main_template.py   --hf "$HF" --out main-template.gguf
python3 gguf-tools/build_dspark_template.py --hf "$HF" --out dspark-template.gguf
```

The chat template itself is the pinned froggeric v20 jinja
(`spark-vllm-docker/mods/froggeric-chat-template/`); `build_main_template.py`
embeds it.

### (b) Oracle intermediate — quantize the full source at the rich format

Quantize the full (256-expert) source to the "oracle-cutlass-mxfp4"
intermediate: every routed expert as byte-lossless CUTLASS_MXFP4 (type 40,
tensor-core B layout), non-experts as MXFP8. This is a plain reproducible
`deepseek4-quantize` run — no REAP yet.

```sh
gguf-tools/deepseek4-quantize \
  --hf "$HF" --template main-template.gguf \
  --experts cutlass_mxfp4 --output fp8_e4m3 \
  --out oracle-cutlass-mxfp4.gguf
```

### (c) REAP transplant — compact to the survivor set (Path B)

Stamp the vendored REAP-25 survivor map onto the oracle: dense-trim the expert
tensors to the 192 survivors on layers 3–42 (layers 0–2 are hash-routed and
preserved), padding the router/bias back to 256 so the CUDA `router_select`
kernels are untouched. This produces the `ds4-compact-v1` base.

```sh
python3 gguf-tools/reap/trim_reap.py \
  --oracle oracle-cutlass-mxfp4.gguf \
  --out    oracle-reap25-compact.gguf
# --survivors defaults to the vendored gguf-tools/reap/reap25-lcb50-survivors.json
```

`oracle-reap25-compact.gguf` is both the REAP-25 base **and** the template for
stage (f): it carries the compacted 192-expert shapes and the `reap.*` KV.

### (d) Imatrix — routed-MoE activation importance

The IQ2_XXS floor needs a real importance vector. Collect one with the pulsar
runtime over the calibration corpus (or download the published `.dat`).

```sh
# calibration corpus (already in-tree; regenerate if desired)
python3 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py

./pulsar -m oracle-reap25-compact.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out routed-moe.dat --ctx 32768
```

A published imatrix is also available at
`antirez/deepseek-v4-gguf` (`imatrix/…routed-moe-ds4-1p5m.dat`).

### (e) Allocation — the per-tensor format map (already committed)

PrismaQuant's measured-KL knapsack chose the per-layer expert formats. The
result is committed as `gguf-tools/prisma/v5mx-format-map.json` (474 entries:
345 MXFP8_LT non-experts, 36 CUTLASS_MXFP4 rich expert tensors, 93 IQ2_XXS floor
expert tensors). To re-derive it, follow `gguf-tools/prisma/README.md`
(measure_layer_kl → dump_inventory → prisma_alloc). You do **not** need to
re-run this stage to reproduce the shipped build; use the committed map.

### (f) Mixed-quant GGUF — apply the format map

Quantize once more, this time with the compacted base as the template and the
committed format map selecting each tensor's precision. Rich (CUTLASS_MXFP4)
expert layers stay byte-lossless; the floor layers drop to IQ2_XXS with the
imatrix; non-experts re-encode to the MXFP8_LT (type-41) swizzle.

```sh
gguf-tools/deepseek4-quantize \
  --hf "$HF" --template oracle-reap25-compact.gguf \
  --imatrix routed-moe.dat \
  --format-map gguf-tools/prisma/v5mx-format-map.json \
  --out v5mx-mixed.gguf
```

> **type-41 native emit — cross-reference.** The MXFP8_LT (type-41) tensors are
> being moved to native emit inside `deepseek4-quantize` by a parallel change
> (touching `gguf-tools/quantize/quants_fp.c`, `quants.h`, `quants_common.c`,
> and the `sfcmp` path). Until that lands, the format map's `MXFP8_LT` entries
> may be produced by the prior swizzle post-step rather than natively; the
> output is byte-identical either way. If you built `deepseek4-quantize` from a
> tree without the native-emit change, confirm the MXFP8_LT swizzle is applied
> (the loader reads type-41 zero-copy). type-40 (CUTLASS_MXFP4) is already
> emitted natively (see `gguf-tools/prisma/README.md`).

### (g) Merge the DSpark drafter

Splice the DSpark drafter into the main file, producing the single
self-contained release artifact that auto-enables spec decode on load.

```sh
# build the drafter GGUF from the DSpark shards using the dspark template …
gguf-tools/deepseek4-quantize \
  --hf "$HF" --template dspark-template.gguf \
  --out dspark-drafter.gguf
# … then merge it in-file
python3 gguf-tools/merge_dspark_gguf.py \
  v5mx-mixed.gguf dspark-drafter.gguf \
  ds4flash-v5mx-reap25-type40-mxfp8lt-dspark-v1.gguf
```

The engine auto-enables DSpark when it sees `dspark.main_proj.weight`; no serve
flags are needed. `bench`/`eval`/`agent` paths opt out via `--no-dspark`.

## Serve

```sh
./pulsar-server -m ds4flash-v5mx-reap25-type40-mxfp8lt-dspark-v1.gguf --ctx 1048576
```

## Reproducibility statement

From a clean clone plus the **two external inputs** above (the
DeepSeek-V4-Flash-DSpark checkpoint and — only if you choose to regenerate the
vendored survivor map — network access to eouya2's + antirez's HF GGUF headers),
every stage of this DAG is reproducible with in-tree tooling. The REAP survivor
map, the PrismaQuant format map, the templates, the imatrix corpus, and the
merge/transplant tools are all committed.

Two honest caveats:

- **type-41 native emit** is still landing (stage (f) note). The bytes are
  identical; only *where* the swizzle is applied moves. This doc will be
  slightly ahead of the tree until that change merges.
- Bit-exact model output depends on the pulsar engine build (`CUDA_ARCH=sm_120f`)
  and the driver/CUDA stack; same-seed reproducibility is guaranteed only on
  identical hardware/build.
