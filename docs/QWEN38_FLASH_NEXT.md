# Qwen3.8 Flash Next

[Back to README](../README.md)

Qwen3.8-Flash-Next (`qwen4exp` in GGUF terms) runs on its own Metal graph:
36 gated delta-net layers and 12 gated GQA layers with the QSA block-sparse
indexer, four-stream hyper-connections, the hashed per-layer n-gram table,
512-expert MoE with a shared expert, and the built-in MTP block.

## Download and run

The [DS4 Q4 release](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4)
contains one combined main/MTP GGUF and a required external PLE sidecar,
about 107 GB (100 GiB) together on disk:

```sh
./download_model.sh qwen38-q4k
./ds4 --ple gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --mtp
./ds4-server --ple gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --mtp --mtp-exact-sampling
```

The downloader links `ds4flash.gguf` to the combined model. Both ordinary and
MTP decode use this same GGUF and PLE sidecar; omit `--mtp` for ordinary decode.
Adjust the PLE path if you set `DS4_GGUF_DIR`. The external PLE table is mapped
separately, allowing this release to run on 128 GB Macs; context size and other
workloads still affect memory use.

Vision needs llama.cpp's mmproj encoder, which is a separate download:

```sh
./download_model.sh qwen38-vision
```

## Build your own GGUF

For calibrated IQ2_XXS gate/up experts with MXFP4 down projections, embedded
MTP and external PLE, use the [two-stage BF16 build](../gguf-tools/README.md#qwen38-iq2_xxs-experiment).
Its main GGUF is 50.34 GB; the Q4_1 PLE sidecar is required separately.
Context buffers, runtime allocations and resident PLE pages also consume
memory. File size alone does not establish whether it fits a 64 GB Mac.
See the [quality and memory measurements](../speed-bench/qwen38-iq2-quality.md)
for the comparison with Q4_K, Q4_0 and Q8.

The older recipes below keep PLE inside the main GGUF, so their file sizes
are not directly comparable with the external-PLE builds.

Build the
GGUF from the Hugging Face checkpoint with the converter in `gguf-tools/`;
the stock `ggml-org` Q8_0 GGUF also loads once its two parts are merged with
`llama-gguf-split --merge`:

```sh
python gguf-tools/qwen4_exp_convert.py --src /path/to/Qwen3.8-Flash-Next \
  --out gguf/Qwen3.8-Flash-Next-Q8.gguf --outtype q8_0            # about 192 GB
python gguf-tools/qwen4_exp_convert.py --src /path/to/Qwen3.8-Flash-Next \
  --out gguf/Qwen3.8-Flash-Next-MXFP4.gguf --outtype q8_0 --experts mxfp4  # about 126 GB
llama-quantize --allow-requantize --tensor-type hc_=f16 --tensor-type ffn_gate_exps=Q4_K \
  --tensor-type ffn_up_exps=Q4_K --tensor-type per_layer_token_embd=Q4_0 \
  gguf/Qwen3.8-Flash-Next-Q8.gguf gguf/Qwen3.8-Flash-Next-Q4K.gguf Q8_0  # about 124 GB
llama-quantize --allow-requantize --tensor-type hc_=f16 --tensor-type ffn_gate_exps=Q2_K \
  --tensor-type ffn_up_exps=Q2_K --tensor-type ffn_down_exps=MXFP4 \
  --tensor-type per_layer_token_embd=Q4_0 \
  gguf/Qwen3.8-Flash-Next-Q8.gguf gguf/Qwen3.8-Flash-Next-Q2K.gguf Q8_0   # about 83 GB
llama-quantize --imatrix imatrix.gguf --allow-requantize --tensor-type hc_=f16 \
  --tensor-type blk.48.ffn_gate_exps=MXFP4 --tensor-type blk.48.ffn_up_exps=MXFP4 \
  --tensor-type ffn_gate_exps=IQ2_XXS --tensor-type ffn_up_exps=IQ2_XXS \
  --tensor-type ffn_down_exps=MXFP4 --tensor-type per_layer_token_embd=Q4_0 \
  gguf/Qwen3.8-Flash-Next-Q8.gguf gguf/Qwen3.8-Flash-Next-IQ2.gguf Q8_0   # about 77 GB
```

The Q8 file keeps every weight at 8 bits except the QSA indexer projections,
which stay at the released BF16. The MXFP4 file keeps the routed experts in
native MXFP4 blocks, the Q4K file requantizes the expert gate/up projections
to Q4_K, and the two-bit files take them to Q2_K or, with an importance
matrix from `llama-imatrix`, IQ2_XXS; the 640-wide expert down projections
are too narrow for 256-value blocks and go to MXFP4. The first matching
`--tensor-type` wins, and the MTP block's experts (`blk.48`) stay MXFP4
because an importance matrix collected with llama.cpp never sees them. Keep
the `hc_=f16` override: the Q8_0 base type would otherwise requantize the
hyper-connection mixers, which slows prefill. Q2_K and IQ2_XXS trade size
against quantization error; compare measured quality for the specific recipe.
These inline-PLE builds need additional memory beyond the file size for
context and runtime buffers.

```sh
./ds4 -m gguf/Qwen3.8-Flash-Next-Q8.gguf --ctx 32768
./ds4 -m gguf/Qwen3.8-Flash-Next-Q8.gguf --mtp --temp 0
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q8.gguf --ctx 65536 --kv-disk-dir ~/.ds4/server-kv
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q8.gguf --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
./ds4-agent -m gguf/Qwen3.8-Flash-Next-MXFP4.gguf
./ds4-server -m gguf/Qwen3.8-Flash-Next-Q4K.gguf --mtp --mtp-exact-sampling --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
```

The MTP block is inside the same GGUF; `--mtp` enables it and speeds up
greedy decoding. At non-zero temperature it keeps target-matching greedy
drafts, like the GLM path, which skews sampled output toward the greedy
choice; `--mtp-exact-sampling` preserves the ordinary sampling distribution
at a smaller speedup. Prefill runs in 8192-token chunks
(`DS4_QWEN4_PREFILL_CHUNK` overrides it; the transient buffers scale with the
chunk size). For A/B checks, `DS4_QWEN4_NO_FUSE=1` selects the unfused decode
kernels, `DS4_QWEN4_NO_IDX_SELECT=1` the argsort top-k and
`DS4_QWEN4_NO_ATTN_MM=1` the per-token attention kernel for prefill batches;
`DS4_QWEN4_TIMING=1` prints stage timings.

MTP decode batches the accepted token's predictor history update with its next
draft. Two-token hyper-connection mixers share weights, the 128-wide GDN scan
processes four value rows per SIMD group, and Q4_K expert gate/up projections
share input loads. These paths are enabled by default. For individual A/B
checks, set `DS4_QWEN4_NO_MTP_BATCH=1`, `DS4_QWEN4_NO_HC_PAIR=1`,
`DS4_QWEN4_NO_GDN_R4=1` (decode only), or `DS4_QWEN4_NO_Q4K_MID=1`.
The GDN verification layout uses eight SIMD groups on M3 Ultra;
`DS4_QWEN4_GDN_NSG` overrides the decode group count from 1 to 8.
GDN prefill uses four SIMD groups on M3 Ultra for batches of at least 8192
tokens with 16 key heads, 48 value heads and head width 128.
`DS4_QWEN4_Q4K_MID_NR` selects 1 or 2 rows per SIMD group for Q4_K gate/up.
One- and two-token decode on M3 Ultra default to one row and eight SIMD groups;
other batches and devices retain two rows and two SIMD groups.
`DS4_QWEN4_Q4K_MID_NSG` overrides the group count from 1 to 8. Setting
`DS4_QWEN4_Q4K_MID_NR=2` restores the former layout without an NSG override.
See [the MTP decode benchmark](../speed-bench/qwen38-mtp-decode.md) for measurements
and reproduction commands. The [earlier 262K comparison](../speed-bench/qwen38-262k-compare.md)
found decode drift in the previous Q4_K optimization. The current gate/up kernel
preserves the original accumulation order. Before the main rebase, it matched
all 896 recorded FP32 decode logit vectors through 262K; see the
[non-MTP benchmark](../speed-bench/qwen38-nonmtp-round3.md) for those measurements.
The [rebase validation](../speed-bench/qwen38-main-rebase.md) matched 256 full
vectors through 8K and preserved the tested MTP outputs; it did not rerun 262K. The
[previous round](../speed-bench/qwen38-nonmtp-decode.md) records the numerical correction.

The trunk submits commands after two layers to overlap GPU execution with host
encoding; `DS4_QWEN4_FLUSH_LAYER=0` restores one submission. On M3 Ultra, MoE
prefill launches use tile caps of 8, 16, and 32 at batch sizes below 4096, from
4096, and from 8192 tokens. `DS4_QWEN4_MOE_MID_TILES` and
`DS4_QWEN4_MOE_DOWN_TILES` override these caps from 1 to 32.
Hyper-connection normalization reuses each stream's RMS for batches of at least
8192 tokens on M3 Ultra with embedding size 2560, four streams and four injection
outputs. It preserves the original chunk reductions. `DS4_QWEN4_HC_NORM_REUSE=0`
disables reuse; `=1` enables it for batches larger than two tokens.
Separate scratch for each chunk removes redundant overwrite barriers without
changing the reduction order. See [the latest MTP optimization report](../speed-bench/qwen38-mtp-round3.md)
for full-vocabulary numerical checks and prefill/decode measurements through 262K.

`--batched-session N` keeps N sessions resident; their decode steps run one
after another rather than as one grouped batch, so it buys concurrency, not
throughput. Thinking is on by default with the model's `xhigh` reasoning
instruction; `reasoning_effort` `low`, `medium` or `xhigh` selects the model
card's levels (`chat_template_kwargs` with `enable_thinking` and
`reasoning_effort` is accepted too), the server's `qwen3.8-flash-next-chat`
alias disables thinking and `qwen3.8-flash-next-reasoner` forces it. Tool
calls use the model's native `<tool_call><function=...><parameter=...>` format
in both the server and the agent. Disk KV checkpoints and live prefix reuse
work as for the other models; the recurrent state travels with the checkpoint.
Rewinding to an arbitrary earlier position resets that state and replays the
kept token prefix on the next evaluation. A shorter prompt is prefilled again.

The model's native window is 262144 tokens. For longer prompts set
`DS4_QWEN4_YARN_FACTOR=4`, which
applies the static YaRN scaling from the model card, up to about 1M tokens;
use a factor of 2 for 512k. Static YaRN costs a little accuracy on short
prompts, so leave it off otherwise.

Images go through the model's Qwen3-VL vision tower. Download llama.cpp's
mmproj file with `./download_model.sh qwen38-vision` (it comes from
`ggml-org/Qwen3.8-Flash-Next-GGUF`; alternatively run
`convert_hf_to_gguf.py --mmproj --outtype f16` on the checkpoint) with
`--vision` and send `image_url` parts as usual. Each image is resized to
multiples of 32 pixels within 64 to 1024 tokens (`DS4_QWEN4_IMAGE_MAX_TOKENS`
raises the cap), encoded on the GPU, and takes the model's 3D rope positions;
live KV reuse keys on the image fingerprints. `make test-qwen4-vision`
compares the tower with the Hugging Face implementation.

To check CLI `/read` image turns and text follow-ups with ordinary and MTP
decode, use two different PNG or JPEG images with the model-backed regression:

```sh
make ds4
uv run tests/test_qwen4_cli_vision.py --model /path/to/main-with-mtp.gguf \
  --ple /path/to/ple.gguf --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --image /path/to/first.png --image /path/to/second.png
```

The test checks that all four turns complete in each mode, including errors
that the interactive CLI can report without a nonzero process exit. It saves
the responses and diagnostics for inspection; it does not grade image content.

Metal only for now. The Metal graph accepts Q8_0, Q4_0, F16, BF16 and F32
dense weights, Q8_0/MXFP4/Q4_0/Q4_K/Q2_K/IQ2_XXS experts, F16/F32/Q8_0
hyper-connection mixers and a Q8_0/Q4_0/MXFP4/F16/F32 n-gram table.
Multi-node tensor parallelism is not implemented yet.

## Qwen3.8 Flash Next (campaign branch merge)

This branch carries the qwen4-exp engine adoption merge. The native fast-pack
qwen4 engine developed during the campaign is superseded; its history, notes
and performance handoff live in [QWEN38_FLASH_NEXT.md](../QWEN38_FLASH_NEXT.md)
and [QWEN38_PERF_HANDOFF.md](../QWEN38_PERF_HANDOFF.md), and its source is kept
in-tree (`ds4_qwen4.c`/`ds4_qwen4.h`, unreferenced by the build).

The published Q4_K-imatrix build combines the main and MTP weights and uses
the external PLE sidecar, as shown in [Download and run](#download-and-run).
To select it explicitly instead of using the default symlink:

```
./ds4 -m gguf/Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf \
      --ple gguf/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --metal --mtp
```

Models still in the old fast-pack format can be converted with
`gguf-tools/qwen4_pack_to_qwen4exp.py` and then run the same way.
