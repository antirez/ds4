# Speculative Decoding

[README](../README.md)

Speculation proposes future tokens with a smaller draft block, then checks
them with the main model. An accepted prefix advances generation by several
tokens in one verification pass. It does not accelerate prefill.

It is opt-in. Gains depend on the prompt, model, backend, and context length;
poor acceptance can make it slower. Measure your workload rather than assuming
that a draft model always helps.

## DeepSeek Flash: DSpark

DSpark is a separate support GGUF, not a standalone language model. It proposes
up to five future tokens. Match its checkpoint to the main model:

| Main checkpoint | Download | Support file |
| --- | --- | --- |
| Flash 0731 | `ds4f-dspark` | `gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf` |
| Flash Vision Experimental | `ds4f-vision-dspark` | `gguf/DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf` |

For the 0731 Q2 model:

```sh
./download_model.sh ds4f-q2
./download_model.sh ds4f-dspark
./ds4 --dspark --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf
```

For Vision Experimental, substitute its matching main model and support file.
Do not mix the two checkpoints. DSpark is not supported for PRO.
The same flags work in `ds4-agent` and non-batched `ds4-server` requests.

The support file adds about 5.6 GiB of weights plus runtime state. The main
model must be resident; external DSpark is not supported with SSD streaming.
DSpark replaces the legacy one-stage MTP drafter for that run; the two are
not stacked.

Resident M5 paths batch supported verifier expert rows, including two-Mac TP.
On DGX Spark, resident Flash 0731 Q2 and two-Spark MXFP4 also batch the seed
with longer drafts and use small-batch Q8 and expert kernels. No extra flags
are needed.
Single-Spark Q2 uses a confidence-pruning threshold of 0.4. Two-Spark
Flash 0731 Q2 and MXFP4 leave confidence pruning off and split the drafter
over RDMA when both peers load its support file. `--dspark-confidence`
overrides the threshold. Exact sampling keeps its default threshold of 0.8.
The scheduler can back off when drafting is unproductive. Defaults select the
fast paths; diagnostic environment variables are not needed for normal use.
Recorded comparisons are in [the QA guide](../QA_BEFORE_RELEASES.md).

For the tested Strix Halo coding configuration, use `--dspark --dspark-confidence 0.7` with the default five-token draft cap and scheduler. Client sampling is temperature `1.0`, `top_p=0.95`, `min_p=0`, and `top_k=0`; high reasoning was also checked on coding and tool-use requests. This uses opportunistic sampling as described below; exact-mode throughput is not qualified by these measurements. `--mtp-draft` controls legacy autoregressive MTP, not the DSpark draft width.

## DeepSeek V4.1 Flash: DSpark

V4.1 has its own 8.04 GiB support GGUF. It retains the native MXFP4 experts
and uses a different draft graph from V4 Flash. Do not interchange the files.
See [conversion](../gguf-tools/README.md#convert-v41-dspark) to build it.

```sh
./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming --dspark \
  --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark.gguf
```

The same options work with `ds4-agent` and `ds4-server`. Metal, CUDA and ROCm
support single-host execution, with either resident target weights or SSD
streaming. The drafter stays resident, so leave another 8 GiB for it.
Two-host tensor parallelism works on Metal and CUDA: add the same `--dspark
--mtp-model FILE` options to both the coordinator and worker. Each peer keeps
the drafter resident; the target and its verification run across both GPUs.

Up to five drafts are checked together. The sampling choices below apply;
`--mtp-exact-sampling` uses sampled proposals and the full acceptance/correction
rule for V4.1. Restored sessions warm the draft window with 128 ordinary tokens
before speculation resumes. Measure speed on your prompts: acceptance and SSD
cache size both matter. V4.1 DSpark can still be slower than ordinary decoding,
including in TP mode.

## GLM: built-in MTP

GLM's draft block is already in its main GGUF:

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --mtp
```

`--mtp-timing` also enables it and prints acceptance and timing counters.
The current GLM cycle commits up to two tokens. No external support file is
needed, and ordinary decode remains the default.

## Qwen3.8: built-in MTP

Both Qwen downloads include MTP and native BF16 n-grams in the main GGUF:

```sh
./download_model.sh qwen38-q4k
./ds4 --mtp
```

Ordinary decode uses the same file with `--mtp` omitted. For non-zero
temperature, add `--mtp-exact-sampling` to preserve the target sampling
distribution. See [Qwen setup](QWEN38_FLASH_NEXT.md) for Metal and CUDA.

The cycle drafts one token ahead by default and engages a **second, chained
draft** (one extra nextn-layer step conditioned on the predictor's own
stream, verified in a 3-row pass) while recent first-draft acceptance is
perfect, disengaging after repeated second-draft rejections.
`DS4_QWEN4_MTP_DEPTH=2` or `=3` fixes the depth;
`0` (default) is the adaptive policy.

## Sampling and reproducibility

At temperature zero, accepted drafts must match the target's greedy
continuation. At non-zero temperature, the default mode is opportunistic:
ordinary tokens use the requested sampling settings, but matching greedy
drafts are accepted directly. Sampling resumes when the proposed suffix does
not match. This is deliberately more deterministic than ordinary sampling.

Use `--mtp-exact-sampling` to preserve the ordinary target sampling
distribution. Greedy proposals are accepted with their target probability;
sampled proposals use the target-to-draft probability ratio. Rejections sample
from the remaining target probability mass.

When a verified block crosses a tool sampling-mode boundary (for example
entering tool-call syntax during server decoding), the server rewinds to the
block start and re-evaluates the boundary token so the next sample uses the
new mode. Under exact sampling that rewind restores a pre-verify snapshot of
the recurrent state instead of resetting the graph, so long retained
contexts are not replayed at every boundary.

Accepted tokens keep the state produced by the batched verifier. Floating-point
reduction order can differ from one-token decode, so long greedy continuations
need not be byte-identical. For DeepSeek comparisons against the ordinary
target-only path, use `--quality` or `--dspark-strict`; these disable the
speculative acceptance path. They do not promise identical output across
different hardware or execution configurations.

Session-batched serving uses ordinary target decoding instead of combining
DSpark/MTP with the session batch. See [serving](SERVER.md#multiple-sessions).
