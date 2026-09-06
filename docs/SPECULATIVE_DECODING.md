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

The support file adds about 5.6 GiB of weights plus runtime state. On Metal,
CUDA, and ROCm, the main model can be resident or SSD-streamed; the support
model stays separately mapped or device-cached. DSpark replaces the legacy
one-stage MTP drafter for that run; the two are not stacked.

Resident M5 paths batch supported verifier expert rows, including two-Mac TP.
Defaults select the established fast paths; additional branch experiments
remain opt-in. See [branch runtime and tester notes](BRANCH_DSPARK.md) for
backend eligibility, rollback controls, and acceptance fixtures.
Recorded comparisons are in [the QA guide](../QA_BEFORE_RELEASES.md).

## GLM: built-in MTP

GLM's draft block is already in its main GGUF:

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --mtp
```

`--mtp-timing` also enables it and prints acceptance and timing counters.
The current GLM cycle commits up to two tokens. No external support file is
needed, and ordinary decode remains the default.

## Sampling and reproducibility

At temperature zero, accepted drafts must match the target's greedy
continuation. At non-zero temperature, the default mode is opportunistic:
ordinary tokens use the requested sampling settings, but matching greedy
drafts are accepted directly. Sampling resumes when the proposed suffix does
not match. This is deliberately more deterministic than ordinary sampling.

Use `--mtp-exact-sampling` to preserve the ordinary target sampling
distribution. Exact mode accepts greedy proposals with their target
probability and samples from the remaining distribution on rejection.

Accepted tokens keep the state produced by the batched verifier. Floating-point
reduction order can differ from one-token decode, so long greedy continuations
need not be byte-identical. For DeepSeek comparisons against the ordinary
target-only path, use `--quality` or `--dspark-strict`; these disable the
speculative acceptance path. They do not promise identical output across
different hardware or execution configurations.

Session-batched serving uses ordinary target decoding instead of combining
DSpark/MTP with the session batch. See [serving](SERVER.md#multiple-sessions).
