# Qwen3.8 MTP decode optimization

On 2026-09-05, an Apple M3 Ultra with 512 GiB unified memory running macOS 27.0
reached **60.51 tokens/s** on the Hamlet prose case, versus **57.40 tokens/s** at
`bd9cfbc`. All three measured candidate runs exceeded 60 tokens/s.

The model was `Qwen3.8-Flash-Next-Q4KImatrix-MTP-qwen4exp-pleext.gguf`, with
`Qwen3.8-Flash-Next-PLE-Q4_1.gguf` supplied through `--ple`. Both sides used
Metal, `--ctx 8192 --temp 0 --nothink --mtp --mtp-timing`. The candidate used
default settings, with no tuning environment variables.

The later [262K comparison](qwen38-262k-compare.md) found identical prefill
logits but numerical drift and some different continuations in long target-only
decode. The short MTP checks below do not establish whole-engine numerical or
task-quality parity at long context.

The subsequent [non-MTP optimization](qwen38-nonmtp-decode.md) corrects the
Q4_K accumulation order and restores all 896 recorded FP32 decode vectors
through 262K to the original engine's values. It includes a fresh MTP regression
check; the measurements below describe the earlier `af4195c` implementation.

## Results

Three interleaved repetitions per prompt, alternating run order. Each executable
received a separate Hamlet warmup, excluded from these medians. Rates come from
the CLI generation timer and exclude model loading and prefill. Results depend
on prompt, context length, model quantization, and hardware.

| Prompt | Baseline t/s | Candidate t/s | Candidate range | Gain | Accepted/cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| hamlet | 57.40 | 60.51 | 60.48–60.55 | 5.4% | 37/56 |
| fibonacci | 69.29 | 73.31 | 73.31–73.43 | 5.8% | 198/201 |
| explanation | 61.49 | 64.87 | 64.79–65.08 | 5.5% | 110/145 |

Generated output bytes and acceptance counts were identical between the two
builds in all nine measured pairs. The harness clears inherited `DS4_*` settings
and pins both changed runtime Metal source files independently for each build.
Binary/source hashes, commands, timings, and output hashes are recorded in
[`results.json`](../OUT/qwen38-mtp/final-defaults/results.json); the same directory
contains each run's stdout and stderr.

## Changes

- Two-token hyper-connection mixers reuse each up-projection weight for both
  tokens and compute low-rank activations once per threadgroup.
- The 128-wide GDN decode scan shares Q/K loads across four value rows. Two-token
  verification uses eight SIMD groups per threadgroup on M3 Ultra. Other devices
  retain one group by default; prefill keeps its existing layout.
- Q4_K expert gate/up projections use the existing tuned dot-product helper,
  sharing input loads across both projections and two output rows. Route weights
  are still applied by Qwen's expert reduction.
- After accepting a draft, one causal two-token predictor pass updates history
  and produces the next draft, reducing command submissions and weight reads.

For individual comparisons, set `DS4_QWEN4_NO_HC_PAIR=1`,
`DS4_QWEN4_NO_GDN_R4=1`, `DS4_QWEN4_NO_Q4K_MID=1`, or
`DS4_QWEN4_NO_MTP_BATCH=1`. The GDN opt-out affects decode only.
`DS4_QWEN4_GDN_NSG` accepts 1–8 SIMD groups for decode experiments.

## Validation

- `make -j8 all`: builds the CLI, server, benchmark, evaluator, and agent.
- `tests/test_qwen4_kernels`: passed against the independent CPU numerical
  references, including F16/F32/Q8 two-token mixers, Q4_K gate/up with MXFP4 down,
  and GDN state/history snapshots immediately after the first verification row.
- `tests/test_qwen4_spec`: passed.
- Exact-sampling regression: with seed 123, temperature 0.7, top-p 0.8, and
  `--mtp-exact-sampling`, both builds produced identical output and accepted
  45 of 56 drafts. Commands and hashes are in
  [`exact-sampling/results.json`](../OUT/qwen38-mtp/exact-sampling/results.json).
- Predictor integration check: 1,324 paired-versus-sequential comparisons over
  2,650 teacher-forced tokens, including pooled-key boundaries and the switch to
  sparse attention. All top predictions matched; maximum logit difference was
  `0.000596046` (tolerance `0.002 + 0.0001 * abs(reference)`). This compares GPU
  predictor paths; `GPU_ONLY` skips the full CPU model reference.
- `make test-qwen4-host` reports eight existing image-patching failures. A fresh
  build from its unchanged sources reproduced all eight. That target covers the
  superseded native qwen4 engine; its failure is independent of these changes.

Kernel, build, and baseline-host logs are in [`OUT/qwen38-mtp`](../OUT/qwen38-mtp).
The predictor check's stdout is
[`mtp-batch-check.stdout`](../OUT/qwen38-mtp/mtp-batch-check.stdout).

## Reproduce

Before changing the baseline checkout, build and save the executable and both
Metal files. They are loaded at runtime, so saving the executable alone is
insufficient:

```sh
make -j8
cp ds4 /tmp/ds4-mtp-before
cp metal/qwen4.metal /tmp/qwen4-mtp-before.metal
cp metal/moe.metal /tmp/moe-mtp-before.metal
```

After building the candidate:

```sh
python3 speed-bench/qwen38_mtp_compare.py \
  --model /path/to/Qwen3.8-Flash-Next-Q4KImatrix-MTP-qwen4exp-pleext.gguf \
  --ple /path/to/Qwen3.8-Flash-Next-PLE-Q4_1.gguf \
  --baseline /tmp/ds4-mtp-before \
  --baseline-source /tmp/qwen4-mtp-before.metal \
  --baseline-moe-source /tmp/moe-mtp-before.metal \
  --repeats 3 --out OUT/qwen38-mtp/reproduction
```

To check the paired predictor over a text prompt containing at least three
tokens, use a longer input to include the sparse-attention boundary:

```sh
DS4_QWEN4_GPU=1 DS4_QWEN4_GPU_ONLY=1 DS4_QWEN4_GPU_CHUNK=128 \
DS4_QWEN4_MTP_OUT=/dev/null DS4_QWEN4_MTP_BATCH_CHECK=1 \
./ds4 -m /path/to/model.gguf --ple /path/to/ple.gguf --metal \
  --prompt-file OUT/qwen38-mtp/validation-prompt.txt --raw-prompt \
  --nothink --first-token-test
```
