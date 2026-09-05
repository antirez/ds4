# Qwen3.8 baseline comparison through 262,144 tokens

This report records the earlier `bd9cfbc` → `af4195c` comparison. The subsequent
[non-MTP optimization](qwen38-nonmtp-decode.md) corrects the Q4_K arithmetic
responsible for the reproduced drift and matches all 896 recorded FP32 decode
vectors to `bd9cfbc`. The historical measurements below are unchanged.

On 2026-09-05, the current optimization patch and its previous baseline
(`bd9cfbc`) completed the standard `ds4-bench` sweep on an Apple M3 Ultra with
512 GiB unified memory, running macOS 27.0.

At 262,144 prompt tokens, target-only decode measured **42.24 → 42.37 tokens/s**
(+0.31%), while prefill measured **1,042.87 → 1,037.44 tokens/s** (-0.52%).
All seven full-vocabulary prefill logit dumps were byte-identical. Greedy
continuations differed at four contexts. A separate controlled decode check
found numerical drift, despite almost unchanged aggregate sampled likelihood.

This benchmark does **not** enable Qwen MTP. The standard driver supports
Qwen target-only decode, so these figures measure the core engine rather than
the earlier short-prompt MTP speedup. Production engine code was left unchanged
throughout this comparison.

## Throughput

The input was the repository's `speed-bench/promessi_sposi.txt` (1,329,139 bytes;
SHA-256 `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`).
Both builds used the same Q4K imatrix model with embedded MTP weights and the
same external Q4_1 PLE table:

- `Qwen3.8-Flash-Next-Q4KImatrix-MTP-qwen4exp-pleext.gguf`
- `Qwen3.8-Flash-Next-PLE-Q4_1.gguf`

Each frontier generated 128 greedy tokens. The sweep used `--ctx-start 4096
--ctx-max 262144 --step-mul 2`, with the default allocation of 262,273 tokens
(prompt plus generation space), default prefill chunks, and no tuning overrides.
The two builds ran sequentially on the GPU, with separate 4K/16-token warmups.
Model loading, logit dumps, and snapshot/replay restoration are outside the
reported timing windows. Prefill rates cover each newly added interval.

| Prompt tokens | Baseline prefill t/s | Current prefill t/s | Baseline decode t/s | Current decode t/s | Decode change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1,110.62 | 1,066.62 | 45.49 | 45.64 | +0.33% |
| 8,192* | 1,150.27 | 1,148.13 | 45.33 | 45.62 | +0.65% |
| 16,384 | 1,171.32 | 1,095.12 | 45.41 | 45.48 | +0.15% |
| 32,768 | 1,166.43 | 1,129.12 | 45.17 | 45.27 | +0.22% |
| 65,536 | 1,157.19 | 1,090.00 | 44.85 | 45.00 | +0.33% |
| 131,072 | 1,116.46 | 1,068.43 | 43.97 | 44.21 | +0.55% |
| 262,144 | 1,042.87 | 1,037.44 | 42.24 | 42.37 | +0.31% |

*The initial 8K baseline timing was an outlier at 34.10 t/s. Three additional
interleaved repetitions returned baseline 45.32/45.33/45.44 and current
45.59/45.69/45.65 t/s. The 8K decode cells use medians of all four observations,
including the initial sweep on each side. Other cells are single-sweep
measurements. All raw observations are retained; small timing differences
should not be treated as established regressions or improvements.*

The baseline sweep took 471.3 seconds and the current sweep 479.9 seconds,
including restoration work that is excluded from the individual throughput
figures. Both used the stock 1 GiB snapshot limit, so larger contexts restored
state by replaying prefixes.

## Numerical comparison

The unmodified benchmark saved every logit at all seven prefill frontiers:
**1,738,240 values (7 × 248,320), with byte-identical JSON dumps**. Each prefill
argmax also matched.

The 128-token greedy continuations matched at 8,192, 32,768, and 65,536 tokens.
They differed at 4,096, 16,384, 131,072, and 262,144 tokens. Thus identical
prefill logits do not imply identical subsequent decode behavior.

To compare distributions on the same histories, a shared local diagnostic
version of the benchmark driver recorded full F32 logits before each of 128
actual following prompt tokens at every frontier. The production engine and
Metal files were unchanged. This diagnostic retained those teacher-forced
tokens while extending the next prefix, exercising incremental prefill/decode
and allowing state differences to carry forward. Its timing CSVs are excluded
from the throughput table above.

Across all **896 recorded positions**:

- Top predictions agreed at **867/896 (96.76%)**.
- Maximum absolute logit difference was **4.508961**.
- Mean KL divergence, baseline to current, was **0.010916 nats**;
  maximum KL was **0.485932 nats**.
- Mean negative log-likelihood changed from **1.682503** to
  **1.682109** nats/token.
- Perplexity over these sampled tokens changed from **5.379005** to
  **5.376882** (**-0.0395%**).

| Context | Top predictions matching | Max logit difference | Baseline mean NLL | Current mean NLL | Sampled perplexity change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 128/128 | 0.001869 | 0.276832 | 0.276829 | -0.000% |
| 8,192 | 121/128 | 3.043321 | 0.886406 | 0.914835 | +2.884% |
| 16,384 | 123/128 | 1.970708 | 2.064057 | 2.064650 | +0.059% |
| 32,768 | 124/128 | 2.168268 | 2.179527 | 2.162353 | -1.703% |
| 65,536 | 126/128 | 2.125294 | 2.087021 | 2.067663 | -1.917% |
| 131,072 | 123/128 | 4.508961 | 1.856138 | 1.847765 | -0.834% |
| 262,144 | 122/128 | 3.994550 | 2.427543 | 2.440666 | +1.321% |

These measurements establish similar aggregate likelihood on this one sampled
text, while showing that the two engines are not numerically equivalent during
long incremental decoding. They do not establish broad task-quality parity.
In particular, the earlier 0.000596 MTP batching check compared paired and
sequential predictor steps; it was not a bound on whole-engine decode drift.
At the 262K checkpoint, sampled perplexity was 1.32% higher in this diagnostic.

## Evidence and reproduction

The `OUT/` links below refer to the local benchmark archive, which is not
included in Git. The measurements and numerical summaries are reproduced above.

All artifacts are under [`OUT/qwen38-mtp/bench-262k`](../OUT/qwen38-mtp/bench-262k):

- [`manifest.json`](../OUT/qwen38-mtp/bench-262k/manifest.json): commands, model
  file metadata, baseline revision, binary hashes, and runtime Metal hashes.
- [`candidate-engine.patch`](../OUT/qwen38-mtp/bench-262k/candidate-engine.patch):
  the engine changes tested against `bd9cfbc`.
- [`baseline/sweep/speed.csv`](../OUT/qwen38-mtp/bench-262k/baseline/sweep/speed.csv)
  and [`candidate/sweep/speed.csv`](../OUT/qwen38-mtp/bench-262k/candidate/sweep/speed.csv):
  unmodified full-sweep measurements.
- [`comparison.json`](../OUT/qwen38-mtp/bench-262k/comparison.json): exact
  prefill-logit and generated-text comparisons, with hashes.
- [`repeat-8k/results.json`](../OUT/qwen38-mtp/bench-262k/repeat-8k/results.json):
  all additional 8K measurements.
- [`teacher-forced/comparison.json`](../OUT/qwen38-mtp/bench-262k/teacher-forced/comparison.json):
  per-position numerical and likelihood statistics.
- [`teacher-forced/ds4_bench_diag.c`](../OUT/qwen38-mtp/bench-262k/teacher-forced/ds4_bench_diag.c):
  the shared diagnostic driver; its manifest records binary/source hashes.

The stock command for each build was:

```sh
./ds4-bench -m /path/to/Qwen3.8-Flash-Next-Q4KImatrix-MTP-qwen4exp-pleext.gguf \
  --ple /path/to/Qwen3.8-Flash-Next-PLE-Q4_1.gguf --metal \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 4096 --ctx-max 262144 --step-mul 2 --gen-tokens 128 \
  --show-output --csv speed.csv --dump-frontier-logits-dir logits
```

Create the logit output directory before running. Build the baseline in a
separate checkout at `bd9cfbc`, and pin its runtime Metal sources. The saved
[`run.py`](../OUT/qwen38-mtp/bench-262k/run.py),
[`compare.py`](../OUT/qwen38-mtp/bench-262k/compare.py), and teacher-forced scripts
record the exact local workflow.
