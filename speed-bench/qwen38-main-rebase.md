# Qwen3.8 main rebase validation

The branch was rebased from `1bf4842` onto main `9ab7053` on the M3 Ultra
with 512 GiB RAM. The tested runtime is `927d845`; the following documentation
commits do not change runtime code. The original branch was 21 commits behind
and 37 ahead. Replaying its non-merge commits and removing three duplicate
patches leaves 30 commits above main.

The final tree was checked against an independent three-way merge of the old
base `b0a147a`, main `9ab7053`, and the saved branch tip `1bf4842`. Every cleanly
merged path is byte-identical. The five conflict resolutions preserve main's
test targets, documentation layout, rendered-prompt scoring, and TP/session
safeguards together with the adopted Qwen engine. Original merge decisions
that removed the superseded native Qwen runtime were retained. Qwen rewind
keeps its existing deferred replay behavior after a recurrent-state reset.

The old tip remains available locally as
`backup/qwen38-before-main-rebase-1bf4842`. Frozen binaries, shaders, commands,
and full logs are under `OUT/qwen38-mtp-round2/rebase/`; the pre-rebase baseline
under `OUT/qwen38-mtp-round2/baseline/` was left intact.
The portable [results summary](qwen38-main-rebase-results.json) records the
revision and source hashes, check outcomes, and comparison results.

## Correctness

- Clean Metal build of all five frontends, focused tests, and `score_official`:
  passed. An isolated CPU build and frontend help checks also passed.
- Server, agent, evaluation case/extractor, Qwen speculative planner, session
  bookkeeping, TP command framing, Linux memory, quality API parsing, prompt
  prefix, and GPU-argument parser checks: passed.
- Metal Qwen kernels, session rollback, GLM-5.3 KDA, and MXFP4 MoE checks: passed.
- Teacher-forced decode at contexts 4096 and 8192: all **256 full-vocabulary
  FP32 vectors** match the original `bd9cfbc` reference byte-for-byte. All
  248,320 logits per vector are finite; maximum absolute difference is zero,
  and all top-1 tokens match. The reference is also the one used to validate
  the pre-rebase `1bf4842` runtime.
- Both stock benchmark frontier dumps and both 128-token greedy continuations
  match the frozen pre-rebase binary exactly.
- A private public-API harness passed **32 full-vector comparisons** across
  snapshots, two live sessions, no-op rewind, and direct rewind/replay from
  prefix+3 to 257, 256, 254, and zero. Prefill chunks were explicitly fixed at
  256; direct evaluation after rewind exercised Qwen's deferred replay.

Existing Metal SDK deprecation warnings and a test format warning remain.

## Throughput

Same Q4_K imatrix model and external Q4_1 PLE table as the prior campaign:
`Qwen3.8-Flash-Next-Q4KImatrix-MTP-qwen4exp-pleext.gguf` and
`Qwen3.8-Flash-Next-PLE-Q4_1.gguf`. The stock `ds4-bench` comparison uses the
frozen Promessi Sposi prompt, doubling frontiers, allocation of 262273 tokens,
and 128 generated tokens per frontier. One serialized before/after pair is a
regression check, not evidence of a speed improvement.

| Context | Prefill before | Prefill rebased | Decode before | Decode rebased |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 1163.56 | 1159.79 | 48.64 | 48.91 |
| 8192 | 1158.13 | 1160.38 | 48.69 | 48.71 |

All rates are tokens/s. Prefill differs by −0.32% / +0.19%; decode differs
by +0.56% / +0.04%.

The greedy MTP check uses `qwen38_mtp_compare.py`, context 8192, two
interleaved measured repetitions per prompt, and separate warmups. Every
generated text matches exactly.

| Prompt | Median before | Median rebased | Accepted drafts before / rebased | Verify cycles before / rebased |
| --- | ---: | ---: | ---: | ---: |
| Hamlet | 60.935 | 60.725 | 37 / 37 | 56 / 56 |
| Fibonacci | 73.880 | 73.900 | 198 / 198 | 201 / 201 |
| Networking explanation | 65.635 | 65.500 | 110 / 110 | 145 / 144 |

The explanation's cycle difference is main's token-budget fix from `233eeb8`:
the public speculative API now clamps output capacity to the remaining token
budget. A separate `DS4_QWEN4_SPEC_TRACE=1` pair confirms identical earlier
events. At final position 303, the baseline verifies and rejects an unnecessary
draft, while the rebased engine evaluates the same token with a plain T1 pass.
Both produce the same 256-token output and accept 110 drafts.

## Scope

This rebase check did not rerun the 262K sweep. The previous full-context
[round-3 results](qwen38-nonmtp-round3.md) remain measurements of the pre-rebase
runtime. MTP validation here compares greedy outputs and decisions, not every
predictor logit vector. CUDA, ROCm, M5 hardware, vision-model inference, and
physical two-rank TP QA were not run on this single M3 Ultra.

The next MTP optimization round should use the rebased frozen runtime as its
baseline and repeat any provisional tuning measurements made before the rebase.
