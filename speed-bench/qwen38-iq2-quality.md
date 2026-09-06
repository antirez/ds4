# Qwen3.8 IQ2_XXS quality experiment

IQ2_XXS is calibrated directly from the original BF16 trunk gate/up weights. All other tensors, including MXFP4 down projections and the MTP block, are copied from the existing Q4_K-imatrix combined GGUF and verified after writing. The required Q4_1 PLE sidecar remains external.

Main model: **50.34 GB (46.89 GiB)**, including MTP. This is model storage, not the complete runtime memory budget.

This is a promising candidate for a 64 GB memory budget: the hard-suite score remains comparable, but the BF16 probability comparison shows substantially more quantization drift than the four-bit recipes. Validate it on the target 64 GB machine and representative tasks before replacing a higher-precision build.

## Hard evaluation

All four recipes use the same frozen evaluator and Metal kernels on M3 Ultra, `--suite hard --ctx 32768 --temp 0 --seed 123`, with a 1024-token prefill chunk. Native per-case generation budgets and the evaluator’s thinking-close controller are retained. Speculation and retries are disabled. Each model is evaluated once on the same 50 cases.

| Recipe | Passed / 50 | Failed | Incomplete |
| --- | ---: | ---: | ---: |
| q4k | 35 | 2 | 13 |
| q40 | 35 | 0 | 15 |
| q8 | 34 | 2 | 14 |
| iq2 | 36 | 3 | 11 |

Passes by source:

| Source | Q4_K | Q4_0 | Q8 | IQ2 |
| --- | ---: | ---: | ---: | ---: |
| MMLU-Pro | 23/30 | 24/30 | 22/30 | 25/30 |
| OlympiadBench | 2/10 | 2/10 | 2/10 | 2/10 |
| LiveBench | 5/5 | 5/5 | 5/5 | 5/5 |
| NIST Juliet | 5/5 | 4/5 | 5/5 | 4/5 |

Paired differences count the exact questions on which the models disagree:

| IQ2 versus | IQ2 alone passed | Baseline alone passed |
| --- | ---: | ---: |
| q4k | 2 | 1 |
| q40 | 2 | 1 |
| q8 | 3 | 1 |

These small, fixed suites do not establish general quality equivalence. Q8 uses Q8 routed down weights as well as gate/up; Q4_0 is an earlier complete recipe. The most controlled quantization comparison is IQ2 versus Q4_K-imatrix.

## BF16-reference continuations

The existing exact-checkpoint fixture is rerun on all four builds. Fifteen cases with mismatched target-token counts are excluded consistently, leaving 85 aligned cases. This is a probability comparison on short continuations, not a long-context reasoning benchmark.

| Recipe | Target NLL ↓ | First token / 85 | Top-1 agreement | Logprob MAE ↓ |
| --- | ---: | ---: | ---: | ---: |
| q4k | 0.21287 | 74 | 96.57% | 0.04307 |
| q40 | 0.20959 | 73 | 96.52% | 0.04297 |
| q8 | 0.19828 | 79 | 98.48% | 0.02070 |
| iq2 | 0.27828 | 58 | 91.32% | 0.13992 |

## Memory and provenance

The 32K→64K ordinary-decode sweep allocates 65,793 context positions with a 1024-token prefill chunk. macOS `time -l` reports peak RSS **33.57 GB (31.27 GiB)** and peak footprint **3.54 GB (3.29 GiB)**. These process counters do not reliably account for all mmap-backed GPU weights and must not be interpreted as the total model memory requirement. A repeat of the same sweep with Metal allocation accounting reports **2.92 GiB** peak tracked runtime tensors. The main model file plus those tracked runtime allocations totals **53.48 GB**; this is a partial budget, excluding OS and driver overhead, other host allocations and resident pages of the external PLE sidecar. PLE pages are demand-paged and reclaimable, but their memory cost is not zero. The test machine has 512 GiB; this does not validate Metal allocation limits or OS headroom on a physical 64 GB Mac. The MTP smoke test is separate from this ordinary-decode memory measurement.

Ordinary decoding measured 47.88 tokens/s at 32K and 47.49 tokens/s at 64K, each for 128 generated tokens. The sweep extends a shared prompt prefix from 32K to 64K; the second prefill measures the additional 32K tokens. The CSV’s generic `kvcache_bytes=0` field is not implemented for this architecture; the runtime allocation report above supplies the memory accounting.

A separate ordinary/MTP pair uses identical greedy settings, a 256-token output cap and 64K allocated context. Generated stdout matches byte-for-byte: **True**. This is a short-prompt correctness smoke test, not a full-context MTP quality evaluation.

The same pinned Unsloth imatrix as the Q4 build is used. Sixteen zero-count gate/up entries use the existing deterministic weight-energy fallback; calibration provenance and per-tensor source hashes are retained in the artifact manifest.

Validation: `make -j8 ds4 ds4-eval`, evaluator case validation and extractor self-tests, 35 offline Qwen pack tests, and all 1,255 written tensor readback hashes passed. All four hard runs completed 50 cases; their nonzero grading exit codes reflect wrong/incomplete answers, not an execution failure. The supplementary MTP check accepted 120 drafts in 135 verification cycles (88.9%).

[Machine-readable results](qwen38-iq2-quality-results.json) include per-case scores, commands, hashes, paired disagreements and calibration details. Full local traces and logits remain under the ignored `OUT/qwen38-iq2-quality` directory.
