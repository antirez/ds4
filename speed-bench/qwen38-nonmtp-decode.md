**Qwen3.8 non-MTP prefill and decode optimization — 2026-09-05**

The [compact results](qwen38-nonmtp-results.json) are tracked alongside this
report. Detailed `OUT/` links below refer to the local benchmark archive;
generated binaries and full logit dumps are not included in Git.

The selected v1 runtime restores the original engine's numerical results in the completed 262K checks and improves decode in both full-sweep orders. Repeated fresh-8K prefill improves 4.50%; long-context prefill varies, with remaining losses at 65K/131K across the two observations. The first decode evaluation is slower.

The **numerical reference is `bd9cfbc`**; the **performance baseline is `af4195c`**, the previous MTP-optimized engine. [Frozen v1](../OUT/qwen38-nonmtp-round2/final/freeze-manifest.json) pins the candidate patch, binaries, and runtime Metal sources. The [selected-source manifest](../OUT/qwen38-nonmtp-round2/selected-source/manifest.json) confirms unchanged runtime code plus test hardening. Compact-worklist code remains isolated.

Measurements use Apple M3 Ultra, 512 GiB unified memory, macOS 27.0, `Qwen3.8-Flash-Next-Q4KImatrix-MTP-qwen4exp-pleext.gguf`, and its `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` sidecar. Non-MTP benchmarks use the frozen Promessi sposi text, 262,273 allocated context positions, sequential GPU jobs, sanitized environment settings, and greedy decode. The [validation manifest](../OUT/qwen38-nonmtp-round2/final/validation-manifest.json) records exact commands, prompt hash, model metadata, and run order.

**Numerical correction.** In the [controlled ablation](../OUT/qwen38-nonmtp-round2/ablation/results.json), `af4195c` matched only 1/256 original FP32 logit vectors, with maximum absolute logit difference 3.043321. Disabling GDN r4 gave the same result; disabling the specialized Q4K mid kernel restored 256/256 exact logit vectors. That helper changed accumulation order, quantization scale/minimum grouping, and SiLU evaluation. Router weighting was already correct.

The replacement shares input loads across gate/up and two rows while preserving original lane/block/element order and `qwen4_silu`. GDN r4 remains unchanged. V1 submits commands after two trunk layers and uses M3 Ultra MoE launch caps of 8/16/32 for batches below 4096/from 4096/from 8192 tokens. Tiles remain 32 tokens wide with unchanged K-loop arithmetic.

The completed [numerical comparison](../OUT/qwen38-nonmtp-round2/final/comparison.json) gives **896/896 full-vocabulary FP32 decode logit vectors byte-identical to `bd9cfbc`**, all top predictions matching, maximum difference 0. It scores 128 following prompt tokens at each doubling frontier from 4096 through 262144, retaining teacher-forced tokens between frontiers to exercise matching mixed prefill/decode histories. Across both sweep orders, [all 28 frontier-logit JSON dumps and all 14 candidate greedy continuations](../OUT/qwen38-nonmtp-round2/completed-comparison.json) match the original engine exactly. The 28 dumps cover both engines at seven frontiers in both orders. Four candidate continuations in the first sweep differ from `af4195c`.

The [hardened kernel suite](../OUT/qwen38-nonmtp-round2/final-test-hardening/kernel-tests.log) and [speculative planner tests](../OUT/qwen38-nonmtp-round2/final/test_qwen4_spec.log) pass. Tests cover Q4K T1/T2, odd tails, shared experts, SIMD groups 1–8, downstream outputs, and T641 tile-cap comparisons. Integer float bits ensure nonfinite rejection under fast-math. These checks remain scoped to the tested inputs, model, and device.

**Non-MTP performance against `af4195c`.** Three fresh 8192-token pairs, each followed by 256 generated tokens and alternating order, give median prefill **1149.37→1201.06 t/s (+4.50%)** and total decode **45.59→47.24 t/s (+3.62%)**. Median first-evaluation latency worsens **23.150→35.498 ms**, while steady decode improves 45.69→47.45 t/s. `gen_first_ms` measures the first post-prefill evaluation, not end-user time to first displayed token.

A separate [two-repeat flush comparison](../OUT/qwen38-nonmtp-round2/flush-latency/manifest.json) found layer 1 submission reduced first-evaluation latency from 31.8–31.9 to 22.0–22.2 ms, but lowered steady decode from 47.51–47.58 to 47.01–47.03 t/s. Layer 2 remains the selected default.

The first stock doubling sweep generates 128 tokens per frontier. Rates are baseline→v1:

| Context | Newly prefilled tokens | Prefill t/s | Change | Decode t/s | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 4,096 | 1101.66 → 1104.77 | +0.28% | 45.63 → 47.31 | +3.68% |
| 8,192 | 4,096 | 1146.35 → 1158.33 | +1.05% | 45.44 → 47.13 | +3.72% |
| 16,384 | 8,192 | 1167.53 → 1192.81 | +2.17% | 45.25 → 46.88 | +3.60% |
| 32,768 | 16,384 | 1164.70 → 1146.74 | -1.54% | 45.25 → 46.79 | +3.40% |
| 65,536 | 32,768 | 1155.76 → 1095.71 | -5.20% | 44.86 → 46.45 | +3.54% |
| 131,072 | 65,536 | 1139.44 → 1068.50 | -6.23% | 44.11 → 45.63 | +3.45% |
| 262,144 | 131,072 | 1044.30 → 1036.92 | -0.71% | 42.40 → 43.64 | +2.92% |

This single sweep ran baseline first; process wall time increased 464.15→478.28 seconds. Prefill rates measure newly appended tokens. Snapshot operations and full-prefix restoration are outside throughput windows; restoration switches to replay after the 32K snapshot exceeds 1 GiB. The 32K timed interval precedes that first replay, so replay alone cannot explain the slowdown's onset. Both graphs keep 8192-token chunks and wait for GPU completion inside the timer.

The [reverse-order sweep](../OUT/qwen38-nonmtp-round2/repeat-262k/manifest.json) then ran candidate first, baseline second; both completed all seven frontiers successfully:

| Context | Prefill t/s, baseline→v1 | Change | Decode t/s, baseline→v1 | Change |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1102.87 → 1159.20 | +5.11% | 45.37 → 47.47 | +4.63% |
| 8,192 | 1147.81 → 1161.12 | +1.16% | 45.40 → 47.17 | +3.90% |
| 16,384 | 1167.27 → 1194.74 | +2.35% | 45.32 → 47.04 | +3.80% |
| 32,768 | 1166.25 → 1189.93 | +2.03% | 45.18 → 46.84 | +3.67% |
| 65,536 | 1157.03 → 1180.03 | +1.99% | 44.86 → 46.71 | +4.12% |
| 131,072 | 1140.97 → 1161.91 | +1.84% | 44.12 → 45.75 | +3.69% |
| 262,144 | 1108.38 → 1128.65 | +1.83% | 42.45 → 43.83 | +3.25% |

Decode improved at every frontier in both orders. Prefill improved throughout the reverse sweep, but two-observation per-build medians still show **-1.60% at 65K and -2.19% at 131K**. At 262K, the combined medians are 1076.34→1082.79 prefill t/s (+0.60%) and 42.425→43.735 decode t/s (+3.09%). Candidate 262K prefill ranged 1036.92–1128.65 t/s; baseline ranged 1044.30–1108.38. Reverse process wall time was 453.36→444.66 seconds. All observations remain in the [combined results](../OUT/qwen38-nonmtp-round2/completed-comparison.json). Two runs establish variability, not a uniform long-prefill improvement or its cause.

**Scheduling follow-up.** A [five-case prefill-only ablation](../OUT/qwen38-nonmtp-round2/long-prefill-ablation-doubling/manifest.json), using the same v1 binary with zero generated tokens, retained every observation:

| Configuration, in execution order | Fresh 32768 t/s | Additional 32768 t/s |
| --- | ---: | ---: |
| Cap 8, flush disabled | 1162.51 | 1157.88 |
| V1 defaults | 1192.15 | 1179.39 |
| Cap 8, flush after layer 2 | 1169.53 | 1155.82 |
| Adaptive caps, flush disabled | 1186.50 | 1165.59 |
| Cap 8, flush disabled, repeated last | 1145.22 | 1129.93 |

All five produced identical saved logits at both frontiers. Defaults improved this workload; the stock regression was not reproduced. The slower final control demonstrates variability without establishing a cause. An [earlier additive-step run](../OUT/qwen38-nonmtp-round2/long-prefill-ablation/manifest.json) remains preserved separately. Decode/replay history and clock/cache effects remain hypotheses.

The isolated worklist passed [focused byte-exact tests](../OUT/qwen38-nonmtp-round2/worklist-focused.log) covering six weight types, empty lists, tails, and guards. Its [8K screen](../OUT/qwen38-nonmtp-round2/worklist-screen/manifest.json) gave 1185.93–1194.04 prefill t/s versus 1166.49–1175.39 for uniform controls, below earlier v1 results. A [65K sweep](../OUT/qwen38-nonmtp-round2/worklist-long/manifest.json) improved prefill about 1% (1179.88→1193.12 t/s at 65K), with all five logit dumps identical and inconsistent decode gains. This marginal evidence did not justify selection.

[64-token tiles](../OUT/qwen38-nonmtp-round2/screen-wide/manifest.json), [GDN r8](../OUT/qwen38-nonmtp-round2/screen-r8/manifest.json), and [direct staging](../OUT/qwen38-nonmtp-round2/screen-ordered/manifest.json) were also rejected. The noisy [4K cap sweep](../OUT/qwen38-nonmtp-round2/caps-4096/manifest.json), including prefill 789.45 t/s and decode 19.62 t/s observations, remains intact.

**MTP regression check.** One measured repetition per prompt, following warmup, produced identical output bytes and acceptance counts between `af4195c` and v1: Hamlet 37/56, Fibonacci 198/201, explanation 110/145. Rates were 60.50→61.53, 73.09→74.37, and 64.40→65.61 t/s respectively. This is a [three-case regression smoke test](../OUT/qwen38-nonmtp-round2/mtp-regression/results.json), not a repeated MTP performance benchmark.
