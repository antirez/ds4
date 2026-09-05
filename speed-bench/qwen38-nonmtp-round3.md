**Qwen3.8 non-MTP optimization, round 3 — 2026-09-06**

The selected build matches the original engine results in the completed 262K checks and improves decode in both sweep orders. Fresh 8K prefill improves slightly; long-context prefill remains mixed. The [compact results](qwen38-nonmtp-round3-results.json) retain commands, inputs, frozen binary/source hashes, and every observation. Detailed `OUT/` links refer to the local archive; large dumps and binaries are not tracked in Git.

The performance reference is `8ec3770`, committed and pushed before this round. The numerical reference remains `bd9cfbc`. Tests use Apple M3 Ultra with 512 GiB unified memory, the Qwen3.8-Flash-Next Q4KImatrix model and Q4_1 PLE sidecar, and the frozen Promessi sposi prompt. Jobs run serially with sanitized settings and pinned Metal sources.

**Repeated fresh 8K measurements.** Three pairs alternate baseline/candidate order, after warmup, with 256 greedy target tokens per run. These are ratios of per-build medians:

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Prefill, t/s | 1199.540 | 1204.400 | +0.41% |
| Decode, t/s | 47.060 | 48.630 | +3.34% |
| Steady decode, t/s | 47.270 | 48.850 | +3.34% |
| First evaluation, ms | 33.679 | 32.381 | -3.85% |

`gen_first_ms` measures the first post-prefill evaluation, not time to the first displayed token. These small samples describe the observations; no statistical significance is claimed.

**Full stock sweeps.** Each build completed all seven doubling frontiers, with 128 generated tokens per frontier and 262,273 context positions allocated. Order was baseline→candidate, then candidate→baseline. Prefill throughput measures newly appended tokens; snapshot/restore or prefix replay work is outside that timing window. Values below are arithmetic averages of the two observations per build.

| Context | Prefill t/s, baseline → candidate | Change | Decode t/s, baseline → candidate | Change |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1123.91 → 1098.60 | -2.25% | 47.170 → 48.745 | +3.34% |
| 8,192 | 1160.53 → 1161.89 | +0.12% | 47.060 → 48.575 | +3.22% |
| 16,384 | 1191.65 → 1180.57 | -0.93% | 46.820 → 48.455 | +3.49% |
| 32,768 | 1150.84 → 1118.24 | -2.83% | 46.595 → 48.190 | +3.42% |
| 65,536 | 1131.55 → 1093.34 | -3.38% | 46.310 → 47.945 | +3.53% |
| 131,072 | 1090.26 → 1076.93 | -1.22% | 45.660 → 47.125 | +3.21% |
| 262,144 | 1030.27 → 1041.22 | +1.06% | 43.575 → 44.915 | +3.08% |

The first sweep showed substantial mid-context prefill losses. The baseline also slowed in the reverse sweep: at 32K it moved from 1187.71 to 1113.96 t/s, while the candidate measured 1115.99 and 1120.50. Run order did not eliminate all differences. Both observations remain in the results; these measurements do not establish a uniform long-prefill improvement or the cause of the variability.

Forward sweep prefill changes by context: 4,096: -4.47%, 8,192: +0.25%, 16,384: -2.11%, 32,768: -6.04%, 65,536: -7.22%, 131,072: -3.71%, 262,144: +0.33%.
Forward sweep: whole-process wall time was 464.89→476.99 seconds, baseline→candidate. This includes work outside the throughput windows.

Reverse sweep prefill changes by context: 4,096: +0.09%, 8,192: -0.01%, 16,384: +0.25%, 32,768: +0.59%, 65,536: +0.80%, 131,072: +1.39%, 262,144: +1.81%.
Reverse sweep: whole-process wall time was 483.16→475.53 seconds, baseline→candidate. This includes work outside the throughput windows.


**Implementation and numerical validation.**

On M3 Ultra, one-token Q4K decode now uses one output row per SIMD group and eight SIMD groups per threadgroup (NR1/NSG8). Two-token MTP verification and other devices retain NR2/NSG2 automatically. Explicit NR2 restores the former geometry when no NSG override is present. The ordered block and element accumulation sequence, lane reductions, and stable SiLU remain unchanged.

HC normalization automatically reuses each stream's RMS only on M3 Ultra with E2560, four HC streams, four injection rows, and batches of at least 8,192 tokens. It retains the original 128-thread RMS reduction and eight separate chunk reductions and partial outputs. A barrier protects shared reduction storage between chunks. Explicit 0/1 settings disable/enable reuse, with the original path retained for one- and two-token batches. Weight precision and quantization are unchanged.

The final kernel suite and speculative planner tests passed. The suite includes 29 HC fixtures comparing forced-off, forced-on, and automatic dispatch across F32/F16/Q8, tails, injection counts, guarded offsets, and full outputs at 8,191/8,192 tokens. Its 68 Q4K comparisons cover NR1/NR2, NSG1–8, defaults, T1/T2, shared experts, and odd tails. The final diagnostic matched all 896 full-vocabulary FP32 logit vectors byte-for-byte against `bd9cfbc`: 128 steps at each of seven contexts from 4,096 through 262,144 tokens, with identical top-1 choices and maximum absolute difference zero. The three-prompt MTP smoke also matched generated outputs and acceptance.

NR4 was slower in the measured decode screens and was removed. Experimental flush policies either failed to improve first evaluation consistently or traded lower first-evaluation latency for slower prefill; they were removed. These results establish equivalence on the tested finite histories and fixtures, rather than general model quality.

Across both stock sweep orders, all 28 serialized frontier-logit dumps and all 14 candidate greedy continuations match the original archive. Serialized dump equality is separate from the direct full-FP32 diagnostic. The latter scores 128 following prompt tokens at each doubling frontier and retains the teacher-forced history between frontiers.

The [frozen selected build](../OUT/qwen38-nonmtp-round3/final/manifest.json) identifies the runtime patch and test binaries. [Full diagnostic evidence](../OUT/qwen38-nonmtp-round3/diagnostics/final-262k/comparison.json) records 896/896 exact vectors and maximum difference zero. The [earlier round](qwen38-nonmtp-decode.md) documents the preceding numerical correction.

**MTP regression smoke.** One measured pair per prompt, following warmup, retained identical output bytes and acceptance counts:

| Prompt | Accepted / verify cycles | Decode t/s, baseline → candidate |
| --- | ---: | ---: |
| Hamlet | 37 / 56 | 60.57 → 60.73 |
| Fibonacci | 198 / 201 | 74.10 → 74.29 |
| Explanation | 110 / 145 | 65.22 → 65.22 |

This checks these three MTP cases; it is not a repeated MTP performance benchmark.

**Selection evidence.** HC kernel timings used seven alternating pairs per weight type and batch size. All three types improved at 8192 tokens (14–25% kernel throughput), whereas F32 at 4096 was slightly slower; the automatic threshold therefore stays at 8192. The full-engine fresh-8K benefit is much smaller. The [experiment inventory](../OUT/qwen38-nonmtp-round3/evidence-review/notes.md) retains the latency and NR screens, raw observations, and provenance limits.
