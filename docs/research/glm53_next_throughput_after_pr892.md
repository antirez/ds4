# GLM-5.3 throughput after PR #892

Date: 2026-08-30  
Decision ticket: #16  
Scope: generated tokens/s on M4 Max after private PR #13, plus production agent throughput where cache replay dominates.

## Decision

Better MTP draft quality is possible, but it is **not the best next DS4 implementation pass**.

The prescribed width-3 experiment is now complete. A transaction prototype
captured KDA state after both retainable prefixes, avoided rejection replay,
and committed 256/256 tokens that teacher-forced as target argmax. It still
failed the performance gate and was removed from the runtime: across two
interleaved runs of three 128-token prompts, width 3 averaged `28.42 t/s`,
versus width 2 at `32.25 t/s` and target-only at `29.08 t/s`. Width 3 was
`11.9%` slower than width 2 and `2.3%` slower than target-only.

The useful pieces remain: width-2 verification now uses a small speculative
state transaction interface, `--mtp-timing` reports committed-token Utility,
and the GLM regression teacher-forces speculative output to require target
equivalence within a narrow numerical-tie band. The next steady-decode pass is therefore
profile-led target-kernel work, not wider speculation or a runtime acceptance
shortcut.

### Reproduction record

All throughput arms used the same command shape on the local Apple M4 Max,
changing only `--mtp-timing` and the diagnostic width environment variable:

```sh
./ds4 -m /Users/chriskz/dev/ds4/gguf/GLM-5.3-Flash-Q2.gguf \
  --metal --ctx 8192 --tokens 128 --temp 0 --nothink \
  --prompt-file PROMPT

DS4_GLM_MTP_WIDTH=2 ./ds4 ... --mtp-timing
DS4_GLM_MTP_WIDTH=3 ./ds4 ... --mtp-timing
```

The width variable existed only in the prototype and was removed after the
failed gate. Runs were interleaved by prompt, then the corpus was repeated:

- code: “Write a compact C function that parses an unsigned decimal integer
  from a byte span without allocation. Return a structured error for empty
  input, non-digit bytes, and uint32 overflow. Include the function, its data
  types, and five table-driven tests. Output code only.”
- JSON: “Return JSON only. Design a deployment record with keys service,
  version, regions, health, dependencies, and rollback. Include four regions,
  eight dependencies with status and latency_ms, and a rollback object
  containing eligible, reason, and checkpoint. Keep the JSON valid and
  deterministic.”
- systems: “Explain how an append-only write-ahead log, checkpoints, and
  idempotent replay combine to recover a stateful service after a crash. Give
  the invariants, recovery sequence, and the three most dangerous edge cases.
  Be precise and concise.”

| Prompt | target-only t/s | width 2 t/s | width 3 t/s |
|---|---:|---:|---:|
| code, run 1 / 2 | 29.11 / 29.09 | 32.86 / 32.98 | 28.84 / 28.86 |
| JSON, run 1 / 2 | 29.06 / 29.06 | 33.75 / 33.92 | 31.10 / 31.08 |
| systems, run 1 / 2 | 29.05 / 29.11 | 30.03 / 29.94 | 25.28 / 25.34 |
| arithmetic mean | **29.08** | **32.25** | **28.42** |

Width-3 telemetry recorded `132/185 = 71.4%` first-depth acceptance and
`63/132 = 47.7%` conditional second-depth acceptance. The correctness test
teacher-forced the deterministic code-copy prompt and required all 256
speculative tokens to remain within `0.1` logit of the serial target argmax
(`worst_argmax_gap=0.019` on the final recorded run). A strict byte-stream comparison was intentionally not retained:
equal-best numerical ties can select different token IDs between serial and
batched target evaluation. The three-prompt corpus above is a throughput
corpus, not an additional exactness claim.

The obvious no-training version of that idea has already been tested. The local `GLM-5.3-Flash-Q2-MTPQ4` artifact raises only the three routed-expert tensors in the nextn support layer from `IQ2_XXS/IQ2_XXS/Q2_K` to `Q4_K`. Across three 128-token M4 Max prompts, stock Q2 and MTPQ4 accepted exactly the same total number of drafts: `152/230` (`66.09%`). MTPQ4 was about 4% slower. More support precision did not improve aggregate draft quality and increased work.

Training a better nextn block could improve acceptance, but DS4 contains conversion and inference code, not a support-model training pipeline. It would require training data or distillation traces, the source checkpoint, a new model artifact, and quality/acceptance validation. That is a model project, not the next runtime optimization.

Those two investigative routes are now resolved. Cost-aware Utility telemetry is
retained for offline diagnosis. The replay-free speculative ceiling experiment
failed its performance gate, so width 3 and its second-prefix checkpoint were
deleted. The higher-confidence implementation route is the public
#904/#905/#903 replay and checkpoint stack; for steady decode, the next step is
an M4 Metal trace followed by an exact target-kernel change justified by that
profile. Do not ship a runtime acceptance gate under the current output
requirement: a prior confidence-selective gate changed target-only tokens, and
PR #892 also records a long-prompt near-tie difference between serial and MTP
execution.

If changing the target quantization is acceptable, a quality-gated reduction of always-active Q8 target weights to Q4_K probably has higher expected raw decode upside than either route. It accelerates every token, but it changes target logits and therefore is not an exact-output optimization.

## Evidence

### 1. Acceptance alone is not the objective

The repository's useful metric is **Utility**: expected committed tokens divided by expected cycle cost. Acceptance is only one input.

- Private [PR #13](http://192.168.2.106:3002/ckz_data_labs/ds4/pulls/13) measured M4 Max verify2 near `48.9 ms`. Its decode-style row pass saved only about `1 ms`; the main win was capturing KDA state after row 0, cutting rejection work from `39.6 ms` to about `5.1 ms`. At `61%` acceptance, generation improved from `23.92` to `29.16 t/s`, approximately target-only parity.
- Public [PR #892](https://github.com/antirez/ds4/pull/892) measured M5 Max short-prompt acceptance of `89.6%`, with `33.0 t/s` target-only and `40.5 t/s` width-2 MTP. Yet on its 4,500-token prompt, target-only was `29.70 t/s` and MTP was `29.26 t/s` despite roughly `90%` acceptance. Verify and fixed cycle cost, not draft quality, bound that case.
- The same PR measured depth-2 acceptance near `45%`; width 3, 4, and 6 fell to `30.6`, `20.8`, and `16 t/s`. Each rejection restored the original KDA state and replayed the accepted prefix through another target forward. The local experiment removed that replay and still measured width 3 at `28.42 t/s`, below both controls.

Local M4 acceptance also varies sharply by prompt. The fixed PR #13 branch at `ce9c2fc` produced:

| Prompt | Q2 support acceptance | Q2 MTP t/s | MTPQ4 acceptance | MTPQ4 t/s |
|---|---:|---:|---:|---:|
| code | 54/73 = 73.97% | 31.81 | 54/73 = 73.97% | 29.18 |
| JSON | 44/84 = 52.38% | 26.45 | 45/83 = 54.22% | 26.33 |
| systems | 54/73 = 73.97% | 30.20 | 53/74 = 71.62% | 29.35 |
| total | **152/230 = 66.09%** | arithmetic mean 29.49 | **152/230 = 66.09%** | arithmetic mean 28.29 |

These are one run per arm, so throughput is directional. The acceptance result is stronger: the precision change redistributed two prompt-level outcomes but changed no aggregate accepted draft count.

### 2. MTPQ4 already tests the practical precision idea

`--inspect` reports:

- Q2: `89.88 GiB`; 43 Q2_K, 68 Q4_K, 86 IQ2_XXS tensors.
- MTPQ4: `91.78 GiB`; 42 Q2_K, 71 Q4_K, 84 IQ2_XXS tensors.

A direct GGUF header comparison found exactly three changed tensors, all in support layer 45:

```text
blk.45.ffn_gate_exps.weight  IQ2_XXS -> Q4_K
blk.45.ffn_up_exps.weight    IQ2_XXS -> Q4_K
blk.45.ffn_down_exps.weight  Q2_K    -> Q4_K
```

The nextn fusion projection is already BF16; nextn norms are F32; DSA projections are Q8/BF16; the shared output head is unchanged. The [conversion plan](../../gguf-tools/glm53_quantize.py) identifies layer 45 as the nextn layer and excludes it from the 45-layer target trunk. Thus this is a clean support-model precision A/B: target weights are identical, while the routed support block gains 1.90 GiB.

The result rules out “raise the support MoE from Q2 to Q4” as the next pass. A full BF16/FP8 support layer could still be tested, but the unchanged acceptance makes a large payoff unlikely enough that it should not precede state checkpointing or a workload utility gate.

### 3. PR #13 supplies the first checkpoint, not the general solution

PR #13 adds one full pre-verify KDA backup and one post-row-0 KDA prefix snapshot. A rejected width-2 draft restores the prefix without a target replay. The local correctness follow-up `ce9c2fc` also preserves DSA indexer pool tail rows so replacement of an unaccepted row rebuilds the same pool as serial decode; see the [speculative pool regression](../../tests/test_glm53_kda.c).

The retained `glm53_spec_transaction` is intentionally narrower than a general
session transaction. It owns the verify-cycle KDA base/prefix snapshots and
dense-cache commit/restore rules for width 2. Server stop-boundary rewind and
long-lived session checkpoints remain in their existing modules because they
operate at different lifetimes. This puts the coupled hot-path state behind one
small interface without creating a speculative abstraction wider than the
winning implementation.

The deleted width-3 prototype added the second prefix checkpoint and raw-hidden
support-head chain needed to test the ceiling without rejection replay. One KDA
snapshot is about 146 MiB for 34 KDA layers. Its measured `47.7%` conditional
depth-2 acceptance was not enough to repay the extra draft and snapshot cost:
width 3 lost to width 2 on all three prompts and to target-only in aggregate.

## Current public branch and open PRs

The current [`glm-5.3-flash` open queue](https://github.com/antirez/ds4/pulls?q=is%3Apr+is%3Aopen+base%3Aglm-5.3-flash) contains seven PRs as of this review: #892, #894, #899, #903, #904, #905, and #909.

Worth integrating or expanding:

- [#905](https://github.com/antirez/ds4/pull/905) rewinds an unconsumed suffix from a speculative result block. Its real reproduction shows one invisible token forcing a rebuild of 109,751 tokens from a 9,703-token cold anchor. This is not steady decode t/s, but it can dominate agent wall time.
- [#904](https://github.com/antirez/ds4/pull/904) canonicalizes GLM thinking/tool replay. Its reproduction lost a valid 65,272-token live prefix because a 166-token tool turn replayed with a different token representation.
- [#903](https://github.com/antirez/ds4/pull/903) saves continued checkpoints after crossing interval frontiers rather than requiring exact modulo alignment. It repairs long-session recovery and complements #904/#905.
- [#892](https://github.com/antirez/ds4/pull/892) is valuable as an experimental branch and source of the width-3 implementation, raw-hidden chaining fixes, and negative results. Its relevant ideas were extracted into the replay-free local prototype; the losing runtime path was then removed rather than merging the PR wholesale.
- [#909](https://github.com/antirez/ds4/pull/909) fixes Metal working-set admission near 128 GiB. It improves reliability, not tokens/s.
- #894 and #899 are server thinking/replay correctness fixes, not generated-token optimizations.

Already implemented or already measured negative:

- PR #13: GLM row verify and one-prefix KDA rollback. The rollback is the large win; verify wiring is only about `1 ms`.
- Local MTPQ4: support routed experts at Q4. No aggregate acceptance gain; slower in the one-run A/B.
- PR #892 plus the local replay-free follow-up: wider embedded MTP. Removing the target replay did not overcome the extra draft/snapshot/verify cost, so width 3 still loses.
- PR #892: enabling the existing fused IQ2 pair-SwiGLU path for GLM's 8-of-288 MoE. Prefill regressed (`296` vs `304 t/s` median), so it was reverted.
- PR #892: GLM prefill chunk 1024/2048/4096 sweep. Noise-level.
- Private PR #12: paired M4 Q4_K KDA Q/K projections. Microbenchmark +5.07% for that subpath, estimated only `0.4–0.7%` whole-token gain.
- Public [#864](https://github.com/antirez/ds4/pull/864): IQ2_XXS/Q2_K prefill kernels, about `5–8%` prefill improvement; decode unchanged.
- Public [#874](https://github.com/antirez/ds4/pull/874): +2.42% exact decode on M3 Ultra for DeepSeek V4, plus useful negative results. Its FP8 KV changes do not directly apply to GLM's KDA/DSA state layout.
- Public branch commit [`10badf6`](https://github.com/antirez/ds4/commit/10badf696c1975d27f6f65a74ca76754e5fa0713): the Q4 artifact already reduces BF16 KDA/embedding/output weights to Q8 with a measured `0.109%` aggregate-NLL increase. This validates a quality-gated artifact route, but does not implement the proposed Q2 Q8-to-Q4 active-byte profile.
- Public branch commit [`6cf658a`](https://github.com/antirez/ds4/commit/6cf658a4da3fc20f4f6717f05746d44a3823cdde): M3 Ultra BF16 QKV dispatch fusion and threadgroup tuning. It is device- and tensor-type-specific; current Q2 on M4 uses Q4 Q/K and Q8 V, so it is not the missing M4 Q2 pass.

## Ranked next passes

### Production agent throughput

1. Integrate #904, #905, and #903 after rebasing and correctness review. Avoided 65K–110K-token re-prefills dwarf single-digit decode gains.
2. Use the retained GLM MTP Utility telemetry to select experiments and document safe manual defaults. A runtime on/off policy is blocked by the target-only output requirement until broad goldens prove otherwise.

### Steady generated tokens/s, exact relative to current Q2

1. Profile-led exact GLM/M4 kernel work, especially concurrent routed/shared FFN execution. Expect incremental gains; PR #12 indicates isolated projection fusion is sub-1% whole-token work.
2. Consider a GLM DFlash2 draft adapter only if a cheaper drafter can improve Utility, not merely acceptance. PR #892 records available same-tokenizer qwen3 draft artifacts, but no GLM integration or throughput evidence.
3. Do not expand embedded MTP beyond width 2 on the current support head. The replay-free width-3 experiment failed both required throughput controls.

### Highest potential if target quality may change

1. Build a quality-gated `q2-m4-fast` artifact that reduces always-active Q8 shared/KDA groups to Q4_K one group at a time. This applies to every generated token and is likely a better expected-return project than retraining the nextn block. It requires the full GLM quality suite, long-context checks, and MTP checks; reject any unacceptable target drift.

## Uncertainties and stop conditions

- MTPQ4 throughput has one run per arm; repeat interleaved runs before publishing the 4% number. The identical aggregate acceptance is enough to deprioritize the artifact.
- M4 conditional depth-2 acceptance measured `63/132 = 47.7%` in the
  replay-free width-3 experiment, close to the M5 result. That acceptance was
  insufficient to repay wider-cycle costs.
- PR #892's measurements predate current `glm-5.3-flash` head. Its width-3 ideas were reimplemented on the current base rather than rebasing the whole experimental branch.
- M5 is dispatch-bound in PR #892; M4 may be more bandwidth-bound. Use a Metal trace before choosing target-kernel fusion versus active-byte work.
- A better trained nextn model remains possible. Reconsider only if multi-prefix rollback is cheap and corpus telemetry shows acceptance is still the dominant Utility term.

## Completed experiment and next action

The interleaved target-only/width-2/width-3 experiment selected width 2. The
next implementation experiment should use a Metal trace to rank exact target
cost, beginning with whether routed and shared FFN work can overlap on M4.
Acceptance remains diagnostic telemetry, not a runtime gate. If target logits
may change, the separate quality-gated active-byte artifact remains the larger
potential project.
