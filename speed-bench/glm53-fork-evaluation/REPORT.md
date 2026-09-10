# M3 Ultra fork evaluation — 2026-09-09

Three decode adaptations produced repeatable gains while preserving every captured logit byte. Router/shared fusion improves short decode by about 1.6%. The final build improves original-model decode at 300K by 6.3% with router plus DSA, and the existing KDA-Q8 model at 8K by 3.7% with router plus mixed inputs, measured from restored prefixes. These are comparisons within each unchanged model, not a quality comparison between models.

Experimental branch: `glm53flash-metal-fork-eval`, based on `glm53flash-metal-exact` at `224e7669abac9fa64580e71a9a1581fa0c852d2a`. The source fork is [IngeniousIdiocy/ds4 at 95eb218](https://github.com/IngeniousIdiocy/ds4/tree/95eb218614868284fb6a9350b4be9f32c0d57575). Final source checked against full model traces and repeated timing: `0f211542950428f2469c0976a1ecf459abc6feab`. Earlier records at `fa00a42` and server follow-up `70befc3` remain in the evidence directory for provenance. The experiment is local; the user's working branch remains at the pushed baseline.

## Measured decode results

Apple M3 Ultra, 80 GPU cores, 512 GiB. Values are mean tokens/s from separate, serial processes with identical prompt, model, context allocation, and 256 greedy decode steps. Traced runs are excluded from throughput results.

| Existing model | Context | Baseline | Router/shared | + DSA selector | + mixed KDA inputs |
|---|---:|---:|---:|---:|---:|
| Original BF16 KDA | 2,048 | 26.65 | 27.06 | — | — |
| Original BF16 KDA | 8,192 | 26.42 | 26.84 | — | — |
| Original BF16 KDA | 62,174 | 25.59 | 25.98 | 26.21 | — |
| Original BF16 KDA | 300,000 | 23.20 | 23.54 | 24.62 | — |
| Existing KDA-Q8 | 2,048 | 30.69 | 31.29 | — | 31.79 |
| Existing KDA-Q8 | 8,192 | 30.44 | 30.99 | — | 31.48 |

Original short comparisons use three runs per arm; long and Q8 comparisons use two per arm. “+ DSA” and “+ mixed KDA” each include router/shared fusion; they are separate comparisons. No model was requantized. Do not compare these results directly with the fork’s 38 tok/s: that used different weights, arithmetic controls, and scheduling.

At 62,174 on KDA-Q8, mixed input fusion adds 0.54 tok/s over router alone: 29.995 → 30.535 (+1.8%), two runs per arm. DSA is disabled in the Q8 timing comparisons to isolate input fusion. All three adapters together pass Q8 trace checks through 62,174; there is no paired timing of that combination or Q8 test at 300K.

An intermediate build at `fa00a42`, after the attention invalid-row repair, was also retimed. An uninstrumented ABBA run, two processes per arm with the same saved original-model prefixes, confirmed the combined router/DSA result:

| Original-model context | Baseline mean | Intermediate mean | Gain |
|---|---:|---:|---:|
| 62,174 | 25.750 | 26.415 | +2.58% |
| 300,000 | 23.340 | 24.790 | +6.21% |

These confirmation runs are recorded separately in `reviewed-timing.json`; they are not pooled with the earlier component screens. Both arms became slightly faster in the later run, while the incremental gain remained consistent.

The final failure-recovery and checkpoint-scan fixes at `0f211542` retain the gains. Each row below uses two processes per arm in ABBA order, 256 greedy steps and identical restored prefixes; trace and dispatch-stat instrumentation are off. These measurements are recorded in `recovery-timing.json` and are not pooled with earlier runs.

| Final confirmation | Baseline mean | Final mean | Gain |
|---|---:|---:|---:|
| Original, 62,174 — router + DSA | 25.705 | 26.355 | +2.53% |
| Original, 300,000 — router + DSA | 23.300 | 24.770 | +6.31% |
| KDA-Q8, 2,048 — router + inputs | 30.625 | 31.780 | +3.77% |
| KDA-Q8, 8,192 — router + inputs | 30.325 | 31.455 | +3.73% |

## Exactness evidence

- Router/shared: 240 poisoned synthetic cases, including tied logits, NaN/Inf bias, counter reuse, and all output tensors. Every captured model logit matches at 2K, 8K, 62,174 and 300,000 tokens.
- DSA: 720 poisoned cases with finite signed scores, ties among winners and at the cut, non-finite values, subnormals, candidate overflow and host-gate limits. 100 accepted fast calls, 380 GPU fallbacks, 240 host refusals. Every captured model logit matches at 62,174 and 300,000. The real-model run accepted 5,479 of 5,632 calls (97.28%); fallback preserved original results.
- Mixed Q8/BF16 KDA: 120 poisoned cases compare all six complete projection outputs against separate existing matvecs. Full logits match at 2K, 8K and 62,174 on the existing KDA-Q8 model.
- Checkpoint maps: the upstream eight-case GLM regression produces 40 failed assertions before the fix. The final regression covers 96 combinations of whitespace, one/two calls, argument-bearing/zero-argument calls, literal closing tags in arguments, and opening tags in earlier prompt text. An additional 4,096-opener case exercises the real serializer, restores the later remembered block, and bounds scan work linearly in the checkpoint text. The server suite passes and fresh tool-map replay renders exactly the original prompt bytes.

## Scope of the adaptations

The router dispatch retains the existing F32 reduction, top-eight selector and Q8 shared gate/up arithmetic. A completion counter belongs to each graph. The shared intermediate has separate storage so the routed expert stage cannot overwrite it.

The DSA prototype adds histogram narrowing and a bounded candidate sort before our existing indexer sort. It accepts only finite, non-subnormal input with distinct top-512 scores and no boundary tie. Unsupported input dispatches the original block sort and merge chain on the GPU. Pool expansion remains the existing separate operation. The adapter is restricted to serial GLM selection, the resident M3 Ultra policy, at least 12,288 scores, and a fallback chain of at most eight dispatches. Larger contexts fall back; the fork’s separate pair-sort/group-merge rewrite was not imported.

The mixed KDA fold handles Q8 Q/K/V plus BF16 f_a/g_a/beta and retains the existing dependent f_b/g_b pair. Each output uses its original dot-product helper. SSD streaming, tensor parallelism, incompatible types/shapes and diagnostic ablation/repetition retain their prior paths.

DSA and mixed KDA are experimental opt-ins (`DS4_GLM_ENABLE_TOPK_FAST=1`, `DS4_GLM_ENABLE_KDA_Q8_INPUTS=1`). Router fusion has `DS4_METAL_DISABLE_GLM53_ROUTER_SHARED=1` as its rollback. The experiment is isolated from the working production branch.

## Method and provenance

The unchanged original model is `GLM-5.3-Flash-Q4_K.gguf` (190,875,526,464 bytes). The existing mixed model is `GLM-5.3-Flash-Q4_K-kdaQ8.gguf` (186,597,336,384 bytes). Models retain their original inode, size and mtime; this campaign records metadata, not a full weight-file SHA-256.

The prompt is `speed-bench/promessi_sposi.txt`, SHA-256 `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`. Short frontiers are 2,048 then 8,192 with context allocation 16,384. Original long frontiers are 62,174 then 300,000 with allocation 301,000. The latter baseline continues the restored 62,174-token prefix; its 351.35 prefill tok/s covers the additional 237,826 tokens, not a separate cold 300K prefill.

Full-logit capture uses a campaign-only wrapper around the public session API. It records every vocabulary float and position before generation and after each generated token. Short traces include 128 decode steps per frontier; long traces include 256. Public session snapshots restore the same long prefix for all arms. Loaded prefixes are recorded as zero measured prefill work, and traced runs are excluded from speed claims.

Fresh processes run from separate immutable copies of the source and binaries. All DS4_, MTL_ and ASTRA_ environment overrides are cleared before explicit experiment controls are set. `frozen-arm-manifest.json` hashes every shader, relevant source and binary. Full commands, overrides and raw CSV rows are in the campaign JSON files.

Local campaign: `/Users/jw/.cache/ds4-bench/glm53-fork-eval-20260909-185810`. Complete logits, raw logs and prefix snapshots stay in that directory. Compact manifests, wrapper script and timing data are supplied beside this report in `evidence/`. `recovery-arm-manifest.json` records the final tested commit, and `recovery-trace-comparisons.json` records its full-logit byte comparisons and hashes. Files prefixed `reviewed-` describe the intermediate `fa00a42` build. The component table uses separately frozen experimental arms before the review corrections described below; the final confirmation table uses `0f211542`. Instrumented model runs are excluded from all throughput comparisons.

This is one prompt on one M3 Ultra, with two or three repetitions per arm. It establishes local incremental gains, not statistical confidence across workloads or a universal speedup. Session continuation and recurrent-state fixtures were tested; this campaign does not claim a byte comparison of every persistent state buffer. No distributed hardware or SSD-streaming model campaign was run.

## Additional screens

### Expert bank: positive, larger integration

Controlled on/off testing inside the pinned fork, using our original BF16-KDA model and `DS4_GLM_EXACT=1`, measured 33,000-token cold prefill at **500.165 tok/s off versus 514.13 on (+2.79%)**, two interleaved runs per arm. All 65 captured full-logit frames match between off/on. Logs show the bank actually executed on all 42 routed layers, with zero bank refusals, rather than silently taking the default fallback.

Both arms use the fork's default 8,192-token prefill chunk and the same untracked-model setting. The enabled arm includes its layer-major schedule, fused command buffers and pipelined layers. A single scheduling-only control, keeping packed weights, measured **489.41 tok/s**, so scheduling alone did not explain the gain. This is a screen of the fork's implementation; this adapter branch does not import that scheduling subsystem. The measured allocation is **13.50 GiB bank + 8.57 GiB staging**, alongside the model and ordinary graph storage. It is a plausible next integration after the smaller decode changes, with only one tested prefill length here.

### Existing untracked-model setting: no resolved benefit

On the router-only adapter, three runs per arm measured **27.01 off / 27.00 on at 2K** and **26.81 off / 26.78 on at 8K**. All captured logits match. These differences sit inside the run spread; there is no reason from this screen to change the current default. Off means the variable is absent: this repository treats even `DS4_METAL_MODEL_UNTRACKED=0` as enabled.

### Combined checks

The final build at `0f211542` passes all three model-free experiment tests, the existing GLM KDA/attention tests, the Metal kernel suite and server suite. CPU session-state and TP command tests also pass. Every full logit matches the original baseline at 2K, 8K, 62,174 and 300,000 with the combined adapters enabled. The unchanged KDA-Q8 model matches at 2K, 8K and 62,174. Coverage logs report 21,504 router fusions and 5,632 DSA attempts for the original long run; the Q8 long run reports 10,752 router fusions, 8,704 mixed-input fusions and 2,816 DSA attempts.

A late refusal test caught the host compiler folding away a NaN check under `-ffast-math`. Volatile integer exponent checks preserve that guard. The checkpoint regression was extended beyond the fork's original eight cases: a literal closing tag inside an argument must use the same structural-tag matcher as the parser, and opening tags in earlier prompt text require retrying inside an unmatched candidate span.

The final checkpoint regression also reproduces 80 failed assertions when an incomplete literal argument wrapper in earlier prompt text hides a later zero-argument call. Scanning now continues from the next opening tag when a candidate has no structural end. The full server suite passes after that follow-up; the inference and shader files are identical to the model-tested commit.

Review also found an inherited exact-attention edge case: invalid selected row IDs loaded row zero before multiplying by zero, allowing NaN/Inf in row zero to poison a result. Invalid rows now stage literal zeros. A GPU regression compares this case against the generic path. This deliberately fixes exceptional invalid-row behavior; ordinary model traces still match the old baseline byte for byte. The complete kernel/model suite passes with all three corrections.

The incremental review found that a failed command buffer could leave partial router arrivals or a half-filled DSA candidate set in reusable storage. Failure invalidation now gives every registered graph counter fresh backing and retires the shared DSA scratch without CPU-mutating buffers still retained by in-flight work. Model-free recovery regressions poison 100 of 144 router arrivals and the exact stale-512 plus current-512 DSA acceptance case, then compare the next complete outputs with the original references. The selector finisher's unused second pool-expansion mode was removed; the graph continues to use the existing separate expansion. The fixed build at `0f211542` passes the complete kernel/server suite, every full-model logit comparison, and the repeated timing confirmation above. The new radix-tree prefix lookup also passes a 30,000-query brute-force oracle with insertion/removal, binary and empty keys, null values and nested prefixes; its standalone source and result are in the evidence directory.

## Reproducing the checks

From the experimental worktree on an M3 Ultra:

```sh
make -j8 ds4-bench ds4_test tests/test_glm53_kda
make test-glm53-fork
./tests/test_glm53_kda
./ds4_test --metal-kernels --server
```

Run binaries from their own matching source directory because this loader resolves Metal source relative to the working directory. Use a clean experiment environment; then enable the two opt-in adapters for a short benchmark:

```sh
DS4_GLM_ENABLE_TOPK_FAST=1 DS4_GLM_ENABLE_KDA_Q8_INPUTS=1 \
  ./ds4-bench -m /path/to/GLM-5.3-Flash-Q4_K-kdaQ8.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 8192 --step-mul 4 --ctx-alloc 16384 \
  --gen-tokens 256 --csv /tmp/glm53-fork-short.csv
```

This short run exercises router and mixed-input fusion; it is below the DSA threshold. The original BF16 model exercises router fusion here. For long contexts and snapshot reuse, consult the exact commands and environment overrides in `topk-screen.json` and `reviewed-screen.json`. To build the capture wrapper after the engine objects, run `python3 evidence/build_capture.py /absolute/path/to/source-tree`; it writes `campaign_bench.c` and `campaign-bench` in that tree. Keep trace-enabled runs separate from throughput measurements.

Roll back router fusion with `DS4_METAL_DISABLE_GLM53_ROUTER_SHARED=1`. Leave the DSA and KDA opt-in variables unset to use their original paths. The existing aggregate GLM rollback controls remain effective.

### Review

Two complete source reviews identified the checkpoint, non-finite and GPU recovery issues described above. All findings were fixed in preserved commits and the final build passed the executable checks. Both post-fix re-review rounds hit the runner's 30-minute limit, so neither run produced a successful pipeline outcome. The final review is scoped to the last fix delta from `88f6f005` plus its evidence; rebase, duplicate model tests and publication remain disabled. This section records that review history, separately from the measured results and manual test outcomes.
