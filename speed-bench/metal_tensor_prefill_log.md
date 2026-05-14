# Metal Tensor Prefill Optimization Log

Branch: `metal-tensor-prefill-next`

Date: 2026-05-14

This branch keeps the current low-drift Tensor default and uses the five-fixture
quality gate before promoting any prefill optimization.

## Drift Gate

Run:

```sh
python3 speed-bench/run_quality_drift_gate.py \
  --out-dir speed-bench/local-runs/20260514-1500-default-moe-gate-up15-down12-quality-drift
```

Fixtures:

- `short_italian_fact`
- `short_code_completion`
- `short_reasoning_plain`
- `long_memory_archive`
- `long_code_audit`

Summary:

| Pair | top1 mismatches | greedy mismatches | worst RMS | worst top20 abs |
| --- | ---: | ---: | ---: | ---: |
| standard vs quality | 0 | 1 | 0.618172 | 2.24006 |
| tensor vs quality | 0 | 1 | 0.618172 | 2.24006 |
| tensor vs standard | 0 | 0 | 0.239946 | 0.55422 |

Gate status: OK.

The direct equivalence test also passed:

```sh
./ds4_test --metal-mpp-equivalence
```

Result after promoting the routed-MoE Tensor window to down from layer 12 and
gate/up from layer 15:
`top1_mismatch=0`, `greedy_fail=0`,
`worst_rms=0.239946`, and `worst_top20_max_abs=0.55422`.

## HC Stable Sigmoid Scope

VariableFate noted that commit `670411d` routed only the standalone
`kernel_dsv4_hc_split_sinkhorn` through `ds4_hc_sigmoid()` and
`ds4_hc_twice_sigmoid()`, while the fused decode kernels kept inline
`1/(1+exp(-z))` forms. That scope is intentional for now.

Inspected paths:

- `ds4_gpu_hc_split_sinkhorn_tensor`: standalone split/sinkhorn path.
- `ds4_gpu_hc_split_weighted_sum_tensor`: fused split plus pre-weighted HC
  reduction, used by batched paths.
- `ds4_gpu_hc_split_weighted_sum_norm_tensor`: decode-only HC-pre plus weighted
  RMSNorm fusion. This is the hot release decode path and is called for both
  attention HC-pre and FFN HC-pre.

Local A/B patch:

- Changed the four fused sites in `kernel_dsv4_hc_split_weighted_sum` and
  `kernel_dsv4_hc_split_weighted_sum_norm4` to call `ds4_hc_sigmoid()` and
  `ds4_hc_twice_sigmoid()`.
- Built with `make ds4 ds4-bench ds4_test`.

Generation throughput on `promessi_sposi`, `ctx=8192`, `gen_tokens=256`:

| Variant | gen t/s |
| --- | ---: |
| production inline exp after revert | 33.28 |
| helper exp with `DS4_METAL_HC_STABLE=0`, repeat 1 | 32.32 |
| helper exp with `DS4_METAL_HC_STABLE=0`, repeat 2 | 31.21 |
| helper tanh with default `DS4_METAL_HC_STABLE=1`, repeat 1 | 31.61 |
| helper tanh with default `DS4_METAL_HC_STABLE=1`, repeat 2 | 31.01 |

Quality result:

- The helper/tanh fused-kernel patch produced non-finite logits in the
  five-fixture drift run. All 15 captured logits dumps reported
  `argmax_logit: nan`, so the summary could not be parsed as valid JSON.
- `./ds4_test --metal-mpp-equivalence` with helper/tanh failed with
  `logits_fail=5` and `top1_mismatch=5`.
- The same helper-call patch with `DS4_METAL_HC_STABLE=0`, which compiles the
  helpers back to the historical exp form, passed equivalence with
  `top1_mismatch=0`, `greedy_fail=0`, `worst_rms=0.066747`, and
  `worst_top20_max_abs=0.191437`.

Decision: keep `DS4_METAL_HC_STABLE` limited to the standalone split/sinkhorn
path and keep the fused decode kernels on the historical inline exp form. A
separate decode flag is not useful until there is a finite, low-drift
decode-specific stable form with measured throughput. The production code keeps
the fused math unchanged and documents this scope near the helper definitions.

## Compact Prefill Timing

Run shape:

```sh
CTX_MAX=8192 GEN_TOKENS=16 \
  OUT_DIR=speed-bench/local-runs/20260514-1510-default-moe-gate-up15-down12-compact \
  OPEN_CHART=0 \
  speed-bench/run_metal_tensor_bench.sh
```

Current routed-MoE Tensor default (`down=12`, `up=15`, `gate=15`) vs standard
Metal:

| ctx | standard prefill t/s | tensor prefill t/s | tensor gain | standard gen t/s | tensor gen t/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 260.99 | 345.19 | 32.3% | 37.18 | 37.45 |
| 1024 | 266.51 | 350.99 | 31.7% | 37.21 | 36.68 |
| 2048 | 319.20 | 398.03 | 24.7% | 36.41 | 35.52 |
| 4096 | 319.02 | 382.11 | 19.8% | 33.27 | 32.30 |
| 8192 | 332.97 | 389.44 | 17.0% | 32.65 | 31.41 |

This keeps the plan focused on prefill. Generation is close to neutral at
shorter contexts in this compact run, with the largest measured drop at 8192
tokens.

## Rejected Knobs

These were evaluated as env-only candidates and not promoted.

| Candidate | Speed result | Drift result | Decision |
| --- | --- | --- | --- |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=18` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=18` | One run showed +2.2% to +5.7% over Tensor auto, but an immediate control run favored the old layer-20 default by 8.7% to 17.1%. | Five-fixture gate passed with `tensor_vs_standard` worst RMS `0.139912` and worst top20 abs `0.316128`. | Not promoted because the speed win was not stable. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=18` alone with up/down defaulting to 19/19 | Two-repeat median vs 19/19/19 Tensor auto: +0.3% at 512, then -0.3%, -0.3%, -0.7%, and +0.6% from 1024..8192. | Not run. | Reject before drift gate because the speed change is noise-level. |
| `DS4_METAL_MPP_MOE_UP_START_LAYER=18` alone with gate/down defaulting to 19/19 | Two-repeat median vs 19/19/19 Tensor auto: -0.2% at 512, -0.9% at 1024, +0.3% at 2048, -0.1% at 4096, and -0.1% at 8192. | Not run. | Reject before drift gate because the speed change is noise-level. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=18` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=18` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=20` | Slower than the promoted Tensor auto default by 0.1% to 3.6% in two-repeat median timing. | Not run. | Reject before drift gate. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=18` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=18` with down defaulting to 19 | Two-repeat median vs 19/19/19 Tensor auto: +0.1% at 512, then -0.7%, -1.9%, -3.0%, and -1.3% from 1024..8192. Generation was within -0.9%..+0.6%. | Not run. | Reject before drift gate because it is slower at most measured contexts. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=18` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=18` with down defaulting to 12 | Two-repeat median vs down-12 Tensor auto: -2.2% at 512, -2.8% at 1024, -2.7% at 2048, -0.1% at 4096, and +1.5% at 8192. Generation was within -0.7%..+1.5%. | Not run. | Reject before drift gate because it is slower at most measured contexts. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=14` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=14` with down defaulting to 12 | Two-repeat median vs down-12 Tensor auto: +2.7% at 512, +2.9% at 1024, +2.2% at 2048, +1.1% at 4096, but -0.8% at 8192. Generation was -3.2% at 8192. | Not run. | Reject before drift gate because it regresses the long-context point and generation more than the layer-15 window. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=14` with down defaulting to 12 and up defaulting to 15 | Two-repeat median vs current Tensor auto: -2.2% at 512, -1.7% at 1024, -0.4% at 2048, +1.0% at 4096, and +2.1% at 8192. Generation was down by 0.4%..1.9%. | Not run. | Reject before drift gate because it is a tradeoff, not a clear prefill win. |
| `DS4_METAL_MPP_MOE_UP_START_LAYER=14` with down defaulting to 12 and gate defaulting to 15 | Two-repeat median vs current Tensor auto: -3.4% at 512, -6.4% at 1024, -4.9% at 2048, -6.2% at 4096, and -5.1% at 8192. | Not run. | Reject before drift gate because it is consistently slower. |
| `DS4_METAL_MPP_MOE_TILE_N=64` | Slower than default by 3.3% to 15.6%. | Not run. | Reject before drift gate. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=9` with gate/up unchanged at 19 | Two-repeat median vs down-12 Tensor auto: +0.3% at 512, +0.1% at 1024, -1.4% at 2048, -0.4% at 4096, and -0.5% at 8192. Generation was within -0.7%..+0.5%. | Not run. | Reject before drift gate because it is slower at most measured contexts. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=10` with gate/up unchanged at 19 | Two-repeat median vs 19/19/19 Tensor auto: +0.8% at 512, flat at 1024, +0.8% at 2048, +2.6% at 4096, and +2.8% at 8192. Generation was within -1.7%..+1.4%. | Five-fixture gate and `./ds4_test --metal-mpp-equivalence` passed, but `tensor_vs_standard` drift rose to worst RMS `0.314905` and worst top20 abs `0.780825`. | Not promoted because layer 12 kept useful speed with lower drift. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=11` with gate/up unchanged at 19 | Two-repeat median vs 19/19/19 Tensor auto: +1.7% at 512, +1.7% at 1024, +3.5% at 2048, +1.7% at 4096, and +1.2% at 8192. Generation was within -1.4%..-0.3%. | Five-fixture gate passed, but `tensor_vs_standard` drift rose to worst RMS `0.314275` and worst top20 abs `0.725578`. | Not promoted because layer 12 had a better drift balance. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=11` with gate/up defaulting to 15 | Two-repeat median vs current Tensor auto: +0.3% at 512, -0.1% at 1024, +0.2% at 2048, +0.5% at 4096, and -2.8% at 8192. Generation was within -1.3%..+0.2%. | Not run. | Reject before drift gate because the new gate/up window removes most of the earlier speed upside and the long-context point regresses. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=18` with gate/up/down defaulting to 19/19/19 | Two-repeat median vs 19/19/19 Tensor auto: -2.1% at 512, -3.1% at 1024, -3.3% at 2048, -0.7% at 4096, and +1.7% at 8192. Generation was within -1.2%..+0.4%. | Not run. | Reject before drift gate because it is slower at most measured contexts. |
| `DS4_METAL_MPP_F16_PAIR=1` | Slower than default by 0.9% to 8.6%. | Previously known safe, but not rerun here. | Keep opt-in. |
| `DS4_METAL_MPP_F16_WIDE=1` | Diagnostic-only wider 512/1024-column compressor Tensor route. | Existing long-code full-model equivalence check fails with wide F16 Tensor (`rms ~= 0.569`, `top20_max_abs ~= 1.48`). | Keep default-off; do not spend more prefill timing effort until the drift issue has a new mitigation. |
| `DS4_METAL_MPP_DIRECT_RHS=0` plus `DS4_METAL_MPP_F16_DIRECT_RHS=1` to isolate staged-RHS attention-output low projection | Two-repeat median vs current Tensor auto: -7.1% at 512, -4.9% at 1024, -4.5% at 2048, -3.4% at 4096, and +0.1% at 8192. Generation was within -0.6%..+0.2%. | Not run. | Reject before drift gate because it is slower at most measured contexts. Keep the direct-RHS attention-output default. |
| `DS4_METAL_MPP_ATTN_OUT_TILE_N=32` | Slower than default by 1.1% to 16.4%. | Not run. | Keep default tile 64. |
| `DS4_METAL_MPP_ATTN_OUT_FILTER=layer=31..42` | Two-repeat median vs 32..42 Tensor auto: flat at 512, then slower by 0.3% to 1.4% from 1024..8192. | Not run. | Reject before drift gate; keep attention-output at 32..42. |
| Local patch: split dense Q8_0 prefill full 32-token tiles from the non-32-token tail (`DS4_METAL_Q8_PREFILL_SPLIT_TAIL=1` prototype) | On `long_code_audit` at `ctx=3836`, two-repeat median vs current Tensor auto was +0.3% prefill and +0.6% generation. | Not run. | Reverted before drift gate because the speed change is noise-level and does not justify keeping another Q8_0 switch. |
| Local patch: dense Q8_0 cooperative Tensor direct-RHS prefill prototype scoped to `attn_q_b` | Two-repeat median vs current Tensor auto was mixed: +2.8% at 512, -1.3% at 1024, -2.2% at 2048, +2.3% at 4096, and +5.1% at 8192. Generation moved -2.5%..+0.8%. | Not run. | Reverted before drift gate because mid-context prefill and generation regressed. |
| Local patch: dense Q8_0 cooperative Tensor direct-RHS prefill prototype scoped to `attn_out`/`attn_output_b` | Two-repeat median vs current Tensor auto was +4.6% at 512, +4.4% at 1024, +6.0% at 2048, +5.2% at 4096, and +3.5% at 8192. A conservative `attn_out@layer=32..42` window was only +0.6%..+0.9% and dropped generation up to 2.2%. | All-layer `attn_out` failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` worst RMS `0.531143` and worst top20 abs `1.17201`. | Reverted despite speed because it violates the no-new-top1/no-new-greedy rule, and the late-only safe-shape hypothesis was noise-level. |
| `DS4_METAL_MPP_MOE_PAIR_GATE_UP=1` with gate/up/down defaulting to 19/19/19 | Two-repeat median vs 19/19/19 Tensor auto: -6.2% at 512, -3.4% at 1024, -2.7% at 2048, -2.5% at 4096, and -2.1% at 8192. Generation was within -0.2%..+1.2%. | Not run. | Reject before drift gate because the paired dispatch is consistently slower. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=18` | Two-repeat median vs current Tensor auto: +0.1% at 512, -0.1% at 1024, -0.6% at 2048, -1.8% at 4096, and -1.2% at 8192. | Not run. | Reject before drift gate because it is not faster than the current 19/19/19 default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=19` | Two-repeat median vs current Tensor auto: -0.9% at 512, -1.9% at 1024, -1.6% at 2048, -2.7% at 4096, and -1.8% at 8192. Generation was within -0.3%..+0.7%. | Not run. | Reject before drift gate because it is consistently slower than the current 19/19/19 default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=10` | Two-repeat median vs current Tensor auto: +7.5% at 512, +8.4% at 1024, +6.0% at 2048, +3.8% at 4096, +4.8% at 8192. Generation was -2.8%, -1.0%, +1.3%, +1.1%, +0.7%. | Failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` also had one top-1 and one greedy mismatch. | Reject despite the speed because it violates the no-new-top1/no-new-greedy rule. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=12` | Two-repeat median vs current Tensor auto: +12.2% at 512, +8.5% at 1024, +8.3% at 2048, +3.2% at 4096, +1.1% at 8192. Generation was +3.4%, -0.2%, +1.5%, -4.6%, -3.6%. | Full `./ds4_test --metal-mpp-equivalence` passed with no top-1 or greedy mismatch, but drift rose to worst RMS `0.300474` and worst top20 abs `1.00957`. | Reject before the full quality gate: long-context speed is weak and drift is much worse than the current conservative default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=15` | Two-repeat median vs current Tensor auto: +2.3% at 512, +2.0% at 1024, +1.5% at 2048, +2.6% at 4096, +2.0% at 8192. Generation was -2.7%, +0.0%, -1.8%, +1.1%, +1.4%. | Full `./ds4_test --metal-mpp-equivalence` passed with no top-1 or greedy mismatch, but drift rose to worst RMS `0.229322` and worst top20 abs `0.511806`. | Reject before the full quality gate: speed is marginal and drift is still worse than default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_EXPERIMENTAL_MOE_MATMUL_START_LAYER=17` | Two-repeat median vs current Tensor auto: +2.2% at 512, +0.5% at 1024, +0.8% at 2048, +1.2% at 4096, +0.7% at 8192. Generation was within -1.7%..+0.5%. | Full `./ds4_test --metal-mpp-equivalence` passed with no top-1 or greedy mismatch, but drift rose to worst RMS `0.190587` and worst top20 abs `0.560192`. | Reject before the full quality gate: speed is within noise and drift is worse than default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus route-specific gate start `8`, up start `15`, down start `12` | Two-repeat median vs current Tensor auto: +2.1% at 512, +2.6% at 1024, +1.5% at 2048, +1.8% at 4096, and +1.4% at 8192. Generation was within -0.6%..+0.4%. | Failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` had one top-1 and one greedy mismatch. | Reject despite the clean timing profile because it violates the no-new-top1/no-new-greedy rule. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus route-specific up start `0`, gate start `15`, down start `12` | Two-repeat median vs current Tensor auto: +6.6% at 512, +6.3% at 1024, +4.5% at 2048, +3.3% at 4096, and +2.9% at 8192. Generation was within -1.4%..+0.5%. | Failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` had one top-1 and one greedy mismatch. | Reject despite speed because it violates the no-new-top1/no-new-greedy rule. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus route-specific down start `0`, gate/up start `15` | Two-repeat median vs current Tensor auto: +4.1% at 512, +4.2% at 1024, +3.5% at 2048, +2.3% at 4096, and +2.2% at 8192. Generation was within -1.7%..+0.1%. | Failed the five-fixture gate: `long_memory_archive` top-1 changed and greedy differed at step 0; `tensor_vs_standard` had one top-1 and one greedy mismatch. | Reject despite speed because it violates the no-new-top1/no-new-greedy rule. |

## Promoted Candidates

| Candidate | Speed result | Drift result | Decision |
| --- | --- | --- | --- |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=19` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=19` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=21` | Two-repeat median vs current Tensor auto: +0.6% at 512, +0.8% at 1024, +2.3% at 2048, +2.0% at 4096, +1.6% at 8192. Generation was within -1.4%..+0.5%. | Five-fixture gate passed, first as env candidate and again as the env-free default after promotion. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.176030`, worst top20 abs `0.360397`. | Promoted, then superseded by the lower-drift 19/19/20 window and the faster 19/19/19 window. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=19` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=19` plus `DS4_METAL_MPP_MOE_DOWN_START_LAYER=20` | Two-repeat median vs 19/19/21 Tensor auto: +0.3% at 512, +1.2% at 1024, +0.9% at 2048, +0.4% at 4096, +0.2% at 8192. Generation was within -0.9%..+1.0%. | Five-fixture gate passed, first as env candidate and again as the env-free default after promotion. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.066747`, worst top20 abs `0.191437`. | Promoted, then superseded by the slightly faster 19/19/19 window. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=19` with gate/up unchanged at 19 | Two-repeat median vs 19/19/20 Tensor auto: +0.9% at 512, +1.2% at 1024, +1.1% at 2048, +0.4% at 4096, +0.9% at 8192. Generation was within -1.0%..+1.4%. | Five-fixture env-candidate gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.136143`, worst top20 abs `0.315292`. | Promoted as the next routed-MoE default window: gate/up/down from layer 19. |
| `DS4_METAL_MPP_MOE_DOWN_START_LAYER=12` with gate/up unchanged at 19 | Two-repeat median vs 19/19/19 Tensor auto: +2.1% at 512, +0.8% at 1024, +2.0% at 2048, +1.1% at 4096, and +1.5% at 8192. Env-free compact timing after promotion showed Tensor prefill +26.7%, +28.8%, +21.9%, +18.7%, and +15.7% vs standard Metal from 512..8192. | Five-fixture env-candidate gate and env-free default gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.229474`, worst top20 abs `0.601166`. `./ds4_test --metal-mpp-equivalence` also passed with the same worst RMS/top20 abs. | Promoted, then superseded by the layer-15 gate/up window. |
| `DS4_METAL_MPP_MOE_GATE_START_LAYER=15` plus `DS4_METAL_MPP_MOE_UP_START_LAYER=15` with down defaulting to 12 | Two-repeat median vs down-12 Tensor auto: +2.2% at 512, +1.5% at 1024, +0.3% at 2048, +0.2% at 4096, and +0.6% at 8192. Env-free compact timing after promotion shows Tensor prefill +32.3%, +31.7%, +24.7%, +19.8%, and +17.0% vs standard Metal from 512..8192. | Five-fixture env-candidate gate and env-free default gate passed. `tensor_vs_quality`: top1 mismatches `0`, greedy mismatches `1` matching standard-vs-quality, worst RMS `0.618172`, worst top20 abs `2.24006`. `tensor_vs_standard`: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.239946`, worst top20 abs `0.55422`. `./ds4_test --metal-mpp-equivalence` also passed with the same worst RMS/top20 abs. | Promoted as the current routed-MoE default window: down from layer 12, gate/up from layer 15. |

## Default-Off Candidates

| Candidate | Speed result | Drift result | Decision |
| --- | --- | --- | --- |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` | Two-repeat median vs current Tensor auto: +15.9% at 512, +19.7% at 1024, +12.5% at 2048, +6.8% at 4096, +11.7% at 8192. Generation was -4.9%, -1.5%, -3.5%, -0.9%, -1.7%. | Five-fixture gate passed. `tensor_vs_quality` stayed inside the current standard-vs-quality envelope with top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, and worst top20 abs `2.24006`. `tensor_vs_standard` had no top1 or greedy mismatch, but drift increased to worst RMS `0.669241` and worst top20 abs `1.30664`. | Keep default-off until an eval confirms the larger Tensor-vs-standard logit movement is acceptable. This is the best prefill candidate so far, but not yet promoted over the lower-drift conservative default. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus route-specific gate start `0`, up start `15`, down start `12` | Two-repeat median vs current Tensor auto: +2.0% at 512, +4.6% at 1024, +6.1% at 2048, +7.3% at 4096, and +4.6% at 8192. Generation was near flat through 4096 and -4.4% at 8192. | Five-fixture gate passed. `tensor_vs_quality` stayed inside the current standard-vs-quality envelope with top1 mismatches `0`, greedy mismatches `1`, worst RMS `0.618172`, and worst top20 abs `2.24006`. `tensor_vs_standard` had no top1 or greedy mismatch, but drift rose to worst RMS `0.529461` and worst top20 abs `1.05153`. | Keep default-off. It is the best route-specific speed candidate that still passes the gate, but it is not promoted because Tensor-vs-standard drift is materially larger than the current conservative default and the 8192 generation point regressed in timing. |
| `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` plus `DS4_METAL_MPP_MOE_FAST_LAYOUT=0` | Two-repeat median vs current Tensor auto: +8.4% at 512, +12.3% at 1024, +0.4% at 2048, +1.2% at 4096, and +4.3% at 8192. Generation was -4.2% at 1024, -3.2% at 2048, -4.4% at 4096, and near flat at 512/8192. | Five-fixture gate passed, but `tensor_vs_standard` was unchanged from the faster experimental layout: top1 mismatches `0`, greedy mismatches `0`, worst RMS `0.669241`, and worst top20 abs `1.30664`. | Reject as the preferred experimental layout because it gives up speed without reducing the larger Tensor-vs-standard movement. |

## Profile Signal

Representative profile:

```sh
env DS4_METAL_GRAPH_TOKEN_PROFILE=1 \
    DS4_METAL_LAYER_STAGE_PROFILE=1 \
    DS4_METAL_MOE_STAGE_PROFILE=1 \
    DS4_METAL_ATTN_OUT_STAGE_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE_FILTER=attn_ \
    ./ds4 --metal -mt auto \
      --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
      -c 8192 -n 1 --system "" --nothink --temp 0
```

Current default result: `prefill: 423.95 t/s`.

Important stage timings at `tokens=3844`:

- Layers 0..11 use legacy routed-MoE projections (`mpp=0/0/0`): median gate
  `32.615 ms`, up `32.579 ms`, down `32.356 ms`.
- Layers 12..14 use Tensor down only (`mpp=0/0/1`): median gate `32.531 ms`,
  up `32.523 ms`, down `13.383 ms`.
- Layers 15..42 use Tensor gate/up/down (`mpp=1/1/1`): median gate
  `13.875 ms`, up `13.859 ms`, down `13.518 ms`.
- Dense attention Q8_0 medians are `attn_q_b=18.069 ms` and
  `attn_out=18.366 ms`.
- The attention output projection stage remains about `37.246 ms/layer`;
  inside the Tensor-enabled late layers the low and output projections are each
  about `18.5-18.7 ms`.

The routed-MoE stage profiler now prints layer, token/pair counts, expert
count, gate/down quant types, `mm_id` vs `mm_id_pair_mpp` path, active Tensor
route mask, tile widths, and intermediate precision. Use
`DS4_METAL_MOE_STAGE_PROFILE_FILTER=<substring>` to limit printed rows while
preserving stage flushes for timing correctness.

Long-shape routed-MoE profile on `long_code_audit`, `tok=3844`,
`pairs=23064`, `experts=6`, `gate=iq2_xxs`, `down=q2_k`:

- Layers before the current conservative Tensor window are still the largest
  remaining routed-MoE opportunity, but the latest one-layer route-window tests
  did not produce a clean prefill win.

This confirms the highest-value routed-MoE target is still the pre-window
specialized `mm_id` path, not the generic dense Q8_0 wrapper. The dense
attention target remains `attn_q_b in=1024 out=32768`.

Comparator check on the all-layer experimental routed-MoE Tensor path:

```sh
env DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1 \
    DS4_METAL_MPP_COMPARE_ROUTE=all \
    DS4_METAL_MPP_COMPARE_MAX=12 \
    DS4_METAL_MPP_COMPARE_VERBOSE=1 \
    ./ds4 --metal -mt auto \
      --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
      -c 8192 -n 1 --system "" --nothink --temp 0
```

The first 12 local projection comparisons, covering `moe_gate`, `moe_up`, and
`moe_down` in layers 0..3, stayed far inside the local comparator target. The
largest observed max abs was about `3.8e-5`, and RMS was about `1e-7` or lower.
That points to accumulated full-model movement from enabling more Tensor
layers, not an obvious single routed-MoE projection breach.

For the next matmul kernel iteration, enable filtered Q8_0 prefill-level timing
with:

```sh
env DS4_METAL_Q8_PREFILL_PROFILE=1 \
    DS4_METAL_Q8_PREFILL_PROFILE_FILTER=attn_q_b \
    ./ds4 --metal -mt auto \
      --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
      -c 8192 -n 1 --system "" --nothink --temp 0
```

This keeps the legacy Q8_0 dispatch but flushes timed prefill batches so each
logged row names the module/layer context, input/output dimensions, token batch,
and elapsed time. Use those rows to pick the first default-off Metal 4
cooperative/tensor Q8_0 matmul target.

Smoke result on `short_code_completion`, `FILTER=moe_gate`: no rows. That is
expected because routed-MoE gate/up/down use the specialized routed-MoE kernels,
not the generic dense Q8_0 prefill wrapper.

Smoke result on `short_code_completion`, `FILTER=attn_q_b`: rows were emitted
for layers 0..42 with shape `in=1024 out=32768 tok=27`. Layer 0 included
first-use overhead at `1.298 ms`; later layers were about `0.33-0.41 ms` each.
This confirms the profile hook works for dense attention Q8_0 projections.

Long-shape smoke result on `long_code_audit`, `FILTER=attn_q_b`, `tok=3844`:
layer 0 reported `27.695 ms`; most layers reported about `18.0-19.2 ms`, with
late layers 40..42 at about `20.0-20.6 ms`. This makes
`attn_q_b in=1024 out=32768` the first dense Q8_0 prototype shape to target
after routed-MoE profiling.

Broader long-shape attention profile on `long_code_audit`, `FILTER=attn_`,
`tok=3844`:

- `attn_q_a in=4096 out=1024`: about `2.45-2.8 ms/layer` after layer-0
  first-use overhead.
- `attn_kv in=4096 out=512`: about `1.35-1.48 ms/layer`.
- `attn_q_b in=1024 out=32768`: about `18.0-18.9 ms/layer`.
- `attn_out in=8192 out=4096`: about `18.0-19.3 ms/layer`.

In this profile `attn_out` names the second/output projection
(`attn_output_b`) that still goes through the generic dense Q8_0 wrapper. The
attention-output low projection (`attn_output_a`) already has a separate
guarded Tensor route and comparator. Dense Q8_0 work should therefore focus on
`attn_q_b` and `attn_output_b`, not on the already-specialized low projection.

## Matmul-First Direction

The current legacy dense Q8_0 prefill kernel already uses
`simdgroup_multiply_accumulate`, so the next meaningful optimization is not just
to rewrite it with the same primitive. The next target is a default-off
quantized prefill matmul family that uses Metal 4 cooperative/tensor matrix
primitives where they help, while preserving the legacy dequantization and
reduction behavior closely enough to pass the quality gate.

This should be treated as a new kernel family, not a revival of the removed
dense Q8_0 Tensor route. The removed route was drift-prone in full-model
comparison; a replacement needs its own dispatch switch, route comparator, and
five-fixture gate evidence before it can be promoted.

Metal 4 and the Neural Accelerator direction should be split into two tracks:

- Near-term: keep DS4 on custom Metal compute shaders over GGUF buffers, and use
  cooperative/tensor matmul primitives inside quantized prefill matmul kernels.
  This is the path that can directly improve current prefill without changing
  model loading or graph ownership.
- Longer-term: evaluate Metal 4 machine-learning passes/Core ML packages only if
  we can package stable repeated subgraphs without losing DS4's quantized
  mmap-backed layout, routed-MoE control, and drift gate. That is not a drop-in
  acceleration path for the current kernels.

Priority order:

1. Early routed-MoE gate/up/down specialized matmuls before the current safe
   Tensor window. Use the existing routed-MoE stage profiler and comparator for
   these routes; they do not pass through the generic dense Q8_0 wrapper.
2. Attention Q/output dense Q8_0 projections. Use
   `DS4_METAL_Q8_PREFILL_PROFILE=1` with a context filter such as `attn_q_b` to
   choose the first prototype shape.
3. Wider route windows only after the new kernel proves low drift in the
   five-fixture quality gate.

Promotion rule: keep a change only if it improves compact prefill timing and
passes the gate with no new top-1 or Tensor-vs-standard greedy regression.

Prototype checklist:

1. Use `DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1` as the first default-off
   experimental quantized prefill matmul dispatch. It moves only the routed-MoE
   Metal 4 cooperative/tensor matmul window and does not use the removed
   dense Q8_0 Tensor controls.
2. First target one high-impact routed-MoE projection shape and compare it with
   `DS4_METAL_MPP_COMPARE_ROUTE=moe_gate|moe_up|moe_down`.
3. Run compact prefill timing twice with an adjacent `-mt off` control to avoid
   promoting thermal/noise wins. Use:

   ```sh
   python3 speed-bench/run_prefill_candidate_gate.py \
     --candidate-label moe-matmul-first \
     --set-env DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1
   ```

4. Add `--run-drift-gate` before promotion. The helper calls
   `speed-bench/run_quality_drift_gate.py`; promotion requires no top-1
   mismatch, no Tensor-vs-standard greedy mismatch, and no regression beyond the
   current standard-vs-quality envelope.
